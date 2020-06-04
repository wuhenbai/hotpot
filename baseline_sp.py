import queue
from numpy.random import shuffle

import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange
from text_to_token import *
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertForQuestionAnswering1, BertForQuestionAnswering3
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from util import evaluate
from hotpot_evaluate_v1 import eval
from config import set_config


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "types"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "type"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    # if not feature.token_is_max_context.get(start_index, False):
                    #     continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            type=result.types))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    type=0))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "type"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.type == 0:
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True
            if pred.type == 1:
                final_text = "yes"
            if pred.type == 2:
                final_text = "no"
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    type=pred.type,
                    ))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        type=0))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(
                    0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, type=0)
                )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        nbest.append(
            _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, type=0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["type"] = entry.type
            nbest_json.append(output)
        assert len(nbest_json) >= 1
        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]

        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
                all_nbest_json[example.qas_id] = nbest_json
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def convert_to_tokens(examples, features, all_results):
    answer_dict = dict()

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    example_index_to_features = collections.defaultdict(list)
    for feature in features:
        example_index_to_features[feature.example_index].append(feature)

    for (example_index, example) in enumerate(examples):
        features = example_index_to_features[example_index]
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            # for (feature_index, feature) in enumerate(features):
            y1 = np.argmax(result.start_logits)
            y2 = np.argmax(result.end_logits)
            q_type = result.types
        # for i, qid in enumerate(ids):

            answer_text = ''
            if q_type == 0:
                doc_tokens = feature.tokens
                tok_tokens = doc_tokens[y1: y2 + 1]
                tok_to_orig_map = feature.token_to_orig_map
                if y2 < len(tok_to_orig_map):
                    orig_doc_start = tok_to_orig_map[y1]
                    orig_doc_end = tok_to_orig_map[y2]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens).strip('[,.;]')

                    final_text = get_final_text(tok_text, orig_text, do_lower_case=False, verbose_logging=False)
                    answer_text = final_text
            elif q_type == 1:
                answer_text = 'yes'
            elif q_type == 2:
                answer_text = 'no'
            answer_dict[example.qas_id] = answer_text
    return answer_dict

# class DataIteratorPack(object):
#     def __init__(self, features, example_dict, graph_dict, bsz, device, sent_limit, entity_limit,
#                  n_layers, entity_type_dict=None, sequential=False,):
#         self.bsz = bsz
#         self.device = device
#         self.features = features
#         self.example_dict = example_dict
#         self.graph_dict = graph_dict
#         self.entity_type_dict = entity_type_dict
#         self.sequential = sequential
#         self.sent_limit = sent_limit
#         self.entity_limit = entity_limit
#         self.example_ptr = 0
#         self.n_layers = n_layers
#         if not sequential:
#             shuffle(self.features)
#
#     def refresh(self):
#         self.example_ptr = 0
#         if not self.sequential:
#             shuffle(self.features)
#
#     def empty(self):
#         return self.example_ptr >= len(self.features)
#
#     def __len__(self):
#         return int(np.ceil(len(self.features)/self.bsz))
#
#     def __iter__(self):
#         # BERT input
#         context_idxs = torch.LongTensor(self.bsz, 512).cuda(self.device)
#         context_mask = torch.LongTensor(self.bsz, 512).cuda(self.device)
#         segment_idxs = torch.LongTensor(self.bsz, 512).cuda(self.device)
#
#         query_mapping = torch.Tensor(self.bsz, 512).cuda(self.device)
#         start_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
#         end_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
#         all_mapping = torch.Tensor(self.bsz, 512, self.sent_limit).cuda(self.device)
#         entity_mapping = torch.Tensor(self.bsz, self.entity_limit, 512).cuda(self.device)
#         entity_graphs = torch.Tensor(self.bsz, self.entity_limit, self.entity_limit).cuda(self.device)
#
#         # Label tensor
#         y1 = torch.LongTensor(self.bsz).cuda(self.device)
#         y2 = torch.LongTensor(self.bsz).cuda(self.device)
#         q_type = torch.LongTensor(self.bsz).cuda(self.device)
#         is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)
#
#         start_mask = torch.FloatTensor(self.bsz, self.entity_limit).cuda(self.device)
#         start_mask_weight = torch.FloatTensor(self.bsz, self.entity_limit).cuda(self.device)
#         entity_label = torch.LongTensor(self.bsz).cuda(self.device)
#         bfs_mask = torch.FloatTensor(self.bsz, self.n_layers, self.entity_limit).cuda(self.device)
#
#         while True:
#             if self.example_ptr >= len(self.features):
#                 break
#             start_id = self.example_ptr
#             cur_bsz = min(self.bsz, len(self.features) - start_id)
#             cur_batch = self.features[start_id: start_id + cur_bsz]
#             cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)
#
#             ids = []
#             unique_ids = []
#             is_support.fill_(IGNORE_INDEX)
#             # is_support.fill_(0)  # BCE
#             max_sent_cnt = 0
#             max_entity_cnt = 0
#
#             for mapping in [start_mapping, end_mapping, all_mapping, query_mapping]:
#                 mapping.zero_()
#
#             for i in range(len(cur_batch)):
#                 case = cur_batch[i]
#                 context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
#                 context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
#                 segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))
#
#                 for j in range(case.sent_spans[0][0] - 1):
#                     query_mapping[i, j] = 1
#
#                 if case.ans_type == 0:
#                     if len(case.end_position) == 0:
#                         y1[i] = y2[i] = 0
#                     elif case.end_position[0] < 512:
#                         y1[i] = case.start_position[0]
#                         y2[i] = case.end_position[0]
#                     else:
#                         y1[i] = y2[i] = 0
#                     q_type[i] = 0
#                 elif case.ans_type == 1:
#                     y1[i] = IGNORE_INDEX
#                     y2[i] = IGNORE_INDEX
#                     q_type[i] = 1
#                 elif case.ans_type == 2:
#                     y1[i] = IGNORE_INDEX
#                     y2[i] = IGNORE_INDEX
#                     q_type[i] = 2
#
#                 for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
#                     is_sp_flag = j in case.sup_fact_ids
#                     start, end = sent_span
#                     if start < end:
#                         is_support[i, j] = int(is_sp_flag)
#                         all_mapping[i, start:end+1, j] = 1
#                         start_mapping[i, j, start] = 1
#                         end_mapping[i, j, end] = 1
#
#                 ids.append(case.qas_id)
#                 unique_ids.append(case.unique_id)
#                 max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))
#
#             input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
#             max_c_len = int(input_lengths.max())
#
#             self.example_ptr += cur_bsz
#
#             yield {
#                 'unique_id': unique_ids,
#                 'ids': ids,
#                 'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
#                 'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
#                 'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
#                 'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),
#                 'context_lens': input_lengths.to(self.device),
#                 'y1': y1[:cur_bsz],
#                 'y2': y2[:cur_bsz],
#                 'q_type': q_type[:cur_bsz],
#                 'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
#                 'start_mapping': start_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
#                 'end_mapping': end_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
#                 'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
#             }
#

def example_dict(examples):

    return {e.qas_id: e for e in examples}

def get_pickle_file(file_name):
    if "gz" in  file_name:
        return gzip.open(file_name, 'rb')
    else:
        return open(file_name, 'rb')

def get_or_load(file):
    with get_pickle_file(file) as fin:
        return pickle.load(fin)

def get_train_feature(args, is_train, tokenizer):
    data_file = args.train_file if is_train else args.predict_file
    select_file = args.train_select_file if is_train else args.predict_select_file
    examples_file = args.train_examples_file if is_train else args.predict_examples_file
    features_file = args.train_features_file if is_train else args.predict_features_file
    graph_file = args.train_graph_file if is_train else args.predict_graph_file
    entity_file = args.train_entity_file if is_train else args.predict_entity_file
    if not os.path.exists(examples_file):
        examples = read_hotpot_examples(select_file, data_file, entity_file)
        features = convert_examples_to_features(examples, tokenizer, max_seq_length=args.max_seq_length,
                                                      max_query_length=args.max_query_length)
        # with gzip.open(examples_file, 'wb') as fout:
        #     pickle.dump(examples, fout)
        # with gzip.open(features_file, 'wb') as fout:
        #     pickle.dump(features, fout)
        # pickle.dump(train_examples, gzip.open(examples_file, 'wb'))
        # pickle.dump(train_features, gzip.open(features_file, 'wb'))
    else:
        examples = get_or_load(examples_file)
        features = get_or_load(features_file)
    with open(graph_file, 'r', encoding='utf-8') as reader:
        graph = json.load(reader)
    return examples, features, graph


def main():
    args = set_config()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    model = BertForQuestionAnswering.from_pretrained(args.bert_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    global_step = 0
    if args.do_train:
        # load train data
        train_examples, train_features, train_graph = get_train_feature(args, args.do_train, tokenizer)
        train_data = DataIteratorPack(train_features, train_examples, train_graph, args.train_batch_size, device, sent_limit=25, entity_limit=80,
                                      sequential=False)

        # load dev data
        eval_examples, eval_features, eval_graph = get_train_feature(args, not args.do_train, tokenizer)
        eval_data = DataIteratorPack(eval_features, eval_examples, train_graph, args.predict_batch_size, device, sent_limit=25, entity_limit=80,
                                      sequential=False)
        with open(args.predict_file) as f:
            gold = json.load(f)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        total_train_loss = 0
        VERBOSE_STEP = 100
        grad_accumulate_step = 1
        best_dev_F1 = None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()

            # learning rate decay
            # if epoch > 1:
            #     args.learning_rate = args.learning_rate * args.decay
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = args.learning_rate
            #     print('lr = {}'.format(args.learning_rate))

            for step, batch in enumerate(train_data):
                # batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids = batch["context_idxs"]
                input_mask = batch["context_mask"]
                segment_ids = batch["segment_idxs"]
                start_positions = batch["y1"]
                end_positions = batch["y2"]
                q_types = batch["q_type"]

                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, q_types, batch=batch)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                if (global_step + 1) % grad_accumulate_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_train_loss += loss
                global_step += 1

                if global_step % VERBOSE_STEP == 0:
                    print("-- In Epoch{}: ".format(epoch))
                    print("Avg-LOSS: {}/batch/step: {}".format(total_train_loss/VERBOSE_STEP, global_step/VERBOSE_STEP))
                    total_train_loss = 0

                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            train_data.refresh()
            if args.do_predict:

                eval_examples_dict = example_dict(eval_examples)
                # eval_features_dict = example_dict(eval_features)

                logger.info("***** Running predictions *****")
                logger.info("  Num split examples = %d", len(eval_features))
                logger.info("  Batch size = %d", args.predict_batch_size)



                model.eval()
                all_results = []
                sp_dict = {}
                logger.info("Start evaluating")
                for step, batch in enumerate(eval_data):
                    # batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                    input_ids = batch["context_idxs"]
                    input_mask = batch["context_mask"]
                    segment_ids = batch["segment_idxs"]

                    if len(all_results) % 1000 == 0:
                        logger.info("Processing example: %d" % (len(all_results)))

                    with torch.no_grad():
                        batch_start_logits, batch_end_logits, batch_types, sp = model(input_ids, segment_ids, input_mask, batch=batch)
                    for i, example_index in enumerate(batch["ids"]):
                        start_logits = batch_start_logits[i].detach().cpu().tolist()
                        end_logits = batch_end_logits[i].detach().cpu().tolist()
                        # eval_feature = eval_features[example_index.item()]
                        unique_id = batch['unique_id'][i]

                        types = batch_types[i].detach().cpu().tolist()
                        all_results.append(RawResult(unique_id=unique_id,
                                                     start_logits=start_logits,
                                                     end_logits=end_logits,
                                                     types=types))
                    predict_support_np = torch.sigmoid(sp[:, :, 1]).data.cpu().numpy()
                    for i in range(predict_support_np.shape[0]):
                        cur_sp_pred = []
                        cur_id = batch['ids'][i]
                        for j in range(predict_support_np.shape[1]):

                            if j >= len(eval_examples_dict[cur_id].sent_names):
                                break
                            if predict_support_np[i, j] > args.sp_threshold:
                                cur_sp_pred.append(eval_examples_dict[cur_id].sent_names[j])
                        sp_dict.update({cur_id: cur_sp_pred})

                answer_dict = convert_to_tokens(eval_examples, eval_features, all_results)
                prediction = {'answer': answer_dict, 'sp': sp_dict}
                output_answer_sp_file = os.path.join(args.output_dir, "predictions_answer_sp_{}.json".format(epoch))
                with open(output_answer_sp_file, "w") as writer:
                    writer.write(json.dumps(prediction, indent=4) + "\n")
                eval(prediction, gold)

                metrics = evaluate(eval_examples_dict, answer_dict)
                print('hotpotqa epoch {:3d} | EM {:.4f} | F1 {:.4f}'.format(
                    epoch, metrics['exact_match'], metrics['f1']))

                output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(epoch))
                output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(epoch))
                output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(epoch))
                all_predictions = write_predictions(eval_examples, eval_features, all_results,
                                  args.n_best_size, args.max_answer_length,
                                  args.do_lower_case, output_prediction_file,
                                  output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                  args.version_2_with_negative, args.null_score_diff_threshold)

                metrics = evaluate(eval_examples_dict, all_predictions)
                print('squad epoch {:3d} | EM {:.4f} | F1 {:.4f}'.format(
                        epoch, metrics['exact_match'], metrics['f1']))
                dev_F1 = metrics['f1']

                eval_data.refresh()
                #learning rate decay
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= args.patience:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] /= 2.0
                        if param_group['lr'] < 1e-6:
                            stop_train = True
                            break
                        cur_patience = 0


if __name__ == "__main__":
    main()


