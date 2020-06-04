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
from qa_bert_based import *
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
IGNORE_INDEX = -100

def get_or_load_data(file):

    with open(file, 'rb') as fin:
        data = pickle.load(fin)

    return data

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 para_spans,
                 sent_spans,
                 entity_spans,
                 sup_fact_ids,
                 ans_type,
                 token_to_orig_map,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.para_spans = para_spans
        self.sent_spans = sent_spans
        self.entity_spans = entity_spans
        self.sup_fact_ids = sup_fact_ids
        self.ans_type = ans_type

        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position

class DataIteratorPack(object):
    def __init__(self, features, example_dict, bsz, device, sent_limit, n_layers, sequential=False):
        self.bsz = bsz
        self.device = device
        self.features = features
        self.example_dict = example_dict
        self.sequential = sequential
        self.sent_limit = sent_limit
        self.example_ptr = 0
        self.n_layers = n_layers
        if not sequential:
            shuffle(self.features)

    def refresh(self):
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        return self.example_ptr >= len(self.features)

    def __len__(self):
        return int(np.ceil(len(self.features)/self.bsz))

    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, 512)
        context_mask = torch.LongTensor(self.bsz, 512)
        segment_idxs = torch.LongTensor(self.bsz, 512)

        # Graph and Mappings
        query_mapping = torch.Tensor(self.bsz, 512).cuda(self.device)
        start_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
        end_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
        all_mapping = torch.Tensor(self.bsz, 512, self.sent_limit).cuda(self.device)

        # Label tensor
        y1 = torch.LongTensor(self.bsz).cuda(self.device)
        y2 = torch.LongTensor(self.bsz).cuda(self.device)
        q_type = torch.LongTensor(self.bsz).cuda(self.device)
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)

        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)
            cur_batch = self.features[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)

            ids = []
            max_sent_cnt = 0
            for mapping in [start_mapping, end_mapping, all_mapping, query_mapping]:
                mapping.zero_()
            is_support.fill_(IGNORE_INDEX)
            # is_support.fill_(0)  # BCE

            for i in range(len(cur_batch)):
                case = cur_batch[i]
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))

                for j in range(case.sent_spans[0][0] - 1):
                    query_mapping[i, j] = 1

                if case.ans_type == 0:
                    if len(case.end_position) == 0:
                        y1[i] = y2[i] = 0
                    elif case.end_position[0] < 512:
                        y1[i] = case.start_position[0]
                        y2[i] = case.end_position[0]
                    else:
                        y1[i] = y2[i] = 0
                    q_type[i] = 0
                elif case.ans_type == 1:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1
                elif case.ans_type == 2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2

                for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
                    is_sp_flag = j in case.sup_fact_ids
                    start, end = sent_span
                    if start < end:
                        is_support[i, j] = int(is_sp_flag)
                        all_mapping[i, start:end+1, j] = 1
                        start_mapping[i, j, start] = 1
                        end_mapping[i, j, end] = 1

                ids.append(case.qas_id)

                max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),
                'context_lens': input_lengths.to(self.device),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
                'start_mapping': start_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
                'end_mapping': end_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
            }


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
                                   ["unique_id", "start_logits", "end_logits"])


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
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

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
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
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
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
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

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(
                    0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0)
                )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

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

def convert_to_tokens(example, features, example_indices, y1, y2, q_type):
    answer_dict = dict()
    for i in range(len(q_type)):
        answer_text = ''
        qas_id = features[example_indices[i]].qas_id
        if q_type[i] == 0:
            doc_tokens = features[example_indices[i]].doc_tokens
            tok_tokens = doc_tokens[y1[i]: y2[i] + 1]
            tok_to_orig_map = features[example_indices[i]].token_to_orig_map
            if y2[i] < len(tok_to_orig_map):
                orig_doc_start = tok_to_orig_map[y1[i]]
                orig_doc_end = tok_to_orig_map[y2[i]]
                orig_tokens = example[example_indices[i]].doc_tokens[orig_doc_start:(orig_doc_end + 1)]
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
        elif q_type[i] == 1:
            answer_text = 'yes'
        elif q_type[i] == 2:
            answer_text = 'no'
        answer_dict[qas_id] = answer_text
    return answer_dict


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default="/DATA/disk1/baijinguo/HotpotQA", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--train_file", default="/DATA/disk1/baijinguo/HotpotQA/hotpot_dev_distractor_v1.json", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default="/DATA/disk1/baijinguo/HotpotQA/hotpot_dev_distractor_v1.json", type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=True, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16, type=int,
                        help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--theta',
                        type=int, default=10,
                        help="theta in loss func")
    parser.add_argument('--alpha',
                        type=float, default=2,
                        help="alpha in loss func")
    parser.add_argument('--beta',
                        type=int, default=4,
                        help="beta in loss func")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory () already exists and is not empty.")
    # os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    examples = read_hotpot_examples(args.train_file)

    # Prepare model
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    # if args.fp16:
    #     model.half()
    device = "cuda:0"
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # try:
    #     from apex.optimizers import FP16_Optimizer
    #     from apex.optimizers import FusedAdam
    # except ImportError:
    #     raise ImportError(
    #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    # optimizer = FusedAdam(optimizer_grouped_parameters,
    #                       lr=args.learning_rate,
    #                       bias_correction=False,
    #                       max_grad_norm=1.0)
    # if args.loss_scale == 0:
    #     optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    # else:
    #     optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    global_step = 0
    if args.do_train:
        # cached_train_features_file = args.train_file + '_{1}_{2}_{3}'.format(
        #     args.bert_model, str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        train_features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=64)

        # train_features = get_or_load_data(args.train_file)
        # print(train_features)
        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs


        logger.info("***** Running training *****")
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        all_input_ids = torch.tensor([f.doc_input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.doc_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.doc_segment_ids for f in train_features], dtype=torch.long)

        y1 = []
        y2 = []
        q_type = []
        # is_support =None
        # start_mapping = None
        # end_mapping = None
        # all_mapping = None
        # label = None
        for case in train_features:
            if case.ans_type == 0:
                if len(case.end_position) == 0:
                    y1.append(0)
                    y2.append(0)
                elif case.end_position[0] < 512:
                    y1.append(case.start_position[0])
                    y2.append(case.end_position[0])
                else:
                    y1.append(0)
                    y2.append(0)
                q_type.append(0)
            elif case.ans_type == 1:
                y1.append(IGNORE_INDEX)
                y2.append(IGNORE_INDEX)

                q_type.append(1)
            elif case.ans_type == 2:
                y1.append(IGNORE_INDEX)
                y2.append(IGNORE_INDEX)

                q_type.append(2)

        all_start_positions = torch.tensor([f for f in y1], dtype=torch.long)
        all_end_positions = torch.tensor([f for f in y2], dtype=torch.long)
        all_q_type = torch.tensor([f for f in q_type], dtype=torch.long)
        # print(all_q_type)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions, all_q_type)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        total_train_loss = 0
        VERBOSE_STEP = 50
        grad_accumulate_step = 1
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions, q_type = batch

                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, q_type)

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
                output_model_file = os.path.join(args.output_dir, "pytorch_model_{}.bin".format(epoch))
                if global_step % VERBOSE_STEP == 0:
                    if args.do_train:
                        torch.save(model_to_save.state_dict(), output_model_file)
                        # Load a trained model that you have fine-tuned
                        # model_state_dict = torch.load(output_model_file)
                        # model = BertForQuestionAnswering.from_pretrained(args.bert_model, state_dict=model_state_dict)
                    else:
                        model = BertForQuestionAnswering.from_pretrained(args.bert_model)

                        model.to(device)

        if args.do_predict:

            eval_examples = read_hotpot_examples(args.predict_file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, max_seq_length=512,
                                                          max_query_length=64)
            logger.info("***** Running predictions *****")
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.predict_batch_size)
            all_input_ids = torch.tensor([f.doc_input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.doc_input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.doc_segment_ids for f in train_features], dtype=torch.long)

            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                         batch_size=args.predict_batch_size)

            model.eval()
            answer_dict = {}
            logger.info("Start evaluating")
            for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
                if len(answer_dict) % 1000 == 0:
                    logger.info("Processing example: %d" % (len(answer_dict)))
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                with torch.no_grad():
                    batch_start, batch_end, batch_types = model(input_ids, segment_ids, input_mask)
                    # for i, example_index in enumerate(example_indices):
                        # print(example_index.item())
                    answer_dict_ = convert_to_tokens(eval_examples, eval_features, example_indices.numpy(), batch_start, batch_end, batch_types)

                    answer_dict.update(answer_dict_)
            prediction = {'answer': answer_dict}
            with open(os.path.join(args.output_dir, "prediction.json"), 'w') as f:
                for ids, values in answer_dict:

                    json.dump({ids: values}, f)
                    f.write("\n")


if __name__ == "__main__":
    main()


