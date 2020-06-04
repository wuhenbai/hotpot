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

from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from model.GNN import BertForQuestionAnswering, GraphFusionNet, DFGN_Bert
from transformers import RobertaTokenizer, BertTokenizer, AdamW

from create_graph import iter_data
from util import evaluate
from hotpot_evaluate_v1 import eval
from config import set_config

import fitlog

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
os.makedirs("./logs", exist_ok=True)
fitlog.set_log_dir("./logs")

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
                    if start_index >= len(feature.doc_tokens):
                        continue
                    if end_index >= len(feature.doc_tokens):
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
                    tok_tokens = feature.doc_tokens[pred.start_index:(pred.end_index + 1)]
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


def convert_to_tokens(example, features, ids, y1, y2, q_type):
    answer_dict = dict()
    for i, qid in enumerate(ids):
        answer_text = ''
        if q_type[i] == 0:
            doc_tokens = features[qid].doc_tokens
            tok_tokens = doc_tokens[y1[i]: y2[i] + 1]
            tok_to_orig_map = features[qid].token_to_orig_map
            if y2[i] < len(tok_to_orig_map):
                orig_doc_start = tok_to_orig_map[y1[i]]
                orig_doc_end = tok_to_orig_map[y2[i]]
                orig_tokens = example[qid].doc_tokens[orig_doc_start:(orig_doc_end + 1)]
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
        answer_dict[qid] = answer_text
    return answer_dict


def example_dict(examples):
    return {e.qas_id: e for e in examples}


def get_pickle_file(file_name):
    if "gz" in file_name:
        return gzip.open(file_name, 'rb')
    else:
        return open(file_name, 'rb')


def get_or_load(file):
    with get_pickle_file(file) as fin:
        return pickle.load(fin)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 unique_id,
                 example_index,
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
        self.unique_id = unique_id
        self.example_index = example_index
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


def get_unique_id(features):
    new_features = []
    unique_id = 1000000000
    for (example_index, feature) in enumerate(tqdm(features)):
        new_features.append(
            InputFeatures(
                qas_id=feature.qas_id,
                example_index=example_index,
                unique_id=unique_id,
                doc_tokens=feature.doc_tokens,
                doc_input_ids=feature.doc_input_ids,
                doc_input_mask=feature.doc_input_mask,
                doc_segment_ids=feature.doc_segment_ids,
                query_tokens=feature.query_tokens,
                query_input_ids=feature.query_input_ids,
                query_input_mask=feature.query_input_mask,
                query_segment_ids=feature.query_segment_ids,
                para_spans=feature.para_spans,
                sent_spans=feature.sent_spans,
                entity_spans=feature.entity_spans,
                sup_fact_ids=feature.sup_fact_ids,
                ans_type=feature.ans_type,
                token_to_orig_map=feature.token_to_orig_map,
                start_position=feature.start_position,
                end_position=feature.end_position))
        unique_id += 1
    return new_features


def get_train_feature(args, is_train, tokenizer):
    data_file = args.train_file if is_train else args.predict_file
    select_file = args.train_select_file if is_train else args.predict_select_file
    examples_file = args.train_examples_file if is_train else args.predict_examples_file
    features_file = args.train_features_file if is_train else args.predict_features_file
    graph_file = args.train_graph_file if is_train else args.predict_graph_file
    entity_file = args.train_entity_file if is_train else args.predict_entity_file
    query_entity_file = args.train_query_entity_file if is_train else args.predict_query_entity_file
    print(examples_file)
    # if not os.path.exists(examples_file):
    #     examples = read_hotpot_examples(select_file, data_file, entity_file)
    #     features = convert_examples_to_features(examples, tokenizer, max_seq_length=args.max_seq_length,
    #                                                   max_query_length=args.max_query_length)
    #     graph = iter_data(features, examples, query_entity_file)
    #     with gzip.open(examples_file, 'wb') as fout:
    #         pickle.dump(examples, fout)
    #     with gzip.open(features_file, 'wb') as fout:
    #         pickle.dump(features, fout)
    #     pickle.dump(graph, gzip.open(graph_file, 'wb'))
    #     # pickle.dump(train_examples, gzip.open(examples_file, 'wb'))
    #     # pickle.dump(train_features, gzip.open(features_file, 'wb'))
    # else:
    #     examples = get_or_load(examples_file)
    #     features = get_or_load(features_file)
    #     # features = get_unique_id(features)
    #     graph = get_or_load(graph_file)
    #     # graph = iter_data(features, examples, query_entity_file)
    # # with open(graph_file, 'r', encoding='utf-8') as reader:
    # #     graph = json.load(reader)
    examples = get_or_load(examples_file)
    features = get_or_load(features_file)
    # features = get_unique_id(features)
    graph = get_or_load(graph_file)
    return examples, features, graph


criterion = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
binary_criterion = torch.nn.BCEWithLogitsLoss(size_average=True)


def compute_loss(batch, start, end, sp, Type, masks, args):
    loss1 = criterion(start, batch['y1']) + criterion(end, batch['y2'])

    loss2 = args.type_lambda * criterion(Type, batch['q_type'])
    loss3 = args.sp_lambda * criterion(sp.view(-1, 2), batch['is_support'].long().view(-1))
    loss = loss1 + loss2 + loss3

    loss4 = 0
    if args.bfs_clf and len(masks) > 0:
        for l in range(args.n_layers):
            pred_mask = masks[l].view(-1)
            gold_mask = batch['bfs_mask'][:, l, :].contiguous().view(-1)
            loss4 += binary_criterion(pred_mask, gold_mask)
        loss += args.bfs_lambda * loss4

    return loss, loss1, loss2, loss3, loss4


def main():
    args = set_config()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Prepare model
    # encoder = BertForQuestionAnswering.from_pretrained(args.bert_model)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # encoder.to(device)
    # encoder.eval()
    # #freeze bert
    # # for name, param in model.named_parameters():
    # #     if "bert" in name:
    # #         param.requires_grad = False
    #
    # model = GraphFusionNet(args)
    model = DFGN_Bert.from_pretrained("/DATA/disk1/baijinguo/BERT_Pretrained/bert-base-uncased", graph_config=args)
    model.to(device)
    model.train()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    global_step = 0

    if args.do_train:
        # load train data
        train_examples, train_features, train_graph = get_train_feature(args, args.do_train, tokenizer)
        train_examples_dict = example_dict(train_examples)
        train_data = DataIteratorPack(train_features, train_examples_dict, train_graph, args.train_batch_size, device,
                                      sent_limit=40, entity_limit=80,
                                      n_layers=args.n_layers, sequential=False)
        # (features, example_dict, graph_dict, bsz, device, sent_limit, entity_limit, n_layers = 2,
        # entity_type_dict = None, sequential = False,)
        # load dev data
        eval_examples, eval_features, eval_graph = get_train_feature(args, not args.do_train, tokenizer)
        eval_examples_dict = example_dict(eval_examples)
        eval_data = DataIteratorPack(eval_features, eval_examples_dict, eval_graph, args.predict_batch_size, device,
                                     sent_limit=40, entity_limit=80,
                                     n_layers=args.n_layers, sequential=False)
        with open(args.predict_file) as f:
            gold = json.load(f)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        cur_patience = 0
        VERBOSE_STEP = 100
        best_dev_F1 = None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # model.train()
            model.train()

            total_train_loss = [0] * 5

            for step, batch in enumerate(train_data):
                # batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids = batch["context_idxs"]
                input_mask = batch["context_mask"]
                segment_ids = batch["segment_idxs"]
                # start_positions = batch["y1"]
                # end_positions = batch["y2"]
                # q_types = batch["q_type"]

                # context_encoding = encoder(input_ids, segment_ids, input_mask)
                #
                # # loss_list = model(context_encoding, batch=batch)
                # start, end, sp, Type, softmask, ent, yp1, yp2 = model(context_encoding, batch=batch, return_yp=True)

                start, end, sp, Type, softmask, ent, yp1, yp2 = model(input_ids, segment_ids, input_mask, batch=batch,
                                                                      return_yp=True, is_train=True)
                loss_list = compute_loss(batch, start, end, sp, Type, softmask, args)

                if args.gradient_accumulation_steps > 1:
                    loss_list = loss_list / args.gradient_accumulation_steps

                loss_list[0].backward()

                if (global_step + 1) % args.grad_accumulate_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                global_step += 1

                for i, l in enumerate(loss_list):
                    if not isinstance(l, int):
                        total_train_loss[i] += l.item()

                if global_step % VERBOSE_STEP == 0:
                    print("-- In Epoch{}: ".format(epoch))
                    for i, l in enumerate(total_train_loss):
                        print("Avg-LOSS{}/batch/step: {}".format(i, l / VERBOSE_STEP))
                    total_train_loss = [0] * 5

                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            train_data.refresh()
            if args.do_predict:

                eval_examples_dict = example_dict(eval_examples)
                eval_features_dict = example_dict(eval_features)

                logger.info("***** Running predictions *****")
                logger.info("  Num split examples = %d", len(eval_features))
                logger.info("  Batch size = %d", args.predict_batch_size)

                model.eval()
                all_results = []
                answer_dict = {}
                sp_dict = {}
                total_test_loss = [0] * 5
                logger.info("Start evaluating")
                for step, batch in enumerate(eval_data):
                    # batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                    input_ids = batch["context_idxs"]
                    input_mask = batch["context_mask"]
                    segment_ids = batch["segment_idxs"]

                    if len(sp_dict) % 1000 == 0:
                        logger.info("Processing example: %d" % (len(all_results)))

                    with torch.no_grad():
                        start, end, sp, Type, softmask, ent, yp1, yp2 = model(input_ids, segment_ids, input_mask,
                                                                              batch=batch, return_yp=True)
                        # context_encoding = encoder(input_ids, segment_ids, input_mask)
                        #
                        # # loss_list = model(context_encoding, batch=batch)
                        # start, end, sp, Type, softmask, ent, yp1, yp2 = model(context_encoding, batch=batch,
                        #                                                       return_yp=True)
                        loss_list = compute_loss(batch, start, end, sp, Type, softmask, args)
                        Type = Type.argmax(dim=1)

                        # batch_start_logits, batch_end_logits, batch_types, sp = model(input_ids, segment_ids, input_mask, batch=batch)
                    for i, l in enumerate(loss_list):
                        if not isinstance(l, int):
                            total_test_loss[i] += l.item()

                    answer_dict_ = convert_to_tokens(eval_examples_dict, eval_features_dict, batch['ids'],
                                                     yp1.data.cpu().numpy().tolist(),
                                                     yp2.data.cpu().numpy().tolist(),
                                                     Type.cpu().numpy())

                    answer_dict.update(answer_dict_)
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

                # for i, l in enumerate(total_train_loss):
                #     print("Avg-LOSS{}/batch/step: {}".format(i, l / len(eval_features)))

                prediction = {'answer': answer_dict, 'sp': sp_dict}
                output_answer_sp_file = os.path.join(args.output_dir, "predictions_answer_sp_{}.json".format(epoch))
                with open(output_answer_sp_file, 'w') as f:
                    json.dump(prediction, f)

                # record results
                metrics = eval(prediction, gold)
                for i, l in enumerate(total_train_loss):
                    metrics["LOSS{}".format(i)] = l / len(eval_features)
                    print("Avg-LOSS{}/batch/step: {}".format(i, l / len(eval_features)))

                # fitlog.add_best_metric({"Test": metrics})

                metrics = evaluate(eval_examples_dict, answer_dict)
                print('hotpotqa | EM {:.4f} | F1 {:.4f}'.format(metrics['exact_match'], metrics['f1']))
                eval_data.refresh()

                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                    logger.info("model save in %s" % output_model_file)
                    # model_to_save.save_pretrained(output_model_file)
                    # tokenizer.save_pretrained(args.output_dir)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    cur_patience = 0

                    # model = AlbertForQuestionAnswering.from_pretrained(args.output_dir, force_download=True)
                    # # tokenizer = AlbertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
                    # model.to(device)
                else:
                    cur_patience += 1
                    if cur_patience >= 3:
                        # for param_group in optimizer.param_groups:
                        #    param_group['lr'] /= 2.0
                        # if param_group['lr'] < 1e-8:
                        #    stop_train = True
                        break


if __name__ == "__main__":
    main()


