from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import gzip
import pickle
import numpy as np
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization import BertTokenizer


class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_text,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 para_start_end_position,
                 sent_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 query_entities=None,
                 entities=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.query_entities = query_entities
        self.entities = entities

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 unique_id,
                 example_index,
                 tokens,
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
                 end_position=None,):

        self.qas_id = qas_id
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
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


def read_hotpot_examples(full_file, para_file, entity_file=None):
    with open(para_file, 'r', encoding='utf-8') as reader:
        para_data = json.load(reader)

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    with open(entity_file, 'r', encoding='utf-8') as reader:
        entity_data = json.load(reader)

    with open(entity_file, 'r', encoding='utf-8') as reader:
        query_entity_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    cnt = 0
    examples = []
    for case in tqdm(full_data):
        key = case['_id']
        qas_type = case['type']
        sup_facts = set([(sp[0], sp[1]) for sp in case['supporting_facts']])
        orig_answer_text = case['answer']

        sent_id = 0
        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        sent_start_end_position = []
        para_start_end_position = []
        entity_start_end_position = []
        ans_start_position, ans_end_position = [], []

        JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no'
        FIND_FLAG = False

        char_to_word_offset = []  # Accumulated along all sentences
        prev_is_whitespace = True

        # for debug
        titles = set()

        for paragraph in para_data[key]:
            title = paragraph[0]
            sents = paragraph[1]
            # if title in entity_data[key]:
            #     entities = entity_data[key][title]
            # else:
            #     entities = []

            titles.add(title)

            para_start_position = len(doc_tokens)

            for local_sent_id, sent in enumerate(sents):
                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)
                sent_names.append(local_sent_name)
                if local_sent_name in sup_facts:
                    sup_facts_sent_id.append(sent_id)
                sent_id += 1
                sent += " "

                sent_start_word_id = len(doc_tokens)
                sent_start_char_id = len(char_to_word_offset)

                for c in sent:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                sent_end_word_id = len(doc_tokens) - 1
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))

                # Answer char position
                answer_offsets = []
                offset = -1
                while True:
                    offset = sent.find(orig_answer_text, offset + 1)
                    if offset != -1:
                        answer_offsets.append(offset)
                    else:
                        break

                # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
                if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0:
                    FIND_FLAG = True
                    for answer_offset in answer_offsets:
                        start_char_position = sent_start_char_id + answer_offset
                        end_char_position = start_char_position + len(orig_answer_text) - 1
                        ans_start_position.append(char_to_word_offset[start_char_position])
                        ans_end_position.append(char_to_word_offset[end_char_position])

                # Find Entity Position
                # entity_pointer = 0
                # find_start_index = 0
                # for entity in entities:
                #     entity_text, entity_type = clean_entity(entity)
                #     entity_offset = sent.find(entity_text, find_start_index)
                #     if entity_offset != -1:
                #         start_char_position = sent_start_char_id + entity_offset
                #         end_char_position = start_char_position + len(entity_text) - 1
                #         ent_start_position = char_to_word_offset[start_char_position]
                #         ent_end_position = char_to_word_offset[end_char_position]
                #         entity_start_end_position.append((ent_start_position, ent_end_position, entity_text, entity_type))
                #         entity_pointer += 1
                #         find_start_index = entity_offset + len(entity_text)
                #     else:
                #         break
                # entities = entities[entity_pointer:]

                # Truncate longer document
                if len(doc_tokens) > 382:
                    break
            para_end_position = len(doc_tokens) - 1
            para_start_end_position.append((para_start_position, para_end_position, title))

        if len(ans_end_position) > 1:
            cnt += 1

        query_entities = dict()
        para_entities = dict()

        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=case['question'],
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            para_start_end_position=para_start_end_position,
            sent_start_end_position=sent_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,
            end_position=ans_end_position,
            query_entities=query_entities,
            entities=para_entities)
        examples.append(example)
    print(cnt)
    return examples


def read_hotpot_examples(full_file, para_file, entity_file=None):
    with open(para_file, 'r', encoding='utf-8') as reader:
        para_data = json.load(reader)

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    # with open(entity_file, 'r', encoding='utf-8') as reader:
    #     entity_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    cnt = 0
    examples = []
    for case in tqdm(full_data[:100]):
        key = case['_id']
        qas_type = case['type']
        sup_facts = set([(sp[0], sp[1]) for sp in case['supporting_facts']])
        orig_answer_text = case['answer']

        sent_id = 0
        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        sent_start_end_position = []
        para_start_end_position = []
        entity_start_end_position = []
        ans_start_position, ans_end_position = [], []

        JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no'
        FIND_FLAG = False

        char_to_word_offset = []  # Accumulated along all sentences
        prev_is_whitespace = True

        # for debug
        titles = set()

        for paragraph in para_data[key]:
            title = paragraph[0]
            sents = paragraph[1]
            # if title in entity_data[key]:
            #     entities = entity_data[key][title]
            # else:
            #     entities = []

            titles.add(title)

            para_start_position = len(doc_tokens)

            for local_sent_id, sent in enumerate(sents):
                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)
                sent_names.append(local_sent_name)
                if local_sent_name in sup_facts:
                    sup_facts_sent_id.append(sent_id)
                sent_id += 1
                sent += " "

                sent_start_word_id = len(doc_tokens)
                sent_start_char_id = len(char_to_word_offset)

                for c in sent:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                sent_end_word_id = len(doc_tokens) - 1
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))

                # Answer char position
                answer_offsets = []
                offset = -1
                while True:
                    offset = sent.find(orig_answer_text, offset + 1)
                    if offset != -1:
                        answer_offsets.append(offset)
                    else:
                        break

                # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
                if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0:
                    FIND_FLAG = True
                    for answer_offset in answer_offsets:
                        start_char_position = sent_start_char_id + answer_offset
                        end_char_position = start_char_position + len(orig_answer_text) - 1
                        ans_start_position.append(char_to_word_offset[start_char_position])
                        ans_end_position.append(char_to_word_offset[end_char_position])

                # Find Entity Position
                # entity_pointer = 0
                # find_start_index = 0
                # for entity in entities:
                #     entity_text, entity_type = clean_entity(entity)
                #     entity_offset = sent.find(entity_text, find_start_index)
                #     if entity_offset != -1:
                #         start_char_position = sent_start_char_id + entity_offset
                #         end_char_position = start_char_position + len(entity_text) - 1
                #         ent_start_position = char_to_word_offset[start_char_position]
                #         ent_end_position = char_to_word_offset[end_char_position]
                #         entity_start_end_position.append((ent_start_position, ent_end_position, entity_text, entity_type))
                #         entity_pointer += 1
                #         find_start_index = entity_offset + len(entity_text)
                #     else:
                #         break
                # entities = entities[entity_pointer:]

                # Truncate longer document
                # if len(doc_tokens) > 382:
                #     break
            para_end_position = len(doc_tokens) - 1
            para_start_end_position.append((para_start_position, para_end_position, title))
        if len(ans_end_position) > 1:
            cnt += 1

        query_entities = dict()
        para_entities = dict()

        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=case['question'],
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            para_start_end_position=para_start_end_position,
            sent_start_end_position=sent_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,
            end_position=ans_end_position,
            query_entities=query_entities,
            entities=para_entities)
        examples.append(example)
    print(cnt)
    return examples


# def read_hotpot_examples(full_file, is_train=False):
#
#     with open(full_file, 'r', encoding='utf-8') as reader:
#         full_data = json.load(reader)
#
#     def is_whitespace(c):
#         if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
#             return True
#         return False
#
#     cnt = 0
#     examples = []
#     sample_paras = dict()
#     for case in tqdm(full_data[:100]):
#         key = case['_id']
#         para_data = case["context"]
#         qas_type = case['type']
#         sup_facts = set([(sp[0], sp[1]) for sp in case['supporting_facts']])
#
#         #TODO:entity
#         query_entities = dict()
#         para_entities = dict()
#         query_entities[key] = get_entities(case["question"])
#
#         #rerank para
#         sup_paras_titles = set()
#         for sp in case['supporting_facts']:
#             # if sp[0] not in sup_paras_titles:
#                 sup_paras_titles.add(sp[0])
#         extend = np.random.randint(len(para_data))
#         sup_paras_titles.add(extend)
#
#         orig_answer_text = case['answer']
#
#         sent_id = 0
#         doc_tokens = []
#         sent_names = []  # (title, sent_id)
#         sup_facts_sent_id = []
#         sent_start_end_position = []
#         para_start_end_position = []
#         ans_start_position, ans_end_position = [], []
#
#         JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no'
#         FIND_FLAG = False
#
#         char_to_word_offset = []  # Accumulated along all sentences
#         prev_is_whitespace = True
#
#         titles = set()
#         # gold_para = set()
#         rerand = []
#
#         #TODO:pada rerank
#         for sup_title in sup_paras_titles:
#             for para_id, paragraph in enumerate(para_data):
#                 title = paragraph[0]
#                 if title == sup_title:
#                     rerand.append(para_id)
#         # # extend = np.random.randint(len(para_data))
#         # #negative example
#         # if is_train:
#         #     if extend not in rerand: rerand.append(extend)
#         sample_paras[key] = []
#         for para_id in rerand:
#         # for para_id, paragraph in enumerate(para_data):
#             paragraph = para_data[para_id]
#
#             sample_paras[key].append(paragraph)
#             title = paragraph[0]
#             sents = paragraph[1]
#
#             #get entities
#             para_entities[key][title] = get_entities(sents)
#
#             titles.add(title)
#
#             # # jungle gold_para
#             # for local_sent_id, sent in enumerate(sents):
#             #     # Determine the global sent id for supporting facts
#             #     local_sent_name = (title, local_sent_id)
#             #
#             #     if local_sent_name in sup_facts:
#             #         gold_para.add(para_id)
#             # if para_id not in gold_para:
#             #     continue
#             para_start_position = len(doc_tokens)
#
#             for local_sent_id, sent in enumerate(sents):
#                 # Determine the global sent id for supporting facts
#                 local_sent_name = (title, local_sent_id)
#
#                 if local_sent_name in sup_facts:
#                     sup_facts_sent_id.append(sent_id)
#
#                 sent_names.append(local_sent_name)
#
#                 sent_id += 1
#                 sent += " "
#
#                 sent_start_word_id = len(doc_tokens)
#                 sent_start_char_id = len(char_to_word_offset)
#                 for c in sent:
#                     if is_whitespace(c):
#                         prev_is_whitespace = True
#                     else:
#                         if prev_is_whitespace:
#                             doc_tokens.append(c)
#                         else:
#                             doc_tokens[-1] += c
#                         prev_is_whitespace = False
#                     char_to_word_offset.append(len(doc_tokens) - 1)
#                 sent_end_word_id = len(doc_tokens) - 1
#                 sent_start_end_position.append((sent_start_word_id, sent_end_word_id))
#
#                 answer_offsets = []
#                 offset = -1
#                 while True:
#                     # find(str, start, end)
#                     offset = sent.find(orig_answer_text, offset + 1)
#                     if offset != -1:
#                         answer_offsets.append(offset)
#                     else:
#                         break
#                 # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
#                 if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0:
#                     FIND_FLAG = True
#                     for answer_offset in answer_offsets:
#                         start_char_position = sent_start_char_id + answer_offset
#                         end_char_position = start_char_position + len(orig_answer_text) - 1
#
#                         ans_start_position.append(char_to_word_offset[start_char_position])
#                         ans_end_position.append(char_to_word_offset[end_char_position])
#                 #         doc_tokens.append(para_tokens)
#                 if len(doc_tokens) > 382:
#                     break
#             para_end_position = len(doc_tokens) - 1
#             para_start_end_position.append((para_start_position, para_end_position, title))
#         if len(ans_end_position) > 1:
#             cnt += 1
#         example = Example(
#             qas_id=key,
#             qas_type=qas_type,
#             doc_tokens=doc_tokens,
#             question_text=case['question'],
#             sent_num=sent_id + 1,
#             sent_names=sent_names,
#             sup_fact_id=sup_facts_sent_id,
#             para_start_end_position=para_start_end_position,
#             sent_start_end_position=sent_start_end_position,
#             orig_answer_text=orig_answer_text,
#             start_position=ans_start_position,
#             end_position=ans_end_position,
#             query_entities=query_entities,
#             entities=para_entities)
#
#         examples.append(example)
#     # with open(r"E:\DATA\HotpotQA\sample_examples.json", 'w') as writer:
#     #     writer.write(json.dumps(sample_paras, indent=4) + "\n")
#     # json.dump(sample_paras, open(r"E:\DATA\HotpotQA\sample_examples.json", 'w'))
#     return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    features = []
    failed = 0
    unique_id = 1000000000
    for (example_index, example) in enumerate(tqdm(examples)):
        if example.orig_answer_text == 'yes':
            ans_type = 1
        elif example.orig_answer_text == 'no':
            ans_type = 2
        else:
            ans_type = 0

        query_tokens = ["[CLS]"] + tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length - 1:
            query_tokens = query_tokens[:max_query_length - 1]
        query_tokens.append("[SEP]")

        para_spans = []
        entity_spans = []
        sentence_spans = []
        all_doc_tokens = []
        orig_to_tok_index = []
        orig_to_tok_back_index = []
        tok_to_orig_index = [0] * len(query_tokens)

        all_doc_tokens += ["[CLS]"] + tokenizer.tokenize(example.question_text)
        if len(all_doc_tokens) > max_query_length - 1:
            all_doc_tokens = all_doc_tokens[:max_query_length - 1]
        all_doc_tokens.append("[SEP]")

        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)

        def relocate_tok_span(orig_start_position, orig_end_position, orig_text, tokenizer):
            if orig_start_position is None:
                return 0, 0

            # global tokenizer
            nonlocal orig_to_tok_index, example, all_doc_tokens

            tok_start_position = orig_to_tok_index[orig_start_position]
            if orig_end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            # Make answer span more accurate.
            return _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

        ans_start_position, ans_end_position = [], []
        for ans_start_pos, ans_end_pos in zip(example.start_position, example.end_position):
            s_pos, e_pos = relocate_tok_span(ans_start_pos, ans_end_pos, example.orig_answer_text, tokenizer)
            ans_start_position.append(s_pos)
            ans_end_position.append(e_pos)

        for sent_span in example.sent_start_end_position:
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                continue
            sent_start_position = orig_to_tok_index[sent_span[0]]
            sent_end_position = orig_to_tok_back_index[sent_span[1]]
            sentence_spans.append((sent_start_position, sent_end_position))

        for para_span in example.para_start_end_position:
            if para_span[0] >= len(orig_to_tok_index) or para_span[0] >= para_span[1]:
                continue
            para_start_position = orig_to_tok_index[para_span[0]]
            para_end_position = orig_to_tok_back_index[para_span[1]]
            para_spans.append((para_start_position, para_end_position, para_span[2]))

        # Padding Document
        all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + ["[SEP]"]
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)

        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_input_mask.append(0)
            doc_segment_ids.append(0)

        # Padding Question
        query_input_mask = [1] * len(query_input_ids)
        query_segment_ids = [0] * len(query_input_ids)

        while len(query_input_ids) < max_query_length:
            query_input_ids.append(0)
            query_input_mask.append(0)
            query_segment_ids.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length

        # Dropout out-of-bound span
        sentence_spans = sentence_spans[:_largest_valid_index(sentence_spans, max_seq_length)]

        sup_fact_ids = example.sup_fact_id
        sent_num = len(sentence_spans)
        sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
        if len(sup_fact_ids) != len(example.sup_fact_id):
            failed += 1

        features.append(
            InputFeatures(
                          qas_id=example.qas_id,
                          unique_id=unique_id,
                          example_index=example_index,
                          tokens=all_doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          doc_segment_ids=doc_segment_ids,
                          query_tokens=query_tokens,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          para_spans=para_spans,
                          sent_spans=sentence_spans,
                          entity_spans=entity_spans,
                          sup_fact_ids=sup_fact_ids,
                          ans_type=ans_type,
                          token_to_orig_map=tok_to_orig_index,
                          start_position=ans_start_position,
                          end_position=ans_end_position)
        )
        unique_id += 1
    return features

def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--example_output", default=r"E:\DATA\HotpotQA\dev_examples.pkl", type=str)
    parser.add_argument("--feature_output", default=r"E:\DATA\HotpotQA\dev_features.pkl", type=str)

    # Other parameters
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size for predictions.")
    parser.add_argument("--full_data", default=r"E:\DATA\HotpotQA\hotpot_dev_distractor_v1.json", type=str)

    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("start read_hotpot_examples")
    examples = read_hotpot_examples(args.full_data, True)
    print("end read_hotpot_examples")
    with gzip.open(args.example_output, 'wb') as fout:
        pickle.dump(examples, fout)
    print("start convert_examples_to_features")
    features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    print("end convert_examples_to_features")
    with gzip.open(args.feature_output, 'wb') as fout:
        pickle.dump(features, fout)
