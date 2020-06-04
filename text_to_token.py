from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import gzip
import pickle
from tqdm import tqdm
import torch
import numpy as np
from numpy.random import shuffle
from utils import create_hierarchical_graph, bfs_step, normalize_answer
IGNORE_INDEX = -100

from pytorch_pretrained_bert.tokenization import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position


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
                 end_position=None,
                 answer=None):

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
        self.answer = answer

def clean_entity(entity):
    # Type = entity[1]
    Text = entity
    # if Type == "DATE" and ',' in Text:
    #     Text = Text.replace(' ,', ',')
    if '?' in Text:
        Text = Text.split('?')[0]
    Text = Text.replace("\'\'", "\"")
    Text = Text.replace("# ", "#")
    return Text


def check_in_full_paras(answer, paras):
    full_doc = ""
    for p in paras:
        full_doc += " ".join(p[1])
    return answer in full_doc



def read_hotpot_examples(para_file, full_file, entity_file):
    with open(para_file, 'r', encoding='utf-8') as reader:
        para_data = json.load(reader)

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    with open(entity_file, 'r', encoding='utf-8') as reader:
        entity_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    cnt = 0
    examples = []
    entities_len = []
    for case in tqdm(full_data[:10]):
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
            if title in entity_data[key]:
                entities = entity_data[key][title]
            else:
                entities = []
            entities_len.append(len(entities))
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
                entity_pointer = 0
                find_start_index = 0
                for entity in entities:
                    entity_text = clean_entity(entity)
                    entity_offset = sent.find(entity_text, find_start_index)
                    if entity_offset != -1:
                        start_char_position = sent_start_char_id + entity_offset
                        end_char_position = start_char_position + len(entity_text) - 1
                        ent_start_position = char_to_word_offset[start_char_position]
                        ent_end_position = char_to_word_offset[end_char_position]

                        entity_start_end_position.append((ent_start_position, ent_end_position, entity_text))
                        entity_pointer += 1
                        find_start_index = entity_offset + len(entity_text)
                    else:
                        break
                entities = entities[entity_pointer:]
                # Truncate longer document
                if len(doc_tokens) > 382:
                    break
            para_end_position = len(doc_tokens) - 1
            para_start_end_position.append((para_start_position, para_end_position, title))


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
            entity_start_end_position=entity_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,
            end_position=ans_end_position,)
        examples.append(example)
    print(cnt)
    return examples


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

        def relocate_tok_span(orig_start_position, orig_end_position, orig_text):
            if orig_start_position is None:
                return 0, 0

            global tokenizer
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
            s_pos, e_pos = relocate_tok_span(ans_start_pos, ans_end_pos, example.orig_answer_text)
            ans_start_position.append(s_pos)
            ans_end_position.append(e_pos)

        for entity_span in example.entity_start_end_position:
            ent_start_position, ent_end_position \
                = relocate_tok_span(entity_span[0], entity_span[1], entity_span[2])

            entity_spans.append((ent_start_position, ent_end_position, entity_span[2]))

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
        entity_spans = entity_spans[:_largest_valid_index(entity_spans, max_seq_length)]
        sentence_spans = sentence_spans[:_largest_valid_index(sentence_spans, max_seq_length)]

        sup_fact_ids = example.sup_fact_id
        sent_num = len(sentence_spans)
        sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
        if len(sup_fact_ids) != len(example.sup_fact_id):
            failed += 1

        features.append(
            InputFeatures(qas_id=example.qas_id,
                          unique_id=unique_id,
                          example_index=example_index,
                          doc_tokens=all_doc_tokens,
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
                          end_position=ans_end_position,
                          answer=example.orig_answer_text)
        )
        unique_id += 1
    print(failed)
    return features


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


class DataIteratorPack(object):
    def __init__(self, features, example_dict, graph_dict, bsz, device, sent_limit, entity_limit, n_layers,
                 entity_type_dict=None, sequential=False,):
        self.bsz = bsz
        self.device = device
        self.features = features
        self.example_dict = example_dict
        self.graph_dict = graph_dict
        self.entity_type_dict = entity_type_dict
        self.sequential = sequential
        self.sent_limit = sent_limit
        self.entity_limit = entity_limit
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
        context_idxs = torch.LongTensor(self.bsz, 512).cuda(self.device)
        context_mask = torch.LongTensor(self.bsz, 512).cuda(self.device)
        segment_idxs = torch.LongTensor(self.bsz, 512).cuda(self.device)

        # Graph and Mappings
        entity_graphs = torch.Tensor(self.bsz, self.entity_limit, self.entity_limit).cuda(self.device)
        query_mapping = torch.Tensor(self.bsz, 512).cuda(self.device)
        start_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
        end_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
        all_mapping = torch.Tensor(self.bsz, 512, self.sent_limit).cuda(self.device)
        entity_mapping = torch.Tensor(self.bsz, self.entity_limit, 512).cuda(self.device)

        # Label tensor
        y1 = torch.LongTensor(self.bsz).cuda(self.device)
        y2 = torch.LongTensor(self.bsz).cuda(self.device)

        answer_mapping = torch.Tensor(self.bsz, 512).cuda(self.device)
        q_type = torch.LongTensor(self.bsz).cuda(self.device)
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)

        start_mask = torch.FloatTensor(self.bsz, self.entity_limit).cuda(self.device)
        start_mask_weight = torch.FloatTensor(self.bsz, self.entity_limit).cuda(self.device)
        bfs_mask = torch.FloatTensor(self.bsz, self.n_layers, self.entity_limit).cuda(self.device)
        entity_label = torch.LongTensor(self.bsz).cuda(self.device)

        while True:
            # if self.example_ptr >= len(self.features):
            if self.example_ptr >= 10:
                break
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)
            cur_batch = self.features[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)

            ids = []
            max_sent_cnt = 0
            max_entity_cnt = 0
            for mapping in [start_mapping, end_mapping, all_mapping, entity_mapping, query_mapping]:
                mapping.zero_()
            entity_label.fill_(IGNORE_INDEX)
            is_support.fill_(IGNORE_INDEX)
            answer_mapping.fill_(IGNORE_INDEX)
            # is_support.fill_(0)  # BCE

            for i in range(len(cur_batch)):
                case = cur_batch[i]
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))

                for j in range(case.sent_spans[0][0] - 1):
                    query_mapping[i, j] = 1

                tem_graph = self.graph_dict[case.qas_id]
                adj = torch.from_numpy(tem_graph['adj'])
                start_entities = torch.from_numpy(tem_graph['start_entities'])
                entity_graphs[i] = adj
                for l in range(self.n_layers):
                    bfs_mask[i][l].copy_(start_entities)
                    start_entities = bfs_step(start_entities, adj)

                start_mask[i].copy_(start_entities)
                start_mask_weight[i, :tem_graph['entity_length']] = start_entities.byte().any().float()
                # if case.ans_type == 0:
                #     num_ans = len(case.start_position)
                #     if num_ans == 0:
                #         y1[i] = y2[i] = 0
                #     else:
                #         ans_id = choice(range(num_ans))
                #         start_position = case.start_position[ans_id]
                #         end_position = case.end_position[ans_id]
                #         if end_position < 512:
                #             y1[i] = start_position
                #             y2[i] = end_position
                #         else:
                #             y1[i] = y2[i] = 0
                #     q_type[i] = 0
                if case.ans_type == 0:
                    if len(case.end_position) == 0:
                        y1[i] = y2[i] = 0
                    elif case.end_position[0] < 512:
                        y1[i] = case.start_position[0]
                        y2[i] = case.end_position[0]

                        answer_mapping[i][:len(case.doc_tokens)].fill_(0)
                        for x in range(y1[i], y2[i]+1):
                            answer_mapping[i][x] = 1
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
                answer = self.example_dict[case.qas_id].orig_answer_text
                for j, entity_span in enumerate(case.entity_spans[:self.entity_limit]):
                    _, _, ent, _ = entity_span
                    if normalize_answer(ent) == normalize_answer(answer):
                        entity_label[i] = j
                        break

                entity_mapping[i] = torch.from_numpy(tem_graph['entity_mapping'])
                max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))
                max_entity_cnt = max(max_entity_cnt, tem_graph['entity_length'])

            entity_lengths = (entity_mapping[:cur_bsz] > 0).float().sum(dim=2)
            entity_lengths = torch.where((entity_lengths > 0), entity_lengths, torch.ones_like(entity_lengths))
            entity_mask = (entity_mapping > 0).any(2).float()

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),
                'entity_graphs': entity_graphs[:cur_bsz, :max_entity_cnt, :max_entity_cnt].contiguous(),
                'context_lens': input_lengths.to(self.device),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
                'start_mapping': start_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
                'end_mapping': end_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                'entity_mapping': entity_mapping[:cur_bsz, :max_entity_cnt, :max_c_len],
                'entity_lens': entity_lengths[:cur_bsz, :max_entity_cnt],
                'entity_mask': entity_mask[:cur_bsz, :max_entity_cnt],
                'entity_label': entity_label[:cur_bsz],
                'start_mask': start_mask[:cur_bsz, :max_entity_cnt].contiguous(),
                'start_mask_weight': start_mask_weight[:cur_bsz, :max_entity_cnt].contiguous(),
                'bfs_mask': bfs_mask[:cur_bsz, :, :max_entity_cnt],
                'answer_mapping': answer_mapping,
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--entity_path", default=r"E:\DATA\HotpotQA\entities\train_entities.json", type=str)
    parser.add_argument("--para_path",  default=r"E:\DATA\HotpotQA\train_selected_paras.json", type=str)
    parser.add_argument("--example_output",  default=r"E:\DATA\HotpotQA\output\example.pkl.gz", type=str)
    parser.add_argument("--feature_output",  default=r"E:\DATA\HotpotQA\output\feature.pkl.gz", type=str)

    # Other parameters
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size for predictions.")
    parser.add_argument("--full_data", type=str,  default=r"E:\DATA\HotpotQA\hotpot_train_v1.1.json")

    args = parser.parse_args()

    examples = read_hotpot_examples(para_file=args.para_path, full_file=args.full_data, entity_file=args.entity_path)
    with gzip.open(args.example_output, 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    with gzip.open(args.feature_output, 'wb') as fout:
        pickle.dump(features, fout)