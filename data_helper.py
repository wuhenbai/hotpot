from os.path import join
import gzip
import pickle
import json
from tqdm import tqdm
import torch
from numpy.random import shuffle
import numpy as np

IGNORE_INDEX = -100

class DataIteratorPack(object):
    def __init__(self, features, example_dict, bsz, device, sent_limit, sequential=False,):
        self.bsz = bsz
        self.device = device
        self.features = features
        self.example_dict = example_dict
        self.sequential = sequential
        self.sent_limit = sent_limit
        self.example_ptr = 0
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

        # Label tensor
        y1 = torch.LongTensor(self.bsz).cuda(self.device)
        y2 = torch.LongTensor(self.bsz).cuda(self.device)
        q_type = torch.LongTensor(self.bsz).cuda(self.device)
        # is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)



        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)
            cur_batch = self.features[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)

            ids = []

            # is_support.fill_(IGNORE_INDEX)
            # is_support.fill_(0)  # BCE

            for i in range(len(cur_batch)):
                case = cur_batch[i]
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))

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

                # for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
                #     is_sp_flag = j in case.sup_fact_ids
                #     start, end = sent_span
                #     if start < end:
                #         is_support[i, j] = int(is_sp_flag)


                ids.append(case.qas_id)



            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                # 'context_lens': input_lengths.to(self.device),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'q_type': q_type[:cur_bsz],
                'ids': ids,
                # 'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
            }

class DataHelper:
    def __init__(self, gz=True, config=None):
        self.DataIterator = DataIteratorPack
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = 'data'
        self.subset_file = join(self.data_dir, 'subset.json')

        self.__train_features__ = None
        self.__dev_features__ = None

        self.__train_examples__ = None
        self.__dev_examples__ = None

        self.__train_graphs__ = None
        self.__dev_graphs__ = None

        self.__train_example_dict__ = None
        self.__dev_example_dict__ = None

        self.config = config

    @property
    def sent_limit(self):
        return 25

    @property
    def n_type(self):
        return 2

    def get_feature_file(self, tag):
        return join(self.data_dir, tag + '_feature' + self.suffix)

    def get_example_file(self, tag):
        return join(self.data_dir, tag + '_example' + self.suffix)

    @property
    def train_feature_file(self):
        return self.get_feature_file('train')

    @property
    def dev_feature_file(self):
        return self.get_feature_file('dev')

    @property
    def train_example_file(self):
        return self.get_example_file('train')

    @property
    def dev_example_file(self):
        return self.get_example_file('dev')

    @staticmethod
    def compress_pickle(pickle_file_name):
        def abbr(obj):
            obj_str = str(obj)
            if len(obj_str) > 100:
                return obj_str[:20] + ' ... ' + obj_str[-20:]
            else:
                return obj_str

        def get_obj_dict(pickle_obj):
            if isinstance(pickle_obj, list):
                obj = pickle_obj[0]
            elif isinstance(pickle_obj, dict):
                obj = list(pickle_obj.values())[0]
            else:
                obj = pickle_obj
            if isinstance(obj, dict):
                return obj
            else:
                return obj.__dict__

        pickle_obj = pickle.load(open(pickle_file_name, 'rb'))

        for k, v in get_obj_dict(pickle_obj).items():
            print(k, abbr(v))
        with gzip.open(pickle_file_name + '.gz', 'wb') as fout:
            pickle.dump(pickle_obj, fout)
        pickle_obj = pickle.load(gzip.open(pickle_file_name + '.gz', 'rb'))
        for k, v in get_obj_dict(pickle_obj).items():
            print(k, abbr(v))

    def __load__(self, file):
        if file.endswith('json'):
            return json.load(open(file, 'r'))
        with self.get_pickle_file(file) as fin:
            print('loading', file)
            return pickle.load(fin)

    def get_pickle_file(self, file_name):
        if self.gz:
            return gzip.open(file_name, 'rb')
        else:
            return open(file_name, 'rb')

    def __get_or_load__(self, name, file):
        if getattr(self, name) is None:
            with self.get_pickle_file(file) as fin:
                print('loading', file)
                setattr(self, name, pickle.load(fin))

        return getattr(self, name)

    # Features
    @property
    def train_features(self):
        return self.__get_or_load__('__train_features__', self.train_feature_file)

    @property
    def dev_features(self):
        return self.__get_or_load__('__dev_features__', self.dev_feature_file)

    # Examples
    @property
    def train_examples(self):
        return self.__get_or_load__('__train_examples__', self.train_example_file)

    @property
    def dev_examples(self):
        return self.__get_or_load__('__dev_examples__', self.dev_example_file)


    # Example dict
    @property
    def train_example_dict(self):
        if self.__train_example_dict__ is None:
            self.__train_example_dict__ = {e.qas_id: e for e in self.train_examples}
        return self.__train_example_dict__

    @property
    def dev_example_dict(self):
        if self.__dev_example_dict__ is None:
            self.__dev_example_dict__ = {e.qas_id: e for e in self.dev_examples}
        return self.__dev_example_dict__

    # Feature dict
    @property
    def train_feature_dict(self):
        return {e.qas_id: e for e in self.train_features}

    @property
    def dev_feature_dict(self):
        return {e.qas_id: e for e in self.dev_features}

    # Load
    def load_dev(self):
        return self.dev_features, self.dev_example_dict, self.dev_graphs

    def load_train(self):
        return self.train_features, self.train_example_dict, self.train_graphs

    def load_train_subset(self, subset):
        assert subset is not None
        keylist = set(json.load(open(self.subset_file, 'r'))[subset])
        train_examples = [e for e in tqdm(self.train_examples, desc='sub_ex') if e.qas_id in keylist]
        train_example_dict = {e.qas_id: e for e in train_examples}
        train_features = [f for f in tqdm(self.train_features, desc='sub_fe') if f.qas_id in keylist]
        train_graphs = {k: self.train_graphs[k] for k in tqdm(keylist, desc='sub_graph')}
        print('subset: {}, total: {}'.format(subset, len(train_graphs)))
        return train_features, train_example_dict, train_graphs

    @property
    def dev_loader(self):
        return self.DataIterator(*self.load_dev(),
                                 bsz=self.config.batch_size,
                                 device='cuda:{}'.format(self.config.model_gpu),
                                 sent_limit=self.sent_limit,
                                 entity_limit=self.entity_limit,
                                 sequential=True,
                                 n_layers=self.config.n_layers)

    @property
    def train_loader(self):
        return self.DataIterator(*self.load_train(),
                                 bsz=self.config.batch_size,
                                 device='cuda:{}'.format(self.config.model_gpu),
                                 sent_limit=self.sent_limit,
                                 entity_limit=self.entity_limit,
                                 sequential=False,
                                 n_layers=self.config.n_layers)

    @property
    def train_sub_loader(self):
        return self.DataIterator(*self.load_train_subset('qat'),
                                 bsz=self.config.batch_size,
                                 device='cuda:{}'.format(self.config.model_gpu),
                                 sent_limit=self.sent_limit,
                                 entity_limit=self.entity_limit,
                                 sequential=False,
                                 n_layers=self.config.n_layers)
