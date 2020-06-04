from utils import *
from tqdm import tqdm
import numpy as np
import gzip
import pickle
from qa_bert_based import InputFeatures, Example
from argparse import ArgumentParser
import gc

def iter_data(features, example_dict, query_entity_path):

    def foo(features, examples, query_entities):
        entity_cnt = []
        entity_graphs = {}
        for case in tqdm(features):
            # case.__dict__['answer'] = examples[case.qas_id].orig_answer_text
            case.__dict__['query_entities'] = [ent[0] for ent in query_entities[case.qas_id]]
            graph = create_entity_graph(case, 80, 512, 'sent', False, False, relational=False)
            entity_cnt.append(graph['entity_length'])

            # Simplify Graph dicts
            targets = ['entity_length', 'start_entities', 'entity_mapping', 'adj']
            simp_graph = dict([(t, graph[t]) for t in targets])

            entity_graphs[case.qas_id] = simp_graph
        entity_cnt = np.array(entity_cnt)
        for thr in range(40, 100, 10):
            print(len(np.where(entity_cnt > thr)[0]) / len(entity_cnt), f'> {thr}')
        # del features
        # del examples
        # del query_entities
        # gc.collect()
        return entity_graphs
        # pickle.dump(entity_graphs, gzip.open(args.graph_path, 'wb'))

    # with gzip.open(args.example_path, 'rb') as fin:
    #     examples = pickle.load(fin)
    #     example_dict = {e.qas_id: e for e in examples}
    #
    # with gzip.open(args.feature_path, 'rb') as fin:
    #     features = pickle.load(fin)
    #
    with open(query_entity_path, 'r') as fin:
        query_entities = json.load(fin)

    # del examples

    entity_graphs = foo(features, example_dict, query_entities)

    # del features
    # del example_dict
    # del query_entities
    gc.collect()
    # with open(args.graph_path, 'w', encoding='utf-8') as f:
    #     f.write(entity_graphs)
    # json.dump(entity_graphs, open(args.graph_path, 'w', encoding='utf-8'), cls=JsonEncoder)
    # pickle.dump(entity_graphs, gzip.open(args.graph_path, 'wb'))
    return entity_graphs

class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # elif isinstance(obj, datetime):
        #     return obj.__str__()
        # else:
        #     return super(MyEncoder, self).default(obj)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--example_path', default=r"E:\DATA\HotpotQA\output\example.pkl.gz", type=str)
    parser.add_argument('--feature_path', default=r"E:\DATA\HotpotQA\output\feature.pkl.gz", type=str)
    parser.add_argument('--query_entity_path', default=r"E:\DATA\HotpotQA\entities\train_query_entities.json", type=str)
    parser.add_argument('--graph_path', default=r"E:\DATA\HotpotQA\entities\train_graph.json", type=str)
    args = parser.parse_args()
    iter_data()
