import argparse
import os
import json
from os.path import join


def process_arguments(args):
    # args.checkpoint_path = join(args.checkpoint_path, args.name)
    # args.prediction_path = join(args.prediction_path, args.name)
    args.n_layers = int(args.gnn.split(':')[1].split(',')[0])
    args.n_heads = int(args.gnn.split(':')[1].split(',')[1])
    args.max_query_len = 50
    args.max_doc_len = 512


def save_settings(args):
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.prediction_path, exist_ok=True)
    json.dump(args.__dict__, open(join(args.checkpoint_path, "run_settings.json"), 'w'))


def set_config():
    parser = argparse.ArgumentParser()
    data_path = 'output'
    # Required parameters
    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    # parser.add_argument("--output_dir", default=r"E:\DATA/HotpotQA", type=str,
    #                     help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--output_dir", default=r"E:\DATA/HotpotQA\output", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--train_file", default=r"E:\DATA/HotpotQA\hotpot_train_v1.1.json", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--train_select_file", default=r"E:\DATA/HotpotQA\train_selected_paras.json", type=str,
                        help="SQuAD json for training. E.g., train_selected_paras.json")
    parser.add_argument("--train_examples_file", default=r"E:\DATA\DFGN-pytorch\DFGN\data\train_example.pkl.gz", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")
    parser.add_argument("--train_features_file", default=r"E:\DATA\DFGN-pytorch\DFGN\data\train_feature.pkl.gz", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")
    parser.add_argument("--train_graph_file", default=r"E:\DATA\DFGN-pytorch\DFGN\data\train_graph.pkl.gz", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")
    parser.add_argument("--train_entity_file", default=r"E:\DATA\HotpotQA\entities\train_entities.json", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")
    parser.add_argument("--train_query_entity_file", default=r"E:\DATA\HotpotQA\entities\train_query_entities.json", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")

    parser.add_argument("--predict_file", default=r"E:\DATA/HotpotQA/hotpot_dev_distractor_v1.json", type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--predict_select_file", default=r"E:\DATA/HotpotQA/dev_selected_paras.json", type=str,
                        help="SQuAD json for training. E.g., dev_selected_paras.json.json")
    parser.add_argument("--predict_examples_file", default=r"E:\DATA\DFGN-pytorch\DFGN\data\dev_example.pkl.gz", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")
    parser.add_argument("--predict_features_file", default=r"E:\DATA\DFGN-pytorch\DFGN\data\dev_feature.pkl.gz", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")
    parser.add_argument("--predict_graph_file", default=r"E:\DATA\DFGN-pytorch\DFGN\data\dev_graph.pkl.gz", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")
    parser.add_argument("--predict_entity_file", default=r"E:\DATA\HotpotQA\entities\dev_entities.json", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")
    parser.add_argument("--predict_query_entity_file", default=r"E:\DATA\HotpotQA\entities\dev_query_entities.json", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.pkl.gz")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    # parser.add_argument("--max_seq_length", default=384, type=int,
    #                     help="The maximum total input sequence length after WordPiece tokenization. Sequences "
    #                          "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=True, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=1, type=int,
                        help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
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
    parser.add_argument('--theta',
                        type=int, default=10,
                        help="theta in loss func")
    parser.add_argument('--alpha',
                        type=float, default=2,
                        help="alpha in loss func")
    parser.add_argument('--beta',
                        type=int, default=4,
                        help="beta in loss func")
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--version_2_with_negative', default=True,
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument("--verbose_logging", default=False,
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument("--sp_threshold", type=float, default=0.5)



    #DFGN
    parser.add_argument('--q_update', default=True, help='Whether update query')
    parser.add_argument('--basicblock_trans', default=True, help='transformer version basicblock')
    parser.add_argument("--prediction_trans", default=False, help='transformer version prediction layer')
    parser.add_argument("--trans_drop", type=float, default=0.5)
    parser.add_argument("--trans_heads", type=int, default=3)
    parser.add_argument("--grad_accumulate_step", default=1, type=int)

    parser.add_argument("--input_dim", type=int, default=768, help="bert-base=768, bert-large=1024")

    # bi attn
    parser.add_argument("--bi_attn_drop", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=300)

    # graph net
    parser.add_argument('--tok2ent', default='mean_max', type=str, help='{mean, mean_max}')
    parser.add_argument('--gnn', default='gat:2,2', type=str, help='gat:n_layer, n_head')
    parser.add_argument("--gnn_drop", type=float, default=0.5)
    parser.add_argument("--gat_attn_drop", type=float, default=0.5)
    parser.add_argument('--q_attn', default=True, help='whether use query attention in GAT')
    parser.add_argument("--lstm_drop", type=float, default=0.3)
    parser.add_argument("--n_type", type=int, default=2)

    # loss
    parser.add_argument("--type_lambda", type=float, default=1)
    parser.add_argument("--sp_lambda", type=float, default=5)
    parser.add_argument('--bfs_clf', default=True, help='Add BCELoss on bfs mask')
    parser.add_argument('--bfs_lambda', type=float, default=1)


    args = parser.parse_args()
    #
    process_arguments(args)
    # save_settings(args)

    return args
