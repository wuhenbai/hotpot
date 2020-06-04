from pytorch_pretrained_bert.modeling import *
from transformers import AlbertModel, AlbertTokenizer, BertPreTrainedModel, RobertaModel, BertModel


from model.layers import *


class GraphFusionNet(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(GraphFusionNet, self).__init__()
        self.config = config
        self.n_layers = config.n_layers
        self.max_query_length = 50

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        h_dim = config.hidden_dim
        q_dim = config.hidden_dim if config.q_update else config.input_dim

        self.basicblocks = nn.ModuleList()
        self.query_update_layers = nn.ModuleList()
        self.query_update_linears = nn.ModuleList()

        for layer in range(self.n_layers):
            self.basicblocks.append(BasicBlock(h_dim, q_dim, layer, config))
            if config.q_update:
                self.query_update_layers.append(BiAttention(h_dim, h_dim, h_dim, config.bi_attn_drop))
                self.query_update_linears.append(nn.Linear(h_dim * 4, h_dim))

        q_dim = h_dim if config.q_update else config.input_dim
        if config.prediction_trans:
            self.predict_layer = TransformerPredictionLayer(self.config, q_dim)
        else:
            self.predict_layer = PredictionLayer(self.config, q_dim)

    def forward(self, context_encoding, batch, return_yp, debug=False):
        query_mapping = batch['query_mapping']
        entity_mask = batch['entity_mask']
        context_mask = batch['context_mask']

        # query_vec = None
        # attn_output, trunc_query_state = self.bi_attention(context_encoding, context_encoding, context_mask)
        # input_state = self.bi_attn_linear(attn_output)

        #TODO: remove query_vec
        query_vec = None
        input_state = context_encoding
        # context_mapping = context_mask - query_mapping
        # #
        # # # extract query encoding
        # trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        #
        # trunc_context_state = context_encoding * context_mapping.unsqueeze(2)
        # trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # # # bert encoding query vec
        # query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        # attn_output, trunc_query_state = self.bi_attention(context_encoding, trunc_query_state, trunc_query_mapping)
        # input_state = self.bi_attn_linear(attn_output)
        #
        # if self.config.q_update:
        #     query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        softmasks = []
        entity_state = None
        # for l in range(self.n_layers):
        #     input_state, entity_state, softmask = self.basicblocks[l](input_state, query_vec, batch)
        #     softmasks.append(softmask)
        #     if self.config.q_update:
        #         query_attn_output, _ = self.query_update_layers[l](trunc_query_state, entity_state, entity_mask)
        #         trunc_query_state = self.query_update_linears[l](query_attn_output)
        #         query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        predictions = self.predict_layer(batch, input_state, query_vec, entity_state, query_mapping, return_yp)
        start, end, sp, Type, ent, yp1, yp2 = predictions

        if return_yp:
            return start, end, sp, Type, softmasks, ent, yp1, yp2
        else:
            return start, end, sp, Type, softmasks, ent



class BertForQuestionAnswering(PreTrainedBertModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.apply(self.init_bert_weights)



    def forward(self, input_ids, token_type_ids=None, attention_mask=None):

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)

        return sequence_output



class DFGN(PreTrainedBertModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, graph_config):
        super(DFGN, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.DFGN = GraphFusionNet(graph_config)
        self.apply(self.init_bert_weights)
        self.type_lambda = graph_config.type_lambda
        self.sp_lambda = graph_config.sp_lambda
        self.bfs_lambda = graph_config.bfs_lambda
        self.bfs_clf = graph_config.bfs_clf
        self.n_layers = graph_config.n_layers


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
                q_type=None, args=None, batch=None, return_yp=None, is_train=False):

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        return self.DFGN(sequence_output, batch, return_yp, debug=is_train)
        # if start_positions is not None and end_positions is not None:
        #     start, end, sp, Type, softmasks, ent, yp1, yp2 = self.DFGN(sequence_output, batch, True)
        # else:
        #     start, end, sp, Type, softmasks, ent = self.DFGN(sequence_output, batch, False)


        # criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
        # binary_criterion = nn.BCEWithLogitsLoss(size_average=True)
        # def compute_loss(batch, start, end, sp, Type, masks):
        #     loss1 = criterion(start, batch['y1']) + criterion(end, batch['y2'])
        #     loss2 = self.type_lambda * criterion(Type, batch['q_type'])
        #     loss3 = self.sp_lambda * criterion(sp.view(-1, 2), batch['is_support'].long().view(-1))
        #     loss = loss1 + loss2 + loss3
        #
        #     loss4 = 0
        #     if self.bfs_clf:
        #         for l in range(self.n_layers):
        #             pred_mask = masks[l].view(-1)
        #             gold_mask = batch['bfs_mask'][:, l, :].contiguous().view(-1)
        #             loss4 += binary_criterion(pred_mask, gold_mask)
        #         loss += self.bfs_lambda * loss4
        #
        #     return loss, loss1, loss2, loss3, loss4
        #
        # if start_positions is not None and end_positions is not None:
        #     loss_list = compute_loss(batch, start, end, sp, Type, softmasks)
        #
        #     return loss_list
        # else:
        #     Type = Type.argmax(dim=1)
        #     return start, end, Type, sp


class DFGN_Roberta(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, graph_config):
        super(DFGN_Roberta, self).__init__(config)
        self.roberta = RobertaModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.DFGN = GraphFusionNet(graph_config)
        # self.apply(self.init_bert_weights)
        self.init_weights()
        self.type_lambda = graph_config.type_lambda
        self.sp_lambda = graph_config.sp_lambda
        self.bfs_lambda = graph_config.bfs_lambda
        self.bfs_clf = graph_config.bfs_clf
        self.n_layers = graph_config.n_layers


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
        inputs_embeds=None, start_positions=None, end_positions=None, args=None, batch=None, return_yp=None, is_train=False):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        return self.DFGN(sequence_output, batch, return_yp, debug=is_train)


class DFGN_Albert(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, graph_config):
        super(DFGN_Albert, self).__init__(config)
        self.albert = AlbertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.DFGN = GraphFusionNet(graph_config)
        # self.apply(self.init_bert_weights)
        self.init_weights()
        self.type_lambda = graph_config.type_lambda
        self.sp_lambda = graph_config.sp_lambda
        self.bfs_lambda = graph_config.bfs_lambda
        self.bfs_clf = graph_config.bfs_clf
        self.n_layers = graph_config.n_layers


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
        inputs_embeds=None, start_positions=None, end_positions=None, args=None, batch=None, return_yp=None, is_train=False):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        return self.DFGN(sequence_output, batch, return_yp, debug=is_train)

class DFGN_Bert(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, graph_config):
        super(DFGN_Bert, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.DFGN = GraphFusionNet(graph_config)
        # self.apply(self.init_bert_weights)
        self.init_weights()
        self.type_lambda = graph_config.type_lambda
        self.sp_lambda = graph_config.sp_lambda
        self.bfs_lambda = graph_config.bfs_lambda
        self.bfs_clf = graph_config.bfs_clf
        self.n_layers = graph_config.n_layers


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
        inputs_embeds=None, start_positions=None, end_positions=None, args=None, batch=None, return_yp=None, is_train=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        return self.DFGN(sequence_output, batch, return_yp, debug=is_train)