# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

r"""
PCR
====
References:
    Chen, Qibin, et al. `"Towards Knowledge-Based Recommender Dialog System."`_ in EMNLP 2019.

.. _`"Towards Knowledge-Based Recommender Dialog System."`:
   https://www.aclweb.org/anthology/D19-1189/

"""

import json
import inspect
import math
import os.path
import random
import pickle
from typing import List
from time import perf_counter

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import RGCNConv, HypergraphConv
from torch_geometric.utils import softmax

from crslab.config import DATA_PATH, DATASET_PATH
from crslab.model.base import BaseModel
from crslab.model.crs.hycorec.attention import MHItemAttention
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionBatch, SelfAttentionSeq
from crslab.model.utils.modules.weighted_hypergraph_conv import WeightedHypergraphConv
from crslab.model.utils.modules.transformer import TransformerEncoder
from crslab.model.crs.hycorec.decoder import TransformerDecoderKG
from crslab.model.utils.modules.viewlearner import ViewLearner

from crslab.model.utils.debug import _numeric_debug_tensor, check

class HyCoRecModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        token_emb_dim: A integer indicating the dimension of token embedding layer.
        pretrain_embedding: A string indicating the path of pretrained embedding.
        n_entity: A integer indicating the number of entities.
        n_relation: A integer indicating the number of relation in KG.
        num_bases: A integer indicating the number of bases.
        kg_emb_dim: A integer indicating the dimension of kg embedding.
        user_emb_dim: A integer indicating the dimension of user embedding.
        n_heads: A integer indicating the number of heads.
        n_layers: A integer indicating the number of layer.
        ffn_size: A integer indicating the size of ffn hidden.
        dropout: A float indicating the dropout rate.
        attention_dropout: A integer indicating the dropout rate of attention layer.
        relu_dropout: A integer indicating the dropout rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        longest_label: A integer indicating the longest length for response generation.
        user_proj_dim: A integer indicating dim to project for user embedding.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.view_optim_opt = opt['view']
        self.device = device
        self.gpu = opt.get("gpu", -1)
        self.dataset = opt.get("dataset", None)
        self.llm = opt.get("llm", "chatgpt-4o")
        assert self.dataset in ['HReDial', 'HTGReDial', 'DuRecDial', 'OpenDialKG', 'ReDial', 'TGReDial']
        # vocab
        self.item_ids = side_data['item_entity_ids']
        self.pad_token_idx = vocab['tok2ind']['__pad__']
        self.start_token_idx = vocab['tok2ind']['__start__']
        self.end_token_idx = vocab['tok2ind']['__end__']
        self.vocab_size = vocab['vocab_size']
        self.token_emb_dim = opt.get('token_emb_dim', 300)
        self.pretrain_embedding = side_data.get('embedding', None)
        self.token2id = json.load(open(os.path.join(DATASET_PATH, self.dataset.lower(), opt["tokenize"], "token2id.json"), "r", encoding="utf-8"))
        self.entity2id = json.load(open(os.path.join(DATASET_PATH, self.dataset.lower(), opt["tokenize"], "entity2id.json"), "r", encoding="utf-8"))
        # kg
        self.n_entity = vocab['n_entity']
        self.entity_kg = side_data['entity_kg']
        self.n_relation = self.entity_kg['n_relation']
        self.edge_idx, self.edge_type = edge_to_pyg_format(self.entity_kg['edge'], 'RGCN')
        self.edge_idx = self.edge_idx.to(device)
        self.edge_type = self.edge_type.to(device)
        self.num_bases = opt.get('num_bases', 8)
        self.kg_emb_dim = opt.get('kg_emb_dim', 300)
        self.user_emb_dim = self.kg_emb_dim
        self.hgcn_layers = opt.get('hgcn_layers', 1)
        self.rgcn_layers = opt.get('rgcn_layers', 1)
        self.cold_start = opt.get('cold_start', False)
        # transformer
        self.n_heads = opt.get('n_heads', 2)
        self.n_layers = opt.get('n_layers', 2)
        self.ffn_size = opt.get('ffn_size', 300)
        self.dropout = opt.get('dropout', 0.1)
        self.attention_dropout = opt.get('attention_dropout', 0.0)
        self.relu_dropout = opt.get('relu_dropout', 0.1)
        self.embeddings_scale = opt.get('embedding_scale', True)
        self.learn_positional_embeddings = opt.get('learn_positional_embeddings', False)
        self.reduction = opt.get('reduction', False)
        self.n_positions = opt.get('n_positions', 1024)
        self.longest_label = opt.get('longest_label', 30)
        self.user_proj_dim = opt.get('user_proj_dim', 512)
        # pooling
        self.pooling = opt.get('pooling', None)
        assert self.pooling == 'Attn' or self.pooling == 'Mean'
        # MHA
        self.mha_n_heads = opt.get('mha_n_heads', 4)
        self.extension_strategy = opt.get('extension_strategy', None)
        self.pretrain = opt.get('pretrain', False)
        self.pretrain_data = None
        self.pretrain_epoch = opt.get('pretrain_epoch', 9999)
        # view learner
        self.view_last = self.view_optim_opt.get('view_last', False)
        self.keep_ratio = self.view_optim_opt.get('ratio', 0.6)  # 用于 early stop 的关键指标
        self.cf_keep_ratio = self.view_optim_opt.get('cf_keep_ratio', 0.2)  # 反事实选边的保留比例
        self.view_hidden_dim = self.view_optim_opt.get('view_hidden_dim', 64)  # ViewLearner 隐藏层维度
        self.view_hyperedge_aggregation = self.view_optim_opt.get('hyperedge_aggregation', 'mean')
        self.same_view = self.view_optim_opt.get('same_view', False)  # 是否让事实视图和反事实视图共享权重学习器（调试用）

        # layers confige
        self.graph_norm = opt.get('graph_norm', 'layernorm').lower()
        self.graph_activation = opt.get('graph_activation', 'relu').lower()
        self.graph_dropout = opt.get('graph_dropout', self.dropout)

        super(HyCoRecModel, self).__init__(opt, device)
        return

    # 构建模型
    def build_model(self, *args, **kwargs):
        if self.pretrain:
            pretrain_file = os.path.join('pretrain', self.dataset, str(self.pretrain_epoch) + '-epoch.pth')
            self.pretrain_data = torch.load(pretrain_file, map_location=torch.device('cuda:' + str(self.gpu[0])))
            logger.info(f"[Load Pretrain Weights from {pretrain_file}]")
        # self._build_hredial_copy_mask()
        self._build_adjacent_matrix()
        # self._build_hllm_data()
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    # 构建 mask
    def _build_hredial_copy_mask(self):
        """只允许 decoder 从 @ 开头的 token 中拷贝，屏蔽掉普通词汇的拷贝 logits。这保证了拷贝机制只复制实体名称，不会照搬对话中的普通词。"""
        token_filename = os.path.join(DATASET_PATH, "hredial", "nltk", "token2id.json")
        token_file = open(token_filename, 'r', encoding="utf-8")
        token2id = json.load(token_file)
        id2token = {token2id[token]: token for token in token2id}
        self.hredial_copy_mask = list()
        for i in range(len(id2token)):
            token = id2token[i]
            if token[0] == '@':
                self.hredial_copy_mask.append(True)
            else:
                self.hredial_copy_mask.append(False)
        self.hredial_copy_mask = torch.as_tensor(self.hredial_copy_mask).to(self.device)
        return
    
    def _build_hllm_data(self):
        self.hllm_data_table = {
            "train": pickle.load(open(os.path.join(DATA_PATH, "hllm", self.dataset.lower(), self.llm, "hllm_train_data.pkl"), "rb")),
            "valid": pickle.load(open(os.path.join(DATA_PATH, "hllm", self.dataset.lower(), self.llm, "hllm_valid_data.pkl"), "rb")),
            "test": pickle.load(open(os.path.join(DATA_PATH, "hllm", self.dataset.lower(),  self.llm, "hllm_test_data.pkl"), "rb")),
        }
        return

    # 构建关联矩阵
    def _build_adjacent_matrix(self):
        entity2id = self.entity2id
        token2id = self.token2id
        item_edger = pickle.load(open(os.path.join(DATA_PATH, "edger", self.dataset.lower(), "item_edger.pkl"), "rb"))
        entity_edger = pickle.load(open(os.path.join(DATA_PATH, "edger", self.dataset.lower(), "entity_edger.pkl"), "rb"))
        word_edger = pickle.load(open(os.path.join(DATA_PATH, "edger", self.dataset.lower(), "word_edger.pkl"), "rb"))

        item_adj = {}
        for item_a in item_edger:
            item_list = item_edger[item_a]
            if item_a not in entity2id:
                continue
            item_a = entity2id[item_a]
            convert_list = []
            for item in item_list:
                if item not in entity2id:
                    continue
                convert_list.append(entity2id[item])
            item_adj[item_a] = convert_list
        self.item_adj = item_adj

        entity_adj = {}
        for entity_a in entity_edger:
            entity_list = entity_edger[entity_a]
            if entity_a not in entity2id:
                continue
            entity_a = entity2id[entity_a]
            convert_list = []
            for entity in entity_list:
                if entity not in entity2id:
                    continue
                convert_list.append(entity2id[entity])
            entity_adj[entity_a] = convert_list
        self.entity_adj = entity_adj

        word_adj = {}
        for word_a in word_edger:
            word_list = word_edger[word_a]
            if word_a not in token2id:
                continue
            word_a = token2id[word_a]
            convert_list = []
            for word in word_list:
                if word not in token2id:
                    continue
                convert_list.append(token2id[word])
            word_adj[word_a] = convert_list
        self.word_adj = word_adj

        logger.info(f"[Adjacent Matrix built.]")
        return

    # 构建编码层
    def _build_embedding(self):
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)

        self.entity_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        nn.init.normal_(self.entity_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.entity_embedding.weight[0], 0)
        self.word_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        nn.init.normal_(self.word_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.word_embedding.weight[0], 0)
        logger.debug('[Build embedding]')
        return

    def _build_rgcn_stack(self):
        return nn.ModuleList([
            RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
            for _ in range(self.rgcn_layers)
        ])

    def _build_hgcn_stack(self):
        return nn.ModuleList([
            WeightedHypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
            for _ in range(self.hgcn_layers)
        ])

    def _build_view_learner_stack(self):
        return nn.ModuleList([
            ViewLearner(
                self.kg_emb_dim,
                self.view_hidden_dim,
                self.device,
                hyperedge_aggregation=self.view_hyperedge_aggregation
            )
            for _ in range(1 if self.view_last else self.hgcn_layers)
        ])

    def _make_graph_norm(self):
        if self.graph_norm in {'layernorm', 'layer_norm', 'ln'}:
            return nn.LayerNorm(self.kg_emb_dim)
        if self.graph_norm in {'batchnorm', 'batch_norm', 'bn'}:
            return nn.BatchNorm1d(self.kg_emb_dim)
        if self.graph_norm in {'none', 'identity', ''}:
            return nn.Identity()
        raise ValueError(f'Unsupported graph_norm: {self.graph_norm}')

    def _build_graph_transitions(self, num_layers):
        transition_layers = max(num_layers - 1, 0)
        norms = nn.ModuleList([self._make_graph_norm() for _ in range(transition_layers)])
        dropouts = nn.ModuleList([nn.Dropout(self.graph_dropout) for _ in range(transition_layers)])
        return norms, dropouts

    def _graph_activate(self, x):
        if self.graph_activation == 'relu':
            return F.relu(x)
        if self.graph_activation == 'gelu':
            return F.gelu(x)
        if self.graph_activation in {'none', 'identity', ''}:
            return x
        raise ValueError(f'Unsupported graph_activation: {self.graph_activation}')

    def _apply_graph_transition(self, x, norms, dropouts, layer_idx):
        if layer_idx >= len(norms):
            return x
        x = norms[layer_idx](x)
        x = self._graph_activate(x)
        x = dropouts[layer_idx](x)
        return x

    def _run_rgcn_stack(self, node_embedding, encoders, norms, dropouts):
        out = node_embedding
        for layer_idx, encoder in enumerate(encoders):
            out = encoder(out, self.edge_idx, self.edge_type)
            out = self._apply_graph_transition(out, norms, dropouts, layer_idx)
        return out

    # 构建超图编码层
    def _build_kg_layer(self):
        # graph encoder
        self.item_encoder = self._build_rgcn_stack()
        self.entity_encoder = self._build_rgcn_stack()
        self.word_encoder = self._build_rgcn_stack()
        self.item_encoder_norms, self.item_encoder_dropouts = self._build_graph_transitions(self.rgcn_layers)
        self.entity_encoder_norms, self.entity_encoder_dropouts = self._build_graph_transitions(self.rgcn_layers)
        self.word_encoder_norms, self.word_encoder_dropouts = self._build_graph_transitions(self.rgcn_layers)
        if self.pretrain:
            self.item_encoder[0].load_state_dict(self.pretrain_data['encoder'])
        # hypergraph convolution
        self.hyper_conv_item = self._build_hgcn_stack()
        self.hyper_conv_entity = self._build_hgcn_stack()
        self.hyper_conv_word = self._build_hgcn_stack()
        self.hyper_conv_item_norms, self.hyper_conv_item_dropouts = self._build_graph_transitions(self.hgcn_layers)
        self.hyper_conv_entity_norms, self.hyper_conv_entity_dropouts = self._build_graph_transitions(self.hgcn_layers)
        self.hyper_conv_word_norms, self.hyper_conv_word_dropouts = self._build_graph_transitions(self.hgcn_layers)
        if self.same_view:
            shared_view_learner = self._build_view_learner_stack()
            self.view_learner_item = shared_view_learner
            self.view_learner_entity = shared_view_learner
            self.view_learner_word = shared_view_learner
        else:
            self.view_learner_item = self._build_view_learner_stack()
            self.view_learner_entity = self._build_view_learner_stack()
            self.view_learner_word = self._build_view_learner_stack()
        # attention type
        self.item_attn = MHItemAttention(self.kg_emb_dim, self.mha_n_heads)
        # pooling
        if self.pooling == 'Attn':
            self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
            self.kg_attn_his = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        logger.debug('[Build kg layer]')
        return

    # 构建推荐模块
    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        logger.debug('[Build recommendation layer]')
        return

    # 构建对话模块
    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.entity_to_token = nn.Linear(self.kg_emb_dim, self.token_emb_dim, bias=True)
        self.related_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.context_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.decoder = TransformerDecoderKG(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.embeddings_scale,
            self.learn_positional_embeddings,
            self.pad_token_idx,
            self.n_positions
        )
        self.user_proj_1 = nn.Linear(self.user_emb_dim, self.user_proj_dim)
        self.user_proj_2 = nn.Linear(self.user_proj_dim, self.vocab_size)
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

        self.copy_proj_1 = nn.Linear(2 * self.token_emb_dim, self.token_emb_dim)
        self.copy_proj_2 = nn.Linear(self.token_emb_dim, self.vocab_size)
        logger.debug('[Build conversation layer]')
        return

    # 获取超图
    def _get_hypergraph(self, related, adj):
        related_items_set = set()
        for related_items in related:
            related_items_set.add(related_items)
        session_related_items = list(related_items_set)

        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for item in session_related_items:
            hypergraph_nodes.append(item)
            hypergraph_edges.append(hyper_edge_counter)
            neighbors = list(adj.get(item, []))
            hypergraph_nodes += neighbors
            hypergraph_edges += [hyper_edge_counter] * len(neighbors)
            hyper_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        return list(set(hypergraph_nodes)), hyper_edge_index

    # 获取聚合
    def _agg_embding(self, hypergraph_items, embedding, tot_sub, adj):
        knowledge_embedding_list = []
        for item in hypergraph_items:
            sub_graph = [item] + list(adj.get(item, []))
            sub_graph = [tot_sub[item] for item in sub_graph]
            sub_graph_embedding = embedding[sub_graph]
            sub_graph_embedding = torch.mean(sub_graph_embedding, dim=0)
            knowledge_embedding_list.append(sub_graph_embedding)
        res_embedding = torch.zeros(1, self.kg_emb_dim).to(self.device)
        if len(knowledge_embedding_list) > 0:
            res_embedding = torch.stack(knowledge_embedding_list, dim=0)
        return res_embedding

    @staticmethod
    def flatten(inputs):
        outputs = set()
        for li in inputs:
            for i in li:
                outputs.add(i)
        return list(outputs)

    # 注意力融合特征向量
    def _attention_and_gating(self, session_embedding, knowledge_embedding, conceptnet_embedding, context_embedding):
        related_embedding = torch.cat((session_embedding, knowledge_embedding, conceptnet_embedding), dim=0)
        if context_embedding is None:
            if self.pooling == 'Attn':
                user_repr = self.kg_attn_his(related_embedding)
            else:
                assert self.pooling == 'Mean'
                user_repr = torch.mean(related_embedding, dim=0)
        elif self.pooling == 'Attn':
            # 保留与context相关度高的related embd
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            # 自注意力aggragation
            user_repr = self.kg_attn_his(attentive_related_embedding)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            # 再融合一次context
            user_repr = self.kg_attn(user_repr)
        else:
            assert self.pooling == 'Mean'
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = torch.mean(attentive_related_embedding, dim=0)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = torch.mean(user_repr, dim=0)
        return user_repr

    def _get_hllm_embedding(self, tot_embedding, hllm_hyper_graph, adj, conv):
        hllm_hyper_edge_A = []
        hllm_hyper_edge_B = []
        for idx, hyper_edge in enumerate(hllm_hyper_graph):
            hllm_hyper_edge_A += [item for item in hyper_edge]
            hllm_hyper_edge_B += [idx] * len(hyper_edge)

        hllm_items = list(set(hllm_hyper_edge_A))
        sub_item2id = {item:idx for idx, item in enumerate(hllm_items)}
        sub_embedding = tot_embedding[hllm_items]

        hllm_hyper_edge = [[sub_item2id[item] for item in hllm_hyper_edge_A], hllm_hyper_edge_B]
        hllm_hyper_edge = torch.LongTensor(hllm_hyper_edge).to(self.device)

        graph_data = {
            'node_embedding': sub_embedding,
            'hyper_edge_index': hllm_hyper_edge,
            'num_hyperedges': len(hllm_hyper_graph)
        }
        embedding = self._run_hypergraph_conv(graph_data, conv)

        return embedding
    def process_hllm(self, hllm_data, id_dict):
        res_data = []
        for raw_hyper_grapth in hllm_data:
            if not isinstance(raw_hyper_grapth, list):
                continue
            temp_hyper_grapth = []
            for meta_data in raw_hyper_grapth:
                if not isinstance(meta_data, int):
                    continue
                if meta_data not in id_dict:
                    continue
                temp_hyper_grapth.append(id_dict[meta_data])
            res_data.append(temp_hyper_grapth)
        return res_data

    # old_version
    def encode_user_repr(self, related_items, related_entities, related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding,):
        # COLD START
        if self.cold_start:
            if len(related_items) == 0 or len(related_words) == 0:
                if len(related_entities) == 0:
                    user_repr = torch.zeros(self.user_emb_dim, device=self.device)
                elif self.pooling == 'Attn':
                    user_repr = tot_entity_embedding[related_entities]
                    user_repr = self.kg_attn(user_repr)
                else:
                    assert self.pooling == 'Mean'
                    user_repr = tot_entity_embedding[related_entities]
                    user_repr = torch.mean(user_repr, dim=0)
                return user_repr
        
        # 获取超图后的数据
        item_embedding = self._run_hypergraph_conv(
            self._prepare_single_hypergraph(related_items, tot_item_embedding, self.item_adj),
            self.hyper_conv_item,
            graph_key='item'
        )
        entity_embedding = self._run_hypergraph_conv(
            self._prepare_single_hypergraph(related_entities, tot_entity_embedding, self.entity_adj),
            self.hyper_conv_entity,
            graph_key='entity'
        )
        word_embedding = self._run_hypergraph_conv(
            self._prepare_single_hypergraph(related_words, tot_word_embedding, self.word_adj),
            self.hyper_conv_word,
            graph_key='word'
        )

        # 注意力机制
        if len(related_entities) == 0:
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, word_embedding, None)
        else:
            context_embedding = tot_entity_embedding[related_entities]
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, word_embedding, context_embedding)
        return user_repr

    # 获取用户编码
    def encode_user(self, batch_related_items, batch_related_entities, batch_related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding):
        user_repr_list = []
        for (related_items, related_entities, related_words) in zip(batch_related_items, batch_related_entities, batch_related_words):
            user_repr = self.encode_user_repr(
                related_items, related_entities, related_words, 
                tot_item_embedding, tot_entity_embedding, tot_word_embedding)
            user_repr_list.append(user_repr)
        user_embedding = torch.stack(user_repr_list, dim=0)
        return user_embedding

    # 推荐模块
    def recommend(self, batch, mode):
        # 获取数据
        conv_id = batch['conv_id']
        related_item = batch['related_item']
        related_entity = batch['related_entity']
        related_word = batch['related_word']
        item = batch['item']
        kg_embeddings = self._encode_kg_embeddings()
        item_embedding = kg_embeddings['item']
        entity_embedding = kg_embeddings['entity']
        token_embedding = kg_embeddings['word']

        # 获取用户编码
        # start = perf_counter()
        user_embedding = self.encode_user(
            related_item,
            related_entity,
            related_word,
            item_embedding,
            entity_embedding,
            token_embedding,
        )  # (batch_size, emb_dim)
        # print(f"{perf_counter() - start:.2f}")

        # 计算各实体得分
        scores = F.linear(user_embedding, entity_embedding, self.rec_bias.bias)  # (batch_size, n_entity)
        loss = self.rec_loss(scores, item)
        return loss, scores
    def _encode_kg_embeddings(self):
        # 对三类节点分别做一次 RGCN 编码，得到整图级节点表示。
        kg_embeddings = {
            'item': self._run_rgcn_stack(
                self.entity_embedding.weight,
                self.item_encoder,
                self.item_encoder_norms,
                self.item_encoder_dropouts
            ),
            'entity': self._run_rgcn_stack(
                self.entity_embedding.weight,
                self.entity_encoder,
                self.entity_encoder_norms,
                self.entity_encoder_dropouts
            ),
            'word': self._run_rgcn_stack(
                self.word_embedding.weight,
                self.word_encoder,
                self.word_encoder_norms,
                self.word_encoder_dropouts
            )
        }
        return kg_embeddings

    def _prepare_single_hypergraph(self, related_nodes, total_embedding, adj):
        # 当前样本没有该类型节点时，不构图，直接返回 None。
        if len(related_nodes) == 0:
            return None

        # 先根据 related_nodes 和邻接表构造当前样本的超图。
        hypergraph_nodes, hyper_edge_index = self._get_hypergraph(related_nodes, adj)
        # 再把整图 embedding 裁成当前样本子图所需的节点与边索引。
        sub_node_embedding, sub_edge_index, tot2sub = self._before_hyperconv(
            total_embedding, hypergraph_nodes, hyper_edge_index, adj
        )

        # 预先记录超边数量，避免后面重复从索引里计算。
        num_hyperedges = 0
        if sub_edge_index.numel() > 0:
            num_hyperedges = int(sub_edge_index[1].max().item()) + 1

        # 返回后续 HGCN 和 ViewLearner 共用的最小子图描述。
        return {
            'node_embedding': sub_node_embedding,
            'hyper_edge_index': sub_edge_index,
            'num_hyperedges': num_hyperedges,
            'hypergraph_nodes': hypergraph_nodes,
            'tot2sub': tot2sub
        }

    def prepare_recommendation_batch(self, batch, kg_embeddings=None):
        # 如果外部没有传入编码结果，就在这里统一做一次 RGCN 编码。
        if kg_embeddings is None:
            kg_embeddings = self._encode_kg_embeddings()

        # batch_graphs 保存 batch 内每个样本对应的 item/entity/word 子图。
        batch_graphs = []
        for related_item, related_entity, related_word in zip(
            batch['related_item'], batch['related_entity'], batch['related_word']
        ):
            # cold-start
            if self.cold_start:
                if len(related_item) == 0 or len(related_word) == 0:
                    if len(related_entity) == 0:
                        batch_graphs.append({
                            'item': None,
                            'entity': None,
                            'word': None,
                            'context_embedding': None,
                            'cold-start': True,
                            'user_repr': torch.zeros(self.user_emb_dim, device=self.device)
                        })
                        continue
                    elif self.pooling == 'Attn':
                        context_embedding = kg_embeddings['entity'][related_entity]
                        batch_graphs.append({
                            'item': None,
                            'entity': None,
                            'word': None,
                            'context_embedding': context_embedding,
                            'cold-start': True,
                            'user_repr': self.kg_attn(context_embedding)
                        })
                        continue
                    else:
                        assert self.pooling == 'Mean'
                        context_embedding = kg_embeddings['entity'][related_entity]
                        batch_graphs.append({
                            'item': None,
                            'entity': None,
                            'word': None,
                            'context_embedding': context_embedding,
                            'cold-start': True,
                            'user_repr': torch.mean(context_embedding, dim=0)
                        })
                        continue
            # related_entity 额外用于构造注意力融合时的上下文表示。
            context_embedding = None
            if len(related_entity) > 0:
                context_embedding = kg_embeddings['entity'][related_entity]

            # 将当前样本三类子图都准备好，供后续反复复用。
            batch_graphs.append({
                'item': self._prepare_single_hypergraph(related_item, kg_embeddings['item'], self.item_adj),
                'entity': self._prepare_single_hypergraph(related_entity, kg_embeddings['entity'], self.entity_adj),
                'word': self._prepare_single_hypergraph(related_word, kg_embeddings['word'], self.word_adj),
                'context_embedding': context_embedding,
                'cold-start': False
            })

        # 返回推荐阶段需要的全部中间结果。
        return {
            'item': batch['item'],
            'kg_embeddings': kg_embeddings,
            'graphs': batch_graphs
        }

    def _hypergraph_transitions(self, graph_key):
        if graph_key == 'item':
            return self.hyper_conv_item_norms, self.hyper_conv_item_dropouts
        if graph_key == 'entity':
            return self.hyper_conv_entity_norms, self.hyper_conv_entity_dropouts
        if graph_key == 'word':
            return self.hyper_conv_word_norms, self.hyper_conv_word_dropouts
        return (), ()

    @staticmethod
    def _copy_sample_graph(sample_graph):
        copied = {}
        for key, value in sample_graph.items():
            if key in {'item', 'entity', 'word'} and value is not None:
                copied[key] = dict(value)
            else:
                copied[key] = value
        return copied

    @staticmethod
    def _select_layer_weight(weight, layer_idx):
        if weight is None:
            return None
        if isinstance(weight, (list, tuple)):
            if layer_idx >= len(weight):
                return None
            return weight[layer_idx]
        return weight

    @staticmethod
    def _update_graph_embedding(sample_graph, graph_key, embedding):
        if sample_graph.get(graph_key) is None:
            return
        sample_graph[graph_key]['node_embedding'] = embedding

    def _run_hypergraph_view_layer(self, working_graph, layer_weights, layer_idx):
        layer_weights = layer_weights or {}
        item_embedding = self._run_hypergraph_conv(
            working_graph['item'],
            self.hyper_conv_item,
            layer_weights.get('item'),
            layer_idx=layer_idx,
            graph_key='item'
        )
        entity_embedding = self._run_hypergraph_conv(
            working_graph['entity'],
            self.hyper_conv_entity,
            layer_weights.get('entity'),
            layer_idx=layer_idx,
            graph_key='entity'
        )
        word_embedding = self._run_hypergraph_conv(
            working_graph['word'],
            self.hyper_conv_word,
            layer_weights.get('word'),
            layer_idx=layer_idx,
            graph_key='word'
        )

        self._update_graph_embedding(working_graph, 'item', item_embedding)
        self._update_graph_embedding(working_graph, 'entity', entity_embedding)
        self._update_graph_embedding(working_graph, 'word', word_embedding)
        return item_embedding, entity_embedding, word_embedding

    def _run_hypergraph_conv(self, graph_data, hyper_conv, hyperedge_weight=None,
                             layer_idx=None, graph_key=None):
        # 缺失图时返回零向量，保持后续拼接逻辑稳定。
        if graph_data is None:
            return torch.zeros((1, self.kg_emb_dim), device=self.device)

        norms, dropouts = self._hypergraph_transitions(graph_key)
        if isinstance(hyper_conv, nn.ModuleList):
            # 原hycorec多层HGCN路径
            if layer_idx is None:
                out = graph_data['node_embedding']
                local_graph = dict(graph_data)
                for current_layer, conv in enumerate(hyper_conv):
                    layer_weight = self._select_layer_weight(hyperedge_weight, current_layer)
                    out = conv(
                        out,
                        local_graph['hyper_edge_index'],
                        incidence_weight=layer_weight
                    )
                    out = self._apply_graph_transition(out, norms, dropouts, current_layer)
                    local_graph['node_embedding'] = out
                return out

            conv = hyper_conv[layer_idx]
            layer_weight = self._select_layer_weight(hyperedge_weight, layer_idx)
            out = conv(
                graph_data['node_embedding'],
                graph_data['hyper_edge_index'],
                incidence_weight=layer_weight
            )
            return self._apply_graph_transition(out, norms, dropouts, layer_idx)

        out = hyper_conv(
            graph_data['node_embedding'],
            graph_data['hyper_edge_index'],
            incidence_weight=hyperedge_weight
        )
        return out

    def encode_user_from_prepared_batch(self, prepared_batch,build_hyperedge_weights = False,
                                        return_hyperedge_weights=False,view_grad=False,
                                        weight_stage='train'):
        # 逐样本并行构造原图、事实图、反事实图三种用户表示。
        user_repr_list = []
        f_user_repr_list = []
        cf_user_repr_list = []
        generated_weights = {'item': [], 'entity': [], 'word': []}
        generated_cf_weights = {'item': [], 'entity': [], 'word': []}
        case_study_topo = []

        for idx, sample_graph in enumerate(prepared_batch['graphs']):
            #cold-start
            if self.cold_start:
                if sample_graph.get('cold-start', False):
                    user_repr_list.append(sample_graph['user_repr'])
                    if build_hyperedge_weights:
                        f_user_repr_list.append(sample_graph['user_repr'])
                        cf_user_repr_list.append(sample_graph['user_repr'])
                    continue

            origin_graph = self._copy_sample_graph(sample_graph)
            item_embedding = entity_embedding = word_embedding = None
            if build_hyperedge_weights:
                factual_graph = self._copy_sample_graph(sample_graph)
                counterfactual_graph = self._copy_sample_graph(sample_graph)
                sample_generated_weights = {'item': [], 'entity': [], 'word': []}
                sample_generated_cf_weights = {'item': [], 'entity': [], 'word': []}
                sample_f_topo_layers = []
                sample_cf_topo_layers = []
                f_item_embedding = f_entity_embedding = f_word_embedding = None
                cf_item_embedding = cf_entity_embedding = cf_word_embedding = None

            for layer_idx in range(self.hgcn_layers):
                item_embedding, entity_embedding, word_embedding = self._run_hypergraph_view_layer(
                    origin_graph,
                    None,
                    layer_idx
                )
                if build_hyperedge_weights:
                    if (not self.view_last) or layer_idx == self.hgcn_layers - 1:
                        if view_grad == False:
                            with torch.no_grad():
                                layer_f_weights, _, layer_f_topo = self._build_sample_hyperedge_weights(
                                    factual_graph,
                                    i=layer_idx,
                                    stage=weight_stage
                                )
                                _, layer_cf_weights, layer_cf_topo = self._build_sample_hyperedge_weights(
                                    counterfactual_graph,
                                    i=layer_idx,
                                    stage=weight_stage
                                )
                        else:
                            layer_f_weights, _, layer_f_topo = self._build_sample_hyperedge_weights(
                                factual_graph,
                                i=layer_idx,
                                stage=weight_stage
                            )
                            _, layer_cf_weights, layer_cf_topo = self._build_sample_hyperedge_weights(
                                counterfactual_graph,
                                i=layer_idx,
                                stage=weight_stage
                            )
                        for graph_key in ('item', 'entity', 'word'):
                            sample_generated_weights[graph_key].append(layer_f_weights[graph_key])
                            sample_generated_cf_weights[graph_key].append(layer_cf_weights[graph_key])
                        sample_f_topo_layers.append(layer_f_topo)
                        sample_cf_topo_layers.append(layer_cf_topo)
                    else:
                        layer_f_weights = {'item': None, 'entity': None, 'word': None}
                        layer_cf_weights = {'item': None, 'entity': None, 'word': None}

                    f_item_embedding, f_entity_embedding, f_word_embedding = self._run_hypergraph_view_layer(
                        factual_graph,
                        layer_f_weights,
                        layer_idx
                    )
                    cf_item_embedding, cf_entity_embedding, cf_word_embedding = self._run_hypergraph_view_layer(
                        counterfactual_graph,
                        layer_cf_weights,
                        layer_idx
                    )

            if build_hyperedge_weights:
                for graph_key in ('item', 'entity', 'word'):
                    generated_weights[graph_key].append(sample_generated_weights[graph_key])
                    generated_cf_weights[graph_key].append(sample_generated_cf_weights[graph_key])

                if len(sample_f_topo_layers) == 0:
                    case_study_topo.append({})
                else:
                    sample_topo = dict(sample_f_topo_layers[-1])
                    sample_topo['f_layers'] = sample_f_topo_layers
                    sample_topo['cf_layers'] = sample_cf_topo_layers
                    case_study_topo.append(sample_topo)

            # print(f"[After HGCN] {torch.cuda.memory_allocated()/1e9:.2f} GB")            
            # 将三路子图表示和上下文表示融合成单个用户向量。
            user_repr = self._attention_and_gating(
                item_embedding,
                entity_embedding,
                word_embedding,
                sample_graph['context_embedding']
            )
            user_repr_list.append(user_repr)

            if build_hyperedge_weights:
                f_user_repr = self._attention_and_gating(
                    f_item_embedding,
                    f_entity_embedding,
                    f_word_embedding,
                    sample_graph['context_embedding']
                )
                cf_user_repr = self._attention_and_gating(
                    cf_item_embedding,
                    cf_entity_embedding,
                    cf_word_embedding,
                    sample_graph['context_embedding']
                )
                f_user_repr_list.append(f_user_repr)
                cf_user_repr_list.append(cf_user_repr)



        # 堆叠成标准 batch 形状。
        user_embedding = torch.stack(user_repr_list, dim=0)
        f_user_embedding = cf_user_embedding = None
        if build_hyperedge_weights:
            f_user_embedding = torch.stack(f_user_repr_list, dim=0)
            cf_user_embedding = torch.stack(cf_user_repr_list, dim=0)
            self._last_case_study_topo = case_study_topo
        if return_hyperedge_weights:
            return user_embedding, f_user_embedding, cf_user_embedding, generated_weights, generated_cf_weights
        return user_embedding, f_user_embedding, cf_user_embedding

    def recommend_from_prepared_batch(self, prepared_batch,view_grad = False,
                                      build_hyperedge_weights=False):
        # 先基于已准备子图得到原图、事实图、反事实图三种 batch 用户表示。
        user_embedding, f_user_embedding, cf_user_embedding = self.encode_user_from_prepared_batch(prepared_batch,build_hyperedge_weights = build_hyperedge_weights,view_grad = view_grad)
        # 推荐打分始终与 entity 编码后的整图实体向量做线性匹配。
        entity_embedding = prepared_batch['kg_embeddings']['entity']

        # 1. 只保留 item 列（与 rec_evaluate 中的逻辑方向一致）
        item_ids_tensor = torch.tensor(self.item_ids, device=self.device, dtype=torch.long)

        # 2. 将 target 从全局 entity ID 映射为 item_ids 内的局部偏移
        #    与 rec_evaluate 中 self.item_ids.index(label) 逻辑完全一致
        item_to_idx = {int(eid): i for i, eid in enumerate(self.item_ids)}
        target_list = [item_to_idx[int(t.item())] for t in prepared_batch['item']]
        target_idx = torch.tensor(target_list, device=self.device)

        def score_loss_rank(current_user_embedding):
            scores = F.linear(current_user_embedding, entity_embedding, self.rec_bias.bias)
            scores = scores[:, item_ids_tensor]  # (B, len_item_ids)
            loss = self.rec_loss(scores, target_idx)
            rank = (
                scores.detach().argsort(dim=-1, descending=True) == target_idx.unsqueeze(1)
            ).float().argmax(dim=-1)
            return loss, scores, rank

        loss, scores, rank = score_loss_rank(user_embedding)
        if build_hyperedge_weights:
            f_loss, f_scores, f_rank = score_loss_rank(f_user_embedding)
            cf_loss, cf_scores, cf_rank = score_loss_rank(cf_user_embedding)
        else:
            f_loss = f_scores = f_rank = None
            cf_loss = cf_scores = cf_rank = None
        return loss, scores, target_idx, rank, f_loss, f_scores, f_rank, cf_loss, cf_scores, cf_rank
    
    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def freeze_parameters(self):
        freeze_models = [
            self.entity_embedding,
            self.token_embedding,
            self.item_encoder,
            self.entity_encoder,
            self.word_encoder,
            self.hyper_conv_item,
            self.hyper_conv_entity,
            self.hyper_conv_word,
            self.item_attn,
            self.rec_bias   
        ]
        if self.pooling == "Attn":
            freeze_models.append(self.kg_attn)
            freeze_models.append(self.kg_attn_his)
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False

    def _before_hyperconv(self, embeddings:torch.FloatTensor, hypergraph_items:List[int], edge_index:torch.LongTensor, adj):
        sub_items = []
        edge_index = edge_index.cpu().numpy()
        sub_items = list(set(hypergraph_items))
        tot2sub = {tot:sub for sub, tot in enumerate(sub_items)}
        sub_embeddings = embeddings[sub_items]
        edge_index = [[tot2sub[v] for v in edge_index[0]], list(edge_index[1])]
        sub_edge_index = torch.tensor(edge_index).long()
        sub_edge_index = sub_edge_index.to(self.device)
        return sub_embeddings, sub_edge_index, tot2sub

    # 获取超图后数据
    def encode_session(self, batch_related_items, batch_related_entities, batch_related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding):
        """
            Return: session_repr (batch_size, batch_seq_len, token_emb_dim), mask (batch_size, batch_seq_len)
        """
        session_repr_list = []
        for session_related_items, session_related_entities, session_related_words in zip(batch_related_items, batch_related_entities, batch_related_words):            
            # COLD START
            if self.cold_start:
                if len(session_related_items) == 0 or len(session_related_words) == 0:
                    if len(session_related_entities) == 0:
                        session_repr_list.append(None)
                    else:
                        session_repr = tot_entity_embedding[session_related_entities]
                        session_repr_list.append(session_repr)
                    continue

            # 获取超图后的数据
            item_embedding = self._run_hypergraph_conv(
                self._prepare_single_hypergraph(session_related_items, tot_item_embedding, self.item_adj),
                self.hyper_conv_item,
                graph_key='item'
            )
            entity_embedding = self._run_hypergraph_conv(
                self._prepare_single_hypergraph(session_related_entities, tot_entity_embedding, self.entity_adj),
                self.hyper_conv_entity,
                graph_key='entity'
            )
            word_embedding = self._run_hypergraph_conv(
                self._prepare_single_hypergraph(session_related_words, tot_word_embedding, self.word_adj),
                self.hyper_conv_word,
                graph_key='word'
            )

            # 数据拼接
            if len(session_related_entities) == 0:
                session_repr = torch.cat((item_embedding, entity_embedding, word_embedding), dim=0)
                session_repr_list.append(session_repr)
            else:
                context_embedding = tot_entity_embedding[session_related_entities]
                session_repr = torch.cat((item_embedding, entity_embedding, context_embedding, word_embedding), dim=0)
                session_repr_list.append(session_repr)

        batch_seq_len = max([session_repr.size(0) for session_repr in session_repr_list if session_repr is not None])
        mask_list = []
        for i in range(len(session_repr_list)):
            if session_repr_list[i] is None:
                mask_list.append([False] * batch_seq_len)
                zero_repr = torch.zeros((batch_seq_len, self.kg_emb_dim), device=self.device, dtype=torch.float)
                session_repr_list[i] = zero_repr
            else:
                mask_list.append([False] * (batch_seq_len - session_repr_list[i].size(0)) + [True] * session_repr_list[i].size(0))
                zero_repr = torch.zeros((batch_seq_len - session_repr_list[i].size(0), self.kg_emb_dim),
                                        device=self.device, dtype=torch.float)
                session_repr_list[i] = torch.cat((zero_repr, session_repr_list[i]), dim=0)

        session_repr_embedding = torch.stack(session_repr_list, dim=0)
        session_repr_embedding = self.entity_to_token(session_repr_embedding)
        # print("session_repr_embedding.shape", session_repr_embedding.shape) # [6, 7, 300]
        return session_repr_embedding, torch.tensor(mask_list, device=self.device, dtype=torch.bool)

    # 生成对话
    def decode_forced(self, related_encoder_state, context_encoder_state, session_state, user_embedding, resp):
        bsz = resp.size(0)
        seqlen = resp.size(1)
        inputs = resp.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, related_encoder_state, context_encoder_state, session_state)
        token_logits = F.linear(latent, self.token_embedding.weight)
        user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

        user_latent = self.entity_to_token(user_embedding)
        user_latent = user_latent.unsqueeze(1).expand(-1, seqlen, -1)
        copy_latent = torch.cat((user_latent, latent), dim=-1)
        copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
        if self.dataset == 'HReDial':
            copy_logits = copy_logits * self.hredial_copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial
        sum_logits = token_logits + user_logits + copy_logits
        _, preds = sum_logits.max(dim=-1)
        return sum_logits, preds

    # 生成对话 - test
    def decode_greedy(self, related_encoder_state, context_encoder_state, session_state, user_embedding):
        bsz = context_encoder_state[0].shape[0]
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(self.longest_label):
            scores, incr_state = self.decoder(xs, related_encoder_state, context_encoder_state, session_state, incr_state)  # incr_state is always None
            scores = scores[:, -1:, :]
            token_logits = F.linear(scores, self.token_embedding.weight)
            user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

            user_latent = self.entity_to_token(user_embedding)
            user_latent = user_latent.unsqueeze(1).expand(-1, 1, -1)
            copy_latent = torch.cat((user_latent, scores), dim=-1)
            copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
            if self.dataset == 'HReDial':
                copy_logits = copy_logits * self.hredial_copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial
            sum_logits = token_logits + user_logits + copy_logits
            probs, preds = sum_logits.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    # 对话模块训练
    def converse(self, batch, mode):
        # 获取数据
        conv_id = batch['conv_id']
        related_item = batch['related_item']
        related_entity = batch['related_entity']
        related_word = batch['related_word']
        response = batch['response']

        related_tokens = batch['related_tokens']
        context_tokens = batch['context_tokens']

        kg_embeddings = self._encode_kg_embeddings()
        item_embedding = kg_embeddings['item']
        entity_embedding = kg_embeddings['entity']
        token_embedding = kg_embeddings['word']

        # 获取对话编码
        session_state = self.encode_session(
            related_item,
            related_entity,
            related_word,
            item_embedding,
            entity_embedding,
            token_embedding,
        )

        # 获取用户编码
        # start = perf_counter()
        user_embedding = self.encode_user(
            related_item,
            related_entity,
            related_word,
            item_embedding,
            entity_embedding,
            token_embedding,
        ) # (batch_size, emb_dim)

        # 获取 X_c、X_h
        related_encoder_state = self.related_encoder(related_tokens)
        context_encoder_state = self.context_encoder(context_tokens)

        # 对话生成
        if mode != 'test':
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self.decode_forced(related_encoder_state, context_encoder_state, session_state, user_embedding, response)
            logits = logits.view(-1, logits.shape[-1])
            labels = response.view(-1)
            return self.conv_loss(logits, labels), preds
        else:
            _, preds = self.decode_greedy(related_encoder_state, context_encoder_state, session_state, user_embedding)
            return preds

    # 推荐模块和对话模块分开训练
    def forward(self, batch, mode, stage):
        if len(self.gpu) >= 2:
            self.edge_idx = self.edge_idx.cuda(torch.cuda.current_device())
            self.edge_type = self.edge_type.cuda(torch.cuda.current_device())
        if stage == "conv":
            return self.converse(batch, mode)
        if stage == "rec":
            # start = perf_counter()
            res = self.recommend(batch, mode)
            # print(f"{perf_counter() - start:.2f}")
            return res
    @staticmethod
    def gumbel_topk_select(weight, keep_ratio=0.9, mode='gumbel_topk'):
        """
        Args:
            logits: 超边连接 logits。
            keep_ratio: 目标保留比例，比如 0.3~0.6。
            mode: 选边策略
                - 'gumbel_topk': (默认) Gumbel 噪声 + top-k 选边，当前标准方法。
                - 'random': 完全随机选边，forward 随机选 k 条边，
                backward 仍通过直通估计传递 sigmoid(logits) 梯度。
                用作 ablation baseline：验证 learned selection 是否优于随机。
        
        返回：硬选择的0/1权重 + 直通梯度估计。
        """
        k = max(1, int(len(weight) * keep_ratio))

        if mode == 'random':
            # 纯随机选边（ablation baseline）
            # 随机打乱索引取前 k 个，完全不依赖 logits 排序
            random_indices = torch.randperm(len(weight), device=weight.device)[:k]

            hinge = 1e-4
            weights = torch.full_like(weight, hinge)
            weights[random_indices] = 1.0 - hinge
            # 直通梯度估计：forward 用随机硬值，backward 仍用 sigmoid(logits) 的梯度
            soft_weights = torch.sigmoid(weight)
            return weights + soft_weights - soft_weights.detach()

        # --- 默认：Gumbel 噪声 + top-k 选边 ---
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(weight) + 1e-8) + 1e-8)
        noisy_logits = weight + gumbel_noise
        
        _, top_indices = torch.topk(noisy_logits, k)
        
        # 硬选择：top-k 为 1-1e-4，其余为 1e-4
        hinge = 1e-4
        weights = torch.full_like(weight, hinge)
        weights[top_indices] = 1.0 - hinge
        # 直通梯度估计：forward用硬值，backward用 soft logits 的梯度(不用sigmoid也行)
        soft_weights = torch.sigmoid(weight)
        return weights + soft_weights - soft_weights.detach()
        
    @staticmethod
    def counterfactual_topk_select(weight, keep_ratio=0.9, cf_keep_ratio=0.2):
        """
        对每个样本的每类图，按 logits 值从高到低分为三档：
        - 前 cf_keep_ratio 高的连接 → 权重设为 hinge（抑制：ViewLearner 最认可的边）
        - 中间 (keep_ratio - cf_keep_ratio) → 权重设为 1 - hinge（保留的核心信息边）
        - 后 (1 - keep_ratio) 低的连接 → 权重设为 hinge（抑制：不重要的边）

        即：仅保留 logits 排名中间段的连接，两端均抑制，构造结构化反事实视图。

        直通梯度估计：forward 使用硬值，backward 梯度流经 sigmoid(logits)。

        Args:
            batch_logits: {'item': [tensor, ...], 'entity': [...], 'word': [...]}
                        每个 tensor 是该样本该图类型所有连接的原始 logits（1D）。
            keep_ratio: 中间+前段的总比例，即 top-keep_ratio 的 logits 参与分配。
            cf_keep_ratio: 前段（最高 logits）比例，被抑制。

        返回：
            {'item': [tensor, ...], 'entity': [...], 'word': [...]}
            与 batch_weights 结构相同的反事实权重字典。
        """
        hinge = 1e-4
        n = len(weight)
        top_k = max(1, int(n * cf_keep_ratio))
        mid_k = max(1, int(n * keep_ratio))  # top_keep_ratio 的总数

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(weight) + 1e-8) + 1e-8)
        noisy_logits = weight + gumbel_noise
        # 按 logits 从高到低排序，取前 mid_k 个索引
        _, sorted_indices = torch.topk(noisy_logits, mid_k)  # 前 mid_k 个（最高的）

        # 中间 (mid_k - top_k) → 保留为 1 - hinge
        mid_mask = torch.zeros_like(weight, dtype=torch.bool)
        mid_mask[sorted_indices[top_k:]] = True
        # mid_mask[sorted_indices[:top_k]] = True

        # forward 权重：中间段 → 1-hinge，其余（前 top_k + 后段）→ hinge
        forward_weights = torch.where(
            mid_mask,
            torch.full_like(weight, 1.0 - hinge),
            torch.full_like(weight, hinge)
        )

        # 直通梯度估计：backward 流经 sigmoid(logits)
        soft_weights = torch.sigmoid(weight)

        return forward_weights + soft_weights - soft_weights.detach()
    
    def _build_batch_hyperedge_weights(self, prepared_batch, stage='train'):
        # batch_weights[graph_key][sample_idx][layer_idx] 直接对应多层 HGCN 的 incidence weights。
        batch_weights = {'item': [], 'entity': [], 'word': []}
        batch_cf_weights = {'item': [], 'entity': [], 'word': []}
        case_study_topo = []

        for sample_graph in prepared_batch['graphs']:
            working_graph = self._copy_sample_graph(sample_graph)
            sample_weights = {'item': [], 'entity': [], 'word': []}
            sample_cf_weights = {'item': [], 'entity': [], 'word': []}
            sample_topo_layers = []

            for layer_idx in range(self.hgcn_layers):
                if (not self.view_last) or layer_idx == self.hgcn_layers - 1:
                    layer_weights, layer_cf_weights, layer_topo = self._build_sample_hyperedge_weights(
                        working_graph,
                        layer_idx,
                        stage=stage
                    )
                    sample_topo_layers.append(layer_topo)
                else:
                    layer_weights = {'item': None, 'entity': None, 'word': None}
                    layer_cf_weights = {'item': None, 'entity': None, 'word': None}

                for graph_key in ('item', 'entity', 'word'):
                    sample_weights[graph_key].append(layer_weights[graph_key])
                    sample_cf_weights[graph_key].append(layer_cf_weights[graph_key])

                item_embedding = self._run_hypergraph_conv(
                    working_graph['item'],
                    self.hyper_conv_item,
                    layer_weights['item'],
                    layer_idx=layer_idx,
                    graph_key='item'
                )
                entity_embedding = self._run_hypergraph_conv(
                    working_graph['entity'],
                    self.hyper_conv_entity,
                    layer_weights['entity'],
                    layer_idx=layer_idx,
                    graph_key='entity'
                )
                word_embedding = self._run_hypergraph_conv(
                    working_graph['word'],
                    self.hyper_conv_word,
                    layer_weights['word'],
                    layer_idx=layer_idx,
                    graph_key='word'
                )
                self._update_graph_embedding(working_graph, 'item', item_embedding)
                self._update_graph_embedding(working_graph, 'entity', entity_embedding)
                self._update_graph_embedding(working_graph, 'word', word_embedding)

            for graph_key in ('item', 'entity', 'word'):
                batch_weights[graph_key].append(sample_weights[graph_key])
                batch_cf_weights[graph_key].append(sample_cf_weights[graph_key])

            if len(sample_topo_layers) == 0:
                case_study_topo.append({})
            else:
                sample_topo = dict(sample_topo_layers[-1])
                sample_topo['layers'] = sample_topo_layers
                case_study_topo.append(sample_topo)

        self._last_case_study_topo = case_study_topo
        return batch_weights, batch_cf_weights

    def _build_sample_hyperedge_weights(self, sample_graph, i=0, stage='train'):
        # 单样本、单层：根据当前 node_embedding 为 item/entity/word 三类超图各算一次权重。
        sample_weights = {'item': None, 'entity': None, 'word': None}
        sample_cf_weights = {'item': None, 'entity': None, 'word': None}
        sample_topo = {}

        for graph_key, learner in self._view_learners(i).items():
            graph_data = sample_graph[graph_key]
            if graph_data is None:
                sample_topo[graph_key] = {
                    'nodes': [],
                    'weights': [],
                    'cf_weights': [],
                    'hyper_edge_index': []
                }
                continue

            weight_logits = learner(
                graph_data['node_embedding'],
                graph_data['hyper_edge_index']
            )
            f_weight = self.gumbel_topk_select(weight_logits, self.keep_ratio)
            cf_weight = self.counterfactual_topk_select(
                weight_logits,
                self.keep_ratio,
                self.cf_keep_ratio
            )
            sample_weights[graph_key] = f_weight
            sample_cf_weights[graph_key] = cf_weight
            sample_topo[graph_key] = self._format_sample_topology(
                graph_data,
                f_weight,
                cf_weight
            )

        return sample_weights, sample_cf_weights, sample_topo

    def _format_sample_topology(self, graph_data, connection_weight, cf_weight):
        edge_index = graph_data['hyper_edge_index']
        tot2sub = graph_data.get('tot2sub', {})
        sub2tot = {v: k for k, v in tot2sub.items()}
        weight_np = connection_weight.detach().cpu().tolist()
        cf_weight_np = cf_weight.detach().cpu().tolist()
        edge_index_np = edge_index.detach().cpu().tolist()
        num_edges = graph_data.get('num_hyperedges', 0)

        topo_per_edge = [[] for _ in range(num_edges)]
        weight_per_edge = [[] for _ in range(num_edges)]
        cf_weight_per_edge = [[] for _ in range(num_edges)]

        for conn_idx in range(len(weight_np)):
            local_node = edge_index_np[0][conn_idx]
            edge_id = edge_index_np[1][conn_idx]
            global_node = sub2tot.get(local_node, local_node)
            topo_per_edge[edge_id].append(global_node)
            weight_per_edge[edge_id].append(weight_np[conn_idx])
            cf_weight_per_edge[edge_id].append(cf_weight_np[conn_idx])

        return {
            'nodes': topo_per_edge,
            'weights': weight_per_edge,
            'cf_weights': cf_weight_per_edge,
            'hyper_edge_index': edge_index_np
        }
    
    def _view_learners(self, i):
        # 统一返回三类超图对应的第 i 层 ViewLearner。
        i = 0 if self.view_last else i
        return {
            'item': self.view_learner_item[i],
            'entity': self.view_learner_entity[i],
            'word': self.view_learner_word[i]
        }
