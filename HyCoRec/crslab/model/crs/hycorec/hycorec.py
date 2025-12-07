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

from crslab.config import DATA_PATH, DATASET_PATH
from crslab.model.base import BaseModel
from crslab.model.crs.hycorec.attention import MHItemAttention
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionBatch, SelfAttentionSeq
from crslab.model.utils.modules.transformer import TransformerEncoder
from crslab.model.crs.hycorec.decoder import TransformerDecoderKG


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
        self.device = device
        self.gpu = opt.get("gpu", -1)
        self.dataset = opt.get("dataset", None)
        self.llm = opt.get("llm", "chatgpt-4o")
        assert self.dataset in ['HReDial', 'HTGReDial', 'DuRecDial', 'OpenDialKG', 'ReDial', 'TGReDial']
        # vocab
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
            item_list = []
            for item in item_list:
                if item not in entity2id:
                    continue
                item_list.append(entity2id[item])
            item_adj[item_a] = item_list
        self.item_adj = item_adj

        entity_adj = {}
        for entity_a in entity_edger:
            entity_list = entity_edger[entity_a]
            if entity_a not in entity2id:
                continue
            entity_a = entity2id[entity_a]
            entity_list = []
            for entity in entity_list:
                if entity not in entity2id:
                    continue
                entity_list.append(entity2id[entity])
            entity_adj[entity_a] = entity_list
        self.entity_adj = entity_adj

        word_adj = {}
        for word_a in word_edger:
            word_list = word_edger[word_a]
            if word_a not in token2id:
                continue
            word_a = token2id[word_a]
            word_list = []
            for word in word_list:
                if word not in token2id:
                    continue
                word_list.append(token2id[word])
            word_adj[word_a] = word_list
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

    # 构建超图编码层
    def _build_kg_layer(self):
        # graph encoder
        self.item_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.entity_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.word_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        if self.pretrain:
            self.item_encoder.load_state_dict(self.pretrain_data['encoder'])
        # hypergraph convolution
        self.hyper_conv_item = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
        self.hyper_conv_entity = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
        self.hyper_conv_word = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
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
    def _get_embedding(self, hypergraph_items, embedding, tot_sub, adj):
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
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = self.kg_attn_his(attentive_related_embedding)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
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

        embedding = conv(sub_embedding, hllm_hyper_edge)

        return embedding
    
    def encode_user_repr(self, related_items, related_entities, related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding,
                          item_hyperedge_weight=None, entity_hyperedge_weight=None, word_hyperedge_weight=None):
        """
        编码用户表示，支持可选的超边权重。
        
        仿照 CACHE/models.py 中 SetGNN.forward() 的 edge_weight 机制：
        - 外部传入的 hyperedge_weight 是每条超边的权重 (0~1)，形状为 (num_hyperedges,)
        - 权重会传递给 HypergraphConv，影响消息传播
        
        Args:
            item_hyperedge_weight: (num_item_hyperedges,) item超图的超边权重，可选
            entity_hyperedge_weight: (num_entity_hyperedges,) entity超图的超边权重，可选
            word_hyperedge_weight: (num_word_hyperedges,) word超图的超边权重，可选
        """
        # 获取超图后的数据
        item_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
        if len(related_items) > 0:
            items, item_hyper_edge_index = self._get_hypergraph(related_items, self.item_adj)
            sub_item_embedding, sub_item_edge_index, item_tot2sub = self._before_hyperconv(tot_item_embedding, items, item_hyper_edge_index, self.item_adj)
            # 传入 hyperedge_weight 进行有权重的消息传播
            raw_item_embedding = self.hyper_conv_item(sub_item_embedding, sub_item_edge_index, hyperedge_weight=item_hyperedge_weight)
            item_embedding = raw_item_embedding

        entity_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
        if len(related_entities) > 0:
            entities, entity_hyper_edge_index = self._get_hypergraph(related_entities, self.entity_adj)
            sub_entity_embedding, sub_entity_edge_index, entity_tot2sub = self._before_hyperconv(tot_entity_embedding, entities, entity_hyper_edge_index, self.entity_adj)
            raw_entity_embedding = self.hyper_conv_entity(sub_entity_embedding, sub_entity_edge_index, hyperedge_weight=entity_hyperedge_weight)
            entity_embedding = raw_entity_embedding
            
        word_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
        if len(related_words) > 0:
            words, word_hyper_edge_index = self._get_hypergraph(related_words, self.word_adj)
            sub_word_embedding, sub_word_edge_index, word_tot2sub = self._before_hyperconv(tot_word_embedding, words, word_hyper_edge_index, self.word_adj)
            raw_word_embedding = self.hyper_conv_word(sub_word_embedding, sub_word_edge_index, hyperedge_weight=word_hyperedge_weight)
            word_embedding = raw_word_embedding

        # 注意力机制
        if len(related_entities) == 0:
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, word_embedding, None)
        else:
            context_embedding = tot_entity_embedding[related_entities]
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, word_embedding, context_embedding)
        return user_repr
    
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

    # 获取用户编码
    def encode_user(self, batch_related_items, batch_related_entities, batch_related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding,
                    batch_item_weights=None, batch_entity_weights=None, batch_word_weights=None):
        """
        批量获取用户编码，支持可选的超边权重。
        
        Args:
            batch_item_weights: list of Tensor 或 None，每个样本的 item 超边权重
            batch_entity_weights: list of Tensor 或 None
            batch_word_weights: list of Tensor 或 None
        """
        user_repr_list = []
        batch_size = len(batch_related_items)
        
        for i, (related_items, related_entities, related_words) in enumerate(zip(batch_related_items, batch_related_entities, batch_related_words)):
            # 获取当前样本的权重（如果有的话）
            item_w = batch_item_weights[i] if batch_item_weights is not None else None
            entity_w = batch_entity_weights[i] if batch_entity_weights is not None else None
            word_w = batch_word_weights[i] if batch_word_weights is not None else None
            
            user_repr = self.encode_user_repr(
                related_items, related_entities, related_words, 
                tot_item_embedding, tot_entity_embedding, tot_word_embedding,
                item_hyperedge_weight=item_w,
                entity_hyperedge_weight=entity_w,
                word_hyperedge_weight=word_w
            )
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
        item_embedding = self.item_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type)
        entity_embedding = self.entity_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type)
        token_embedding = self.word_encoder(self.word_embedding.weight, self.edge_idx, self.edge_type)

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

    def recommend_with_weight(self, batch, mode, batch_item_weights=None, batch_entity_weights=None, batch_word_weights=None):
        """
        带权重的推荐模块，仿照 CACHE/models.py 的 edge_weight 向前传播机制。
        
        外部传入 0~1 的超边权重，模型根据这些权重进行有权重的超图卷积，
        最终输出带权重影响的推荐 logits。
        
        Args:
            batch: 输入批次数据
            mode: 'train' / 'valid' / 'test'
            batch_item_weights: list of Tensor，每个样本的 item 超边权重 (num_hyperedges,)
            batch_entity_weights: list of Tensor，每个样本的 entity 超边权重
            batch_word_weights: list of Tensor，每个样本的 word 超边权重
        
        Returns:
            loss: 推荐损失
            scores: (batch_size, n_entity) 各实体得分
        """
        # 获取数据
        conv_id = batch['conv_id']
        related_item = batch['related_item']
        related_entity = batch['related_entity']
        related_word = batch['related_word']
        item = batch['item']
        
        # RGCN 编码
        item_embedding = self.item_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type)
        entity_embedding = self.entity_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type)
        token_embedding = self.word_encoder(self.word_embedding.weight, self.edge_idx, self.edge_type)

        # 带权重的用户编码
        user_embedding = self.encode_user(
            related_item,
            related_entity,
            related_word,
            item_embedding,
            entity_embedding,
            token_embedding,
            batch_item_weights=batch_item_weights,
            batch_entity_weights=batch_entity_weights,
            batch_word_weights=batch_word_weights
        )  # (batch_size, emb_dim)

        # 计算各实体得分
        scores = F.linear(user_embedding, entity_embedding, self.rec_bias.bias)  # (batch_size, n_entity)
        loss = self.rec_loss(scores, item)
        return loss, scores

    def recommend_with_weight_fn(self, batch, mode, item_weight_fn=None, entity_weight_fn=None, word_weight_fn=None):
        """
        带权重生成函数的推荐模块，仿照 CACHE/train.py 的训练机制。
        
        与 recommend_with_weight 不同的是，这里接受的是权重生成函数，
        函数会在超图卷积前被调用，根据节点特征和超边索引动态生成权重。
        
        Args:
            batch: 输入批次数据
            mode: 'train' / 'valid' / 'test'
            item_weight_fn: 函数 (node_features, hyper_edge_index) -> (weight, weight_logits)
            entity_weight_fn: 同上
            word_weight_fn: 同上
        
        Returns:
            loss: 推荐损失
            scores: (batch_size, n_entity) 各实体得分
            weight_info: list of dict，每个样本的权重信息（用于正则化）
        """
        # 获取数据
        conv_id = batch['conv_id']
        related_item = batch['related_item']
        related_entity = batch['related_entity']
        related_word = batch['related_word']
        item = batch['item']
        
        # RGCN 编码
        item_embedding = self.item_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type)
        entity_embedding = self.entity_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type)
        token_embedding = self.word_encoder(self.word_embedding.weight, self.edge_idx, self.edge_type)

        # 带权重函数的用户编码
        user_embedding, weight_info = self.encode_user_with_weight_fn(
            related_item,
            related_entity,
            related_word,
            item_embedding,
            entity_embedding,
            token_embedding,
            item_weight_fn=item_weight_fn,
            entity_weight_fn=entity_weight_fn,
            word_weight_fn=word_weight_fn
        )  # (batch_size, emb_dim), list of dict

        # 计算各实体得分
        scores = F.linear(user_embedding, entity_embedding, self.rec_bias.bias)  # (batch_size, n_entity)
        loss = self.rec_loss(scores, item)
        return loss, scores, weight_info

    def encode_user_with_weight_fn(self, batch_related_items, batch_related_entities, batch_related_words,
                                    tot_item_embedding, tot_entity_embedding, tot_word_embedding,
                                    item_weight_fn=None, entity_weight_fn=None, word_weight_fn=None):
        """
        批量获取用户编码，使用权重生成函数动态生成超边权重。
        
        Args:
            item_weight_fn: 函数 (node_features, hyper_edge_index) -> (weight, weight_logits)
            entity_weight_fn: 同上
            word_weight_fn: 同上
        
        Returns:
            user_embedding: (batch_size, emb_dim)
            weight_info: list of dict，每个样本的权重信息
        """
        user_repr_list = []
        weight_info_list = []
        
        for related_items, related_entities, related_words in zip(
            batch_related_items, batch_related_entities, batch_related_words
        ):
            user_repr, weight_info = self.encode_user_repr_with_weight_fn(
                related_items, related_entities, related_words,
                tot_item_embedding, tot_entity_embedding, tot_word_embedding,
                item_weight_fn=item_weight_fn,
                entity_weight_fn=entity_weight_fn,
                word_weight_fn=word_weight_fn
            )
            user_repr_list.append(user_repr)
            weight_info_list.append(weight_info)
        
        user_embedding = torch.stack(user_repr_list, dim=0)
        return user_embedding, weight_info_list

    def encode_user_repr_with_weight_fn(self, related_items, related_entities, related_words,
                                         tot_item_embedding, tot_entity_embedding, tot_word_embedding,
                                         item_weight_fn=None, entity_weight_fn=None, word_weight_fn=None):
        """
        编码单个用户表示，使用权重生成函数动态生成超边权重。
        
        仿照 CACHE/models.py 和 train.py 的机制：
        1. 构建超图
        2. 调用权重生成函数得到边权重
        3. 将边权重聚合为超边权重
        4. 使用超边权重进行超图卷积
        
        Args:
            item_weight_fn: 函数 (node_features, hyper_edge_index) -> (weight, weight_logits)
        
        Returns:
            user_repr: 用户表示向量
            weight_info: dict，包含 item/entity/word 的权重信息
        """
        weight_info = {'item': None, 'entity': None, 'word': None}
        
        # Item 超图
        item_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
        if len(related_items) > 0:
            items, item_hyper_edge_index = self._get_hypergraph(related_items, self.item_adj)
            sub_item_embedding, sub_item_edge_index, item_tot2sub = self._before_hyperconv(
                tot_item_embedding, items, item_hyper_edge_index, self.item_adj
            )
            
            hyperedge_weight = None
            if item_weight_fn is not None:
                # 调用权重生成函数
                edge_weight, weight_logits = item_weight_fn(sub_item_embedding, sub_item_edge_index)
                # 将边权重聚合为超边权重（mean pooling）
                hyperedge_weight = self._edge_weight_to_hyperedge_weight(edge_weight, sub_item_edge_index)
                weight_info['item'] = {'weight': edge_weight, 'logits': weight_logits}
            
            raw_item_embedding = self.hyper_conv_item(sub_item_embedding, sub_item_edge_index, hyperedge_weight=hyperedge_weight)
            item_embedding = raw_item_embedding

        # Entity 超图
        entity_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
        if len(related_entities) > 0:
            entities, entity_hyper_edge_index = self._get_hypergraph(related_entities, self.entity_adj)
            sub_entity_embedding, sub_entity_edge_index, entity_tot2sub = self._before_hyperconv(
                tot_entity_embedding, entities, entity_hyper_edge_index, self.entity_adj
            )
            
            hyperedge_weight = None
            if entity_weight_fn is not None:
                edge_weight, weight_logits = entity_weight_fn(sub_entity_embedding, sub_entity_edge_index)
                hyperedge_weight = self._edge_weight_to_hyperedge_weight(edge_weight, sub_entity_edge_index)
                weight_info['entity'] = {'weight': edge_weight, 'logits': weight_logits}
            
            raw_entity_embedding = self.hyper_conv_entity(sub_entity_embedding, sub_entity_edge_index, hyperedge_weight=hyperedge_weight)
            entity_embedding = raw_entity_embedding
            
        # Word 超图
        word_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
        if len(related_words) > 0:
            words, word_hyper_edge_index = self._get_hypergraph(related_words, self.word_adj)
            sub_word_embedding, sub_word_edge_index, word_tot2sub = self._before_hyperconv(
                tot_word_embedding, words, word_hyper_edge_index, self.word_adj
            )
            
            hyperedge_weight = None
            if word_weight_fn is not None:
                edge_weight, weight_logits = word_weight_fn(sub_word_embedding, sub_word_edge_index)
                hyperedge_weight = self._edge_weight_to_hyperedge_weight(edge_weight, sub_word_edge_index)
                weight_info['word'] = {'weight': edge_weight, 'logits': weight_logits}
            
            raw_word_embedding = self.hyper_conv_word(sub_word_embedding, sub_word_edge_index, hyperedge_weight=hyperedge_weight)
            word_embedding = raw_word_embedding

        # 注意力机制
        if len(related_entities) == 0:
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, word_embedding, None)
        else:
            context_embedding = tot_entity_embedding[related_entities]
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, word_embedding, context_embedding)
        
        return user_repr, weight_info

    def _edge_weight_to_hyperedge_weight(self, edge_weight, hyper_edge_index, aggregation='mean'):
        """
        将边权重聚合为超边权重。
        
        仿照 CACHE/models.py 中的 edge_weight 机制：
        - edge_weight: (num_connections,) 每条 node-hyperedge 连接的权重
        - 聚合为 hyperedge_weight: (num_hyperedges,) 每条超边的权重
        
        Args:
            edge_weight: (num_connections,) 边权重
            hyper_edge_index: (2, num_connections) [节点ID, 超边ID]
            aggregation: 聚合方式 'mean', 'sum', 'max'
        
        Returns:
            hyperedge_weight: (num_hyperedges,) 超边权重
        """
        if edge_weight is None:
            return None
            
        # 确保 edge_weight 是 1D
        if edge_weight.dim() > 1:
            edge_weight = edge_weight.squeeze(-1)
        
        hedge_ids = hyper_edge_index[1]
        num_hedges = hedge_ids.max().item() + 1
        
        if aggregation == 'mean':
            hyperedge_weight = torch.zeros(num_hedges, device=edge_weight.device)
            hedge_count = torch.zeros(num_hedges, device=edge_weight.device)
            hyperedge_weight.scatter_add_(0, hedge_ids, edge_weight)
            hedge_count.scatter_add_(0, hedge_ids, torch.ones_like(edge_weight))
            hedge_count = hedge_count.clamp(min=1)
            hyperedge_weight = hyperedge_weight / hedge_count
        elif aggregation == 'sum':
            hyperedge_weight = torch.zeros(num_hedges, device=edge_weight.device)
            hyperedge_weight.scatter_add_(0, hedge_ids, edge_weight)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return hyperedge_weight

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
        for item in hypergraph_items:
            sub_items += [item] + list(adj.get(item, []))
        sub_items = list(set(sub_items))
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
            # if len(session_related_items) == 0 or len(session_related_words) == 0:
            #     if len(session_related_entities) == 0:
            #         session_repr_list.append(None)
            #     else:
            #         session_repr = tot_entity_embedding[session_related_entities]
            #         session_repr_list.append(session_repr)
            #     continue

            # 获取超图后的数据
            item_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
            if len(session_related_items) > 0:
                items, item_hyper_edge_index = self._get_hypergraph(session_related_items, self.item_adj)
                sub_item_embedding, sub_item_edge_index, item_tot2sub = self._before_hyperconv(tot_item_embedding, items, item_hyper_edge_index, self.item_adj)
                raw_item_embedding = self.hyper_conv_item(sub_item_embedding, sub_item_edge_index)
                item_embedding = raw_item_embedding
                # item_embedding = self._get_embedding(items, raw_item_embedding, item_tot2sub, self.item_adj)

            entity_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
            if len(session_related_entities) > 0:
                entities, entity_hyper_edge_index = self._get_hypergraph(session_related_entities, self.entity_adj)
                sub_entity_embedding, sub_entity_edge_index, entity_tot2sub = self._before_hyperconv(tot_entity_embedding, entities, entity_hyper_edge_index, self.entity_adj)
                raw_entity_embedding = self.hyper_conv_entity(sub_entity_embedding, sub_entity_edge_index)
                entity_embedding = raw_entity_embedding
                # entity_embedding = self._get_embedding(entities, raw_entity_embedding, entity_tot2sub, self.entity_adj)
                
            word_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
            if len(session_related_words) > 0:
                owrds, word_hyper_edge_index = self._get_hypergraph(session_related_words, self.word_adj)
                sub_word_embedding, sub_word_edge_index, word_tot2sub = self._before_hyperconv(tot_word_embedding, owrds, word_hyper_edge_index, self.word_adj)
                raw_word_embedding = self.hyper_conv_word(sub_word_embedding, sub_word_edge_index)
                word_embedding = raw_word_embedding
                # word_embedding = self._get_embedding(owrds, raw_word_embedding, word_tot2sub, self.word_adj)

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

        item_embedding = self.item_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type)
        entity_embedding = self.entity_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type)
        token_embedding = self.word_encoder(self.word_embedding.weight, self.edge_idx, self.edge_type)

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


class ViewLearner(nn.Module):
    """
    为超图中的每条 node-hyperedge 连接学习概率权重。
    
    逻辑：
    1. 使用独立的 HypergraphConv 作为 encoder 提取特征
    2. 聚合得到超边表示（与主模型 _get_embedding 逻辑一致，使用 mean pooling）
    3. 将节点表示和对应超边表示拼接，通过 MLP 得到权重 logits
    """
    
    def __init__(self, input_dim, hidden_dim=64, device=None):
        """
        Args:
            input_dim: 输入特征维度 (对应 HyCoRec 的 kg_emb_dim)
            hidden_dim: MLP 隐藏层维度
            device: 计算设备
        """
        super(ViewLearner, self).__init__()
        
        self.input_dim = input_dim
        self.device = device
        
        # 独立的 encoder (与主模型结构相同但参数独立)
        self.encoder = HypergraphConv(input_dim, input_dim)
        
        # MLP: [node_feat || hedge_feat] -> weight
        self.mlp_edge_model = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _aggregate_hyperedge_embedding(self, node_embedding, hyper_edge_index):
        """
        聚合得到超边表示（与主模型 _get_embedding 逻辑一致，使用 mean pooling）
        
        Args:
            node_embedding: (num_nodes, input_dim) encoder 输出的节点特征
            hyper_edge_index: (2, num_connections) [节点ID, 超边ID]
        
        Returns:
            hedge_embedding: (num_hyperedges, input_dim) 超边表示
        """
        node_ids = hyper_edge_index[0]
        hedge_ids = hyper_edge_index[1]
        num_hedges = hedge_ids.max().item() + 1
        
        # scatter mean: 对每条超边内的节点取平均
        node_emb = node_embedding[node_ids]  # (num_connections, dim)
        hedge_emb = torch.zeros(num_hedges, self.input_dim, device=node_embedding.device)
        
        # 计算每条超边的节点数（用于 mean）
        ones = torch.ones(node_ids.size(0), 1, device=node_embedding.device)
        hedge_count = torch.zeros(num_hedges, 1, device=node_embedding.device)
        hedge_count.scatter_add_(0, hedge_ids.unsqueeze(-1), ones)
        hedge_count = hedge_count.clamp(min=1)  # 避免除零
        
        # 累加节点特征
        hedge_emb.scatter_add_(
            0,
            hedge_ids.unsqueeze(-1).expand(-1, self.input_dim),
            node_emb
        )
        # 取平均
        hedge_emb = hedge_emb / hedge_count
        
        return hedge_emb
    
    def forward(self, node_features, hyper_edge_index):
        """
        Args:
            node_features: (num_nodes, input_dim) 输入节点特征（来自 RGCN 编码后）
            hyper_edge_index: (2, num_connections) 超图连接 [节点ID, 超边ID]
        
        Returns:
            weight_logits: (num_connections, 1) 每条连接的权重 logits
        """
        # 1. 通过独立的 encoder 提取特征
        encoded_node_feat = self.encoder(node_features, hyper_edge_index)
        # encoded_node_feat: (num_nodes, input_dim)
        
        # 2. 聚合得到超边表示
        hedge_embedding = self._aggregate_hyperedge_embedding(encoded_node_feat, hyper_edge_index)
        # hedge_embedding: (num_hyperedges, input_dim)
        
        # 3. 获取每条连接对应的节点和超边特征
        node_ids = hyper_edge_index[0]
        hedge_ids = hyper_edge_index[1]
        
        emb_node = encoded_node_feat[node_ids]   # (num_connections, dim)
        emb_hedge = hedge_embedding[hedge_ids]    # (num_connections, dim)
        
        # 4. 拼接后通过 MLP 得到权重
        total_emb = torch.cat([emb_node, emb_hedge], dim=1)  # (num_connections, dim*2)
        weight_logits = self.mlp_edge_model(total_emb)       # (num_connections, 1)
        
        return weight_logits