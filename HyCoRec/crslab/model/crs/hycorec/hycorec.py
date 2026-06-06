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


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Compute mean aggregation over the first dimension with scatter-add."""
    # 输出张量的首维对应聚合后的 group 数量，后续维度保持与输入一致。
    out_shape = (dim_size,) + tuple(values.shape[1:])
    # 先创建零张量，后面通过 scatter_add 把同组元素累加进去。
    out = values.new_zeros(out_shape)
    # 为了适配高维特征，这里把一维 index 扩展到与 values 同形状。
    expand_shape = (index.size(0),) + (1,) * (values.dim() - 1)
    # 按照 index 指定的 group 做加和。
    out.scatter_add_(0, index.view(*expand_shape).expand_as(values), values)

    # 统计每个 group 中有多少元素，后面用于求均值。
    count = values.new_zeros(dim_size)
    count.scatter_add_(0, index, values.new_ones(index.size(0)))
    # 防止某个 group 为空导致除零。
    count = count.clamp_min(1.0)
    # 将计数 reshape 成可广播形状。
    view_shape = (dim_size,) + (1,) * (values.dim() - 1)
    # 返回按 group 聚合后的均值。
    return out / count.view(*view_shape)

def check(name, x, log_file="debug.log"):
    if not isinstance(x, torch.Tensor):
        all_finite = "non-tensor"
        min_val = "non-tensor"
        max_val = "non-tensor"
    else:
        finite_mask = torch.isfinite(x)
        all_finite = finite_mask.all().item()
        any_finite = finite_mask.any().item()

        if any_finite:
            finite_vals = x[finite_mask]
            min_val = finite_vals.min().item()
            max_val = finite_vals.max().item()
        else:
            min_val = "nan"
            max_val = "nan"

    log_path = os.path.join(os.path.dirname(__file__), log_file) if "__file__" in globals() else log_file

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{name} {all_finite} {min_val} {max_val}\n")


def _numeric_debug_location():
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return 'unknown'
    caller = frame.f_back.f_back
    return f'{os.path.basename(caller.f_code.co_filename)}:{caller.f_lineno}'


def _numeric_debug_tensor(owner, name, tensor):
    if not getattr(owner, 'nan_debug', False):
        return True
    if not isinstance(tensor, torch.Tensor):
        return True
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    if torch.isfinite(tensor).all().item():
        return True

    with torch.no_grad():
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        total = detached.numel()
        stats = {
            'shape': tuple(detached.shape),
            'dtype': str(detached.dtype),
            'device': str(detached.device),
            'finite': f'{finite_count}/{total}',
            'nan': int(torch.isnan(detached).sum().item()) if detached.is_floating_point() else 0,
            '+inf': int(torch.isposinf(detached).sum().item()) if detached.is_floating_point() else 0,
            '-inf': int(torch.isneginf(detached).sum().item()) if detached.is_floating_point() else 0,
        }
        if finite_count > 0:
            finite_vals = detached[finite_mask].float()
            stats.update({
                'min': float(finite_vals.min().item()),
                'max': float(finite_vals.max().item()),
                'mean': float(finite_vals.mean().item()),
            })

    message = f"[NUMERIC DEBUG] non-finite tensor '{name}' at {_numeric_debug_location()} stats={stats}"
    logger.error(message)
    if getattr(owner, 'nan_debug_raise', True):
        raise FloatingPointError(message)
    return False

def gumbel_softmax(logits, temperature=1.0, clip_eps=0.0):
    """Gumbel-Softmax trick for differentiable sampling"""
    bias = 0.0001
    eps = (bias - (1 - bias)) * torch.rand(logits.size(), device=logits.device) + (1 - bias)
    gate_inputs = torch.log(eps) - torch.log(1 - eps)
    gate_inputs = (gate_inputs + logits) / temperature
    probs = torch.sigmoid(gate_inputs).reshape(-1)
    if clip_eps > 0:
        probs = probs.clamp(min=clip_eps, max=1.0 - clip_eps)
    return probs

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
        self.hgcn_layers = int(opt.get('hgcn_layers', 2))
        if self.hgcn_layers < 1:
            raise ValueError(f'hgcn_layers must be >= 1, got {self.hgcn_layers}')
        self.hidden_dim = int(opt.get('hidden_dim', self.kg_emb_dim))
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
        # view learner
        view_opt = opt.get('view', {})
        self.temperature = view_opt.get('temperature', 1.0)
        logger.info(f"[View Learner] temperature={self.temperature}")
        self.weight_clip_eps = view_opt.get('weight_clip_eps', 0.0)

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

    def _build_hyper_conv_stack(self):
        layers = []
        for layer_idx in range(self.hgcn_layers):
            in_dim = self.kg_emb_dim if layer_idx == 0 else self.hidden_dim
            out_dim = self.kg_emb_dim if layer_idx == self.hgcn_layers - 1 else self.hidden_dim
            layers.append(WeightedHypergraphConv(in_dim, out_dim))
        return nn.ModuleList(layers)

    @staticmethod
    def _get_layer_module(module_or_list, layer_idx):
        if module_or_list is None:
            return None
        if isinstance(module_or_list, (nn.ModuleList, list, tuple)):
            if layer_idx >= len(module_or_list):
                return None
            return module_or_list[layer_idx]
        return module_or_list if layer_idx == 0 else None

    # 构建超图编码层
    def _build_kg_layer(self):
        # graph encoder
        self.item_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.entity_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.word_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        if self.pretrain:
            self.item_encoder.load_state_dict(self.pretrain_data['encoder'])
        # hypergraph convolution
        self.hyper_conv_item = self._build_hyper_conv_stack()
        self.hyper_conv_entity = self._build_hyper_conv_stack()
        self.hyper_conv_word = self._build_hyper_conv_stack()
        logger.info(f'convlayers = {self.hgcn_layers}, hidden_dim = {self.hidden_dim}')
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
    
    def encode_user_repr(self, related_items, related_entities, related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding,):
        # 获取超图后的数据
        # COLD START
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
        item_graph = self._prepare_single_hypergraph(related_items, tot_item_embedding, self.item_adj)
        item_embedding = self._run_hypergraph_conv(item_graph, self.hyper_conv_item, view_learner=None)

        entity_graph = self._prepare_single_hypergraph(related_entities, tot_entity_embedding, self.entity_adj)
        entity_embedding = self._run_hypergraph_conv(entity_graph, self.hyper_conv_entity, view_learner=None)

        word_graph = self._prepare_single_hypergraph(related_words, tot_word_embedding, self.word_adj)
        word_embedding = self._run_hypergraph_conv(word_graph, self.hyper_conv_word, view_learner=None)

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
    def _encode_kg_embeddings(self):
        # 对三类节点分别做一次 RGCN 编码，得到整图级节点表示。
        kg_embeddings = {
            'item': self.item_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type),
            'entity': self.entity_encoder(self.entity_embedding.weight, self.edge_idx, self.edge_type),
            'word': self.word_encoder(self.word_embedding.weight, self.edge_idx, self.edge_type)
        }
        for graph_key, embedding in kg_embeddings.items():
            _numeric_debug_tensor(self, f'_encode_kg_embeddings.{graph_key}', embedding)
        return kg_embeddings

    def _prepare_single_hypergraph(self, related_nodes, total_embedding, adj):
        # 当前样本没有该类型节点时，不构图，直接返回 None。
        if len(related_nodes) == 0:
            return None

        # 先根据 related_nodes 和邻接表构造当前样本的超图。
        hypergraph_nodes, hyper_edge_index = self._get_hypergraph(related_nodes, adj)
        # 再把整图 embedding 裁成当前样本子图所需的节点与边索引。
        sub_node_embedding, sub_edge_index, _ = self._before_hyperconv(
            total_embedding, hypergraph_nodes, hyper_edge_index, adj
        )
        _numeric_debug_tensor(self, '_prepare_single_hypergraph.sub_node_embedding', sub_node_embedding)

        # 预先记录超边数量，避免后面重复从索引里计算。
        num_hyperedges = 0
        if sub_edge_index.numel() > 0:
            num_hyperedges = int(sub_edge_index[1].max().item()) + 1

        # 返回后续 HGCN 和 ViewLearner 共用的最小子图描述。
        return {
            'node_embedding': sub_node_embedding,
            'hyper_edge_index': sub_edge_index,
            'num_hyperedges': num_hyperedges
        }

    def prepare_recommendation_batch(self, batch, kg_embeddings=None):
        # 如果外部没有传入编码结果，就在这里统一做一次 RGCN 编码。
        if kg_embeddings is None:
            kg_embeddings = self._encode_kg_embeddings()

        # batch_graphs 保存 batch 内每个样本对应的 item/entity/word 子图。
        batch_graphs = []
        tot_entity = kg_embeddings['entity']
        for related_item, related_entity, related_word in zip(
            batch['related_item'], batch['related_entity'], batch['related_word']
        ):
            # cold-start
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
                elif self.pooling == 'Attn':
                    batch_graphs.append({
                        'item': None,
                        'entity': None,
                        'word': None,
                        'context_embedding': None,                       
                        'cold-start': True,
                        'user_repr': self.kg_attn(tot_entity[related_entity])
                    })
                else:
                    batch_graphs.append({
                        'item': None,
                        'entity': None,
                        'word': None,
                        'context_embedding': None,                      
                        'cold-start': True,
                        'user_repr': torch.mean(tot_entity[related_entity], dim=0)
                    })
                continue
            # related_entity 额外用于构造注意力融合时的上下文表示。
            context_embedding = None
            if len(related_entity) > 0:
                context_embedding = tot_entity[related_entity]

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

    def _run_hypergraph_conv(self, graph_data, hyper_conv, incidence_weight=None, view_learner=None):
        # 缺失图时返回零向量，保持后续拼接逻辑稳定。
        if graph_data is None:
            return torch.zeros((1, self.kg_emb_dim), device=self.device)
        # 对当前样本子图按层执行 HGCN。这里的 hyperedge_weight 实际是连接级 incidence 权重。
        node_embedding = graph_data['node_embedding']
        hyper_edge_index = graph_data['hyper_edge_index']
        num_hyperedges = graph_data.get('num_hyperedges', None)
        hyper_convs = hyper_conv if isinstance(hyper_conv, nn.ModuleList) else [hyper_conv]

        _numeric_debug_tensor(self, '_run_hypergraph_conv.node_embedding_input', node_embedding)
        _numeric_debug_tensor(self, '_run_hypergraph_conv.hyperedge_weight_input', incidence_weight)
        for layer_idx, conv in enumerate(hyper_convs):
            if view_learner is not None:
                if layer_idx == 0:
                    incidence_weight = incidence_weight
                else:
                    learner = self._get_layer_module(view_learner, layer_idx)
                    assert learner is not None, f"View learner for layer {layer_idx} is not found."
                    weight_logits = learner(node_embedding, hyper_edge_index)
                    incidence_weight = gumbel_softmax(
                        weight_logits,
                        self.temperature,
                        self.weight_clip_eps
                    )

            _numeric_debug_tensor(self, f'_run_hypergraph_conv.layer_{layer_idx}.incidence_weight', incidence_weight)
            node_embedding = conv(
                node_embedding,
                hyper_edge_index,
                num_edges=num_hyperedges,
                incidence_weight=incidence_weight
            )
            _numeric_debug_tensor(self, f'_run_hypergraph_conv.layer_{layer_idx}.output', node_embedding)

        _numeric_debug_tensor(self, '_run_hypergraph_conv.output', node_embedding)
        return node_embedding

    def encode_user_from_prepared_batch(self, prepared_batch, batch_item_weights=None,
                                        batch_entity_weights=None, batch_word_weights=None,
                                        item_view_learner=None, entity_view_learner=None,
                                        word_view_learner=None):
        # 逐样本构造最终用户表示，因为每个样本的子图结构都不同。
        user_repr_list = []

        for idx, sample_graph in enumerate(prepared_batch['graphs']):
            #cold-start
            if sample_graph.get('cold-start', False):
                user_repr_list.append(sample_graph['user_repr'])
                continue
            # 逐样本取出三类图的超边权重；若无权版本则为 None。
            item_weight = batch_item_weights[idx] if batch_item_weights is not None else None
            entity_weight = batch_entity_weights[idx] if batch_entity_weights is not None else None
            word_weight = batch_word_weights[idx] if batch_word_weights is not None else None
            # item/entity/word 三类子图各自执行带权或不带权的 HGCN。
            item_embedding = self._run_hypergraph_conv(
                sample_graph['item'], self.hyper_conv_item, item_weight, item_view_learner
            )
            entity_embedding = self._run_hypergraph_conv(
                sample_graph['entity'], self.hyper_conv_entity, entity_weight, entity_view_learner
            )
            word_embedding = self._run_hypergraph_conv(
                sample_graph['word'], self.hyper_conv_word, word_weight, word_view_learner
            )
            _numeric_debug_tensor(self, f'encode_user_from_prepared_batch[{idx}].item_embedding', item_embedding)
            _numeric_debug_tensor(self, f'encode_user_from_prepared_batch[{idx}].entity_embedding', entity_embedding)
            _numeric_debug_tensor(self, f'encode_user_from_prepared_batch[{idx}].word_embedding', word_embedding)
            _numeric_debug_tensor(self, f'encode_user_from_prepared_batch[{idx}].context_embedding', sample_graph['context_embedding'])
            # 将三路子图表示和上下文表示融合成单个用户向量。
            user_repr = self._attention_and_gating(
                item_embedding,
                entity_embedding,
                word_embedding,
                sample_graph['context_embedding']
            )
            _numeric_debug_tensor(self, f'encode_user_from_prepared_batch[{idx}].user_repr', user_repr)
            # 收集 batch 内所有样本的用户表示。
            user_repr_list.append(user_repr)

        # 堆叠成标准 batch 形状。
        user_embedding = torch.stack(user_repr_list, dim=0)
        _numeric_debug_tensor(self, 'encode_user_from_prepared_batch.user_embedding', user_embedding)
        return user_embedding

    def recommend_from_prepared_batch(self, prepared_batch, batch_item_weights=None,
                                      batch_entity_weights=None, batch_word_weights=None,
                                      item_view_learner=None, entity_view_learner=None,
                                      word_view_learner=None):
        # 先基于已准备子图得到 batch 用户表示。
        user_embedding = self.encode_user_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_item_weights,
            batch_entity_weights=batch_entity_weights,
            batch_word_weights=batch_word_weights,
            item_view_learner=item_view_learner,
            entity_view_learner=entity_view_learner,
            word_view_learner=word_view_learner
        )
        _numeric_debug_tensor(self, 'recommend_from_prepared_batch.user_embedding', user_embedding)
        # 推荐打分始终与 entity 编码后的整图实体向量做线性匹配。
        entity_embedding = prepared_batch['kg_embeddings']['entity']
        _numeric_debug_tensor(self, 'recommend_from_prepared_batch.entity_embedding', entity_embedding)
        scores = F.linear(user_embedding, entity_embedding, self.rec_bias.bias)
        _numeric_debug_tensor(self, 'recommend_from_prepared_batch.scores', scores)
        # 交叉熵损失仍使用原始 item 标签监督。
        loss = self.rec_loss(scores, prepared_batch['item'])
        _numeric_debug_tensor(self, 'recommend_from_prepared_batch.loss', loss)
        return loss, scores, prepared_batch['item']

    def build_batch_hyperedge_weights(self, prepared_batch, item_weight_fn=None,
                                      entity_weight_fn=None, word_weight_fn=None):
        # 保存每个样本、每类图的超边权重。
        batch_weights = {'item': [], 'entity': [], 'word': []}
        # 额外保存拍平后的权重，供外部做统一统计。
        flat_weight_info = {'item': [], 'entity': [], 'word': []}

        # 逐样本生成权重，适配可变大小子图。
        for sample_graph in prepared_batch['graphs']:
            # 三类图共用同一套接口，但权重函数各自独立。
            for graph_key, weight_fn in (
                ('item', item_weight_fn),
                ('entity', entity_weight_fn),
                ('word', word_weight_fn)
            ):
                # 取出当前样本当前图类型的子图。
                graph_data = sample_graph[graph_key]
                # 无图或未提供权重函数时，表示该分支不启用权重。
                if graph_data is None or weight_fn is None:
                    batch_weights[graph_key].append(None)
                    continue

                # 先生成连接级权重。
                connection_weight, _ = weight_fn(
                    graph_data['node_embedding'],
                    graph_data['hyper_edge_index']
                )
                # 保存逐样本超边权重。
                batch_weights[graph_key].append(connection_weight)
                # 也保存展平版本供统计使用。
                flat_weight_info[graph_key].append(connection_weight.reshape(-1))

        # 将拍平后的列表转成张量字典。
        weight_info = {}
        for graph_key, weights in flat_weight_info.items():
            # 若某一图类型在当前 batch 中不存在，则返回零张量避免空列表问题。
            if len(weights) == 0:
                weight_info[graph_key] = torch.zeros(1, device=self.device)
            else:
                # 否则直接拼接所有样本的权重。
                weight_info[graph_key] = torch.cat(weights, dim=0)

        return batch_weights, weight_info

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
            if len(session_related_items) == 0 or len(session_related_words) == 0:
                if len(session_related_entities) == 0:
                    session_repr_list.append(None)
                else:
                    session_repr = tot_entity_embedding[session_related_entities]
                    session_repr_list.append(session_repr)
                continue

            # 获取超图后的数据
            item_graph = self._prepare_single_hypergraph(session_related_items, tot_item_embedding, self.item_adj)
            item_embedding = self._run_hypergraph_conv(item_graph, self.hyper_conv_item, view_learner=None)

            entity_graph = self._prepare_single_hypergraph(session_related_entities, tot_entity_embedding, self.entity_adj)
            entity_embedding = self._run_hypergraph_conv(entity_graph, self.hyper_conv_entity, view_learner=None)

            word_graph = self._prepare_single_hypergraph(session_related_words, tot_word_embedding, self.word_adj)
            word_embedding = self._run_hypergraph_conv(word_graph, self.hyper_conv_word, view_learner=None)

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


class HyperedgeAttentionPooling(nn.Module):
    """CACHE-style PMA-inspired pooling from incident nodes to hyperedges."""

    def __init__(self, input_dim, hidden_dim):
        super(HyperedgeAttentionPooling, self).__init__()
        # 保存输入维度，供后面创建参数和输出张量使用。
        self.input_dim = input_dim
        # 将节点特征映射到 attention key 空间。
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        # 将节点特征映射到 value 空间，最终参与超边表示聚合。
        self.value_proj = nn.Linear(input_dim, input_dim)
        # 类似 PMA 中的 seed，作为每个超边共享的初始查询基底。
        self.seed = nn.Parameter(torch.empty(1, input_dim))
        # 全局可学习 query，用于衡量节点对超边表示的贡献度。
        self.query = nn.Parameter(torch.empty(1, hidden_dim))
        # 两层 LayerNorm 对应 attention 聚合后和 FFN 后的稳定化。
        self.norm0 = nn.LayerNorm(input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        # 一个轻量 FFN，用于增强聚合后的超边表示。
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        # 显式初始化所有参数。
        self.reset_parameters()

    def reset_parameters(self):
        # 线性层使用 Xavier 初始化，和项目内其他 MLP 保持一致。
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        # seed 和 query 也是可学习矩阵，同样采用 Xavier 初始化。
        nn.init.xavier_uniform_(self.seed)
        nn.init.xavier_uniform_(self.query)
        # 归一化层重置为默认状态。
        self.norm0.reset_parameters()
        self.norm1.reset_parameters()
        # FFN 内部线性层单独初始化。
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, node_embedding, hyper_edge_index):
        # 从 incidence 矩阵中取出“节点索引”和“超边索引”。
        node_ids = hyper_edge_index[0]
        hedge_ids = hyper_edge_index[1]
        # 超边数量默认由索引最大值推断。
        num_hyperedges = int(hedge_ids.max().item()) + 1

        # 只取实际参与当前超边连接的节点表示。
        node_repr = node_embedding[node_ids]
        # 节点先映射到 key 空间，再与全局 query 做点积得到 attention logits。
        attn_logits = (self.key_proj(node_repr) * self.query).sum(dim=-1)
        # 标准缩放，避免 logits 过大。
        attn_logits = attn_logits / math.sqrt(self.query.size(-1))
        # 在同一条超边内部做 softmax，得到每个连接的注意力分数。
        attn_scores = softmax(attn_logits, hedge_ids, num_nodes=num_hyperedges)

        # 为每条超边创建输出槽位。
        hedge_embedding = node_embedding.new_zeros(num_hyperedges, self.input_dim)
        # 将 value 按注意力加权后累加到对应超边上。
        hedge_embedding.index_add_(
            0,
            hedge_ids,
            self.value_proj(node_repr) * attn_scores.unsqueeze(-1)
        )

        # 加上 seed 后做第一层归一化，对齐 PMA 风格的残差结构。
        hedge_embedding = self.norm0(hedge_embedding + self.seed)
        # 再经过 FFN 和第二层归一化，增强超边表示能力。
        hedge_embedding = self.norm1(hedge_embedding + self.ffn(hedge_embedding))
        return hedge_embedding


class ViewLearner(nn.Module):
    """Learn connection logits from node-hyperedge pairs."""

    def __init__(self, input_dim, hidden_dim=64, device=None, hyperedge_aggregation='mean'):
        super(ViewLearner, self).__init__()

        # 只允许两种超边表示方式，便于配置管理。
        if hyperedge_aggregation not in {'mean', 'attention'}:
            raise ValueError(f'Unsupported hyperedge_aggregation: {hyperedge_aggregation}')

        # 记录基础配置。
        self.input_dim = input_dim
        self.device = device
        self.hyperedge_aggregation = hyperedge_aggregation

        # 独立 encoder 先对节点做一次超图编码，再交给边权预测头使用。
        self.encoder = HypergraphConv(input_dim, input_dim)
        # 边权头输入是“连接上的节点表示 + 对应超边表示”的拼接。
        self.mlp_edge_model = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 默认不启用 attention pooling，只有配置为 attention 时才创建模块。
        self.attention_pool = None
        if self.hyperedge_aggregation == 'attention':
            self.attention_pool = HyperedgeAttentionPooling(input_dim, hidden_dim)

        # 初始化所有可学习参数。
        self._init_weights()

    def _init_weights(self):
        # 统一初始化当前模块下所有线性层。
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _aggregate_hyperedge_embedding(self, node_embedding, hyper_edge_index):
        # incidence 矩阵第一行是节点，第二行是超边。
        node_ids = hyper_edge_index[0]
        hedge_ids = hyper_edge_index[1]
        # 计算当前子图中的超边数量。
        num_hyperedges = int(hedge_ids.max().item()) + 1
        # 取出所有连接对应的节点表示，mean 聚合时会直接用到。
        incident_node_embedding = node_embedding[node_ids]

        # attention 模式下，使用 PMA 风格聚合得到超边表示。
        if self.hyperedge_aggregation == 'attention':
            return self.attention_pool(node_embedding, hyper_edge_index)
        # mean 模式下，直接按超边对 incident 节点求均值。
        return _scatter_mean(incident_node_embedding, hedge_ids, num_hyperedges)

    def forward(self, node_features, hyper_edge_index):
        # 先用独立 HypergraphConv 对输入节点特征做一次编码。
        _numeric_debug_tensor(self, 'ViewLearner.forward.node_features_input', node_features)
        encoded_node_feat = self.encoder(node_features, hyper_edge_index)
        _numeric_debug_tensor(self, 'ViewLearner.forward.encoded_node_feat', encoded_node_feat)
        # 再按配置的聚合方式生成每条超边的表示。
        hedge_embedding = self._aggregate_hyperedge_embedding(encoded_node_feat, hyper_edge_index)
        _numeric_debug_tensor(self, 'ViewLearner.forward.hedge_embedding', hedge_embedding)

        # 重新取出连接级的节点索引与超边索引。
        node_ids = hyper_edge_index[0]
        hedge_ids = hyper_edge_index[1]
        # 对每条连接，拼接“该连接的节点表示”和“该连接所属超边表示”。
        total_emb = torch.cat(
            [encoded_node_feat[node_ids], hedge_embedding[hedge_ids]],
            dim=1
        )
        _numeric_debug_tensor(self, 'ViewLearner.forward.total_emb', total_emb)
        # 通过 MLP 输出每条连接的权重 logits，并展平成一维。
        logits = self.mlp_edge_model(total_emb).reshape(-1)
        _numeric_debug_tensor(self, 'ViewLearner.forward.logits', logits)
        return logits
