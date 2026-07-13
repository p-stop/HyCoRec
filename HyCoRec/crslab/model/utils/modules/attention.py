# -*- coding: utf-8 -*-
# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import softmax

class SelfAttentionBatch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionBatch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        # h: (N, dim)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e, dim=0)  # (N)
        return torch.matmul(attention, h)  # (dim)


class SelfAttentionSeq(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionSeq, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask=None, return_logits=False):
        """
        For the padding tokens, its corresponding mask is True
        if mask==[1, 1, 1, ...]
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)  # (batch, seq_len, 1)
        if mask is not None:
            full_mask = -1e30 * mask.float()
            batch_mask = torch.sum((mask == False), -1).bool().float().unsqueeze(-1)  # for all padding one, the mask=0
            mask = full_mask * batch_mask
            e += mask.unsqueeze(-1)
        attention = F.softmax(e, dim=1)  # (batch, seq_len, 1)
        # (batch, dim)
        if return_logits:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1), attention.squeeze(-1)
        else:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1)

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