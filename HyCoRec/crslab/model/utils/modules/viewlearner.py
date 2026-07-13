from HyCoRec.crslab.model.utils.modules.attention import HyperedgeAttentionPooling
import torch
from torch import nn
from torch_geometric.nn import HypergraphConv

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
        encoded_node_feat = self.encoder(node_features, hyper_edge_index)
        # encoded_node_feat = node_features
        # 再按配置的聚合方式生成每条超边的表示。
        hedge_embedding = self._aggregate_hyperedge_embedding(encoded_node_feat, hyper_edge_index)

        # 重新取出连接级的节点索引与超边索引。
        node_ids = hyper_edge_index[0]
        hedge_ids = hyper_edge_index[1]
        # 对每条连接，拼接“该连接的节点表示”和“该连接所属超边表示”。
        total_emb = torch.cat(
            [encoded_node_feat[node_ids], hedge_embedding[hedge_ids]],
            dim=1
        )
        # 通过 MLP 输出每条连接的权重 logits，并展平成一维。
        logits = self.mlp_edge_model(total_emb).reshape(-1)
        return logits