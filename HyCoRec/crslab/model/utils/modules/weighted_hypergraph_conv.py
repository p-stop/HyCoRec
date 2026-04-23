from typing import Optional
import inspect
import os

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from loguru import logger

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax


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


class WeightedHypergraphConv(MessagePassing):
    r"""PyG-compatible HypergraphConv with optional incidence weights.

    When ``incidence_weight`` is provided, each node-hyperedge incidence is
    treated as a weighted entry in the incidence matrix instead of collapsing
    it to a single scalar per hyperedge.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = "node",
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        assert attention_mode in ["node", "edge"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(
                in_channels,
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(
                in_channels,
                out_channels,
                bias=False,
                weight_initializer="glorot",
            )

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    @disable_dynamic_shapes(required_args=["num_edges"])
    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_weight: Optional[Tensor] = None,
        hyperedge_attr: Optional[Tensor] = None,
        num_edges: Optional[int] = None,
        incidence_weight: Optional[Tensor] = None,
    ) -> Tensor:
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        else:
            hyperedge_weight = hyperedge_weight.view(-1).to(x.device, dtype=x.dtype)
            if hyperedge_weight.numel() != num_edges:
                raise ValueError(
                    "hyperedge_weight must have one value per hyperedge "
                    f"({num_edges}), got {hyperedge_weight.numel()}."
                )

        if incidence_weight is not None:
            incidence_weight = incidence_weight.view(-1).to(x.device, dtype=x.dtype)
            if incidence_weight.numel() != hyperedge_index.size(1):
                raise ValueError(
                    "incidence_weight must have one value per node-hyperedge "
                    f"incidence ({hyperedge_index.size(1)}), got "
                    f"{incidence_weight.numel()}."
                )

        _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.x_input', x)
        _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.hyperedge_weight_input', hyperedge_weight)
        _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.incidence_weight_input', incidence_weight)
        x = self.lin(x)
        _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.x_after_lin', x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == "node":
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if incidence_weight is None:
            # Fall back to the official HypergraphConv normalization:
            # D^{-1} H W B^{-1} H^T X Theta
            d_inv = scatter(
                hyperedge_weight[hyperedge_index[1]],
                hyperedge_index[0],
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.unweighted.d_before_inv', d_inv)
            d_inv = 1.0 / d_inv
            d_inv[d_inv == float("inf")] = 0
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.unweighted.d_inv', d_inv)

            b_inv = scatter(
                x.new_ones(hyperedge_index.size(1)),
                hyperedge_index[1],
                dim=0,
                dim_size=num_edges,
                reduce="sum",
            )
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.unweighted.b_before_inv', b_inv)
            b_inv = 1.0 / b_inv
            b_inv[b_inv == float("inf")] = 0
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.unweighted.b_inv', b_inv)

            out = self.propagate(
                hyperedge_index,
                x=x,
                norm=b_inv,
                alpha=alpha,
                incidence_weight=None,
                size=(num_nodes, num_edges),
            )
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.unweighted.after_node_to_edge', out)
            out = self.propagate(
                hyperedge_index.flip([0]),
                x=out,
                norm=d_inv,
                alpha=alpha,
                incidence_weight=None,
                size=(num_edges, num_nodes),
            )
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.unweighted.after_edge_to_node', out)
        else:
            node_ids = hyperedge_index[0]
            hyperedge_ids = hyperedge_index[1]

            # Weighted incidence case:
            # H is no longer binary, so node degree becomes sum_e W_e * H_{v,e}.
            d_inv = scatter(
                hyperedge_weight[hyperedge_ids] * incidence_weight,
                node_ids,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.weighted.d_before_inv', d_inv)
            d_inv = 1.0 / d_inv
            d_inv[d_inv == float("inf")] = 0
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.weighted.d_inv', d_inv)

            # Hyperedge degree becomes the weighted sum of its incident nodes.
            b_inv = scatter(
                incidence_weight,
                hyperedge_ids,
                dim=0,
                dim_size=num_edges,
                reduce="sum",
            )
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.weighted.b_before_inv', b_inv)
            b_inv = 1.0 / b_inv
            b_inv[b_inv == float("inf")] = 0
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.weighted.b_inv', b_inv)

            # First pass: node -> hyperedge using B^{-1} H^T.
            out = self.propagate(
                hyperedge_index,
                x=x,
                norm=b_inv,
                alpha=alpha,
                incidence_weight=incidence_weight,
                size=(num_nodes, num_edges),
            )
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.weighted.after_node_to_edge', out)
            # Second pass: hyperedge -> node using D^{-1} H W.
            out = self.propagate(
                hyperedge_index.flip([0]),
                x=out,
                norm=d_inv,
                alpha=alpha,
                incidence_weight=incidence_weight * hyperedge_weight[hyperedge_ids],
                size=(num_edges, num_nodes),
            )
            _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.weighted.after_edge_to_node', out)

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        _numeric_debug_tensor(self, 'WeightedHypergraphConv.forward.output', out)

        return out

    def message(
        self,
        x_j: Tensor,
        norm_i: Tensor,
        alpha: Optional[Tensor],
        incidence_weight: Optional[Tensor],
    ) -> Tensor:
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        if incidence_weight is not None:
            # incidence_weight here is the per-incidence scalar injected by forward().
            out = incidence_weight.view(-1, 1, 1) * out

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
            f"use_attention={self.use_attention})"
        )
