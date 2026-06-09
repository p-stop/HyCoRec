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