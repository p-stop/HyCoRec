# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

import os
import json
from time import perf_counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt
from crslab.model.crs.hycorec.hycorec import ViewLearner

def _dump_debug_tensors(scores_orig_norm, scores_f, scores_cf):
    """
    Dump 3 tensors to local files for inspection.
    Saved as .pt so you can reload with torch.load().
    """
    dump_dir =  "./debug_dumps"
    os.makedirs(dump_dir, exist_ok=True)

    # move to cpu to make files portable and smaller (no GPU tensors)
    torch.save(scores_orig_norm.detach().cpu(), os.path.join(dump_dir, "scores_orig_norm"))
    torch.save(scores_f.detach().cpu(),        os.path.join(dump_dir, "scores_f"))
    torch.save(scores_cf.detach().cpu(),       os.path.join(dump_dir, "scores_cf"))

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

class HyCoRecSystem(BaseSystem):
    """
    HyCoRec System with CACHE-style factual/counterfactual training.
    
    仿照 CACHE/train.py 实现交替训练机制：
    - STEP 1: 训练 ViewLearner（冻结主模型）
    - STEP 2: 训练主模型（ViewLearner 在 eval 模式）
    """

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False):
        super(HyCoRecSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug)

        self.ind2tok = vocab['ind2tok']
        self.end_token_idx = vocab['tok2ind']['__end__']
        self.item_ids = side_data['item_entity_ids']

        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.view_optim_opt = opt['view']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

        self.rec_early_stop_metric = self.rec_optim_opt.get('early_stop_metric', 'rec_loss')
        self.bef_loss = 100000.0
        
        # ViewLearner 超参数（从配置中读取，设置默认值）
        # 仿照 CACHE/train.py 的参数设置
        self.f_mode = self.view_optim_opt.get('f_mode', 2)  # factual loss 版本选择
        self.cf_mode = self.view_optim_opt.get('cf_mode', 3)  # counterfactual loss 版本选择
        self.kg_emb_dim = opt.get('kg_emb_dim', 128)
        self.view_hidden_dim = self.view_optim_opt.get('view_hidden_dim', 64)  # ViewLearner 隐藏层维度
        self.view_hyperedge_aggregation = self.view_optim_opt.get('hyperedge_aggregation', 'mean')
        self.view_lr = self.view_optim_opt.get('view_lr', 0.01)       # CACHE 默认 1e-2
        self.view_wd = self.view_optim_opt.get('view_wd', 0.001)      # CACHE 默认 1e-3
        self.view_alpha = self.view_optim_opt.get('view_alpha', 0.5)  # factual vs counterfactual 权重
        self.view_lambda = self.view_optim_opt.get('view_lambda', 5.0)  # 边权重正则化系数
        self.model_lambda = self.view_optim_opt.get('model_lambda', 0.1)  # 主模型损失中的对比损失权重
        self.gamma = self.view_optim_opt.get('gamma', 0.5)            # hinge loss margin
        self.temperature = self.view_optim_opt.get('temperature', 1.0)  # gumbel softmax 温度
        self.use_counterfactual = self.view_optim_opt.get('use_counterfactual', True)
        self.same_view = self.view_optim_opt.get('same_view', False)  # 是否让事实视图和反事实视图共享权重学习器（调试用）
        self.tem_decay = self.view_optim_opt.get('tem_decay', False)  # 是否启用温度衰减（调试用）
        self.loss_tau = self.view_optim_opt.get('loss_tau', self.view_optim_opt.get('tau', 1.0))  # KL 蒸馏温度
        self.loss_topk = self.view_optim_opt.get('loss_topk', self.view_optim_opt.get('topk', 50))  # 排名损失使用的 top-k
        logger.info(f"loss_topk: {self.loss_topk}")
        self.loss_neg_k = self.view_optim_opt.get('loss_neg_k', self.view_optim_opt.get('neg_k', 50))  # top-k 对比负样本数
        logger.info(f"loss_neg_k: {self.loss_neg_k}")
        self.weight_clip_eps = float(self.view_optim_opt.get('weight_clip_eps', 1e-4))
        self.degree_clip_eps = float(self.view_optim_opt.get('degree_clip_eps', 0.0))
        self.nan_dump_dir = self.view_optim_opt.get('nan_dump_dir', './debug_dumps/view_nan')
        if not (0.0 <= self.weight_clip_eps < 0.5):
            raise ValueError(f'view.weight_clip_eps must be in [0, 0.5), got {self.weight_clip_eps}')
        if self.degree_clip_eps < 0:
            raise ValueError(f'view.degree_clip_eps must be >= 0, got {self.degree_clip_eps}')


        self.nan_debug = self.view_optim_opt.get('debug_nan', True)
        self.nan_debug_raise = self.view_optim_opt.get('debug_nan_raise', True)
        self.nan_debug_anomaly = self.view_optim_opt.get('debug_nan_anomaly', True)
        if self.nan_debug:
            logger.warning(
                '[NUMERIC DEBUG enabled] training will stop at the first NaN/Inf; '
                'set view.debug_nan=false after locating the source.'
            )
        
        # 预训练模型加载配置
        # pretrain_model_path: 预训练模型路径，如果指定则跳过预训练直接加载
        # save_pretrain_model: 是否在预训练后保存模型
        self.pretrain_model_path = self.view_optim_opt.get('pretrain_model_path', None)
        self.save_pretrain_model = self.view_optim_opt.get('save_pretrain_model', True)
        self.pretrain_save_path = self.view_optim_opt.get('pretrain_save_path', './pretrain_models')
        self.pretrain_model_name = self.view_optim_opt.get('pretrain_model_name',None)
        
        # 构建 ViewLearner（为三种超图各一个）
        # 注意：ViewLearner 需要能直接处理节点特征和超边索引
        if(self.same_view):
            self.view_learner_item = ViewLearner(
                self.kg_emb_dim,
                self.view_hidden_dim,
                self.device,
                hyperedge_aggregation=self.view_hyperedge_aggregation
            ).to(self.device)
            self.view_learner_entity = self.view_learner_item  # 共享权重
            self.view_learner_word = self.view_learner_item    # 共享权
        else:
            self.view_learner_item = ViewLearner(
                self.kg_emb_dim,
                self.view_hidden_dim,
                self.device,
                hyperedge_aggregation=self.view_hyperedge_aggregation
            ).to(self.device)
            self.view_learner_entity = ViewLearner(
                self.kg_emb_dim,
                self.view_hidden_dim,
                self.device,
                hyperedge_aggregation=self.view_hyperedge_aggregation
            ).to(self.device)
            self.view_learner_word = ViewLearner(
                self.kg_emb_dim,
                self.view_hidden_dim,
                self.device,
                hyperedge_aggregation=self.view_hyperedge_aggregation
            ).to(self.device)
        self._configure_numeric_debug_modules()

    def _core_model(self):
        """Return the underlying model for both plain and DataParallel wrappers."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model
    def _set_requires_grad(self, module: nn.Module, requires_grad: bool) -> None:
        """Enable/disable gradients for all parameters in a module."""
        for p in module.parameters():
            p.requires_grad_(requires_grad)
    def _view_learners(self):###
        # 统一返回三类超图对应的 ViewLearner，便于后续循环处理。
        return {
            'item': self.view_learner_item,
            'entity': self.view_learner_entity,
            'word': self.view_learner_word
        }

    def _configure_numeric_debug_modules(self):
        # 将调试开关传给模型层和子模块，这样 NaN/Inf 能在更靠近源头的位置报出。
        modules = list(self._core_model().modules())
        for learner in self._view_learners().values():
            modules.extend(list(learner.modules()))
        for module in modules:
            module.nan_debug = self.nan_debug
            module.nan_debug_raise = self.nan_debug_raise
            object.__setattr__(module, 'degree_clip_eps', self.degree_clip_eps)

    def _clip_hyperedge_weight(self, weight, name='hyperedge_weight'):
        if weight is None or self.weight_clip_eps <= 0:
            return weight
        clipped = weight.clamp(min=self.weight_clip_eps, max=1.0 - self.weight_clip_eps)
        if self.nan_debug:
            raw_low = int((weight < self.weight_clip_eps).sum().item())
            raw_high = int((weight > (1.0 - self.weight_clip_eps)).sum().item())
            if raw_low > 0 or raw_high > 0:
                logger.warning(
                    f"[WEIGHT CLIP] {name} eps={self.weight_clip_eps} "
                    f"raw_low={raw_low} raw_high={raw_high} "
                    f"context={getattr(self, '_debug_context', {})}"
                )
        return clipped

    def _debug_monitor_degree_health(self, name, prepared_batch, batch_weights):
        if not self.nan_debug:
            return
        degree_zero_eps = max(self.weight_clip_eps * 0.5, 1e-12)
        for sample_idx, sample_graph in enumerate(prepared_batch['graphs']):
            for graph_key in ('item', 'entity', 'word'):
                weight_list = batch_weights.get(graph_key, None)
                if weight_list is None or sample_idx >= len(weight_list):
                    continue
                weight = weight_list[sample_idx]
                graph_data = sample_graph[graph_key]
                if weight is None or graph_data is None:
                    continue
                hyper_edge_index = graph_data['hyper_edge_index']
                if hyper_edge_index.numel() == 0:
                    continue

                node_ids = hyper_edge_index[0]
                hedge_ids = hyper_edge_index[1]
                node_degree = weight.new_zeros(graph_data['node_embedding'].size(0))
                node_degree.index_add_(0, node_ids, weight[hedge_ids])

                if node_degree.numel() == 0:
                    continue
                min_weight = float(weight.min().item())
                max_weight = float(weight.max().item())
                min_degree = float(node_degree.min().item())
                zero_degree_count = int((node_degree <= degree_zero_eps).sum().item())
                non_finite_degree = int((~torch.isfinite(node_degree)).sum().item())

                if zero_degree_count > 0 or non_finite_degree > 0:
                    logger.error(
                        f"[DEGREE DEBUG] {name}.{graph_key}[{sample_idx}] "
                        f"min_weight={min_weight:.6e} max_weight={max_weight:.6e} "
                        f"min_degree={min_degree:.6e} zero_degree_nodes={zero_degree_count}/{node_degree.numel()} "
                        f"non_finite_degree={non_finite_degree} "
                        f"context={getattr(self, '_debug_context', {})}"
                    )

    def _dump_view_backward_snapshot(self, reason, scores_orig, scores_f, scores_cf,
                                     loss_f, loss_cf, aug_weight_mean, view_loss,
                                     batch_weights_f, batch_weights_cf):
        if not self.nan_debug:
            return

        os.makedirs(self.nan_dump_dir, exist_ok=True)
        debug_ctx = getattr(self, '_debug_context', {})
        epoch = debug_ctx.get('epoch', 'na')
        batch_idx = debug_ctx.get('batch_idx', 'na')
        dump_name = f'view_backward_nan_e{epoch}_b{batch_idx}.pt'
        dump_path = os.path.join(self.nan_dump_dir, dump_name)

        payload = {
            'reason': str(reason),
            'context': debug_ctx,
            'weight_clip_eps': self.weight_clip_eps,
            'degree_clip_eps': self.degree_clip_eps,
            'view_alpha': self.view_alpha,
            'view_lambda': self.view_lambda,
            'loss_f': loss_f.detach().cpu(),
            'loss_cf': loss_cf.detach().cpu(),
            'aug_weight_mean': aug_weight_mean.detach().cpu(),
            'view_loss': view_loss.detach().cpu(),
            'scores_orig': scores_orig.detach().cpu(),
            'scores_f': scores_f.detach().cpu(),
            'scores_cf': scores_cf.detach().cpu(),
        }
        for graph_key in ('item', 'entity', 'word'):
            factual = [w.detach().cpu() for w in batch_weights_f[graph_key] if w is not None]
            counterfactual = [w.detach().cpu() for w in batch_weights_cf[graph_key] if w is not None]
            payload[f'batch_weights_f.{graph_key}'] = factual
            payload[f'batch_weights_cf.{graph_key}'] = counterfactual

        torch.save(payload, dump_path)
        logger.error(f'[VIEW BACKWARD SNAPSHOT] saved to {dump_path}')

    def _debug_check_weight_lists(self, name, batch_weights):
        if not self.nan_debug:
            return
        for graph_key, weights in batch_weights.items():
            for idx, weight in enumerate(weights):
                if weight is not None:
                    self._debug_check_tensor(f'{name}.{graph_key}[{idx}]', weight)

    def _debug_check_prepared_batch(self, name, prepared_batch):
        if not self.nan_debug:
            return
        self._debug_check_tensor(f'{name}.item', prepared_batch['item'])
        for emb_key, embedding in prepared_batch['kg_embeddings'].items():
            self._debug_check_tensor(f'{name}.kg_embeddings.{emb_key}', embedding)
        for sample_idx, sample_graph in enumerate(prepared_batch['graphs']):
            for graph_key in ('item', 'entity', 'word'):
                graph_data = sample_graph[graph_key]
                if graph_data is None:
                    continue
                self._debug_check_tensor(
                    f'{name}.graphs[{sample_idx}].{graph_key}.node_embedding',
                    graph_data['node_embedding']
                )
            if sample_graph['context_embedding'] is not None:
                self._debug_check_tensor(
                    f'{name}.graphs[{sample_idx}].context_embedding',
                    sample_graph['context_embedding']
                )

    def _debug_check_module_grads(self, name, modules):
        if not self.nan_debug:
            return
        for module_name, module in modules.items():
            for param_name, param in module.named_parameters():
                if param.grad is not None:
                    self._debug_check_tensor(f'{name}.{module_name}.grad.{param_name}', param.grad)

    def _prepare_recommendation_batch(self, batch):
        # 将“RGCN 编码 + 当前 batch 子图提取”委托给模型层统一完成。
        # 这样系统层只关心训练顺序，不再直接操作图构建细节。
        return self._core_model().prepare_recommendation_batch(batch)

    def _build_batch_hyperedge_weights(self, prepared_batch):
        # 拿到底层模型，后面要复用其“连接权重 -> 超边权重”的聚合逻辑。
        core_model = self._core_model()
        # batch_weights 保存每个样本、每种图的超边权重列表，直接供 HGCN 使用。
        batch_weights = {'item': [], 'entity': [], 'word': []}
        # flat_weight_info 把所有样本的权重拍平，便于做正则项统计。
        flat_weight_info = {'item': [], 'entity': [], 'word': []}

        # 逐样本处理，因为每个样本的子图大小都可能不同。
        for sample_graph in prepared_batch['graphs']:
            # item/entity/word 三种超图共享同一套权重生成流程。
            for graph_key, learner in self._view_learners().items():
                # 取出当前样本在某一类图上的子图数据。
                graph_data = sample_graph[graph_key]
                # 如果该样本没有这一类图，就占位为 None。
                if graph_data is None:
                    batch_weights[graph_key].append(None)
                    continue

                # ViewLearner 先基于“节点特征 + 关联关系”输出连接级 logits。
                weight_logits = learner(
                    graph_data['node_embedding'],
                    graph_data['hyper_edge_index']
                )
                # check(f"{graph_key}_weight_logits", weight_logits)
                # 用 gumbel-softmax 将 logits 变成可微的连接保留概率。
                connection_weight = gumbel_softmax(weight_logits, self.temperature,self.weight_clip_eps)
                # c_w = torch.sigmoid(weight_logits)
                # check(f"{graph_key}_connection_weight", connection_weight)
                # 保存当前样本当前图类型的超边权重，稍后直接喂给 HGCN。
                batch_weights[graph_key].append(connection_weight)
                # 同时把它展平收集起来，用于正则项统计。
                flat_weight_info[graph_key].append(connection_weight.reshape(-1))

        # 将拍平后的权重列表整理成张量字典。
        weight_info = {}
        for graph_key, weights in flat_weight_info.items():
            # 某一类图在整个 batch 中都不存在时，构造零张量避免下游 .mean() 报错。
            if len(weights) == 0:
                weight_info[graph_key] = torch.zeros(1, device=self.device)
            else:
                # 否则将所有样本的权重拼起来，供正则项统一统计。
                weight_info[graph_key] = torch.cat(weights, dim=0)

        return batch_weights, weight_info

    def _build_counterfactual_weights(self, batch_weights):
        # 反事实视图直接使用补权重：保留概率变为删除概率。
        counterfactual_weights = {}
        for graph_key, weights in batch_weights.items():
            graph_weights = []
            for weight in weights:
                # 没有该图时继续保持 None。
                if weight is None:
                    graph_weights.append(None)
                else:
                    graph_weights.append(1 - weight)
            counterfactual_weights[graph_key] = graph_weights
        return counterfactual_weights

    def rec_evaluate(self, rec_predict, item_label,type=""):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        # start = perf_counter()
        for rec_rank, label in zip(rec_ranks, item_label):
            label = self.item_ids.index(label)
            self.evaluator.rec_evaluate(rec_rank, label,type=type)
        # print(f"{perf_counter() - start}")

    def conv_evaluate(self, prediction, response, batch_user_id=None, batch_conv_id=None):
        prediction = prediction.tolist()
        response = response.tolist()
        if batch_user_id is None:
            for p, r in zip(prediction, response):
                p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
                r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
                self.evaluator.gen_evaluate(p_str, [r_str], p)
        else:
            for p, r, uid, cid in zip(prediction, response, batch_user_id, batch_conv_id):
                p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
                r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
                self.evaluator.gen_evaluate(p_str, [r_str], p)

    def step(self, batch, stage, mode):
        assert stage in ('rec', 'conv')
        assert mode in ('train', 'valid', 'test')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        if stage == 'rec':
            rec_loss, rec_scores = self.model.forward(batch, mode, stage)
            rec_loss = rec_loss.sum()
            if mode == 'train':
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_scores, batch['item'])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        else:
            if mode != 'test':
                gen_loss, preds = self.model.forward(batch, mode, stage)
                if mode == 'train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(preds, batch['response'])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add('gen_loss', AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                preds = self.model.forward(batch, mode, stage)
                self.conv_evaluate(preds, batch['response'], batch.get('user_id', None), batch['conv_id'])

    def rec_eval_with_weight(self, batch):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        core_model = self._core_model()

        prepared_batch = self._prepare_recommendation_batch(batch)

        rec_loss, scores_orig,ground_truth= core_model.recommend_from_prepared_batch(prepared_batch)

        batch_weights_f, weight_info = self._build_batch_hyperedge_weights(prepared_batch)

        rec_f_loss, scores_f , _= core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_f['item'],
            batch_entity_weights=batch_weights_f['entity'],
            batch_word_weights=batch_weights_f['word']
        )

        batch_weights_cf = self._build_counterfactual_weights(batch_weights_f)

        rec_cf_loss, scores_cf ,_= core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_cf['item'],
            batch_entity_weights=batch_weights_cf['entity'],
            batch_word_weights=batch_weights_cf['word']
        )

        loss_f = self.factual_loss(scores_orig, scores_f)
        loss_cf = self.counterfactual_loss(scores_orig, scores_cf)

        self.rec_evaluate(scores_orig, ground_truth)
        self.rec_evaluate(scores_f, ground_truth,type="f")
        self.rec_evaluate(scores_cf, ground_truth,type="cf")

        rec_loss = rec_loss.sum().item()

        self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        self.evaluator.optim_metrics.add("loss_f", AverageMetric(loss_f.item()))
        self.evaluator.optim_metrics.add("loss_cf", AverageMetric(loss_cf.item()))




    def train_recommender(self):
        """推荐模块训练（交替训练 ViewLearner 和主模型）"""
        # 初始化主模型优化器
        self.init_optim(self.rec_optim_opt, self.model.parameters())
        
        # 初始化 ViewLearner 优化器
        view_params = (
            list(self.view_learner_item.parameters()) + 
            list(self.view_learner_entity.parameters()) + 
            list(self.view_learner_word.parameters())
        )
        self.view_optimizer = torch.optim.Adam(view_params, lr=self.view_lr, weight_decay=self.view_wd)

 # ==================== 预训练阶段 ====================
        # 根据配置决定是预训练还是加载已有模型
        if self.pretrain_model_path and os.path.exists(self.pretrain_model_path):
            # 加载已有的预训练模型
            logger.info(f'[Loading pretrained model from {self.pretrain_model_path}]')
            self._load_pretrain_model(self.pretrain_model_path)
        else:
            # 执行预训练
            pretrain_epochs = self.view_optim_opt.get('pretrain_epochs', 0)
            logger.info(f'[Pretraining main model for {pretrain_epochs} epochs]')
            for epoch in range(pretrain_epochs):
                self.evaluator.reset_metrics()
                logger.info(f'[Pretrain epoch {epoch}]')
                for batch in self.train_dataloader.get_rec_data(self.rec_batch_size):
                    self.step(batch, stage='rec', mode='train')
                self.evaluator.report(epoch=epoch, mode='train')
            
            # 预训练后保存模型
            if self.save_pretrain_model:
                self._save_pretrain_model()
        
        # 重置 early_stop 状态，准备交叉训练
        self.best_metric = None

        # #v_train
        # self.evaluator.reset_metrics()
        # logger.info(f'[Pretrain view learner]')
        # for batch in self.train_dataloader.get_rec_data(self.rec_batch_size):
        #     self.train_view_learner_step(batch)
        # self.evaluator.report(epoch=epoch, mode='train')

        # 交叉训练阶段：交替训练 ViewLearner 和主模型
        logger.info('[Starting alternating training]')
        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            
            if self.tem_decay :
                # 每 50 个 epoch 衰减 view_lambda
                if (epoch + 1) % 2 == 0:
                    self.view_lambda *= 0.9
                    logger.info(f'[Decay view_lambda to {self.view_lambda}]')
            
            logger.info('[Train]')
            for batch_idx, batch in enumerate(self.train_dataloader.get_rec_data(self.rec_batch_size)):
                self._debug_context = {
                    'stage': 'rec',
                    'mode': 'train',
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                }
                # 数据移到 GPU
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                if self.use_counterfactual:
                    # ========== STEP 1: 训练 ViewLearner ==========
                    self.train_view_learner_step(batch)
                    
                    # ========== STEP 2: 训练主模型 ==========
                    self.train_main_model_step(batch)
                else:
                    # 不使用反事实推理，使用原始训练
                    self.step(batch, stage='rec', mode='train')
            
            # logger.info(f"[DEBUG] optim_metrics keys before report: {list(self.evaluator.optim_metrics.metrics.keys()) if hasattr(self.evaluator.optim_metrics,'metrics') else self.evaluator.optim_metrics}")
            # logger.info(f"[DEBUG] rec_metrics keys before report: {list(self.evaluator.rec_metrics.metrics.keys()) if hasattr(self.evaluator.rec_metrics,'metrics') else self.evaluator.rec_metrics}")
            # logger.info(f"[DEBUG] gen_metrics keys before report: {list(self.evaluator.gen_metrics.metrics.keys()) if hasattr(self.evaluator.gen_metrics,'metrics') else self.evaluator.gen_metrics}")
            train_report = self.evaluator.report(epoch=epoch, mode='train')
            self.log_wandb_metrics(train_report, stage='rec', mode='train', epoch=epoch)
            
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.rec_eval_with_weight(batch)
                    # self.step(batch, stage='rec', mode='valid')
                valid_report = self.evaluator.report(epoch=epoch, mode='valid')
                self.log_wandb_metrics(valid_report, stage='rec', mode='valid', epoch=epoch)
                # early stop
                metric = valid_report.get(
                    self.rec_early_stop_metric,
                )
                # now_loss = valid_report.get("view_loss", None)
                # if now_loss is None:
                #     # valid 阶段未统计 view_loss（例如只调用了 rec_eval_with_weight）
                #     # logger.warning('[Valid_lr] view_loss not found in valid_report; skip view_lr decay logic.')
                #     a = 0
                # else:
                #     delta_loss = self.bef_loss - now_loss
                #     if now_loss > 0.6:
                #         if (-self.bef_loss * 0.05) < delta_loss < (self.bef_loss * 0.05):
                #             self.view_lr *= 0.1
                #             self.view_wd *= 0.1
                #             logger.info(
                #                 f'[{epoch}] no significant improvement in view_loss. '
                #                 f'Decay view_lr to {self.view_lr}, view_wd to {self.view_wd}'
                #             )
                #     # 更新基准 loss，供下个 epoch 比较
                #     self.bef_loss = now_loss           
                if self.early_stop(metric):
                    break
        
            # test
            logger.info('[Test]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.rec_eval_with_weight(batch)
                    # self.step(batch, stage='rec', mode='test')
                test_report = self.evaluator.report(mode='test')
                self.log_wandb_metrics(test_report, stage='rec', mode='test')

                    # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.rec_eval_with_weight(batch)
                # self.step(batch, stage='rec', mode='test')
            test_report = self.evaluator.report(mode='test')
            self.log_wandb_metrics(test_report, stage='rec', mode='test')

    def train_view_learner_step(self, batch):
        """
        训练 ViewLearner（冻结主模型）
        
        仿照 CACHE/train.py 的 STEP ONE:
        1. 主模型 eval 模式，获取原始预测
        2. ViewLearner 生成边权重
        3. 用边权重做事实预测和反事实预测
        4. 计算 view_loss 并更新 ViewLearner
        """
        self.view_learner_item.train()
        self.view_learner_entity.train()
        self.view_learner_word.train()
        # 训练 ViewLearner 时冻结主模型，只让权重学习器更新。
        self.model.eval()
        self._set_requires_grad(self.model, False)
        # 取到底层模型，避免 DataParallel 包装影响自定义方法调用。
        core_model = self._core_model()
        # 先完成当前 batch 的 GCN 编码和子图准备，这是后续所有视图的共同输入。
        prepared_batch = self._prepare_recommendation_batch(batch)
        self._debug_check_prepared_batch('train_view_learner_step.prepared_batch', prepared_batch)
        # item：ground-truth ;RGCN后的kg_embding ; graph*4
        # 1. 原始预测（不带权重）
        with torch.no_grad():
            # 原始分数只作为目标参照，因此不保留梯度。
            _, scores_orig ,targets= core_model.recommend_from_prepared_batch(prepared_batch)
        self._debug_check_tensor('train_view_learner_step.scores_orig', scores_orig)
        self._debug_check_tensor('train_view_learner_step.targets', targets)

        # 2. 当前 batch 先完成 GCN 与子图提取，再交给 ViewLearner 生成超边权重
        # 这里得到的是 factual 视图对应的超边权重，以及用于正则化统计的权重信息。
        batch_weights_f, weight_info = self._build_batch_hyperedge_weights(prepared_batch)
        self._debug_check_weight_lists('train_view_learner_step.batch_weights_f', batch_weights_f)
        for graph_key, weight in weight_info.items():
            self._debug_check_tensor(f'train_view_learner_step.weight_info.{graph_key}', weight)
        # batch_weights_f:(item/entity/word:各个bathc对应图的weight的list)；weight_info:(item/entity/word:三个长tensor用于正则化)
        # 3. 事实预测（带学习到的超边权重）
        # 将 factual 超边权重送入 HGCN，得到事实视图下的推荐分数。
        _, scores_f ,_= core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_f['item'],
            batch_entity_weights=batch_weights_f['entity'],
            batch_word_weights=batch_weights_f['word']
        )
        self._debug_check_tensor('train_view_learner_step.scores_f', scores_f)

        # 4. 反事实预测（使用补权重）
        # factual 的补权重对应 counterfactual 视图。
        batch_weights_cf = self._build_counterfactual_weights(batch_weights_f)
        self._debug_check_weight_lists('train_view_learner_step.batch_weights_cf', batch_weights_cf)
        self._debug_monitor_degree_health('train_view_learner_step.batch_weights_f', prepared_batch, batch_weights_f)
        self._debug_monitor_degree_health('train_view_learner_step.batch_weights_cf', prepared_batch, batch_weights_cf)
        # 将反事实权重送入 HGCN，得到 counterfactual 分数。
        _, scores_cf ,_= core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_cf['item'],
            batch_entity_weights=batch_weights_cf['entity'],
            batch_word_weights=batch_weights_cf['word']
        )
        self._debug_check_tensor('train_view_learner_step.scores_cf', scores_cf)
        
        # 5. 计算事实损失和反事实损失
        # 原始分数只作为 teacher logits，不参与梯度传播。
        scores_orig = scores_orig.detach()
        self._debug_check_tensor('train_view_learner_step.scores_orig_norm', scores_orig)
        # factual 分数应尽量贴近原始分数的偏好方向。
        loss_f = self.factual_loss(scores_orig, scores_f)
        self._debug_check_tensor('train_view_learner_step.loss_f', loss_f)
        # counterfactual 目标：尽量与原始预测相反。
        loss_cf = self.counterfactual_loss(scores_orig, scores_cf)
        self._debug_check_tensor('train_view_learner_step.loss_cf', loss_cf)
        # 6. 计算边权重正则化（鼓励保留更多边）
        # 分别取三类图的超边权重。
        item_weight = weight_info['item']
        entity_weight = weight_info['entity']
        word_weight = weight_info['word']
        # 将三类图的平均保留权重相加，作为整体图稀疏度约束。
        aug_weight_mean = item_weight.mean() + entity_weight.mean() + word_weight.mean()
        if self.same_view:
            aug_weight_mean /= 3  # 如果三类图共享权重学习器，平均一下避免数值过大。
        self._debug_check_tensor('train_view_learner_step.aug_weight_mean', aug_weight_mean)

        # 7. view_loss = α * loss_f + (1-α) * loss_cf + λ * mean(aug_weight)
        # factual/counterfactual 损失控制视图质量，正则项控制不要过度删边。
        view_loss = (self.view_alpha * loss_f + 
                     (1 - self.view_alpha) * loss_cf + 
                     self.view_lambda * aug_weight_mean)
        self._debug_check_tensor('train_view_learner_step.view_loss', view_loss)
        
        if not torch.isfinite(view_loss):
            logger.error(
                f"[NaN/Inf] view_loss={view_loss.item()} "
                f"loss_f={loss_f.item()} loss_cf={loss_cf.item()} "
                f"aug_weight_mean={aug_weight_mean.item()} "
                f"view_alpha={self.view_alpha} view_lambda={self.view_lambda}"
            )
            # record something so train.report is not empty and you can see the failure rate
            self.evaluator.optim_metrics.add("view_loss_nan", AverageMetric(1.0))
            self._set_requires_grad(self.model, True)
            return
        else:
            self.evaluator.optim_metrics.add("view_loss_nan", AverageMetric(0.0))
        
        # 8. 更新 ViewLearner
        # 先清空 ViewLearner 梯度。
        self.view_optimizer.zero_grad()
        # 只对 ViewLearner 相关参数反向传播。
        try:
            with self._debug_anomaly_context():
                view_loss.backward()
            self.evaluator.optim_metrics.add("view_backward_nan", AverageMetric(0.0))
        except RuntimeError as error:
            logger.error(
                f"[VIEW BACKWARD FAILED] {error}; "
                f"context={getattr(self, '_debug_context', {})}"
            )
            self._dump_view_backward_snapshot(
                error,
                scores_orig,
                scores_f,
                scores_cf,
                loss_f,
                loss_cf,
                aug_weight_mean,
                view_loss,
                batch_weights_f,
                batch_weights_cf
            )
            self.evaluator.optim_metrics.add("view_backward_nan", AverageMetric(1.0))
            self._set_requires_grad(self.model, True)
            if self.nan_debug_raise:
                raise
            return
        self._debug_check_module_grads('train_view_learner_step.after_backward', self._view_learners())
        # 做梯度裁剪，避免权重学习过程不稳定。
        view_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.view_learner_item.parameters()) + 
            list(self.view_learner_entity.parameters()) + 
            list(self.view_learner_word.parameters()), 
            1.0
        )
        self._debug_check_tensor('train_view_learner_step.view_grad_norm_after_clip', view_grad_norm)
        # 执行一次优化更新。
        self.view_optimizer.step()
        for graph_key, learner in self._view_learners().items():
            for param_name, param in learner.named_parameters():
                self._debug_check_tensor(f'train_view_learner_step.after_step.{graph_key}.param.{param_name}', param)
        
        # 记录指标
        self.evaluator.optim_metrics.add("view_loss", AverageMetric(view_loss.item()))
        self.evaluator.optim_metrics.add("loss_f", AverageMetric(loss_f.item()))
        self.evaluator.optim_metrics.add("loss_cf", AverageMetric(loss_cf.item()))
        self.evaluator.optim_metrics.add("aug_weight_mean", AverageMetric(aug_weight_mean.item()))

        self._set_requires_grad(self.model, True)

    def train_main_model_step(self, batch):
        """
        训练主模型（ViewLearner 在 eval 模式）
        
        仿照 CACHE/train.py 的 STEP TWO:
        1. 主模型 train 模式
        2. ViewLearner eval 模式，生成边权重（不更新）
        3. 计算原始损失 + 对比损失
        4. 更新主模型
        """
        self.model.train()
        self.view_learner_item.eval()
        self.view_learner_entity.eval()
        self.view_learner_word.eval()
        # 主模型更新阶段仍然复用同一批预处理后的图数据。
        core_model = self._core_model()
        # 先做 batch 级图准备，再进入权重预测和 HGCN。
        prepared_batch = self._prepare_recommendation_batch(batch)
        self._debug_check_prepared_batch('train_main_model_step.prepared_batch', prepared_batch)

        # 1. 每个 batch 先完成 GCN 编码和当前 batch 子图提取
        # 原始视图不带权重，直接作为主任务基线损失。
        rec_loss, scores_orig,targets= core_model.recommend_from_prepared_batch(prepared_batch)
        self._debug_check_tensor('train_main_model_step.rec_loss', rec_loss)
        self._debug_check_tensor('train_main_model_step.scores_orig', scores_orig)
        self._debug_check_tensor('train_main_model_step.targets', targets)

        # 2. 再交给 ViewLearner 生成超边权重
        # 主模型训练时不更新 ViewLearner，因此这里用 no_grad。
        with torch.no_grad():
            batch_weights_f, _ = self._build_batch_hyperedge_weights(prepared_batch)
        self._debug_check_weight_lists('train_main_model_step.batch_weights_f', batch_weights_f)

        # 3. 选择是否带权重地执行 HGCN
        # factual 视图：使用学习到的超边权重。
        _, scores_f ,_= core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_f['item'],
            batch_entity_weights=batch_weights_f['entity'],
            batch_word_weights=batch_weights_f['word']
        )
        self._debug_check_tensor('train_main_model_step.scores_f', scores_f)

        # 反事实视图：使用 factual 权重的补集。
        batch_weights_cf = self._build_counterfactual_weights(batch_weights_f)
        self._debug_check_weight_lists('train_main_model_step.batch_weights_cf', batch_weights_cf)
        # 反事实 HGCN 前向，用于构造对比约束。
        _, scores_cf ,_= core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_cf['item'],
            batch_entity_weights=batch_weights_cf['entity'],
            batch_word_weights=batch_weights_cf['word']
        )
        self._debug_check_tensor('train_main_model_step.scores_cf', scores_cf)
        
        # 5. 计算事实损失和反事实损失
        # 原始分数只作为 teacher logits，不参与梯度传播。
        scores_orig = scores_orig.detach()
        self._debug_check_tensor('train_view_learner_step.scores_orig_norm', scores_orig)
        # factual 分数应尽量贴近原始分数的偏好方向。
        loss_f = self.factual_loss(scores_orig, scores_f)
        self._debug_check_tensor('train_view_learner_step.loss_f', loss_f)
        # counterfactual 目标：尽量与原始预测相反。
        loss_cf = self.counterfactual_loss(scores_orig, scores_cf)
        self._debug_check_tensor('train_view_learner_step.loss_cf', loss_cf)
        
        # 5. model_loss = rec_loss + λ_model * (α * loss_f + (1-α) * loss_cf)
        # 主任务交叉熵仍是核心，视图对比损失作为辅助约束。
        model_loss = rec_loss.sum() + self.model_lambda * (
            self.view_alpha * loss_f + (1 - self.view_alpha) * loss_cf
        )
        self._debug_check_tensor('train_main_model_step.model_loss', model_loss)
        
        # 6. 更新主模型
        # 使用系统已有的 backward 逻辑更新主模型参数。
        self.backward(model_loss)
        
        # 记录指标
        self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss.item()))
        self.evaluator.optim_metrics.add("model_loss", AverageMetric(model_loss.item()))
        self.evaluator.optim_metrics.add("main_loss_f", AverageMetric(loss_f.item()))    
        self.evaluator.optim_metrics.add("main_loss_cf", AverageMetric(loss_cf.item()))

    def train_conversation(self):
        self._core_model().freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size):
                self.step(batch, stage='conv', mode='train')
            train_report = self.evaluator.report(epoch=epoch, mode='train')
            self.log_wandb_metrics(train_report, stage='conv', mode='train', epoch=epoch)
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='valid')
                valid_report = self.evaluator.report(epoch=epoch, mode='valid')
                self.log_wandb_metrics(valid_report, stage='conv', mode='valid', epoch=epoch)
                # early stop
                metric = self.evaluator.optim_metrics['gen_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            test_report = self.evaluator.report(mode='test')
            self.log_wandb_metrics(test_report, stage='conv', mode='test')

    def fit(self):
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        pass

    def _get_teacher_topk(self, scores_orig, tk):
        #特判
        B, N = scores_orig.shape
        if N <= 1:
            empty_idx = torch.empty(B, 0, dtype=torch.long, device=scores_orig.device)
            empty_w = scores_orig.new_empty(B, 0)
            return empty_idx, empty_w
        tk = min(max(int(tk), 1), N - 1)

        topk_vals, topk_idx = torch.topk(scores_orig, tk, dim=1)
        topk_w = F.softmax(topk_vals, dim=1)
        return topk_idx, topk_w

    def factual_loss(self, scores_orig, scores_f):
        """
        事实损失：宏观上保证 factual 分布与原始分布一致。
        """
        if self.f_mode == 1:
        # 版本 1（当前启用）：KL(softmax(z_orig / tau) || softmax(z_f / tau))
            tau = max(float(self.loss_tau), 1e-8)
            teacher_prob = F.softmax(scores_orig / tau, dim=-1)
            factual_log_prob = F.log_softmax(scores_f / tau, dim=-1)
            return F.kl_div(factual_log_prob, teacher_prob, reduction='batchmean')

        # 版本 2（备用，已注释）：topk 仍是 topk
        # 特判
        elif self.f_mode == 2:
            B, N = scores_f.shape
            tk = min(max(int(self.loss_topk), 1), N - 1)
            neg_k = min(max(int(self.loss_neg_k), 1), N - tk)
            
            pos_idx, pos_w = self._get_teacher_topk(scores_orig, tk)
            pos_scores = scores_f.gather(1, pos_idx)
            
            mask = torch.zeros_like(scores_f, dtype=torch.bool)
            mask.scatter_(1, pos_idx, True)
            
            neg_pool = scores_f.masked_fill(mask, -1e9)
            neg_scores = torch.topk(neg_pool, neg_k, dim=1).values
            
            pair_loss = F.softplus(
                self.gamma - pos_scores.unsqueeze(-1) + neg_scores.unsqueeze(1)
            )
            return (pair_loss * pos_w.unsqueeze(-1)).mean()

        # 版本 3（备用，已注释）：topk > bottomk
        else:
            B, N = scores_f.shape
            if N <= 1:
                return scores_f.sum() * 0.0
            
            tk = min(max(int(self.loss_topk), 1), N // 2)
            top_idx = torch.topk(scores_orig.detach(), tk, dim=1).indices
            bottom_idx = torch.topk(scores_orig.detach(), tk, dim=1, largest=False).indices
            
            top_scores = scores_f.gather(1, top_idx)
            bottom_scores = scores_f.gather(1, bottom_idx)
            pair_loss = torch.clamp(
                self.gamma - top_scores.unsqueeze(-1) + bottom_scores.unsqueeze(1),
                min=0
            )
            return pair_loss.mean()

    def counterfactual_loss(self, scores_orig, scores_cf):
        """
        反事实损失：多分类场景下鼓励 counterfactual 结果与事实预测不同。

        注意：这里沿用 scores_orig 的旧参数名，但实际传入的是未 softmax 的原始 logits。
        """
        # # 版本 1（当前启用）：原 top1 被原 top(k+1) 反超
        # #特判
        if self.cf_mode == 1:
            B, N = scores_cf.shape
            if N <= 1:
                return scores_cf.sum() * 0.0
            tk = min(max(int(self.loss_topk), 1), N - 1)

            ranked_idx = torch.topk(scores_orig, tk + 1, dim=1).indices
            top1_idx = ranked_idx[:, :1]
            challenger_idx = ranked_idx[:, tk:tk + 1]

            top1_score = scores_cf.gather(1, top1_idx)
            challenger_score = scores_cf.gather(1, challenger_idx)
            return F.softplus(self.gamma + top1_score - challenger_score).mean()

        # 版本 2（备用，已注释）：topk < bottomk
        elif self.cf_mode == 2:
            B, N = scores_cf.shape
            if N <= 1:
                return scores_cf.sum() * 0.0
            
            tk = min(max(int(self.loss_topk), 1), N // 2)
            top_idx = torch.topk(scores_orig.detach(), tk, dim=1).indices
            bottom_idx = torch.topk(scores_orig.detach(), tk, dim=1, largest=False).indices
            
            top_scores = scores_cf.gather(1, top_idx)
            bottom_scores = scores_cf.gather(1, bottom_idx)
            pair_loss = torch.clamp(
                self.gamma + top_scores.unsqueeze(-1) - bottom_scores.unsqueeze(1),
                min=0
            )
            return pair_loss.mean()

        # 版本 3（备用，已注释）：topk 不是 topk
        # 特判
        else:
            B, N = scores_cf.shape
            tk = min(max(int(self.loss_topk), 1), N - 1)
            neg_k = min(max(int(self.loss_neg_k), 1), N - tk)
            
            pos_idx, pos_w = self._get_teacher_topk(scores_orig, tk)
            pos_scores = scores_cf.gather(1, pos_idx)
            
            mask = torch.zeros_like(scores_cf, dtype=torch.bool)
            mask.scatter_(1, pos_idx, True)
            
            neg_pool = scores_cf.masked_fill(mask, -1e9)
            neg_scores = torch.topk(neg_pool, neg_k, dim=1).values
            
            pair_loss = F.softplus(
                self.gamma + pos_scores.unsqueeze(-1) - neg_scores.unsqueeze(1)
            )
            return (pair_loss * pos_w.unsqueeze(-1)).mean()
    def _get_pretrain_model_filename(self):
        """Return the configured pretrain model filename."""
        if self.pretrain_model_name:
            if self.pretrain_model_name.endswith('.pth'):
                return self.pretrain_model_name
            return f'{self.pretrain_model_name}.pth'

        model_name = self.opt.get('model_name', 'hycorec')
        dataset_name = self.opt.get('dataset', 'unknown')
        return f'{model_name}_{dataset_name}_pretrain.pth'

    def _save_pretrain_model(self):
        """保存预训练后的主模型"""
        os.makedirs(self.pretrain_save_path, exist_ok=True)
        save_file = os.path.join(self.pretrain_save_path, self._get_pretrain_model_filename())

        if os.environ.get("CUDA_VISIBLE_DEVICES") == '-1':
            state_dict = self.model.state_dict()
        else:
            state_dict = self.model.module.state_dict()

        torch.save(state_dict, save_file)
        logger.info(f'[Pretrained model saved to {save_file}]')

    def _load_pretrain_model(self, load_path):
        """加载预训练的主模型"""
        state_dict = torch.load(load_path, map_location=self.device)
        
        if os.environ.get("CUDA_VISIBLE_DEVICES") == '-1':
            self.model.load_state_dict(state_dict)
        else:
            self.model.module.load_state_dict(state_dict)
        
        logger.info(f'[Pretrained model loaded from {load_path}]')
