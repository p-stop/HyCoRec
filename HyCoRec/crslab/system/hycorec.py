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


        # case study: ID→实体名称映射
        self.entity2id = vocab.get('entity2id', {})
        self.id2entity = vocab.get('id2entity', {})
        # case study: 样本收集
        self._case_study_samples = []
        self._case_study_uid_counter = 0
        self._last_case_study_topo = []

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
        self.deitem_f = self.view_optim_opt.get('deitem_f', False)  # 是否在 ViewLearner 中去掉 item 图（调试用）
        self.deentity_f = self.view_optim_opt.get('deentity_f', False)  # 是否在 ViewLearner 中去掉 entity 图（调试用）
        self.deword_f = self.view_optim_opt.get('deword_f', False)  # 是否在 ViewLearner 中去掉 word 图（调试用）
        self.deitem_cf = self.view_optim_opt.get('deitem_cf', False)  # 是否在 ViewLearner 中去掉 item 图（调试用）
        self.deentity_cf = self.view_optim_opt.get('deentity_cf', False)  # 是否在 ViewLearner 中去掉 entity 图（调试用）
        self.deword_cf = self.view_optim_opt.get('deword_cf', False)  # 是否在 ViewLearner 中去掉 word 图（调试用）
        self.en_case_study = self.view_optim_opt.get('en_case_study', False)  # 是否启用 case study
        

        self.view_enhence = self.view_optim_opt.get('view_enhence', False)  # 是否启用 ViewLearner 增强（反事实训练）
        self.view_epoch = self.view_optim_opt['epoch']
        self.cf_mix_ratio = self.view_optim_opt.get('cf_mix_ratio', 0.5)  # 反事实样本混入比例
        self.keep_ratio = self.view_optim_opt.get('ratio', 0.6)
        logger.info(f"keep_ratio: {self.keep_ratio}")
        self.f_mode = self.view_optim_opt.get('f_mode', 2)  # factual loss 版本选择
        self.cf_mode = self.view_optim_opt.get('cf_mode', 3)  # counterfactual loss 版本选择
        self.kg_emb_dim = opt.get('kg_emb_dim', 128)
        self.view_lr = self.view_optim_opt.get('view_lr', 0.01)       # CACHE 默认 1e-2
        self.view_wd = self.view_optim_opt.get('view_wd', 0.001)      # CACHE 默认 1e-3
        self.view_alpha = self.view_optim_opt.get('view_alpha', 0.5)  # factual vs counterfactual 权重
        self.model_lambda = self.view_optim_opt.get('model_lambda', 0.1)  # 主模型损失中的对比损失权重
        self.gamma = self.view_optim_opt.get('gamma', 0.5)            # hinge loss margin
        self.use_counterfactual = self.view_optim_opt.get('use_counterfactual', True)
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


        self.nan_debug = self.view_optim_opt.get('debug_nan', False)
        self.nan_debug_raise = self.view_optim_opt.get('debug_nan_raise', False)
        self.nan_debug_anomaly = self.view_optim_opt.get('debug_nan_anomaly', False)
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
        
    def _core_model(self):
        """Return the underlying model for both plain and DataParallel wrappers."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model
    def _set_requires_grad(self, module: nn.Module, requires_grad: bool) -> None:
        """Enable/disable gradients for all parameters in a module."""
        for p in module.parameters():
            p.requires_grad_(requires_grad)

    def _view_learner_modules(self):
        core_model = self._core_model()
        return (
            core_model.view_learner_item,
            core_model.view_learner_entity,
            core_model.view_learner_word
        )

    def _unique_view_learner_modules(self):
        modules = []
        seen = set()
        for module in self._view_learner_modules():
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            modules.append(module)
        return modules

    def _view_learner_parameters(self):
        params = []
        seen = set()
        for module in self._unique_view_learner_modules():
            for param in module.parameters():
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        return params

    def _set_view_learners_mode(self, training: bool):
        for module in self._unique_view_learner_modules():
            module.train(training)

    def _set_view_learner_requires_grad(self, requires_grad: bool):
        for param in self._view_learner_parameters():
            param.requires_grad_(requires_grad)

    def _sync_view_hparams_to_model(self):
        core_model = self._core_model()
        if hasattr(core_model, 'keep_ratio'):
            core_model.keep_ratio = self.keep_ratio

    def _debug_check_weight_lists(self, name, batch_weights):
        """check if any weight is nan/inf"""
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

    def _prepare_recommendation_batch(self, batch):
        # 将“RGCN 编码 + 当前 batch 子图提取”委托给模型层统一完成。
        # 这样系统层只关心训练顺序，不再直接操作图构建细节。
        return self._core_model().prepare_recommendation_batch(batch)

    def rec_evaluate(self, rec_predict, item_label, type=""):
        rec_predict = rec_predict.cpu()
        # rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        # start = perf_counter()
        for rec_rank, label in zip(rec_ranks, item_label):
            # label = self.item_ids.index(label)
            self.evaluator.rec_evaluate(rec_rank, label, type=type)
        # print(f"{perf_counter() - start}")

    def _rec_conditional_report(self, scores_orig, scores_f, scores_cf, ground_truth, k=50):
        """条件召回指标：原始模型命中时，事实/反事实视图的命中率。

        回答："主模型已经能推荐对的样本中，f 和 cf 各自还能不能对？"
        Store results in evaluator.optim_metrics.
        """
        if ground_truth.numel() == 0:
            return

        # top-k indices per view  — 所有 scores 已 slice 到 item 列，排名对应局部索引
        _, ranks_orig = torch.topk(scores_orig, k, dim=-1)   # [B, k]
        _, ranks_f    = torch.topk(scores_f,    k, dim=-1)
        _, ranks_cf   = torch.topk(scores_cf,   k, dim=-1)

        # ground_truth 已经是 recommend_from_prepared_batch 返回的局部索引
        gt_idx = ground_truth  # [B], dtype long, values in [0, len(item_ids))

        # per-sample hit masks
        hit_orig = (ranks_orig == gt_idx.unsqueeze(1)).any(dim=1)  # [B]
        hit_f    = (ranks_f    == gt_idx.unsqueeze(1)).any(dim=1)
        hit_cf   = (ranks_cf   == gt_idx.unsqueeze(1)).any(dim=1)

        n_orig = int(hit_orig.sum().item())
        n_total = hit_orig.numel()

        if n_orig > 0:
            n_f_given_orig  = int((hit_orig & hit_f).sum().item())
            n_cf_given_orig = int((hit_orig & hit_cf).sum().item())
            self.evaluator.optim_metrics.add(
                "cond_f_recall@50", AverageMetric(n_f_given_orig, n_orig))
            self.evaluator.optim_metrics.add(
                "cond_cf_recall@50", AverageMetric(n_cf_given_orig, n_orig))

        # 原始模型自身的召回支持率（sanity check: recall@50 本身）
        self.evaluator.optim_metrics.add(
            "cond_orig_support@50", AverageMetric(n_orig, n_total))

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

    def rec_eval_with_weight(self, batch,batch_idx=0):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        core_model = self._core_model()

        prepared_batch = self._prepare_recommendation_batch(batch)

        (
            rec_loss, scores_orig, ground_truth, origin_rank,
            _, scores_f, f_rank,
            _, scores_cf, cf_rank
        ) = core_model.recommend_from_prepared_batch(
            prepared_batch,
            build_hyperedge_weights=True
        )

        loss_f = self.factual_loss(scores_orig.detach(), scores_f)
        loss_cf = self.counterfactual_loss(scores_orig.detach(), scores_cf)

        self.rec_evaluate(scores_orig, ground_truth)
        self.rec_evaluate(scores_f, ground_truth, type="f")
        self.rec_evaluate(scores_cf, ground_truth, type="cf")
        self._rec_conditional_report(scores_orig, scores_f, scores_cf, ground_truth, k=50)

        rec_loss = rec_loss.sum().item()
    
        self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        self.evaluator.optim_metrics.add("loss_f", AverageMetric(loss_f.item()))
        self.evaluator.optim_metrics.add("loss_cf", AverageMetric(loss_cf.item()))
        # ---- case study: 收集并保存本batch的数据 ----
        if self.en_case_study:
            self._last_case_study_topo = getattr(core_model, '_last_case_study_topo', [])
            self._collect_case_study_batch(prepared_batch, origin_rank, f_rank, cf_rank)




    def train_recommender(self):
        """推荐模块训练（交替训练 ViewLearner 和主模型）"""
        # initialize optimalizers
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        view_params = self._view_learner_parameters()
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

        # 交叉训练阶段：交替训练 ViewLearner 和主模型
        logger.info('[Starting alternating training]')
        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            
            if self.en_case_study and epoch != 11 and epoch != 12:
                self._case_study_samples = []
                self._case_study_uid_counter = 0
            
            logger.info('[Train]')
            for batch_idx, batch in enumerate(self.train_dataloader.get_rec_data(self.rec_batch_size)):
                # 数据移到 GPU
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                if self.use_counterfactual:
                    # ========== STEP 1: 训练 ViewLearner ==========
                    self.train_view_learner_step(batch,epoch)
                    
                    # ========== STEP 2: 训练主模型 ==========
                    self.train_main_model_step(batch,epoch)
                else:
                    # 不使用反事实推理，使用原始训练
                    self.step(batch, stage='rec', mode='train')
            
            train_report = self.evaluator.report(epoch=epoch, mode='train')
            self.log_wandb_metrics(train_report, stage='rec', mode='train', epoch=epoch)
            # valid
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    if self.use_counterfactual:
                        self.rec_eval_with_weight(batch,epoch)
                    else:
                        self.step(batch, stage='rec', mode='valid')
                valid_report = self.evaluator.report(epoch=epoch, mode='valid')
                self.log_wandb_metrics(valid_report, stage='rec', mode='valid', epoch=epoch)
                # early stop
                metric = valid_report.get(
                    self.rec_early_stop_metric,
                )
                if self.early_stop(metric):
                    break
            # test
            logger.info('[Test]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    if self.use_counterfactual:
                        self.rec_eval_with_weight(batch,epoch)
                    else:
                        self.step(batch, stage='rec', mode='test')
                test_report = self.evaluator.report(mode='test')
                self.log_wandb_metrics(test_report, stage='rec', mode='test')

            ## case-study
            if self.en_case_study and (epoch == 11 or epoch == 12):
                self._save_id_mappings()
                self._save_case_study()

        if self.use_counterfactual and self.view_enhence:
            logger.info('[Starting view training]')
            for epoch in range(self.view_epoch):
                self.evaluator.reset_metrics()
                self.keep_ratio = max(self.keep_ratio * 0.95, 0.7)  # 每个 epoch 衰减 keep_ratio，保持在 [0.1, 1.0] 范围内
                logger.info(f'[view epoch {str(epoch)}，ratio={self.keep_ratio}]')
                
                logger.info('[Train]')
                for batch_idx, batch in enumerate(self.train_dataloader.get_rec_data(self.rec_batch_size)):
                    # 数据移到 GPU
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)

                    self.train_view_learner_step(batch,batch_idx)
                train_report = self.evaluator.report(epoch=epoch, mode='train')
                self.log_wandb_metrics(train_report, stage='rec', mode='train', epoch=epoch)
                
                # valid
                logger.info('[Valid]')
                with torch.no_grad():
                    self.evaluator.reset_metrics()
                    for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                        self.rec_eval_with_weight(batch,epoch)
                    valid_report = self.evaluator.report(epoch=epoch, mode='valid')
                    self.log_wandb_metrics(valid_report, stage='rec', mode='valid', epoch=epoch)        
            logger.info('[Test]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.rec_eval_with_weight(batch)
                test_report = self.evaluator.report(mode='test')
                self.log_wandb_metrics(test_report, stage='rec', mode='test')

    def train_view_learner_step(self, batch,batch_idx=0):
        """
        训练 ViewLearner（冻结主模型）
        
        仿照 CACHE/train.py 的 STEP ONE:
        1. 主模型 eval 模式，获取原始预测
        2. ViewLearner 生成边权重
        3. 用边权重做事实预测和反事实预测
        4. 计算 view_loss 并更新 ViewLearner
        """
        # 训练 ViewLearner 时冻结主模型，只让权重学习器更新。
        self.model.eval()
        self._set_requires_grad(self.model, False)
        self._set_view_learners_mode(True)
        self._set_view_learner_requires_grad(True)
        # 取到底层模型，避免 DataParallel 包装影响自定义方法调用。
        core_model = self._core_model()

        # 针对view_enhence模式，确保模型内部的 keep_ratio 与系统层同步
        self._sync_view_hparams_to_model()

        # 先完成当前 batch 的 GCN 编码和子图准备，这是后续所有视图的共同输入。
        prepared_batch = self._prepare_recommendation_batch(batch)

        # 一次前向同时得到原图、事实图和反事实图的分数；权重由 model 内部逐层生成。
        (
            _, scores_orig, _, _,
            _, scores_f, _,
            _, scores_cf, _
        ) = core_model.recommend_from_prepared_batch(
            prepared_batch,
            build_hyperedge_weights=True,
            view_grad=True  # 允许 ViewLearner 梯度传播
        )

        # 5. 计算事实损失和反事实损失
        # 原始分数只作为 teacher logits，不参与梯度传播。
        scores_orig = scores_orig.detach()
        # factual 分数应尽量贴近原始分数的偏好方向。
        loss_f = self.factual_loss(scores_orig, scores_f)
        # counterfactual 目标：尽量与原始预测相反。
        loss_cf = self.counterfactual_loss(scores_orig, scores_cf)

        # 7. view_loss = α * loss_f + (1-α) * loss_cf
        # factual/counterfactual 损失控制视图质量，正则项控制不要过度删边。
        # aug_weight_mean=0
        view_loss = (self.view_alpha * loss_f + 
                     (1 - self.view_alpha) * loss_cf)
        # 8. 更新 ViewLearner
        # 先清空 ViewLearner 梯度。
        self.view_optimizer.zero_grad()
        # 只对 ViewLearner 相关参数反向传播。
        view_loss.backward()
        # 做梯度裁剪，避免权重学习过程不稳定。
        view_grad_norm = torch.nn.utils.clip_grad_norm_(self._view_learner_parameters(), 1.0)
        # 执行一次优化更新。
        self.view_optimizer.step()
        # 记录指标
        self.evaluator.optim_metrics.add("view_loss", AverageMetric(view_loss.item()))
        self.evaluator.optim_metrics.add("loss_f", AverageMetric(loss_f.item()))
        self.evaluator.optim_metrics.add("loss_cf", AverageMetric(loss_cf.item()))

        self._set_requires_grad(self.model, True)

    def train_main_model_step(self, batch,batch_idx):
        """
        训练主模型（ViewLearner 在 eval 模式）
        
        仿照 CACHE/train.py 的 STEP TWO:
        1. 主模型 train 模式
        2. ViewLearner eval 模式，生成边权重（不更新）
        3. 计算原始损失 + 对比损失
        4. 更新主模型
        """
        self.model.train()
        self._set_view_learners_mode(False)
        self._set_view_learner_requires_grad(False)
        # 主模型更新阶段仍然复用同一批预处理后的图数据。
        core_model = self._core_model()
        self._sync_view_hparams_to_model()
        # 先做 batch 级图准备，再进入权重预测和 HGCN。
        prepared_batch = self._prepare_recommendation_batch(batch)
        self._debug_check_prepared_batch('train_main_model_step.prepared_batch', prepared_batch)

        # 一次前向同时得到原图、事实图和反事实图三套 loss/score/rank。
        (
            rec_loss, scores_orig, targets, origin_rank,
            _, scores_f, f_rank,
            _, scores_cf, cf_rank
        ) = core_model.recommend_from_prepared_batch(
            prepared_batch,
            build_hyperedge_weights=True
        )
        self._debug_check_tensor('train_main_model_step.rec_loss', rec_loss)
        self._debug_check_tensor('train_main_model_step.scores_orig', scores_orig)
        self._debug_check_tensor('train_main_model_step.targets', targets)
        self._debug_check_tensor('train_main_model_step.scores_f', scores_f)
        self._debug_check_tensor('train_main_model_step.scores_cf', scores_cf)

        # ---- case study: 收集并保存本batch的数据 ----
        if self.en_case_study:
            self._last_case_study_topo = getattr(core_model, '_last_case_study_topo', [])
            self._collect_case_study_batch(prepared_batch, origin_rank, f_rank, cf_rank)

        # 5. 计算事实损失和反事实损失
        # 原始分数只作为 teacher logits，不参与梯度传播。
        scores_orig = scores_orig.detach()
        # factual 分数应尽量贴近原始分数的偏好方向。
        loss_f = self.factual_loss(scores_orig, scores_f)
        self._debug_check_tensor('train_view_learner_step.loss_f', loss_f)
        # counterfactual 目标：尽量与原始预测相反。
        loss_cf = self.counterfactual_loss(scores_orig, scores_cf)
        self._debug_check_tensor('train_view_learner_step.loss_cf', loss_cf)

        # 5. model_loss = rec_loss + λ_model * (α * loss_f + (1-α) * loss_cf)
        # 主任务交叉熵仍是核心，视图对比损失作为辅助约束。
        view_loss = self.view_alpha*loss_f + (1 - self.view_alpha)*loss_cf
        # view_loss = loss_f
        model_loss = rec_loss.sum() + self.model_lambda * view_loss
        self._debug_check_tensor('train_main_model_step.model_loss', model_loss)
        
        # 6. 更新主模型
        # 使用系统已有的 backward 逻辑更新主模型参数。
        self.backward(model_loss)
        self._set_view_learner_requires_grad(True)

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
            # valid
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
        elif self.cf_mode == 3:
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
            # pair_loss = torch.clamp(
            #     self.gamma + pos_scores.unsqueeze(-1) - neg_scores.unsqueeze(1),
            #     min=0
            # )
            # return pair_loss.mean()
        else:
            B, N = scores_cf.shape
            if N <= 1:
                return scores_cf.sum() * 0.0
            
            # ---- 公共部分：原 top-k 在反事实视图中的得分 ----
            tk = min(max(int(self.loss_topk), 1), N - 1)
            neg_k = min(max(int(self.loss_neg_k), 1), N - tk)
            pos_idx, pos_w = self._get_teacher_topk(scores_orig, tk)
            pos_scores = scores_cf.gather(1, pos_idx)               # (B, tk)

            # ---- mode3 分支：pos vs best non-top ----
            mask = torch.zeros_like(scores_cf, dtype=torch.bool)
            mask.scatter_(1, pos_idx, True)
            neg_pool = scores_cf.masked_fill(mask, -1e9)
            neg_scores_best = torch.topk(neg_pool, neg_k, dim=1).values  # (B, neg_k)

            loss_mode3 = F.softplus(
                self.gamma + pos_scores.unsqueeze(-1) - neg_scores_best.unsqueeze(1)
            )                                                       # (B, tk, neg_k)
            loss_mode3 = (loss_mode3 * pos_w.unsqueeze(-1)).mean()

            # ---- mode2 分支：pos vs bottom-k ----
            bottom_k = min(tk, N // 2)
            bottom_idx = torch.topk(scores_orig.detach(), bottom_k, dim=1, largest=False).indices
            bottom_scores = scores_cf.gather(1, bottom_idx)         # (B, bottom_k)

            loss_mode2 = torch.clamp(
                self.gamma + pos_scores.unsqueeze(-1) - bottom_scores.unsqueeze(1),
                min=0
            ).mean()

            # ---- 加权融合 ----
            return self.cf_mix_ratio * loss_mode2 + (1 - self.cf_mix_ratio) * loss_mode3
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

    def _log_weight_distribution(self, weight_info, tag="factual"):
        """输出超边权重的极端性分布，用于诊断是否趋近 0/1 还是停在 0.5。
        
        Args:
            weight_info: dict, {'item': tensor, 'entity': tensor, 'word': tensor}
                        来自 _build_batch_hyperedge_weights 的返回值。
            tag: str, 标签，用于区分 factual / counterfactual。
        """
        all_w = torch.cat([w for w in weight_info.values() if w.numel() > 0], dim=0)
        if all_w.numel() == 0:
            logger.info(f"[DISTRIB] {tag}: empty weights")
            return

        # --- 基础统计 ---
        w_min  = all_w.min().item()
        w_max  = all_w.max().item()
        w_mean = all_w.mean().item()
        w_std  = all_w.std().item()
        w_med  = all_w.median().item()

        # --- 各区间占比 ---
        bins = [
            ("0.00-0.10",   (0.0,  0.1)),
            ("0.10-0.25",   (0.1,  0.25)),
            ("0.25-0.40",   (0.25, 0.4)),
            ("0.40-0.50",   (0.4,  0.5)),
            ("0.50-0.60",   (0.5,  0.6)),
            ("0.60-0.75",   (0.6,  0.75)),
            ("0.75-0.90",   (0.75, 0.9)),
            ("0.90-1.00",   (0.9,  1.0)),
        ]
        total = all_w.numel()
        bin_strs = []
        for label, (lo, hi) in bins:
            cnt = int(((all_w >= lo) & (all_w < hi + 1e-9)).sum().item())
            pct = cnt / total * 100
            bin_strs.append(f"{label}:{cnt:>6d}({pct:5.1f}%)")

        # --- 极端性指标 ---
        eps = 1e-4
        near_0 = int((all_w <= eps).sum().item())          # 几乎为 0
        near_1 = int((all_w >= 1.0 - eps).sum().item())    # 几乎为 1
        extreme = near_0 + near_1
        # 模糊区间：[0.3, 0.7] 内的权重缺乏明确偏好
        mid_count = int(((all_w >= 0.3) & (all_w <= 0.7)).sum().item())
        
        # 极化率：越靠近 0 或 1 的比例越高越好
        polarization = extreme / total * 100
        ambig_rate = mid_count / total * 100

        # --- 熵（越低越极化）---
        eps_ent = 1e-8
        H_per_weight = -(all_w * torch.log(all_w + eps_ent) + (1 - all_w) * torch.log(1 - all_w + eps_ent))
        H_mean = H_per_weight.mean().item()
        # H_max = ln(2) ≈ 0.693 在 w=0.5 处。越小越极化。

        # --- 拼接输出 ---
        bin_line = "  ".join(bin_strs)
        summary = (
            f"[DISTRIB] {tag}"
            f" | N={total} min={w_min:.4f} max={w_max:.4f} mean={w_mean:.4f} std={w_std:.4f} med={w_med:.4f}"
            f" | near-0={near_0}({near_0/total*100:.1f}%) near-1={near_1}({near_1/total*100:.1f}%)"
            f" | polarization={polarization:.1f}% ambig=[0.3,0.7]={ambig_rate:.1f}%"
            f" | H={H_mean:.4f} (max=0.693 at w=0.5)"
            f"\n  {bin_line}"
        )
        logger.info(summary)

    def _collect_case_study_batch(self, prepared_batch, origin_rank, f_rank, cf_rank):
        """
        将当前batch的case study数据收集到 self._case_study_samples 中。
        """
        topo_list = getattr(self, '_last_case_study_topo', [])
        ground_truth = prepared_batch['item']
        origin_rank_cpu = origin_rank.detach().cpu().tolist()
        f_rank_cpu = f_rank.detach().cpu().tolist()
        cf_rank_cpu = cf_rank.detach().cpu().tolist()
        gt_cpu = ground_truth.cpu().tolist()

        for batch_idx in range(len(topo_list)):
            sample_entry = {
                "uid": self._case_study_uid_counter,
                "item": topo_list[batch_idx].get('item', {}).get('nodes', []),
                "item_weight": topo_list[batch_idx].get('item', {}).get('weights', []),
                "entity": topo_list[batch_idx].get('entity', {}).get('nodes', []),
                "entity_weight": topo_list[batch_idx].get('entity', {}).get('weights', []),
                "word": topo_list[batch_idx].get('word', {}).get('nodes', []),
                "word_weight": topo_list[batch_idx].get('word', {}).get('weights', []),
                "ground_truth": gt_cpu[batch_idx] if batch_idx < len(gt_cpu) else -1,
                "origin_rank": origin_rank_cpu[batch_idx] if batch_idx < len(origin_rank_cpu) else -1,
                "f_rank": f_rank_cpu[batch_idx] if batch_idx < len(f_rank_cpu) else -1,
                "cf_rank": cf_rank_cpu[batch_idx] if batch_idx < len(cf_rank_cpu) else -1,
            }
            self._case_study_samples.append(sample_entry)
            self._case_study_uid_counter += 1

    def _save_case_study(self, save_dir=None, file_prefix="case_study"):
        """将收集到的case study数据保存为JSON文件，同时生成实体名称版本。"""
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "case_study")
        os.makedirs(save_dir, exist_ok=True)

        if len(self._case_study_samples) == 0:
            logger.warning("[Case Study] No samples to save.")
            return

        id_path = os.path.join(save_dir, f"{file_prefix}_ids.json")
        with open(id_path, 'w', encoding='utf-8') as f:
            json.dump(self._case_study_samples, f, indent=2, ensure_ascii=False)
        logger.info(f"[Case Study] Saved {len(self._case_study_samples)} samples (IDs) to {id_path}")

        if hasattr(self, 'id2entity') and self.id2entity:
            named_samples = self._convert_ids_to_names(self._case_study_samples)
            name_path = os.path.join(save_dir, f"{file_prefix}_names.json")
            with open(name_path, 'w', encoding='utf-8') as f:
                json.dump(named_samples, f, indent=2, ensure_ascii=False)
            logger.info(f"[Case Study] Saved {len(named_samples)} samples (Names) to {name_path}")

    def _save_id_mappings(self, save_dir=None):
        """将entity2id和id2entity映射保存到本地。"""
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "case_study")
        os.makedirs(save_dir, exist_ok=True)

        core_model = self._core_model()
        if hasattr(core_model, 'entity2id'):
            entity2id_path = os.path.join(save_dir, "entity2id.json")
            with open(entity2id_path, 'w', encoding='utf-8') as f:
                json.dump(core_model.entity2id, f, indent=2, ensure_ascii=False)
            logger.info(f"[Case Study] Saved entity2id mapping to {entity2id_path}")

        if hasattr(self, 'id2entity') and self.id2entity:
            id2entity_path = os.path.join(save_dir, "id2entity.json")
            id2entity_str_keys = {str(k): v for k, v in self.id2entity.items()}
            with open(id2entity_path, 'w', encoding='utf-8') as f:
                json.dump(id2entity_str_keys, f, indent=2, ensure_ascii=False)
            logger.info(f"[Case Study] Saved id2entity mapping to {id2entity_path}")

    def _convert_ids_to_names(self, samples):
        """将case study样本中的全局entity ID转换为实体名称。"""
        if not hasattr(self, 'id2entity') or not self.id2entity:
            logger.warning("[Case Study] id2entity mapping not available.")
            return samples

        named_samples = []
        for sample in samples:
            named = {
                "uid": sample["uid"],
                "item": self._convert_graph_ids(sample.get("item", [])),
                "item_weight": sample.get("item_weight", []),
                "entity": self._convert_graph_ids(sample.get("entity", [])),
                "entity_weight": sample.get("entity_weight", []),
                "word": self._convert_graph_ids(sample.get("word", [])),
                "word_weight": sample.get("word_weight", []),
                "ground_truth": self._id_to_name(sample.get("ground_truth", -1)),
                "origin_rank": sample.get("origin_rank", -1),
                "f_rank": sample.get("f_rank", -1),
                "cf_rank": sample.get("cf_rank", -1),
            }
            named_samples.append(named)
        return named_samples

    def _convert_graph_ids(self, graph_data):
        """将超图拓扑中的全局ID列表转换为实体名称。"""
        if not graph_data:
            return []
        result = []
        for edge_nodes in graph_data:
            named_edge = [self._id_to_name(nid) for nid in edge_nodes]
            result.append(named_edge)
        return result

    def _id_to_name(self, entity_id):
        """将单个实体ID转换为名称。"""
        if entity_id is None or entity_id == -1:
            return None
        eid = int(entity_id)
        return self.id2entity.get(eid, str(eid))
