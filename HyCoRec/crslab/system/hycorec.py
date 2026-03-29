# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

import os
import json
from time import perf_counter
import torch
import torch.nn as nn
import pickle as pkl
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt
from crslab.model.crs.hycorec.hycorec import ViewLearner


def gumbel_softmax(logits, temperature=1.0):
    """Gumbel-Softmax trick for differentiable sampling"""
    bias = 0.0001
    eps = (bias - (1 - bias)) * torch.rand(logits.size(), device=logits.device) + (1 - bias)
    gate_inputs = torch.log(eps) - torch.log(1 - eps)
    gate_inputs = (gate_inputs + logits) / temperature
    return torch.sigmoid(gate_inputs).reshape(-1)


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
        
        # ViewLearner 超参数（从配置中读取，设置默认值）
        # 仿照 CACHE/train.py 的参数设置
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
        
        # 预训练模型加载配置
        # pretrain_model_path: 预训练模型路径，如果指定则跳过预训练直接加载
        # save_pretrain_model: 是否在预训练后保存模型
        self.pretrain_model_path = self.view_optim_opt.get('pretrain_model_path', None)
        self.save_pretrain_model = self.view_optim_opt.get('save_pretrain_model', True)
        self.pretrain_save_path = self.view_optim_opt.get('pretrain_save_path', './pretrain_models')
        self.pretrain_model_name = self.view_optim_opt.get('pretrain_model_name',None)
        
        # 构建 ViewLearner（为三种超图各一个）
        # 注意：ViewLearner 需要能直接处理节点特征和超边索引
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

    def _core_model(self):
        """Return the underlying model for both plain and DataParallel wrappers."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _view_learners(self):###
        # 统一返回三类超图对应的 ViewLearner，便于后续循环处理。
        return {
            'item': self.view_learner_item,
            'entity': self.view_learner_entity,
            'word': self.view_learner_word
        }

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
                # 用 gumbel-softmax 将 logits 变成可微的连接保留概率。
                connection_weight = gumbel_softmax(weight_logits, self.temperature)
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

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        # start = perf_counter()
        for rec_rank, label in zip(rec_ranks, item_label):
            label = self.item_ids.index(label)
            self.evaluator.rec_evaluate(rec_rank, label)
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
        
        # 交叉训练阶段：交替训练 ViewLearner 和主模型
        logger.info('[Starting alternating training]')
        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            # self.log_wandb_metrics({'view_lambda': self.view_lambda}, stage='rec', mode='train', epoch=epoch)
            
            # # 每 50 个 epoch 衰减 view_lambda
            # if (epoch + 1) % 50 == 0:
            #     self.view_lambda *= 0.5
            #     logger.info(f'[Decay view_lambda to {self.view_lambda}]')
            #     self.log_wandb_metrics({'view_lambda': self.view_lambda}, stage='rec', mode='train', epoch=epoch)
            
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size):
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
            
            train_report = self.evaluator.report(epoch=epoch, mode='train')
            self.log_wandb_metrics(train_report, stage='rec', mode='train', epoch=epoch)
            
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
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
                self.step(batch, stage='rec', mode='test')
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
        # 取到底层模型，避免 DataParallel 包装影响自定义方法调用。
        core_model = self._core_model()
        # 先完成当前 batch 的 GCN 编码和子图准备，这是后续所有视图的共同输入。
        prepared_batch = self._prepare_recommendation_batch(batch)
        # item：ground-truth ;RGCN后的kg_embding ; graph*4
        # 1. 原始预测（不带权重）
        with torch.no_grad():
            # 原始分数只作为目标参照，因此不保留梯度。
            _, scores_orig = core_model.recommend_from_prepared_batch(prepared_batch)

        # 2. 当前 batch 先完成 GCN 与子图提取，再交给 ViewLearner 生成超边权重
        # 这里得到的是 factual 视图对应的超边权重，以及用于正则化统计的权重信息。
        batch_weights_f, weight_info = self._build_batch_hyperedge_weights(prepared_batch)
        # batch_weights_f:(item/entity/word:各个bathc对应图的weight的list)；weight_info:(item/entity/word:三个长tensor用于正则化)
        # 3. 事实预测（带学习到的超边权重）
        # 将 factual 超边权重送入 HGCN，得到事实视图下的推荐分数。
        _, scores_f = core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_f['item'],
            batch_entity_weights=batch_weights_f['entity'],
            batch_word_weights=batch_weights_f['word']
        )

        # 4. 反事实预测（使用补权重）
        # factual 的补权重对应 counterfactual 视图。
        batch_weights_cf = self._build_counterfactual_weights(batch_weights_f)
        # 将反事实权重送入 HGCN，得到 counterfactual 分数。
        _, scores_cf = core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_cf['item'],
            batch_entity_weights=batch_weights_cf['entity'],
            batch_word_weights=batch_weights_cf['word']
        )
        
        # 5. 计算事实损失和反事实损失
        # 原始分数先 detach，再 sigmoid 成为用于判别方向的软标签。
        scores_orig_norm = torch.sigmoid(scores_orig.detach())
        # factual 目标：尽量保持与原始预测一致。
        loss_f = self.factual_loss(scores_orig_norm, scores_f)
        # counterfactual 目标：尽量与原始预测相反。
        loss_cf = self.counterfactual_loss(scores_orig_norm, scores_cf)
        
        # 6. 计算边权重正则化（鼓励保留更多边）
        # 分别取三类图的超边权重。
        item_weight = weight_info['item']
        entity_weight = weight_info['entity']
        word_weight = weight_info['word']
        # 将三类图的平均保留权重相加，作为整体图稀疏度约束。
        aug_weight_mean = item_weight.mean() + entity_weight.mean() + word_weight.mean()

        # 7. view_loss = α * loss_f + (1-α) * loss_cf + λ * mean(aug_weight)
        # factual/counterfactual 损失控制视图质量，正则项控制不要过度删边。
        view_loss = (self.view_alpha * loss_f + 
                     (1 - self.view_alpha) * loss_cf + 
                     self.view_lambda * aug_weight_mean)
        
        # 8. 更新 ViewLearner
        # 先清空 ViewLearner 梯度。
        self.view_optimizer.zero_grad()
        # 只对 ViewLearner 相关参数反向传播。
        view_loss.backward()
        # 做梯度裁剪，避免权重学习过程不稳定。
        torch.nn.utils.clip_grad_norm_(
            list(self.view_learner_item.parameters()) + 
            list(self.view_learner_entity.parameters()) + 
            list(self.view_learner_word.parameters()), 
            1.0
        )
        # 执行一次优化更新。
        self.view_optimizer.step()
        
        # 记录指标
        self.evaluator.optim_metrics.add("view_loss", AverageMetric(view_loss.item()))
        self.evaluator.optim_metrics.add("loss_f", AverageMetric(loss_f.item()))
        self.evaluator.optim_metrics.add("loss_cf", AverageMetric(loss_cf.item()))
        self.evaluator.optim_metrics.add("aug_weight_mean", AverageMetric(aug_weight_mean.item()))

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

        # 1. 每个 batch 先完成 GCN 编码和当前 batch 子图提取
        # 原始视图不带权重，直接作为主任务基线损失。
        rec_loss_orig, scores_orig = core_model.recommend_from_prepared_batch(prepared_batch)

        # 2. 再交给 ViewLearner 生成超边权重
        # 主模型训练时不更新 ViewLearner，因此这里用 no_grad。
        with torch.no_grad():
            batch_weights_f, _ = self._build_batch_hyperedge_weights(prepared_batch)

        # 3. 选择是否带权重地执行 HGCN
        # factual 视图：使用学习到的超边权重。
        _, scores_f = core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_f['item'],
            batch_entity_weights=batch_weights_f['entity'],
            batch_word_weights=batch_weights_f['word']
        )

        # 反事实视图：使用 factual 权重的补集。
        batch_weights_cf = self._build_counterfactual_weights(batch_weights_f)
        # 反事实 HGCN 前向，用于构造对比约束。
        _, scores_cf = core_model.recommend_from_prepared_batch(
            prepared_batch,
            batch_item_weights=batch_weights_cf['item'],
            batch_entity_weights=batch_weights_cf['entity'],
            batch_word_weights=batch_weights_cf['word']
        )
        
        # 4. 计算事实/反事实损失
        # 原始分数只作为方向标签，不参与梯度传播。
        scores_orig_norm = torch.sigmoid(scores_orig.detach())
        # factual 分数应尽量贴近原始分数的偏好方向。
        loss_f = self.factual_loss(scores_orig_norm, scores_f)
        # counterfactual 分数应尽量背离原始分数的偏好方向。
        loss_cf = self.counterfactual_loss(scores_orig_norm, scores_cf)
        
        # 5. model_loss = rec_loss + λ_model * (α * loss_f + (1-α) * loss_cf)
        # 主任务交叉熵仍是核心，视图对比损失作为辅助约束。
        model_loss = rec_loss_orig + self.model_lambda * (
            self.view_alpha * loss_f + (1 - self.view_alpha) * loss_cf
        )
        
        # 6. 更新主模型
        # 使用系统已有的 backward 逻辑更新主模型参数。
        self.backward(model_loss)
        
        # 记录指标
        rec_loss_value = rec_loss_orig.sum().item()
        self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss_value))
        self.evaluator.optim_metrics.add("model_loss", AverageMetric(model_loss.item()))
        self.evaluator.optim_metrics.add("main_loss_f", AverageMetric(loss_f.item()))    
        self.evaluator.optim_metrics.add("main_loss_cf", AverageMetric(loss_cf.item()))

    def factual_loss(self, scores_orig_norm, scores_f):
        """
        事实损失：鼓励事实预测与原始预测一致
        
        当原始预测高分时（认为是正样本），事实预测也应该高
        当原始预测低分时（认为是负样本），事实预测也应该低
        """
        # 使用 scores 的相对排名来确定正负
        coef = scores_orig_norm.detach().clone()
        coef[scores_orig_norm >= 0.5] = 1
        coef[scores_orig_norm < 0.5] = -1
        
        # hinge loss: max(0, γ + coef * (0 - scores_f))
        # 当 coef=1 时，希望 scores_f 高，所以 0 - scores_f 应该负
        # 当 coef=-1 时，希望 scores_f 低，所以 0 - scores_f 应该正
        loss = torch.mean(torch.clamp(self.gamma + coef * (0 - scores_f), min=0))
        return loss

    def counterfactual_loss(self, scores_orig_norm, scores_cf):
        """
        反事实损失：鼓励反事实预测与原始预测相反
        
        当原始预测高分时，反事实预测应该低
        当原始预测低分时，反事实预测应该高
        """
        coef = scores_orig_norm.detach().clone()
        coef[scores_orig_norm >= 0.5] = -1  # 原来高，现在希望低
        coef[scores_orig_norm < 0.5] = 1    # 原来低，现在希望高
        
        loss = torch.mean(torch.clamp(self.gamma + coef * (0 - scores_cf), min=0))
        return loss

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
