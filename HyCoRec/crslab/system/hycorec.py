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
    return torch.sigmoid(gate_inputs).squeeze()


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
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']
        
        # ViewLearner 超参数（从配置中读取，设置默认值）
        # 仿照 CACHE/train.py 的参数设置
        self.kg_emb_dim = opt.get('kg_emb_dim', 128)
        self.view_lr = self.rec_optim_opt.get('view_lr', 0.01)       # CACHE 默认 1e-2
        self.view_wd = self.rec_optim_opt.get('view_wd', 0.001)      # CACHE 默认 1e-3
        self.view_alpha = self.rec_optim_opt.get('view_alpha', 0.5)  # factual vs counterfactual 权重
        self.view_lambda = self.rec_optim_opt.get('view_lambda', 5.0)  # 边权重正则化系数
        self.model_lambda = self.rec_optim_opt.get('model_lambda', 0.1)  # 主模型损失中的对比损失权重
        self.gamma = self.rec_optim_opt.get('gamma', 0.5)            # hinge loss margin
        self.temperature = self.rec_optim_opt.get('temperature', 1.0)  # gumbel softmax 温度
        self.use_counterfactual = self.rec_optim_opt.get('use_counterfactual', True)
        
        # 构建 ViewLearner（为三种超图各一个）
        # 注意：ViewLearner 需要能直接处理节点特征和超边索引
        self.view_learner_item = ViewLearner(self.kg_emb_dim, hidden_dim=64, device=self.device).to(self.device)
        self.view_learner_entity = ViewLearner(self.kg_emb_dim, hidden_dim=64, device=self.device).to(self.device)
        self.view_learner_word = ViewLearner(self.kg_emb_dim, hidden_dim=64, device=self.device).to(self.device)

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

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            
            # 每 50 个 epoch 衰减 view_lambda
            if (epoch + 1) % 50 == 0:
                self.view_lambda *= 0.5
                logger.info(f'[Decay view_lambda to {self.view_lambda}]')
            
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
            
            self.evaluator.report(epoch=epoch, mode='train')
            
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')
                # early stop
                metric = self.evaluator.optim_metrics['rec_loss']
                if self.early_stop(metric):
                    break
        
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

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
        self.model.eval()
        
        # 1. 原始预测（不带权重）
        with torch.no_grad():
            rec_loss_orig, scores_orig = self.model.forward(batch, 'train', 'rec')
        
        # 2. 获取 RGCN 编码后的嵌入（用于 ViewLearner）
        # 需要从模型中获取嵌入，然后构建超图，再调用 ViewLearner
        # 这里简化处理：直接调用带权重的推荐方法，让权重在模型内部通过回调生成
        
        # 为了让 ViewLearner 能够生成权重，我们需要一种方式让模型内部调用 ViewLearner
        # 方案：传入一个权重生成函数给模型
        
        def make_weight_fn(view_learner):
            """创建一个权重生成函数"""
            def fn(node_features, hyper_edge_index):
                weight_logits = view_learner(node_features, hyper_edge_index)
                # 应用 gumbel-softmax
                weight = gumbel_softmax(weight_logits, self.temperature)
                return weight, weight_logits  # 返回权重和 logits（用于正则化）
            return fn
        
        item_weight_fn = make_weight_fn(self.view_learner_item)
        entity_weight_fn = make_weight_fn(self.view_learner_entity)
        word_weight_fn = make_weight_fn(self.view_learner_word)
        
        # 3. 事实预测（带学习到的权重）
        rec_loss_f, scores_f, weight_info = self.model.recommend_with_weight_fn(
            batch, 'train',
            item_weight_fn=item_weight_fn,
            entity_weight_fn=entity_weight_fn,
            word_weight_fn=word_weight_fn
        )
        
        # 4. 反事实预测（用 1 - weight）
        def make_cf_weight_fn(view_learner):
            """创建反事实权重生成函数"""
            def fn(node_features, hyper_edge_index):
                weight_logits = view_learner(node_features, hyper_edge_index)
                weight = gumbel_softmax(weight_logits, self.temperature)
                cf_weight = 1 - weight  # 反事实权重
                return cf_weight, weight_logits
            return fn
        
        item_cf_fn = make_cf_weight_fn(self.view_learner_item)
        entity_cf_fn = make_cf_weight_fn(self.view_learner_entity)
        word_cf_fn = make_cf_weight_fn(self.view_learner_word)
        
        rec_loss_cf, scores_cf, _ = self.model.recommend_with_weight_fn(
            batch, 'train',
            item_weight_fn=item_cf_fn,
            entity_weight_fn=entity_cf_fn,
            word_weight_fn=word_cf_fn
        )
        
        # 5. 计算事实损失和反事实损失
        loss_f = self.factual_loss(scores_orig, scores_f)
        loss_cf = self.counterfactual_loss(scores_orig, scores_cf)
        
        # 6. 计算边权重正则化（鼓励保留更多边）
        aug_weight_mean = 0.0
        count = 0
        for info in weight_info:
            for key in ['item', 'entity', 'word']:
                if info.get(key) is not None and info[key].get('logits') is not None:
                    aug_weight = torch.sigmoid(info[key]['logits'])
                    aug_weight_mean += torch.mean(aug_weight)
                    count += 1
        if count > 0:
            aug_weight_mean = aug_weight_mean / count
        
        # 7. view_loss = α * loss_f + (1-α) * loss_cf + λ * mean(aug_weight)
        view_loss = (self.view_alpha * loss_f + 
                     (1 - self.view_alpha) * loss_cf + 
                     self.view_lambda * aug_weight_mean)
        
        # 8. 更新 ViewLearner
        self.view_optimizer.zero_grad()
        view_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.view_learner_item.parameters()) + 
            list(self.view_learner_entity.parameters()) + 
            list(self.view_learner_word.parameters()), 
            1.0
        )
        self.view_optimizer.step()
        
        # 记录指标
        self.evaluator.optim_metrics.add("view_loss", AverageMetric(view_loss.item()))

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
        
        # 1. 原始预测（不带权重）
        rec_loss_orig, scores_orig = self.model.forward(batch, 'train', 'rec')
        
        # 2. 事实预测（带 ViewLearner 生成的权重）
        def make_weight_fn(view_learner):
            def fn(node_features, hyper_edge_index):
                with torch.no_grad():  # ViewLearner 不更新
                    weight_logits = view_learner(node_features, hyper_edge_index)
                weight = gumbel_softmax(weight_logits.detach(), self.temperature)
                return weight, weight_logits.detach()
            return fn
        
        item_weight_fn = make_weight_fn(self.view_learner_item)
        entity_weight_fn = make_weight_fn(self.view_learner_entity)
        word_weight_fn = make_weight_fn(self.view_learner_word)
        
        rec_loss_f, scores_f, _ = self.model.recommend_with_weight_fn(
            batch, 'train',
            item_weight_fn=item_weight_fn,
            entity_weight_fn=entity_weight_fn,
            word_weight_fn=word_weight_fn
        )
        
        # 3. 反事实预测
        def make_cf_weight_fn(view_learner):
            def fn(node_features, hyper_edge_index):
                with torch.no_grad():
                    weight_logits = view_learner(node_features, hyper_edge_index)
                weight = gumbel_softmax(weight_logits.detach(), self.temperature)
                cf_weight = 1 - weight
                return cf_weight, weight_logits.detach()
            return fn
        
        item_cf_fn = make_cf_weight_fn(self.view_learner_item)
        entity_cf_fn = make_cf_weight_fn(self.view_learner_entity)
        word_cf_fn = make_cf_weight_fn(self.view_learner_word)
        
        rec_loss_cf, scores_cf, _ = self.model.recommend_with_weight_fn(
            batch, 'train',
            item_weight_fn=item_cf_fn,
            entity_weight_fn=entity_cf_fn,
            word_weight_fn=word_cf_fn
        )
        
        # 4. 计算事实/反事实损失
        loss_f = self.factual_loss(scores_orig.detach(), scores_f)
        loss_cf = self.counterfactual_loss(scores_orig.detach(), scores_cf)
        
        # 5. model_loss = rec_loss + λ_model * (α * loss_f + (1-α) * loss_cf)
        model_loss = rec_loss_orig + self.model_lambda * (
            self.view_alpha * loss_f + (1 - self.view_alpha) * loss_cf
        )
        
        # 6. 更新主模型
        self.backward(model_loss)
        
        # 记录指标
        rec_loss_value = rec_loss_orig.sum().item()
        self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss_value))

    def factual_loss(self, scores_orig, scores_f):
        """
        事实损失：鼓励事实预测与原始预测一致
        
        当原始预测高分时（认为是正样本），事实预测也应该高
        当原始预测低分时（认为是负样本），事实预测也应该低
        """
        # 使用 scores 的相对排名来确定正负
        scores_orig_norm = torch.sigmoid(scores_orig)
        coef = scores_orig_norm.detach().clone()
        coef[scores_orig_norm >= 0.5] = 1
        coef[scores_orig_norm < 0.5] = -1
        
        # hinge loss: max(0, γ + coef * (0 - scores_f))
        # 当 coef=1 时，希望 scores_f 高，所以 0 - scores_f 应该负
        # 当 coef=-1 时，希望 scores_f 低，所以 0 - scores_f 应该正
        loss = torch.mean(torch.clamp(self.gamma + coef * (0 - scores_f), min=0))
        return loss

    def counterfactual_loss(self, scores_orig, scores_cf):
        """
        反事实损失：鼓励反事实预测与原始预测相反
        
        当原始预测高分时，反事实预测应该低
        当原始预测低分时，反事实预测应该高
        """
        scores_orig_norm = torch.sigmoid(scores_orig)
        coef = scores_orig_norm.detach().clone()
        coef[scores_orig_norm >= 0.5] = -1  # 原来高，现在希望低
        coef[scores_orig_norm < 0.5] = 1    # 原来低，现在希望高
        
        loss = torch.mean(torch.clamp(self.gamma + coef * (0 - scores_cf), min=0))
        return loss

    def train_conversation(self):
        if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
            self.model.freeze_parameters()
        else:
            self.model.module.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')
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
            self.evaluator.report(mode='test')

    def fit(self):
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        pass
