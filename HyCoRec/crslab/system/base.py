# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/9
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

# UPDATE:
# @Time   : 2021/11/5
# @Author : Zhipeng Zhao
# @Email  : oran_official@outlook.com

import os
import contextlib
import inspect
from abc import ABC, abstractmethod
import numpy as np
import random
import torch
from loguru import logger
from torch import optim
from transformers import Adafactor
try:
    import wandb
except ImportError:
    wandb = None

from crslab.config import SAVE_PATH
from crslab.evaluator import get_evaluator
from crslab.evaluator.metrics.base import AverageMetric
from crslab.model import get_model
from crslab.system.utils import lr_scheduler
from crslab.system.utils.functions import compute_grad_norm

optim_class = {}
optim_class.update({k: v for k, v in optim.__dict__.items() if not k.startswith('__') and k[0].isupper()})
optim_class.update({'AdamW': optim.AdamW, 'Adafactor': Adafactor})
lr_scheduler_class = {k: v for k, v in lr_scheduler.__dict__.items() if not k.startswith('__') and k[0].isupper()}
transformers_tokenizer = ('bert', 'gpt2')


class BaseSystem(ABC):
    """Base class for all system"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.

        """
        self.opt = opt
        if opt["gpu"] == [-1]:
            self.device = torch.device('cpu')
        elif len(opt["gpu"]) == 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # seed
        if 'seed' in opt:
            seed = int(opt['seed'])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            logger.info(f'[Set seed] {seed}')
        # data
        if debug:
            self.train_dataloader = valid_dataloader
            self.valid_dataloader = valid_dataloader
            self.test_dataloader = test_dataloader
        else:
            self.train_dataloader = train_dataloader
            self.valid_dataloader = valid_dataloader
            self.test_dataloader =  test_dataloader
        self.vocab = vocab
        self.side_data = side_data
        # model
        if 'model' in opt:
            self.model = get_model(opt, opt['model'], self.device, vocab, side_data).to(self.device)
        else:
            if 'rec_model' in opt:
                self.rec_model = get_model(opt, opt['rec_model'], self.device, vocab['rec'], side_data['rec']).to(
                    self.device)
            if 'conv_model' in opt:
                self.conv_model = get_model(opt, opt['conv_model'], self.device, vocab['conv'], side_data['conv']).to(
                    self.device)
            if 'policy_model' in opt:
                self.policy_model = get_model(opt, opt['policy_model'], self.device, vocab['policy'],
                                              side_data['policy']).to(self.device)
        model_file_name = opt.get('model_file', f'{opt["model_name"]}.pth')
        self.model_file = os.path.join(SAVE_PATH, model_file_name)
        if restore_system:
            self.restore_model()

        if not interact:
            self.evaluator = get_evaluator('standard', opt['dataset'], opt['rankfile'])
        self.wandb_run = None
        self.use_wandb = False
        self._init_wandb(interact=interact, debug=debug)

    def _init_wandb(self, interact=False, debug=False):
        wandb_opt = self.opt.get('wandb', {})
        self.use_wandb = bool(self.opt.get('use_wandb', wandb_opt.get('enable', False)))
        if interact:
            self.use_wandb = False
        if not self.use_wandb:
            return
        if wandb is None:
            logger.warning('[WandB disabled] package `wandb` is not installed.')
            self.use_wandb = False
            return

        if isinstance(self.opt, dict):
            wandb_config = dict(self.opt)
        elif hasattr(self.opt, 'opt') and isinstance(self.opt.opt, dict):
            wandb_config = dict(self.opt.opt)
        else:
            wandb_config = {}

        init_kwargs = {
            'project': wandb_opt.get('project', 'HyCoRec_yuan'),
            'name': wandb_opt.get('name'),
            'entity': wandb_opt.get('entity'),
            'group': wandb_opt.get('group'),
            'tags': wandb_opt.get('tags'),
            'job_type': 'debug' if debug else 'train',
            'config': wandb_config,
        }
        mode = wandb_opt.get('mode')
        if mode:
            init_kwargs['mode'] = mode
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        try:
            self.wandb_run = wandb.init(**init_kwargs)
            self.wandb_run.define_metric("*", step_metric="epoch")
            logger.info('[WandB initialized]')
        except Exception as exc:
            logger.warning(f'[WandB disabled] init failed: {exc}')
            self.use_wandb = False
            self.wandb_run = None

    # 2) 统一 log 时显式带 epoch 字段（不要依赖默认 step）
    def log_wandb_metrics(self, metrics, stage=None, mode=None, epoch=None):
        if not self.use_wandb or self.wandb_run is None or not metrics:
            return

        prefix = '/'.join([part for part in (stage, mode) if part])
        payload = {}
        for key, value in metrics.items():
            metric_key = f'{prefix}/{key}' if prefix else key
            payload[metric_key] = value

        if epoch is not None and epoch >= 0:
            payload["epoch"] = epoch

        self.wandb_run.log(payload)


    def finish_wandb(self):
        if not self.use_wandb or self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def init_optim(self, opt, parameters):
        self.optim_opt = opt
        parameters = list(parameters)
        if isinstance(parameters[0], dict):
            for i, d in enumerate(parameters):
                parameters[i]['params'] = list(d['params'])

        # gradient acumulation
        self.update_freq = opt.get('update_freq', 1)
        self._number_grad_accum = 0

        self.gradient_clip = opt.get('gradient_clip', -1)

        self.build_optimizer(parameters)
        self.build_lr_scheduler()

        if isinstance(parameters[0], dict):
            self.parameters = []
            for d in parameters:
                self.parameters.extend(d['params'])
        else:
            self.parameters = parameters

        # early stop
        self.need_early_stop = self.optim_opt.get('early_stop', False)
        if self.need_early_stop:
            logger.debug('[Enable early stop]')
            self.reset_early_stop_state()

    def build_optimizer(self, parameters):
        optimizer_opt = self.optim_opt['optimizer']
        optimizer = optimizer_opt.pop('name')
        self.optimizer = optim_class[optimizer](parameters, **optimizer_opt)
        logger.info(f"[Build optimizer: {optimizer}]")

    def build_lr_scheduler(self):
        """
        Create the learning rate scheduler, and assign it to self.scheduler. This
        scheduler will be updated upon a call to receive_metrics. May also create
        self.warmup_scheduler, if appropriate.

        :param state_dict states: Possible state_dict provided by model
            checkpoint, for restoring LR state
        :param bool hard_reset: If true, the LR scheduler should ignore the
            state dictionary.
        """
        if self.optim_opt.get('lr_scheduler', None):
            lr_scheduler_opt = self.optim_opt['lr_scheduler']
            lr_scheduler = lr_scheduler_opt.pop('name')
            self.scheduler = lr_scheduler_class[lr_scheduler](self.optimizer, **lr_scheduler_opt)
            logger.info(f"[Build scheduler {lr_scheduler}]")

    def reset_early_stop_state(self):
        self.best_valid = None
        self.drop_cnt = 0
        self.impatience = self.optim_opt.get('impatience', 3)
        if self.optim_opt['stop_mode'] == 'max':
            self.stop_mode = 1
        elif self.optim_opt['stop_mode'] == 'min':
            self.stop_mode = -1
        else:
            raise
        logger.debug('[Reset early stop state]')

    @abstractmethod
    def fit(self):
        """fit the whole system"""
        pass

    @abstractmethod
    def step(self, batch, stage, mode):
        """calculate loss and prediction for batch data under certrain stage and mode

        Args:
            batch (dict or tuple): batch data
            stage (str): recommendation/policy/conversation etc.
            mode (str): train/valid/test
        """
        pass

    def backward(self, loss):
        """empty grad, backward loss and update params

        Args:
            loss (torch.Tensor):
        """
        if getattr(self, 'nan_debug', False):
            self._debug_check_tensor('BaseSystem.backward.loss_before_zero_grad', loss)
        self._zero_grad()

        if self.update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum + 1) % self.update_freq
            loss /= self.update_freq
            if getattr(self, 'nan_debug', False):
                self._debug_check_tensor('BaseSystem.backward.loss_after_update_freq', loss)

        if getattr(self, 'nan_debug', False):
            grad_tensor = loss.clone().detach()
            self._debug_check_tensor('BaseSystem.backward.explicit_grad_tensor', grad_tensor)
            with self._debug_anomaly_context():
                loss.backward(grad_tensor)
            self._debug_check_gradients('BaseSystem.backward.after_backward')
        else:
            loss.backward(loss.clone().detach())

        self._update_params()

    def _zero_grad(self):
        if self._number_grad_accum != 0:
            # if we're accumulating gradients, don't actually zero things out yet.
            return
        self.optimizer.zero_grad()

    def _update_params(self):
        if self.update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            # self._number_grad_accum is updated in backward function
            if self._number_grad_accum != 0:
                return

        if self.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, self.gradient_clip
            )
            if getattr(self, 'nan_debug', False):
                self._debug_check_tensor('BaseSystem._update_params.grad_norm_after_clip', grad_norm)
            self.evaluator.optim_metrics.add('grad norm', AverageMetric(grad_norm))
            self.evaluator.optim_metrics.add(
                'grad clip ratio',
                AverageMetric(float(grad_norm > self.gradient_clip)),
            )
        else:
            grad_norm = compute_grad_norm(self.parameters)
            if getattr(self, 'nan_debug', False):
                self._debug_check_tensor('BaseSystem._update_params.grad_norm', torch.as_tensor(grad_norm))
            self.evaluator.optim_metrics.add('grad norm', AverageMetric(grad_norm))

        self.optimizer.step()
        if getattr(self, 'nan_debug', False):
            self._debug_check_parameters('BaseSystem._update_params.after_optimizer_step')

        if hasattr(self, 'scheduler'):
            self.scheduler.train_step()

    def _debug_anomaly_context(self):
        if getattr(self, 'nan_debug', False) and getattr(self, 'nan_debug_anomaly', True):
            return torch.autograd.detect_anomaly()
        return contextlib.nullcontext()

    def _debug_location(self):
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None or frame.f_back.f_back is None:
            return 'unknown'
        caller = frame.f_back.f_back
        return f'{os.path.basename(caller.f_code.co_filename)}:{caller.f_lineno}'

    def _debug_tensor_stats(self, tensor):
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
            return stats

    def _debug_check_tensor(self, name, tensor):
        if not getattr(self, 'nan_debug', False):
            return True
        if not isinstance(tensor, torch.Tensor):
            return True
        if not (tensor.is_floating_point() or tensor.is_complex()):
            return True
        if torch.isfinite(tensor).all().item():
            return True

        message = (
            f"[NUMERIC DEBUG] non-finite tensor '{name}' at {self._debug_location()} "
            f"stats={self._debug_tensor_stats(tensor)} context={getattr(self, '_debug_context', {})}"
        )
        logger.error(message)
        if getattr(self, 'nan_debug_raise', True):
            raise FloatingPointError(message)
        return False

    def _debug_check_gradients(self, name):
        for param_name, param in self.model.named_parameters():
            if param.grad is not None and not self._debug_check_tensor(f'{name}.grad.{param_name}', param.grad):
                return False
        return True

    def _debug_check_parameters(self, name):
        for param_name, param in self.model.named_parameters():
            if not self._debug_check_tensor(f'{name}.param.{param_name}', param):
                return False
        return True

    def adjust_lr(self, metric=None):
        """adjust learning rate w/o metric by scheduler

        Args:
            metric (optional): Defaults to None.
        """
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            return
        self.scheduler.valid_step(metric)
        logger.debug('[Adjust learning rate after valid epoch]')

    def early_stop(self, metric):
        if not self.need_early_stop:
            return False
        if self.best_valid is None or metric * self.stop_mode > self.best_valid * self.stop_mode:
            self.best_valid = metric
            self.drop_cnt = 0
            logger.info('[Get new best model]')
            return False
        else:
            self.drop_cnt += 1
            if self.drop_cnt >= self.impatience:
                logger.info('[Early stop]')
                return True

    def save_model(self):
        r"""Store the model parameters."""
        state = {}
        if hasattr(self, 'model'):
            state['model_state_dict'] = self.model.state_dict()
        if hasattr(self, 'rec_model'):
            state['rec_state_dict'] = self.rec_model.state_dict()
        if hasattr(self, 'conv_model'):
            state['conv_state_dict'] = self.conv_model.state_dict()
        if hasattr(self, 'policy_model'):
            state['policy_state_dict'] = self.policy_model.state_dict()

        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(state, self.model_file)
        logger.info(f'[Save model into {self.model_file}]')

    def restore_model(self):
        r"""Store the model parameters."""
        if not os.path.exists(self.model_file):
            raise ValueError(f'Saved model [{self.model_file}] does not exist')
        checkpoint = torch.load(self.model_file, map_location=self.device)
        if hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'rec_model'):
            self.rec_model.load_state_dict(checkpoint['rec_state_dict'])
        if hasattr(self, 'conv_model'):
            self.conv_model.load_state_dict(checkpoint['conv_state_dict'])
        if hasattr(self, 'policy_model'):
            self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
        logger.info(f'[Restore model from {self.model_file}]')

    @abstractmethod
    def interact(self):
        pass
