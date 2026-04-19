# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/9
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import argparse
import os
import warnings
import yaml

from crslab.config import Config

warnings.filterwarnings('ignore')


def _parse_override_value(raw_value):
    if isinstance(raw_value, bool):
        return raw_value
    try:
        return yaml.safe_load(raw_value)
    except Exception:
        return raw_value


def _assign_nested(config_dict, dotted_key, value):
    cursor = config_dict
    key_parts = dotted_key.split('.')
    for part in key_parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[key_parts[-1]] = value


def _parse_unknown_overrides(unknown_args):
    overrides = {}
    idx = 0
    while idx < len(unknown_args):
        token = unknown_args[idx]
        if not token.startswith('--'):
            idx += 1
            continue

        option = token[2:]
        if '=' in option:
            key, raw_value = option.split('=', 1)
            value = _parse_override_value(raw_value)
            idx += 1
        else:
            next_token = unknown_args[idx + 1] if idx + 1 < len(unknown_args) else None
            if next_token is None or next_token.startswith('--'):
                key = option
                value = True
                idx += 1
            else:
                key = option
                value = _parse_override_value(next_token)
                idx += 2

        _assign_nested(overrides, key, value)

    return overrides


def _merge_dict(target, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value


def _build_sweep_agent_run_name(explicit_name=None):
    if explicit_name:
        return explicit_name

    sweep_id = os.environ.get('WANDB_SWEEP_ID')
    agent_gpu = os.environ.get('HYCOREC_SWEEP_AGENT_GPU')
    if not sweep_id or agent_gpu is None:
        return explicit_name

    agent_pid = os.getppid()
    return f'gpu{agent_gpu}_pid{agent_pid}'


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='config/crs/hycorec/redial.yaml', help='config file(yaml) path')
    parser.add_argument('-g', '--gpu', type=str, default=None,
                        help='specify GPU id(s) to use. Defaults to CUDA_VISIBLE_DEVICES when set, otherwise CPU(-1).')
    parser.add_argument('-sd', '--save_data', action='store_true',
                        help='save processed dataset')
    parser.add_argument('-rd', '--restore_data', action='store_true',
                        help='restore processed dataset')
    parser.add_argument('-ss', '--save_system', action='store_true',
                        help='save trained system')
    parser.add_argument('-rs', '--restore_system', action='store_true',
                        help='restore trained system')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='use valid dataset to debug your system')
    parser.add_argument('-i', '--interact', action='store_true',
                        help='interact with your system instead of training')
    parser.add_argument('-s', '--seed', type=int, default=2020)
    parser.add_argument('-p', '--pretrain', action='store_true', help='use pretrain weights')
    parser.add_argument('-e', '--pretrain_epoch', type=int, default=9999, help='pretrain epoch')
    parser.add_argument('--disw', action='store_true', help='disable wandb logging')
    parser.add_argument('--wp', type=str, default='redail', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/team')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_group', type=str, default=None, help='wandb run group')
    parser.add_argument('--wandb_mode', type=str, default=None, help='wandb mode (online/offline/disabled)')
    parser.add_argument('--wandb_tags', nargs='*', default=None, help='wandb tags separated by spaces')
    args, unknown_args = parser.parse_known_args()

    gpu = args.gpu if args.gpu is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '-1')
    config = Config(args.config, gpu, args.debug, args.seed, args.pretrain, args.pretrain_epoch)
    override_config = _parse_unknown_overrides(unknown_args)
    if override_config:
        _merge_dict(config.opt, override_config)

    config['use_wandb'] = not args.disw
    config['wandb'] = {
        'enable': not args.disw,
        'project': args.wp,
        'entity': args.wandb_entity,
        'name': _build_sweep_agent_run_name(args.wandb_name),
        'group': args.wandb_group,
        'mode': args.wandb_mode,
        'tags': args.wandb_tags,
    }
    config['sweep_overrides'] = override_config

    from crslab.quick_start import run_crslab

    run_crslab(config, args.save_data, args.restore_data, args.save_system, args.restore_system, args.interact,
               args.debug)
