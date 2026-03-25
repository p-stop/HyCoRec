import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Launch one wandb sweep agent per GPU.'
    )
    parser.add_argument(
        'sweep_id',
        help='W&B sweep id, e.g. entity/project/abc12345',
    )
    parser.add_argument(
        '--runs-per-agent',
        type=int,
        default=0,
        help='Maximum runs per agent. Use 0 for unlimited.',
    )
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='*',
        default=list(range(8)),
        help='GPU ids to use. Defaults to 0 1 2 3 4 5 6 7.',
    )
    parser.add_argument(
        '--workdir',
        type=str,
        default='.',
        help='Working directory for launching agents.',
    )
    return parser.parse_args()


def build_agent_command(sweep_id, runs_per_agent):
    command = ['wandb', 'agent']
    if runs_per_agent > 0:
        command.extend(['--count', str(runs_per_agent)])
    command.append(sweep_id)
    return command


def main():
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    processes = []

    for gpu in args.gpus:
        agent_dir = workdir / f'wandb_gpu{gpu}'
        log_path = workdir / f'agent_gpu{gpu}.log'
        agent_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['WANDB_DIR'] = str(agent_dir)

        command = build_agent_command(args.sweep_id, args.runs_per_agent)
        log_file = open(log_path, 'a', encoding='utf-8')
        process = subprocess.Popen(
            command,
            cwd=str(workdir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu, process, log_file))
        print(f'Started agent on GPU {gpu}, pid={process.pid}, log={log_path}')

    exit_code = 0
    try:
        for gpu, process, log_file in processes:
            return_code = process.wait()
            log_file.close()
            if return_code != 0:
                print(f'Agent on GPU {gpu} exited with code {return_code}', file=sys.stderr)
                exit_code = return_code if exit_code == 0 else exit_code
    except KeyboardInterrupt:
        print('Stopping agents...')
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, log_file in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            log_file.close()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
