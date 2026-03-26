import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def parse_args():
    default_workdir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Launch one detached wandb sweep agent per GPU.'
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
        default=list(range(3,8)),
        help='GPU ids to use. Defaults to 0 1 2 3 4 5 6 7.',
    )
    parser.add_argument(
        '--workdir',
        type=str,
        default=str(default_workdir),
        help='Working directory for launching agents. Defaults to the HyCoRec project root.',
    )
    parser.add_argument(
        '--foreground',
        action='store_true',
        help='Keep this launcher attached and wait for all agents.',
    )
    return parser.parse_args()


def build_agent_command(sweep_id, runs_per_agent):
    command = [sys.executable, '-m', 'wandb', 'agent']
    if runs_per_agent > 0:
        command.extend(['--count', str(runs_per_agent)])
    command.append(sweep_id)
    return command


def start_process(command, workdir, env, log_file, foreground):
    if foreground:
        return subprocess.Popen(
            command,
            cwd=str(workdir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    return subprocess.Popen(
        command,
        cwd=str(workdir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def write_pid_file(pid_path, pid):
    pid_path.write_text(str(pid), encoding='utf-8')


def main():
    args = parse_args() #接受命令行参数
    workdir = Path(args.workdir).resolve() #获取工作目录
    processes = [] #存储进程信息

    for gpu in args.gpus: # 遍历每个GPU
        agent_dir = workdir / 'wandb_sweep' / f'wandb_gpu{gpu}' #创建每个GPU的工作目录
        tmp_dir = agent_dir / 'tmp'
        log_path = agent_dir / f'agent_gpu{gpu}.log'
        pid_path = agent_dir / f'agent_gpu{gpu}.pid'
        agent_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['HYCOREC_SWEEP_AGENT_GPU'] = str(gpu)
        env['WANDB_DIR'] = str(agent_dir)
        env['TMPDIR'] = str(tmp_dir)
        env['TMP'] = str(tmp_dir)
        env['TEMP'] = str(tmp_dir)

        command = build_agent_command(args.sweep_id, args.runs_per_agent) #构建命令行参数
        log_file = open(log_path, 'a', encoding='utf-8') #打开日志文件
        process = start_process(command, workdir, env, log_file, args.foreground) #启动进程
        write_pid_file(pid_path, process.pid)
        processes.append((gpu, process, log_file))
        print(f'Started agent on GPU {gpu}, pid={process.pid}, log={log_path}')

    if not args.foreground: #如果后台运行
        for _, _, log_file in processes:
            log_file.close()
        print('All agents started in background.')
        return
    # 如果前台运行，等待所有进程结束；如果有进程异常退出，记录退出码；如果收到键盘中断，终止所有进程。
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
                os.killpg(process.pid, signal.SIGTERM)
        for _, process, log_file in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
            log_file.close()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
