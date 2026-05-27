# Copyright (c) 2025 Huawei Technologies Co., Ltd
# 模拟在 N 个终端里各自执行 run.sh：每个子进程独立跑一份 infer.py，互不调度。

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


def prepare_run_log_dir(log_base):
    run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run_log_dir = os.path.join(log_base, run_name)
    if os.path.isdir(run_log_dir):
        shutil.rmtree(run_log_dir)
    os.makedirs(run_log_dir, exist_ok=True)
    return run_log_dir


def build_infer_cmd(python_exe, model_path, infer_count, warm_up_times, output_dir, stream):
    cmd = [
        python_exe,
        'infer.py',
        '--model_path', model_path,
        '--infer_count', str(infer_count),
        '--warm_up_times', str(warm_up_times),
        '--output_dir', output_dir,
    ]
    if stream:
        cmd.append('--stream')
    return cmd


def spawn_client(client_id, python_exe, model_path, infer_count, warm_up_times,
                 output_base, stream, run_log_dir, work_dir):
    output_dir = os.path.join(output_base, 'client_{}'.format(client_id))
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(run_log_dir, 'client_{}.log'.format(client_id))
    cmd = build_infer_cmd(
        python_exe, model_path, infer_count, warm_up_times, output_dir, stream)

    log_fp = open(log_path, 'w', buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=work_dir,
    )
    return {
        'client_id': client_id,
        'pid': proc.pid,
        'proc': proc,
        'log_path': log_path,
        'output_dir': output_dir,
        'log_fp': log_fp,
        'spawn_time': time.time(),
        'start_time': time.time(),
    }


def wait_clients(clients):
    results = []
    batch_start = min(c['spawn_time'] for c in clients)
    for client in clients:
        exit_code = client['proc'].wait()
        client['log_fp'].close()
        elapsed = time.time() - client['start_time']
        result = {
            'client_id': client['client_id'],
            'pid': client['pid'],
            'exit_code': exit_code,
            'elapsed': elapsed,
            'log_path': client['log_path'],
            'output_dir': client['output_dir'],
            'success': exit_code == 0,
        }
        results.append(result)
        status = 'OK' if result['success'] else 'FAILED'
        print('[INFO] client_{} pid={} {} exit_code={} elapsed={:.3f}s log={}'.format(
            client['client_id'], client['pid'], status, exit_code, elapsed, client['log_path']),
            flush=True)
    batch_elapsed = time.time() - batch_start
    return results, batch_elapsed


def write_summary(run_log_dir, args, results, batch_elapsed):
    summary_path = os.path.join(run_log_dir, 'summary.log')
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count

    with open(summary_path, 'w', buffering=1) as f:
        f.write('[SUMMARY] log_dir={}\n'.format(run_log_dir))
        f.write('[SUMMARY] mode=manual_terminal_spawn, concurrency={}, infer_count={}\n'.format(
            args.concurrency, args.infer_count))
        f.write('[SUMMARY] each_client=infer.py (same as run.sh), stream={}\n'.format(args.stream))
        f.write('[SUMMARY] success={}, failed={}, batch_wall={:.3f}s\n'.format(
            success_count, failed_count, batch_elapsed))
        if results:
            elapsed_list = sorted(r['elapsed'] for r in results)
            p95_idx = max(0, min(len(elapsed_list) - 1, int(len(elapsed_list) * 0.95) - 1))
            f.write('[SUMMARY] avg_client_elapsed={:.3f}s, p95_client_elapsed={:.3f}s\n'.format(
                sum(elapsed_list) / len(elapsed_list), elapsed_list[p95_idx]))
        for result in sorted(results, key=lambda x: x['client_id']):
            f.write('[CLIENT] client_{} pid={} success={} exit_code={} elapsed={:.3f}s log={}\n'.format(
                result['client_id'], result['pid'], result['success'], result['exit_code'],
                result['elapsed'], result['log_path']))
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description='Spawn N independent infer.py processes (like N terminals running run.sh)')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--concurrency', default=10, type=int, help='number of parallel infer.py')
    parser.add_argument('--infer_count', default=5, type=int, help='infer loop count per client')
    parser.add_argument('--warm_up_times', default=5, type=int, help='warm up times per client')
    parser.add_argument(
        '--output_dir',
        default='/home/ma-user/work/test/model/CosyVoice-claude/testout/manual_concurrent',
        type=str,
        help='base output dir; each client writes to output_dir/client_N/',
    )
    parser.add_argument('--log_dir', default='logs/manual', type=str, help='log base dir')
    parser.add_argument('--stream', action='store_true', help='stream infer')
    parser.add_argument(
        '--python',
        default=sys.executable,
        type=str,
        help='python executable for child infer.py (default: current interpreter)',
    )
    args = parser.parse_args()

    work_dir = os.path.dirname(os.path.abspath(__file__))
    infer_py = os.path.join(work_dir, 'infer.py')
    if not os.path.isfile(infer_py):
        print('[ERROR] infer.py not found: {}'.format(infer_py), file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.python):
        print('[ERROR] python not found: {}'.format(args.python), file=sys.stderr)
        sys.exit(1)

    run_log_dir = prepare_run_log_dir(args.log_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print('[INFO] run_log_dir={}'.format(run_log_dir), flush=True)
    print('[INFO] mode=manual_terminal_spawn, concurrency={}'.format(args.concurrency), flush=True)
    print('[INFO] python={}'.format(args.python), flush=True)
    print('[INFO] model_path={}'.format(args.model_path), flush=True)
    print('[INFO] each client runs: infer.py --infer_count={} --warm_up_times={} --stream={}'.format(
        args.infer_count, args.warm_up_times, args.stream), flush=True)
    print('[INFO] spawning {} clients at once (like opening {} terminals)...'.format(
        args.concurrency, args.concurrency), flush=True)

    clients = []
    spawn_start = time.time()
    for client_id in range(args.concurrency):
        client = spawn_client(
            client_id,
            args.python,
            args.model_path,
            args.infer_count,
            args.warm_up_times,
            args.output_dir,
            args.stream,
            run_log_dir,
            work_dir,
        )
        clients.append(client)
        print('[INFO] spawned client_{} pid={} log={}'.format(
            client_id, client['pid'], client['log_path']), flush=True)

    print('[INFO] all {} clients spawned in {:.3f}s, waiting...'.format(
        args.concurrency, time.time() - spawn_start), flush=True)

    results, batch_elapsed = wait_clients(clients)
    summary_path = write_summary(run_log_dir, args, results, batch_elapsed)

    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count

    print('=' * 72, flush=True)
    print('[SUMMARY] log_dir={}'.format(run_log_dir), flush=True)
    print('[SUMMARY] summary_log={}'.format(summary_path), flush=True)
    print('[SUMMARY] concurrency={}, success={}, failed={}, batch_wall={:.3f}s'.format(
        args.concurrency, success_count, failed_count, batch_elapsed), flush=True)
    if results:
        elapsed_list = sorted(r['elapsed'] for r in results)
        p95_idx = max(0, min(len(elapsed_list) - 1, int(len(elapsed_list) * 0.95) - 1))
        print('[SUMMARY] avg_client_elapsed={:.3f}s, p95_client_elapsed={:.3f}s'.format(
            sum(elapsed_list) / len(elapsed_list), elapsed_list[p95_idx]), flush=True)
    print('=' * 72, flush=True)

    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
