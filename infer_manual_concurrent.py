# Copyright (c) 2025 Huawei Technologies Co., Ltd
# 模拟在 N 个终端里各自执行 run.sh：每个子进程独立跑一份 infer.py，互不调度。
# 支持进程错峰启动、CPU 亲和性绑定、串行 warmup 等优化手段。

import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime


ENV_FINGERPRINT_KEYS = (
    'ASCEND_RT_VISIBLE_DEVICES',
    'ASCEND_GEO_W8A16',
    'DYNAMIC_QUANT',
    'ACLNN_CACHE_LIMIT',
    'ENABLE_DYNAMIC_SHAPE_MULTI_STREAM',
    'TASK_QUEUE_ENABLE',
    'COSYVOICE2_QWEN_HIDDEN_ONLY',
    'COSYVOICE2_SAMPLING_MODE',
    'COSYVOICE2_FAST_TOPK_K',
    'COSYVOICE2_FAST_TOPK_RAW_LOGITS',
    'COSYVOICE2_REUSE_EMBEDDING_OUTPUT',
    'COSYVOICE2_CACHE_DECODE_METADATA',
    'COSYVOICE2_PIPELINE_LLM_FLOW',
    'COSYVOICE2_FLOW_STATIC_SET_CONTEXT',
    'COSYVOICE2_DEVICE_TOKEN_DECODE',
    'COSYVOICE2_DISABLE_TQDM',
    'COSYVOICE2_EMPTY_CACHE_EACH_UTT',
    'COSYVOICE2_MARK_STATIC_INPUTS',
    'COSYVOICE2_HIFT_DECODE_OM',
    'COSYVOICE2_OM_SET_CONTEXT_EACH_CHUNK',
    'COSYVOICE2_FLOW_HIFT_STREAM',
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'TORCH_INTEROP_THREADS',
    'OMP_PROC_BIND',
    'OMP_PLACES',
    'OPENBLAS_NUM_THREADS',
    'PYTORCH_NPU_ALLOC_CONF',
    'LD_PRELOAD',
)


def environment_fingerprint():
    return {key: os.environ.get(key, '<unset>') for key in ENV_FINGERPRINT_KEYS}


# ---------------------------------------------------------------------------
# 平台感知：获取 CPU 总数
# ---------------------------------------------------------------------------
def get_cpu_count():
    """返回系统 CPU 总数（物理核或逻辑核，与 nproc 一致）"""
    return os.cpu_count() or 192  # fallback 192 for Ascend platform


def parse_cpu_list(cpu_spec):
    """Parse CPU list like '144-167' or '144-151,160-167'."""
    cpus = []
    for part in cpu_spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def cpu_affinity_ranges(concurrency, total_cpus=None, cpu_list=None,
                        share_cpu_list=False):
    """将 CPU 核均分为 concurrency 段，返回 [(start, end), ...] 列表"""
    if cpu_list is None:
        if total_cpus is None:
            total_cpus = get_cpu_count()
        chunk_size = total_cpus // concurrency
        ranges = []
        for i in range(concurrency):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < concurrency - 1 else total_cpus - 1
            ranges.append((start, end))
        return ranges
    if len(cpu_list) < concurrency:
        raise ValueError('cpu_list has fewer CPUs ({}) than concurrency ({})'.format(
            len(cpu_list), concurrency))
    if share_cpu_list:
        return [(cpu_list[0], cpu_list[-1]) for _ in range(concurrency)]
    base_size = len(cpu_list) // concurrency
    remainder = len(cpu_list) % concurrency
    ranges = []
    offset = 0
    for i in range(concurrency):
        size = base_size + (1 if i < remainder else 0)
        chunk = cpu_list[offset: offset + size]
        ranges.append((chunk[0], chunk[-1]))
        offset += size
    return ranges


# ---------------------------------------------------------------------------
# 日志 & 目录
# ---------------------------------------------------------------------------
def prepare_run_log_dir(log_base):
    run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run_log_dir = os.path.join(log_base, run_name)
    if os.path.isdir(run_log_dir):
        shutil.rmtree(run_log_dir)
    os.makedirs(run_log_dir, exist_ok=True)
    return run_log_dir


# ---------------------------------------------------------------------------
# 构建 infer.py 命令行
# ---------------------------------------------------------------------------
def build_infer_cmd(python_exe, model_path, infer_count, warm_up_times,
                    output_dir, stream, no_save_audio, warmup_full=False,
                    sync_start_dir='', sync_start_timeout=900.0,
                    text_file='', spk_id=''):
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
    if no_save_audio:
        cmd.append('--no_save_audio')
    if warmup_full:
        cmd.append('--warmup_full')
    if sync_start_dir:
        cmd.extend([
            '--sync_start_dir', sync_start_dir,
            '--sync_start_timeout', str(sync_start_timeout),
        ])
    if text_file:
        cmd.extend(['--text_file', text_file])
    if spk_id:
        cmd.extend(['--spk_id', spk_id])
    return cmd


# ---------------------------------------------------------------------------
# 启动单个客户端进程
# ---------------------------------------------------------------------------
def spawn_client(client_id, python_exe, model_path, infer_count, warm_up_times,
                 output_base, stream, no_save_audio, run_log_dir, work_dir,
                 env_extra=None, warmup_full=False, sync_start_dir='',
                 sync_start_timeout=900.0, text_file='', spk_id=''):
    output_dir = os.path.join(output_base, 'client_{}'.format(client_id))
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(run_log_dir, 'client_{}.log'.format(client_id))
    cmd = build_infer_cmd(
        python_exe, model_path, infer_count, warm_up_times, output_dir, stream,
        no_save_audio, warmup_full, sync_start_dir, sync_start_timeout,
        text_file, spk_id)

    # 合并额外环境变量
    child_env = os.environ.copy()
    if env_extra:
        child_env.update(env_extra)

    log_fp = open(log_path, 'w', buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=work_dir,
        env=child_env,
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
        'cmd': cmd,
        'env_extra': env_extra,
    }


# ---------------------------------------------------------------------------
# 等待客户端完成 & 收集结果
# ---------------------------------------------------------------------------
def wait_client(client):
    """等待单个 client 完成，返回 result dict"""
    exit_code = client['proc'].wait()
    client['log_fp'].close()
    elapsed = time.time() - client['start_time']
    return {
        'client_id': client['client_id'],
        'pid': client['pid'],
        'exit_code': exit_code,
        'elapsed': elapsed,
        'log_path': client['log_path'],
        'output_dir': client['output_dir'],
        'success': exit_code == 0,
    }


def wait_clients(clients):
    results = []
    batch_start = min(c['spawn_time'] for c in clients)
    for client in clients:
        result = wait_client(client)
        results.append(result)
        status = 'OK' if result['success'] else 'FAILED'
        print('[INFO] client_{} pid={} {} exit_code={} elapsed={:.3f}s log={}'.format(
            result['client_id'], result['pid'], status, result['exit_code'],
            result['elapsed'], result['log_path']),
            flush=True)
    batch_elapsed = time.time() - batch_start
    return results, batch_elapsed


def wait_sync_ready(clients, sync_dir, timeout):
    os.makedirs(sync_dir, exist_ok=True)
    expected = {
        'client_{}.ready'.format(client['client_id'])
        for client in clients
    }
    start = time.time()
    while True:
        ready = set(os.path.basename(path) for path in
                    [os.path.join(sync_dir, name)
                     for name in os.listdir(sync_dir)]
                    if path.endswith('.ready'))
        missing = sorted(expected - ready)
        if not missing:
            go_path = os.path.join(sync_dir, 'go')
            with open(go_path, 'w', buffering=1) as f:
                f.write('{:.6f}\n'.format(time.time()))
            print('[SYNC] all {} clients ready in {:.3f}s, go={}'.format(
                len(clients), time.time() - start, go_path), flush=True)
            return time.time() - start
        failed = [client for client in clients
                  if client['proc'].poll() is not None]
        if failed:
            ids = ','.join(str(client['client_id']) for client in failed)
            raise RuntimeError(
                'clients exited before sync ready: {}'.format(ids))
        if timeout > 0 and time.time() - start > timeout:
            raise TimeoutError(
                'sync ready timeout after {:.1f}s, missing={}'.format(
                    timeout, ','.join(missing)))
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# Summary 写入
# ---------------------------------------------------------------------------
def write_summary(run_log_dir, args, results, batch_elapsed, extra_info=None):
    summary_path = os.path.join(run_log_dir, 'summary.log')
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count

    with open(summary_path, 'w', buffering=1) as f:
        f.write('[SUMMARY] log_dir={}\n'.format(run_log_dir))
        f.write('[SUMMARY] mode=manual_terminal_spawn, concurrency={}, infer_count={}\n'.format(
            args.concurrency, args.infer_count))
        f.write('[SUMMARY] each_client=infer.py (same as run.sh), stream={}\n'.format(args.stream))
        if extra_info:
            for k, v in extra_info.items():
                f.write('[SUMMARY] {}={}\n'.format(k, v))
        f.write('[SUMMARY] success={}, failed={}, batch_wall={:.3f}s\n'.format(
            success_count, failed_count, batch_elapsed))
        if results:
            elapsed_list = sorted(r['elapsed'] for r in results)
            p95_idx = max(0, min(len(elapsed_list) - 1,
                                 int(len(elapsed_list) * 0.95) - 1))
            f.write('[SUMMARY] avg_client_elapsed={:.3f}s, p95_client_elapsed={:.3f}s\n'.format(
                sum(elapsed_list) / len(elapsed_list), elapsed_list[p95_idx]))
        for result in sorted(results, key=lambda x: x['client_id']):
            f.write('[CLIENT] client_{} pid={} success={} exit_code={} elapsed={:.3f}s log={}\n'.format(
                result['client_id'], result['pid'], result['success'],
                result['exit_code'], result['elapsed'], result['log_path']))
    return summary_path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Spawn N independent infer.py processes (like N terminals running run.sh)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='model path')
    parser.add_argument('--concurrency', default=10, type=int,
                        help='number of parallel infer.py')
    parser.add_argument('--infer_count', default=5, type=int,
                        help='infer loop count per client')
    parser.add_argument('--warm_up_times', default=5, type=int,
                        help='warm up times per client')
    parser.add_argument(
        '--output_dir',
        default='/home/ma-user/work/test/model/CosyVoice-claude/testout/manual_concurrent',
        type=str,
        help='base output dir; each client writes to output_dir/client_N/',
    )
    parser.add_argument('--log_dir', default='logs/manual', type=str,
                        help='log base dir')
    parser.add_argument('--text_file', default='', type=str,
                        help='utf-8 text file passed to infer.py; one non-empty line is one inference text')
    parser.add_argument('--spk_id', default='', type=str,
                        help='SFT speaker id passed to infer.py')
    parser.add_argument('--stream', action='store_true',
                        help='stream infer')
    parser.add_argument('--no_save_audio', action='store_true',
                        help='consume inference output without writing wav files')
    parser.add_argument(
        '--python',
        default=sys.executable,
        type=str,
        help='python executable for child infer.py (default: current interpreter)',
    )

    # --- 新增优化参数 ---
    parser.add_argument(
        '--stagger_delay',
        default=0.0, type=float,
        help='每个进程启动之间的最大随机延迟（秒），0 表示同时启动',
    )
    parser.add_argument(
        '--enable_cpu_affinity',
        action='store_true',
        help='为每个进程绑定专属 CPU 核范围，减少 CPU 竞争',
    )
    parser.add_argument(
        '--serial_warmup',
        action='store_true',
        help='串行执行所有进程的 warmup（减少 NPU 编译风暴），warmup 结束后再并发推理',
    )
    parser.add_argument(
        '--warmup_full',
        action='store_true',
        help='warmup 时完整消费流式输出；不设置时只等待首包，行为与 run.sh 旧版本一致',
    )
    parser.add_argument(
        '--sync_start',
        action='store_true',
        help='所有子进程 warmup 完成后等待统一 go，再同时开始正式推理',
    )
    parser.add_argument(
        '--sync_start_timeout',
        default=900.0, type=float,
        help='等待所有子进程进入正式推理屏障的超时时间（秒）',
    )
    parser.add_argument(
        '--total_cpus',
        default=None, type=int,
        help='系统 CPU 总数（默认自动检测）',
    )
    parser.add_argument(
        '--cpu_affinity_cpus',
        default='',
        type=str,
        help='CPU list/range used for affinity split, e.g. 144-167. Overrides --total_cpus when set.',
    )
    parser.add_argument(
        '--cpu_affinity_share',
        action='store_true',
        help='bind every client to the same --cpu_affinity_cpus range instead of splitting it',
    )
    args = parser.parse_args()

    work_dir = os.path.dirname(os.path.abspath(__file__))
    infer_py = os.path.join(work_dir, 'infer.py')
    if not os.path.isfile(infer_py):
        print('[ERROR] infer.py not found: {}'.format(infer_py),
              file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.python):
        print('[ERROR] python not found: {}'.format(args.python),
              file=sys.stderr)
        sys.exit(1)
    if args.text_file and not os.path.isfile(args.text_file):
        print('[ERROR] text_file not found: {}'.format(args.text_file),
              file=sys.stderr)
        sys.exit(1)

    run_log_dir = prepare_run_log_dir(args.log_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    affinity_cpu_list = parse_cpu_list(args.cpu_affinity_cpus) if args.cpu_affinity_cpus else None
    total_cpus = len(affinity_cpu_list) if affinity_cpu_list else (args.total_cpus or get_cpu_count())

    print('[INFO] run_log_dir={}'.format(run_log_dir), flush=True)
    print('[INFO] mode=manual_terminal_spawn, concurrency={}'.format(
        args.concurrency), flush=True)
    print('[INFO] python={}'.format(args.python), flush=True)
    print('[INFO] model_path={}'.format(args.model_path), flush=True)
    if args.text_file:
        print('[INFO] text_file={}'.format(args.text_file), flush=True)
    print('[INFO] each client runs: infer.py --infer_count={} --warm_up_times={} --stream={}'.format(
        args.infer_count, args.warm_up_times, args.stream), flush=True)
    print('[INFO] total_cpus={}, stagger_delay={:.1f}s, cpu_affinity={}, serial_warmup={}'.format(
        total_cpus, args.stagger_delay, args.enable_cpu_affinity,
        args.serial_warmup), flush=True)
    print('[INFO] environment_fingerprint={}'.format(environment_fingerprint()), flush=True)

    # --- 计算 CPU 亲和性范围 ---
    cpu_ranges = None
    if args.enable_cpu_affinity:
        cpu_ranges = cpu_affinity_ranges(
            args.concurrency, total_cpus, affinity_cpu_list,
            args.cpu_affinity_share)
        for i, (start, end) in enumerate(cpu_ranges):
            print('[INFO] client_{} cpu_range={}-{}'.format(i, start, end),
                  flush=True)

    extra_info = {
        'stagger_delay': args.stagger_delay,
        'cpu_affinity': args.enable_cpu_affinity,
        'serial_warmup': args.serial_warmup,
        'sync_start': args.sync_start,
        'sync_start_timeout': args.sync_start_timeout,
        'total_cpus': total_cpus,
        'cpu_affinity_cpus': args.cpu_affinity_cpus,
        'cpu_affinity_share': args.cpu_affinity_share,
        'no_save_audio': args.no_save_audio,
        'text_file': args.text_file,
        'spk_id': args.spk_id,
        'environment_fingerprint': environment_fingerprint(),
    }

    clients = []
    sync_dir = os.path.join(run_log_dir, 'sync_start')
    sync_wait_elapsed = None

    # ====================================================================
    # Phase 1: 串行 Warmup（可选）
    # ====================================================================
    if args.serial_warmup:
        print('=' * 72, flush=True)
        print('[PHASE 1] Serial warmup: {} clients one-by-one'.format(
            args.concurrency), flush=True)

        for client_id in range(args.concurrency):
            print('[INFO] serial warmup client_{} starting...'.format(client_id),
                  flush=True)

            env_extra = {}
            if cpu_ranges:
                start, end = cpu_ranges[client_id]
                env_extra.update({
                    'CPU_AFFINITY_START': str(start),
                    'CPU_AFFINITY_END': str(end),
                    'CPU_AFFINITY_CLIENT_ID': str(client_id),
                })

            client = spawn_client(
                client_id,
                args.python,
                args.model_path,
                0,  # infer_count=0: 仅 warmup
                args.warm_up_times,
                args.output_dir,
                args.stream,
                args.no_save_audio,
                run_log_dir,
                work_dir,
                env_extra=env_extra,
                warmup_full=args.warmup_full,
                text_file=args.text_file,
                spk_id=args.spk_id,
            )

            # CPU 亲和性：在父进程侧对子进程 PID 设置
            if cpu_ranges:
                start, end = cpu_ranges[client_id]
                try:
                    os.sched_setaffinity(client['pid'], range(start, end + 1))
                except Exception as e:
                    print('[WARN] failed to set affinity for pid={}: {}'.format(
                        client['pid'], e), flush=True)

            result = wait_client(client)
            status = 'OK' if result['success'] else 'FAILED'
            print('[INFO] serial warmup client_{} {} elapsed={:.3f}s'.format(
                client_id, status, result['elapsed']), flush=True)

            if not result['success']:
                print('[ERROR] warmup client_{} failed, abort'.format(client_id),
                      file=sys.stderr)
                sys.exit(1)

        print('[PHASE 1] Serial warmup completed for {} clients'.format(
            args.concurrency), flush=True)
        print('=' * 72, flush=True)

    # ====================================================================
    # Phase 2: 并发推理
    # ====================================================================
    print('[PHASE 2] Spawning {} concurrent inference clients...'.format(
        args.concurrency), flush=True)

    spawn_start = time.time()
    for client_id in range(args.concurrency):
        env_extra = {}
        if cpu_ranges:
            start, end = cpu_ranges[client_id]
            env_extra.update({
                'CPU_AFFINITY_START': str(start),
                'CPU_AFFINITY_END': str(end),
                'CPU_AFFINITY_CLIENT_ID': str(client_id),
            })
        if args.serial_warmup:
            # 通知 infer.py 跳过 warmup
            env_extra['SKIP_WARMUP'] = '1'

        child_sync_dir = ''
        if args.sync_start:
            child_sync_dir = sync_dir

        client = spawn_client(
            client_id,
            args.python,
            args.model_path,
            args.infer_count,
            args.warm_up_times,
            args.output_dir,
            args.stream,
            args.no_save_audio,
            run_log_dir,
            work_dir,
            env_extra=env_extra,
            warmup_full=args.warmup_full,
            sync_start_dir=child_sync_dir,
            sync_start_timeout=args.sync_start_timeout,
            text_file=args.text_file,
            spk_id=args.spk_id,
        )

        # CPU 亲和性
        if cpu_ranges:
            start, end = cpu_ranges[client_id]
            try:
                os.sched_setaffinity(client['pid'], range(start, end + 1))
            except Exception as e:
                print('[WARN] failed to set affinity for pid={}: {}'.format(
                    client['pid'], e), flush=True)

        clients.append(client)
        print('[INFO] spawned client_{} pid={} cpu={}-{} log={}'.format(
            client_id, client['pid'],
            cpu_ranges[client_id][0] if cpu_ranges else '-',
            cpu_ranges[client_id][1] if cpu_ranges else '-',
            client['log_path']), flush=True)

        # 错峰延迟：在启动下一个进程之前等待一段随机时间
        if args.stagger_delay > 0 and client_id < args.concurrency - 1:
            delay = random.uniform(0, args.stagger_delay)
            print('[INFO] stagger delay {:.2f}s before next spawn...'.format(delay),
                  flush=True)
            time.sleep(delay)

    print('[INFO] all {} clients spawned in {:.3f}s, waiting...'.format(
        args.concurrency, time.time() - spawn_start), flush=True)

    if args.sync_start:
        sync_wait_elapsed = wait_sync_ready(
            clients, sync_dir, args.sync_start_timeout)
        extra_info['sync_wait_elapsed'] = '{:.3f}'.format(sync_wait_elapsed)

    results, batch_elapsed = wait_clients(clients)
    summary_path = write_summary(run_log_dir, args, results, batch_elapsed,
                                  extra_info=extra_info)

    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count

    print('=' * 72, flush=True)
    print('[SUMMARY] log_dir={}'.format(run_log_dir), flush=True)
    print('[SUMMARY] summary_log={}'.format(summary_path), flush=True)
    print('[SUMMARY] concurrency={}, success={}, failed={}, batch_wall={:.3f}s'.format(
        args.concurrency, success_count, failed_count, batch_elapsed), flush=True)
    if results:
        elapsed_list = sorted(r['elapsed'] for r in results)
        p95_idx = max(0, min(len(elapsed_list) - 1,
                             int(len(elapsed_list) * 0.95) - 1))
        print('[SUMMARY] avg_client_elapsed={:.3f}s, p95_client_elapsed={:.3f}s'.format(
            sum(elapsed_list) / len(elapsed_list), elapsed_list[p95_idx]),
            flush=True)
    print('=' * 72, flush=True)

    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
