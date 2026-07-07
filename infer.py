# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import os
import sys
import time
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


def apply_cpu_affinity():
    """从环境变量读取 CPU 亲和性范围并应用"""
    if 'CPU_AFFINITY_START' in os.environ and 'CPU_AFFINITY_END' in os.environ:
        try:
            start = int(os.environ['CPU_AFFINITY_START'])
            end = int(os.environ['CPU_AFFINITY_END'])
            os.sched_setaffinity(0, range(start, end + 1))
            client_id = os.environ.get('CPU_AFFINITY_CLIENT_ID', '?')
            print('[INFO] CPU affinity set: cores {}-{} (client_{})'.format(
                start, end, client_id), flush=True)
        except Exception as e:
            print('[WARN] Failed to set CPU affinity: {}'.format(e), flush=True)


def configure_threads():
    """限制 PyTorch CPU 线程数，减少多进程下 CPU 竞争"""
    omp_threads = int(os.environ.get('OMP_NUM_THREADS', 8))
    mkl_threads = int(os.environ.get('MKL_NUM_THREADS', 8))

    # 设置环境变量（对底层 BLAS / MKL 生效）
    os.environ.setdefault('OMP_NUM_THREADS', str(omp_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(mkl_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(mkl_threads))
    os.environ.setdefault('NUMEXPR_NUM_THREADS', str(mkl_threads))

    # 设置 PyTorch 原生线程数
    torch.set_num_threads(omp_threads)
    try:
        torch.set_num_interop_threads(min(4, omp_threads))
    except Exception:
        pass  # 某些 torch 版本可能不支持

    print('[INFO] CPU threads configured: OMP={}, MKL={}, torch={}'.format(
        omp_threads, mkl_threads, torch.get_num_threads()), flush=True)


def wait_sync_start(sync_dir, client_id, timeout):
    if not sync_dir:
        return
    os.makedirs(sync_dir, exist_ok=True)
    ready_path = os.path.join(sync_dir, 'client_{}.ready'.format(client_id))
    go_path = os.path.join(sync_dir, 'go')
    with open(ready_path, 'w', buffering=1) as f:
        f.write('{}\n'.format(os.getpid()))
    print('[SYNC] client_{} ready, waiting for go'.format(client_id),
          flush=True)
    start = time.time()
    while not os.path.exists(go_path):
        if timeout > 0 and time.time() - start > timeout:
            raise TimeoutError(
                'sync start timeout after {:.1f}s'.format(timeout))
        time.sleep(0.05)
    print('[SYNC] client_{} go, wait={:.3f}s'.format(
        client_id, time.time() - start), flush=True)


if __name__ == '__main__':
    # ================================================================
    # 启动时立即设置 CPU 亲和性和线程数（在其他 import 之前尽可能早）
    # ================================================================
    apply_cpu_affinity()
    configure_threads()

    # print("go go go!")
    # torch.set_num_threads(8)
    torch_npu.npu.set_compile_mode(jit_compile=False)

    parser = argparse.ArgumentParser(description="CosyVoice infer")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument('--warm_up_times', default=5, type=int,
                        help='warm up times')
    parser.add_argument('--infer_count', default=5, type=int,
                        help='infer loop count')
    parser.add_argument('--output_dir',
                        default='/home/ma-user/work/test/model/CosyVoice-claude/testout/demo11',
                        type=str, help='output dir')
    parser.add_argument('--stream', action="store_true", help='stream infer')
    parser.add_argument('--no_save_audio', action="store_true",
                        help='consume inference output without writing wav files')
    parser.add_argument('--sync_start_dir', default='', type=str,
                        help='directory used as formal inference start barrier')
    parser.add_argument('--sync_start_timeout', default=900.0, type=float,
                        help='seconds to wait for formal inference start barrier')
    args = parser.parse_args()

    # 检查是否跳过 warmup（由 infer_manual_concurrent 的 serial_warmup 模式控制）
    skip_warmup = os.environ.get('SKIP_WARMUP', '0') == '1'
    client_id = os.environ.get('CPU_AFFINITY_CLIENT_ID', 'main')

    os.makedirs(args.output_dir, exist_ok=True)
    print('[INFO] client_id={} skip_warmup={} '.format(client_id, skip_warmup),
          flush=True)

    # ---- 模型加载 ----
    print('[INFO] loading model...', flush=True)
    cosyvoice = CosyVoice2(args.model_path, load_om=True, fp16=True)
    cosyvoice.model.llm.eval()
    cosyvoice.model.llm.llm.model.model.half()

    # 对hift模型结构进行torchair图模式适配
    cosyvoice.model.hift.remove_weight_norm()  # 删除推理过程中的weight_norm
    # config = CompilerConfig()
    # config.experimental_config.frozen_parameter = True
    # config.experimental_config.tiling_schedule_optimize = True
    # npu_backend = tng.get_npu_backend(compiler_config=config)
    # cosyvoice.model.hift.decode = torch.compile(
    #     cosyvoice.model.hift.decode, dynamic=True, fullgraph=True,
    #     backend=npu_backend)

    # 输入数据加载
    prompt_texts = [
        '是的，您现在还有大概1个G的流量。',
        '不全是通用的哦，里面有800兆是通用流量，还有900兆是定向流量。',
        '查到了，您现在的通用流量还剩800兆，定向流量还剩900兆。',
        '好的，稍后如果您收到评价短信，麻烦您对我的服务做出评价，感谢您的来电，祝您生活愉快，再见！',
        '您好，中国移动，很高兴为您服务。请问有什么可以帮您？',
        '好的，请问您是要为当前拨打的这个号码办理吗？另外需要和您核实一下，机主是您本人吗？',
        '好的。这款"青春畅想5G套餐"主要是针对年轻用户的专属优惠，月费59元，包含30G通用流量、30G定向流量和100分钟语音通话。您看这个流量够您平时使用吗？',
    ]

    with torch.no_grad():
        if args.warm_up_times > 0 and not skip_warmup:
            print('warm up start', flush=True)
            warmup_start = time.time()
            for warmup_idx in range(args.warm_up_times):
                warmup_text = prompt_texts[warmup_idx % len(prompt_texts)]
                for _ in cosyvoice.inference_sft(warmup_text, '03729',
                                                 stream=args.stream):
                    pass
            print('warm up end, elapsed={:.1f}s'.format(
                time.time() - warmup_start), flush=True)

        # 如果 infer_count=0 表示仅 warmup，直接返回
        if args.infer_count <= 0:
            print('[INFO] infer_count=0, exiting after warmup', flush=True)
            sys.exit(0)

        wait_sync_start(args.sync_start_dir, client_id,
                        args.sync_start_timeout)

        for infer_idx in range(args.infer_count):
            for text_idx, prompt_txt in enumerate(prompt_texts):
                print('[INFO] infer round {}, text {}: {}'.format(
                    infer_idx, text_idx, prompt_txt))
                if args.no_save_audio:
                    for _ in cosyvoice.inference_sft(
                            prompt_txt, '03729', stream=args.stream):
                        pass
                else:
                    speech_chunks = []
                    for _, j in enumerate(cosyvoice.inference_sft(
                            prompt_txt, '03729', stream=args.stream)):
                        speech_chunks.append(j['tts_speech'])
                    if speech_chunks:
                        full_speech = torch.cat(speech_chunks, dim=1)
                        output_path = os.path.join(
                            args.output_dir,
                            'sft_full_{}_{}.wav'.format(infer_idx, text_idx))
                        torchaudio.save(output_path, full_speech,
                                        cosyvoice.sample_rate)
                        print('[INFO] save full speech to {}'.format(output_path))
