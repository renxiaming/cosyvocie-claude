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
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)
    
    parser = argparse.ArgumentParser(description="CosyVoice infer")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument('--warm_up_times', default=5, type=int, help='warm up times')
    parser.add_argument('--infer_count', default=5, type=int, help='infer loop count')
    parser.add_argument('--output_dir', default='/home/ma-user/work/test/model/CosyVoice-claude/testout/demo2',
                        type=str, help='output dir')
    parser.add_argument('--stream', action="store_true", help='stream infer')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cosyvoice = CosyVoice2(args.model_path, load_om=True, fp16=True)
    cosyvoice.model.llm.eval()
    cosyvoice.model.llm.llm.model.model.half()

    # 对hift模型结构进行torchair图模式适配
    cosyvoice.model.hift.remove_weight_norm() #删除推理过程中的weight_norm
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.hift.decode = torch.compile(cosyvoice.model.hift.decode, dynamic=True, fullgraph=True, backend=npu_backend)


    # 输入数据加载
    prompt_texts = [
        '是的，您现在还有大概1个G的流量。',
        '不全是通用的哦，里面有800兆是通用流量，还有900兆是定向流量。',
        '查到了，您现在的通用流量还剩800兆，定向流量还剩900兆。',
        '好的，稍后如果您收到评价短信，麻烦您对我的服务做出评价，感谢您的来电，祝您生活愉快，再见！',
        '您好，中国移动，很高兴为您服务。请问有什么可以帮您？',
        '好的，请问您是要为当前拨打的这个号码办理吗？另外需要和您核实一下，机主是您本人吗？',
        '好的。这款“青春畅想5G套餐”主要是针对年轻用户的专属优惠，月费59元，包含30G通用流量、30G定向流量和100分钟语音通话。您看这个流量够您平时使用吗？',
    ]

    with torch.no_grad():
        # import ipdb;ipdb.set_trace()
        print('warm up start')
        for _ in range(args.warm_up_times):
            next(cosyvoice.inference_sft(prompt_texts[0], '03729', stream=args.stream))
        print('warm up end')
        # import ipdb;ipdb.set_trace()
        for infer_idx in range(args.infer_count):
            for text_idx, prompt_txt in enumerate(prompt_texts):
                print('[INFO] infer round {}, text {}: {}'.format(infer_idx, text_idx, prompt_txt))
                speech_chunks = []
                for _, j in enumerate(cosyvoice.inference_sft(prompt_txt, '03729', stream=args.stream)):
                    speech_chunks.append(j['tts_speech'])
                if speech_chunks:
                    full_speech = torch.cat(speech_chunks, dim=1)
                    output_path = os.path.join(args.output_dir, 'sft_full_{}_{}.wav'.format(infer_idx, text_idx))
                    torchaudio.save(output_path, full_speech, cosyvoice.sample_rate)
                    print('[INFO] save full speech to {}'.format(output_path))
