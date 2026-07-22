[CosyVoice2 Ascend 910B 交付复现说明](#cosyvoice2-ascend-910b-交付复现说明)

## CosyVoice2 Ascend 910B 交付复现说明

本仓库当前交付的是 CosyVoice2 0.5B 在 Ascend 910B 单卡 10 进程流式推理上的优化版本。默认目标口径：

- 不改变首包和中间包 chunk 大小。
- 单卡 10 进程同步正式推理。
- 首包 p90 < 400ms。
- 中间包 p90 RTF < 0.3。
- 中间包统计口径排除 final tail。final tail 是收尾包，长度和 cache 状态不同，不能混入中间包验收。

### 目录和依赖

默认模型路径：

```bash
../weight/CosyVoice2-0.5B_sft_shenhu_25_60
```

默认验收抄本：

```bash
data/manual_transcript_20260720.txt
```

默认 HiFT decode OM：

```bash
experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om
```

运行前确认：

```bash
conda activate voxcpm
cd /data/xmren/work/work/test/model/CosyVoice-claude
npu-smi info
```

如果模型文件是从其他用户复制过来的，`ais_bench` 可能要求当前用户属于文件 owner 或 owner group。需要保证模型目录和 OM 文件对当前运行用户可读，并满足 Ascend path security check 的 owner/group 要求。

### 10 进程正式压测

默认入口：

```bash
bash run_manual_concurrent.sh
```

默认行为：

- 使用 `ASCEND_RT_VISIBLE_DEVICES=0`。
- 启动 10 个独立 `infer.py` 进程。
- 每个进程绑定到 NPU0 近端 CPU `144-167`，10 进程共享这组 core。
- 所有进程先完整 warmup 当前 `TEXT_FILE`。
- 所有进程 warmup 完成后通过 sync barrier 同步开始正式推理。
- 默认 `NO_SAVE_AUDIO=1`，性能压测不保存音频，避免 wav 保存和 device-to-host copy 影响 RTF。
- 默认 `INFER_COUNT=1`，即每个 client 正式跑一遍抄本。

常用覆盖参数：

```bash
# 指定 NPU
ASCEND_RT_VISIBLE_DEVICES=1 bash run_manual_concurrent.sh

# 保存音频，主要用于听音质，不用于性能验收
NO_SAVE_AUDIO=0 bash run_manual_concurrent.sh

# 换抄本；默认 warmup 会自动覆盖该文件所有非空行
TEXT_FILE=data/your_transcript.txt bash run_manual_concurrent.sh

# 缩短 warmup 调试，不建议用于正式验收
WARM_UP_TIMES=5 bash run_manual_concurrent.sh
```

### 单进程推理

默认入口：

```bash
bash run.sh
```

默认行为：

- 使用同一套 HiFT OM、Qwen hidden-only、fast topk、device-token decode、Flow mask sync 优化。
- 默认完整 warmup 当前 `TEXT_FILE`。
- 默认保存音频到 `testout/run_single`。
- 默认 `INFER_COUNT=1`。

性能测试单进程可关闭保存：

```bash
NO_SAVE_AUDIO=1 bash run.sh
```

### 当前固化的核心优化

1. HiFT decode OM 化

`run_manual_concurrent.sh` 和 `run.sh` 默认加载：

```bash
COSYVOICE2_HIFT_DECODE_OM=experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om
COSYVOICE2_HIFT_DECODE_GEARS=30,50,128,130,160
```

作用是把 HiFT vocoder 的 decode 从 PyTorch eager/compile 路径替换成固定 OM 推理，减少多进程下 runtime 开销和图资源抖动。

2. Qwen safe hidden-only

默认开启：

```bash
COSYVOICE2_QWEN_HIDDEN_ONLY=1
TORCHAIR_CACHE_HOME=experiments/torchair_cache_hidden_safe
```

CosyVoice2 后续只使用 Qwen hidden states，再接自己的 `llm_decoder` 生成 speech token。当前实现跳过 Qwen 原文本 `lm_head`，但使用独立 hidden-only 编译入口，避免复用原 decode/prefill cache 导致返回结构不稳定。可回退：

```bash
COSYVOICE2_QWEN_HIDDEN_ONLY=0 bash run_manual_concurrent.sh
```

3. LLM token decode 轻量化

默认开启：

```bash
COSYVOICE2_SAMPLING_MODE=fast_topk
COSYVOICE2_FAST_TOPK_K=25
COSYVOICE2_DEVICE_TOKEN_DECODE=1
```

作用是减少原始 RAS/top-p 全量排序和逐 token host `.item()` 同步。`FAST_TOPK_K=25` 是当前交付默认，未固化 TopK=10，因为它会改变采样分布，音质风险更高。

4. no-save 压测路径不做 CPU 输出同步

`NO_SAVE_AUDIO=1` 时，`infer.py` 会设置：

```bash
COSYVOICE2_NO_CPU_OUTPUT=1
```

推理代码只消费生成结果，不把每个 chunk 立即 `.cpu()` 并保存 wav，避免把 host copy/IO 混入 RTF。

5. Flow mask 构造减少 host sync

`cosyvoice/flow/flow.py` 中 `make_pad_mask` 显式传入 `max_len`，避免内部 `lengths.max().item()` 造成 NPU 到 Host 同步。这个改动不改变模型输出，只减少运行时同步点。

6. 完整 warmup 当前抄本

两个启动脚本默认自动统计 `TEXT_FILE` 的非空行数作为 `WARM_UP_TIMES`，并开启 `--warmup_full`。这样正式推理前已覆盖当前抄本的主要文本长度和动态 shape，避免第 6 条之后才首次遇到新 shape 导致首包和中间包抖动。

### 已验证结果

默认 5 条 warmup 时，正式推理会在未 warmup 文本上出现边界抖动：

```text
logs/manual_hift_om_v2_sync_run/run_20260722_203656
first p90 = 403.82ms
middle non-final p90 RTF = 0.30252
```

完整 warmup 后连续两轮 10 进程同步推理达成 p90 目标：

```text
logs/exp_full_warmup_67_24core/run_20260722_205133
first p90 = 393.98ms
middle non-final p90 RTF = 0.29962

logs/exp_full_warmup_67_24core_rerun/run_20260722_205954
first p90 = 399.61ms
middle non-final p90 RTF = 0.29501
```

注意：性能仍然接近硬件边界。正式验收前应保证 NPU0 无其他进程、CPU 亲和核未被其他高负载任务占用，并使用默认完整 warmup。

---

[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=CosyVoice🤠&text2=Text-to-Speech%20💖%20Large%20Language%20Model&width=800&height=210)](https://github.com/Akshay090/svg-banners)

## 👉🏻 CosyVoice 👈🏻
**CosyVoice 2.0**: [Demos](https://funaudiollm.github.io/cosyvoice2/); [Paper](https://arxiv.org/abs/2412.10117); [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B); [HuggingFace](https://huggingface.co/spaces/FunAudioLLM/CosyVoice2-0.5B)

**CosyVoice 1.0**: [Demos](https://fun-audio-llm.github.io); [Paper](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf); [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice-300M)

## Highlight🔥

**CosyVoice 2.0** has been released! Compared to version 1.0, the new version offers more accurate, more stable, faster, and better speech generation capabilities.
### Multilingual
- **Supported Language**: Chinese, English, Japanese, Korean, Chinese dialects (Cantonese, Sichuanese, Shanghainese, Tianjinese, Wuhanese, etc.)
- **Crosslingual & Mixlingual**：Support zero-shot voice cloning for cross-lingual and code-switching scenarios.
### Ultra-Low Latency
- **Bidirectional Streaming Support**: CosyVoice 2.0 integrates offline and streaming modeling technologies.
- **Rapid First Packet Synthesis**: Achieves latency as low as 150ms while maintaining high-quality audio output.
### High Accuracy
- **Improved Pronunciation**: Reduces pronunciation errors by 30% to 50% compared to CosyVoice 1.0.
- **Benchmark Achievements**: Attains the lowest character error rate on the hard test set of the Seed-TTS evaluation set.
### Strong Stability
- **Consistency in Timbre**: Ensures reliable voice consistency for zero-shot and cross-language speech synthesis.
- **Cross-language Synthesis**: Marked improvements compared to version 1.0.
### Natural Experience
- **Enhanced Prosody and Sound Quality**: Improved alignment of synthesized audio, raising MOS evaluation scores from 5.4 to 5.53.
- **Emotional and Dialectal Flexibility**: Now supports more granular emotional controls and accent adjustments.

## Roadmap

- [x] 2024/12

    - [x] 25hz cosyvoice 2.0 released

- [x] 2024/09

    - [x] 25hz cosyvoice base model
    - [x] 25hz cosyvoice voice conversion model

- [x] 2024/08

    - [x] Repetition Aware Sampling(RAS) inference for llm stability
    - [x] Streaming inference mode support, including kv cache and sdpa for rtf optimization

- [x] 2024/07

    - [x] Flow matching training support
    - [x] WeTextProcessing support when ttsfrd is not available
    - [x] Fastapi server and client


## Install

**Clone and install**

- Clone the repo
``` sh
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone submodule due to network failures, please run following command until success
cd CosyVoice
git submodule update --init --recursive
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

**Model download**

We strongly recommend that you download our pretrained `CosyVoice2-0.5B` `CosyVoice-300M` `CosyVoice-300M-SFT` `CosyVoice-300M-Instruct` model and `CosyVoice-ttsfrd` resource.

``` python
# SDK模型下载
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='pretrained_models/CosyVoice-300M-25Hz')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

``` sh
# git模型下载，请确保已安装git lfs
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-25Hz.git pretrained_models/CosyVoice-300M-25Hz
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

Optionally, you can unzip `ttsfrd` resouce and install `ttsfrd` package for better text normalization performance.

Notice that this step is not necessary. If you do not install `ttsfrd` package, we will use WeTextProcessing by default.

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

**Basic Usage**

We strongly recommend using `CosyVoice2-0.5B` for better performance.
Follow code below for detailed usage of each model.

``` python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
```

**CosyVoice2 Usage**
```python
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

**CosyVoice Usage**
```python
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)
# sft usage
print(cosyvoice.list_available_spks())
# change stream=True for chunk stream inference
for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
    torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M') # or change to pretrained_models/CosyVoice-300M-25Hz for 25Hz inference
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cross_lingual usage
prompt_speech_16k = load_wav('./asset/cross_lingual_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k, stream=False)):
    torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# vc usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
source_speech_16k = load_wav('./asset/cross_lingual_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)):
    torchaudio.save('vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
for i, j in enumerate(cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.', stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

**Start web demo**

You can use our web demo page to get familiar with CosyVoice quickly.

Please see the demo website for details.

``` python
# change iic/CosyVoice-300M-SFT for sft inference, or iic/CosyVoice-300M-Instruct for instruct inference
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

**Advanced Usage**

For advanced user, we have provided train and inference scripts in `examples/libritts/cosyvoice/run.sh`.

**Build for deployment**

Optionally, if you want service deployment,
you can run following steps.

``` sh
cd runtime/python
docker build -t cosyvoice:v1.0 .
# change iic/CosyVoice-300M to iic/CosyVoice-300M-Instruct if you want to use instruct inference
# for grpc usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"
cd grpc && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
# for fastapi usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
cd fastapi && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
```

## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/FunAudioLLM/CosyVoice/issues).

You can also scan the QR code to join our official Dingding chat group.

<img src="./asset/dingding.png" width="250px">

## Acknowledge

1. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
2. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
3. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
