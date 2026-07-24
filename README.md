# CosyVoice2 Ascend 910B 推理加速交付说明

本仓库是在开源 [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) 基础上，面向 **Ascend 910B aarch64** 的单卡 **10 进程流式推理**交付版本。文档覆盖环境配置、权重准备、代码改动、运行方式与验收口径。

## 交付目标

- 业务 chunk 配置不变：首包 token `25`，中间包 hop `60`。
- 单张 Ascend 910B，10 个独立进程同步正式推理。
- 正式推理首包 p90 `< 400ms`。
- 正式推理中间包 p90 RTF `< 0.3`。
- 中间包统计排除 final tail（每句话最后一个 chunk）。

## 交付清单

接收方至少需要以下内容：

| 类别 | 内容 | 位置 |
|---|---|---|
| 代码 | 本仓库 + `transformers` 子模块 | 当前目录 |
| 业务权重 | SFT 模型目录 | 默认 `../weight/CosyVoice2-0.5B_sft_shenhu_25_60` |
| HiFT decode OM | 已编译 OM + 导出 ONNX | `experiments/hift_decode_om_20260706_230701/` |
| 验收抄本 | 67 行业务文本 | `data/manual_transcript_20260720.txt` |
| 运行环境 | CANN + torch/torch_npu + conda 依赖 | 见第 1 节 |

仓库 **不包含** 数 GB 的业务权重；Flow/speech OM 在权重目录中，需单独准备。

## 1. 环境与依赖

### 1.1 硬件与系统

- 芯片：Ascend 910B（aarch64）
- 操作系统：Linux aarch64
- 驱动：与 CANN 版本匹配的 Ascend 驱动
- 建议：整卡推理，不要对目标 NPU 做 vNPU 切分

### 1.2 CANN 与 Python 环境

当前脚本按 **CANN 8.1.RC1** 验证，`run.sh` / `run_manual_concurrent.sh` 会自动执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

Python 侧建议使用 conda 环境（验证环境名：`voxcpm`，Python 3.11）：

```bash
conda create -n voxcpm python=3.11 -y
conda activate voxcpm
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt
```

还需要安装与 CANN 匹配的 **torch / torch_npu**（不在 `requirements.txt` 中，需按 Ascend 官方 wheel 安装）。安装完成后确认：

```bash
python3 -c "import torch; import torch_npu; print(torch.__version__)"
npu-smi info
```

### 1.3 获取代码

```bash
git clone --recursive https://github.com/renxiaming/cosyvocie-claude.git CosyVoice-claude
cd CosyVoice-claude
git submodule update --init --recursive
```

如果 `transformers` 子模块未拉取，Qwen hidden-only 路径无法复现。也可按第 7 节用 `docs/transformers_replacement/` 中的文件手动替换。

### 1.4 权重目录

默认模型路径：

```bash
export MODEL_PATH=/path/to/CosyVoice2-0.5B_sft_shenhu_25_60
```

该目录必须包含第 4 节列出的 `.pt` / `.om` / `.onnx` 文件。默认脚本通过 `MODEL_PATH` 或相对路径 `../weight/CosyVoice2-0.5B_sft_shenhu_25_60` 加载。

默认 SFT 音色：

```bash
SFT_SPK_ID=03729
```

该 speaker 必须已写入 `$MODEL_PATH/spk2info.pt`，注册方法见第 4.2 节。

## 2. 快速开始

```bash
conda activate voxcpm
cd /path/to/CosyVoice-claude
export MODEL_PATH=/path/to/CosyVoice2-0.5B_sft_shenhu_25_60
npu-smi info
```

单进程推理并保存音频：

```bash
bash run.sh
```

10 进程正式压测（默认 NPU0，脚本会自动绑定近端 CPU）：

```bash
bash run_manual_concurrent.sh
```

推荐验收配置（当前最优组合：NPU2 + CPU 144-167）：

```bash
ASCEND_RT_VISIBLE_DEVICES=2 bash run_manual_concurrent.sh
```

常用覆盖：

```bash
# 指定 NPU
ASCEND_RT_VISIBLE_DEVICES=0 bash run_manual_concurrent.sh

# 多进程保存音频，只用于听音质，不用于性能验收
NO_SAVE_AUDIO=0 bash run_manual_concurrent.sh

# 使用指定 SFT 音色
SFT_SPK_ID=03729 bash run_manual_concurrent.sh

# 换抄本；默认会对新抄本做完整 warmup
TEXT_FILE=data/your_transcript.txt bash run_manual_concurrent.sh

# 调试时缩短 warmup；正式验收不要这样做
WARM_UP_TIMES=5 bash run_manual_concurrent.sh
```

默认 HiFT decode OM：

```bash
experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om
```

Docker 打包见 `docker/README_ascend.md`。

## 3. 从开源代码到当前版本的整体步骤

当前版本不是纯开源 CosyVoice2 直接运行，而是在开源结构上做了 Ascend 910B 推理适配和多进程低延迟优化。

1. 获取开源 CosyVoice2 代码。

   基础结构保持 FunAudioLLM/CosyVoice 的目录组织，包括 `cosyvoice/`、`third_party/Matcha-TTS/`、`runtime/`、`examples/` 等。

2. 使用本地 Ascend 适配过的 `transformers` 子仓库。

   主仓库中的 `transformers` 是 gitlink/submodule 形式。当前指针：

   ```text
   transformers -> e352348b1442775fda3d6faf2aa716d6dd581ff5
   ```

   相比早期可跑通版本 `05e48ed` 使用的指针：

   ```text
   transformers -> 6581349d4be9fb6853d7b8b1fc204883c10f19db
   ```

   当前又增加了 hidden-only runner。具体修改见第 6 节。

3. 准备业务模型目录。

   模型目录默认放在仓库外：

   ```bash
   ../weight/CosyVoice2-0.5B_sft_shenhu_25_60
   ```

   大模型权重、Flow OM、speech token OM 等文件不放进本仓库，避免把数 GB 模型权重提交到代码仓库。

4. 保留 Flow 和 speech token 的原有 OM 推理。

   `CosyVoice2(..., load_om=True)` 会加载模型目录下的：

   ```text
   flow_linux_aarch64.om
   flow_static.om
   speech_linux_aarch64.om
   ```

   这部分沿用 Ascend 适配版本的 OM 路径。

5. 额外把 HiFT vocoder 的 `decode` 子图导出并编译成 OM。

   该文件放在本仓库：

   ```text
   experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om
   ```

   代码运行时用环境变量加载，替换 PyTorch eager/torch.compile 的 HiFT decode 路径。

6. 改造 Qwen LLM 推理路径。

   CosyVoice2 的 Qwen 输出只需要 hidden states，后面接 CosyVoice 自己的 `llm_decoder` 生成 speech token；Qwen 原文本 `lm_head` 对 TTS 推理没有实际用途。当前版本增加 safe hidden-only runner，跳过 Qwen `lm_head`，但保持独立 torchair 编译入口，避免早期 hidden-only 复用编译缓存导致音频杂音。

7. 增加严格同步、多进程 CPU 亲和、完整 warmup 和 no-save 压测路径。

   默认入口 `run_manual_concurrent.sh` 会启动 10 个独立进程，各自完整 warmup 当前抄本，然后统一进入正式推理，避免“部分进程先跑完，部分进程还没开始”的假低 RTF。

## 4. 模型文件处理说明

默认外部模型目录中的核心文件如下：

| 文件 | 当前处理方式 | 是否提交本仓库 | 用途 |
|---|---|---:|---|
| `llm.pt` | 保留在外部模型目录 | 否 | CosyVoice LLM 权重 |
| `flow.pt` | 保留在外部模型目录 | 否 | Flow PyTorch 权重，主要用于构建模型对象 |
| `hift.pt` | 保留在外部模型目录 | 否 | HiFT PyTorch 权重，除 decode OM 外仍需模型结构和部分逻辑 |
| `spk2info.pt` / `spk2info1.pt` | 保留在外部模型目录 | 否 | SFT 音色信息 |
| `flow_linux_aarch64.om` | 运行时通过 `load_om=True` 加载 | 否 | Flow 动态 OM |
| `flow_static.om` | 运行时通过 `load_om=True` 加载 | 否 | Flow 静态 OM |
| `speech_linux_aarch64.om` | 运行时通过 `load_om=True` 加载 | 否 | speech tokenizer/token 模块 OM |
| `speech_tokenizer_v2.onnx` / `speech_token_md.onnx` | 保留在外部模型目录 | 否 | speech token 相关 ONNX 文件 |
| `flow.decoder.estimator.fp32.onnx` | 保留在外部模型目录 | 否 | Flow estimator ONNX 源文件 |
| `CosyVoice-BlankEN/model.safetensors` | 保留在外部模型目录 | 否 | Qwen/HF 模型相关文件 |
| `vllm/model.safetensors` | 保留在外部模型目录 | 否 | 早期 vLLM/MindIE 方向探索产物，当前默认推理不使用 |
| `experiments/hift_decode_om_20260706_230701/hift.decode.fp32.onnx` | 本仓库跟踪 | 是 | HiFT decode 导出的 ONNX |
| `experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om` | 本仓库跟踪 | 是 | 当前默认使用的 HiFT decode OM |
| `experiments/hift_decode_om_20260706_230701/real_hift_decode_shapes.txt` | 本仓库跟踪 | 是 | 实测 HiFT decode shape/gear 记录 |

注意：

- 外部模型目录必须存在，默认脚本不会自动下载或复制权重。
- `ais_bench` 会做路径安全检查。如果 OM 文件不是当前用户或用户组可访问，可能报 owner/ownergroup 相关错误。需要保证模型文件和目录 owner/group 满足当前运行用户要求。
- 本仓库 `.gitignore` 明确忽略新生成的运行日志、NPU 编译缓存、profiling 产物和临时权重，避免交付代码被本机产物污染。

### 4.1 从开源权重处理成当前可运行模型目录

如果别人只有开源 `CosyVoice2-0.5B` 权重，不能直接跑当前 10 进程脚本。需要先把权重目录处理成当前代码要求的 Ascend 推理目录。

假设：

```bash
export MODEL_DIR=/path/to/CosyVoice2-0.5B_sft_shenhu_25_60
export SOC_VERSION=Ascend910B1
```

最终目录至少需要包含：

```text
$MODEL_DIR/llm.pt
$MODEL_DIR/flow.pt
$MODEL_DIR/hift.pt
$MODEL_DIR/spk2info.pt
$MODEL_DIR/flow.decoder.estimator.fp32.onnx
$MODEL_DIR/flow_linux_aarch64.om
$MODEL_DIR/flow_static.om
$MODEL_DIR/speech_tokenizer_v2.onnx
$MODEL_DIR/speech_linux_aarch64.om
```

当前默认脚本还会使用仓库内的：

```text
experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om
```

如果开源权重中的 `hift.pt` 和当前交付模型不一致，建议按第 3.6 节重新导出并编译 HiFT decode OM。

### 4.2 注册 SFT speaker 音色

SFT 模式通过 `spk2info.pt` 查音色 embedding。当前推理默认：

```bash
SFT_SPK_ID=03729
```

所以模型目录里必须存在：

```python
spk2info["03729"]["embedding"]
```

注册方式：

```bash
python3 register_wav.py \
  --model_dir "$MODEL_DIR" \
  --spk 03729=/path/to/03729_16k.wav \
  --overwrite
```

也可以一次注册多个音色：

```bash
python3 register_wav.py \
  --model_dir "$MODEL_DIR" \
  --spk 03729=/path/to/03729_16k.wav \
  --spk shenhu=/path/to/shenhu_prompt_16k.wav \
  --overwrite
```

要求：

- 输入 wav 建议是单声道、16k 采样率、干净人声。
- 音频内容建议 3-30 秒，背景噪声越少越好。
- `register_wav.py` 会调用 `CosyVoice2.frontend._extract_spk_embedding()`，使用 `campplus.onnx` 抽取 speaker embedding，然后写入 `$MODEL_DIR/spk2info.pt`。

注册完成后可以用指定音色推理：

```bash
SFT_SPK_ID=03729 MODEL_PATH="$MODEL_DIR" bash run.sh
SFT_SPK_ID=03729 MODEL_PATH="$MODEL_DIR" bash run_manual_concurrent.sh
```

如果换成自己的音色，例如 `customer_a`：

```bash
python3 register_wav.py \
  --model_dir "$MODEL_DIR" \
  --spk customer_a=/path/to/customer_a_16k.wav

SFT_SPK_ID=customer_a MODEL_PATH="$MODEL_DIR" bash run.sh
```

### 4.3 导出 Flow estimator ONNX

仓库提供了参数化脚本：

```bash
MODEL_DIR="$MODEL_DIR" bash export.sh
```

等价于：

```bash
export PYTHONPATH=third_party/Matcha-TTS:transformers/src:${PYTHONPATH:-}
python3 cosyvoice/bin/export_onnx.py --model_dir "$MODEL_DIR"
```

导出参数在 `cosyvoice/bin/export_onnx.py` 中固定：

```text
opset_version=18
input_names=x,mask,mu,t,spks,cond
output_names=estimator_out
dynamic_axes:
  x/mask/mu/cond/estimator_out 的第 2 维为 seq_len
dummy input:
  x     [2, 80, seq_len]
  mask  [2, 1, seq_len]
  mu    [2, 80, seq_len]
  t     [2]
  spks  [2, 80]
  cond  [2, 80, seq_len]
```

输出：

```text
$MODEL_DIR/flow.decoder.estimator.fp32.onnx
```

### 4.4 编译 Flow OM

当前代码会优先使用 `flow_static.om`，并按 `COSYVOICE2_FLOW_GEARS` 手动 padding 到固定档位；没有命中时才退回动态 `flow_linux_aarch64.om`。

当前默认 Flow gear：

```bash
COSYVOICE2_FLOW_GEARS=50,74,98,122,146,170,200,230,260,290,320,350,380,410
```

ATC 编译模板如下。不同 CANN 版本对 `--dynamic_dims`/动态 shape 参数格式可能有差异，核心要求是输入名和 shape 必须与 ONNX 一致，动态维度必须覆盖上面的 gear。

```bash
atc \
  --framework=5 \
  --model="$MODEL_DIR/flow.decoder.estimator.fp32.onnx" \
  --output="$MODEL_DIR/flow_static" \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --input_shape="x:2,80,-1;mask:2,1,-1;mu:2,80,-1;t:2;spks:2,80;cond:2,80,-1" \
  --dynamic_dims="50;74;98;122;146;170;200;230;260;290;320;350;380;410" \
  --precision_mode=force_fp16
```

输出文件应为：

```text
$MODEL_DIR/flow_static.om
```

动态 Flow OM 可按同一个 ONNX 编译，输出名需要符合代码加载约定：

```bash
atc \
  --framework=5 \
  --model="$MODEL_DIR/flow.decoder.estimator.fp32.onnx" \
  --output="$MODEL_DIR/flow_linux_aarch64" \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --input_shape="x:2,80,-1;mask:2,1,-1;mu:2,80,-1;t:2;spks:2,80;cond:2,80,-1" \
  --precision_mode=force_fp16
```

输出文件应为：

```text
$MODEL_DIR/flow_linux_aarch64.om
```

### 4.5 编译 speech tokenizer OM

当前前端 `_extract_speech_token()` 期望 speech OM 输入为：

```text
feats        [1, 128, T] float32
feats_length [1] int64
```

输出：

```text
indices int64
```

ATC 编译模板：

```bash
atc \
  --framework=5 \
  --model="$MODEL_DIR/speech_tokenizer_v2.onnx" \
  --output="$MODEL_DIR/speech_linux_aarch64" \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --input_shape="feats:1,128,-1;feats_length:1" \
  --precision_mode=force_fp16
```

输出文件应为：

```text
$MODEL_DIR/speech_linux_aarch64.om
```

### 4.6 重新导出并编译 HiFT decode OM

如果使用的 `hift.pt` 和当前交付模型不同，重新导出 HiFT decode ONNX：

```bash
python3 tools/export_hift_decode_onnx.py \
  --model_dir "$MODEL_DIR" \
  --output experiments/hift_decode_om_20260706_230701/hift.decode.fp32.onnx \
  --mel_len 50 \
  --opset 18
```

导出参数：

```text
input_names=x,s_stft
output_names=magnitude,phase
dynamic_axes:
  x 第 2 维为 mel_len
  s_stft/magnitude/phase 第 2 维为 stft_len
dummy input:
  x      [1, 80, mel_len]
  s_stft [1, 18, 120 * mel_len + 1]
```

当前默认 HiFT gear：

```bash
COSYVOICE2_HIFT_DECODE_GEARS=30,50,128,130,160
```

ATC 编译模板：

```bash
atc \
  --framework=5 \
  --model=experiments/hift_decode_om_20260706_230701/hift.decode.fp32.onnx \
  --output=experiments/hift_decode_om_20260706_230701/hift_decode_static_v2 \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --input_shape="x:1,80,-1;s_stft:1,18,-1" \
  --dynamic_dims="30,3601;50,6001;128,15361;130,15601;160,19201" \
  --precision_mode=force_fp16
```

这里 `stft_len = 120 * mel_len + 1`。如果新增 HiFT gear，需要同时更新：

```bash
COSYVOICE2_HIFT_DECODE_GEARS
```

以及 ATC 的动态档位。

## 5. HiFT decode 导出 OM 和运行时接入

HiFT 原始路径在 `cosyvoice/hifigan/generator.py` 中：

```python
magnitude, phase = self.decode(x=speech_feat, s_stft=s_stft, index=index)
```

当前处理方式：

1. 从真实流式推理中采集 HiFT decode 输入 shape。

   记录文件：

   ```text
   experiments/hift_decode_om_20260706_230701/real_hift_decode_shapes.txt
   ```

   当前记录覆盖的 mel 长度包括：

   ```text
   30, 50, 128
   ```

   运行时脚本配置的 gear：

   ```bash
   COSYVOICE2_HIFT_DECODE_GEARS=30,50,128,130,160
   ```

2. 导出 `HiFTGenerator.decode` 为 ONNX。

   交付文件：

   ```text
   experiments/hift_decode_om_20260706_230701/hift.decode.fp32.onnx
   ```

3. 通过 ATC 编译 OM。

   当前默认使用：

   ```text
   experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om
   ```

4. 在 `CosyVoice2` 初始化时加载 HiFT OM。

   文件：

   ```text
   cosyvoice/cli/cosyvoice.py
   ```

   逻辑：

   ```python
   hift_decode_om = os.environ.get('COSYVOICE2_HIFT_DECODE_OM', '')
   if hift_decode_om:
       self.model.hift.load_decode_om(hift_decode_om, restore_session=flow_om)
   ```

5. 在 `HiFTGenerator` 内部实现 OM 推理和 fallback。

   文件：

   ```text
   cosyvoice/hifigan/generator.py
   ```

   新增逻辑：

   - `load_decode_om()`：用 `ais_bench.InferSession` 加载 OM，并读取 `COSYVOICE2_HIFT_DECODE_GEARS`。
   - `decode_by_om()`：把输入按 gear padding，执行 OM 推理，再裁回原始 `s_stft` 长度。
   - `forward()`：如果 `decode_om is not None`，走 OM；否则回退原 PyTorch `decode()`。
   - OM 推理后调用 `restore_session.set_context()`，把上下文切回 Flow OM，避免多个 `InferSession` 混用时上下文错乱。

当前限制：

- 如果实际 mel 长度超过 `COSYVOICE2_HIFT_DECODE_GEARS` 覆盖范围，会抛出 `no hift decode om gear`。
- 当前 chunk 配置下已覆盖默认流式推理档位。换 chunk 或换 HiFT 结构，需要重新采集 shape 并补导出/补编译。

## 6. CosyVoice 主仓库代码改动

### 6.1 启动脚本

多进程入口：

```text
run_manual_concurrent.sh
```

默认配置：

```bash
ASCEND_RT_VISIBLE_DEVICES=0
CONCURRENCY=10
INFER_COUNT=1
NO_SAVE_AUDIO=1
SYNC_START=1
SYNC_START_TIMEOUT=1800
CPU_AFFINITY_CPUS=144-167
CPU_AFFINITY_SHARE=1
OMP_NUM_THREADS=2
MKL_NUM_THREADS=1
COSYVOICE2_FIRST_CHUNK_SIZE=25
COSYVOICE2_TOKEN_HOP_LEN=60
COSYVOICE2_FLOW_CONTEXT_TOKENS=25
COSYVOICE2_FLOW_HIFT_STREAM=0
```

脚本会自动统计 `TEXT_FILE` 非空行数作为默认 `WARM_UP_TIMES`。当前交付抄本 67 行，所以默认完整 warmup 67 次。

单进程入口：

```text
run.sh
```

单进程同样使用 HiFT OM、Qwen hidden-only、fast topk、device-token decode、完整 warmup。区别是单进程默认 `NO_SAVE_AUDIO=0`，会保存音频到 `testout/run_single`。

### 6.2 并发调度和同步

文件：

```text
infer_manual_concurrent.py
infer.py
```

关键改动：

- `infer_manual_concurrent.py` 负责启动 N 个独立 `infer.py` 子进程，模拟 N 个终端独立推理。
- 支持 `--cpu_affinity_cpus` 和 `--cpu_affinity_share`，将进程绑定到指定 CPU 范围。
- 支持 `--sync_start`，父进程等待所有 client 写入 `client_N.ready` 后再写 `go` 文件。
- `infer.py` 在 warmup 后进入 `wait_sync_start()`，看到 `go` 后才进入正式推理。
- 支持 `--text_file`，一行一条推理文本。
- 支持 `--warmup_full`，warmup 阶段完整消费流式输出，而不是只取首包。
- 支持 `--no_save_audio`，用于性能压测。

### 6.3 no-save 路径减少 CPU 同步

文件：

```text
infer.py
cosyvoice/cli/model.py
```

`infer.py` 在 `--no_save_audio` 时设置：

```bash
COSYVOICE2_NO_CPU_OUTPUT=1
```

`CosyVoice2Model._wrap_tts_output()` 根据该变量决定是否 `.cpu()`：

```python
if self._no_cpu_output:
    return {'tts_speech': tts_speech}
return {'tts_speech': tts_speech.cpu()}
```

这样性能压测时不会把每个 chunk 的 device-to-host copy 和 wav 保存开销计入 RTF。保存音频时仍保持原行为。

### 6.4 Flow mask 同步点优化

文件：

```text
cosyvoice/flow/flow.py
cosyvoice/utils/mask.py
```

改动：

- `make_pad_mask(token_len, max_len=token.shape[1])` 显式传入 `max_len`，避免内部 `lengths.max().item()` 触发 NPU 到 Host 同步。
- `torch.tensor([mel_len], device=h.device)` 放到目标 device 上，减少不必要的 Host tensor。
- `COSYVOICE2_SKIP_MASK_SANITY=1` 时跳过推理期 mask 全 false 检查，避免 `.item()` 强制同步。

这些改动不改变模型数学结果，主要减少同步点和日志/检查开销。

### 6.5 LLM 采样和 token tensor 优化

文件：

```text
cosyvoice/llm/llm.py
cosyvoice/utils/common.py
cosyvoice/cli/model.py
```

改动：

- `COSYVOICE2_SAMPLING_MODE=fast_topk` 时使用 `fast_topk_sampling()`，避免原始 RAS/top-p 的全量排序路径。
- `COSYVOICE2_DEVICE_TOKEN_DECODE=1` 时，LLM decode 阶段直接保留 device token tensor，减少逐 token `.item()` 同步。
- `_speech_tokens_to_tensor()` 支持 token 列表里是 Tensor 的情况，避免中间把 token 拉回 CPU。
- 默认保留 `COSYVOICE2_FAST_TOPK_K=25`。TopK=10 曾作为探索方向，但会改变采样分布，未作为交付默认。

## 7. transformers 从官方版本替换到当前版本

当前推理必须使用本仓库适配过的 Qwen2 transformers 文件。主仓库通过：

```bash
export PYTHONPATH=transformers/src:${PYTHONPATH:-}
```

优先加载本地 `transformers`，而不是环境里 pip 安装的 HuggingFace transformers。

### 7.1 改动文件清单

只改了一个 transformers 文件：

```text
transformers/src/transformers/models/qwen2/modeling_qwen2.py
```

本仓库已经单独保存了一份改好的替换文件：

```text
docs/transformers_replacement/src/transformers/models/qwen2/modeling_qwen2.py
```

这份文件对应当前 transformers 子仓库提交：

```text
e352348b1442775fda3d6faf2aa716d6dd581ff5
```

官方 transformers 基础版本是：

```text
8e3e145b42
```

### 7.2 从官方 transformers 复现到当前版本

从官方 HuggingFace transformers 拉代码：

```bash
git clone https://github.com/huggingface/transformers.git transformers
cd transformers
git checkout 8e3e145b42
cd ..
```

然后只做一次文件替换：

```bash
install -D \
  docs/transformers_replacement/src/transformers/models/qwen2/modeling_qwen2.py \
  transformers/src/transformers/models/qwen2/modeling_qwen2.py
```

替换关系如下：

```text
把官方文件：
transformers/src/transformers/models/qwen2/modeling_qwen2.py

替换成：
docs/transformers_replacement/src/transformers/models/qwen2/modeling_qwen2.py
```

替换后可以在 transformers 子仓库里单独提交：

```bash
cd transformers
git add src/transformers/models/qwen2/modeling_qwen2.py
git commit -m "adapt qwen2 for cosyvoice2 ascend hidden-only"
cd ..
```

### 7.3 为什么要替换这个文件

这个文件里包含当前性能路径依赖的 Qwen2 改动：

```text
Ascend NPU RMSNorm / AddRMSNorm
Ascend prompt/incremental FlashAttention
固定 KV cache buffer
TorchAIR cache_compile prefill/decode
safe hidden-only runner，跳过 lm_head，只返回 hidden states
```

CosyVoice 侧通过 `cosyvoice/llm/llm.py` 调用：

```python
self.model.model.forward_hidden_only(...)
```

如果不替换这个文件，`COSYVOICE2_QWEN_HIDDEN_ONLY=1` 不会生效，当前 10 进程低延迟路径无法复现。

### 7.4 可选：把 transformers 作为子仓库上传

如果你希望别人 clone 后不手动替换文件，可以把替换后的 `transformers` 目录作为单独 GitHub 仓库上传，然后在主仓库 `.gitmodules` 里登记这个 URL。

示例：

```bash
cd transformers
git remote add github https://github.com/<your-user>/transformers-cosyvoice2-ascend.git
git push github HEAD:main
cd ..

git config -f .gitmodules submodule.transformers.path transformers
git config -f .gitmodules submodule.transformers.url https://github.com/<your-user>/transformers-cosyvoice2-ascend.git
git add .gitmodules transformers
git commit -m "add transformers ascend submodule"
```

别人复现时：

```bash
git clone --recursive https://github.com/<your-user>/<your-cosyvoice-repo>.git
cd <your-cosyvoice-repo>
git submodule update --init --recursive
```

## 8. 默认性能配置和原因

多进程默认参数在 `run_manual_concurrent.sh` 中固化：

```bash
COSYVOICE2_HIFT_DECODE_OM=experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om
COSYVOICE2_HIFT_DECODE_GEARS=30,50,128,130,160
COSYVOICE2_QWEN_HIDDEN_ONLY=1
COSYVOICE2_SAMPLING_MODE=fast_topk
COSYVOICE2_FAST_TOPK_K=25
COSYVOICE2_DEVICE_TOKEN_DECODE=1
COSYVOICE2_FIRST_CHUNK_SIZE=25
COSYVOICE2_TOKEN_HOP_LEN=60
COSYVOICE2_FLOW_CONTEXT_TOKENS=25
COSYVOICE2_FLOW_HIFT_STREAM=0
COSYVOICE2_SKIP_MASK_SANITY=1
TASK_QUEUE_ENABLE=2
ENABLE_DYNAMIC_SHAPE_MULTI_STREAM=1
OMP_NUM_THREADS=2
MKL_NUM_THREADS=1
CPU_AFFINITY_CPUS=144-167
CPU_AFFINITY_SHARE=1
```

选择这些默认值的原因：

- 10 进程下额外 Flow/HiFT stream 会增加 SQ/CQ 压力，所以默认 `COSYVOICE2_FLOW_HIFT_STREAM=0`。
- 10 进程下 CPU/BLAS 抢占会放大抖动，所以多进程默认 `OMP=2`、`MKL=1`。
- 当前机器 NPU0 近端 CPU 观测为 `144-167`，默认 10 进程共享这 24 个 core，比硬切小段更稳。
- 完整 warmup 能显著降低正式阶段首次遇到新文本长度/shape 的抖动。
- `infer.py` 会在导入 `torch/torch_npu` 之前先执行 CPU 亲和和线程环境设置，避免 torch_npu 初始化阶段创建的 Host 线程落到非亲和 CPU 上。

## 9. 验收口径和已验证结果

正式统计只解析 `[INFO] infer round ...` 之后的输出，排除 warmup。

首包：

- 每条文本第一个 `yield speech len ..., rtf ...`。
- 首包耗时按 `speech_len * rtf * 1000` 计算 ms。

中间包：

- 每条文本非首包、非 final tail 的 chunk（`yields[1:-1]`）。
- RTF 直接使用日志里的 `rtf`。

final tail：

- 每条文本最后一个 chunk。
- 单独统计，不混入中间包验收。

解析工具：

```bash
python3 tools/parse_run_metrics.py logs/manual_hift_om_v2_sync_run/run_YYYYMMDD_HHMMSS
```

### 9.1 当前推荐验收配置

```bash
ASCEND_RT_VISIBLE_DEVICES=2 bash run_manual_concurrent.sh
```

脚本会自动将 10 进程绑定到 NPU2 近端 CPU `144-167`。如需手动指定：

```bash
ASCEND_RT_VISIBLE_DEVICES=2 CPU_AFFINITY_CPUS=144-167 bash run_manual_concurrent.sh
```

### 9.2 最新复现结果（2026-07-24）

在 `data/manual_transcript_20260720.txt`、67 行完整 warmup、10 进程 sync_start 条件下：

| run | NPU | CPU | 首包 p90 | 首包 p95 | 中间包 p90 RTF | 中间包 p95 RTF | 平均中间包 RTF | RTF>0.3 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `run_20260724_131659` | 2 | 144-167 | 392.0 ms | 398.8 ms | 0.2909 | 0.2929 | 0.2831 | 0.43% |
| `run_20260724_033300` | 2 | 144-167 | 394.1 ms | 397.0 ms | 0.2910 | 0.2935 | 0.2835 | 1.29% |

两次 run 均满足首包 p90 `< 400ms`、中间包 p90 RTF `< 0.3`。

### 9.3 验收注意事项

- 压测时不要保存音频：保持默认 `NO_SAVE_AUDIO=1`。
- 不要缩短 warmup：正式验收使用默认完整 warmup。
- 不要在压测时持续运行 `watch npu-smi info`，会干扰设备管理接口。
- 目标 NPU 必须是整卡，不要存在 vNPU 切分残留。
- 性能仍接近单卡 10 进程硬件边界；正式验收前确认 NPU 上无其他推理进程，CPU 亲和核无外部高负载。

更完整的优化过程记录见 `docs/cosyvoice2_10proc_latency_optimization_record.md`。

## 10. 交付文件和运行产物清理策略

已纳入版本控制的交付关键文件：

```text
run_manual_concurrent.sh
run.sh
export.sh
register_wav.py
infer.py
infer_manual_concurrent.py
cosyvoice/cli/cosyvoice.py
cosyvoice/cli/model.py
cosyvoice/hifigan/generator.py
cosyvoice/llm/llm.py
cosyvoice/flow/flow.py
cosyvoice/flow/flow_matching.py
cosyvoice/utils/common.py
cosyvoice/utils/mask.py
data/manual_transcript_20260720.txt
experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om
tools/export_hift_decode_onnx.py
tools/parse_run_metrics.py
docker/README_ascend.md
docs/transformers_replacement/src/transformers/models/qwen2/modeling_qwen2.py
transformers
```

本版本已从 git 中清理的中间/本机产物：

```text
logs/
testout/
kernel_meta/
extra-info/
huawei_model/
.torchair_cache/
experiments/torchair_cache_*/
experiments/hift_decode_om_20260706_230701/*.log
cosyvoice/cli/model copy.py
cosyvoice/flow/flow_matching-Copy1.py
cosyvoice/flow/flow_matching.py.bak
fusion_result.json
exception_cb_index_*.bin
```

`.gitignore` 会忽略上述运行产物以及新生成的音频、大权重、profiling/CANN 临时文件。重新跑压测不会污染 git 状态。

## 附录

- 上游 CosyVoice 原始 README：`docs/UPSTREAM_COSYVOICE_README.md`
- 10 进程延迟优化过程记录：`docs/cosyvoice2_10proc_latency_optimization_record.md`
- Ascend Docker 打包说明：`docker/README_ascend.md`
