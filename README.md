# CosyVoice2 Ascend 910B 推理加速交付说明

本文档说明当前仓库如何从开源 CosyVoice2 代码演进到交付版本、每类模型文件如何处理、代码做了哪些关键修改，以及如何复现单卡 10 进程流式推理指标。

当前交付目标：

- 不改变当前业务 chunk 配置：首包 `25`，中间包 hop `60`。
- 单张 Ascend 910B，10 个独立进程同时正式推理。
- 正式推理首包 p90 `< 400ms`。
- 正式推理中间包 p90 RTF `< 0.3`。
- 中间包统计口径排除 final tail。final tail 是每句话结束时的收尾包，长度、cache 状态和普通中间包不同，不能混入中间包验收。

## 1. 快速复现

进入环境和代码目录：

```bash
conda activate voxcpm
cd /data/xmren/work/work/test/model/CosyVoice-claude
npu-smi info
```

10 进程正式压测：

```bash
bash run_manual_concurrent.sh
```

单进程推理并保存音频：

```bash
bash run.sh
```

常用覆盖方式：

```bash
# 指定 NPU
ASCEND_RT_VISIBLE_DEVICES=0 bash run_manual_concurrent.sh

# 多进程保存音频，只用于听音质，不用于性能验收
NO_SAVE_AUDIO=0 bash run_manual_concurrent.sh

# 使用指定 SFT 音色。该音色必须已经注册到 MODEL_DIR/spk2info.pt
SFT_SPK_ID=03729 bash run_manual_concurrent.sh

# 换抄本。默认会自动完整 warmup 新抄本所有非空行
TEXT_FILE=data/your_transcript.txt bash run_manual_concurrent.sh

# 调试时缩短 warmup；正式验收不要这样做
WARM_UP_TIMES=5 bash run_manual_concurrent.sh
```

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

## 2. 从开源代码到当前版本的整体步骤

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

## 3. 模型文件处理说明

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
| `experiments/hift_decode_om_20260706_230701/hift_decode_static.om` | 本仓库跟踪 | 是 | 早期 HiFT decode OM，首版验证产物 |
| `experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om` | 本仓库跟踪 | 是 | 当前默认使用的 HiFT decode OM |
| `experiments/hift_decode_om_20260706_230701/real_hift_decode_shapes.txt` | 本仓库跟踪 | 是 | 实测 HiFT decode shape/gear 记录 |

注意：

- 外部模型目录必须存在，默认脚本不会自动下载或复制权重。
- `ais_bench` 会做路径安全检查。如果 OM 文件不是当前用户或用户组可访问，可能报 owner/ownergroup 相关错误。需要保证模型文件和目录 owner/group 满足当前运行用户要求。
- 本仓库 `.gitignore` 明确忽略新生成的运行日志、NPU 编译缓存、profiling 产物和临时权重，避免交付代码被本机产物污染。

### 3.1 从开源权重处理成当前可运行模型目录

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

### 3.2 注册 SFT speaker 音色

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

### 3.3 导出 Flow estimator ONNX

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

### 3.4 编译 Flow OM

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

### 3.5 编译 speech tokenizer OM

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

### 3.6 重新导出并编译 HiFT decode OM

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

## 4. HiFT decode 导出 OM 和运行时接入

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

## 5. CosyVoice 主仓库代码改动

### 5.1 启动脚本

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

### 5.2 并发调度和同步

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

### 5.3 no-save 路径减少 CPU 同步

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

### 5.4 Flow mask 同步点优化

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

### 5.5 LLM 采样和 token tensor 优化

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

## 6. transformers 从官方版本替换到当前版本

当前推理必须使用本仓库适配过的 Qwen2 transformers 文件。主仓库通过：

```bash
export PYTHONPATH=transformers/src:${PYTHONPATH:-}
```

优先加载本地 `transformers`，而不是环境里 pip 安装的 HuggingFace transformers。

### 6.1 改动文件清单

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

### 6.2 从官方 transformers 复现到当前版本

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

### 6.3 为什么要替换这个文件

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

### 6.4 可选：把 transformers 作为子仓库上传

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

## 7. 默认性能配置和原因

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

## 8. 验收口径和已验证结果

正式统计只解析 `[INFO] infer round ...` 之后的输出，排除 warmup。

首包：

- 每条文本第一个 `yield speech len ..., rtf ...`。
- 首包耗时按 `speech_len * rtf * 1000` 计算 ms。

中间包：

- 每条文本非首包、非 final tail 的 chunk。
- RTF 直接使用日志里的 `rtf`。

final tail：

- 每条文本最后一个 chunk。
- 单独统计，不混入中间包验收。

默认 5 条 warmup 时会在未 warmup 文本上出现边界抖动：

```text
logs/manual_hift_om_v2_sync_run/run_20260722_203656
first p90 = 403.82ms
middle non-final p90 RTF = 0.30252
```

完整 warmup 当前 67 行抄本后，连续两轮 10 进程同步推理达成 p90 目标：

```text
logs/exp_full_warmup_67_24core/run_20260722_205133
first p90 = 393.98ms
middle non-final p90 RTF = 0.29962

logs/exp_full_warmup_67_24core_rerun/run_20260722_205954
first p90 = 399.61ms
middle non-final p90 RTF = 0.29501
```

新增 CPU 初始化顺序优化后，默认 24 核共享配置连续两轮达成 p90 目标：

```text
logs/exp_cpu_early_affinity_default/run_20260722_223830
first p90 = 398.48ms
middle non-final p90 RTF = 0.29907

logs/exp_cpu_early_affinity_default_rerun/run_20260722_234203
first p90 = 394.37ms
middle non-final p90 RTF = 0.29782
```

已尝试但不作为默认的 CPU/系统侧方案：

```text
独立 2 核/进程，144-163 分段：首包 p90 404.99ms，中间包 p90 0.30014，放弃。
OMP_NUM_THREADS=1：首包 p90 397.65ms，但中间包 p90 0.30165，放弃。
关闭 kernel.numa_balancing：首包 p90 398.26ms，但中间包 p90 0.30589，放弃。
停止 irqbalance 并绑定 dev0_sq_task：首包 p90 403.93ms，中间包 p90 0.30650，放弃。
共享 144-163 预留 164-167：单轮中间包 p90 0.29825，但复跑退化到 0.30170，放弃。
```

注意：性能仍接近单卡 10 进程硬件边界。正式验收前需要保证 NPU 上没有其他推理进程，CPU 亲和核没有明显外部高负载，并使用默认完整 warmup。

## 9. 交付文件和运行产物清理策略

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
docs/transformers_replacement/src/transformers/models/qwen2/modeling_qwen2.py
transformers
```

已从交付版本移除的旧入口/产物：

```text
run copy.sh
run1.sh
run_infer_py.sh
run_streaming.sh
run_streaming copy.sh
infer_streaming.py
run_stream_chunk_web.sh
stream_chunk_web.py
stream_chunk_probe.html
STREAM_CHUNK_WEB_README.md
fusion_result.json
exception_cb_index_*.bin
xmren_log.log
```

移除原因：

- `run.sh` 和 `run_manual_concurrent.sh` 已经覆盖单进程和 10 进程交付推理。
- 旧 streaming/web probe 脚本仍使用早期 hard-coded 模型路径、旧音色名或 HiFT torch.compile 路径，容易误导复现。
- `fusion_result.json`、`exception_cb_index_*.bin`、`xmren_log.log` 是本机运行/编译产物，不属于代码交付。

`.gitignore` 会忽略：

- `logs/`
- `testout/`
- `*.log`
- `kernel_meta/`
- `extra-info/`
- `fusion_result.json`
- `exception_cb_index_*.bin`
- `.torchair_cache/`
- `experiments/torchair_cache_hidden_safe/`
- profiling / CANN 临时产物
- 新生成的音频和大权重文件

如果重新跑压测，生成的日志和音频不会污染 git 状态。

---

以下为上游 CosyVoice 原始 README 内容，保留用于查询开源项目基础用法。


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
