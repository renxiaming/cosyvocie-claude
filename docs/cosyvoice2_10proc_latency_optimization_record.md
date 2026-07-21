# CosyVoice2 10 进程推理延迟优化记录

日期：2026-07-07

基线版本：

```text
commit 05e48ed93f5c1c19c228163c7953bc209665fd39
Author: renxiaming <1024404107@qq.com>
Date:   Mon Jul 6 21:55:59 2026 +0800

    10进程跑通
```

本文记录从上述“10 进程跑通”版本到当前优化版本的主要工程改动、验证方法、收益和风险点。

## 目标

目标指标：

- 10 进程并发推理可稳定启动并完成。
- 首包非异常值小于 500ms，最好接近或小于 400ms。
- 中间包非异常值 RTF 小于 0.3。
- 保存音频时音质不能劣化。

## 基线问题

`05e48ed` 已经解决了 10 进程能跑通的问题，但仍有几个工程瓶颈：

1. **正式推理开始不够严格同步**
   不同进程可能在 warmup / 编译 / 加载阶段耗时不同。如果一部分进程已经开始正式推理，另一部分还卡在 warmup 前后，则表面 RTF 会偏乐观，不能代表真正 10 进程同时推理。

2. **HiFT / Flow / LLM 组合链路仍有较多运行时开销**
   CosyVoice2 的流式路径不是单个 LLM，它包含 LLM token 生成、Flow、HiFT vocoder、流式拼接和保存输出。单纯优化其中一段，需要确认不会引入音质问题。

3. **保存音频和 no-save 压测路径混在一起**
   性能压测时如果每个 chunk 都 `.cpu()` 或保存 wav，会把设备同步、host copy、IO 时间混进推理 RTF。

4. **Qwen LLM 路径存在无用计算**
   CosyVoice2 实际使用 Qwen 最后一层 hidden state，然后接自己的 `llm_decoder` 预测 speech token。Qwen 自带的文本 `lm_head` logits 对 CosyVoice 推理结果没有用途，但原路径仍会计算它。

## 当前主要改动

### 1. 严格同步正式推理开始

涉及文件：

- `infer_manual_concurrent.py`
- `infer.py`
- `run_manual_concurrent.sh`

核心逻辑：

- 每个子进程完成模型加载和 warmup 后，写入 `client_N.ready`。
- 父进程等待所有 ready 文件出现后，写入 `go` 文件。
- 所有子进程看到 `go` 后同时进入正式推理。

当前脚本默认：

```bash
export SYNC_START="${SYNC_START:-1}"
export SYNC_START_TIMEOUT="${SYNC_START_TIMEOUT:-1800}"
```

价值：

- 保证测到的指标是真正 10 进程同时正式推理。
- 避免“部分进程先跑完，另一部分才开始”的假低 RTF。

### 2. HiFT decode OM 化

涉及文件：

- `run_manual_concurrent.sh`
- `cosyvoice/cli/model.py`

当前脚本默认加载 HiFT decode OM：

```bash
export COSYVOICE2_HIFT_DECODE_OM="${COSYVOICE2_HIFT_DECODE_OM:-experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om}"
export COSYVOICE2_HIFT_DECODE_GEARS="${COSYVOICE2_HIFT_DECODE_GEARS:-30,50,128,130,160}"
```

核心作用：

- 将 HiFT vocoder 的 `decode` 从 PyTorch eager/compile 路径替换为固定档位 OM 推理。
- 减少 HiFT 侧 Python/runtime 开销。
- 降低动态编译和多进程 runtime 抖动。

注意：

- OM 档位必须覆盖实际流式 chunk 的 shape。
- 当前使用的是固定 gears：`30,50,128,130,160`。
- 如果出现未覆盖档位，需要补导出/补编译，而不能让运行时落入异常路径。

### 3. 流式参数和 NPU runtime 参数固定

涉及文件：

- `run_manual_concurrent.sh`

当前关键参数：

```bash
export COSYVOICE2_FIRST_CHUNK_SIZE="${COSYVOICE2_FIRST_CHUNK_SIZE:-25}"
export COSYVOICE2_TOKEN_HOP_LEN="${COSYVOICE2_TOKEN_HOP_LEN:-60}"
export COSYVOICE2_FLOW_CONTEXT_TOKENS="${COSYVOICE2_FLOW_CONTEXT_TOKENS:-25}"
export COSYVOICE2_FLOW_GEARS="${COSYVOICE2_FLOW_GEARS:-50,74,98,122,146,170,200,230,260,290,320,350,380,410}"
export COSYVOICE2_FLOW_HIFT_STREAM="${COSYVOICE2_FLOW_HIFT_STREAM:-0}"
```

说明：

- `FIRST_CHUNK_SIZE=25`：控制首包 token 数。
- `TOKEN_HOP_LEN=60`：控制中间包推进步长。
- `FLOW_CONTEXT_TOKENS=25`：保留 Flow 上下文。
- `FLOW_HIFT_STREAM=0`：10 进程下关闭额外 Flow/HiFT NPU stream，避免多进程 SQ/CQ 资源进一步紧张。

### 4. no-save 压测路径避免强制 CPU 同步

涉及文件：

- `infer.py`
- `cosyvoice/cli/model.py`

当前逻辑：

```python
if args.no_save_audio:
    os.environ.setdefault('COSYVOICE2_NO_CPU_OUTPUT', '1')
```

`CosyVoice2Model` 中增加：

```python
def _wrap_tts_output(self, tts_speech):
    if self._no_cpu_output:
        return {'tts_speech': tts_speech}
    return {'tts_speech': tts_speech.cpu()}
```

价值：

- 性能压测 `--no_save_audio` 时不再每个 chunk 都 `.cpu()`。
- 避免把 device-to-host 同步时间混入 RTF。
- 保存音频路径不受影响，仍然会返回 CPU tensor 并写 wav。

### 5. Qwen safe hidden-only

涉及文件：

- `cosyvoice/llm/llm.py`
- `transformers/src/transformers/models/qwen2/modeling_qwen2.py`
- `run_manual_concurrent.sh`

背景：

CosyVoice2 的 LLM 使用方式是：

1. 输入 text / prompt speech token embedding。
2. Qwen 输出最后一层 hidden state。
3. CosyVoice 自己的 `llm_decoder` 根据 hidden state 预测 speech token。

因此 Qwen 自带的文本 `lm_head` logits 对最终 TTS 推理没有用途。

#### 最初失败版本

最初尝试是在 `Qwen2Model._forward()` 中，当 `lm_head is None` 时直接返回 `BaseModelOutputWithPast`。这能降低 RTF，但会带来音频杂音风险。

失败原因：

- 原 Qwen `cache_compile` 图的返回结构是 `(out, logits)`。
- 失败版本 hidden-only 返回结构变成单个 `out`。
- torchair 编译缓存对这种同函数、不同返回结构的路径不稳定。
- 最终可能导致某些段拿到错误 hidden/cache，LLM token 错误后，Flow/HiFT 生成一段全是噪声。

这个版本已经废弃。

#### 当前 safe hidden-only 实现

当前实现不再复用原 `decode/prefill` 编译入口，而是新增独立入口：

```python
self.cached_decode_hidden = tng.inference.cache_compile(self.decode_hidden, config=config)
self.cached_prefill_hidden = tng.inference.cache_compile(self.prefill_hidden, config=config)
```

新增：

```python
forward_hidden_only()
decode_hidden()
prefill_hidden()
```

关键点：

- hidden-only 使用独立 torchair 编译函数。
- 不污染原 `decode/prefill` 编译缓存。
- `_forward(..., lm_head=None)` 返回 `(out, hidden_states)`，仍保持双返回结构。
- `CosyVoice Qwen2Encoder` 只取 `last_hidden_state` 和 `past_key_values`。

脚本默认开启：

```bash
export COSYVOICE2_QWEN_HIDDEN_ONLY="${COSYVOICE2_QWEN_HIDDEN_ONLY:-1}"
export TORCHAIR_CACHE_HOME="${TORCHAIR_CACHE_HOME:-experiments/torchair_cache_hidden_safe}"
```

回退方式：

```bash
COSYVOICE2_QWEN_HIDDEN_ONLY=0 bash run_manual_concurrent.sh
```

## 音质验证

用户反馈旧 hidden-only 保存音频存在一段时间全是杂音。

排查结果：

- `testout/clean_qwen_save/client_0` 的旧 Qwen 路径音质 OK。
- 杂音不是峰值超过 1.0 造成的。
- 根因是 unsafe hidden-only 版本对 torchair 编译图返回结构不安全。

safe hidden-only 音质验证：

```text
旧路径样本：testout/clean_qwen_save_clamped/client_0
safe hidden-only 样本：testout/hidden_safe_save/client_0
```

逐采样对比结果：

```text
sft_full_0_0.wav max_abs_diff=0.0000000000 mean_abs_diff=0.0000000000
sft_full_0_1.wav max_abs_diff=0.0000000000 mean_abs_diff=0.0000000000
sft_full_0_2.wav max_abs_diff=0.0000000000 mean_abs_diff=0.0000000000
sft_full_0_3.wav max_abs_diff=0.0000000000 mean_abs_diff=0.0000000000
sft_full_0_4.wav max_abs_diff=0.0000000000 mean_abs_diff=0.0000000000
sft_full_0_5.wav max_abs_diff=0.0000000000 mean_abs_diff=0.0000000000
sft_full_0_6.wav max_abs_diff=0.0000000000 mean_abs_diff=0.0000000000
```

结论：

- safe hidden-only 不改变音频内容。
- unsafe hidden-only 的杂音问题已通过独立编译入口和稳定返回结构解决。

## 性能统计口径

日志来源：

- `05e48ed` 代表日志：`logs/manual/run_20260706_183253`
- HiFT OM 同步基线：`logs/manual_hift_om_v2_sync_run/run_20260707_143321`
- no-save/no-cpu-output：`logs/manual_hift_om_no_cpu_output_test/run_20260707_193014`
- safe hidden-only：`logs/manual_qwen_hidden_safe_10p/run_20260707_230918`

统计方法：

- 每条文本出现 `infer round` 后的第一个 `yield speech` 作为首包。
- 首包耗时换算：

```text
first_ms = speech_len * rtf * 1000
```

- 中间包 RTF：非首包且 `speech_len > 0.9s` 的 chunk。
- 异常值：`rtf > 5`，通常对应首次编译或 warmup，统计时排除。

## 性能对比

### 10 进程跑通版本 vs 当前 safe hidden-only

| 指标 | 05e48ed 10进程跑通 | 当前 safe hidden-only | 改善 |
| --- | ---: | ---: | ---: |
| 首包 avg | 448.6ms | 400.5ms | -48.1ms / -10.7% |
| 首包 p50 | 456.8ms | 396.2ms | -60.6ms / -13.3% |
| 首包 p95 | 536.5ms | 438.2ms | -98.3ms / -18.3% |
| 中间包 RTF avg | 0.335369 | 0.291429 | -0.043940 / -13.1% |
| 中间包 RTF p50 | 0.344347 | 0.297494 | -0.046853 / -13.6% |
| 中间包 RTF p95 | 0.402520 | 0.317905 | -0.084615 / -21.0% |

### 分阶段结果

| 阶段 | 首包 avg | 首包 p95 | 中间包 RTF avg | 中间包 RTF p95 |
| --- | ---: | ---: | ---: | ---: |
| `05e48ed` 10进程跑通 | 448.6ms | 536.5ms | 0.335369 | 0.402520 |
| HiFT OM + 同步基线 | 456.2ms | 497.5ms | 0.334537 | 0.373477 |
| no-save/no-cpu-output | 449.2ms | 475.4ms | 0.325208 | 0.354735 |
| safe hidden-only | 400.5ms | 438.2ms | 0.291429 | 0.317905 |

结论：

- 首包 p95 从 `536.5ms` 降到 `438.2ms`，已明显低于 500ms。
- 首包 p50 为 `396.2ms`，已经接近 400ms 目标。
- 中间包 RTF avg 从 `0.335369` 降到 `0.291429`，已经小于 0.3。
- 中间包 RTF p95 为 `0.317905`，相比基线明显降低，但如果以 p95 严格要求小于 0.3，还需要继续优化。

## 2026-07-21 追加优化：CPU/NPU 拓扑亲和与线程收敛

本轮目标不改变首包/中间包 chunk 长度，不改变 Flow 步数，不引入影响音质的 OM device pointer 路径，仅从工程调度和 CPU runtime 竞争侧继续压低 10 进程中间包 RTF。

### Profiling 结论

新增默认关闭的 chunk profiling 开关：

```bash
export COSYVOICE2_CHUNK_PROFILE=0
```

开启后会在每个流式 chunk 打印：

```text
[CHUNK_PROFILE] offset=... speech_len=... llm_ms=... t2w_ms=... rtf=...
```

2 进程 profiling 日志：

```text
logs/exp_chunk_profile_2p/run_20260721_142427
```

聚合结论：

| 分段 | LLM avg | Flow+HiFT avg | RTF avg | LLM 占比 |
| --- | ---: | ---: | ---: | ---: |
| 中间包 | 345.57ms | 78.17ms | 0.162213 | 81.6% |
| 全部 chunk | 289.10ms | 86.59ms | 0.300242 | 77.0% |

结论：

- 10 进程下中间包主要瓶颈仍是 LLM token 生成和调度排队。
- Flow/HiFT 小优化收益有限，继续改 OM padding / HiFT 输入搬运不能解决主要长尾。
- 后续大收益方向仍是 LLM 执行层、调度层，或者 vLLM/SGLang/MindIE 类自定义 LLM 后端。

### CPU/NPU 拓扑亲和

机器拓扑：

```text
npu-smi info -t topo
NPU0 CPU Affinity: 144-167
```

原并发脚本把 10 个进程均分到全机器 `0-191`，大量进程会在远端 NUMA 上做 Python runtime、TorchNPU runtime、OM Host 输入输出准备，容易放大 LLM/OM 调度长尾。

本轮给 `infer_manual_concurrent.py` 增加两个参数：

```bash
--cpu_affinity_cpus 144-167
--cpu_affinity_share
```

含义：

- `--cpu_affinity_cpus`：指定一段 CPU 核范围。
- `--cpu_affinity_share`：10 个进程共享同一段 CPU 核，而不是再切成 10 份。

最终采用默认：

```bash
export CPU_AFFINITY_CPUS="${CPU_AFFINITY_CPUS:-144-167}"
export CPU_AFFINITY_SHARE="${CPU_AFFINITY_SHARE:-1}"
```

注意：

- `144-167` 是当前机器 NPU0 的就近 CPU。换机器或换 NPU 后必须重新执行 `npu-smi info -t topo` 确认。
- 如果使用 NPU1/NPU4 等其他卡，需要把 `CPU_AFFINITY_CPUS` 改成对应卡的 CPU Affinity。
- 实验中 `144-191` 虽然给了更多 CPU，但 strict 中间包 `RTF>0.3` 比例不如 `144-167`。

### CPU 线程数收敛

10 进程同时跑时，每个进程开太多 CPU/BLAS 线程会抢占 CPU runtime、TorchNPU runtime 和 Host 侧调度。当前最优默认值：

```bash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
```

验证结论：

- `OMP=4/MKL=2`：比全机绑核好，但仍有中间包长尾。
- `OMP=2/MKL=1`：当前最优，首包和 strict 中间包同时改善。
- `OMP=1/MKL=1`：CPU 侧过紧，strict 中间包 `RTF>0.3` 比例变差。

### 当前 2026-07-21 最佳配置

当前并发脚本默认使用：

```bash
export CPU_AFFINITY_CPUS="${CPU_AFFINITY_CPUS:-144-167}"
export CPU_AFFINITY_SHARE="${CPU_AFFINITY_SHARE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export ENABLE_DYNAMIC_SHAPE_MULTI_STREAM="${ENABLE_DYNAMIC_SHAPE_MULTI_STREAM:-1}"
```

最佳验证日志：

```text
logs/exp_cpu_share_144_167_threads2_10p/run_20260721_143733
```

对比日志：

```text
旧 round0 基线:
logs/manual_hift_om_v2_sync_run/run_20260720_225800
```

同口径 round0 统计：

| 指标 | 2026-07-20 基线 | 2026-07-21 当前最佳 | 改善 |
| --- | ---: | ---: | ---: |
| 首包 avg | 380.73ms | 371.69ms | -9.04ms |
| 首包 p50 | 388.17ms | 379.01ms | -9.16ms |
| 首包 p95 | 407.94ms | 396.69ms | -11.25ms |
| 首包 p99 | 413.01ms | 403.33ms | -9.68ms |
| strict 中间包 RTF avg | 0.293577 | 0.285056 | -0.008521 |
| strict 中间包 RTF p50 | 0.294036 | 0.284493 | -0.009543 |
| strict 中间包 RTF p95 | 0.312281 | 0.305498 | -0.006783 |
| strict 中间包 RTF p99 | 0.322722 | 0.313682 | -0.009040 |
| strict 中间包 RTF>0.3 | 23.73% | 7.91% | -15.82 pct |

说明：

- strict 中间包定义：非首包且 `speech_len > 0.9s` 的 chunk。
- 全量 mid 仍包含较短尾包，短尾包天然 RTF 偏高，因此 strict 指标更能反映稳定中间包。
- 当前首包 p95 已低于 400ms。
- strict 中间包 p95 仍为 `0.305498`，还没有完全低于 0.3，但 `RTF>0.3` 比例已经从 `23.73%` 降到 `7.91%`。

### 已尝试但不保留的方案

| 方案 | 结果 | 处理 |
| --- | --- | --- |
| OM device input / `aclruntime.BaseTensor(data_ptr)` | 数值略好，但音质明显变差 | 已回退 |
| OM host padding buffer reuse | 单独略好，和最佳 CPU 亲和组合不稳定 | 已回退 |
| OM NPU FP32 copy | 收益很小，p95/尾包不稳 | 已回退 |
| 长度 tensor cache | strict `RTF>0.3` 略降，但 p95 和尾包变差 | 已回退 |
| `CPU_AFFINITY_CPUS=144-191` | CPU 更宽，但 strict `RTF>0.3` 变差 | 不采用 |
| `OMP_NUM_THREADS=1` | CPU 侧过紧，strict `RTF>0.3` 变差 | 不采用 |
| `TASK_QUEUE_ENABLE=1` | 首包和中间包明显变差 | 不采用 |
| 跳过 fast_topk 前的 `log_softmax` | 没有收益，生成包数变化，存在采样差异风险 | 已回退 |
| `ENABLE_DYNAMIC_SHAPE_MULTI_STREAM=0` | 部分 p95 好，但 strict `RTF>0.3` 不如当前最佳 | 暂不采用 |

### 当前音质验证

当前保留代码重新生成的单进程音频：

```text
/home/ma-user/work/test/model/CosyVoice-claude/testout/current_best_audio_check
```

示例文件：

```text
/home/ma-user/work/test/model/CosyVoice-claude/testout/current_best_audio_check/sft_full_0_0.wav
/home/ma-user/work/test/model/CosyVoice-claude/testout/current_best_audio_check/sft_full_0_4.wav
/home/ma-user/work/test/model/CosyVoice-claude/testout/current_best_audio_check/sft_full_0_20.wav
/home/ma-user/work/test/model/CosyVoice-claude/testout/current_best_audio_check/sft_full_0_63.wav
```

该音频用于验证当前保留优化后的音质，不包含已回退的 OM device input 实验。

## 当前启动方式

默认启动：

```bash
bash run_manual_concurrent.sh
```

默认行为：

- 10 进程。
- 完整 warmup。
- warmup 后同步正式开始。
- 默认 no-save 压测。
- 默认开启 HiFT decode OM。
- 默认开启 safe hidden-only。
- 默认使用 `experiments/torchair_cache_hidden_safe` 作为 torchair cache。
- 默认绑定 NPU0 就近 CPU `144-167`，10 个进程共享该 CPU 范围。
- 默认 CPU 线程数为 `OMP=2, MKL=1, OPENBLAS=1, NUMEXPR=1`。

保存音频验证：

```bash
NO_SAVE_AUDIO=0 CONCURRENCY=1 SYNC_START=0 bash run_manual_concurrent.sh
```

关闭 safe hidden-only 回退：

```bash
COSYVOICE2_QWEN_HIDDEN_ONLY=0 bash run_manual_concurrent.sh
```

指定干净 torchair cache：

```bash
TORCHAIR_CACHE_HOME=experiments/torchair_cache_hidden_safe bash run_manual_concurrent.sh
```

## 当前状态

已经达成：

- 10 进程并发推理成功。
- 首包 p95 小于 500ms。
- 首包 p50 接近 400ms。
- 中间包 RTF avg 小于 0.3。
- safe hidden-only 音频与旧路径逐采样一致。

尚未完全达成：

- 中间包 RTF p95 仍约 `0.318`，如果“稳定”定义为 p95 小于 0.3，还需要继续优化。

下一步可继续尝试：

- 单进程多请求调度，减少 10 独立进程带来的 NPU runtime 竞争。
- LLM token step 层面的轻量 batch/scheduler。
- vLLM Ascend / SGLang Ascend 自定义 CosyVoice2 模型适配，但这属于更大改造，优先级低于当前 safe hidden-only 这类低风险优化。
