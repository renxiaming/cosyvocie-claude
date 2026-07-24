# Ascend Docker packaging

This packaging path targets the current aarch64 Ascend NPU runtime. It keeps
the original CUDA Dockerfile untouched.

## Build

Run from the project root:

```bash
bash docker/build_ascend_image.sh
```

Defaults:

- Image: `cosyvoice2-ascend:manual`
- Base image: `swr.cn-southwest-2.myhuaweicloud.com/aslp/cosyvoice2:62g`
- Project in image: `/opt/CosyVoice-claude`
- Model in image: `/opt/weight/CosyVoice2-0.5B_sft_shenhu_25_60`

Override examples:

```bash
IMAGE=cosyvoice2-ascend:20260724 bash docker/build_ascend_image.sh
MODEL_DIR=/path/to/CosyVoice2-0.5B_sft_shenhu_25_60 bash docker/build_ascend_image.sh
PROJECT_DIR=/path/to/CosyVoice-claude bash docker/build_ascend_image.sh
```

## Run

Single process:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 bash docker/run_ascend.sh bash run.sh
```

Ten-process benchmark:

```bash
ASCEND_RT_VISIBLE_DEVICES=2 bash docker/run_ascend.sh bash run_manual_concurrent.sh
```

The run helper mounts host Ascend driver devices and stores container outputs
under `docker_outputs/logs` and `docker_outputs/testout`.

## Export

```bash
bash docker/save_ascend_image.sh
```

On another compatible Ascend aarch64 host:

```bash
docker load -i cosyvoice2-ascend-manual.tar
ASCEND_RT_VISIBLE_DEVICES=0 bash docker/run_ascend.sh bash run.sh
```

The target host still needs a compatible Ascend driver and visible NPU devices.
The model weights and project code are already inside the image.
