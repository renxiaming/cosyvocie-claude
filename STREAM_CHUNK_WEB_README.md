# CosyVoice Chunk Stream Probe

这个小工具用于验证 CosyVoice 流式 chunk 是否能顺序衔接播放：

- 后端每生成一个 `tts_speech` chunk 就立刻通过 `/api/stream` 推给浏览器。
- 浏览器用 Web Audio API 把 PCM chunk 顺序排进播放队列。
- 页面会显示每个 chunk 的生成间隔、音频时长、到达时队列剩余量和是否出现播放空隙。

## 启动

在能正常运行 `run_manual_concurrent.sh` 的同一个 Docker/conda 环境里执行：

```bash
cd /data/xmren/work/work/test/model/CosyVoice-claude
PORT=50080 ./run_stream_chunk_web.sh
```

如果需要改模型或音色：

```bash
MODEL_PATH=../weight/CosyVoice2-0.5B_sft_shenhu_25_60 DEFAULT_SPK=03729 PORT=50080 ./run_stream_chunk_web.sh
```

服务监听 `0.0.0.0:${PORT}`，容器内日志会打印访问地址。

## Docker 端口映射

如果还没有启动 Docker 容器，需要在 `docker run` 时发布端口：

```bash
docker run ... -p 50080:50080 ...
```

如果容器已经启动但没有映射端口，Docker 不能给运行中的容器直接补 `-p`。可选方案：

1. 重新以 `-p 50080:50080` 启动容器。
2. 如果你是 `--network host` 启动的容器，直接访问服务器的 `50080` 端口。

## SSH 访问

本地电脑开隧道：

```bash
ssh -L 50080:127.0.0.1:50080 user@server_ip
```

然后浏览器打开：

```text
http://127.0.0.1:50080/
```

如果 Docker 做了端口映射，隧道目标是宿主机的 `127.0.0.1:50080`。如果你在服务器上再套了一层跳板机，按你的 SSH 登录链路把 `-L` 加在最外层即可。

## 页面判断方式

- `server` 列的 `ahead` 表示本 chunk 的生成间隔不大于上一个 chunk 的音频时长。
- `playback` 列的 `ahead` 表示浏览器收到本 chunk 时，上一个 chunk 还没有播完。
- `play gap` 大于 `0ms` 表示播放队列已经空过，真实听感可能会断。
- 顶部 `queued` 越稳定越好；持续接近 `0.00s` 说明快到播放边界了。
