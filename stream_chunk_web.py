import argparse
import base64
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401
import torchair as tng
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from torchair.configs.compiler_config import CompilerConfig

from cosyvoice.cli.cosyvoice import CosyVoice2


ROOT_DIR = Path(__file__).resolve().parent
WEB_PAGE = ROOT_DIR / "stream_chunk_probe.html"

app = FastAPI(title="CosyVoice Chunk Stream Probe")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cosyvoice = None
server_config = {}
stream_lock = threading.Lock()


class StreamRequest(BaseModel):
    text: str = Field(..., min_length=1)
    spk_id: str = Field(default="03729", min_length=1)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    text_frontend: bool = True


def apply_cpu_affinity():
    if "CPU_AFFINITY_START" in os.environ and "CPU_AFFINITY_END" in os.environ:
        try:
            start = int(os.environ["CPU_AFFINITY_START"])
            end = int(os.environ["CPU_AFFINITY_END"])
            os.sched_setaffinity(0, range(start, end + 1))
            print("[INFO] CPU affinity set: cores {}-{}".format(start, end), flush=True)
        except Exception as exc:
            print("[WARN] Failed to set CPU affinity: {}".format(exc), flush=True)


def configure_threads():
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", 8))
    mkl_threads = int(os.environ.get("MKL_NUM_THREADS", 8))
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(mkl_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(mkl_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(mkl_threads))

    torch.set_num_threads(omp_threads)
    try:
        torch.set_num_interop_threads(min(4, omp_threads))
    except Exception:
        pass

    print(
        "[INFO] CPU threads configured: OMP={}, MKL={}, torch={}".format(
            omp_threads, mkl_threads, torch.get_num_threads()
        ),
        flush=True,
    )


def setup_cosyvoice(args):
    print("[INFO] loading model: {}".format(args.model_path), flush=True)
    model = CosyVoice2(args.model_path, load_om=args.load_om, fp16=args.fp16)
    model.model.llm.eval()

    if args.fp16:
        model.model.llm.llm.model.model.half()

    if args.compile_hift:
        print("[INFO] compiling hift.decode with torchair backend", flush=True)
        model.model.hift.remove_weight_norm()
        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        model.model.hift.decode = torch.compile(
            model.model.hift.decode,
            dynamic=True,
            fullgraph=True,
            backend=npu_backend,
        )

    return model


def tensor_to_pcm16_b64(speech):
    audio = speech.detach().float().cpu().flatten().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype("<i2", copy=False)
    return base64.b64encode(pcm16.tobytes()).decode("ascii"), int(pcm16.shape[0])


def ndjson(payload):
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def stream_events(req):
    if cosyvoice is None:
        yield ndjson({"event": "error", "message": "model is not ready"})
        return

    if not stream_lock.acquire(blocking=False):
        yield ndjson({"event": "error", "message": "another stream is running; stop it first"})
        return

    started_at = time.perf_counter()
    previous_ready_at = started_at
    chunk_count = 0
    audio_duration_total = 0.0

    try:
        sample_rate = int(cosyvoice.sample_rate)
        yield ndjson(
            {
                "event": "start",
                "sample_rate": sample_rate,
                "spk_id": req.spk_id,
                "speed": req.speed,
                "text_frontend": req.text_frontend,
                "server_start_unix": time.time(),
            }
        )

        with torch.no_grad():
            model_output = cosyvoice.inference_sft(
                req.text,
                req.spk_id,
                stream=True,
                speed=req.speed,
                text_frontend=req.text_frontend,
            )
            for index, chunk in enumerate(model_output):
                ready_at = time.perf_counter()
                speech = chunk["tts_speech"]
                pcm16_b64, samples = tensor_to_pcm16_b64(speech)
                duration_s = samples / sample_rate if sample_rate else 0.0
                audio_duration_total += duration_s
                chunk_count += 1

                yield ndjson(
                    {
                        "event": "chunk",
                        "index": index,
                        "sample_rate": sample_rate,
                        "samples": samples,
                        "duration_s": duration_s,
                        "server_elapsed_s": ready_at - started_at,
                        "server_delta_s": ready_at - previous_ready_at,
                        "pcm16_b64": pcm16_b64,
                    }
                )
                previous_ready_at = ready_at

        yield ndjson(
            {
                "event": "end",
                "chunks": chunk_count,
                "audio_duration_s": audio_duration_total,
                "server_elapsed_s": time.perf_counter() - started_at,
            }
        )
    except GeneratorExit:
        raise
    except Exception as exc:
        yield ndjson({"event": "error", "message": "{}".format(exc)})
    finally:
        stream_lock.release()


def warm_up(model, args):
    if args.warm_up_times <= 0:
        return
    print("[INFO] warm up start, times={}".format(args.warm_up_times), flush=True)
    warmup_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.warm_up_times):
            next(
                model.inference_sft(
                    args.warm_up_text,
                    args.default_spk,
                    stream=True,
                    speed=1.0,
                )
            )
    print("[INFO] warm up end, elapsed={:.1f}s".format(time.perf_counter() - warmup_start), flush=True)


@app.get("/")
def index():
    return FileResponse(WEB_PAGE)


@app.get("/api/config")
def config():
    if cosyvoice is None:
        return JSONResponse({"ready": False})
    spks = []
    try:
        spks = cosyvoice.list_available_spks()
    except Exception:
        spks = []
    return {
        "ready": True,
        "sample_rate": int(cosyvoice.sample_rate),
        "default_spk": server_config.get("default_spk", "03729"),
        "spks": spks,
        "model_path": server_config.get("model_path", ""),
    }


@app.post("/api/stream")
def stream(req: StreamRequest):
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        stream_events(req),
        media_type="application/x-ndjson",
        headers=headers,
    )


@app.get("/health")
def health():
    return {"ok": cosyvoice is not None}


def parse_args():
    parser = argparse.ArgumentParser(description="CosyVoice streaming chunk web probe")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="bind host")
    parser.add_argument("--port", type=int, default=50080, help="bind port")
    parser.add_argument("--warm_up_times", type=int, default=2, help="warm up times before serving")
    parser.add_argument("--default_spk", type=str, default="03729", help="default SFT speaker id")
    parser.add_argument(
        "--warm_up_text",
        type=str,
        default="是的，您现在还有大概1个G的流量。",
        help="warm up text",
    )
    parser.add_argument("--load_om", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile_hift", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    global cosyvoice, server_config

    args = parse_args()
    if not WEB_PAGE.is_file():
        raise FileNotFoundError("web page not found: {}".format(WEB_PAGE))

    apply_cpu_affinity()
    configure_threads()
    torch_npu.npu.set_compile_mode(jit_compile=False)

    cosyvoice = setup_cosyvoice(args)
    warm_up(cosyvoice, args)
    server_config = {
        "default_spk": args.default_spk,
        "model_path": args.model_path,
    }

    print("[INFO] serving on http://{}:{}/".format(args.host, args.port), flush=True)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
