import argparse
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "transformers", "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"))


class HiFTDecodeWrapper(torch.nn.Module):
    def __init__(self, hift):
        super().__init__()
        self.hift = hift
        self.index = hift.istft_params["n_fft"] // 2 + 1

    def forward(self, x, s_stft):
        return self.hift.decode(x=x, s_stft=s_stft, index=self.index)


def parse_args():
    parser = argparse.ArgumentParser(description="Export HiFT decode to ONNX")
    parser.add_argument("--model_dir", required=True,
                        help="CosyVoice2 model directory")
    parser.add_argument("--output", required=True,
                        help="Output ONNX path")
    parser.add_argument("--mel_len", default=50, type=int,
                        help="Dummy mel length for export")
    parser.add_argument("--opset", default=18, type=int,
                        help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    from cosyvoice.cli.cosyvoice import CosyVoice2

    model = CosyVoice2(args.model_dir, load_om=False, fp16=False)
    hift = model.model.hift.eval()
    hift.remove_weight_norm()
    wrapper = HiFTDecodeWrapper(hift).eval()

    device = model.model.device
    mel_len = args.mel_len
    stft_len = 120 * mel_len + 1
    x = torch.randn(1, 80, mel_len, dtype=torch.float32, device=device)
    s_stft = torch.randn(1, 18, stft_len, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x, s_stft),
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["x", "s_stft"],
            output_names=["magnitude", "phase"],
            dynamic_axes={
                "x": {2: "mel_len"},
                "s_stft": {2: "stft_len"},
                "magnitude": {2: "stft_len"},
                "phase": {2: "stft_len"},
            },
        )
    print("saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
