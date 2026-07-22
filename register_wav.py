import argparse
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "transformers", "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register SFT speaker embeddings into spk2info.pt")
    parser.add_argument("--model_dir", required=True,
                        help="CosyVoice2 model directory")
    parser.add_argument(
        "--spk",
        action="append",
        required=True,
        metavar="SPK_ID=WAV_PATH",
        help="Speaker registration item. Can be passed multiple times.",
    )
    parser.add_argument("--output", default="",
                        help="Output spk2info path. Default: MODEL_DIR/spk2info.pt")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing speaker id")
    return parser.parse_args()


def parse_spk_items(items):
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError("--spk must be formatted as SPK_ID=WAV_PATH: {}".format(item))
        spk_id, wav_path = item.split("=", 1)
        spk_id = spk_id.strip()
        wav_path = wav_path.strip()
        if not spk_id:
            raise ValueError("empty speaker id in --spk {}".format(item))
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(wav_path)
        parsed.append((spk_id, wav_path))
    return parsed


def main():
    args = parse_args()
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    model_dir = os.path.abspath(args.model_dir)
    spk2info_path = args.output or os.path.join(model_dir, "spk2info.pt")
    spk_items = parse_spk_items(args.spk)

    cosyvoice = CosyVoice2(model_dir, fp16=False)
    if os.path.exists(spk2info_path):
        spk2info = torch.load(spk2info_path, map_location="cpu")
    else:
        spk2info = {}

    for spk_id, wav_path in spk_items:
        if spk_id in spk2info and not args.overwrite:
            raise ValueError(
                "speaker id already exists: {}. Use --overwrite to replace it.".format(spk_id))
        speech_16k = load_wav(wav_path, 16000)
        embedding = cosyvoice.frontend._extract_spk_embedding(speech_16k).cpu()
        spk2info[spk_id] = {"embedding": embedding}
        print("registered: {} <- {}".format(spk_id, wav_path), flush=True)

    torch.save(spk2info, spk2info_path)
    print("saved: {} keys={}".format(spk2info_path, sorted(spk2info.keys())),
          flush=True)


if __name__ == "__main__":
    main()
