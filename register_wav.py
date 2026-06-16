import os
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

MODEL_DIR = "/home/ma-user/work/test/model/weight/CosyVoice2-0.5B"   # 完整模型目录
SPK2INFO_PATH = os.path.join(MODEL_DIR, "spk2info.pt")

WAV_MAP = {
    "shenhu": "/home/ma-user/work/test/model/CosyVoice-claude/huawei_model/shenhu_prompt_16k.wav",
    "03729": "/home/ma-user/work/test/model/CosyVoice-claude/huawei_model/03729_16k.wav",
}

cosyvoice = CosyVoice2(MODEL_DIR, fp16=False)

if os.path.exists(SPK2INFO_PATH):
    spk2info = torch.load(SPK2INFO_PATH, map_location="cpu")
else:
    spk2info = {}

for spk_id, wav_path in WAV_MAP.items():
    speech_16k = load_wav(wav_path, 16000)
    embedding = cosyvoice.frontend._extract_spk_embedding(speech_16k)
    spk2info[spk_id] = {"embedding": embedding}
    print(f"registered: {spk_id}")

torch.save(spk2info, SPK2INFO_PATH)
print("saved:", SPK2INFO_PATH, "keys:", list(spk2info.keys()))