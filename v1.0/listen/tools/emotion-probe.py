"""listen 情感识别基线探针（2026-11-13 初版）

两条路线各跑一遍，只输出原始判断，不做任何"翻译成形容词"的解释：
  A. SenseVoice（FunASR）—— ASR + 内置情感/事件标签
  B. wav2vec2（superb ER）—— 四类情感概率

用法：
  env_ear\\Scripts\\python.exe tools\\emotion-probe.py <wav> [--device cuda:0|cpu]
"""

import sys
import json
import argparse
import io

sys.stdout.reconfigure(encoding="utf-8")


def run_sensevoice(path, device):
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    model = AutoModel(
        model="FunAudioLLM/SenseVoiceSmall",
        trust_remote_code=True,
        hub="hf",
        device=device,
        disable_update=True,
    )
    res = model.generate(
        input=path,
        cache={},
        language="zh",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    out = []
    for r in res:
        raw = r.get("text", "")
        out.append({"raw": raw, "clean": rich_transcription_postprocess(raw)})
    return out


def run_wav2vec2(path, device_index):
    from transformers import pipeline

    clf = pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=device_index,
    )
    return clf(path, top_k=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device_index = 0 if args.device.startswith("cuda") else -1
    result = {"file": args.wav, "device": args.device}

    try:
        result["sensevoice"] = run_sensevoice(args.wav, args.device)
    except Exception as exc:  # noqa: BLE001
        result["sensevoice_error"] = repr(exc)

    try:
        result["wav2vec2_er"] = run_wav2vec2(args.wav, device_index)
    except Exception as exc:  # noqa: BLE001
        result["wav2vec2_error"] = repr(exc)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
