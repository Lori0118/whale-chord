# -*- coding: utf-8 -*-
"""录音体检 —— 素材入库前先过这一道，别拿坏录音去判断音色。

2026-11-14 晚的教训：作者录的《腐草为萤》64 秒，唱法其实很稳（颤音 7.7Hz/60¢、
跳变率 0.02），但 2.5kHz 以上只剩 0.28% 能量（同一套算法量真人干声是 7.8%，差 28 倍），
录进来就已经是闷的。RVC 忠实地把这份「闷」原样带过去了（质心 1194→1195Hz），
所以那条素材不能用来判断 weixi2 的音色 —— 会冤枉模型。

用法：
    python rec_check.py <音频文件> [更多文件...] [--blocks 5]

读法（判据来自 2026-11-14 的两条基线：雨爱原唱干声 7.8%、RVC 纯换声 1.5% 量级）：
    >2.5k ≥ 3%  可以当素材
    1–3%        偏闷，能验管线，别用来评音色
    < 1%        太闷，重录
另外看：电平（活跃段 rms 别低于 -30dB）、峰值（>0.99 有削波）、
底噪（活跃-静音差 <20dB 说明环境吵或降噪把声音一起吃掉了）。
"""
import argparse
import os
import subprocess
import sys
import tempfile

import librosa
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SR = 44100
HOP = 441


def load(path):
    """librosa 读不了的容器（m4a/aac）先用 ffmpeg 转 wav。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".flac"):
        y, _ = librosa.load(path, sr=SR, mono=True)
        return y
    tmp = os.path.join(tempfile.gettempdir(), "rec_check_tmp.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1",
                    "-ar", str(SR), "-c:a", "pcm_s16le", tmp], check=True)
    y, _ = librosa.load(tmp, sr=SR, mono=True)
    return y


def analyse(path):
    y = load(path)
    if y.size == 0:
        return None
    dur = len(y) / SR
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    peak = float(np.abs(y).max())
    clip = int(np.sum(np.abs(y) > 0.99))
    act = rms > rms.max() * 0.1
    rms_act = float(librosa.amplitude_to_db(np.array([rms[act].mean()]), ref=1.0)[0]) if act.any() else -99.0
    floor = float(librosa.amplitude_to_db(np.array([np.percentile(rms, 10)]), ref=1.0)[0])
    snr = rms_act - floor

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    P = (S ** 2).mean(axis=1)
    tot = P.sum() + 1e-12
    hf = P[f > 2500].sum() / tot * 100
    mid = P[(f >= 300) & (f < 800)].sum() / tot * 100
    cen = float(librosa.feature.spectral_centroid(S=S, sr=SR)[0].mean())

    f0, _, _ = librosa.pyin(y, fmin=80, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)
    voiced = float(np.mean(~np.isnan(f0))) if f0 is not None else 0.0
    f0med = float(np.nanmedian(f0)) if f0 is not None and np.any(~np.isnan(f0)) else 0.0

    # 分段看高频，判断是不是整条都闷 / 只有某段被挡
    blocks = []
    n = int(5 * SR)
    for s in range(0, len(y) - n + 1, n):
        Sb = np.abs(librosa.stft(y[s:s + n], n_fft=2048, hop_length=HOP))
        Pb = (Sb ** 2).mean(axis=1)
        blocks.append(float(Pb[f > 2500].sum() / (Pb.sum() + 1e-12) * 100))

    return dict(dur=dur, rms_act=rms_act, peak=peak, clip=clip, snr=snr,
                hf=hf, mid=mid, cen=cen, voiced=voiced, f0=f0med, blocks=blocks)


def verdict(r):
    issues = []
    if r["hf"] < 1.0:
        issues.append(f"太闷（>2.5k 只有 {r['hf']:.2f}%）→ 重录")
    elif r["hf"] < 3.0:
        issues.append(f"偏闷（>2.5k {r['hf']:.2f}%）→ 能验管线，别评音色")
    if r["rms_act"] < -30:
        issues.append(f"太小声（活跃段 {r['rms_act']:.1f}dBFS）")
    if r["peak"] > 0.99 or r["clip"] > 0:
        issues.append(f"削波 {r['clip']} 处")
    if r["snr"] < 20:
        issues.append(f"底噪偏高（信噪比 {r['snr']:.0f}dB）")
    return issues or ["可以当素材"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--blocks", type=float, default=5.0)
    a = ap.parse_args()
    for path in a.files:
        r = analyse(path)
        name = os.path.basename(path)
        if r is None:
            print(f"{name}: 读不出来")
            continue
        print(f"\n=== {name} ===")
        print(f"  时长 {r['dur']:.1f}s  活跃段 {r['rms_act']:.1f}dBFS  峰值 {r['peak']:.2f}  信噪比 {r['snr']:.0f}dB")
        print(f"  >2.5kHz {r['hf']:.2f}%   300–800Hz {r['mid']:.1f}%   质心 {r['cen']:.0f}Hz")
        print(f"  f0 中位 {r['f0']:.0f}Hz  有声帧 {r['voiced']*100:.0f}%")
        if r["blocks"]:
            bs = " ".join(f"{b:.1f}" for b in r["blocks"])
            print(f"  分段高频%: {bs}")
        print(f"  判定: {'；'.join(verdict(r))}")


if __name__ == "__main__":
    main()
