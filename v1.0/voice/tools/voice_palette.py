# -*- coding: utf-8 -*-
"""语气指纹：给一段音频打「它是什么语气」的量化标签，并按目标语气挑样本。

思路来自 ATRIE/merry-atri 的 reference_library.json（情感 → 推荐参考音频），
但他们那个库没开源，而且我们是 UTAU 拼接连 RVC，没有 zero-shot 架构，
所以不能照抄「换参考音频重合成」——只能搬「按语气挑样本」这个决策层。

这里的指纹全部是可测的物理量，不是猜的：
  f0_med      基频中位数（Hz）      —— 决定低沉/高亢
  f0_iqr      基频四分位距（音分）  —— 决定语调起伏大不大
  tilt        频谱倾斜（dB/oct）    —— 负得多=厚/暗，接近 0=薄/亮
  hnr         谐噪比（dB）          —— 低=气声重，高=干净
  dyn         能量动态范围（dB）    —— 大=有强弱对比
  rate        每秒音节数            —— 决定急促/舒缓
  tail_slope  句尾音高斜率（音分/秒）—— 负=句尾往下掉（叹气感）
  onset_slope 起音上升时间（ms）    —— 长=慢半拍起音（慵懒）

用法：
  # 打指纹
  python tools/voice_palette.py fingerprint a.wav b.wav c.wav

  # 按目标语气挑（给一段「目标语气」的参考，或直接给数值目标）
  python tools/voice_palette.py pick --target sleepy --pool ref_speak ref4

  # 导出成 json 供别的脚本用
  python tools/voice_palette.py fingerprint --json out.json ref_speak/*.wav
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import librosa
import numpy as np

SR = 22050


# ---------------------------------------------------------------- 基础测量

def _load(path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=SR, mono=True)
    # 去掉首尾静音，否则所有指标都被空白稀释
    y, _ = librosa.effects.trim(y, top_db=35)
    return y, sr


def _f0_stats(y: np.ndarray, sr: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    f0, voiced, _ = librosa.pyin(
        y, fmin=70, fmax=800, sr=sr, frame_length=2048, hop_length=256
    )
    ok = voiced & np.isfinite(f0) & (f0 > 0)
    if not ok.any():
        return 0.0, 0.0, np.array([]), np.array([])
    v = f0[ok]
    med = float(np.median(v))
    cents = 1200 * np.log2(v / med)
    iqr = float(np.percentile(cents, 75) - np.percentile(cents, 25))
    return med, iqr, v, f0


def _tilt(y: np.ndarray, sr: int, lo: float = 100.0, hi: float = 6000.0) -> float:
    """频谱倾斜 dB/oct。只取能量前 60% 的帧，避开静音与尾音。"""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    e = S.sum(axis=0)
    if e.size == 0:
        return 0.0
    keep = e >= np.percentile(e, 40)
    mag = S[:, keep].mean(axis=1) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= lo) & (freqs <= hi)
    if band.sum() < 4:
        return 0.0
    lx = np.log2(freqs[band])
    ly = 20 * np.log10(mag[band])
    # 稳健一次拟合，别让单点尖峰带偏
    A = np.vstack([lx - lx.mean(), np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(coef[0])


def _hnr(y: np.ndarray, sr: int) -> float:
    """用自相关峰估谐噪比。低于 0 的截断到 0。"""
    frame = int(0.04 * sr)
    hop = frame // 2
    vals = []
    for i in range(0, max(1, len(y) - frame), hop):
        seg = y[i:i + frame]
        if np.max(np.abs(seg)) < 1e-4:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lo = int(sr / 800)
        hi = min(int(sr / 70), len(ac) - 1)
        if hi <= lo + 1:
            continue
        peak = float(np.max(ac[lo:hi]))
        peak = min(max(peak, 1e-6), 0.999999)
        vals.append(10 * np.log10(peak / (1 - peak)))
    return float(np.median(vals)) if vals else 0.0


def _dyn(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms = rms[rms > 1e-5]
    if rms.size < 4:
        return 0.0
    return float(20 * np.log10(np.percentile(rms, 95) / (np.percentile(rms, 10) + 1e-9)))


def _rate(y: np.ndarray, sr: int) -> float:
    """每秒音节数：用能量包络的峰来数，音节量级。"""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    peaks = librosa.util.peak_pick(
        env, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.3, wait=8
    )
    dur = len(y) / sr
    return float(len(peaks) / dur) if dur > 0.2 else 0.0


def _tail_slope(f0: np.ndarray, voiced: np.ndarray, sr: int) -> float:
    """最后 25% 有声段的音高斜率（音分/秒）。负=句尾往下掉。"""
    idx = np.where(voiced & np.isfinite(f0) & (f0 > 0))[0]
    if idx.size < 12:
        return 0.0
    tail = idx[int(idx.size * 0.75):]
    if tail.size < 6:
        return 0.0
    v = f0[tail]
    med = float(np.median(v))
    if med <= 0:
        return 0.0
    cents = 1200 * np.log2(v / med)
    hop_s = 256 / sr
    t = np.arange(tail.size) * hop_s
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, cents, rcond=None)
    return float(coef[0])


def _onset_slope(y: np.ndarray, sr: int) -> float:
    """平均起音上升时间（ms）：局部能量从谷底爬到峰值花多久。长=慢半拍。

    做法：先算 RMS 包络，用 find_peaks(prominence) 找局部极大，
    再从该峰往前找谷底，量谷底到峰爬了多久。

    为什么不用 librosa.util.peak_pick：说话/长音的能量是连续的，
    局部均值贴近峰值，peak_pick 的 delta 判据会一个峰都找不到（实测 peaks=0）。
    find_peaks 用 prominence 衡量「这个峰比周围高出多少」，对连续能量更稳。
    """
    from scipy.signal import find_peaks

    hop = 128
    env = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    if env.size < 12:
        return 0.0
    env = np.convolve(env, np.ones(3) / 3, mode="same")
    pk, _ = find_peaks(env, prominence=0.12 * float(env.max()), distance=6)
    if pk.size == 0:
        return 0.0
    hop_s = hop / sr
    rises = []
    for p in pk[:15]:
        back = max(0, p - int(0.20 / hop_s))
        valley = back + int(np.argmin(env[back:p + 1]))
        if p - valley >= 1:
            rises.append((p - valley) * hop_s * 1000)
    return float(np.median(rises)) if rises else 0.0


# ---------------------------------------------------------------- 指纹

def fingerprint(path: str) -> dict:
    y, sr = _load(path)
    if len(y) < sr // 10:
        raise ValueError(f"{path}: 有效音频太短")
    f0_med, f0_iqr, _, f0_all = _f0_stats(y, sr)
    voiced = np.isfinite(f0_all) & (f0_all > 0)
    return {
        "path": path,
        "name": os.path.basename(path),
        "dur": round(len(y) / sr, 2),
        "f0_med": round(f0_med, 1),
        "f0_iqr": round(f0_iqr, 1),
        "tilt": round(_tilt(y, sr), 2),
        "hnr": round(_hnr(y, sr), 1),
        "dyn": round(_dyn(y), 1),
        "rate": round(_rate(y, sr), 2),
        "tail_slope": round(_tail_slope(f0_all, voiced, sr), 1),
        "onset_slope": round(_onset_slope(y, sr), 1),
    }


# ---------------------------------------------------------------- 语气目标

# 每条 = 对物理量的期望方向与权重。不是「情感标签」，是可算的偏好。
# 参考来源：作者的声音定词「偏低沉 / 慵懒 / 云淡风轻」+ merry-atri 的情感参数表。
TARGETS: dict[str, dict] = {
    # 作者定的定妆音：低、慢半拍、尾音懒、气声重
    "weixi": {
        "desc": "微汐定妆：偏低沉、慵懒、云淡风轻",
        "want": {"f0_med": (-1.0, 0.30), "onset_slope": (+1.0, 0.22),
                 "hnr": (-1.0, 0.18), "tail_slope": (-1.0, 0.16),
                 "f0_iqr": (-1.0, 0.14)},
    },
    # 说话场景：不要起伏太大，但要清楚
    "speak": {
        "desc": "说话：平稳、清楚、起伏克制",
        "want": {"f0_iqr": (-1.0, 0.35), "dyn": (-1.0, 0.25),
                 "hnr": (+1.0, 0.22), "rate": (-1.0, 0.18)},
    },
    # 唱歌场景：要有动态、要有起伏
    "sing": {
        "desc": "唱歌：起伏大、有强弱对比",
        "want": {"f0_iqr": (+1.0, 0.35), "dyn": (+1.0, 0.30),
                 "tail_slope": (-1.0, 0.20), "rate": (-1.0, 0.15)},
    },
    # 懒散/困：极慢起音、气声重
    "sleepy": {
        "desc": "慵懒困倦：起音极慢、气声重、音高往下",
        "want": {"onset_slope": (+1.0, 0.30), "hnr": (-1.0, 0.28),
                 "tail_slope": (-1.0, 0.22), "f0_iqr": (-1.0, 0.20)},
    },
    # 精神/活泼：快、亮、干净
    "bright": {
        "desc": "精神活泼：快、亮、干净",
        "want": {"rate": (+1.0, 0.30), "hnr": (+1.0, 0.25),
                 "tilt": (+1.0, 0.25), "f0_med": (+1.0, 0.20)},
    },
}


def _zscore(fps: list[dict], key: str) -> np.ndarray:
    v = np.array([f[key] for f in fps], dtype=float)
    s = v.std()
    if s < 1e-9:
        return np.zeros_like(v)
    return (v - v.mean()) / s


def score_pool(fps: list[dict], target: str) -> list[dict]:
    if target not in TARGETS:
        raise KeyError(f"未知语气 {target}，可选：{', '.join(TARGETS)}")
    spec = TARGETS[target]
    total = np.zeros(len(fps))
    parts = {}
    for key, (sign, weight) in spec["want"].items():
        z = _zscore(fps, key) * sign
        parts[key] = z
        total += z * weight
    out = []
    for i, f in enumerate(fps):
        out.append({**f, "score": round(float(total[i]), 3),
                    "why": {k: round(float(v[i]), 2) for k, v in parts.items()}})
    out.sort(key=lambda r: -r["score"])
    return out


# ---------------------------------------------------------------- CLI

def _collect(pool: list[str]) -> list[str]:
    files: list[str] = []
    for p in pool:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "*.wav"))
        elif any(ch in p for ch in "*?"):
            files += glob.glob(p)
        elif os.path.isfile(p):
            files.append(p)
    return sorted(set(files))


def main() -> None:
    ap = argparse.ArgumentParser(description="语气指纹 / 按语气挑样本")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fingerprint", help="打印每个文件的语气指纹")
    f.add_argument("files", nargs="+")
    f.add_argument("--json", help="把结果写到这个 json")

    p = sub.add_parser("pick", help="按目标语气给候选排序")
    p.add_argument("--target", required=True, choices=list(TARGETS))
    p.add_argument("--pool", nargs="+", required=True, help="目录或通配符")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", help="把结果写到这个 json")

    a = ap.parse_args()

    if a.cmd == "fingerprint":
        files = _collect(a.files)
        if not files:
            print("没找到音频文件", file=sys.stderr)
            sys.exit(1)
        rows = []
        for path in files:
            try:
                rows.append(fingerprint(path))
            except Exception as e:
                print(f"  跳过 {os.path.basename(path)}: {e}", file=sys.stderr)
        hdr = f"{'name':<26}{'dur':>6}{'f0':>7}{'iqr':>7}{'tilt':>7}{'hnr':>7}{'dyn':>6}{'rate':>6}{'tail':>8}{'onset':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['name'][:25]:<26}{r['dur']:>6.1f}{r['f0_med']:>7.0f}{r['f0_iqr']:>7.0f}"
                  f"{r['tilt']:>7.1f}{r['hnr']:>7.1f}{r['dyn']:>6.1f}{r['rate']:>6.2f}"
                  f"{r['tail_slope']:>8.0f}{r['onset_slope']:>7.0f}")
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False, indent=2)
            print(f"\n→ {a.json}")

    elif a.cmd == "pick":
        files = _collect(a.pool)
        if not files:
            print("没找到音频文件", file=sys.stderr)
            sys.exit(1)
        fps = []
        for path in files:
            try:
                fps.append(fingerprint(path))
            except Exception as e:
                print(f"  跳过 {os.path.basename(path)}: {e}", file=sys.stderr)
        if not fps:
            print("没有可用样本", file=sys.stderr)
            sys.exit(1)
        spec = TARGETS[a.target]
        print(f"目标语气 [{a.target}] {spec['desc']}")
        print(f"候选 {len(fps)} 条\n")
        ranked = score_pool(fps, a.target)[:a.top]
        for i, r in enumerate(ranked, 1):
            why = " ".join(f"{k}{v:+.1f}" for k, v in sorted(r["why"].items(), key=lambda x: -abs(x[1])))
            print(f"{i:>2}. {r['score']:>+6.2f}  {r['name']}")
            print(f"      f0={r['f0_med']:.0f}Hz iqr={r['f0_iqr']:.0f}¢ tilt={r['tilt']:+.1f} "
                  f"hnr={r['hnr']:.1f}dB dyn={r['dyn']:.1f}dB rate={r['rate']:.2f}/s "
                  f"tail={r['tail_slope']:+.0f}¢/s onset={r['onset_slope']:.0f}ms")
            print(f"      贡献: {why}")
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump({"target": a.target, "spec": spec, "ranked": ranked}, fh,
                          ensure_ascii=False, indent=2)
            print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
