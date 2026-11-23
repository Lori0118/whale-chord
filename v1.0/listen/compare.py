# -*- coding: utf-8 -*-
"""compare —— 原唱 vs 自己唱的多维度对照。

（2026-11-14 从 whale-chord/tools 迁入；作者在那天指出「不要只靠 ASR，拿原唱同段做多维度对比」，
这套测量就是 listen 的第二个核心能力：不给「好不好听」，只给两段人声差在哪。）

## 六个维度

| 维度 | 量什么 | 为什么重要 |
|---|---|---|
| voiced 占比 | 有声帧比例 | 断得多不多 |
| 颤音 | 速率 Hz / 深度音分 | 真人长音自带 4–8Hz 颤音；没有 = 学不到「唱」 |
| 跳变率 | 10ms 帧间变化 > 半音的比例 | 「一个字一个字蹦」的直接量化 |
| 断音 | 能量掉到中位数以下的帧占比 | 字与字之间是不是切开 |
| 频谱质心 | 亮度 | 嗓子亮还是暗（变调、切片会变暗） |
| 动态 | rms 95/10 分位差（dB） | 音量是否失控 |

## 用法

    python compare.py <原唱.wav> <自己唱1.wav> [自己唱2.wav ...] [--plot out.png]

原唱如果带伴奏，先用 demucs 分出人声（见 README「分轨」一节）：
    python -m demucs --two-stems=vocals -n htdemucs -o out 原曲.wav

## 读法

- 跳变率高 + 断音高 = 拼接/不连贯的唱法；真人原唱通常断音 <10%
- 颤音深度 <20 音分 = 基本没有颤音（单音节数据集训出来的模型就这样）
- 质心比原唱低 30% 以上 = 音色偏暗，多半是切片或变调造成的
- 这几项都是「同一套方法横向比」才有意义，绝对值别外推

依赖：librosa / numpy / matplotlib（画图才需要）。
"""
import argparse
import sys

import librosa
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SR = 44100
HOP = 441


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def f0_track(y):
    f0, voiced, _ = librosa.pyin(y, fmin=110, fmax=700, sr=SR,
                                 frame_length=2048, hop_length=HOP)
    return f0, voiced


def vibrato(f0, min_dur=0.45):
    """稳定段里的颤音速率与深度（与 listen/voice.py 同一套思路）。"""
    if f0 is None:
        return 0.0, 0.0, 0
    v = ~np.isnan(f0)
    segs, start = [], None
    for i, b in enumerate(v):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if (i - start) * HOP / SR >= min_dur:
                segs.append((start, i))
            start = None
    if start is not None and (len(v) - start) * HOP / SR >= min_dur:
        segs.append((start, len(v)))
    rates, depths = [], []
    for a, b in segs:
        seg = f0[a:b]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 30:
            continue
        cents = 1200 * np.log2(seg / np.median(seg))
        d = np.diff(cents)
        sign_change = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
        if len(sign_change) >= 4:
            period = np.median(np.diff(sign_change)) * HOP / SR
            if period > 0.02:
                rates.append(1.0 / period)
        depths.append((np.percentile(cents, 90) - np.percentile(cents, 10)) / 2)
    if not rates:
        return 0.0, float(np.median(depths)) if depths else 0.0, len(segs)
    return float(np.median(rates)), float(np.median(depths)), len(segs)


def continuity(f0):
    v = ~np.isnan(f0)
    pairs = v[:-1] & v[1:]
    if pairs.sum() == 0:
        return 0.0, 0.0, 0.0
    st = 12 * np.log2(f0[1:][pairs] / f0[:-1][pairs])
    return float(np.mean(np.abs(st) > 0.5)), float(np.median(np.abs(st))), float(np.percentile(np.abs(st), 95))


def gaps(y):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    thr = np.median(rms) * 0.15
    return float(np.mean(rms < thr))


def timbre(y):
    cent = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP)[0]
    return float(np.median(cent))


def dyn(y):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    rms = rms[rms > 1e-5]
    if len(rms) == 0:
        return 0.0
    return float(20 * np.log10(np.percentile(rms, 95) / max(np.percentile(rms, 10), 1e-9)))


def identity(y, f0):
    """音色同一性口径——跨表比「像不像这把嗓子」只能用这一套。

    与 ml/tools/corpus_probe.py 是同一套算法：能量前 40% 的帧、功率谱、加权质心。
    （2026-11-15 加。上面 report 里的 centroid 是 librosa 幅度加权全帧中位数，
    两套能差 5–13 倍：E1-raw.wav 在这个文件里报 2226，在 corpus_probe 里报 252，
    在 rec_check 里报 3345。谁都不是错的，混着比就是错的。）
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    e = S.sum(axis=0)
    keep = e >= np.percentile(e, 60)
    if keep.sum() < 8:
        keep = np.ones(e.shape, dtype=bool)
    P = (S[:, keep] ** 2).mean(axis=1) + 1e-12
    tot = P.sum()
    f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    return {
        "cen_energy": float((P * f).sum() / tot),
        "hf_pct": float(P[f > 2500].sum() / tot * 100),
        "mid_pct": float(P[(f >= 300) & (f < 800)].sum() / tot * 100),
        "f0_med_hz": float(np.median(f0v)) if f0v.size else 0.0,
    }


def report(path):
    y = load(path)
    f0, voiced = f0_track(y)
    vr, vd, nseg = vibrato(f0)
    jr, jm, j95 = continuity(f0)
    ident = identity(y, f0)
    return {
        "file": path.replace("\\", "/").split("/")[-1],
        "dur": len(y) / SR,
        "voiced": float(np.mean(voiced)),
        "vib_rate": vr,
        "vib_depth": vd,
        "n_stable_seg": nseg,
        "jump_rate": jr,
        "jump_med": jm,
        "jump_p95": j95,
        "gap_ratio": gaps(y),
        "centroid": timbre(y),
        "dyn_db": dyn(y),
        "rms": float(np.sqrt(np.mean(y ** 2))),
        "peak": float(np.abs(y).max()),
        **ident,
    }


def plot(files, names, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cols = ["#222222", "#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for i, f in enumerate(files):
        y = load(f)
        f0, _ = f0_track(y)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP)[0]
        n = min(len(f0), len(rms), len(cent))
        t = np.linspace(0, 1, n)
        semi = 12 * np.log2(f0[:n] / 440.0)
        db = 20 * np.log10(np.maximum(rms[:n], 1e-6))
        db -= np.percentile(db, 95)
        c = cols[i % len(cols)]
        axes[0].plot(t, semi, c=c, lw=1.1, label=names[i], alpha=0.9)
        axes[1].plot(t, db, c=c, lw=0.9, label=names[i], alpha=0.9)
        axes[2].plot(t, cent[:n] / 1000, c=c, lw=0.9, label=names[i], alpha=0.9)

    axes[0].set_ylabel("音高（半音，相对 A4）")
    axes[0].set_title("原唱 vs 自己唱：音高轨迹 / 能量包络 / 频谱亮度", fontsize=13)
    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("能量（dB，峰值归一）")
    axes[1].legend(loc="lower right", fontsize=9, ncol=2)
    axes[1].grid(alpha=0.25)
    axes[2].set_ylabel("频谱质心（kHz）")
    axes[2].set_xlabel("相对时间（各段自身归一化到 0–1）")
    axes[2].legend(loc="upper right", fontsize=9, ncol=2)
    axes[2].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print("图已保存:", out)


def main():
    ap = argparse.ArgumentParser(description="原唱 vs 自己唱的多维度对照")
    ap.add_argument("files", nargs="+", help="第一个是原唱（或参照），后面是待比较的版本")
    ap.add_argument("--plot", help="输出三联对照图的路径（.png）")
    ap.add_argument("--no-plot", action="store_true", help="只打印数字不画图")
    a = ap.parse_args()
    files = a.files
    rows = [report(p) for p in files]
    names = [r["file"].rsplit(".", 1)[0] for r in rows]

    print("\n=== 唱法对照 ===")
    print("%-26s %6s %7s %8s %7s %6s %8s %8s %8s %6s %8s %7s" % (
        "file", "dur", "voiced", "颤速率", "颤深¢", "稳定段", "跳变率",
        "跳变中¢", "跳变p95", "断音", "质心Hz", "动态dB"))
    for r in rows:
        print("%-26s %6.2f %7.2f %8.2f %7.0f %6d %8.2f %8.1f %8.1f %6.2f %8.0f %7.1f" % (
            r["file"][:26], r["dur"], r["voiced"], r["vib_rate"], r["vib_depth"],
            r["n_stable_seg"], r["jump_rate"], r["jump_med"], r["jump_p95"],
            r["gap_ratio"], r["centroid"], r["dyn_db"]))
    print("\n[读法] 跳变率 = 10ms 帧间变化超半音的比例（越高越「蹦」）；断音 = 能量低于中位数的帧占比；"
          "颤速率真人 4–8Hz、深 20–120 音分。")

    print("\n=== 音色同一性（跨表只认这一套）===")
    print("%-26s %10s %8s %9s %9s" % ("file", "能量质心Hz", ">2.5k%", "300-800%", "f0中位Hz"))
    for r in rows:
        print("%-26s %10.0f %7.2f%% %8.1f%% %9.0f" % (
            r["file"][:26], r["cen_energy"], r["hf_pct"], r["mid_pct"], r["f0_med_hz"]))
    print("[口径] 能量前 40% 帧的功率谱加权（与 ml/tools/corpus_probe.py 完全同一套，E1-raw 在此报 252Hz）。"
          "\n       上面「质心Hz」是另一套（librosa 幅度加权全帧中位数，E1-raw 报 2226）——两套不许跨表比。")

    if a.plot and not a.no_plot:
        plot(files, names, a.plot)


if __name__ == "__main__":
    main()
