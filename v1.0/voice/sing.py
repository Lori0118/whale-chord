#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sing.py —— 用 UTAU 音源把「音符 + 拼音歌词」唱成 wav。

不依赖 OpenUtau（它没有命令行导出），自己干这些事：
  1. 读 oto.ini，拿到每个音节 / 每个「元音→辅音」过渡的取样区间
  2. 按最近音高切片、变调、拉伸
  3. 给长音加颤音（电子模式加得更狠）
  4. 用 _VC 里的过渡样本补上音节之间的连读
  5. 加一点微走音和时值抖动，让节奏不精确到毫秒
  6. 按 preutterance 对齐、交叉淡化拼起来

用法：
    python sing.py score.json out.wav [--mode human|cyber]

score.json 形如：
    [{"midi": 60, "dur": 0.5, "lyric": "wo"}, ...]

--mode human（默认）：颤音 5.2Hz ±18 音分、微走音 ±6 音分、时值抖动 ±12ms、带连读
--mode cyber：颤音 6.5Hz ±45 音分、不走音、不抖动、不带连读（要的就是电子味）
"""
from __future__ import annotations

import io
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None

import pinyin_rules

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.environ.get(
    "WHALE_VOICEBANK",
    os.path.join(HERE, "voicebank", "hitsuboku-kumi-zh-mirror-main"),
)
SR = 44100

PITCH_HZ = {"A3": 220.0, "A4": 440.0, "D5": 587.33}
PITCH_MIDI = {"A3": 57, "A4": 69, "D5": 74}

INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]

# 音源的元音词表只有 27 种（a/ai/an/ang/ao/e/eh/ei/en/eng/er/ong/ou/
# wo/wu/yan/ye/yi/yin/ying/yong/yu/yuan/yue/yuen/-i/h-i），
# 缺 ing/in/ui/iu/ie/ian/uan/ue 这些，得映射到它的同类写法上。
FINAL_MAP = {
    "ing": "ying", "in": "yin", "i": "yi", "u": "wu", "uo": "wo",
    "ue": "yue", "ve": "yue", "ve": "yue", "uan": "yuan", "un": "yuen",
    "ie": "ye", "ian": "yan", "iang": "yang", "iong": "ong",
    "ui": "ei", "iu": "ou", "v": "yu", "iong": "yong",
}

MODES = {
    "human": dict(vib_hz=3.4, vib_cents=28.0, drift_cents=6.0, jitter_ms=12.0, vib_from=0.50),
    "cyber": dict(vib_hz=6.5, vib_cents=45.0, drift_cents=0.0, jitter_ms=0.0, vib_from=0.15),
    # speak：说话用。不颤、时值抖动大一点，其余交给四声的调形（tools/apply_tones.py）
    "speak": dict(vib_hz=0.0, vib_cents=0.0, drift_cents=1.5, jitter_ms=25.0, vib_from=9.0),
}


# ---------- oto.ini ----------

def parse_oto(path: str) -> dict:
    table = {}
    if not os.path.isfile(path):
        return table
    for line in io.open(path, encoding="utf-8-sig", errors="replace"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        fn, rest = line.split("=", 1)
        parts = rest.split(",")
        if len(parts) < 6:
            continue
        try:
            offset, cons, cutoff, pre, ovl = (float(x) for x in parts[1:6])
        except ValueError:
            continue
        table[parts[0]] = (fn, offset, cons, cutoff, pre, ovl)
    return table


def load_bank():
    return (parse_oto(os.path.join(BANK, "_CV", "oto.ini")),
            parse_oto(os.path.join(BANK, "_VC", "oto.ini")))


def pitch_of(alias: str):
    tail = alias.rsplit("_", 1)
    if len(tail) == 2 and tail[1] in PITCH_HZ:
        return tail[1]
    return None


# ---------- 拼音切分 ----------

def split_syllable(s: str):
    for ini in INITIALS:
        if s.startswith(ini) and len(s) > len(ini):
            return ini, s[len(ini):]
    return "", s


# ---------- 选样 ----------

def pick(table: dict, key: str, target_midi: float):
    """在 table 里找 key 对应的样本，取音高最近的一档（平手时优先降调）。"""
    cands = []
    for alias, info in table.items():
        p = pitch_of(alias)
        if p is None:
            continue
        raw = alias.rsplit("_", 1)[0].strip()
        if raw == key:
            form = 0
        elif raw == "- " + key:
            form = 1
        elif raw == key + "+":
            form = 2
        else:
            continue
        cands.append((form, PITCH_MIDI[p], alias, info))
    if not cands:
        return None
    # 带 "- " 前缀的是「含声母的完整音节」，不带前缀的只有元音部分。
    # 音符起头必须用含声母的那一档，否则 /d/ /zh/ /g/ 全被吃掉（"当"会听成"张"）。
    if os.environ.get("WHALE_ONSET", "1") != "0":
        full = [c for c in cands if c[0] == 1]
        if full:
            cands = full
    best = min(c[0] for c in cands)
    cands = [c for c in cands if c[0] == best]
    force = os.environ.get("WHALE_FORCE_PITCH", "").strip()
    if force in PITCH_MIDI:
        # 强制整句用同一档源样本，避免换样本时音色跳变（代价：变调幅度变大）
        forced = [c for c in cands if c[1] == PITCH_MIDI[force]]
        if forced:
            cands = forced
    cands.sort(key=lambda c: (abs(c[1] - target_midi), 0 if c[1] >= target_midi else 1))
    return cands[0][2], cands[0][3]


# ---------- 音频 ----------

def stretch(y: np.ndarray, ratio: float) -> np.ndarray:
    if abs(ratio - 1.0) < 0.01 or len(y) < 1024:
        return y
    if librosa is None:
        n = max(1, int(len(y) * ratio))
        return np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
    return librosa.effects.time_stretch(y, rate=1.0 / ratio)


def shift_pitch(y: np.ndarray, semitones: float) -> np.ndarray:
    if abs(semitones) < 0.05 or librosa is None:
        return y
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=semitones)
def spectral_envelope(mag: np.ndarray, sr: int) -> np.ndarray:
    """用倒谱法估频谱包络（逐帧），返回与 mag 同形状的对数包络。"""
    logmag = np.log(mag + 1e-10)
    cep = np.fft.irfft(logmag, axis=0)
    lifter = np.zeros(cep.shape[0])
    q = int(0.0025 * sr)          # 2.5ms 以内的倒谱分量 = 包络
    lifter[:q] = 1.0
    lifter[-q + 1:] = 1.0
    env = np.fft.rfft(cep * lifter[:, None], axis=0)
    return env.real


def shift_pitch_formant(y: np.ndarray, semitones: float,
                        n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """变调但保留原共振峰位置（音高升高时不会变成童声）。"""
    if abs(semitones) < 0.05 or librosa is None or len(y) < n_fft * 2:
        return y
    # 1) 先普通变调
    shifted = librosa.effects.pitch_shift(y, sr=SR, n_steps=semitones)
    n = min(len(y), len(shifted))
    if n < n_fft * 2:
        return shifted
    y0, y1 = y[:n], shifted[:n]

    # 2) 分别取包络
    S0 = np.abs(librosa.stft(y0, n_fft=n_fft, hop_length=hop))
    S1 = np.abs(librosa.stft(y1, n_fft=n_fft, hop_length=hop))
    E0 = spectral_envelope(S0, SR)
    E1 = spectral_envelope(S1, SR)

    # 3) 把包络换回去，相位沿用变调后的
    S1c = librosa.stft(y1, n_fft=n_fft, hop_length=hop)
    k = float(os.environ.get("WHALE_FORMANT_K", "1.0") or 1.0)
    if abs(k - 1.0) > 0.02:
        # 共振峰整体缩放：把包络沿频率轴重采样（k<1 = 下移 = 声道更长 = 更厚重）
        n_bins = E0.shape[0]
        src = np.clip(np.arange(n_bins) / k, 0, n_bins - 1)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, n_bins - 1)
        w = (src - lo)[:, None]
        E0 = E0[lo] * (1 - w) + E0[hi] * w
    ratio = np.exp(np.clip(E0 - E1, -6.0, 6.0))
    S1c = S1c * ratio
    out = librosa.istft(S1c, hop_length=hop, length=n)
    return out.astype(np.float32)


def add_vibrato(y: np.ndarray, rate_hz: float, depth_cents: float,
                fade_in: float) -> np.ndarray:
    n = len(y)
    if n < 2048 or depth_cents <= 0:
        return y
    t = np.arange(n) / SR
    amp = np.clip((t - fade_in) / max(fade_in, 0.05), 0.0, 1.0)
    ratio = 2.0 ** (depth_cents / 1200.0)
    dev = (ratio - 1.0) * amp * np.sin(2 * np.pi * rate_hz * t)
    idx = np.cumsum(1.0 + dev)
    idx = idx[idx < n - 1]
    if len(idx) < 256:
        return y
    return np.interp(idx, np.arange(n), y)


def add_portamento(y: np.ndarray, start_semitones: float, tau: float = 0.055) -> np.ndarray:
    """在音头做一个向目标音高滑过去的过程——真人换音不是跳过去的。

    start_semitones 相对目标音高的初始偏差（负=从低往上滑），按 exp(-t/tau) 衰减。
    """
    n = len(y)
    if n < 1024 or abs(start_semitones) < 0.25:
        return y
    t = np.arange(n) / SR
    dev = start_semitones * np.exp(-t / max(tau, 0.01))
    ratio = 2.0 ** (dev / 12.0)
    idx = np.cumsum(ratio)
    keep = idx < n - 1
    idx = idx[keep]
    if len(idx) < 256:
        return y
    return np.interp(idx, np.arange(n), y)


def get_segment(info, folder: str):
    fn, offset, cons, cutoff, pre, ovl = info
    path = os.path.join(BANK, folder, fn)
    if not os.path.isfile(path):
        return None, pre
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR and librosa is not None:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    s = max(0, int((offset - pre) / 1000 * SR))
    e = min(len(y), int((offset - cutoff) / 1000 * SR))
    if e - s < 512:
        return None, pre
    return y[s:e], pre


def render_cv(cv: dict, syl: str, dur: float, midi: float, cfg: dict, prev_midi=None):
    """合成一个音节：**声母段不拉长，只拉长元音段**。

    oto.ini 的 pre（先行发声）标的是「元音起点」，样本开头那截就是声母。
    以前把整段一起 stretch，声母被拖长成一小口气，听感就是「每个字前停了一下」；
    按能量阈值剪静音又会把 /d/ 这类短爆破音剪掉（「当」变「让」）。
    """
    hit = pick(cv, syl, midi)
    if hit is None:
        return None, 0.0
    alias, info = hit
    seg, pre = get_segment(info, "_CV")
    if seg is None:
        return None, 0.0

    src_hz = PITCH_HZ[pitch_of(alias)]
    tgt_hz = 440.0 * 2 ** ((midi - 69) / 12)
    semis = 12 * np.log2(tgt_hz / src_hz)
    if os.environ.get("WHALE_FORMANT", "1") != "0":
        seg = shift_pitch_formant(seg, semis)
    else:
        seg = shift_pitch(seg, semis)

    # 定位元音起点：用 pyin 找第一个稳定浊音帧。辅音（f/x/r/s…）在它前面，
    # 这些样本的辅音长达 60~475ms（带气声），整段铺进去会盖住上一个音、听成「停顿」。
    # 只保留元音起点前 60ms 的那一小截（辅音到元音的过渡最关键的部位）。
    cons_len = 0
    if os.environ.get("WHALE_SPLIT", "1") != "0" and len(seg) > 4096 and librosa is not None:
        try:
            _f0, _vf, _ = librosa.pyin(seg, fmin=90, fmax=900, sr=SR, frame_length=2048)
            voiced = np.where(_vf & np.isfinite(_f0))[0]
            if len(voiced):
                vstart = int(voiced[0] * 512)
                if vstart > 1024:
                    cons_len = min(int(0.060 * SR), vstart)
        except Exception:
            cons_len = 0
    if cons_len > 0:
        cons = seg[vstart - cons_len:vstart]
        vow = seg[vstart:]
    else:
        cons, vow = None, seg

    if cons is not None and len(vow) > 256:
        vow_len = int((dur + 0.02) * SR)
        vow = stretch(vow, vow_len / len(vow)) if len(vow) > 256 else vow
        vow = vow[:vow_len] if len(vow) > vow_len else np.pad(vow, (0, max(0, vow_len - len(vow))))
        f = min(int(0.008 * SR), len(cons), len(vow))
        if f > 32:
            cons = cons.copy(); vow = vow.copy()
            cons[-f:] *= np.linspace(1.0, 0.0, f)
            vow[:f] *= np.linspace(0.0, 1.0, f)
        out = np.concatenate([cons, vow])
    else:
        want = int((dur + pre / 1000) * SR)
        if want > 256:
            out = stretch(seg, want / len(seg))
            out = out[:want] if len(out) > want else np.pad(out, (0, want - len(out)))
        else:
            out = seg

    if prev_midi is not None:
        out = add_portamento(out, prev_midi - midi)
    if dur >= cfg["vib_from"]:
        out = add_vibrato(out, cfg["vib_hz"], cfg["vib_cents"], cfg["vib_from"])
    lead = cons_len / SR if cons_len > 0 else pre / 1000
    return out, lead


def render_vc(vc: dict, prev: str, nxt: str, midi_next: float):
    """连读：用拼音规则算出「前一个的元音 + 后一个的声母」该用哪段样本。

    只取尾部 80ms（纯辅音），保共振峰变调——否则辅音被变调成刺耳的爆音
    （作者听到的「喇叭声」）。音量在调用处降到 0.22。
    """
    key = pinyin_rules.vc_key(prev, nxt)
    hit = pick(vc, key, midi_next)
    if hit is None:
        return None
    alias, info = hit
    seg, _pre = get_segment(info, "_VC")
    if seg is None or len(seg) < 1024:
        return None
    keep = min(len(seg), int(float(os.environ.get("WHALE_VC_LEN", "0.08")) * SR))
    seg = seg[-keep:]
    src_hz = PITCH_HZ[pitch_of(alias)]
    tgt_hz = 440.0 * 2 ** ((midi_next - 69) / 12)
    semis = 12 * np.log2(tgt_hz / src_hz)
    if os.environ.get("WHALE_FORMANT", "1") != "0":
        return shift_pitch_formant(seg, semis)
    return shift_pitch(seg, semis)


def ensure(buf: np.ndarray, end: int) -> np.ndarray:
    if end > len(buf):
        return np.concatenate([buf, np.zeros(end - len(buf) + SR, dtype=np.float32)])
    return buf


def highpass(y: np.ndarray, cutoff: float = 85.0) -> np.ndarray:
    """二阶高通，去掉低频隆隆声（否则整段会像「低吼」）。"""
    if cutoff <= 0 or librosa is None or len(y) < 1024:
        return y
    try:
        import scipy.signal as ss
        sos = ss.butter(2, cutoff / (SR / 2), btype="highpass", output="sos")
        return ss.sosfilt(sos, y).astype(np.float32)
    except Exception:
        S = librosa.stft(y, n_fft=2048)
        fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
        ramp = np.clip((fr - cutoff * 0.6) / (cutoff * 0.4), 0.0, 1.0)
        return librosa.istft(S * ramp[:, None], length=len(y)).astype(np.float32)


def measure_tilt(y: np.ndarray, lo: float = 100.0, hi: float = 6000.0) -> float:
    """量频谱倾斜（dB/oct）：STFT 帧平均，**只取能量高的帧**，避开静音段把曲线拉平。"""
    S = np.abs(librosa.stft(y, n_fft=4096)) ** 2
    frame_energy = S.sum(axis=0)
    if len(frame_energy) < 4:
        return 0.0
    thresh = np.percentile(frame_energy, 60)
    keep = frame_energy >= thresh
    ps = 10 * np.log10(S[:, keep].mean(axis=1) + 1e-12)
    fr = librosa.fft_frequencies(sr=SR, n_fft=4096)
    sel = (fr > lo) & (fr < hi)
    if sel.sum() < 32:
        return 0.0
    return float(np.polyfit(np.log2(fr[sel]), ps[sel], 1)[0])


def match_tilt(y: np.ndarray, target_db_per_oct: float = -8.5,
               passes: int = 3) -> np.ndarray:
    """把频谱倾斜拉到目标值附近，迭代几次逼近。"""
    if len(y) < 8192 or librosa is None:
        return y
    out = y.astype(np.float32).copy()
    for _ in range(max(1, passes)):
        cur = measure_tilt(out)
        delta = target_db_per_oct - cur
        if abs(delta) < 0.4:
            break
        n = len(out)
        spec = np.fft.rfft(out)
        freqs = np.fft.rfftfreq(n, 1.0 / SR)
        gain = np.ones_like(freqs)
        mask = freqs > 60
        gain[mask] = (freqs[mask] / 1000.0) ** (delta / 6.02)
        # 限幅：抬低频最多 +4dB、削高频最多 -8dB，否则低频会暴涨成「低吼」
        gain = np.clip(gain, 10 ** (-8 / 20), 10 ** (4 / 20))
        out = np.fft.irfft(spec * gain, n=n).astype(np.float32)
    return out


def synth(score: list, out: str, mode: str = "human", seed: int = 7):
    cfg = MODES.get(mode, MODES["human"])
    rnd = random.Random(seed)
    cv, vc = load_bank()
    total = sum(n["dur"] for n in score) + 2.5
    buf = np.zeros(int(total * SR) + SR, dtype=np.float32)
    # 开头留白：唱歌时留 0.6s 让片子好剪；说话时设 WHALE_LEAD=0，
    # 否则 RVC 会在那段静音上凭空长出一个「啊」。
    t = float(os.environ.get("WHALE_LEAD", "0.6"))
    placed = 0
    vc_used = 0
    prev_syl = None
    prev_midi = None

    for note in score:
        syl = (note.get("lyric") or "").strip().lower()
        if not syl:
            t += note["dur"]
            prev_syl = None
            continue

        midi = note["midi"] + float(os.environ.get("WHALE_TRANSPOSE", "0")) \
            + rnd.uniform(-1, 1) * cfg["drift_cents"] / 100.0
        dur = max(0.08, note["dur"] + rnd.uniform(-1, 1) * cfg["jitter_ms"] / 1000.0)

        # 过渡音：铺在下一个音起点之前，和它交叉
        if prev_syl and mode in ("human", "speak"):
            seg = render_vc(vc, prev_syl, syl, midi)
            if seg is not None:
                start = int((t - len(seg) / SR) * SR)
                if start >= 0:
                    seg = seg.copy()
                    fade = min(len(seg) // 3, int(0.03 * SR))
                    if fade > 32:
                        seg[:fade] *= np.linspace(0, 1, fade)
                        seg[-fade:] *= np.linspace(0.35, 1, fade)
                    end = start + len(seg)
                    buf = ensure(buf, end)
                    buf[start:end] += seg * float(os.environ.get("WHALE_VC_GAIN", "0.0"))
                    vc_used += 1

        seg, pre = render_cv(cv, syl, dur, midi, cfg, prev_midi=prev_midi)
        if seg is None:
            print(f"[warn] 找不到音节 {syl!r}，跳过", file=sys.stderr)
            t += note["dur"]
            prev_syl = syl
            continue

        # 辅音要能听得见，前一个音必须让路。两种模式让路的长度不同：
        # - 唱歌（human/cyber）：贴着当前字真正的起点让路，用 render_cv 返回的 lead
        #   （= 实际保留下来的辅音长，≤60ms）。不能用 oto 的 pre（150~200ms）——
        #   那会让前一个字被削掉 200ms，而新字只从 60ms 前开始，中间空出 140ms 静音。
        #   听起来是「每个字前顿一下」，实测会让 ASR 几乎认不出字
        #   （49 字平铺测试 12%；修好后 v2 谱 1/49 → 17/49）。
        # - 说话（speak）：保留 v6 已验收的行为（那版是作者耳朵过的），不动。
        _lead = max(0.0, float(pre))
        if mode in ("human", "cyber"):
            d1 = int((t - _lead) * SR)
            d0 = int((t - _lead - 0.020) * SR)
        else:
            d0 = int((t - _lead - 0.015) * SR)
            d1 = int(t * SR)
        if os.environ.get("WHALE_DUCK", "1") != "0" and d0 >= 0 and d1 > d0 + 64:
            buf[max(0, d0):d1] *= np.linspace(1.0, 0.0, d1 - max(0, d0))
            # 唱歌时前一个音的尾巴必须在这里断干净：它原本一直延伸到 t，
            # 现在只让路到 t-lead，剩下那段会和新字的声母叠在一起，实测会糊成噪声。
            if mode in ("human", "cyber") and d1 < len(buf):
                buf[d1:] = 0.0
        start = int((t - pre) * SR)
        if start < 0:
            seg = seg[-start:]
            start = 0
        end = start + len(seg)
        buf = ensure(buf, end)
        fade = min(len(seg) // 4, int(0.02 * SR))
        if fade > 32:
            seg = seg.copy()
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        buf[start:end] += seg

        placed += 1
        prev_syl = syl
        prev_midi = midi
        t += note["dur"]

    nz = np.nonzero(np.abs(buf) > 1e-4)[0]
    if len(nz):
        buf = buf[: min(len(buf), nz[-1] + int(0.35 * SR))]
    buf = highpass(buf, float(os.environ.get("WHALE_HPF", "85")))
    buf = match_tilt(buf, float(os.environ.get("WHALE_TILT", "-8.5")))
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = buf / peak * 0.89
    sf.write(out, buf, SR, subtype="PCM_16")
    return placed, t, vc_used


def main() -> None:
    args = list(sys.argv[1:])
    mode = "human"
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i + 1]
        del args[i:i + 2]
    lyrics = None
    if "--lyrics" in args:
        i = args.index("--lyrics")
        lyrics = args[i + 1]
        del args[i:i + 2]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    score = json.load(io.open(args[0], encoding="utf-8-sig"))
    if lyrics:
        syl = pinyin_rules.to_pinyin(lyrics)
        if syl is None:
            print("没装 pypinyin，--lyrics 用不了", file=sys.stderr)
            sys.exit(1)
        for note, s in zip(score, syl):
            note["lyric"] = s
        if len(syl) != len(score):
            print(f"[warn] 歌词 {len(syl)} 字，音符 {len(score)} 个，按短的来",
                  file=sys.stderr)
    n, dur, v = synth(score, args[1], mode=mode)
    print(f"[{mode}] 合成 {n}/{len(score)} 个音，过渡 {v} 处，时长 {dur:.2f}s → {args[1]}")


if __name__ == "__main__":
    main()
