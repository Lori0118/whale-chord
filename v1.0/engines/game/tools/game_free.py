# -*- coding: utf-8 -*-
"""GAME 1.0.3 自由模式（无声部约束）最小推理 —— 只依赖 onnxruntime + numpy + librosa。

移植自 Vocal2Midi (Apache-2.0) inference/game/onnx_runtime.py 的 forward 顺序：
encoder -> [dur2bd(若无已知边界则跳过)] -> segmenter x nsteps -> bd2dur -> estimator

权重 GAME-1.0.3-medium-onnx 为 CC BY-NC-SA 4.0（非商用），仅作实验，勿进商用路径。
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import librosa

MODEL_DIR = Path(r'.\models\GAME-1.0.3-medium-onnx')

FILENAMES = {
    "encoder": "encoder.onnx",
    "segmenter": "segmenter.onnx",
    "estimator": "estimator.onnx",
    "dur2bd": "dur2bd.onnx",
    "bd2dur": "bd2dur.onnx",
}


def load_sessions():
    cfg = json.loads((MODEL_DIR / 'config.json').read_text(encoding='utf-8'))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False
    sess = {}
    for key, fn in FILENAMES.items():
        sess[key] = ort.InferenceSession(str(MODEL_DIR / fn), sess_options=so,
                                        providers=['CPUExecutionProvider'])
    return cfg, sess


def t0_nstep_to_ts(t0: float, nsteps: int):
    step = (1 - t0) / nsteps
    return [t0 + i * step for i in range(nsteps)]


def run_free(path, seg_threshold=0.2, seg_radius=0.02, est_threshold=0.2,
             nsteps=8, t0=0.0, language=0):
    cfg, sess = load_sessions()
    sr = int(cfg['samplerate'])
    dt = float(cfg['timestep'])
    loop = bool(cfg.get('loop', True))

    y, fs = librosa.load(path, sr=sr, mono=True)
    y = np.ascontiguousarray(y, dtype=np.float32)
    dur = np.array([len(y) / sr], dtype=np.float32)
    wav = y[np.newaxis, :]
    ts = t0_nstep_to_ts(t0, nsteps) if loop else []

    enc = sess['encoder'].run(None, {'waveform': wav, 'duration': dur})
    x_seg, x_est, mask_t = enc[0], enc[1], enc[2]

    batch = wav.shape[0]
    boundaries = np.zeros_like(mask_t, dtype=bool)
    lang = np.full((batch,), int(language), dtype=np.int64)
    radius = int(round(seg_radius / dt))

    seq = ts if (loop and ts) else [0.0]
    for t in seq:
        boundaries = sess['segmenter'].run(None, {
            'x_seg': x_seg,
            'language': lang,
            'known_boundaries': np.zeros_like(mask_t, dtype=bool),
            'prev_boundaries': boundaries,
            't': np.full((batch,), float(t), dtype=np.float32),
            'maskT': mask_t,
            'threshold': np.array(seg_threshold, dtype=np.float32),
            'radius': np.array(radius, dtype=np.int64),
        })[0]

    durs_out, mask_n = sess['bd2dur'].run(None, {'boundaries': boundaries, 'maskT': mask_t})
    presence, scores = sess['estimator'].run(None, {
        'x_est': x_est,
        'boundaries': boundaries,
        'maskT': mask_t,
        'maskN': mask_n,
        'threshold': np.array(est_threshold, dtype=np.float32),
    })

    valid = mask_n[0].astype(bool)
    durs = durs_out[0][valid].astype(np.float64)
    onsets = np.concatenate([[0.0], np.cumsum(durs)[:-1]]) if len(durs) else np.zeros(0)
    return {
        'sr': sr,
        'dur': float(len(y) / sr),
        'onsets': onsets,
        'durations': durs,
        'presence': presence[0][valid],
        'scores': scores[0][valid].astype(np.float64),
        # 别名，方便下游统一成 note 三元组
        'voiced': presence[0][valid].astype(bool),
        'pitch': scores[0][valid].astype(np.float64),
    }


def main():
    if len(sys.argv) < 2:
        print('usage: game_free.py <wav> [out.json]')
        sys.exit(1)
    wav = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None

    r = run_free(wav)
    d = r['durations']
    p = r['presence'].astype(bool)
    s = r['scores']

    onset = np.concatenate([[0.0], np.cumsum(d)[:-1]])
    n_note = len(d)
    n_uv = int((~p).sum())

    print(f'[GAME] {Path(wav).name}  dur={r["dur"]:.3f}s  notes={n_note}  unvoiced={n_uv} ({n_uv/max(n_note,1):.1%})')
    if n_note:
        dv = d[p] if p.any() else d
        print(f'[GAME] note-dur  min={dv.min()*1000:.0f}ms  median={np.median(dv)*1000:.0f}ms  max={dv.max()*1000:.0f}ms')
    if p.any():
        sv = s[p]
        print(f'[GAME] pitch     min={sv.min():.2f}  median={np.median(sv):.2f}  max={sv.max():.2f}  span={sv.max()-sv.min():.2f} semi')
        print(f'[GAME] first 25 notes:')
        for i in range(min(25, n_note)):
            tag = f'{s[i]:6.2f}' if p[i] else '  rest'
            print(f'        {i:3d}  t={onset[i]:7.3f}  dur={d[i]*1000:6.0f}ms  pitch={tag}')

    if out:
        Path(out).write_text(json.dumps({
            'source': str(wav),
            'duration': r['dur'],
            'onsets': onset.tolist(),
            'durations': d.tolist(),
            'pitch': s.tolist(),
            'voiced': p.tolist(),
        }, ensure_ascii=False), encoding='utf-8')
        print(f'[GAME] -> {out}')


if __name__ == '__main__':
    main()
