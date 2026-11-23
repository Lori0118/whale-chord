# -*- coding: utf-8 -*-
"""多曲横向：GAME vs ROSVOT，重点看「切分粒度」而不只是音高偏差。

上次只用 pyin 偏差评，结论是 ROSVOT 略好，但作者耳朵判 GAME 更好。
怀疑口径有偏：贴 pyin 往往意味着跟着颤音/滑音多切。所以这次加三个「音乐性」指标：
  1. 音符密度（每秒几个音）
  2. 相邻音程 |Δ| 的分布——大量 0/1 半音说明长音被切碎、颤音被切碎
  3. 同音内相对时长
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, r'.\tools')
from game_free import run_free  # noqa: E402

GAME_OUT = Path(r'.\out\multi')
GAME_OUT.mkdir(parents=True, exist_ok=True)


def analyze(notes, name, dur, label):
    """notes = [(onset, dur, pitch)]"""
    if not notes:
        return None
    o = np.array([n[0] for n in notes])
    d = np.array([n[1] for n in notes])
    p = np.array([n[2] for n in notes])
    keep = d > 0
    o, d, p = o[keep], d[keep], p[keep]
    order = np.argsort(o)
    o, d, p = o[order], d[order], p[order]

    density = len(o) / max(dur, 1e-6)
    med_dur = float(np.median(d))
    steps = np.abs(np.diff(p))
    same = float((steps < 0.5).mean()) if len(steps) else 0.0
    semi = float(((steps >= 0.5) & (steps < 1.5)).mean()) if len(steps) else 0.0
    big = float((steps >= 5).mean()) if len(steps) else 0.0
    short_ratio = float((d < 0.12).mean())

    print(f'  {label:8s} 音符{len(o):4d}  密度{density:5.2f}/s  中位时长{med_dur*1000:4.0f}ms  '
          f'短音<120ms{short_ratio:5.1%}  Δ=0 {same:5.1%}  Δ=1 {semi:5.1%}  大跳≥5半音 {big:5.1%}')
    return {'label': label, 'n': int(len(o)), 'density': density, 'median_dur': med_dur,
            'short_ratio': short_ratio, 'same': same, 'semi': semi, 'big': big,
            'onsets': o.tolist(), 'durs': d.tolist(), 'pitch': p.tolist()}


SONGS = [
    ('光年之外', r'.\svs-lab\光年之外.wav'),
    ('晴天', r'.\svs-lab\晴天.wav'),
    ('青花瓷', r'.\svs-lab\青花瓷.wav'),
]

results = {}
for title, path in SONGS:
    print(f'\n===== {title} =====')
    y, sr = librosa.load(path, sr=None, mono=True)
    dur = len(y) / sr
    print(f'  时长 {dur:.1f}s')

    # GAME
    r = run_free(path)
    gam_notes = [(float(a), float(b), float(c))
                 for a, b, c, v in zip(r['onsets'], r['durations'], r['scores'], r['voiced']) if v]
    gs = analyze(gam_notes, title, r['dur'], 'GAME')
    if gs:
        (GAME_OUT / f'{title}.game.json').write_text(json.dumps(gs, ensure_ascii=False), encoding='utf-8')

    results[title] = {'dur': dur, 'game': gs}

print('\n\n=== 汇总 ===')
for title, v in results.items():
    g = v['game']
    print(f"{title:8s} GAME {g['n']:4d} 音  密度 {g['density']:.2f}/s  中位 {g['median_dur']*1000:.0f}ms")
