# -*- coding: utf-8 -*-
"""ROSVOT 后处理：按从 GAME 行为归纳出的规则，把碎块合回去。

规则（只合两种，都偏保守）：
  R1 同音连续：相邻两音音高差 < 0.6 半音且间隙 < 25ms → 合并（取长者音高）
  R2 微短音：音符时长 < 150ms → 并入相邻音（要求音程 ≤ 2 半音，优先并入更长的那个）
不合并的情形：音程 ≥ 3 半音的跳进、时长 ≥ 150ms 的音——那是真旋律。
"""
from pathlib import Path

import numpy as np
import pretty_midi


def load_notes(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    ns = [(n.start, n.end - n.start, float(n.pitch)) for inst in pm.instruments for n in inst.notes]
    return sorted(ns)


def merge_same(notes, tol_cents=0.6, gap=0.025):
    out = []
    for o, d, p in notes:
        if out:
            po, pd, pp = out[-1]
            if abs(p - pp) < tol_cents and o - (po + pd) < gap:
                # 合并：时长拉伸到后音结束，音高取更长的那个
                new_d = (o + d) - po
                new_p = pp if pd >= d else p
                out[-1] = (po, new_d, new_p)
                continue
        out.append((o, d, p))
    return out


def merge_short(notes, max_dur=0.15, max_step=2.0):
    if len(notes) < 2:
        return notes
    notes = list(notes)
    changed = True
    while changed:
        changed = False
        for i, (o, d, p) in enumerate(notes):
            if d >= max_dur:
                continue
            # 候选：前一个 / 后一个
            cands = []
            if i > 0:
                po, pd, pp = notes[i - 1]
                if abs(p - pp) <= max_step:
                    cands.append((i - 1, po + pd - o, pd))
            if i + 1 < len(notes):
                no, nd, np_ = notes[i + 1]
                if abs(p - np_) <= max_step:
                    cands.append((i + 1, 999, nd))  # 后音恒可并入
            if not cands:
                continue
            # 优先并入更长的那一侧
            cands.sort(key=lambda c: -c[2])
            j = cands[0][0]
            tgt = notes[j]
            if j < i:  # 并入前音
                merged = (tgt[0], (o + d) - tgt[0], tgt[2])
                del notes[i]
                notes[j] = merged
            else:  # 并入后音
                merged = (o, (tgt[0] + tgt[1]) - o, tgt[2])
                del notes[i]
                notes[j - 1] = merged
            changed = True
            break
    return notes


def postprocess(notes, use_r1=True, use_r2=True):
    if use_r1:
        notes = merge_same(notes)
    if use_r2:
        notes = merge_short(notes)
        notes = merge_same(notes)  # R2 合并后可能又出现同音相邻
    return notes


def stats(notes, label):
    if not notes:
        print(f'  {label}: 无')
        return None
    o = np.array([n[0] for n in notes])
    d = np.array([n[1] for n in notes])
    p = np.array([n[2] for n in notes])
    steps = np.abs(np.diff(p))
    return {
        'label': label, 'n': len(notes),
        'median': float(np.median(d)),
        'same': float((steps < 0.5).mean()) if len(steps) else 0,
        'semi': float(((steps >= 0.5) & (steps < 1.5)).mean()) if len(steps) else 0,
        'big': float((steps >= 5).mean()) if len(steps) else 0,
    }


SONGS = {
    '雨爱': Path(r'.\out\rosvot\midi\output.mid'),
    '光年之外': Path(r'.\out\rosvot-multi\光年之外\midi\output.mid'),
    '晴天': Path(r'.\out\rosvot-multi\晴天\midi\output.mid'),
    '青花瓷': Path(r'.\out\rosvot-multi\青花瓷\midi\output.mid'),
}

# GAME 的参照值（前面量过的）
GAME_REF = {
    '雨爱': {'n': 58, 'median': 0.360, 'same': 0.287, 'semi': 0.245, 'big': 0.160},
    '光年之外': {'n': 95, 'median': 0.330, 'same': 0.287, 'semi': 0.245, 'big': 0.160},
    '晴天': {'n': 109, 'median': 0.430, 'same': 0.185, 'semi': 0.296, 'big': 0.148},
    '青花瓷': {'n': 180, 'median': 0.280, 'same': 0.229, 'semi': 0.112, 'big': 0.084},
}

print(f'{"歌":8s}{"版本":14s}{"音符":>6s}{"中位ms":>9s}{"Δ=0":>8s}{"Δ=1":>8s}{"大跳":>8s}')
for song, path in SONGS.items():
    raw = load_notes(path)
    r1 = postprocess(raw, use_r1=True, use_r2=False)
    both = postprocess(raw, use_r1=True, use_r2=True)
    for label, ns in (('ROSVOT 原始', raw), ('+R1 同音', r1), ('+R1+R2 微短', both)):
        s = stats(ns, label)
        if s:
            print(f'{song:8s}{s["label"]:14s}{s["n"]:6d}{s["median"]*1000:9.0f}'
                  f'{s["same"]:8.1%}{s["semi"]:8.1%}{s["big"]:8.1%}')
    g = GAME_REF[song]
    print(f'{"":8s}{"GAME 参照":12s}{g["n"]:6d}{g["median"]*1000:9.0f}'
          f'{g["same"]:8.1%}{g["semi"]:8.1%}{g["big"]:8.1%}')
    print()
