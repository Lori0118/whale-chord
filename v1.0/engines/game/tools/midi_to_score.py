# -*- coding: utf-8 -*-
"""把 thr=0.98 的 MIDI 排成 DiffSinger 的 score.json。

字边界用 GAME 的 31 个音符累积（GAME 在这一段是一字一音，正好 31 字）；
每个 thr0.98 音符按中心点归到某个字下，一个字多个音就用空格连起来。
"""
import json
from pathlib import Path

import numpy as np
import pretty_midi

OUT = Path(r'.\out')
LYRICS = '听雨的声音一滴滴清晰你的呼吸像雨滴渗入我的爱里真希望雨能下不停'
NAME = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def ns(m):
    m = int(round(m))
    return f'{NAME[m % 12]}{m // 12 - 1}'


def load_midi(p):
    pm = pretty_midi.PrettyMIDI(str(p))
    a = sorted([(n.start, n.end - n.start, float(n.pitch)) for inst in pm.instruments for n in inst.notes])
    return [x for x in a if x[0] < 12.0]


game_pm = json.load(open(OUT / 'rainie-game-free.json', encoding='utf-8'))
game = [(o, d, p) for o, d, p, v in zip(game_pm['onsets'], game_pm['durations'],
                                         game_pm['pitch'], game_pm['voiced']) if v and o < 12.0]

notes = load_midi(OUT / 'rosvot-thr-0.98' / 'midi' / 'output.mid')

print(f'字數 {len(LYRICS)}   GAME 音符 {len(game)}   thr0.98 音符 {len(notes)}')

# 字边界 = GAME 音符的起止
word_bounds = [(o, o + d) for o, d, _ in game]

groups = [[] for _ in word_bounds]
for o, d, p in notes:
    c = o + d / 2
    idx = None
    for i, (a, b) in enumerate(word_bounds):
        if a - 0.03 <= c < b + 0.03:
            idx = i
            break
    if idx is None:
        # 落到间隙里，归最近的字
        idx = int(np.argmin([min(abs(c - a), abs(c - b)) for a, b in word_bounds]))
    groups[idx].append((o, d, p))

empty = [i for i, g in enumerate(groups) if not g]
print(f'没有分到音符的字: {empty}')

# 补齐空字：用 GAME 该字的音高和时长
for i in empty:
    groups[i].append((game[i][0], game[i][1], game[i][2]))

notes_str = []
durs_str = []
for i, g in enumerate(groups):
    g = sorted(g)
    # 音符之间有空隙就补 rest（DiffSinger 用 AP 表示换气，这里用 rest）
    notes_str.append(' '.join(ns(p) for _, _, p in g))
    durs_str.append(' '.join(f'{d:.3f}' for _, d, _ in g))

score = {
    'text': LYRICS,
    'notes': ' | '.join(notes_str),
    'notes_duration': ' | '.join(durs_str),
    'input_type': 'word',
}

out = OUT / 'score-rainie12-rosvot098.json'
out.write_text(json.dumps(score, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'-> {out}')
print('前 6 个字:', ' | '.join(f'{c}{n}' for c, n in zip(LYRICS, notes_str))[:200])
total = sum(float(x) for s in durs_str for x in s.split())
print(f'总时长 {total:.2f}s')
