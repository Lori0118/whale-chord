# GAME（歌声转换）+ Vocal2Midi（扒谱）

**代码 MIT；权重 CC BY-NC-SA，不随仓库分发。**

## 这一条路是干什么的

```
原唱 → Vocal2Midi 扒谱（可选）→ GAME 转换 → 换成目标音色，保留原唱的唱法
```

和 DiffSinger 那条路的区别：

| | DiffSinger 路线 | GAME 路线 |
|---|---|---|
| 韵律从哪来 | 谱（人写的） | 原唱（照搬） |
| 音色从哪来 | RVC 再换一次 | 转换模型直接给 |
| 擅长 | 完全新写一首 | 「这首歌唱得像那个人」 |
| 缺 | 谱上没有的不会补 | 必须先有一版原唱 |

我们这一版的主线走 DiffSinger；GAME 是并存的替代路线，代码留着、权重自己下。

## 目录

```
GAME/            上游代码（MIT）
Vocal2Midi/      上游代码（扒谱，MIT）
tools/           我们自己的接线脚本
  game_free.py     最简调用：输入/输出/权重路径
  multi_game.py    批量跑
  midi_to_score.py  MIDI → 本项目的谱格式
  postprocess.py    响度与拼接的后处理
```

## 许可提醒

GAME 的官方权重是 **CC BY-NC-SA**：非商用、相同方式共享。
所以这一条路默认**不进主流程**，只作研究对照；要商用必须换权重（自己训，且训练数据许可要干净）。

## 装法

1. 按 `GAME/README.md` 装依赖（它有自己的 requirements）。
2. 权重放 `models/game/`，用 `tools/game_free.py` 指定路径。
3. 先跑 4 秒、一句话验证，再上整首。
