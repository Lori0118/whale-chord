# 复现路线：从一段音频到一首 cover

这是 v1.0 实际跑通的那条路。每一步都写了「输入 / 输出 / 怎么验」，没有写的步骤就是不存在的步骤。

## 0. 环境

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python whale.py env              # 自检：依赖齐不齐
```

RVC 单独一个环境（它和 core 的依赖互相顶），路径登记在 `shared/envs.json` 的 `rvc` 角色。

## 1. 听：先量，再动手

```bash
python whale.py listen  song.wav --stem vocals   # 结构 / 音符 / 和弦 / 调性
python whale.py voice   song.wav                 # 颤音速率与幅度、谐噪比、动态、音域
python whale.py check   song.wav                 # 入口体检：这条录音能不能当素材
```

**验**：`listen` 给出的调性，自己唱一遍或对一下原曲的调，能对上才算对。
`check` 里 >2.5kHz 占比 <1% 就直接重录——不要拿去训模型。

## 2. 扒谱

```bash
python <rospot>/infer.py --input vocals.wav --output score.json
```

**验**：抽 4 秒、8 个音，听谱上出来的音高与原唱是否一致。长音被切成几个音的，要手动合并。

## 3. 写谱（人干的活）

谱是简单形：

```json
[{"midi": 60, "dur": 0.5, "lyric": "wo"}]
```

要人工标的：拖腔（相邻同字的第二个音）、气口、长音。这一版不会自动标。

## 4. 唱

```bash
python whale.py sing score.json out-raw.wav
```

**验**：ASR 对一遍歌词（只能当筛查），再看音高走向。

## 5. 换音色

```bash
python whale.py convert out-raw.wav out-e1.wav --model <你的模型>
```

**验**：换声前后分别量一次咬字命中率。掉得多的就是这一版的老毛病（见 `docs/LIMITS.md` 第 2 条）。

## 6. 混音

人声与伴奏对齐、人声增益、峰值控制。这一版没有混音脚本，用的是 ffmpeg 的 `amix` + `alimiter`。

## 时间成本（在一块 12G 卡上）

| 步骤 | 三分钟的歌 |
|---|---|
| 听 + 体检 | 十秒级 |
| 扒谱 | 分钟级 |
| 唱（DiffSinger） | 分钟级 |
| 换声（RVC） | 分钟级 |
| 修谱（人） | **小时级**——这是真正的瓶颈 |

## 一句话总结

这条流水线能出整首，但**修谱和修咬字是人工的**。把它当成「一个会按谱唱的执行者」，不是「一个会唱歌的人」。
