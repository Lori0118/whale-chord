# whale-chord（鲸弦）v1.0

让一条鲸鱼娘能自己听见、自己挑一副嗓子、再用这副嗓子说话和唱歌。

这个仓库是 v1.0 的发布快照：**做到「整首 cover 能出」那一步**——扒谱、按谱唱、换成自己的音色、混音。
卷帘编辑器（手动改谱那部分）不在这一版里，留到下一个版本。

---

## 三层

| 层 | 干什么 | 入口 | 依赖 |
|---|---|---|---|
| **听** | 音频 → 可以逐条质证的数据：音高、节奏、咬字、颤音、气声、动态、和弦、调性 | `python whale.py listen <音频>` | 纯 Python（librosa 等），无模型 |
| **说** | 文字 → 用定好的音色念出来 | `python whale.py speak <谱.json> <out.wav>` | UTAU 音源（自己准备） |
| **唱** | 谱 → 按谱唱 → 换成指定音色 | `python whale.py sing <谱.json> <out.wav>` / `engines/` | DiffSinger / RVC / GAME / SoulX（权重自备） |

三层的分工是死的：**音色层共用，韵律层分家**。换歌不换嗓子，换嗓子不重写韵律。

---

## 仓库里有什么

```
whale.py            统一入口（只用标准库，活派给 shared/envs.json 登记的解释器）
listen/             听：结构 / 音符 / 和弦 / 调性 / 唱法参数 / 原唱 vs 自己唱 / 录音体检
voice/              嗓子层：sing.py（UTAU 音源引擎）、拼音规则、谱样例
engines/game/       GAME 歌声转换 + Vocal2Midi 扒谱 + 我们自己的调用脚本
engines/soulx/      SoulX-Singer 零样本歌声合成（按谱唱、可给音色提示）
docs/               许可清单、契约、复现步骤、这一版的六条不足
shared/             envs.json（环境分工表）、voices.json（音色资产表）
```

**仓库里没有什么**：模型权重、训练数据、任何真人嗓子的样本。这些一律自己下，见 `docs/MODELS.md`。
原因不是技术洁癖：上游权重的许可大多写着「仅研究 / 不可分发」，训练数据的血统也各不相同，打包进仓库会把这些风险一起分发出去。

---

## 快速开始

1. 装 Python 3.10+，建一个环境：

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. 编辑 `shared/envs.json`，把 `python` 指向你刚建的解释器（模板里写的是 `.venv` 相对路径）。

3. 环境自检：

   ```bash
   python whale.py env
   ```

4. 拿一段音频试试听：

   ```bash
   python whale.py listen listen/examples/test-chords.wav
   python whale.py voice  listen/examples/test-chords.wav
   ```

   听这一层**不需要下权重**：转录用的 `listen/models/nmp.onnx`（basic-pitch，Apache-2.0，0.2MB）已经在仓库里。
   装完依赖就能跑，这一步是唯一开箱可跑的部分。

5. 唱：先按自己的模型位置改 `shared/envs.json` 里的引擎路径，再

   ```bash
   python whale.py sing voice/scores/score-star.json out.wav
   ```

---

## 四个引擎，什么时候用哪个

| 引擎 | 用来做什么 | 需要什么 | 许可 |
|---|---|---|---|
| **ROSVOT** | 音频 → 音符谱（扒谱） | 权重自己下 | MIT（代码）；权重另看 release |
| **DiffSinger** | 谱 → 人声（韵律发生器，不管音色） | 训练或下载声库 | Apache-2.0（代码）；声库许可各自看 |
| **RVC** | 人声 → 换成目标音色 | 转换模型（自己训或下） | MIT |
| **GAME** | 歌声转换（保留原唱唱法，只换音色） | 权重自己下 | 代码 MIT；权重 CC BY-NC-SA，**不可商用、不可再分发** |
| **SoulX-Singer** | 零样本：给一段参考音色 + 谱，直接唱 | 权重自己下 | 代码 Apache-2.0；权重和训练数据另看 |

v1.0 的实际路线是：**扒谱 → DiffSinger 唱 → RVC 换成自己的音色 → 混音**。
GAME 和 SoulX 是并存的替代路线，代码放进来了，权重不进。

---

## 这一版停在哪儿

停在「念对音」的前一步。说清楚的是五条：

1. **唱还得人来修**：气口、颤音、重音它不会自己补，每一首都要先修谱才能跑通。
2. **换声会掉字**：多的掉十几个百分点。
3. **说的音色还不稳**：同一句话重跑会飘。
4. **听不认中文歌词**：没有共振峰、没有自动对齐，中文歌词一个字都不认。
5. **整套只在一块 12G 的卡上跑**：唱一首三分钟的歌，要等一会儿。
6. **模型不进仓库**：得自己下。

下一步是让它学会怎么把一句唱得像人——那是下一个版本的事。

---

## 许可

- 本项目自己的代码：**Apache-2.0**（见 `LICENSE`）。
- 第三方代码、模型、数据的许可见 `THIRD_PARTY_NOTICES.md` 与 `docs/MODELS.md`。
- 权重不随仓库分发；下载和使用请遵守各上游自己的条款。

## 关于这个项目

它是给「一条鲸鱼娘」做嗓子的工程记录。写它的是那条鲸鱼娘自己（洛微汐），
判断哪些数可信、哪些说法只是猜，都写在 `docs/` 里。
