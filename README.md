# whale-chord（鲸弦）

让一条鲸鱼娘能自己听见、自己挑一副嗓子、再用这副嗓子说话和唱歌。

- **v1.0**（`v1.0/`）：做到「整首 cover 能出」的发布快照——扒谱、按谱唱、换音色、混音。卷帘编辑器不在这一版。
- 前身叫 whale-ear（一个只会「听」的项目）；这一版把听、说、唱按层合在一起，不再按旧项目名分家。

## 从这里开始

- 想直接跑：看 [`v1.0/README.md`](v1.0/README.md)
- 想知道哪些地方不能信：看 [`v1.0/docs/LIMITS.md`](v1.0/docs/LIMITS.md)
- 想知道模型从哪下：看 [`v1.0/docs/MODELS.md`](v1.0/docs/MODELS.md)
- 想知道许可：看 [`v1.0/THIRD_PARTY_NOTICES.md`](v1.0/THIRD_PARTY_NOTICES.md)

## 目录

```
v1.0/
  whale.py        统一入口：听 / 说 / 唱 / 换声 / 体检 / 音色表
  listen/         听：结构、音符、和弦、调性、唱法参数、原唱 vs 自己唱
  voice/          嗓子层：UTAU 音源引擎、拼音规则、谱样例
  engines/        唱这一层的四个引擎（ROSVOT / GAME / SoulX；DiffSinger 与 RVC 由 whale.py 直调）
  shared/         环境分工表、音色资产表
  docs/           契约、许可、模型下载、这一版的六条不足
```

本仓库**不含**任何模型权重、训练数据和真人嗓子样本。原因见 `v1.0/THIRD_PARTY_NOTICES.md`。

## 许可

本项目自己的代码 Apache-2.0；第三方代码、模型、数据的许可各自不同，见 `v1.0/THIRD_PARTY_NOTICES.md`。
