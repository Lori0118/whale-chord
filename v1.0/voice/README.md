# voice：嗓子层

这一层管两件事：**定一副嗓子**，然后**用这副嗓子说和唱**。

```
voice/
  sing.py            UTAU 音源引擎：谱 + 拼音 → wav（不依赖模型，也不需要 GPU）
  pinyin_rules.py    中文 → 声母/韵母切分
  scores/            谱样例（简单形，见 docs/CONTRACTS.md）
  tools/
    rec_check.py     入口体检：这条录音能不能当素材
    voice_palette.py 音色表：从候选里挑一副嗓子
```

## 说和唱是同一副嗓子，两种韵律

- 说：`python whale.py speak score.json out.wav`
- 唱：`python whale.py sing score.json out.wav`

`sing.py` 做的是：读音源的 `oto.ini` → 按最近音高切片 → 变调拉伸 → 给长音加颤音 → 用过渡样本补连读 → 按 `preutterance` 对齐拼接。
`--mode human` 是带微走音和时值抖动的版本，`--mode cyber` 是电子味版本（颤音更狠、不抖动、不带连读）。

## 音源从哪来

**本仓库不带音源。** UTAU 音源库是第三方资产，各自有许可（有的允许改、有的只允许个人使用）。
你自己准备的音源放进 `voice/voicebank/<名字>/`，再到 `shared/voices.json` 登记。

> 音色必须存成文件：同一段描述、同一个 seed 重跑出来的不是同一条嗓子。

## 这一层的硬限制

1. **拼接合成会有「蹦字」感**：长音之间没有真正的过渡，唱慢歌时明显。
2. **说话母本不等于唱歌母本**：能说话的腔调，唱起来不一定好听——两个母本要分开准备。
3. **换声会掉字**：RVC 换完音色，咬字会丢一部分（多的十几个百分点）。
