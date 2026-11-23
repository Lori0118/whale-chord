# -*- coding: utf-8 -*-
"""pinyin_rules.py —— 汉语拼音的声母/韵母规则，专门用来对齐 UTAU 音源的元音词表。

音源的 _VC 表只收 27 种元音写法，而且用的是「零声母拼法」：
    ing → ying、üe → yue、ü → yu、ün → yuen（非标准拼法）、u → wu、uo → wo
所以不能凭感觉写映射，得按下面的表来。

数据来源：汉语拼音韵母表（39 个）+ 零声母拼写规则。
pypinyin 负责「汉字 → 拼音」（含多音字），这里只负责「拼音 → 音源词表」。
"""
from __future__ import annotations

# 21 个声母（zh/ch/sh 要放在最前，避免被 z/c/s 抢先匹配）
INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]

# 韵母表：单韵母 / 复韵母 / 鼻韵母 / 撮口呼
FINALS = [
    "a", "o", "e", "ê", "er",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong",
    "i", "ia", "ie", "iao", "iou", "ian", "in", "iang", "ing", "iong",
    "u", "ua", "uo", "uai", "uei", "uan", "uen", "uang", "ueng",
    "ü", "üe", "üan", "ün",
]

# 韵母的「零声母拼法」——音源里用的就是这个写法
ZERO_INITIAL = {
    "i": "yi", "ia": "ya", "ie": "ye", "iao": "yao", "iou": "you",
    "ian": "yan", "in": "yin", "iang": "yang", "ing": "ying", "iong": "yong",
    "u": "wu", "ua": "wa", "uo": "wo", "uai": "wai", "uei": "wei",
    "uan": "wan", "uen": "wen", "uang": "wang", "ueng": "weng",
    "ü": "yu", "üe": "yue", "üan": "yuan", "ün": "yun",
    "a": "a", "o": "o", "e": "e", "ê": "ê", "er": "er",
    "ai": "ai", "ei": "ei", "ao": "ao", "ou": "ou",
    "an": "an", "en": "en", "ang": "ang", "eng": "eng", "ong": "ong",
}

# 音源的元音词表里实际没有的写法 → 退到音感最接近的那个
# （音源只收 27 种，缺的口子在这儿补；补不上的宁可退化，不要瞎配）
FALLBACK = {
    "ya": "a", "yao": "ao", "you": "ou", "yang": "ang", "yong": "ong",
    "wa": "a", "wai": "ai", "wei": "ei", "wan": "an", "wen": "en",
    "wang": "ang", "weng": "eng", "yun": "yuen",
    "o": "wo", "ê": "e", "eh": "e",
    "iou": "ou", "uei": "ei", "uen": "en", "ueng": "eng",
}

# 舌尖前/后音（zi ci si / zhi chi shi ri）的韵母是特殊的「-i」，音源里单列了
APICAL_INITIALS = {"z", "c", "s", "zh", "ch", "sh", "r"}

# 拼音的缩写写法（iou/uei/uen 在声母后写成 iu/ui/un）
ABBREV = {"iu": "iou", "ui": "uei", "un": "uen",
          "ǖ": "ü", "v": "ü"}


def split_syllable(syllable: str):
    """把拼音音节切成 (声母, 韵母)。韵母已还原缩写（iu→iou 等）。"""
    s = syllable.strip().lower().replace("v", "ü")
    for ini in INITIALS:
        if s.startswith(ini) and len(s) > len(ini):
            raw = s[len(ini):]
            # j/q/x 后面写的 u 其实都是 ü（ju=举、xue=靴、quan=全）。
            # 这一步必须在「缩写还原」之前：jun 的 un 是 ün，不是 uen。
            if ini in ('j', 'q', 'x') and raw.startswith('u'):
                fin = 'ü' + raw[1:]
            else:
                fin = ABBREV.get(raw, raw)
            if fin == "i" and ini in APICAL_INITIALS:
                return ini, "-i"
            return ini, fin
    # 零声母：整段就是韵母的零声母拼法，先还原回原韵母
    for fin, zero in ZERO_INITIAL.items():
        if s == zero:
            return "", fin
    return "", ABBREV.get(s, s)


def bank_vowel(fin: str) -> str:
    """把韵母换成音源词表里的写法。"""
    if fin == "-i":
        return "-i"
    zero = ZERO_INITIAL.get(fin, fin)
    if zero in FALLBACK:
        return FALLBACK[zero]
    return zero


def vc_key(prev: str, nxt: str):
    """两个音节之间的连读键：左边是前一个的元音，右边是后一个的声母（或元音）。"""
    _pi, pf = split_syllable(prev)
    ni, nf = split_syllable(nxt)
    left = bank_vowel(pf)
    right = ni if ni else bank_vowel(nf)
    return f"{left} {right}"


def to_pinyin(text: str):
    """汉字歌词 → 无声调拼音。没装 pypinyin 时返回 None。"""
    try:
        from pypinyin import lazy_pinyin, Style
    except ImportError:
        return None
    out = []
    for word in lazy_pinyin(text, style=Style.NORMAL, errors="ignore"):
        if word.strip():
            out.append(word.strip())
    return out


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sing

    _cv, vc = sing.load_bank()
    firsts = {a.rsplit("_", 1)[0].split()[0] for a in vc
              if sing.pitch_of(a) and len(a.rsplit("_", 1)[0].split()) == 2}

    print(f"音源元音写法 {len(firsts)} 种:", " ".join(sorted(firsts)))
    hit, miss = [], []
    for fin in FINALS:
        tok = bank_vowel(fin)
        (hit if tok in firsts else miss).append((fin, tok))
    print(f"\n命中的韵母 {len(hit)}/{len(FINALS)}")
    if miss:
        print("落空的:", ", ".join(f"{f}→{t}" for f, t in miss))

    print("\n示例句连读键：")
    syl = to_pinyin("让鲸鱼娘学会唱歌") or ["rang", "jing", "yu", "niang",
                                         "xue", "hui", "chang", "ge"]
    print("  拼音:", syl)
    for a, b in zip(syl, syl[1:]):
        k = vc_key(a, b)
        ok = any(al.rsplit("_", 1)[0] == k for al in vc)
        print(f"  {a:6}→{b:6} 键 {k:8} {'OK' if ok else '缺'}")
