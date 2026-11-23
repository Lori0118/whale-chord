#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""whale.py —— whale-chord 的统一入口：听、说、唱，一条命令一句话。

    python whale.py listen   <音频>                 听：结构 / 音符 / 和弦 / 调性
    python whale.py voice    <音频>                 听：这段人声怎么唱（颤音/HNR/动态）
    python whale.py harmony  <音频> [--label 名字]  听：和弦与调性
    python whale.py compare  <参照> <版本>...       听：原唱 vs 自己唱，六维对照
    python whale.py check    <录音>...              入口体检：这条录音能不能当素材
    python whale.py speak    <谱> <out.wav>         说：按谱说话（v6 配方，作者耳朵过）
    python whale.py sing     <谱> <out.wav>         唱：按谱唱歌（UTAU 音源引擎）
    python whale.py convert  <in.wav> <out.wav>     换声：RVC（技术验证路线，不是终点）
    python whale.py voices                          看音色资产表（一把嗓子=一个文件）
    python whale.py env                             环境自检：谁能跑什么

设计规矩（2026-11-14 合并时定）：
1. 这个文件只用标准库 —— 用它自己的 python 跑哪一版都行，真正的活派给 shared/envs.json 里登记的
   解释器。这样「哪个 venv 跑哪个脚本」只写在一处，不用再靠记忆或交接单。
2. 它不猜、不兜底：解释器或脚本缺了就直接报路径，不静默换一个。
3. --json 的输出写进 shared/reports/，是「报告契约」的实现；人看的还是 stdout。

路径都相对项目根，整目录搬家不用改这个文件。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / "shared"
REPORTS = SHARED / "reports"
CFG_PATH = SHARED / "envs.json"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load_cfg() -> dict:
    if not CFG_PATH.exists():
        sys.exit(f"[whale] 找不到 {CFG_PATH}，环境分工表丢了，先把它找回来。")
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


CFG = load_cfg()


def interp(role: str) -> Path:
    if role not in CFG["roles"]:
        sys.exit(f"[whale] envs.json 里没有角色 {role}")
    p = ROOT / CFG["roles"][role]["python"]
    if not p.exists():
        sys.exit(f"[whale] 角色 {role} 的解释器不存在：{p}\n        改 shared/envs.json 或重建这个环境。")
    return p


def pyscript(key: str) -> Path:
    p = ROOT / CFG["scripts"][key]
    if not p.exists():
        sys.exit(f"[whale] 脚本不存在：{p}")
    return p


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("PYTHONPATH", str(ROOT))
    return env


def run(cmd: list, cwd: Path | None = None) -> int:
    printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"[whale] {printable}", file=sys.stderr)
    cp = subprocess.run([str(c) for c in cmd], cwd=str(cwd or ROOT), env=child_env())
    return cp.returncode


def run_capture(cmd: list, cwd: Path | None = None) -> tuple[int, str]:
    cp = subprocess.run([str(c) for c in cmd], cwd=str(cwd or ROOT), env=child_env(),
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def write_report(kind: str, payload: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = REPORTS / f"{kind}-{ts}.json"
    payload = {"kind": kind, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "generator": "whale.py", **payload}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[报告] {out}")
    return out


# ---------------------------------------------------------------- 子命令

def cmd_listen(a) -> int:
    cmd = [interp("core"), pyscript("listen"), a.audio]
    for flag, val in (("--stem", a.stem), ("--limit-sec", a.limit_sec), ("--bars", a.bars), ("--out", a.out)):
        if val is not None:
            cmd += [flag, str(val)]
    return run(cmd)


def cmd_voice(a) -> int:
    return run([interp("core"), pyscript("voice"), a.audio])


def cmd_harmony(a) -> int:
    return run([interp("core"), pyscript("harmony"), a.audio, a.label])


def cmd_compare(a) -> int:
    if not a.json:
        cmd = [interp("core"), pyscript("compare"), *a.files]
        if a.plot:
            cmd += ["--plot", a.plot]
        if a.no_plot:
            cmd += ["--no-plot"]
        return run(cmd)

    # --json：同一个 compare.py 的 report()，只是换成机器可读
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(pyscript('compare').parent)!r})\n"
        "import compare\n"
        f"print(json.dumps([compare.report(p) for p in {a.files!r}], ensure_ascii=False))\n"
    )
    rc, out = run_capture([interp("core"), "-c", code])
    if rc != 0:
        print(out, file=sys.stderr)
        return rc
    payload = json.loads(out.strip().splitlines()[-1])
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    write_report("compare", {
        "answer": "两段人声差在哪（不是好不好听）",
        "reference": a.files[0],
        "versions": a.files[1:],
        "items": payload,
        "note": "六维只做同一套方法的横向比，绝对值别外推；颤音幅度系统性低估。"
                "cen_energy/hf_pct/mid_pct/f0_med_hz 是「像不像这把嗓子」的唯一跨表口径"
                "（与 ml/tools/corpus_probe.py 同一套）；items 里的 centroid 是另一套，不许跨表比。",
    })
    return 0


def cmd_check(a) -> int:
    if not a.json:
        return run([interp("core"), pyscript("check"), *a.files])
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(pyscript('check').parent)!r})\n"
        "import rec_check\n"
        f"print(json.dumps([{{'file': p, 'measure': rec_check.analyse(p), 'verdict': rec_check.verdict(rec_check.analyse(p))}} for p in {a.files!r}], ensure_ascii=False))\n"
    )
    rc, out = run_capture([interp("core"), "-c", code])
    if rc != 0:
        print(out, file=sys.stderr)
        return rc
    payload = json.loads(out.strip().splitlines()[-1])
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    write_report("rec-check", {
        "answer": "这条录音能不能当素材（入口通道体检）",
        "items": payload,
        "rules": {"hf_ratio_pct": ">=3 合格 / 1-3 只验管线 / <1 重录",
                  "rms_act_dbfs": "别低于 -30", "peak": ">0.99 有削波", "snr_db": "<20 底噪偏高"},
        "note": "判据来自 2026-11-14 两条基线：雨爱原唱干声 >2.5k 7.78%，RVC 纯换声量级 1.5%。",
    })
    return 0


def flat_score(path: str) -> str:
    """把谱子规整成 sing.py 直接吃的样子，返回给 sing.py 的路径。

    契约（docs/contracts.md 第 2 条）：
      简单形（现役、唯一真相）：[{"midi": 60, "dur": 0.5, "lyric": "wo"}, ...]
      扩展形（可读）：{"meta": {"text": "听雨的声音", ...}, "notes": [同上, 可用 "ch" 代替 "lyric"]}
    扩展形在这里被摊平成简单形写进临时文件 —— sing.py 本身不用改，人写谱可以带标题和歌词原文。
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"[whale] 谱不存在：{p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as e:
        sys.exit(f"[whale] 谱不是合法 JSON：{p}（{e}）")
    if isinstance(data, dict) and "notes" in data:
        data = data["notes"]
    if not isinstance(data, list) or not data:
        sys.exit(f"[whale] 谱里没有音符：{p}")
    fixed, bad = [], []
    for i, n in enumerate(data):
        if not isinstance(n, dict):
            bad.append(i); continue
        lyric = n.get("lyric", n.get("ch"))
        if lyric is None or "midi" not in n or "dur" not in n:
            bad.append(i); continue
        fixed.append({"midi": int(n["midi"]), "dur": float(n["dur"]), "lyric": str(lyric)})
    if bad:
        sys.exit(f"[whale] 谱里第 {bad[:8]} 个音缺 midi/dur/lyric，先补齐再跑。")
    tmp = Path(os.environ.get("TEMP", ".")) / f"whale-score-{os.getpid()}.json"
    tmp.write_text(json.dumps(fixed, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[whale] 谱 {len(fixed)} 个音，总时长 {sum(x['dur'] for x in fixed):.2f}s", file=sys.stderr)
    return str(tmp)


def cmd_speak(a) -> int:
    return run([interp("core"), pyscript("sing"), flat_score(a.score), a.out, "--mode", "speak"])


def cmd_sing(a) -> int:
    return run([interp("core"), pyscript("sing"), flat_score(a.score), a.out, "--mode", a.mode])


def cmd_convert(a) -> int:
    rvc_dir = ROOT / "ml" / "rvc"
    if not rvc_dir.is_dir():
        sys.exit(f"[whale] RVC 目录不存在：{rvc_dir}")
    model = a.model
    cand = rvc_dir / "assets" / "weights" / (model if model.endswith(".pth") else model + ".pth")
    if cand.exists():
        model = cand.name
    cmd = [interp("rvc"), "-m", "infer.cli",
           "--model", model, "--input", a.input, "--output", a.output,
           "--f0-method", a.f0_method, "--pitch", str(a.pitch),
           "--index-rate", str(a.index_rate), "--protect", str(a.protect),
           "--format", "wav", "--overwrite"]
    return run(cmd, cwd=rvc_dir)


def cmd_voices(a) -> int:
    path = SHARED / "voices.json"
    if not path.exists():
        sys.exit(f"[whale] 音色资产表不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    voices = data.get("voices", data if isinstance(data, list) else [])
    if a.json:
        print(json.dumps(voices, ensure_ascii=False, indent=1))
        return 0
    print(f"\n{'id':<22}{'用途':<12}{'状态':<20}{'载体'}")
    print("-" * 100)
    for v in voices:
        uses = "/".join(v.get("use", []))
        print(f"{v.get('id',''):<22}{uses:<12}{v.get('status',''):<20}{v.get('file','')}")
    print("\n[读法] 载体是文件不是描述词：同一段描述、同一个 seed 重跑出来的不是同一条嗓子。")
    print("       status=locked 是作者耳朵过掉的；candidate 是能用但没定；retired 是路线判死、文件留着。")
    return 0


def cmd_env(a) -> int:
    print(f"项目根  {ROOT}")
    print(f"分工表  {CFG_PATH}")
    print()
    ok = True
    for role, spec in CFG["roles"].items():
        p = ROOT / spec["python"]
        mark = "OK  " if p.exists() else "MISS"
        if not p.exists():
            ok = False
            print(f"[{mark}] {role:<6} {spec['python']}")
            continue
        rc, out = run_capture([p, "-c",
                               "import sys;print(sys.version.split()[0])"])
        ver = out.strip().splitlines()[-1] if out.strip() else "?"
        print(f"[{mark}] {role:<6} {spec['python']}  (py {ver})  —— {spec.get('desc','')}")
    core = interp("core")
    print("\n核心环境依赖：")
    for mod in ("numpy", "librosa", "soundfile", "scipy", "onnxruntime", "pretty_midi", "demucs_onnx", "matplotlib"):
        rc, out = run_capture([core, "-c", f"import {mod} as m;print(getattr(m,'__version__','?'))"])
        print(f"  {'OK  ' if rc == 0 else 'MISS'} {mod:<12}{out.strip().splitlines()[-1] if rc == 0 and out.strip() else ''}")
        ok = ok and rc == 0
    print("\n结论：" + ("都能跑" if ok else "有缺项，见上面的 MISS"))
    return 0 if ok else 1


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="whale", description="whale-chord 统一入口：听 / 说 / 唱",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("listen", help="听：结构/音符/和弦/调性")
    p.add_argument("audio")
    p.add_argument("--stem", choices=["vocals", "bass", "piano", "guitar", "drums"])
    p.add_argument("--limit-sec", type=float, dest="limit_sec")
    p.add_argument("--bars", type=int)
    p.add_argument("--out")
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser("voice", help="听：这段人声怎么唱")
    p.add_argument("audio")
    p.set_defaults(func=cmd_voice)

    p = sub.add_parser("harmony", help="听：和弦与调性")
    p.add_argument("audio")
    p.add_argument("--label", default="audio")
    p.set_defaults(func=cmd_harmony)

    p = sub.add_parser("compare", help="听：原唱 vs 自己唱，六维对照")
    p.add_argument("files", nargs="+")
    p.add_argument("--plot")
    p.add_argument("--no-plot", action="store_true", dest="no_plot")
    p.add_argument("--json", action="store_true", help="机器可读报告，写进 shared/reports/")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("check", help="入口体检：录音能不能当素材")
    p.add_argument("files", nargs="+")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("speak", help="说：按谱说话")
    p.add_argument("score")
    p.add_argument("out")
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("sing", help="唱：按谱唱歌")
    p.add_argument("score")
    p.add_argument("out")
    p.add_argument("--mode", choices=["human", "cyber"], default="human")
    p.set_defaults(func=cmd_sing)

    p = sub.add_parser("convert", help="换声：RVC（技术验证路线）")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model", default="weixi2")
    p.add_argument("--pitch", type=float, default=0)
    p.add_argument("--f0-method", default="rmvpe", dest="f0_method")
    # index-rate 默认 0.3（2026-11-15 扫描结论，见 shared\reports\sweep-20261115-star.json）：
    # 0.75 会把 2.5k 以上从源里的 0.29% 抬到 0.42%，比 E1 本体的 0.20% 亮一倍（RVC 特征泄漏）；
    # 0~0.3 是个平台（质心 318~321），0.3 保住索引对咬字的稳定作用，又不加那层亮度。
    p.add_argument("--index-rate", type=float, default=0.3, dest="index_rate")
    p.add_argument("--protect", type=float, default=0.33)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("voices", help="音色资产表")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_voices)

    p = sub.add_parser("env", help="环境自检")
    p.set_defaults(func=cmd_env)

    a = ap.parse_args()
    if a.cmd == "harmony" and not getattr(a, "label", None):
        a.label = "audio"
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
