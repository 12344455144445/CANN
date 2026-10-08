#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
selfcheck_all.py —— 提交前自检总驱动（九道）

把「每次提交前必跑」的自检固化成一条命令，避免漏跑。
任一 hard 项失败即返回非 0。

用法
----
    python ref/selfcheck_all.py --cand <候选.asc> [--base <基线.asc>]
默认 base = records/experiments/v20-mdl/kernel_v20_mdl.asc
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
DEFAULT_BASE = os.path.join(ROOT, "records", "experiments", "v20-mdl",
                            "kernel_v20_mdl.asc")

# (名称, 命令行, 是否 hard)
CHECKS = [
    ("结构+UB+精度红线+数值序列", ["submit_check.py", "--cand", "{cand}",
                                    "--base", "{base}"], True),
    ("作用域（E15）", ["scope_check.py", "{cand}"], True),
    ("预处理/配平（E19 前哨）", ["cpp_preprocess_check.py", "{cand}"], True),
    ("跨核同步 + signal 开关（E22）", ["sync_check.py", "{cand}"], True),
    ("形参==实参", ["sig_check.py", "{cand}"], True),
    ("精度守卫（三红线 fingerprint）", ["precision_guard.py",
                                        "--baseline", "{base}",
                                        "--cand", "{cand}"], True),
    ("v17c 行覆盖+flag 记账（公式回归）", ["cover_check_v17c.py"], True),
    ("v22 新增行划分穷举", ["cover_check_v22.py"], True),
    ("活代码剔注释 diff", ["livecode_diff.py", "--a", "{base}", "--b",
                           "{cand}", "--show", "0"], False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cand = os.path.abspath(args.cand)
    base = os.path.abspath(args.base)
    if not os.path.isfile(cand):
        print("[FAIL] 候选不存在: %s" % cand)
        return 2
    if not os.path.isfile(base):
        print("[FAIL] 基线不存在: %s" % base)
        return 2

    print("=" * 74)
    print("提交前自检总驱动")
    print("  候选 : %s" % cand)
    print("  基线 : %s" % base)
    print("=" * 74)

    hard_fail, soft_fail = [], []
    for idx, (name, argv, hard) in enumerate(CHECKS, 1):
        cmd = [PY, os.path.join(HERE, argv[0])]
        cmd += [x.format(cand=cand, base=base) for x in argv[1:]]
        print("\n[%d/%d] %s" % (idx, len(CHECKS), name))
        print("-" * 74)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        out = (r.stdout or "") + (r.stderr or "")
        if not args.quiet:
            print(out.rstrip()[-1800:])
        ok = (r.returncode == 0)
        # 兜底：有些脚本恒返回 0，靠输出里的关键字判断
        if ok and ("未通过" in out or "请勿提交" in out or "[FAIL]" in out):
            ok = False
        tag = "PASS" if ok else ("FAIL(hard)" if hard else "FAIL(soft)")
        print("  ==> %s" % tag)
        if not ok:
            (hard_fail if hard else soft_fail).append(name)

    print("\n" + "=" * 74)
    if hard_fail:
        print("结论：**不可提交** —— hard 项未通过：")
        for n in hard_fail:
            print("   - %s" % n)
    elif soft_fail:
        print("结论：hard 项全通过；soft 项需人工确认：")
        for n in soft_fail:
            print("   - %s" % n)
    else:
        print("结论：九道全通过 —— 允许提交")
    print("=" * 74)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
