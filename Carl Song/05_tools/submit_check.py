#!/usr/bin/env python3
"""
提交前自检（单进程版）

为什么要有这一版
----------------
最初的 submit_check.sh 用 shell 拼了几十个 grep/sed/mktemp/rm。
在本机实测，**每次创建子进程约 450ms**（20 次 /bin/true 要 9 秒），
于是整个自检要跑 27 秒，其中 sys 时间 17 秒 —— 几乎全花在进程创建上，
真正干活的两个 Python 检查各只要 0.9 秒。

本脚本把全部检查合并进一个 Python 进程，不再产生任何子进程，
并直接 import 复用 precision_guard / static_check 的现有实现，
避免逻辑在两处重复、两边结论不一致。

检查项
------
  [1] kernel.asc 是否合乎"被 #include 的片段"规范
  [2] 精度关键路径是否被改动（红线，必须 PASS）
  [3] 源码结构（括号配对、UB 预算、队列配对）
  [4] 数值路径的浮点乘法顺序关键序列

退出码: 0 = 可以提交; 1 = 有问题
"""

import argparse
import contextlib
import io
import os
import re
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import precision_guard as pg          # noqa: E402
import static_check as sc             # noqa: E402


# 本工具所在目录的上一级。两种布局都成立:
#   分析工作区:  <root>/ref/submit_check.py
#   自包含包:    <root>/_tools/submit_check.py
ROOT = os.path.dirname(HERE)


def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def default_candidate():
    """候选 kernel.asc 的默认位置。"""
    return _first_existing([
        os.path.join(ROOT, 'kernel.asc'),            # 自包含包 / 竞赛工程根
        os.path.join(ROOT, 'submission', 'kernel.asc'),
    ]) or os.path.join(ROOT, 'kernel.asc')


def resolve_baseline(explicit):
    """基线搜索顺序：显式指定 > 已验证 checkpoint > 随包基线 > 用户提供版。

    返回 (路径, 是否退回到非验证版本)。
    """
    if explicit:
        return explicit, False

    real = _first_existing([
        os.path.join(ROOT, 'records', 'checkpoints', '551961-kernel.asc'),
    ])
    if real:
        return real, False

    packaged = _first_existing([
        os.path.join(HERE, 'baseline_kernel.asc'),   # 自包含包自带
    ])
    if packaged:
        return packaged, True

    fallback = _first_existing([
        os.path.join(ROOT, 'records', 'pasted_version', 'kernel.asc'),
        os.path.join(ROOT, 'baseline', 'kernel.asc'),
    ])
    if fallback:
        return fallback, True

    return None, False


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def strip_comments(text):
    """去掉块注释与行注释，保留行结构（避免注释里的字样造成误报）。"""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    return text


def capture(fn, *args, **kwargs):
    """在进程内调用 fn 并把它的 stdout 收进字符串，返回 (返回值, 输出)。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*args, **kwargs)
    return rc, buf.getvalue()


# ---------------------------------------------------------------------------
# [1] 规范性
# ---------------------------------------------------------------------------
def check_normative(path):
    lines = []
    ok = True

    if not os.path.isfile(path):
        return False, [f"  FAIL 找不到 {path}"]

    with open(path, encoding='utf-8', errors='replace') as f:
        code = strip_comments(f.read())

    if re.search(r'\b(?:int|void)\s+main\s*\(', code):
        lines.append("  FAIL 含 main() —— 本文件会被 main.asc 以 #include 方式引入，"
                     "不可定义 main")
        ok = False

    if re.search(r'^\s*#\s*pragma\s+once', code, re.M):
        lines.append("  FAIL 含 #pragma once —— 被 #include 的片段不应有")
        ok = False

    if re.search(r'^\s*#\s*ifndef\s+\w*_(?:H|H_|HPP)\s*$', code, re.M):
        lines.append("  WARN 疑似 include guard —— 被 #include 的片段通常不需要")

    if re.search(r'extern\s+"C"\s+void\s+run_kernel\s*\(', code):
        lines.append("  OK   存在 extern \"C\" void run_kernel 入口")
    else:
        lines.append("  FAIL 找不到 extern \"C\" void run_kernel —— "
                     "评测框架依赖此入口")
        ok = False

    if ok:
        lines.append("  OK   规范性检查通过")
    return ok, lines


# ---------------------------------------------------------------------------
# [3] 结构
# ---------------------------------------------------------------------------
def check_structure(path, ub_reference_kb):
    lines = []
    ok = True

    if not os.path.isfile(path):
        return False, ["  FAIL 文件不存在，无法做结构检查"], {}

    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()

    # --- 3.1 括号配对 ---
    errs = sc.check_braces(text, path)
    if errs:
        lines.append("  FAIL 括号配对异常:")
        lines.extend("    " + e.strip() for e in errs)
        ok = False
    else:
        lines.append("  OK   括号与大括号配对正常")

    # --- 3.2 UB 预算 ---
    saved = sc.UB_REFERENCE_KB
    try:
        sc.UB_REFERENCE_KB = ub_reference_kb
        ub_lines, ub_total = sc.check_ub_budget(text, path)
    finally:
        sc.UB_REFERENCE_KB = saved

    lines.append("")
    lines.append("  UB 预算:")
    lines.extend("  " + l for l in ub_lines)
    if ub_total is not None and ub_reference_kb > 0 \
            and ub_total > ub_reference_kb * 1024:
        ok = False

    # --- 3.3 队列配对 ---
    lines.append("")
    lines.append("  队列操作配对:")
    pair_lines = sc.check_tensor_pairs(text, path)
    # 只保留结论行与异常行，明细太长
    for l in pair_lines:
        if '[PASS]' in l or '[FAIL]' in l:
            lines.append("  " + l.strip())
    if any('[FAIL]' in l for l in pair_lines):
        ok = False

    return ok, lines, {'ub_total': ub_total}


# ---------------------------------------------------------------------------
# [4] 数值路径
# ---------------------------------------------------------------------------
def numeric_section(guard_out):
    """从守卫输出里截出"数值路径函数"那一段。"""
    start = guard_out.find('[3] 数值路径函数')
    if start < 0:
        return ''
    end = guard_out.find('\n\n', start)
    return guard_out[start:end if end > 0 else len(guard_out)]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="提交前自检")
    ap.add_argument('--cand', default=default_candidate(),
                    help='候选 kernel.asc')
    ap.add_argument('--base', default=None, help='基线 kernel.asc')
    ap.add_argument('--ub-reference', type=float, default=sc.UB_REFERENCE_KB,
                    help=f'UB 基准 KB（默认 {sc.UB_REFERENCE_KB}，'
                         '即已验证版本实测占用）')
    ap.add_argument('--quiet', action='store_true', help='只输出结论行')
    args = ap.parse_args()

    cand = os.path.abspath(args.cand)
    baseline, is_fallback = resolve_baseline(
        os.path.abspath(args.base) if args.base else None)

    out = []
    out.append("=" * 70)
    out.append("提交前自检")
    out.append("=" * 70)
    out.append(f"  候选项  : {cand}")
    out.append(f"  基线    : {baseline or '<未找到>'}")
    if is_fallback:
        out.append("")
        out.append("  !! 未找到真实验证过的 records/checkpoints/551961-kernel.asc")
        out.append("     当前退回到「用户提供的版本」作为基线。")
        out.append("     它同样是可用的精度基准，")
        out.append("     但不等于平台上已验证 15/15 的那份。")
    out.append("")

    n_fail = 0
    n_warn = 0

    # ---- [1] ----
    ok1, l1 = check_normative(cand)
    out.append("[1] kernel.asc 规范性")
    out.extend(l1)
    out.append("")
    if not ok1:
        n_fail += 1

    # ---- [2] + [4] 守卫只跑一次 ----
    guard_out = ''
    guard_ran = False
    guard_ok = False

    if not os.path.isfile(cand):
        guard_msg = ["  WARN 候选文件不存在，跳过精度检查"]
        n_warn += 1
    elif not baseline:
        guard_msg = ["  WARN 找不到任何基线 —— 精度检查被跳过",
                     "       请把已验证版本放到 records/checkpoints/551961-kernel.asc",
                     "       或用 --base <路径> 指定"]
        n_warn += 1
    else:
        rc, guard_out = capture(pg.compare, baseline, cand, False)
        guard_ran = True
        guard_ok = (rc == 0)
        guard_msg = []

    out.append("[2] 精度关键路径（红线）")
    if guard_ran:
        if guard_ok:
            out.append("  OK   精度关键路径未被改动")
        else:
            out.append("  FAIL 精度关键路径被改动 —— 不要提交！")
            for l in guard_out.split('\n'):
                if re.match(r'^  (FAIL|PASS) ', l) or l.startswith('    ['):
                    out.append("     " + l.strip())
            n_fail += 1
    else:
        out.extend(guard_msg)
    out.append("")

    # ---- [3] ----
    ok3, l3, _ = check_structure(cand, args.ub_reference)
    out.append("[3] 源码结构")
    out.extend(l3)
    out.append("")
    if not ok3:
        n_fail += 1

    # ---- [4] ----
    out.append("[4] 数值路径关键序列")
    if not guard_ran:
        out.append("  WARN 守卫未执行，跳过")
    else:
        sec = numeric_section(guard_out)
        if not sec:
            out.append("  WARN 未解析到数值路径段落")
        else:
            has_diff = '**差异**' in sec
            for l in sec.split('\n'):
                if re.match(r'^\s*(PASS|NOTE|\?)', l) or '差异' in l:
                    out.append("  " + l.strip())
            if has_diff:
                out.append("  FAIL 关键序列存在差异"
                           "（可能是浮点乘法顺序被改）")
                n_fail += 1
    out.append("")

    # ---- 汇总 ----
    out.append("=" * 70)
    if n_fail == 0:
        out.append(f"自检通过（警告 {n_warn} 项）")
        out.append("")
        out.append("接下来:")
        out.append("  1. 真机跑完整回归:  cd build && bash ../run_all_cases.sh")
        out.append("  2. 用 msprof 采集性能数据")
        out.append("  3. 上传 kernel.asc 到评测平台")
    else:
        out.append(f"自检未通过（失败 {n_fail} 项，警告 {n_warn} 项）")
        out.append("")
        out.append("精度不过等于 0 分，请先修复再提交。")
    out.append("=" * 70)

    print('\n'.join(out))
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
