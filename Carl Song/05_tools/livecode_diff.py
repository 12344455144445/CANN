#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
livecode_diff.py —— 剔注释后的「活代码」逐行对比（通用版）

`check_v20_mdl.py` 是 v17c-vs-v20 的专用版（路径写死）。本脚本把它参数化，
供后续版本（v22 起）复用。

为什么必须剔注释：整行注释的增删会污染 `submit_check.py` 的「字符数变化」
NOTE（它是剔注释启发式），也会掩盖真正的语义差异。语义差异只以本脚本为准。

用法
----
    python ref/livecode_diff.py --a <基线.asc> --b <候选.asc> [--show 40]
"""

import argparse
import difflib
import os
import sys


def read(path):
    return open(path, "r", encoding="utf-8", newline="").read()


def strip_comments(text):
    """去掉整行注释（// 开头，允许前导空白）与空行。

    只处理**整行**注释，不处理行尾注释 —— 本工程注释一律独占整行，
    行尾注释若被误删反而可能掩盖差异，保守起见不动。
    """
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s == "" or s.startswith("//"):
            continue
        out.append(ln.rstrip())
    return out


def device_region(text):
    """从第一个 __aicore__ 函数开始到文件末尾（device 侧全部代码）。"""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "__aicore__" in ln:
            return lines[i:]
    return []


def hr(title=None):
    print("=" * 74)
    if title:
        print(title)
        print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="基线 .asc")
    ap.add_argument("--b", required=True, help="候选 .asc")
    ap.add_argument("--show", type=int, default=40, help="最多显示多少行 diff")
    args = ap.parse_args()

    for p in (args.a, args.b):
        if not os.path.isfile(p):
            print("[FAIL] 文件不存在: %s" % p)
            return 1

    a_raw, b_raw = read(args.a), read(args.b)
    a, b = strip_comments(a_raw), strip_comments(b_raw)

    hr("活代码等价性对比器（剔注释）")
    print("A(基线) : %s" % args.a)
    print("B(候选) : %s" % args.b)
    print()

    # ---------------------------------------------------------------- [1]
    hr("[1] 活代码逐行 diff（剔掉整行注释与空行）")
    print("  活代码行数：A = %d，B = %d" % (len(a), len(b)))
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    n_add = sum(o[4] - o[3] for o in ops if o[0] in ("insert", "replace"))
    n_del = sum(o[2] - o[1] for o in ops if o[0] in ("delete", "replace"))
    print("  diff 块数：%d   (+%d / -%d 行)" % (len(ops), n_add, n_del))
    shown = 0
    for tag, i1, i2, j1, j2 in ops:
        print()
        print("    ---- [%s] A行 %d..%d  ->  B行 %d..%d ----"
              % (tag, i1 + 1, i2, j1 + 1, j2))
        for x in a[i1:i2]:
            if shown < args.show:
                print("    - %s" % x.strip())
                shown += 1
        for y in b[j1:j2]:
            if shown < args.show:
                print("    + %s" % y.strip())
                shown += 1
        if shown >= args.show:
            print("    ... (达到 --show 上限，其余省略)")
            break

    # ---------------------------------------------------------------- [2]
    hr("[2] device 侧区域（第一个 __aicore__ 起）剔注释后对比")
    da = strip_comments("\n".join(device_region(a_raw)))
    db = strip_comments("\n".join(device_region(b_raw)))
    print("  device 活代码行数：A = %d，B = %d" % (len(da), len(db)))
    print("  [%s] device 侧逐行相同" % ("OK " if da == db else "!! "))
    if da != db:
        sm2 = difflib.SequenceMatcher(a=da, b=db, autojunk=False)
        for tag, i1, i2, j1, j2 in sm2.get_opcodes():
            if tag == "equal":
                continue
            print("    [%s] A %d..%d -> B %d..%d" % (tag, i1 + 1, i2, j1 + 1, j2))
            for x in da[i1:i2]:
                print("      - %s" % x.strip())
            for y in db[j1:j2]:
                print("      + %s" % y.strip())

    # ---------------------------------------------------------------- [3]
    hr("[3] host 侧关键契约逐项核对")

    RED_FUNCS = ("QmqQuantizeNearHalf", "QmqDiv", "QmqCorrectDiv")

    def extract_redline(src):
        """抽三个精度红线函数的**定义体**（签名行到花括号配平的末行）。

        必须按「函数体」而不是「含关键词的行」来抽：QmqDiv 的调用点
        （如 `QmqDiv(fp32Local, activationLocal, negScaleLocal, alignedN);`）
        同样含 `QmqDiv` 子串，按行抽会把**合法的调用点参数改写**误报成
        红线改动 —— 而红线约束的只是**函数实现**逐字节不变。
        """
        out, i = [], 0
        while i < len(src):
            ln = src[i]
            is_def = any(n + "(" in ln and ("inline" in ln or "aicore" in ln)
                         for n in RED_FUNCS)
            if not is_def:
                i += 1
                continue
            j = i
            while j < len(src) and "{" not in src[j]:
                j += 1
            depth, k = 0, j
            while k < len(src):
                depth += src[k].count("{") - src[k].count("}")
                if depth <= 0:
                    break
                k += 1
            out.extend(src[i:k + 1])
            i = k + 1
        return out

    def extract_key(src, names, custom=None):
        if custom is not None:
            return custom(src)
        return [ln for ln in src if any(k in ln for k in names)]

    # hard=True  -> 必须逐行完全一致（语义红线 / 数值路径）
    # hard=False -> 只报告「B 新增了哪些行」，供人工确认是否都在预期补丁点
    groups = [
        ("精度红线", True, None, extract_redline),
        ("tiling 调用", True, ["tilingApi.", "GetTiling"], None),
        ("workspace 获取", True, ["QmqAcquireWorkspace", "QmqWorkspaceCache",
                                  "resultWorkspaceSize", "systemWorkspaceSize",
                                  "requestedSystemWorkspaceSize"], None),
        ("核间同步", False, ["CrossCoreWaitFlag", "CrossCoreSetFlag",
                            "QmqCubeSignalTile", "QmqCubeSignalAllFlags",
                            "QmqCubeMaybeSignal"], None),
        ("常量块", False, ["constexpr uint32_t", "constexpr uint16_t",
                          "constexpr float", "constexpr uint64_t",
                          "constexpr auto"], None),
        ("14 形参签名", False, ["run_kernel", "GM_ADDR", "TensorGroupInfo",
                               "availableCoreNum", "aclrtStream"], None),
    ]
    hard_fail = []
    for name, hard, keys, custom in groups:
        ga, gb = extract_key(a, keys, custom), extract_key(b, keys, custom)
        if hard:
            same = (ga == gb)
            if not same:
                hard_fail.append(name)
            print("  [%s] %-16s A %3d 行 / B %3d 行   %s"
                  % ("OK " if same else "!! ", name, len(ga), len(gb),
                     "逐行一致" if same else "**不一致（红线）**"))
            if not same:
                import difflib as _d
                for ln in _d.unified_diff(ga, gb, "A", "B", lineterm="", n=0):
                    if ln[:1] in "+-" and ln[:3] not in ("+++", "---"):
                        print("           %s" % ln.strip())
        else:
            # 只报告 B 里新增的行（A 中不存在的），这才是真实改动
            from collections import Counter
            ca = Counter(ga)
            new = []
            for ln in gb:
                if ca[ln] > 0:
                    ca[ln] -= 1
                else:
                    new.append(ln)
            print("  [ok ] %-16s A %3d 行 / B %3d 行   新增 %d 行"
                  % (name, len(ga), len(gb), len(new)))
            for ln in new[:14]:
                print("           + %s" % ln.strip())
            if len(new) > 14:
                print("           ... 另有 %d 行" % (len(new) - 14))

    hr("[4] 结论")
    print("  活代码净增：+%d 行" % (n_add - n_del))
    print("  host 红线：%s" % ("全部一致 ✓" if not hard_fail
                              else "**不一致: %s**" % ", ".join(hard_fail)))
    print("  说明：v22 起 device 侧**必然有改动**（新增 kernel + signal 形参），")
    print("        故 [2] 的『完全相同』不再是验收标准；验收标准改为：")
    print("        ① 精度红线 0 改动 ② tiling 调用 0 改动")
    print("        ③ 同步协议自洽（sync_check 的 [6] 段）")
    print("        ④ 新增行只落在预期补丁点（见上面「新增」清单）")
    hr()
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
