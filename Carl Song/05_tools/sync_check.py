# -*- coding: utf-8 -*-
"""
sync_check.py —— 跨核同步协议自检（v17b 新增，防死锁）

背景：v17b 把"整组一次 Set/Wait"改成"每个 mBlock 一次 Set/Wait"。
一旦某个 AIV 会 Wait 的 flag 在 AIC 侧某条路径上没有 Set，**内核会永久挂死**
（平台表现为超时 / Runtime Error），而且本地完全无法用编译器发现。

本脚本做 4 项静态审查：
  [1] flag 清单：每个 flagID 的 Set 点位 / Wait 点位计数
  [2] Set/Wait 对称性：Set 与 Wait 的 flag 集合必须一致
  [3] AIC 侧"提前返回"补偿：cube 函数（及 AIC 分支）中每条 return 之前
      必须已经置位全部 flag
  [4] 条件置位补偿：出现"可能跳过置位"的分支（break/continue）时，
      必须在退出前补齐剩余 flag

用法:
  python ref/sync_check.py <候选.asc> [--verbose]
"""
import argparse
import re
import sys

SET_RE = re.compile(r"CrossCoreSetFlag\s*<\s*([^>]+?)\s*>\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
WAIT_RE = re.compile(r"CrossCoreWaitFlag\s*<\s*([^>]+?)\s*>\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")


def strip_comments(text):
    """去掉 // 行注释与 /* */ 块注释，保留行号（用空行占位）"""
    out = []
    in_block = False
    for line in text.split("\n"):
        buf = []
        i = 0
        while i < len(line):
            if in_block:
                j = line.find("*/", i)
                if j == -1:
                    i = len(line)
                else:
                    in_block = False
                    i = j + 2
                continue
            if line.startswith("//", i):
                break
            if line.startswith("/*", i):
                in_block = True
                i += 2
                continue
            buf.append(line[i])
            i += 1
        out.append("".join(buf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    raw = open(args.cand, "r", encoding="utf-8").read()
    lines = strip_comments(raw)

    print("=" * 74)
    print("跨核同步协议自检: %s" % args.cand)
    print("=" * 74)

    sets = []   # (lineno, mode, flag)
    waits = []
    for idx, ln in enumerate(lines):
        for m in SET_RE.finditer(ln):
            sets.append((idx + 1, m.group(1).strip(), m.group(2)))
        for m in WAIT_RE.finditer(ln):
            waits.append((idx + 1, m.group(1).strip(), m.group(2)))

    # ---------- [1] flag 清单 ----------
    print("\n[1] flag 清单（Set / Wait 点位）")
    all_flags = sorted(set(f for _, _, f in sets) | set(f for _, _, f in waits))
    if not all_flags:
        print("  未发现任何 CrossCore 同步调用 —— 该版本可能是单核版本")
    for f in all_flags:
        s = [x for x in sets if x[2] == f]
        w = [x for x in waits if x[2] == f]
        smodes = sorted(set(x[1] for x in s))
        wmodes = sorted(set(x[1] for x in w))
        print("  %-24s Set %d 处 %-14s | Wait %d 处 %-14s"
              % (f, len(s), "[%s]" % (",".join(smodes) or "-"),
                 len(w), "[%s]" % (",".join(wmodes) or "-")))
        if args.verbose:
            for ln, mo, _ in s:
                print("        Set  行%d  mode=%s" % (ln, mo))
            for ln, mo, _ in w:
                print("        Wait 行%d  mode=%s" % (ln, mo))

    # ---------- [2] 对称性 ----------
    print("\n[2] Set/Wait 对称性")
    set_flags = set(f for _, _, f in sets)
    wait_flags = set(f for _, _, f in waits)
    ok = True
    if wait_flags - set_flags:
        print("  [FAIL] 以下 flag 被 Wait 但从未被 Set -> **必然死锁**")
        for f in sorted(wait_flags - set_flags):
            print("         %s" % f)
        ok = False
    if set_flags - wait_flags:
        print("  [WARN] 以下 flag 被 Set 但从未被 Wait（多余信号，通常无害）")
        for f in sorted(set_flags - wait_flags):
            print("         %s" % f)
    # mode 一致性
    for f in sorted(wait_flags & set_flags):
        smodes = set(x[1] for x in sets if x[2] == f)
        wmodes = set(x[1] for x in waits if x[2] == f)
        if smodes != wmodes:
            print("  [FAIL] %s 的 mode 不对称: Set=%s Wait=%s" % (f, smodes, wmodes))
            ok = False
    if ok:
        print("  OK  每个 Wait 的 flag 都有对应 Set，且 mode 对称")

    # ---------- [3] return 前必须已置位 ----------
    print("\n[3] 提前返回路径的置位补偿")
    # 找 cube 函数的起止
    start = None
    for i, ln in enumerate(lines):
        if "quant_matmul_relu_quant_cube(" in ln and "void" in ln:
            start = i
            break
    if start is None:
        print("  [SKIP] 未定位到 quant_matmul_relu_quant_cube")
    else:
        depth = 0
        end = None
        seen_open = False
        for i in range(start, len(lines)):
            for ch in lines[i]:
                if ch == "{":
                    depth += 1
                    seen_open = True
                elif ch == "}":
                    depth -= 1
                    if seen_open and depth == 0:
                        end = i
                        break
            if end is not None:
                break
        print("  cube 函数范围: 行 %d ~ %d" % (start + 1, (end or start) + 1))
        cube_has_set = any(start + 1 <= ln <= (end or start) + 1 for ln, _, _ in sets)
        # 也可能通过 Signal helper 间接置位（helper 定义在函数外）
        if not cube_has_set:
            body_txt = "\n".join(lines[start:(end or start) + 1])
            if re.search(r"\b\w*Signal\w*\s*\(", body_txt):
                cube_has_set = True
        if cube_has_set:
            print("  结构：置位在 cube 函数**内部** -> 要求每条 return 前都已置位")
            bad = []
            for i in range(start, (end or start) + 1):
                s = lines[i].strip()
                if s == "return;":
                    win = "\n".join(lines[max(start, i - 12):i])
                    if not re.search(
                            r"CrossCoreSetFlag|SignalAllFlags|SignalTile"
                            r"|MaybeSignalAll|MaybeSignalTile", win):
                        bad.append(i + 1)
            if bad:
                print("  [FAIL] 以下 return 之前未见置位调用 -> **可能死锁**")
                for ln in bad:
                    print("         行 %d" % ln)
                ok = False
            else:
                print("  OK  所有 return 之前均已置位")
        else:
            print("  结构：置位在 cube 函数**外部**（调用点之后无条件执行）")
            call_idx = None
            for i in range(start, (end or start) + 2):
                if i > start and "quant_matmul_relu_quant_cube(" in lines[i]:
                    call_idx = i
                    break
            if call_idx is None:
                # 调用点在 cube 定义之后，全局再搜一次
                for i, ln in enumerate(lines):
                    if i > (end or start) and "quant_matmul_relu_quant_cube(" in ln:
                        call_idx = i
                        break
            if call_idx is None:
                print("  [SKIP] 未找到调用点，请人工确认")
            else:
                tail = "\n".join(lines[call_idx + 1:call_idx + 6])
                if re.search(r"CrossCoreSetFlag", tail):
                    print("  OK  调用点之后无条件置位（行 %d 起）" % (call_idx + 1))
                else:
                    print("  [FAIL] 调用点之后未见置位 -> **可能死锁**（行 %d）"
                          % (call_idx + 1))
                    ok = False

    # ---------- [4] 可能跳过置位的分支 ----------
    print("\n[4] 可能跳过置位的控制流")
    suspects = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("break;") or s.startswith("continue;"):
            back = "\n".join(lines[max(0, i - 10):i])
            fwd = "\n".join(lines[i + 1:i + 8])
            # break/continue 所在分支若通向"置位"或"退出"，需要有人补 flag
            if not re.search(
                    r"SignalAllFlags|SignalTile|MaybeSignalAll|MaybeSignalTile"
                    r"|CrossCoreSetFlag|CrossCoreWaitFlag",
                    back + fwd):
                suspects.append((i + 1, s))
    if suspects:
        print("  [INFO] 以下 break/continue 附近未见置位调用，请人工确认这些分支")
        print("         不处于『AIV 已进入 Wait』的路径上：")
        for ln, s in suspects:
            print("         行 %-5d %s" % (ln, s))
    else:
        print("  OK  未发现可疑的 break/continue")

    # ---------- [5] 置位辅助调用点 ----------
    print("\n[5] 置位辅助调用点（人工核对：每次 launch 每个 flag 恰好 Set 一次）")
    helper_names = []
    for ln in lines:
        m = re.match(r"\s*__aicore__\s+inline\s+void\s+(\w+)\s*\(", ln)
        if m and ("Signal" in m.group(1) or "SetFlag" in m.group(1)):
            helper_names.append(m.group(1))
    if not helper_names:
        print("  未使用置位辅助；请直接核对上面的 [1] flag 清单")
    for h in helper_names:
        call_lines = []
        for i, ln in enumerate(lines):
            if re.search(r"\b%s\s*\(" % re.escape(h), ln) and \
                    not re.match(r"\s*__aicore__\s+inline", ln):
                call_lines.append(i + 1)
        print("  %-26s 调用 %d 处: %s"
              % (h + "()", len(call_lines),
                 ", ".join("行%d" % x for x in call_lines) or "-"))
    print("  说明：信号 helper 的**定义体**会被 [1] 各计 1 处；")
    print("        这里列出的是真正的运行期调用点。两者结合起来核对：")
    print("        「每个 AIC group 每次 launch 对每个 flag 的 Set 次数」应为 1。")

    # ---------- [6] signal 开关一致性（V22 新增；V24 升级为『守卫等价』判据） ----------
    # 动机：把 cube 段搬进独立 kernel、或用 mode 参数复用同一 kernel 后，AIC 不一定
    # 有 AIV 消费者；若仍置位就会在 flag 计数器上留下悬空 +1，污染同一 core 上紧随
    # 其后的 launch（E22）。
    #
    # 判据（V24 起）—— 取每个 cube 调用点的 signal 实参：
    #   · 字面量 true   —— 要求该 kernel 体内确有 CrossCoreWaitFlag；
    #   · 字面量 false  —— 要求该 kernel 体内没有 CrossCoreWaitFlag；
    #   · 其它表达式 E  —— **每一条** CrossCoreWaitFlag 的外层守卫链里都必须含 E，
    #                      即「等待发生的路径 ⊆ E 为真的路径」，两者同开关。
    #   （E31：检查器只认字面量会在改名/重构后假阳性；这里改为校验守卫表达式，
    #     判据不放松 —— 表达式分支比字面量分支要求更多。）
    print("\n[6] cube 函数 signal 开关与 AIV 等待的一致性")
    raw = "\n".join(lines)
    mdef = re.search(r"void\s+quant_matmul_relu_quant_cube\s*\(([^)]*)\)", raw, re.S)
    if not mdef:
        print("  [SKIP] 未定位到 cube 函数定义")
    elif "signal" not in mdef.group(1):
        print("  [INFO] cube 函数无 signal 形参 -> 跳过（旧版无条件置位行为）")
    else:
        kstarts = [mm.start() for mm in re.finditer(r"__global__", raw)]
        kstarts.append(len(raw))

        def normalize(expr):
            return re.sub(r"\s+", " ", expr or "").strip()

        def wait_guard_chains(body):
            """返回每条 CrossCoreWaitFlag 的**外层守卫链**（由外到内，忽略 None）。"""
            stack = []
            pending = None
            out = []
            i = 0
            n = len(body)
            while i < n:
                ch = body[i]
                if ch == '{':
                    stack.append(pending)
                    pending = None
                elif ch == '}':
                    if stack:
                        stack.pop()
                else:
                    m = re.compile(r"if\s*\(").match(body, i)
                    if m:
                        j = body.index('(', i)
                        d = 0
                        k = j
                        while k < n:
                            if body[k] == '(':
                                d += 1
                            elif body[k] == ')':
                                d -= 1
                                if d == 0:
                                    break
                            k += 1
                        pending = normalize(body[j + 1:k])
                        i = k + 1
                        continue
                    if body.startswith("CrossCoreWaitFlag", i):
                        out.append([g for g in stack if g])
                        i += len("CrossCoreWaitFlag")
                        continue
                i += 1
            return out

        def kernel_of(idx):
            found = None
            for j in range(len(kstarts) - 1):
                if kstarts[j] <= idx < kstarts[j + 1]:
                    found = j
            return found

        bad = []
        for mm in re.finditer(r"quant_matmul_relu_quant_cube\s*\(", raw):
            pre = raw[max(0, mm.start() - 40):mm.start()]
            if "void" in pre or "inline" in pre:
                continue                      # 定义本身
            open_idx = raw.index("(", mm.start())
            depth = 0
            close_idx = None
            for p in range(open_idx, len(raw)):
                if raw[p] == "(":
                    depth += 1
                elif raw[p] == ")":
                    depth -= 1
                    if depth == 0:
                        close_idx = p
                        break
            if close_idx is None:
                continue
            last = normalize(raw[open_idx + 1:close_idx].rsplit(",", 1)[-1])
            ln = raw[:mm.start()].count("\n") + 1
            k = kernel_of(mm.start())
            if k is None:
                print("  [SKIP] 行 %d 的调用点不在任何 __global__ 内" % ln)
                continue
            km = re.search(r"void\s+(\w+)\s*\(", raw[kstarts[k]:kstarts[k] + 400])
            kn = km.group(1) if km else ("kernel@%d" % k)
            chains = wait_guard_chains(raw[kstarts[k]:kstarts[k + 1]])
            if last == "true":
                good = bool(chains)
                why = "字面 true，且有 AIV 等待"
            elif last == "false":
                good = not chains
                why = "字面 false，且无 AIV 等待"
            else:
                good = bool(chains) and all(last in ch for ch in chains)
                why = ("表达式守卫一致 x%d" % len(chains)) if good else \
                      "表达式未覆盖全部等待路径"
            if not good:
                bad.append(ln)
            print("  [%s] %-24s 行%-5d signal=%-24s AIV等该flag=%s  (%s)"
                  % ("OK " if good else "FAIL", kn, ln, last,
                     "Y" if chains else "N", why))
        if bad:
            print("  [FAIL] signal 与 AIV 等待不一致 -> 会死锁或残留悬空 flag")
            ok = False
        else:
            print("  OK  signal 的开关语义与 AIV 的等待路径完全一致")

    print("\n" + "=" * 74)
    if ok:
        print("同步协议自检通过")
    else:
        print("同步协议自检 **未通过** —— 存在死锁风险，请勿提交")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
