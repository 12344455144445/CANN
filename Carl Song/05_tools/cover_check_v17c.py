# -*- coding: utf-8 -*-
"""
cover_check_v17c.py —— 穷举验证 v17c 的
  (1) AIV 行覆盖：不漏、不重
  (2) ★ 跨核 flag 记账：Set 次数 == Wait 次数，且**不存在悬空 flag**
  (3) 宿主分组不变式：groupCount <= usableCoreNum（单波次，不超订）

v17c 相对 v17b 的唯一结构性改动：
    mBlocksPerGroup : 固定 2  ->  ceilDiv(mBlock 总数, usableCoreNum)，钳到 [1, 2]
    groupCount      : ceilDiv(mBlock 数, 2)   ->  ceilDiv(mBlock 数, mBlocksPerGroup)
    mBlockIndex     : groupIdx*2 + t          ->  groupIdx*mBlocksPerGroup + t
    firstMBlock     : groupIndex*2            ->  groupIndex*mBlocksPerGroup
    SignalAllFlags  : 固定置位 2 个            ->  按 mBlocksPerGroup 置位

(2) 是本轮**新增**的、也是最关键的检查。
    mBlocksPerGroup==1 时 AIV 只 Wait flag 8，从不消费 flag 9。
    若 AIC 仍按「固定 2 个」置位 flag 9，就会在 flag 计数器里留下悬空 +1，
    污染同一 core 上的后续 launch（下一次 flag 9 的 Wait 会提前通过，
    使 AIV 读到未算完的中间结果 -> 精度错）。
    本脚本用「总 Set 数 == 总 Wait 数」逐 flag 记账，专门抓这一类缺陷。
"""
import sys

MAX_GROUP = 16
MAX_FLAG_TILE = 2          # kMaxMBlocksPerGroup
LITERAL_CAP = 2            # QmqCubeSignalTile 内部钳位


def ceil_div(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------- 宿主分组
def plan(m, blockM, usable_core_num):
    """复现宿主侧分组决策（与 kernel_v17c.asc 宿主代码逐式对应）"""
    n_mb = ceil_div(m, blockM)                     # activeAivCount
    per_group = ceil_div(n_mb, usable_core_num)    # mBlocksPerGroup
    if per_group < 1:
        per_group = 1
    if per_group > MAX_FLAG_TILE:
        per_group = MAX_FLAG_TILE
    group_count = ceil_div(n_mb, per_group)        # activeGroupCount
    return n_mb, per_group, group_count


# ---------------------------------------------------------------- 行覆盖
def v17c_rows(m, blockM, per_group, group_count):
    """v17c 的 AIV 行划分（与 kernel 实现逐式对应）"""
    covered = []
    halfM = (blockM + 1) // 2
    for g in range(group_count):
        for t in range(per_group):
            off = (g * per_group + t) * blockM
            if off >= m:
                continue
            mBlockEnd = min(off + blockM, m)
            for s in (0, 1):
                start = off + (0 if s == 0 else halfM)
                if start >= mBlockEnd:
                    continue
                end = min(off + halfM, mBlockEnd) if s == 0 else mBlockEnd
                covered.append((start, end, (g, t, s)))
    return covered


def v17c_rows_buggy(m, blockM, per_group, group_count):
    """v17b 最初的有缺陷写法 min(halfM, m - start)，用于回归对照"""
    covered = []
    halfM = (blockM + 1) // 2
    for g in range(group_count):
        for t in range(per_group):
            off = (g * per_group + t) * blockM
            if off >= m:
                continue
            for s in (0, 1):
                start = off + s * halfM
                if start >= m:
                    continue
                rows = min(halfM, m - start)
                if rows == 0:
                    continue
                covered.append((start, start + rows, (g, t, s)))
    return covered


def audit_rows(covered, m, quiet=False, tag=""):
    errors = []
    for i in range(len(covered)):
        for j in range(i + 1, len(covered)):
            a0, a1, ai = covered[i]
            b0, b1, bj = covered[j]
            if max(a0, b0) < min(a1, b1):
                errors.append("区间重叠 %s 与 %s: [%d,%d) x [%d,%d)"
                              % (ai, bj, a0, a1, b0, b1))
    flag = [0] * m
    for a0, a1, _ in covered:
        if a0 < 0 or a1 > m:
            errors.append("越界 [%d,%d) m=%d" % (a0, a1, m))
            continue
        for r in range(a0, a1):
            flag[r] += 1
    missing = [r for r in range(m) if flag[r] == 0]
    dup = [r for r in range(m) if flag[r] > 1]
    if missing:
        errors.append("漏行 %d 个，例: %s" % (len(missing), missing[:12]))
    if dup:
        errors.append("重复行 %d 个，例: %s" % (len(dup), dup[:12]))
    if errors:
        if not quiet:
            print("[FAIL] %s m=%d:" % (tag, m))
            for e in errors[:6]:
                print("        %s" % e)
        return False
    return True


# ---------------------------------------------------------------- flag 记账
def signal_tile(sets, t):
    """QmqCubeSignalTile：localTile==0 -> flag8；1<=localTile<2 -> flag9；否则不置位"""
    if t == 0:
        sets[0] += 1
    elif t < LITERAL_CAP:
        sets[1] += 1


def wait_tile(waits, t):
    """AIV：localTile==0 等 flag8，否则等 flag9"""
    waits[0 if t == 0 else 1] += 1


def simulate_flags(m, blockM, per_group, group_count, backfill="per_group"):
    """
    backfill:
      "per_group" —— v17c 正确实现：按 per_group 补齐
      "fixed_two" —— v17b 旧实现：固定置位 flag8 + flag9（在 per_group==1 时悬空）
    返回 (sets, waits)，各为 {0: n, 1: n}
    """
    sets = {0: 0, 1: 0}
    waits = {0: 0, 1: 0}

    def backfill_range(a, b):
        if backfill == "fixed_two":
            sets[0] += 1
            sets[1] += 1
        else:
            for rest in range(a, b):
                signal_tile(sets, rest)

    for g in range(group_count):
        # ---------------- AIC 侧 ----------------
        first_off = (g * per_group) * blockM
        if first_off >= m:
            # 早退：本 group 无任何 mBlock
            backfill_range(0, per_group)
        else:
            for t in range(per_group):
                off = (g * per_group + t) * blockM
                if off >= m:
                    # 原 break 分支：补齐 rest = t .. per_group-1
                    backfill_range(t, per_group)
                    break
                signal_tile(sets, t)          # IterateAll 之后置位
        # ---------------- AIV 侧 ----------------
        for t in range(per_group):
            wait_tile(waits, t)

    return sets, waits


def check_flags(m, blockM, per_group, group_count, quiet=False, tag=""):
    sets, waits = simulate_flags(m, blockM, per_group, group_count, "per_group")
    errs = []
    for fid in (0, 1):
        if sets[fid] != waits[fid]:
            errs.append("flag%d 悬空: Set=%d, Wait=%d (差 %d)"
                        % (8 + fid, sets[fid], waits[fid], sets[fid] - waits[fid]))
    if errs:
        if not quiet:
            print("[FAIL] %s m=%d blockM=%d perGroup=%d groups=%d:" %
                  (tag, m, blockM, per_group, group_count))
            for e in errs:
                print("        %s" % e)
        return False
    return True


def check_flags_buggy(m, blockM, per_group, group_count):
    """v17b 旧实现（固定置位 2 个 flag）是否会被抓到？"""
    sets, waits = simulate_flags(m, blockM, per_group, group_count, "fixed_two")
    return sets[0] == waits[0] and sets[1] == waits[1]


# ---------------------------------------------------------------- main
def main():
    ok_all = True

    # ============ (1) 行覆盖穷举 ============
    cases = 0
    for U in (1, 2, 3, 8, 12, 16, 24):
        for m in list(range(1, 130)) + [255, 256, 257, 383, 384, 385,
                                        511, 512, 767, 768, 1023, 1024,
                                        2048, 4096]:
            for blockM in range(1, 40):
                n_mb, per_group, group_count = plan(m, blockM, U)
                if group_count > MAX_GROUP:
                    continue
                cases += 1
                cov = v17c_rows(m, blockM, per_group, group_count)
                if not audit_rows(cov, m, tag="v17c"):
                    ok_all = False
                    return 1
    print("(1) v17c 行覆盖穷举：%d 组 (U, m, blockM) 全部通过（不漏不重）" % cases)

    # ============ (2) flag 记账穷举 ============
    cases = 0
    per_group_hist = {1: 0, 2: 0}
    for U in (1, 2, 3, 8, 12, 16, 24):
        for m in list(range(1, 130)) + [255, 256, 257, 383, 384, 385,
                                        511, 512, 767, 768, 1023, 1024,
                                        2048, 4096]:
            for blockM in range(1, 40):
                n_mb, per_group, group_count = plan(m, blockM, U)
                if group_count > MAX_GROUP:
                    continue
                per_group_hist[per_group] = per_group_hist.get(per_group, 0) + 1
                cases += 1
                if not check_flags(m, blockM, per_group, group_count, tag="v17c"):
                    ok_all = False
                    return 1
    print("(2) v17c flag 记账穷举：%d 组全部通过（逐 flag 的 Set 数 == Wait 数）"
          % cases)
    print("    mBlocksPerGroup 分布：1 -> %d 组，2 -> %d 组"
          % (per_group_hist.get(1, 0), per_group_hist.get(2, 0)))
    if per_group_hist.get(1, 0) == 0:
        print("    [FAIL] 没有任何用例覆盖 mBlocksPerGroup==1，检查有效性不足")
        ok_all = False

    # ---- 回归对照 A：v17b 旧实现（固定置位 2 个 flag）是否被检出 ----
    print()
    print("回归对照 A：v17b 旧实现（SignalAllFlags 固定置位 2 个）")
    caught = tot = 0
    for U in (1, 2, 3, 8, 12, 16, 24):
        for m in list(range(1, 130)) + [256, 512, 1024]:
            for blockM in range(1, 40):
                n_mb, per_group, group_count = plan(m, blockM, U)
                if group_count > MAX_GROUP:
                    continue
                tot += 1
                if not check_flags_buggy(m, blockM, per_group, group_count):
                    caught += 1
    print("    被判为「悬空 flag」的用例数: %d / %d  <- 必须 > 0（证明脚本有检出能力）"
          % (caught, tot))
    if caught == 0:
        print("    [FAIL] 脚本检不出已知缺陷，审计逻辑无效")
        ok_all = False

    # ---- 回归对照 B：v17b 最初的行分配缺陷是否仍被检出 ----
    print()
    print("回归对照 B：行分配缺陷（min(halfM, m-localStart)）")
    caught2 = tot2 = 0
    for U in (8, 24):
        for m in list(range(1, 200)) + [256, 512]:
            for blockM in range(1, 40):
                n_mb, per_group, group_count = plan(m, blockM, U)
                if group_count > MAX_GROUP:
                    continue
                tot2 += 1
                if not audit_rows(v17c_rows_buggy(m, blockM, per_group, group_count),
                                  m, quiet=True):
                    caught2 += 1
    print("    被判为「覆盖不合格」的用例数: %d / %d  <- 必须 > 0" % (caught2, tot2))
    if caught2 == 0:
        print("    [FAIL] 脚本检不出已知缺陷")
        ok_all = False

    # ============ (3) 宿主不变式 ============
    print()
    print("(3) 宿主分组不变式：per_group ∈ {1,2} 且 groupCount <= U（单波次）")
    bad = 0
    checked = 0
    for U in range(1, 49):
        for m in list(range(1, 400)) + [511, 512, 768, 1024, 2048, 4096, 8192]:
            for blockM in range(1, 64):
                n_mb = ceil_div(m, blockM)
                # 宿主的硬前置：nMBlocks <= aivCandidate == 2*U，否则宿主直接 return
                if n_mb > 2 * U:
                    continue
                _, per_group, group_count = plan(m, blockM, U)
                checked += 1
                if per_group not in (1, 2):
                    bad += 1
                if group_count > U:
                    bad += 1
                # 期望：per_group == ceilDiv(n_mb, U)（钳位不生效）
                if per_group != ceil_div(n_mb, U):
                    print("    [WARN] 钳位生效: U=%d m=%d blockM=%d n_mb=%d -> per_group=%d"
                          % (U, m, blockM, n_mb, per_group))
    print("    抽查 %d 组，违反不变式 %d 次" % (checked, bad))
    if bad:
        ok_all = False

    print()
    if ok_all:
        print("结论：v17c 的 AIV 行划分**不漏不重**，flag **逐 id 无悬空**，"
              "分组恒在单波次内 —— 允许提交。")
    else:
        print("结论：存在缺陷，禁止提交")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
