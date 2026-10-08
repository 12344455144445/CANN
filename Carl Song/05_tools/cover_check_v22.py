#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
cover_check_v22.py —— v22 新增代码的穷举证明

v22 相对 v20 只改了两件事：
  (A) cube 段搬进 qmq_cube_only，任务映射**逐字节不变**（groupIdx = GetBlockIdx()，
      firstMBlock = groupIndex * mBlocksPerGroup）⇒ 原有 cover_check_v17c 的结论
      直接继承，无需重证。
  (B) vector 段搬进 qmq_vector_only，**换了一套行划分**：
        rowsPerAiv = ceil(m / aivCount)
        rowStart   = aivIdx * rowsPerAiv
        rows       = min(rowsPerAiv, m - rowStart)      (rowStart < m 才算)
      内层再按 kVectorTileRows 分批。
      这套划分 v17c 的脚本**不覆盖**，必须单独穷举证明「不漏不重」。

本脚本做三件事：
  [1] 行划分穷举：所有 (m, aivCount) 下，各子核行区间并集 == [0, m)，且两两不交。
  [2] 内层分批穷举：rows 在 [1, 65536] 内，分批区间并集 == [0, rows)，且两两不交。
  [3] 判据决策表：QmqUseSplitPath 在 U=24 下的落点（预登记签名）。

用法: python ref/cover_check_v22.py
"""

import sys

K_VECTOR_TILE_ROWS = 1024
K_SPLIT_MIN_WORK = 4096


def ceil_div(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------- [1] 行划分 ---
# 闭式证明（本脚本只做数值复核，证明本身是初等事实）：
#   rp = ceil(m/A)，第 i 个子核区间 = [i·rp, i·rp + rp)（最后一个截断到 m）。
#   · 不重叠：各区间左端单调递增且步长恰为 rp ⇒ 相邻区间首尾相接，天然不交；
#   · 不漏：参与的 i 有 K = ceil(m/rp) 个，并集 = [0, K·rp) ∩ [0, m)，
#     而 K·rp >= m ⇒ 并集 = [0, m)；
#   · 不越界：每段都夹在 min(rp, m - rowStart) 内 ⇒ 并集 <= [0, m)。
def check_row_partition(m_list, aiv_counts):
    bad = []
    total = 0
    for aiv_count in aiv_counts:
        for m in m_list:
            rp = ceil_div(m, aiv_count)
            prev_end = 0
            count = 0
            for aiv in range(aiv_count):
                rs = aiv * rp
                if rs >= m:
                    continue
                rows = min(rp, m - rs)
                if rows <= 0:
                    bad.append((aiv_count, m, aiv, "rows<=0"))
                    continue
                if rs != prev_end:
                    bad.append((aiv_count, m, aiv,
                                "空洞/重叠: rs=%d prev_end=%d" % (rs, prev_end)))
                prev_end = rs + rows
                count += rows
            total += 1
            if count != m or prev_end != m:
                bad.append((aiv_count, m, None,
                            "count=%d prev_end=%d != m=%d" % (count, prev_end, m)))
    return total, bad


# ------------------------------------------------------------ [2] 内层分批 ---
# 同理：ro 以 kVectorTileRows 递增、每段长 min(T, rows-ro) ⇒ 首尾相接、并集 [0,rows)。
def check_inner_batch(rows_list):
    bad = []
    total = 0
    for rows in rows_list:
        prev_end = 0
        count = 0
        ro = 0
        guard = 0
        while ro < rows:
            guard += 1
            if guard > 100000:
                bad.append((rows, ro, "死循环"))
                break
            cur = min(K_VECTOR_TILE_ROWS, rows - ro)
            if cur <= 0:
                bad.append((rows, ro, "cur<=0"))
                break
            if ro != prev_end:
                bad.append((rows, ro, "空洞/重叠"))
            prev_end = ro + cur
            count += cur
            ro += K_VECTOR_TILE_ROWS
        total += 1
        if count != rows or prev_end != rows:
            bad.append((rows, None, "count=%d prev_end=%d != %d"
                        % (count, prev_end, rows)))
    return total, bad


# -------------------------------------------------------------- [3] 判据表 ---
def gate(m, n, usable=24, single_core_m=None):
    """复刻 C++ 侧 QmqUseSplitPath（perGroup 由 m/singleCoreM 推出）。"""
    aiv_candidate = min(m, 2 * usable)
    balanced = ceil_div(m, aiv_candidate)
    tile_m = min(m, max(16, balanced))
    scm = tile_m if single_core_m is None else single_core_m
    active_aiv = ceil_div(m, scm)
    mblocks_pg = max(1, min(2, ceil_div(active_aiv, usable)))
    active_group = ceil_div(active_aiv, mblocks_pg)
    if mblocks_pg != 1:
        return False, mblocks_pg, scm
    split_rows = ceil_div(m, 2 * usable)
    fused_rows = ceil_div(scm, 2)
    if split_rows >= fused_rows:
        return False, mblocks_pg, scm
    saved = (fused_rows - split_rows) * n
    return saved >= K_SPLIT_MIN_WORK, mblocks_pg, scm


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 74)
    print("v22 新增代码穷举证明")
    print("=" * 74)

    print("\n[1] qmq_vector_only 行划分穷举（不漏不重）")
    aiv_counts = list(range(2, 98, 2))              # 2U, U = 1..48
    m_list = list(range(1, 4001))
    # 大 m 的边界点：每个 A 的 rp 整数倍处（区间截断最容易出错的地方）
    for A in aiv_counts:
        for K in (1000, 4096, 16384):
            for d in (-1, 0, 1):
                v = K * A + d
                if v >= 1:
                    m_list.append(v)
    m_list = sorted(set(m_list))
    total, bad = check_row_partition(m_list, aiv_counts)
    print("    用例数 : %d  (m 1..4000 全域 + 各 A 的 rp 边界点，A=2,4,...,96)"
          % total)
    if bad:
        print("    [FAIL] %d 例不合格，前 5 例：" % len(bad))
        for b in bad[:5]:
            print("           %s" % (b,))
    else:
        print("    [PASS] 全部通过：并集恰为 [0, m)，无重叠、无空洞")

    print("\n[2] 内层分批穷举（kVectorTileRows = %d）" % K_VECTOR_TILE_ROWS)
    rows_list = list(range(1, 20001))
    for K in (1, 2, 3, 7, 63, 64, 65, 1024):
        base = K * K_VECTOR_TILE_ROWS
        for d in (-1, 0, 1):
            if base + d >= 1:
                rows_list.append(base + d)
    rows_list = sorted(set(rows_list))
    total2, bad2 = check_inner_batch(rows_list)
    print("    用例数 : %d  (rows 1..20000 + kVectorTileRows 的整数倍边界)" % total2)
    if bad2:
        print("    [FAIL] %d 例不合格，前 5 例：" % len(bad2))
        for b in bad2[:5]:
            print("           %s" % (b,))
    else:
        print("    [PASS] 全部通过：分批并集恰为 [0, rows)")

    print("\n[3] QmqUseSplitPath 决策表（U = 24, kQmqSplitMinWork = %d）"
          % K_SPLIT_MIN_WORK)
    ns = [256, 512, 1024, 2048, 4096, 8192, 16384, 65536]
    print("    %-10s %-8s %-8s %-9s %s" % ("m", "scm", "pg", "splitRows", "走拆分路径? (n=)"))
    print("    " + "-" * 66)
    marks = {}
    for m in list(range(1, 40)) + [48, 64, 96, 128, 160, 192, 224, 240, 256,
                                   288, 320, 336, 337, 352, 353, 368, 369, 384,
                                   385, 400, 512, 768, 1024, 2048]:
        cells = []
        for n in ns:
            ok, pg, scm = gate(m, n)
            cells.append("Y" if ok else ".")
            marks.setdefault(pg, 0)
            if ok:
                marks[pg] += 1
        _, pg, scm = gate(m, ns[-1])
        print("    %-10d %-8d %-8d %-9d %s"
              % (m, scm, pg, ceil_div(m, 48), "".join(cells)))
    print("    n 列依次为: %s" % ", ".join(str(x) for x in ns))
    print("    说明: m=337~384 因 ceil(m/48)==ceil(scm/2)==8 被拒（零收益段）；")
    print("          m>=385 落在 perGroup=2 被拒（保留 v20 的 mBlock 间重叠）。")

    print("\n" + "=" * 74)
    if bad or bad2:
        print("结论: 未通过 —— 请勿提交")
        print("=" * 74)
        return 1
    print("结论: 行划分不漏不重、分批不漏不重 —— 允许提交")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
