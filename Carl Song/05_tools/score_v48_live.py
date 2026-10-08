# -*- coding: utf-8 -*-
"""v48（行体链减负：去一层 UB→UB 搬运 + int32 中转改址）读数裁决器。

★ T 口径（E114 提醒）：本文件的 T 是**今日（v47 实测当日）**的平台最优用时。
  历史版本的绝对分数不可比 —— 只有同一窗口内的相对变化可信（E104）。

用法
----
    # 直接改下面的 V48 数组，或命令行传入 15 个逐点用时：
    python ref/score_v48_live.py
    python ref/score_v48_live.py --v48 "11.10 10.35 11.55 21.10 20.80 37.30 48.40 109.90 60.00 123.00 235.00 287.00 383.00 756.00 1590.00"

预登记判读表（records/experiments/v48-rowbody/SUBMIT.md §四）
------------------------------------------------------------
  总耗时 <= 3680 µs（−0.7% 以上）  ⇒ 兑现：UB->UB 搬运在关键路径 ⇒ 行体吞吐/依赖受限
                                       下一步：继续删链上冗余指令（合并尾部三级 Cast 中转）
  3680 ~ 3720 µs（打平 |Δ|<0.4%）  ⇒ 该搬运被 MTE 与 V 流水重叠 ⇒ 行体【延迟受限】
                                       ★ 下一步：行间双缓冲（kVectorTileN 6144->4096
                                         + fp32/half/reduce/mask 各加一份），目标 +1.5~2.5 分
  >= 3760 µs（+1.5% 以上）         ⇒ 有害（编译器调度悬崖，E112 型）⇒ 立刻回退 v45
  错误占比 > 0                     ⇒ 立即回退 v45
"""
import argparse
from math import log

# ---------------------------------------------------------------- T（今日口径）
# ★ E114：T 是移动靶。点 10 已由 36.46 下移到 36.30（v48 实测当日的平台最优用时）。
T = [2.08, 1.77, 2.50, 8.46, 8.28, 12.46, 15.46, 15.83, 19.27, 36.30,
     53.59, 74.35, 84.11, 174.06, 348.97]

# ---------------------------------------------------------------- 逐点读数（µs）
V40 = [11.04, 10.72, 11.85, 20.83, 20.72, 37.62, 48.92, 109.87, 60.35,
       122.82, 237.62, 285.91, 382.82, 766.86, 1590.00]
V45 = [11.15, 10.37, 11.55, 21.17, 20.86, 37.38, 48.53, 110.05, 60.11,
       123.04, 235.30, 287.24, 383.58, 756.24, 1590.00]
V47 = [11.21, 11.58, 12.13, 21.86, 21.22, 39.03, 49.23, 112.02, 61.86,
       122.92, 234.79, 282.89, 381.91, 742.73, 1570.00]

V48 = [11.46, 10.43, 12.03, 22.50, 22.88, 39.40, 50.37, 112.79, 62.54,
       128.39, 236.14, 285.15, 393.13, 768.74, 1620.00]   # 15/15 Pass，错误占比全 0.00%

GRP_HI = [8, 11, 12, 13, 14, 15]   # N <  2048
GRP_LO = [4, 5, 6, 7, 9, 10]       # N >= 2048
GRP_TINY = [1, 2, 3]

BASE_TOTAL = sum(V45)              # 3706.57


def score(ts):
    return sum(100.0 if t <= tt else 100.0 / (1.0 + log(t / tt, 1.5))
               for t, tt in zip(ts, T)) / len(T)


def point_scores(ts):
    return [100.0 if t <= tt else 100.0 / (1.0 + log(t / tt, 1.5))
            for t, tt in zip(ts, T)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v48", default=None, help="15 个逐点用时，空格分隔")
    args = ap.parse_args()
    global V48
    if args.v48:
        V48 = [float(x) for x in args.v48.replace(",", " ").split()]
        assert len(V48) == 15, "需要 15 个数值，收到 %d" % len(V48)

    print("=" * 96)
    print("  [0] 基线对照（今日 T 口径；T[2]=2.50、T[9]=36.46 已随平台最优下移）")
    print("=" * 96)
    for tag, ts in (("v45 = 基线（当前最高分）", V45), ("v40 = 判读基准锚", V40),
                    ("v47 = 最低总耗时但掉分", V47)):
        print("      %-26s 总分 %8.4f   总耗时 %9.2f µs" % (tag, score(ts), sum(ts)))
    print()

    if V48 is None:
        print("  ★ V48 未填 —— 把实测的 15 个逐点用时写入 V48 或 --v48 再跑。")
        print()
        print("  参考：判读阈（相对 v45 的 %.2f µs）" % BASE_TOTAL)
        print("      兑现    <= %.2f µs" % (BASE_TOTAL * 0.993))
        print("      打平    %.2f ~ %.2f µs" % (BASE_TOTAL * 0.996, BASE_TOTAL * 1.004))
        print("      回退    >= %.2f µs" % (BASE_TOTAL * 1.015))
        return

    s45, s48 = score(V45), score(V48)
    t45, t48 = sum(V45), sum(V48)
    print("=" * 96)
    print("  [1] ★ 预登记判读")
    print("=" * 96)
    print("      v48 总耗时 %.2f µs（相对 v45 %+.2f µs / %+.3f%%）"
          % (t48, t48 - t45, t48 / t45 - 1))
    print("      v48 总分   %.4f（相对 v45 %+.4f；噪声带 ±0.16）" % (s48, s48 - s45))
    print()
    if t48 >= BASE_TOTAL * 1.015:
        print("      ⇒ 【第二支：有害】总耗时涨到 +1.5% 以上")
        print("      ⇒ 立刻回退 v45（cp v45 -> submission/kernel.asc）")
    elif t48 > BASE_TOTAL * 1.004:
        print("      ⇒ 【中间区】略慢，但未到回退线 ⇒ 若总分也降，回退 v45")
    elif t48 >= BASE_TOTAL * 0.996:
        print("      ⇒ 【打平（落噪声带）】★ 结论：那条 UB->UB 搬运被 MTE 与向量流水重叠掉了")
        print("      ⇒ 行体是【延迟受限】、有重叠空间")
        print("      ⇒ ★ 下一步（结构级）：行间双缓冲")
        print("         · kVectorTileN 6144 -> 4096，UB 占用 170784 -> 103072 B（余 ~93 KB）")
        print("         · fp32 / half / reduce / correctionMask 四个缓冲各加一份（+49664 B）")
        print("         · 让相邻两行的独立行体链在向量流水里重叠，理论 −30~50% 行体时间")
    else:
        print("      ⇒ 【第一支：兑现】总耗时 <= −0.4%")
        print("      ⇒ 该搬运确实在关键路径上 ⇒ 行体是【吞吐/依赖受限】")
        print("      ⇒ 下一步：继续删链上冗余指令（合并尾部三级 Cast 中转 float->int32->half->int8）")
    print()

    print("=" * 96)
    print("  [2] 逐点对照（t/T 升序 = 边际价值升序；小 t 点每 1% 值钱得多）")
    print("=" * 96)
    print("      点   T      v45      v48     Δ%       v45分   v48分    Δ分   组")
    pt45, pt48 = point_scores(V45), point_scores(V48)
    for i in sorted(range(15), key=lambda k: V45[k] / T[k]):
        grp = "tiny" if i + 1 in GRP_TINY else ("HI" if i + 1 in GRP_HI else "LO")
        print("     %4d %6.2f %8.2f %8.2f %+7.2f%%  %7.2f %7.2f %+6.2f  %s"
              % (i + 1, T[i], V45[i], V48[i], (V48[i] / V45[i] - 1) * 100,
                 pt45[i], pt48[i], pt48[i] - pt45[i], grp))
    print()
    d = [pt48[i] - pt45[i] for i in range(15)]
    worst = sorted(range(15), key=lambda k: d[k])[:3]
    best = sorted(range(15), key=lambda k: -d[k])[:3]
    print("      最受伤的点：%s" % "、".join("点%d %+.2f" % (k + 1, d[k]) for k in worst))
    print("      最受益的点：%s" % "、".join("点%d %+.2f" % (k + 1, d[k]) for k in best))
    print("      ★ 若总耗时降而【小点】变慢，总分仍可能下降（v47 教训，E109）")
    print()
    print("=" * 96)
    if s48 > s45:
        print("  结论：v48 总分 %.4f > v45 %.4f ⇒ 留 v48 在提交位" % (s48, s45))
    else:
        print("  结论：v48 总分 %.4f < v45 %.4f ⇒ 提交位保持 v45" % (s48, s45))
    print("=" * 96)


if __name__ == "__main__":
    main()
