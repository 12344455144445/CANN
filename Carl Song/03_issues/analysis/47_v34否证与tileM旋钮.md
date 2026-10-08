# 47 — v34 否证与 `tileM` 旋钮（2026-10-07）

> 配套：`records/evals/eval_2026-10-07_v34.md`（实测全表 + 根因三连）、
> `records/experiments/v35-tilem8/SUBMIT.md`（v35 设计 + 预登记判读表 + 穷举安全性）。

## §1 v34 = AIV 只切行 ⇒ 9/15，机制否证

| 项 | 内容 |
|---|---|
| 结果 | `Pass 9/15` |
| 失败点 | **1(100.00%) · 3(56.11%) · 4(48.18%) · 5(48.45%) · 7(47.97%) · 11(71.84%)**，用时栏全 `-` |
| 通过点 | 2 / 6 / 8 / 9 / 10 / 12 / 13 / 14 / 15，`t ≈ v20`（点 8 +14.2% 属历史抖动） |
| 判读 | **失败 ⟺ 走切分路径**；闸门关闭的 `{2,14,15}` 全过 ⇒ **闸门按设计工作，机制本身不成立** |
| 信息价值 | ★ **对「AIV 占比」为 0** —— 失败点无耗时读数（E71）⇒ 预登记的耗时判据不可判读 |

## §2 根因（行号实证，不是推测）

### E76 — 闲置 cube group 会**立刻**放行自己的 AIV
`quant_matmul_relu_quant_cube` **行 352~356**：

```cpp
const uint32_t firstMOffset = firstMBlock * static_cast<uint32_t>(tiling.singleCoreM);
if (firstMOffset >= m) {
    QmqCubeSignalAllFlags(mBlocksPerGroup);   // 立刻置位 flag 8
    return;
}
```

- **v20 不变式**：`activeGroupCount = baseGroupCount = ceil(activeAivCount / mBlocksPerGroup)`，
  `activeAivCount = ceil(m / singleCoreM)` ⇒ **每个启动的 group 都有 1 个真实 mBlock**，
  上面那条分支在 v20 里**从不进入**。
- **v34 破坏它**：闸门把 `activeGroupCount = usableCoreNum = 24`，真实 mBlock 数只有 `ceil(m/16)`
  （点 1 时 = **1**）⇒ 凭空造出 23 个闲置组。
- **后果链**：闲置组 AIC 瞬间置位 flag 8 → 它的 2 个 AIV 立刻通过 `CrossCoreWaitFlag<2,PIPE_FIX>(8)`
  → 读别的 group 尚未算完的 workspace 行 → `rowMax` 偏小 → `yScale` 偏大 → **整行错**。

### E77 — mode 0 会合在 48 个 AIV 上不构成可用栅栏
`CrossCoreSetFlag/WaitFlag<0, PIPE_FIX>(7)` 想让 48 个 AIV 到齐，**实测无效**。
★**`flagId 7` 的每一次使用都与失败共现**：

| 版本 | 用了 flag 7 | 结果 | 没用 flag 7 | 结果 |
|---|---|---|---|---|
| v29 | 5 / 7 / 11 | 全 WA | 其余 12 点 | **全过** |
| v34 | 1 / 3 / 4 / 5 / 7 / 11 | 全 WA | 其余 9 点 | **全过** |

嫌疑：①`<0>` 计数器上限（官方「同一 flagId 最多设 **15** 次」）容不下 48 核；
②flag 7 与 Matmul 内部 flag 冲突（E54）。
⇒ **不用 `flagId 7`；不靠 mode 0 做融合算子的全局栅栏。**

### E78 — AIV 并行度被 `blockDim` 限死
在岗 AIV 数 `= 2·baseGroupCount ≤ 2·activeAivCount = 2·ceil(m/singleCoreM)`。
⇒ 抬 AIV 数**必须**抬 `activeGroupCount`；抬了就落进 E76。⇒ **AIV 只切行结构上死掉。**

## §3 为什么 v34 对「占比」一无所获

预登记表的两支都需要**候选点的耗时**变化；但**所有候选点都 WA ⇒ 没有耗时读数**（E71）。
⇒ 结论：**任何以「失败点的 `t`」为判据的预登记都是不可判读的** —— 判读表必须区分
「正确性签名」与「耗时签名」，且优先选**预期全点通过**的实验。

## §4 v35 的旋钮：host `tileM`（E79）

```
SetDim(1)  （v20 行 1012）  ⇒  singleCoreM == tileM
AIV 每核行数 = halfM = ceil(singleCoreM/2)      （AIV 段 v20 行 815）
⇒ 每核行数 ≈ tileM / 2
⇒ tileM 16 → 8  ⇒  每核行数 8 → 4  ⇒  AIV 墙钟减半
⇒ g = 1.3266 / (1 + 0.3266 × 0.5) = 1.1404×   （+14.04%，analysis/41 §3.2 闭式）
```

**★ 更正 `analysis/41:144`**：判词「降 `tileM`：每 mBlock 仍烧一个 16 行分形 ⇒ 墙钟持平」
**只覆盖 cube 侧**（对 cube 成立），**漏了 AIV 侧** —— v20 的 AIV 每核行数正比于 `singleCoreM`。
这就是 v19/v12 试 `tileM` 时看不到收益的原因：当时 AIV 只占 24.6%，而被改动的那部分只是它的 1/2。

**门控**：`ceil(m/8) ≤ usableCoreNum` ⇒ `m ≤ 192` 才下探（受影响集 `m ∈ [9,192]`）；`m > 192` 逐字节 = v20。

**为什么安全**（`ref/cover_check_v35.py`，两种 API 行为模型 × `m ∈ [1,4000]`，P1~P9 全通过）：

| 模型 | API 行为 | v35 结果 |
|---|---|---|
| **N** | 不夹紧 `singleCoreM = tileM` | `m ∈ [9,192]`：`singleCoreM = 8`、`active ≤ 24`、**每核行数 8→4** ⇒ +14% |
| **C** | 夹紧到 baseM=16 | `singleCoreM ≡ 16` ⇒ **逐行等于 v20 ⇒ null、无回归** |

⇒ 两种行为下都不可能触发 `return` / 越界 / 变慢。
★**本实验因此是一个干净的二元问题**：「`singleCoreM` 是否被夹紧到 baseM」。
A 支（有增益）⇒ 未夹紧 ⇒ AIV 行数路线活着；B 支（无增益）⇒ 夹紧 ⇒ 该路线关闭，转 A-2/P0 或 D 路线。

## §5 下一步候选（按头寸）
1. **D 路线**：`quant_matmul_relu_quant_vector` 的 Pass-1/Pass-2「逐行标量回读」向量化
   （v20 行 655~674 / 778~800；台账记「主嫌」；`ReduceMax`/`GetValue`/`PipeBarrier` 均已在 15/15 路径出现 ⇒ 不违 E19）。
2. **A-2 / P0**：去高层 `Matmul` 换低阶原语（`O ≈ 9 µs`，点 1/2 几乎全是它；头寸最大，E19 不可验证）。
3. **`analysis/46` §5**：`SetOrgShape` 的 `orgN` + workspace 窗口主序（头寸 3~6×，需 AIV 取数大改，先做 §5.6 单点验证）。
