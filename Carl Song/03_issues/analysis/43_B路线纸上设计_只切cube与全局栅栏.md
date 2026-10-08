# 43 · 路线 B（只切 cube）的纸上设计与论证

> 立项依据：`records/evals/eval_2026-10-06_v28.md`（v28 定标：噪声带 **±0.20**、`O ≈ 9 µs`、`rowMax` 障碍）
> 目标：`m ≤ 384` 的 9 个点里，把闲置核用起来；`m > 384` 的 6 个点**逐字节不动**（天然对照组，E26）
> 状态：**纸上论证完成（本轮）**，实现待下一步。

---

## §1 · v20 的同步骨架（读码，行号为 `records/experiments/v20-mdl/kernel_v20_mdl.asc`）

| 位置 | 内容 |
|---|---|
| 774 | `__global__ __mix__(1, 2) void quant_matmul_relu_quant_fused(...)`，**13 形参** |
| 794 | AIC：`groupIdx = GetBlockIdx()`；`groupIdx < groupCount` ⇒ 调 cube 段 |
| 301 / 305 | cube 段：`QmqCubeSignalTile(localTile)` ⇒ 在 `IterateAll` 之后 `CrossCoreSetFlag<2, PIPE_FIX>(flagId)`（`localTile==0` 用 **8**，否则用 **9**） |
| 805–807 | AIV：`aivIdx = GetBlockIdx()`；`groupIdx = aivIdx / 2`；`subIdx = GetSubBlockIdx()` |
| 819 / 821 | AIV：`CrossCoreWaitFlag<2, PIPE_FIX>(8 或 9)` |
| 814–815 | `blockM = tiling.singleCoreM`；`halfM = (blockM+1)/2` |
| 824–844 | `mBlockIndex = groupIdx * mBlocksPerGroup + localTile`；两个子核把**同一 mBlock 的行**按 `localTile/halfM` 切一半 |
| 561 / 566 / 568 | **`rowMax` 沿 N tile 逐 tile 累积**（⇒ 必须扫遍整行） |
| 1048 | host 守卫：`tiling.singleCoreN != n` ⇒ 直接返回（**v20 假设「每 group 覆盖整 N」**） |
| 1072 / 1097 | `activeGroupCount` 计算 与 `<<<activeGroupCount, 0, stream>>>` 启动 |

**⇒ 现状：`grid = B`（`B = ceil(m/16)` 个 mBlock），每个 group = 1 AIC + 2 AIV。**
`m ≤ 16` ⇒ `B = 1` ⇒ **只有 1 个 AIC + 2 个 AIV 干活，其余 22 AIC / 46 AIV 全闲**。

---

## §2 · ★★★★ 官方权威结论（本轮查证，决定设计走向）

来源：昇腾社区 CANN 文档（`atlas_ascendc_10_0011.html`、`SyncAll.md`）、`asc.gitcode.com` 核间同步能力概述、`pypto-lib cce-extern-kernel.md`。

### 2.1 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 的**可见范围**

> **模式 0**：AI Core 核间的同步控制。对于 AIC 场景，**同步所有的 AIC 核**……
> **模式 1**：AI Core 内部，**AIV 核之间**的同步。
> **模式 2**：AI Core 内部，**AIC 与 AIV 之间**的同步（*一个 AIC ↔ 两个 AIV*）。

> ★ 外部权威总结（pypto-lib）：**"CrossCoreSetFlag / CrossCoreWaitFlag are intra-group only
> (mode 0x2 = a cube and its two paired vectors). **They are not a global barrier.**"**

**⇒ 关键判定：v20 用的 `<2, PIPE_FIX>` 是模式 2 = 只对「同一个 group 内的 1 AIC + 2 AIV」有效，
无法让「mBlock m 的某个 group 的 AIV」等到「同 mBlock 其它 group 的 AIC」。**
**⇒ 「每个 mBlock 的 AIV 等 c 个 cube flag」这个最早的设计（`eval_..._v28.md` §4 的 (iii)）在 API 层不成立。**

### 2.2 其它硬约束（E22 的官方依据）

| 约束 | 原文 |
|---|---|
| flagId 范围 | **0 ~ 10**（v20 用 8/9，合法） |
| 计数器上限 | **同一 flagId 的计数器最多可以设置 15 次** |
| 禁止连设 | **不允许连续设置同一个 flagId**，以防计数器状态混乱 |
| 严格配对 | set / wait 的**模板参数与 flagId 必须完全一致**；否则视为不同 flagId |
| ⚠️ 与 Matmul 冲突 | **「不建议开发者同时使用该接口和 Matmul 高阶 API，否则会有 flagId 冲突的风险」** |

> ⚠️ **最后一条说明：v20 自己就在「犯错」** —— 它同时用了 Matmul 高阶 API 与手写 flag 8/9。
> 目前 15/15 正常，但这是一条**潜在风险**，记入观察项。

### 2.3 ★★★★★ `SyncAll` 可用，且**支持 Atlas A3**

> **硬同步**：`template <bool isAIVOnly = true, ...> __aicore__ inline void SyncAll()`
> **`isAIVOnly = false`**：**融合算子（含 Cube 与 Vector）的全核同步** ——
> 「先分别完成 Vector 核和 Cube 核的全核同步，再执行两者之间的同步」
> 产品支持：**Atlas A3 训练系列 / A3 推理系列 —— 软同步 √ 硬同步 √**（A3 两者都支持）
> ⚠️ 硬同步 `SyncAll` 使用 **FFTS flag ID 11~14**，**不要复用这几个 flag 做自己的 `CrossCoreSetFlag`**（我们用 8/9 ⇒ 不冲突 ✓）

**⇒ 一个被官方背书的「融合算子全局栅栏」存在，且一次调用、无需我们自己做 flag 记账（避开 E22）。**

---

## §3 · 设计 B′（修正版）：**cube 切 N + `SyncAll<false>` 栅栏 + vector 段逐字节不动**

```
grid = B · c        （c = n / tiling.singleCoreN，编码进**已有** tiling 字段，形参表不变）
group g  →  mBlock = g / c,  nSplit = g % c

① AIC(g)：算 mBlock 的**第 nSplit 个 N 窗口**的矩阵乘  → workspace
           （SetTail 的 N 用窗口宽度；x2 偏移 = nSplit·(n/c)）
② SyncAll<false>()      ← 全核栅栏：保证「mBlock m 的**全部** N 窗口」都已落到 workspace
③ AIV：按行分摊（把 mBlock 的行分给 2·c 个子核），**每个子核用完整 N** 走原来的 vector 函数
        ⇒ rowMax / yScale / y 的算术与顺序 **与 v20 逐字节相同**
```

### 3.1 正确性论证（三个不变量）

1. **覆盖性**：`y` 的每个 `(r, col)` 恰被一个子核写一次 —— 行区间是 `mBlock` 行集的一个划分，子核内覆盖完整 N。
2. **`rowMax` 正确**：栅栏保证子核读到的整行数据已完整（所有 N 窗口都已写完）。
   ⇒ `rowMax`、`yScale`、`y` 的**算术路径与 v20 完全同一**（同一个不会被改动的 vector 函数、同一 tile 顺序） ⇒ 位级等价。
3. **无悬空 flag**：`<2, PIPE_FIX>` 的 flag 8/9 在 `m ≤ 384` 路径上**全部不再使用**（改由栅栏负责 cube→vector），
   悬空计数器问题（E22）从根上消失。

### 3.2 ★★★ 成本分析：**这条路上我们没有失去什么**

上一轮担心「换成全局栅栏会丢掉 cube/vector 重叠」。**读码后这条担心不成立**：
**v17b 的重叠是「前一个 mBlock 的 vector 与后一个 mBlock 的 cube 重叠」**，
而 `mBlocksPerGroup == 1`（即 `m ≤ 384`，我们唯一要改的 9 个点）时 **每个 group 只有一个 mBlock**
⇒ **本来就不存在可重叠的两个 mBlock** ⇒ 换栅栏**损失≈0**。
（`m > 384` 的 6 个点保留 v20 原路径，一个字不动。）

### 3.3 ★★ 真正要付的代价：**新增一条运行期分支**

`c == 1` 时走 v20 原路径、`c > 1` 时走新路径 ⇒ 核函数里多一个运行期 `if`。
**这个代价有实测数**：v27 在 v20 核函数里多挂三处死分支（运行期从不进入），实测 **−1.16 分**（总耗时 +1.5%）。
⇒ 本设计的净收益估计必须扣掉这一项：

```
净收益 ≈ （§3.4 的增益） − （分支代价 0 ~ 1.16）
```

### 3.4 增益（把 `O ≈ 9 µs` 计入 —— `analysis/41` §3 的闭式漏了 `O`，见 `eval_..._v28.md` §5）

`增益(点) = t / ( O + (t − O)/c )`，`c = floor(24/B)`，`B ≥ 18 ⇒ c = 1`（自动回落 = 对照组）

| 点 | B | c | t → t′ | Δ分 |
|---|---|---|---|---|
| 1 | 1 | 24 | 16.41 → 9.31 | **+5.0** |
| 2 | 1 | 24 | 10.74 → 9.07 | +1.5 |
| 3 | ≤3 | 8 | 16.62 → 11.62 | **+3.4** |
| 4 | ≤4 | 6 | 22.37 → 15.52 | **+10.6** |
| 5 | ≤5 | 4 | 26.77 → 21.20 | **+4.5** |
| 7 | ≤7 | 3 | 64.45 → 53.48 | +2.5 |
| 6/8/9/10/12/13 | `m>384` | 1 | 不变 | 0（**对照组**） |
| 14/15 | 14~24 | 1 | 不变 | 0（**对照组**） |

**⇒ 毛收益 ≈ +1.83 分**（`O` 的估值是**上界**，实际更小 ⇒ 收益只会更大）
**⇒ 扣掉分支代价后：净 ≈ +0.7 ~ +1.8 分。仍在 ±0.20 噪声带的 3.5 倍以上 ⇒ 可判定（E47）。**

---

## §4 · 预登记判读表（提交前写死）

| 观察 | 结论 | 动作 |
|---|---|---|
| 15/15 且总分 ≥ **+1.0** | B′ 成立 | 转正；下一步继续加大 `c` 或上 A-2 |
| 15/15 且总分 ∈ **[−0.4, +1.0]** | 分支代价吃掉了收益 | 改做「统一路径」（所有点都走新写法，去掉分支），再测一次 |
| 15/15 且总分 ≤ **−0.4** | 分支代价 > 收益 | 撤回，**B 线封版**，只留 A-2 |
| 非 15/15，`用时 -` | 启动失败（栅栏用错/`SyncAll` 不存在） | 见 §5；回退 v28 |
| 非 15/15，有 `用时` 但错误占比 > 0 | 行划分/覆盖性错 | 回退 v28，用 `cover_check_*` 穷举重查行划分 |
| 对照组（6/8/9/10/12/13/14/15）**逐点变化 > 0** | **实现越界**（新代码影响了不该影响的点） | 立即回退，按 E37 重查改动边界 |

---

## §5 · 本次设计的**不可本地验证项**（E19，必须单独成版）

| 项 | 风险 | 若失败的现象 |
|---|---|---|
| `AscendC::SyncAll<false>()` 是否在此 CANN 版本存在且可编译 | 中 | Compile Error |
| `SyncAll<false>` 是否与 Matmul 高阶 API 的 FFTS flag（11~14）冲突 | 低（官方说 11~14 归它） | 启动失败 / 挂死（`507015`） |
| `SyncAll` 是否要求**全部**核都参与（`grid = B·c` 里的每一核） | 中 | 挂死 |
| 把 `c` 编码进 `tiling.singleCoreN` 后，host 守卫（行 1048）需同步放宽 | 低 | 直接 return（全 0 输出） |

**⇒ 因此本版必须：单变量（只做 B′，不捆绑 A-2）、单独成版、并保留 v28 一键回退。**

---

## §6 · 下一步动作清单（实现顺序）

1. **先做 host 侧**：算 `c`（`c = floor(24/B)`，且 `c | n` 或按窗口取整，`c ≤ floor(n/16)`），
   写入 `tiling.singleCoreN`，`activeGroupCount = B·c`，放宽行 1048 的守卫。
2. **再改 AIC 段**：把 `groupIdx` 拆成 `(mBlock, nSplit)`，`SetTail` 的 N 用窗口宽度，`x2Gm` 偏移加 `nSplit·(n/c)`。
3. **加栅栏**：AIC 段末尾 `SyncAll<false>()`；`c > 1` 时**不再置位** flag 8/9。
4. **再改 AIV 段**：行分摊分母由 2 改为 `2c`，其余**逐字节不动**（仍调同一个 vector 函数）。
5. **离线穷举**：扩写 `cover_check_v22.py`（★新行划分穷举）证明「15 点 × 每个 (r,col) 恰被写一次」。
6. **九道自检**：`ref/selfcheck_all.py --cand <候选>.asc`，并**跑一次 v28 基线做回归**（E31）。
7. CRLF 生成器按 E21/E44 写（`make_v29.py`，保留 v27 生成器的 `CALL_EXPECT` 逐字节断言）。

---

## §7 · ★★★★ 平台模板核对（用户提供官方脚手架原文后新增）

提交页 `https://cannjudge.cn/public/op_challenge_hangxia_prelim/quantmatmulreluquant/submit` **需登录**，WebFetch 返回「访问受限」。
但**模板原文本身给出了比网页更硬的契约**。

### 7.1 四条核对结论

| # | 模板给出的 | 我们 v20 的现状 | 结论 |
|---|---|---|---|
| **F1** | `TensorInfo { const int64_t* shape; int64_t numDims; int32_t dtype; }`；`TensorGroupInfo { const TensorInfo* tensors; int64_t numTensors; }`；用法 `info_x.tensors[0].shape[0]` | 第 **957~994** 行**全部读到并校验**：`numTensors==1`、`numDims`、`dtype`、`shape` 交叉一致（`x2.shape[1]==k`、`y.shape==(m,n)` …） | ✅ **完全合规，且更严**。**★ 由此：`m,n,k` 在 host 侧是精确已知的** |
| **F2** | dtype 表：`0=fp32 … 3=int8 …` | `kFloat32Dtype=0`、`kInt8Dtype=3`（第 952~953 行） | ✅ 与平台文档一致 |
| **F3** | 文件被 `#include`，**禁 `main()` / `#pragma once` / include guard** | `grep`：`#pragma once` **0**、`int main` **0**、`#ifndef` **0** | ✅ 三条禁令全部满足 |
| **F4** | 建议核函数形如 **`__global__ __cube__ void quant_matmul_relu_quant_custom(...)`**，可用 API 列的是 **`TPipe / TQue / DataCopy / Add`（低阶原语）**，**未出现 `lib/matmul_intf.h`** | 我们用 `__global__ __mix__(1, 2)` + **高层 `matmul::Matmul`** | ⚠️ **外部旁证**：平台的参考形状是「**纯 cube + 低阶原语**」⇒ 与 E51（`O ≈ 9 µs` 来自高层 `Matmul` 的 `Init`）方向一致，**支持路线 A-2**。但模板只是脚手架提示，不是硬约束（我们 15/15 已证） |

> **F5 观察项**：`availableCoreNum` 是入参（`main.asc` 用 `aclrtGetDeviceInfo(..., ACL_DEV_ATTR_CUBE_CORE_NUM, ...)` 取）。
> 我们有一条 `activeGroupCount > usableCoreNum ⇒ return` 的守卫。若真实平台核数与我们推断的 U=24 不一致，该守卫行为会变（目前 15/15 ⇒ 未触发）。

### 7.2 ★★★★★ F1 的直接后果：**「形状问题」正式关闭**

`MEMORY` §九 一直挂着「仅剩途径 = `run_kernel` 里加 `printf` 探形状（耗一次提交）」。
**⇒ 这条作废：`m`、`n` 在 host 侧运行时就是精确值，不需要任何探测提交。**

对 v29 的直接好处：**闸门可以同时用 `m` 和 `n`**（原来只知道 `m`）：

```
B = ceil(m / tiling.singleCoreM)          // singleCoreM 已由 host 算出
c = 1
若 B < 18:                                 // B ≥ 18 ⇒ 自动回落（天然对照组）
    取最大 c，使  c ≤ floor(24 / B)  且  c 整除 (n / 16)     // ← 16 的倍数约束
```

### 7.3 ★★ `c` 的编码方式（读 run_kernel 后确定）

v20 的 tiling 建法（第 1011~1050 行）：`tilingApi.SetDim(1)` + `SetShape(tileM, n, k)`，
并且第 1048 行**断言 `tiling.singleCoreN == n`**（即 v20 假设「每个 group 覆盖整 N」）。

**⇒ 最优编码：把 `SetShape` 的第二参由 `n` 改为 `n / c`。**
这样 `tiling.singleCoreN = n / c`，device 侧一行算出 `c = n / tiling.singleCoreN`，
**形参表完全不动**（不需要新形参、不需要新类型宏 ⇒ 绕开 E34/E39 的全部坑）。

**★★ 并且「`(n/c) % 16 == 0`」这条约束正好保证：**
1. 每个 N 窗口宽度**恰好**是 `n/c` ⇒ **没有尾块** ⇒ 位级精确性由构造保证；
2. 窗口宽度是 16 的倍数 ⇒ 落在 cube 的合法 N tile 网格上（避免触碰「边界校正 guard 只在 `currentN ≥ 2048` 生效」那条红线）；
3. `c` 不满足时**自动取 1** ⇒ **最坏情况逐字节等于 v20，零风险**。

**⇒ 闸门是「参数无关」的（E30）：只由运行时已知的 `m`、`n`、`singleCoreM` 决定，无自由常数。**

---

## §8 · ★★ 实现修正（v29 落地后回填，2026-10-06）

本节记录**实现与上面纸上设计的四处偏离**，以及一处**旧结论的更正**。
候选文件：`records/experiments/v29-nsplit/kernel_v29_nsplit.asc`
（sha256 `a6c989ac807c10d8a2f28d9a04e2f02ac8512570ecfd10c27fa655f152f01145`，59,693 B，1,255 行）。

### 8.1 栅栏由 `SyncAll<false>()` 改为**模式 0 的 `CrossCoreSetFlag/WaitFlag`**

§3 原本用官方融合算子栅栏 `SyncAll<false>()`。**改为模式 0，两条理由**：

* **官方语义确认（本轮再查证）**：asc.gitcode.com「关键特性说明」原文 ——
  「**模式 0**：AI Core 核间的同步控制。……**对于 AIV 全核场景，同步所有的 AIV 核**，
  直到所有的 AIV 核都执行到 `CrossCoreSetFlag` 时，`CrossCoreWaitFlag` 后续的指令才会执行。」
  ⇒ 在 AIV 侧调用模式 0 **只统计 AIV**，不会因为 AIC 不参与而死锁。
  另查证 `modeId` 支持：**A2/A3 支持 0/1/2**（950 才多一个 4）⇒ A3 可用。
* **代价对比**：`SyncAll<false>()` 要求 **AIC 侧也必须进入栅栏** ⇒ AIC 分支多一条运行期
  分支；而 `m > 384` 的 6 个点会白付这笔开销（v27 实测：核函数里多挂运行期死分支
  ⇒ 小点显著变慢，−1.16 分）。模式 0 放 AIV 侧 ⇒ **AIC 段活代码逐字节不动**。

**⇒ 净效果：AIC 的 flag 协议（模式 2 / flag 8、9）完全不变，新增面收敛为「一个常量 + 一个形参 + 一处新用法」。**

### 8.2 `c` 用**新形参**传，不用 §7.3 建议的 `SetShape(tileM, n/c, k)`

§7.3 曾建议「把 c 编码进 `SetShape` 的第二参 ⇒ 形参表完全不动」。**实现时否决，三条理由**：

1. 改 `SetShape` 的 N 会连带改 `tiling.N` / `tiling.singleCoreN` 与 **matmul 内部
   L1/L0 buffer 尺寸**，属于「不可本地验证的深层契约」；加一个形参是**纯局部**改动。
2. 第 1048 行 `tiling.singleCoreN != n` 是 v20 的**显式 host 契约**（「每 group 覆盖整 N」）。
   走 §7.3 就得改这条守卫 ⇒ 动 host 契约。
3. 「加形参」的形状已被 **v27 证过无害**（v27 把融合核由 13 形参加到 15 形参，
   其 −1.16 分被归因于**运行期死分支**，不是形参表长度）。

⇒ 实际做法：`SetOrgShape(m, n, k, k)` **仍给全形状**（保证 C 的行步长仍是 `n`、
B 的行步长仍是 `k`），只把 `SetTail`/`SetTensorB`/`IterateAll` 的三个窗口参数改掉。

### 8.3 ⚠️ 更正：「`m > 384` 的 6 个点逐字节不变」**这句话是错的**

上一轮（`MEMORY` §五、`eval_2026-10-06_v28.md` §6）把 6 个 `mBlocksPerGroup == 2` 的点
写成「**逐字节不变**」。**实际只是语义不变**：

* `halfM = (blockM+1)/2` 被换成 `sliceCount = 2*c` + `stepM = ceil(blockM/sliceCount)`；
* `localStart` / `localEnd` 由三元式换成 `sliceIdx*stepM` 的统一式；
* AIV 头部多两次整除 + 一次取模；tile 循环里多一个运行期 `if (nSplitCount > 1U)`。

数值上完全等价（`c==1` 时 `stepM == halfM`、`sliceIdx == subIdx`，已由
`ref/cover_check_v29.py` 的 51,227 组 `c==1` 组合穷举确认），但**机器码不同**。
⇒ 这 6 个点是**语义对照组**，其逐点变化里既有共模漂移、也有少量代码体积代价，
**不能当严格的零基准**。新增纪律 **E52**。

### 8.4 ★ M 切分（下调 `tileM` 下界）**已死**，N 切分确是唯一手段

`analysis/41:144` 早已写下判词：

> 「降 `tileM`：**每 mBlock 仍烧一个 16 行分形** ⇒ 墙钟持平」

物理原因：cube 的 `baseM` 是 16，`tileM < 16` 时每块仍要烧满一个 16 行分形的时间
⇒ 把 mBlock 拆得更细只增加并行度、不减少单块耗时，而 `O` 已经吃掉大部分墙钟。
⇒ 本轮**复核并通过**这条：**沿 N 切 c 份是唯一能真正降低「单 group 成本」的手段。**

### 8.5 实际落地的 13 个锚定替换

| # | 标签 | 内容 |
|---|---|---|
| 1 | `host/c-grid` | `baseGroupCount` 改名 + 新增 `c` 闸门（含 32B 对齐条件）+ `activeGroupCount = base * c` |
| 2 | `host/launch-args` | launch 实参 +1 |
| 3 | `kernel/signature` | `__global__` 融合核 13→14 形参 |
| 4 | `aic/call` | 把 `nSplitCount` 传给 cube 段 |
| 5 | `cube/signature` | cube 段 10→11 形参 |
| 6 | `cube/mapping` | `mbGroupIndex / nSplitIndex / windowN`，`firstMBlock` 改基 |
| 7 | `cube/window` | `x2WindowOffset` / `cWindowOffset` / `SetTail(...windowN)` |
| 8 | `aiv/barrier` | 模式 0 全 AIV 会合（在全部 `continue` 之前） |
| 9 | `aiv/mapping` | AIV 侧 `mbGroupIndex / nSplitIndex` |
| 10 | `aiv/slice` | `sliceCount = 2*c`、`stepM` |
| 11 | `aiv/slice-range` | 统一切片区间式 |
| 12 | `const/flag` | `kQmqNsplitBarrierFlag = 7` |
| 13 | `header/v29` | 文件头新增 v29 说明（纯注释） |

**校验结果**（全绿）：

```
make_v29.py           13/13 替换成功、括号配对、CRLF 1255/1255、末尾带 CRLF
livecode_diff         16 个 diff 块，+52 / −15 行（净增 +37 行活代码）
                       与 13 个锚点一一对应，0 处意外改动
                       契约：精度红线 / tiling 调用 / workspace 获取 / 14 形参签名 —— 全部 0 改动
cover_check_v29.py    5 节全绿（202,306 组行划分穷举，其中 c>=2 共 151,079 组）
selfcheck_all.py      九道全通过（并用 v20 自身做了一次工具回归，E31）
合规用词              跳过/占位/投机/简化计算 全 0；等价性论证 9 处
```
