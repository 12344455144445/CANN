# v49 提交手册 —— 极小形状：跨核握手整段免除

> sha256(16) = **`8786ef39afd84394`** ｜ 70,214 B ｜ 1,449 个 `\r\n` 元素（v45 1,428，净 +21）
> 基线：**v45**（`1052cfef14de53aa`，23.3537 / 3706.57 µs，平台 15/15）—— 当前最优
> 生成器：`ref/make_v49.py`（A1~A12 断言全过，**A11 逐字节逆变换还原 v45**）

---

## §〇 · 替换指令（给工程包）

```
把  E:\CANN\records\experiments\v49-nohandshake\kernel_v49_nohandshake.asc
覆盖工程包里的  code/kernel.asc                ← sha256 前 16 位 8786ef39afd84394
```

回退（一键放回当前最高分）：

```bash
cp records/experiments/v45-allocfirst/kernel_v45_allocfirst.asc submission/kernel.asc
```

> ⚠️ 提交位始终由 `submission/kernel.asc` 决定；只有上平台跑的那一份才需要覆盖工程包。

---

## §一 · 一句话

**极小形状（`QmqTinyShape(m,n,k)` 为真）下，AIV 对 AIC 没有任何数据依赖 ⇒ 把那条跨核
`Wait` 与 AIC 侧配套的三处 `Set` 一起免掉**。这是把「点 1/2/3 的固定开销」串行链上
**唯一还没被标定过的一段**拿掉。

### 1.1 为什么打这里（三个候选已被实测否证）

| 候选 | 否证依据 | 结论 |
|---|---|---|
| **host 侧**（tiling 构造 / `GetTiling` / workspace 分配） | v21：`PATCH-H1`（workspace 缓存）+ `PATCH-H2`（tiling 缓存）实测只值 **0.41%**，噪声内（`analysis/38` §3.1 / `analysis/48` §3.2） | ✗ 永久关闭；**且 `main.asc` 每进程只调一次 `run_kernel`，任何 host 缓存都不会命中** |
| **AIC 入口**（`TPipe` + `matmul::Matmul` 构造 + `REGIST_MATMUL_OBJ`） | v40：整段拿掉、点 1 = `(4,8,16)` 必然命中门控，实测 11.32 → 11.85 µs（扣本底后收益 ≈ 0）＝ **E97** | ✗ 死路 |
| **AIV 的 `QmqComputeTinyC` 全长**（自带 TPipe + 3 TBuf + 2 `DataCopyPad` + 3 `PIPE_ALL` + 标量三重循环 + GM 写回） | v40 旁证：整个函数只值 **+0.30 µs** ＝ E97 后半 | ✗ 太小 |

⇒ **剩下唯一未标定的就是「AIV 被 AIC 的 flag 挡住」这一段串行等待**。

### 1.2 为什么这段等待确实存在（时序论证）

两条路径的**长度不对称**：

| 谁 | 到达同步点前要做什么 |
|---|---|
| **AIV** | `InitSocState()` → `GetBlockIdx()` ×2 → `GetSubBlockIdx()` → 4 条标量运算 → **`CrossCoreWaitFlag`（阻塞）** |
| **AIC** | `InitSocState()` → `SetSysWorkspace()` → `GetBlockIdx()` → 函数入口 → 5 项入参校验 → `QmqTinyShape` 判定 → `CrossCoreSetFlag<2, PIPE_FIX>` → **flag 传播到 AIV** |

AIV 的 15 次 `InitBuffer` 全部发生在 **Wait 之后**（在 `quant_matmul_relu_quant_vector` 内）
⇒ AIV 几乎立刻到达 Wait，而 AIC 还在走它自己的入口 ⇒ **AIV 真的会在 Wait 上阻塞**。

★ 反证：若这段等待为 0，则 v40（把 AIC 入口整段拿掉）在点 1 上就该有明显收益 —— 实测为 0。
**两者只能同时成立于「AIC 入口本身便宜、但 AIV 仍在等它跑完入口」这一种情形**，
而本版正是把这唯一的串行等待删掉。

### 1.3 价值（`analysis/39` §4）

点 1 = `(M=4, N=8, K=16)` ⇒ 整个矩阵乘只有 **1 条 `mmad`**（16×16×16 分形全包住），
耗时几十 ns 量级；点 2 的 `T` 只有 **1.77 µs**。而我们在 v20/v35/v37/v38/v39 五版里的
读数是 **10.28 / 10.29 / 10.43 / 10.64 / 10.68 µs** —— 五版机器码各异、读数不变 ⇒ 固定开销。

| 若点 1/2/3 各减 | 三点 ΔΣ分 | 总分 |
|---|---|---|
| 0（打平） | 0 | 23.3492 |
| 1 µs | ≈ +1.8 ~ +3.0 | ≈ 23.5 |
| **2.5 µs** | **≈ +4.6** | ≈ 23.7 |
| **5 µs** | **≈ +24.6** | **≈ 25.0** ← 兑现上限 |

---

## §二 · 唯一改动（4 处，全部在**行体之外**）

| # | 位置（v45 行号） | 改动 |
|---|---|---|
| **H1** | AIC 610~612（入参非法分支） | `QmqCubeSignalAllFlags(...)` 包进 `if (!QmqTinyShape(m, n, k))` |
| **H1** | AIC 616~624（v40 极小形状旁路） | 删掉 `QmqCubeSignalAllFlags(...)`（只留 `return;`） |
| **H1** | AIC 630~632（本 group 无 mBlock） | 同上包 `if (!QmqTinyShape(m, n, k))` |
| **H2** | AIV 1095~1100（`localTile` 循环头） | `CrossCoreWaitFlag(8/9)` 包进 `if (!tinyShape)` |
| **H3** | AIV +1094 附近 | `const bool tinyShape = QmqTinyShape(m, n, k);`（纯 CSE） |

**AIC 的置位本体（`QmqCubeSignalTile` 里的两条 `CrossCoreSetFlag`）一行未动**；
**AIV 的取 C、Pass-1、Pass-2、行体、掩码扫描区、yScale 合并段全部一行未动**。

---

## §三 · 等价性论证

### 3.1 AIV 对 AIC 没有数据依赖（逐位）

* 极小形状下 AIC 在 `matmul::Matmul` 之前就 `return`（v40 已有）⇒ **C 由本 AIV 自己算**
  （`QmqComputeTinyC` 逐行点积写 `workspace`）；
* AIV 的 Pass-1 / Pass-2 只读本 AIV 刚写进 `workspace` 的**行区间**
  （`localStart + rowOffset` 与 `QmqComputeTinyC(rowStart=localStart, rows=localRows)` 同一区间）；
* `y` / `yScale` 也由本 AIV 写出；
* `x1Scale` / `x2Scale` 是**算子入参**（GM），与 AIC 无关。

⇒ 逐行核对后 **AIV 侧没有任何一个读操作的数据源来自 AIC** ⇒ 那条 Wait 是空等。

### 3.2 Set / Wait 配平零变化（E22 的硬约束）

两侧门控用的是**同一个纯函数** `QmqTinyShape(m, n, k)`，而 `m` / `n` / `k` 由 host 以
**同一组值**分别传给 `quant_matmul_relu_quant_cube` 与 `quant_matmul_relu_quant_fused`
的形参 ⇒ **两侧判定恒等**。

| 情形 | AIC 置位数 | AIV 等待数 | 配平 |
|---|---|---|---|
| `QmqTinyShape` 为真（任一路径 return） | 0 | 0 | ✅ 零变化 |
| `QmqTinyShape` 为真，正常走完 | 0（本版删掉那处） | 0 | ✅ 零变化 |
| `QmqTinyShape` 为假，入参非法 | `mBlocksPerGroup` | 同 | ✅ 与 v45 逐字相同 |
| `QmqTinyShape` 为假，本 group 无 mBlock | `mBlocksPerGroup` | 同 | ✅ |
| `QmqTinyShape` 为假，循环内提前 break | 补 `rest..mBlocksPerGroup` | 同 | ✅ |

★ 关键：**不存在「一边置位、另一边不等待」的组合** ⇒ 不产生悬空 flag
（v17b 注释里担心的正是悬空 flag 污染同一 core 的后续 launch）。
唯一新引入的组合是「两边都不收发」，它对 flag 计数器是恒等变换。

### 3.3 红线与危险区

* `precision_guard.py`：三函数指纹全 PASS（`a5b35da0316e52b6` / `98a7bb5425b2e7a9` /
  `d9e81fce5089189f`），危险模式零新增；
* `livecode_diff.py` 的精度红线口径：**A 164 行 / B 164 行 逐行一致**；
* `precision_guard` 的数值路径段：**`quant_matmul_relu_quant_vector` 未改动**
  （即**整个 AIV 行体函数逐字节相同**）；仅 `quant_matmul_relu_quant_cube` 已改动
  （1985 → 1998 字符，全在早期返回分支里），五项标记全 OK；
* 生成器 **A10b** 断言：`fp32Local = fp32Buffer.Get<float>()` 到
  `x2ScaleQueue.FreeTensor(x2ScaleLocal)` 之间（Pass-2 行体 + 掩码扫描区）逐字节一致；
* **E112 安全**：`PipeBarrier<PIPE_V>()` 47 → 47、`Compares(` 1 → 1、`ReduceMin(` 1 → 1、
  `if (currentN >= 2048U)` 1 → 1 —— 扫描区一个字符没动。

---

## §四 · ★ 预登记判读表（**测前写死**）

> 尺子：`v45 = 3706.57 µs`（同一次对比内使用；E104/E114 —— 只信**同一窗口内的相对变化**）

| 读数 | 判读 | 下一步 |
|---|---|---|
| **点 1/2/3 各减 ≥ 1.0 µs**，且其余 12 点 \|Δ%\| < 0.5%，总耗时 ≤ 3697 µs | ✅ **兑现** —— 握手即串行链第一段 | 照实测 µs 反推 O 的握手成分；**继续打 AIV 侧入场**（C 的 GM 往返 → UB 直通，`analysis/48` §5.4 note 2） |
| **点 1/2/3 \|Δ%\| < 0.5%**（打平） | ⚠️ 握手被 AIV 自身入场覆盖 ⇒ 该方向关闭 | **转主攻 E100：行间双缓冲**（`kVectorTileN` 6144→4096 + 四缓冲各加一份，理论 +1.5~2.5 分） |
| 点 1/2/3 **任一变慢 ≥ 1%** | ❌ Wait 曾起节流/重叠作用 | **立刻回退 v45** |
| 其余 12 点任一点 \|Δ%\| ≥ 0.5% | ❌ 门控没生效 / 影响了非 tiny 路径 | **回退 v45**，查 `QmqTinyShape` 的 host/device 取值一致性 |
| 错误占比 > 0 | ❌ | **立刻回退 v45** |
| 总耗时 ≥ 3762 µs | ❌ | **立刻回退 v45** |

★ 必须**同时看逐点**：总耗时可能被大点盖住，而分数由小点决定（`dS/dt ∝ S²/t`，E109）。

---

## §五 · 兑现表（预期）

| 项 | 预期 |
|---|---|
| 总耗时 | −3 ~ −9 µs（3697 ~ 3703 µs，−0.1% ~ −0.25%） |
| 总分 | **+0.1 ~ +0.6**（若握手 ≥ 2.5 µs 则 ≥ +1.5） |
| 15/15 + 错误占比 0.00% | 必达（纯同步删减，无算术改动） |

**诚实说明**：本版的**主要价值是「标定」而非「得分」** —— 它给出一条此前从未测过的
O 成分。若读数落在「打平」，我们就能**定量地把 O 从嫌疑人名单里划掉**，从而把资源
全部压到 E100（行间双缓冲）这条唯一的大头方向上。

---

## §六 · 风险与砍法

| 风险 | 概率 | 砍法 |
|---|---|---|
| 编译不过（本机无 CANN，E19） | 低 —— 只新增 1 条 `const bool` 与 3 个静态 `if`，零新 API | 若返回「用时 - ＋ 全点败」= 启动失败 ⇒ 回退 v45 |
| 非 tiny 路径被扰动 | 极低 —— 该路径的机器码语义零变化；`precision_guard` 证整个 vector 函数未改 | 回退 v45 |
| 悬空 flag | 极低 —— 见 §3.2 的配平表，不存在单边收发 | 回退 v45 |
| 收益落进噪声带（±0.16 分） | **中** | 这本身就是 §四「打平」支 ⇒ 有信息量，不算失败 |

---

## §七 · 与各版对照

| 版本 | sha16 | 总分（今日 T） | 总耗时 | 说明 |
|---|---|---|---|---|
| **v49（本版）** | `8786ef39afd84394` | 待测 | 待测 | 极小形状免除跨核握手 |
| v48 | `f9886a774e01414c` | 22.8381 | 3775.95 | ❌ 回归 −0.5111（行体内部改动，弃用） |
| **v45** | `1052cfef14de53aa` | **23.3492** | **3706.57** | ★★ 当前最高分 = 基线 |
| v44 | `ac19fa2fd39edc76` | — | 3690.41 | A 无罪（但 A 掉分） |
| v40 | `57a7ac3ee4803e79` | 23.3231 | 3717.95 | 判读基准锚 |
