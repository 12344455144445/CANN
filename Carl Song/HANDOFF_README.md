# CANN-handoff —— 交接包总说明

> **项目**：CANN（华为昇腾）算子竞赛 · 题目 `QuantMatmulReluQuant`（INT8 matmul + ReLU + 动态逐 token 量化）
> **平台**：CANNJudge（`cannjudge.cn`）· 芯片 **Atlas A3（910C）** · arch `dav-2201`
> **当前成绩**：**23.3492 分 / 3706.57 µs**（v45）· 目标 **25 分**
> **本包生成时间**：随 v49 立项一起交付

---

## §1 ★ 三句话交接

1. **提交位 = `02_code/submission_current.asc`（= v45，sha `1052cfef14de53aa`）= 当前最高分 23.3492 = 唯一可行基线。**
2. **`02_code/candidate_v49.asc`（sha `8786ef39afd84394`）= 已生成、九道自检全 PASS、只差平台实测。**
3. **别碰**：行体内部、host 侧、Pass-2 扫描区 —— 这三处已有 **六次以上**的实测否证（详见 `01_rules/GAME_RULES.md` §5）。

---

## §2 阅读顺序（强烈建议）

| 顺序 | 文件 | 为什么 |
|---|---|---|
| 1 | **`01_rules/GAME_RULES.md`** | 题目规格 / 评分公式 / **不要做什么清单** / 硬约束速查。**读完能避免 90% 的坑。** |
| 2 | **`04_next/NEXT_STEPS.md`** | 现在在哪、下一步干什么、怎么判读。 |
| 3 | **`02_code/v49_SUBMIT.md`** | 七节提交手册 —— 也是未来每一版的**模板**。 |
| 4 | `03_issues/MEMORY.md` | 索引层，一页看懂当前状态 + 已关闭路线。 |
| 5 | `03_issues/LEDGER.md` | **85 条编号纪律（E15~E116）全文**（`grep -n E116` 等可展开）。|
| 6 | `03_issues/analysis/*.md` | 需要深挖某个结论时再读（39/48/49/56/57 最关键）。|

---

## §3 包内容地图

```
CANN-handoff/
├── HANDOFF_README.md          ← 本文件（总入口）
│
├── 01_rules/                  【规则】游戏规则与硬契约
│   └── GAME_RULES.md          · 题目规格、评分公式、平台契约、精度红线、⛔禁做清单、常量速查
│
├── 02_code/                   【代码】可直接替换的 kernel 源码 + 生成器
│   ├── submission_current.asc        = v45（提交位，sha 1052cfef14de53aa）★
│   ├── baseline_v45.asc              = 同上（语义名，作基线）
│   ├── candidate_v49.asc             = v49（待测件，sha 8786ef39afd84394）★
│   ├── v49_vs_v45.diff               = 5 hunk / 136 行统一 diff
│   ├── v49_SUBMIT.md                 = 七节提交手册（含★预登记判读表）
│   ├── make_v49.py                   = ★ 生成器模板（A1~A12 全套断言）
│   ├── make_v48.py                   = 上一版生成器（参考）
│   ├── calib/
│   │   ├── kernel_v20_mdl.asc        = 历史基线（噪声带标定来源）
│   │   └── kernel_v40_tiny.asc       = 最保守回退
│   └── pkg_snapshot/                 = 工程包快照（评测脚手架）
│       ├── main.asc                  · 第 32 行 #include "kernel.asc"；★ main() 每进程只调一次 run_kernel
│       ├── run.sh                    · 第 2 步 rm -rf build（全新构建）
│       ├── CMakeLists.txt            · CXX_STANDARD 14
│       └── data_utils.h
│
├── 03_issues/                 【问题】发现的问题与改进措施
│   ├── LEDGER.md                     = 85 条编号纪律 E15~E116（全文）
│   ├── MEMORY.md                     = 索引层（当前状态 / 已关闭路线 / 工具 / 契约）
│   ├── 2026-10-07.md / 2026-10-08.md = 工作日志（含 R78：v48 结算 + E115/E116 + v49 立项）
│   └── analysis/                     = 10 份关键分析
│       ├── 38_v20有效_固定开销杠杆_与路线重排.md
│       ├── 39_题目规格与合规复核.md            ★ §4.2 点 1/2/3 杠杆表
│       ├── 43_B路线纸上设计_只切cube与全局栅栏.md
│       ├── 46_v33判读C_口径纠正与v34只切AIV.md
│       ├── 47_v34否证与tileM旋钮.md
│       ├── 48_v39结算_tileM家族终结_与v40去Matmul设计.md   ★ §4 O 成分表
│       ├── 49_v41否证与N摊销发现.md            ★ 逐点形状分区表
│       ├── 54_A无罪与B夹逼及v45纯重排诊断.md
│       ├── 55_顺序无罪与回归的绝对us账本及v46立项.md
│       ├── 56_C是唯一元凶_形状分区退场与v48扫描区方向.md
│       └── 57_v47定案与行体链读法及v48立项.md  ★ §七 行间双缓冲设计
│
├── 04_next/                   【改进措施】下一步
│   └── NEXT_STEPS.md          · v49 判读 → AIV 入场 → E100 行间双缓冲 → 纪律
│
└── 05_tools/                  【工具】12 个 Python 自检脚本（依赖闭包已补全）
    ├── selfcheck_all.py       ★ 九道一键
    ├── precision_guard.py     ★ 三红线指纹 + 数值路径（权威）
    ├── livecode_diff.py       ★ 判活代码归属（唯一可信口径）
    ├── score_v48_live.py      ★ 读数裁决器（今日 T + 边际价值 + 预登记判读）
    ├── submit_check.py / scope_check.py / cpp_preprocess_check.py / sync_check.py
    ├── sig_check.py / static_check.py / cover_check_v17c.py / cover_check_v22.py
```

---

## §4 ★ 如何把工具跑起来（**已冒烟测试验证**）

工具脚本里的路径是**相对工作目录**写的（`ref/`、`records/experiments/`、`submission/`）。要运行它们，需要把本包**还原成原工程的目录布局**：

```bash
# 1) 建一个工作目录
mkdir -p work/ref work/records/experiments work/submission

# 2) 工具 → ref/
cp CANN-handoff/05_tools/*.py                     work/ref/

# 3) 源码 → records/experiments/<版本目录名>/<原文件名>
cp CANN-handoff/02_code/baseline_v45.asc          work/records/experiments/v45-allocfirst/kernel_v45_allocfirst.asc
cp CANN-handoff/02_code/candidate_v49.asc         work/records/experiments/v49-nohandshake/kernel_v49_nohandshake.asc
cp CANN-handoff/02_code/calib/kernel_v20_mdl.asc  work/records/experiments/v20-mdl/kernel_v20_mdl.asc
cp CANN-handoff/02_code/calib/kernel_v40_tiny.asc work/records/experiments/v40-tiny/kernel_v40_tiny.asc

# 4) 提交位
cp CANN-handoff/02_code/submission_current.asc    work/submission/kernel.asc

# 5) 从 work/ 目录下运行
cd work
PY=<你的 python>
$PY ref/selfcheck_all.py --cand records/experiments/v49-nohandshake/kernel_v49_nohandshake.asc \
                         --base records/experiments/v45-allocfirst/kernel_v45_allocfirst.asc
```

> ★ **注意**：`selfcheck_all.py` 的 `DEFAULT_BASE` 指向 `records/experiments/v20-mdl/…`，**必须显式给 `--base`**，否则会拿 v20 当基线得出错误结论。
> ✅ 上述布局在打包前**已实测**：`selfcheck_all` **九道全通过**、`precision_guard` PASS、`score_v48_live` 正常输出。

### 工具用法速查

```bash
PY=...                                    # 换成你的 Python 路径

# ★ 九道一键（提交前必跑）
$PY ref/selfcheck_all.py --cand <候选>.asc --base <基线>.asc

# ★ 判活代码归属（唯一可信口径：哪些行真的被编译进内核）
$PY ref/livecode_diff.py --a <基线>.asc --b <候选>.asc

# ★ 三红线指纹 + 数值路径函数是否改动（权威）
$PY ref/precision_guard.py --baseline <基线>.asc --candidate <候选>.asc

# ★ 读数裁决（把平台 15 个读数喂进去）
$PY ref/score_v48_live.py --v48 "11.46 10.43 12.03 ..."
```

---

## §5 关键 sha256(16) 表

| 文件 | sha256(16) | 分数 / 耗时 | 说明 |
|---|---|---|---|
| `02_code/submission_current.asc` | **`1052cfef14de53aa`** | **23.3492 / 3706.57 µs** | ★★ **提交位 = 当前最高分 = 可行基线** |
| `02_code/baseline_v45.asc` | `1052cfef14de53aa` | 同上 | 语义名，作 diff/自检基线 |
| `02_code/candidate_v49.asc` | **`8786ef39afd84394`** | 待平台实测 | ★ **本步测件**（九道 PASS）|
| `02_code/calib/kernel_v40_tiny.asc` | `57a7ac3ee4803e79` | 23.3231 / 3717.95 | 最保守回退 + 判读基准锚 |
| `02_code/calib/kernel_v20_mdl.asc` | `cac63ad78c25c167` | 22.1103 | 历史基线（噪声带标定）|

**⛔ 永久弃用**：v41 / v42 / v43 / v46 / v47 / v48（全部含 A / B2 / C 之一，E115 六连败零胜）。

---

## §6 当前状态一页纸

- **得分 23.3492 / 25 分目标**，差 **1.65 分**。
- **要拿这 1.65 分只有两条路**（`01_rules/GAME_RULES.md` §2 杠杆表）：
  - **银弹 A**：把 tiny 3 点（点 1/2/3）从 ~10.4 µs 压到 4.0 µs ⇒ 27.27 分（+3.92）。**共值 +4.9 分，是全题最肥的肉。**
  - **银弹 B**：把高组 6 点（点 8/11~15）的「剥 `O` 比值」从 2~7 压到 2.5 ⇒ 26.61 分（+3.26）。
- **全局提速 11.2% 才刚到 25 分** ⇒ **不要指望微优化**。
- **单条指令级改动只值 +0.09 分**（< 噪声带 ±0.16）⇒ **收益必须结构级**。
- 下一步 = 把 v49 送上平台（`04_next/NEXT_STEPS.md` §1）。

---

## §7 环境与其他

- **本机没有任何 CANN 编译能力**（E19）⇒ **所有改动都无法本地编译验证**，只能靠**生成器断言 + 九道自检**守门，然后上平台实测。
- **Python**：原工程环境 `D:/WorkBuddy/.workbuddy/binaries/python/envs/default/Scripts/python.exe`（接手环境请自行替换）。
- **换行符**：`kernel.asc` 是 **CRLF**；`LEDGER.md` / `memory/*.md` 是 **LF**（改文档时注意）。
- **临时目录**：`handoff_codex/_smoke/`（冒烟测试残留）与 `handoff_codex/_deps.txt`（工具依赖闭包清单）**未打进本包**。
