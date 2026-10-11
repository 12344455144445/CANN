---
document_type: cann_optimization_reference
schema_version: 1
updated_beijing: 2026-10-11
language: zh-CN
target_operator: QuantMatmulReluQuant
source_issue: https://gitcode.com/cann/cann-ops-competitions/issues/37
source_attachment: https://gitcode.com/user-attachments/files/9516645/981cc9a3399f4b5fab645394b6363ebd.docx
learning_hub: https://gitcode.com/cann/cann-learning-hub
purpose: 后续代码优化前读取的学习要点与执行检查
---

# CANN 学习指南与下次优化备忘录

后续代码优化优先读取 [已读源码的可执行优化方案](CANN_算子优化方案_可执行版.md)。该文件是在用户进一步要求下新增的落地资料，包含实际读过的官方方案、与 v84 的逐项对照及完整候选；本文件保留基础学习摘要。

本文件以 Issue #37 所附《CANN挑战赛学习指南（零基础参赛版）》正文为主要依据，并纳入用户补充的学习仓及视频入口。第 1—3 节是指南摘要；第 4 节是评论中的补充提醒；第 5—7 节是便于后续优化执行的整理建议。学习资料不替代赛题规范、对应设备的 API 文档和实际评测结果。

## 0. 下次接手先读这里

1. 先读仓库 `Carl Memory/CANN_续跑记忆.md` 的最新状态，再读本文件；版本选择、提交次数和授权状态以最新记录及用户指令为准。
2. 本摘要本身不授权提交评测或申请云资源。配套优化候选已生成，并按用户要求上传至 GitHub；后续运行范围以用户最新指令为准。
3. 截至本次读取，最新续跑记录推荐 `baseline/kernel.asc`，最后提交是 v90；最高单次观测与稳定复现收益需要分别说明。未来开始优化时重新核对，不把旧版本或旧分数当实时状态。
4. 已记录的“停止提交”和“暂不申请云算力”状态继续保留，恢复相关操作以用户新指令为准。
5. `sources/` 与其他同步项目文件只读；候选代码、测试与产出写入独立工作目录。

**执行主线：固定题意与环境 → 功能基线 → 多形状及边界验证 → 性能基线 → 一个瓶颈、一个假设 → 正确性回归 → 同方法性能对比 → 保存可复现结果。**

## 1. 指南的关键原则

- 学习安排为 10 个学习日，建议分 3—4 周完成；每周 2—3 日，前九日每天约 2—4 小时，不要求连续十天。
- C/C++、Linux、Tensor 概念按需补齐。Notebook 可在线运行，也可下载源码本地练习。
- 教程仅在 Atlas A2 系列完成验证；比赛硬件、软件版本、赛题、评分和截止时间以比赛指定环境及组委会通知为准。
- 开始前收集报名组队、算子要求、评分、提交物、评测入口、日程和答疑信息。
- 每一步留下实际产出：代码、命令、环境、参考结果、测试结果、问题及复盘。看懂教程与完成上板运行分别记录。
- 云实验结束要停止实例；关闭浏览器通常不会释放算力。这里是使用提醒，不代表当前项目应创建实例。
- CANNBot 可用于讲解、排错和自测；AI 的解释需要结合代码及运行结果核实，发送日志前删除敏感内容。

## 2. 十天路线与优化时的查阅位置

教程根目录：[Ascend C Kernel 直调版](https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development_light)。章节编号取自下载的指南，目录更新后以仓库实际文件为准。

| 学习日 | 章节 / 主题 | 必须掌握或留下的产出 | 后续优化的用途 |
|---|---|---|---|
| Day 1 | 1.1—1.5：算子、CANN、NPU、AI Core | 概念关系与章节练习 | 区分软件栈、硬件和算子职责 |
| Day 2 | 2.1—2.2：Hello World、CANNBot | 硬件 / CANN / 工作目录、编译运行记录 | 优先区分环境、编译、运行问题 |
| Day 3 | 2.3—2.4：编程范式、Add | GlobalTensor / LocalTensor / 队列；CopyIn → Compute → CopyOut；参考对比 | 检查搬运、生命周期与同步 |
| Day 4 | 2.5—2.6：泛化 Tiling、PyTorch 直调 | 找到 Tiling 生成与读取位置；至少 3 组 shape | 证明分块支持多种输入，核对调用接口 |
| Day 5 | 2.7—2.8：核函数、Vector | 至少 3 组输入，含非整块长度；尾块说明 | 检查向量范围、尾块与非对齐输入 |
| Day 6 | 3.1—3.3、3.6：MatMul | M/N/K 和分块关系；至少两组尺寸与参考一致 | 分析矩阵乘分块、数据复用与 Cube 路径 |
| Day 7 | 3.4—3.5、3.7：CV 融合 | 融合前后计算流程、中间数据与正确性 | 分析矩阵乘及后处理融合 |
| Day 8 | 4.1—4.2：NPU 上板调试 | 调试方法、日志、复现与定位；无故障也主动验证 | 排查设备上的精度、越界、同步问题 |
| Day 9 | 4.3—4.4：Profiling、仿真 | 正确算子的性能基线、瓶颈判断、一次假设及前后数据 | 用证据确定优化对象 |
| Day 10 | GELU 实战 | 读题、实现、边界测试、有效提交与复盘 | 练习完整开发闭环，流程可迁移到当前算子 |

指南用 GELU 练习流程，不表示当前 QuantMatmulReluQuant 应改成 GELU。

## 3. 从指南提炼的开发与验收流程

### 3.1 开始实现前

记录算子定义、输入输出、属性、shape 范围、dtype、format、精度规则、工程模板和提交方式。先实现功能，再处理性能；代码应覆盖规定输入范围，不针对单一测试用例硬编码。

### 3.2 功能测试

覆盖常规、随机、边界、非对齐和大尺寸输入；同时覆盖主要 shape 与 dtype。与预先确定的参考实现比较，保存通过情况和错误信息。Tiling 学习至少用 3 组 shape；MatMul 学习至少用两组尺寸；这些是入门练习下限，不是比赛验证充分性的证明。

### 3.3 性能优化

先对功能正确的版本采集基线，用 Profiling / 仿真定位主要瓶颈；选择一个优化假设，保存修改前后数据。对比必须使用相同环境与测量方法，优化后重新确认功能和泛化能力。

### 3.4 提交前检查

- [ ] 输入、输出、属性、dtype、format 与赛题原型一致。
- [ ] 基础、随机、边界、大尺寸与非对齐用例通过。
- [ ] 精度满足原定标准，主要 shape / dtype 已覆盖。
- [ ] 性能前后使用同环境、同测量方法。
- [ ] 团队赛由另一名成员复现，个人赛在新目录独立复现。
- [ ] 目录、命名和提交方式符合比赛要求。
- [ ] 最终代码、测试结果、版本号与提交回执已保存。

## 4. 评论里的补充提醒

来源：[Issue #37 评论](https://gitcode.com/cann/cann-ops-competitions/issues/37)，重点为 2026-10-01—10-10 的学习笔记。它们多数明确属于文档研读或纸面推导，不能称为已编译、上板验证的结论。以下作为复查主题；具体接口以当前芯片、版本和重载为准。

| 复查主题 | 后续改代码时应核对 |
|---|---|
| 多核与尾块 | 起点、有效长度及余数分配须无遗漏、无重叠；不等长分核不能简单用核号乘本核长度 |
| 双缓冲 | 队列深度与实际缓冲块数是不同参数；开启后重新核算容量、块长、循环次数和偏移 |
| 非对齐搬运 | 区分有效字节与物理缓冲长度；核对接口长度 / stride 的单位及设备支持，向上对齐不扩大 GM 合法边界 |
| Vector 掩码 | 核对掩码模式、单次覆盖范围、repeat、步长及恢复操作；掩码不能代替搬运边界检查 |
| Cast / 舍入 | 正负半整数都要测；枚举存在不代表具体转换支持，标量与向量接口约束不能互套 |
| 归约 | 有效元素数与缓冲容量分开；补位是否影响 max / mean 等结果需分别判断 |
| 浮点累加 | 分块和合并顺序可能改变误差；先排除漏算、重复和越界，再分析精度，不因失败放宽标准 |
| SoftMax / GELU | 参考公式、补位、计算模式和临时空间需一致；数学恒等式不能替代逐元素验收 |

## 5. 应用于 QuantMatmulReluQuant 的优化建议

本节是根据指南流程与当前项目性质提出的应用建议，不是指南原文，也不承诺性能收益。

### 5.1 固定计算与精度语义

从赛题参考实现明确矩阵乘、反量化、ReLU、归约、尺度和再量化的实际顺序；记录每一步的输入输出类型、累加精度、舍入和饱和规则。行级尺度不得因合批混用。融合或改写表达式后，对照这些语义逐项回归。

### 5.2 根据瓶颈选择路线

| 观察到的瓶颈 | 可以验证的优化假设 | 必须保留的证据 |
|---|---|---|
| 小矩阵固定开销高 | 减少标量循环与调度、SIMD 化、按条件合批 | 适用 shape、逐点耗时、额外缓冲和精度 |
| 数据搬运等待明显 | 提高连续搬运和数据复用，减少中间 GM 往返 | 实际搬运量、访问布局、生命周期和同步 |
| Cube 工作不足或分核不均 | 按实际 M/N/K 调整分块与工作分配 | 合法 shape、核间覆盖、尾部负载与耗时 |
| Vector 后处理开销高 | 反量化 / 激活 / 归约 / 量化的向量化或融合 | 每行语义、归约范围、舍入、UB 峰值 |
| 搬运与计算可重叠 | 在容量允许时验证流水或双缓冲 | 依赖关系、完整资源预算、同环境对比 |

没有设备剖析时，将瓶颈判断标为“待验证假设”。先从已确认的逐点耗时与历史失败路线筛选候选；不猜测隐藏测试点并据此硬编码分支。

### 5.3 每次修改的硬性自查

- 分块：主块、尾块、主核、尾核分别计算；核间区间覆盖完整，不重叠。
- 内存：按实际同时存活的缓冲区统计 UB 峰值，计入对齐、双缓冲、临时工作区和所有 dtype 的字节数；互斥路径分别算峰值。L1/L0 等资源单独核算，使用实际设备上限。
- 搬运：每次 GM 读写均在合法范围；补齐空间、有效长度、mask 和归约 count 相互对应。
- 同步：明确 CopyIn → Compute → CopyOut 及 Cube / Vector 的生产消费依赖；删屏障或复用内存前证明上一使用者已完成。
- 精度：覆盖正负值、零、范围极值、量化阈值附近、正负半整数、非整除尺寸及各分支边界；按赛题实际允许范围选择数据。
- 接口：查当前 CANN 版本与设备支持表；旧评论、其他芯片样例或枚举名称不能替代编译和硬件验证。
- 证据：本机 CPU 数值检查用于辅助验证，平台编译、NPU 正确性与硬件性能分别记录。

## 6. 下一轮实验操作清单

1. 读取最新续跑记录，确认用户当前范围；若仅分析，就只产出分析。
2. 确认可恢复的基线源码和 SHA256；读取逐点耗时及失败记录。
3. 用一句话写明假设：哪个 shape / 路径、什么瓶颈、改什么、为何可能更快。
4. 在独立候选目录修改一个主要变量，保存 diff 与资源预算。
5. 完成与改动相关的静态及数值检查；具备授权和环境时，再执行编译、设备正确性与性能验证。
6. 按逐点结果分析收益和退化，使用一致的评分口径；单次峰值、独立提交估算、队伍榜单分别标注。
7. 只有结果支持时才选为推荐版本；复测以用户指令和可用预算为准，未复测就明确标注。
8. 增量保存代码、SHA、环境、原始结果、提交 URL、结论和下一步；回退版本始终可定位。

建议每个候选记录以下字段，尚未完成的检查填 `not_run`，不要填通过：

```yaml
candidate_id: 待填写
baseline_id: 待填写
source_sha256: 待填写
hardware_and_cann: 待核对
hypothesis: 待填写
changed_path_and_shapes: 待填写
ub_peak_bytes: 待核算
correctness_cases: []
cpu_checks: not_run
compile_check: not_run
npu_correctness: not_run
performance_before: null
performance_after: null
submission_url: null
conclusion: pending
rollback_source: 待填写
```

## 7. 学习资料索引

### 核心入口

- [指南及学习打卡帖](https://gitcode.com/cann/cann-ops-competitions/issues/37)：本摘要主来源；原始指南可从该帖的 Word 附件读取；附件链接见本文件开头的 `source_attachment`。
- [CANN 学习中心仓](https://gitcode.com/cann/cann-learning-hub)：已读取首页；后续优先查“算子开发”主线及其教程 README。芯片、依赖、命令按所选教程说明核对。
- [手把手学习视频入口](https://space.bilibili.com/1190614918)：用户提供；本次未读取到视频列表，后续按教程标题匹配，不把视频首页当接口规范。
- [短视频资料入口](https://space.bilibili.com/1218389749)：用户提供；本次未读取到视频列表，适合概念补充，关键实现回查文档和源码。

### 按问题查课程

| 当前问题 | 优先查阅 |
|---|---|
| 基础概念、Add 数据流、多 shape、上板、Profiling | [Kernel 直调版教程](https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development_light)，对应第 2 节章节 |
| 完整工程和社区提交规范 | [算子工程版教程](https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development) |
| 矩阵乘分块与搬运 | [MatMul 样例](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/matmul_story) |
| Vector / 量化后处理 | [RmsNormQuant 样例](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/rms_norm_quant_story)；借鉴优化方法，公式仍按当前赛题 |
| SIMD 与标量开销 | [SIMD VF 样例](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/simd_vf_story)、[Scalar 样例](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/scalar_story) |
| 更多分步优化案例 | [性能优化样例总览](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance) |
| 入门完整实战 | [指南中的 GELU 练习题](https://cannjudge.cn/public/s2/gelu) |

上表样例地址来自已读取的学习仓首页；本次未逐一审核样例实现与设备兼容性，使用前先读各自 README。

### 给后续优化助手的读取提示

> 先读取 Competition/README.md 与仓库 Carl Memory/CANN_续跑记忆.md 的最新状态，再读取 Competition/CANN_算子优化方案_可执行版.md 及本摘要。按当前用户授权确定工作范围；沿用已验证基线，先保证精度和多 shape 泛化，再依据实测瓶颈做单变量优化。核算资源、边界和同步，保存源码 SHA、逐点结果与可恢复版本。区分指南正文、未上板的评论推导和应用建议；需要接口细节时查询对应设备与 CANN 版本的官方文档。
