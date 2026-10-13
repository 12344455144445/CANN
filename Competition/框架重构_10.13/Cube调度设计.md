> 当前状态：用户已停止自动测评。以下是归档技术资料；未经用户重新授权，不得执行比赛提交。新候选尚未经过 CANN 编译及 NPU 验证。

# 10.13 Cube 调度框架设计

用户选择的 v124 是本轮源码基线，`Competition/Sheldon10.13am` 是最新研究参考。所有新增 Cube 候选均未编译、未提交、未设备验证。自动测评已按用户要求停止。

## v139：M/N 网格与完整行量化

旧融合路径主要按 M 分配 Cube；小 M、大 N 时有核没有工作。新入口 `qmq_partition_fused_entry` 按 rowGroup 与 columnGroup 划分 C：每个 AIC 计算一个行块和列块，随后所有启动的 AIC/AIV 在唯一的 `SyncAll<false>()` 汇合。只有此后，AIV 按完整行独立归属执行 v124 量化助手。因此尺度仍来自全部 N，未把局部列片段尺度当成整行尺度。

请求门控包括 M<=384、N>=1024 且 N 为 32 倍数、K>=256 且 K 为 32 倍数、N*K>=1 MiB，要求至少两个列组。行块 M<=192 时为 16，否则为 32；请求列宽按 128 对齐。网格预算不超过 availableCoreNum、实际 AIC 核数和实际 AIV 核数的一半。较大 M 还要求相比原 M 网格增加足够有效核；返回的 singleCoreM/N 必须精确匹配请求，否则回退原路径。完整条件以对应 kernel.asc 为准。

A 起点 row*K；物理 B 为 [N,K]，起点 column*K；C 起点 row*N+column。模板、host 和 runtime 的 B 转置声明一致。五参数 `SetOrgShape(m,n,k,k,n)` 指定完整 C 行距 N，随后设置每块的 Tail 和张量地址。部分列块写回不能采用局部列宽作为完整输出行距。

所有已启动核心都参与同一个无条件屏障，零行 AIV 也先参与屏障；屏障前没有提前 return/continue。AIC 的 IterateAll、End 在屏障之前，AIV 在其后读取 C。使用 `__schedmode__(1)` 和 `__mix__(1,2)`，只执行一次 launch。显式 GM 仍为 INT32 C 的 4MN 加 Matmul 系统 workspace，未增加分片激活矩阵。

337680 个几何案例复核了覆盖、网格、尾块和完整行归属；14 个数值布局模型覆盖 10719456 元素。模型没有模拟真实 Matmul、混合调度或 API 事件。全局屏障可能损失原 M tile 间搬算重叠，小 M 的后处理仍只有少数 AIV 持有完整行。编译器语法、部分 N 写回跨度、混合硬屏障、实际命中率和端到端收益都需要设备验证。

## 两个隔离变化的辅助候选

v135 只将 INT8 BType 模板转置参数显式置 true，与 host/runtime 声明一致。原 v124 已通过真实设备正确性验证，不能将声明差异认定为已证实的设备错误或保证收益。

v136 只把不变的 SetOrgShape 移到注册 Matmul 对象之后、块循环之前。Tail 仍按块更新；目的是减少重复设置，收益未测。不要未经独立验证就组合这些候选。

## 官方依据与验证边界

[SyncAll 官方文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0204.html) 描述 A3 混合硬同步和逻辑核数限制，并建议 batch 调度。

[SetOrgShape 官方文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0651.html) 描述五参数形式、C 的 N 跨度及调用顺序。

源码门控遵循这些设计约束，但文档支持不代表候选已通过比赛平台编译。原静态 workspace cache 未隔离 stream/context，容量扩张时释放旧地址，不能据此宣称多流并发安全。当前交付只保存候选和审查，原推荐代码不替换。
