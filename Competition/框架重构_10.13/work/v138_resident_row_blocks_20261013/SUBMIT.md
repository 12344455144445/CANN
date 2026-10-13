> 当前状态：用户已停止自动测评。以下是归档技术资料；未经用户重新授权，不得执行比赛提交。新候选尚未经过 CANN 编译及 NPU 验证。

# V138 容量推导完整行块框架

基线：v124，用户已确认25.25版本为代码基线。

适用所有N为32倍数且32<=N<=8192的正常形状，不要求整个AIV行区间放入UB。按floor(8192/N)批行驻留，每批C只读一次、FP32 Cast和两次缩放各执行一次。批内行最大值统一用原QmqCorrectDiv修复scale；反量化结果保留在fp32Buffer，消费完的INT32输入bank依次作为尺度scratch和量化quotient，免activation UB copy。量化Div、近半值筛查与精修、夹紧和ODD/RINT链均沿用v124。宽N和非32整行对齐保留原fallback。

动态tensor/scale/mask容量分别按64元素/64元素/32B对齐；所有mask padding在比较前初始化。单TPipe分支在fallback TPipe创建前分流。峰值显式UB173952 B，加8192 B API预留共182144 B <=196608 B。4096个维度/行数/尾块组合已检查资源、访问和逐行唯一归属。

不保证提速：各批尺度修复与depth1输入缺少搬算双buffer可能产生回归。需要平台CANN编译、15/15零错误，再按当天v124相邻对照比较并原样复测，不以CPU或最高单次估分替代NPU证据。
