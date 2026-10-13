> 当前状态：用户已停止自动测评。以下是归档技术资料；未经用户重新授权，不得执行比赛提交。新候选尚未经过 CANN 编译及 NPU 验证。

显式开启 BType 的 ISTRANS=true，使 INT8 B[N,K] 的模板、host SetBType(true)及 kernel SetTensorB(true)一致。只改一个 MatmulType 声明。

源码逆变换精确恢复 v124；CPU 检查不代表 CANN 编译或 NPU 通过。需 15/15 零误差、来源 hash 绑定及邻近对照再决定是否保留。

https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0614.html
