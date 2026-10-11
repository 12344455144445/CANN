# v86：从用户确认提分的v84继续

2026-10-11，用户确认略有提分的版本为v84；具体分数与15点记录未提供。基线SHA b17cf7e9e7da2dca60ffbefd35851c769dadaa1e53c764d301dc608265b07ebc。本目录只产生一个独立候选，未把v83/v85混入基线。

## 改动与理由

已有tiny SIMD点积后，仍有逐列标量读取、两次浮点乘法、求最大值和逐列激活写回。v86在N>=16且尺度安全时改为Muls→Mul→Maxs→WholeReduceMax，然后整块拷贝激活到既有activationBuffer。N<16仍采用原v84标量后处理，避免未经测量地替换最小形状。

两次乘法分开执行；精确除127、近半修正、零行尺度1、输出RINT和钳位完全保留。新增WholeReduceMax只覆盖真实N，不含未初始化lane；拷贝补齐到8浮点元素前先把额外lane清零。复用sums第二半做归约结果，无新增UB分配。

## 浮点安全与回退

device侧检查尺度：零或绝对值在[2^-60,2^40]内。tiny原门控K<=128，INT8点积绝对值<=2^21，因此非零第一步乘法至少2^-60，第二步至少2^-120；上界至多2^101。中间非零乘积保持正常有限FP32，避免用不同指令路径处理subnormal/overflow。

范围外尺度、NaN/Inf及N<16继续执行v84原标量逻辑；原非有限结果回退到device workspace的路径保留。该检查用于数值等价保障，不按官方点编号或输入答案专门分支。

FP32乘法次序不变仍需设备核验。新增指令和同步也有成本，不能保证性能提升。

## 验证与测试

- 初始源码契约检查因v86缺失失败；生成与审查补充后4组CPU/静态检查通过。
- 包含N16..32、K跨32/64、零行、负尺度、范围边界、行padding；比较独立逐元素FP32参考与向量CPU模型的激活、尺度位模式和舍入输出。
- 直接解释实际源码中安全尺度谓词，验证零、正负界限、界外数、subnormal及NaN/Inf分流。
- 解释实际vectorPost谓词，验证N8/15/16/32、零/负/安全边界尺度和不安全尺度选择原路径。独立审查提出这两项源码关联/路径验证缺口，已补齐；未发现阻断性新增源码错误。
- 数值helper、一般Vector/Cube/融合入口、tiny入口与host run_kernel函数体均逐字符保持。
- 这是CPU模型和静态scope证据；没有运行提交源码的ASC编译或NPU执行。

## 操作

解压后双击COPY_V86.cmd，绿色SUCCESS后在网页kernel.asc编辑区Ctrl+A、Ctrl+V完整替换，确认末尾run_kernel后提交。工具会替换剪贴板，只做完整源码校验与复制。

建议同窗口v84→v86→v84，保存任务号、15点用时、页面最优T与总分。NPU编译/15点正确性/性能均待验证；任一点失败，先使用COPY_BASELINE_V84.cmd还原后另行定位。

官方指令参考：https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0079.html

离线复现需要Python/NumPy：python build_v86.py，python test_v86.py；交付包保留相邻2026-10-11-v83-v85/v84的生成基线。通过manifest校验代码字节与版本，LF网页导出可与CRLF原文规范化比较。
