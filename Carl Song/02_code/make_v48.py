# -*- coding: utf-8 -*-
"""
v48 生成器 —— 以 v45（当前最高分）为基线，只做【行体内部】的等价改写。

基线：records/experiments/v45-allocfirst/kernel_v45_allocfirst.asc
      sha256(16) = 1052cfef14de53aa（今日 T 口径 23.3537 分 / 3706.57 µs，当前最高分）

立项依据（analysis/57，v47 实测的逐点解剖）：
  · v47 = v40 + A + B1 + B2 实测 15/15、错误占比全 0.00%，总耗时 3675.38 µs（全项目最低），
    但总分 23.1175 < v45 的 23.3537 ⇒ 「总分 ≠ 总耗时」第三次实证（E109 副产物）。
  · 单条改动逐点核算（今日 T 口径）：
      A （删行体恒等 Maxs） 总耗时 −27.54 µs 但总分 −0.1523（点 4/5 各掉 1.2/1.1 分）⇒ 负项；
      B1（AllocTensor 上移，纯重排） 只值 +0.025（噪声带内）⇒ v45 ≈ v40；
      B2（两个行标量循环合并） 掉分，点 2 由 10.49 → 10.37 → 11.58（全版本最差）；
      C （取消 currentN>=2048 的 ReduceMin 预筛） 是 v42/v43/v46 慢簇的元凶（+748 µs / +20%）。
  · 本轮把「行体」当一条**串行依赖链**读（E108）：Pass-2 行体 19 条指令（17 条向量
    + 1 条 UB->UB 搬运 + 1 条 DataCopyPad），每条读上一条的输出 ⇒ 删 1 条 ≈ −5.3% 行体。
  ⇒ 唯一还能安全拿的是「链上真正多余的那几条」，且必须避开 E112 的扫描区。

本版唯一目标（三处改动，全部逐位等价、全部不碰 E112 扫描区）：
  [OPT-V48-B2] 删掉行内 `DataCopy(activationLocal, fp32Local, alignedN)`：
               QmqDiv 直接写 activationLocal，fp32Local 反过来保留 relu 后的激活值
               ⇒ 少 1 条 UB->UB 搬运 + 少 1 条 PipeBarrier<PIPE_V>。
  [OPT-V48-B3] 末尾 int32 中转不再写 inputLocal（角色互换后它已被 activationLocal
               别名、且正是 Cast 的源，不能原地转写），改写到 fp32Local 的 int32 视图。
  [OPT-V48-B4] `SetDeqScale(1.0f)` 从行内上提到 Pass-2 的 nOffset 循环之前（循环不变量）。

★ 为什么本版**不做** B5（近半残差链 floor -> rint，删 1 条 Adds）：
  推导后发现原设计有两处硬伤，收益（约 −0.7% 总耗时 ≈ +0.09 分）远小于风险：
    ① 数学：令 d = activation / rowScale、f = frac(d)。旧残差 = |d − floor(d) − 0.5|
       = |f − 0.5|；新残差 |d − rint(d)| = min(f, 1−f)。两者**互补**（min(f,1−f) +
       |f−0.5| ≡ 0.5）而**非恒等** ⇒ 判据必须整体换成「> 0.5 − tol」。
    ② 致命：预筛那行 `hasNearHalf = !(scalarMinLocal.GetValue(0) >= kHalfScreenTolerance)`
       吃的是 **ReduceMin**。残差换成新定义后，该式等价于「存在 lane 使旧残差 > tol」，
       与原意「存在 lane 使旧残差 < tol」**方向相反** ⇒ hasNearHalf 几乎恒 false
       ⇒ 扫描被跳过 ⇒ **输出直接错误**。要修就得把 ReduceMin 换成 ReduceMax，
       而 E112 已三次实证该扫描区「微扰即跨 +17~20% 悬崖」（v42/v43/v46）。
  ⇒ 放弃 B5。近半残差链留待「结构级重构 + 自带形状对照」时一并处理。

等价性（逐位）：
  B2：原实现用 `DataCopy(activationLocal, fp32Local)` 把 relu 后的激活值从 fp32Local
      位拷贝到 activationLocal（= inputLocal 的 float 视图），再把 fp32Local 原地
      覆盖成商 d。本版让 QmqDiv 把商写进 activationLocal，fp32Local 保留激活值。
      · 掩码扫描读到的两个标量：原 (activationLocal, fp32Local) = (激活, d)；
        本版 (fp32Local, activationLocal) = (激活, d) —— DataCopy 是位拷贝，逐位相同。
      · 掩码命中写回：原写 fp32Local（d 的宿主）；本版写 activationLocal（d 的宿主）。
      · 未命中 lane：原 fp32Local 保持 d；本版 activationLocal 保持 d。
      · else 分支（rowMax <= 0）：原 Duplicate(fp32Local, 0.0f)；本版 Duplicate(
        activationLocal, 0.0f) —— 二者都是收尾段的输入，收尾段结果同为 0。
      · 收尾段：原对 fp32Local 做 Maxs/Mins/Cast；本版对 activationLocal 做同样三步，
        输入值逐位相同 ⇒ 输出逐位相同。
      · activationLocal 的声明上移到 if/else 之外，只为满足 C++ 作用域（它同时被 else
        分支与收尾段引用），是一行零开销的编译期别名，不产生任何指令。
  B3：原 `Cast(inputLocal, fp32Local, CAST_RINT)` 把 float 量化值写进 inputLocal
      （workspaceQueue 的 int32 缓冲）。角色互换后 inputLocal 已被 activationLocal
      别名且正是 Cast 的源 ⇒ 必须换目的地。改写到 fp32Local.ReinterpretCast<int32_t>()
      （fp32Buffer 的 int32 视图）：与 activationLocal 不同源、不别名，float->int32
      的 RoundMode 与源值不变 ⇒ 逐位相同。fp32Local 此刻已完成全部使命。
  B4：`SetDeqScale` 是标量寄存器写，全文唯一一次；nOffset / row 循环体内没有任何
      一处改写 deq scale（QmqDiv 只调 Div，QmqQuantizeNearHalf 是纯标量函数），
      故把它提到行循环之外设置一次，与逐行重复设置完全等价。

纪律要点：E37（整块锚点）｜E21（CRLF）｜E56/E75/E85（剔注释同口径计数）｜E74（自后向前）｜
          E87（横幅随版本更新）｜A11（逆变换逐字节还原基线）｜A12（活代码差集）｜
          A14（E112 扫描区结构零改动守门）｜A10（精度红线三区逐字节不变）
"""

import hashlib
import io
import os
from collections import Counter

BASE = 'records/experiments/v45-allocfirst/kernel_v45_allocfirst.asc'
OUTDIR = 'records/experiments/v48-rowbody'
OUT = os.path.join(OUTDIR, 'kernel_v48_rowbody.asc')

BASE_SHA16 = '1052cfef14de53aa'
BASE_ELEMS = 1428           # split('\r\n') 后的元素数（末尾空元素）


def load(path):
    with io.open(path, encoding='utf-8', newline='') as f:
        text = f.read()
    return text, text.split('\r\n')


def strip_comment(lines):
    out = []
    for x in lines:
        t = x.strip()
        if t == '' or t.startswith('//'):
            continue
        out.append(t)
    return '\n'.join(out)


def find_all(lines, block):
    n, m = len(lines), len(block)
    return [i for i in range(n - m + 1) if lines[i:i + m] == block]


def replace_at(lines, start, old, new):
    assert lines[start:start + len(old)] == old, '块内容不匹配'
    return lines[:start] + new + lines[start + len(old):]


# ---------------------------------------------------------------- 改动块定义
# T0：[OPT-V48-B4] SetDeqScale 外提
T0_OLD = [
    '    DataCopyPad(yScaleGm[rowStart], yScaleLocal, yScaleCopy);',
    '    yScaleQueue.FreeTensor(yScaleLocal);',
    '',
    '    for (uint32_t nOffset = 0; nOffset < n; nOffset += kVectorTileN) {',
]
T0_NEW = [
    '    DataCopyPad(yScaleGm[rowStart], yScaleLocal, yScaleCopy);',
    '    yScaleQueue.FreeTensor(yScaleLocal);',
    '',
    '    // [OPT-V48-B4] deq scale 是 Pass-2 的循环不变量（全程 1.0f），上提到行循环',
    '    // 之外只设置一次；nOffset / row 两层循环体内没有任何一处改写 deq scale',
    '    // （QmqDiv 只调用 Div，QmqQuantizeNearHalf 是纯标量函数）⇒ 本搬移等价。',
    '    SetDeqScale(static_cast<half>(1.0f));',
    '',
    '    for (uint32_t nOffset = 0; nOffset < n; nOffset += kVectorTileN) {',
]

# T1：[OPT-V48-B2] activationLocal 声明上移到行循环开头（作用域需要，零开销）
#     锚点取 949+950 两行：Pass-1 的 824 行同名单行声明不构成连续两行，故本块唯一。
T1_OLD = [
    '                LocalTensor<float> fp32Local = fp32Buffer.Get<float>();',
    '                Cast(fp32Local, inputLocal, RoundMode::CAST_RINT, alignedN);',
]
T1_NEW = [
    '                LocalTensor<float> fp32Local = fp32Buffer.Get<float>();',
    '                // [OPT-V48-B2] 商将直接写进 activationLocal（inputLocal 的 float',
    '                // 视图），fp32Local 反过来保管 relu 后的激活值 —— 两块缓冲区角色',
    '                // 互换，省掉行内那一次 UB->UB 搬运。声明上移到 if/else 之外只是',
    '                // 为满足 C++ 作用域（它同时被 else 分支与收尾段引用），零开销。',
    '                LocalTensor<float> activationLocal =',
    '                    inputLocal.ReinterpretCast<float>();',
    '                Cast(fp32Local, inputLocal, RoundMode::CAST_RINT, alignedN);',
]

# T2：[OPT-V48-B2] 删 DataCopy + QmqDiv 改址
T2_OLD = [
    '                LocalTensor<float> activationLocal = inputLocal.ReinterpretCast<float>();',
    '                LocalTensor<float> negScaleLocal = halfBuffer.Get<float>();',
    '                LocalTensor<float> residualLocal = reduceBuffer.Get<float>();',
    '                LocalTensor<float> scratchLocal = halfBuffer.Get<float>();',
    '                LocalTensor<uint8_t> correctionMaskLocal =',
    '                    correctionMaskBuffer.Get<uint8_t>();',
    '                const float scaleValue = rowScalePrefetch[row];',
    '                DataCopy(activationLocal, fp32Local, alignedN);',
    '                PipeBarrier<PIPE_V>();',
    '                Duplicate(negScaleLocal, scaleValue, alignedN);',
    '                PipeBarrier<PIPE_V>();',
    '                QmqDiv(fp32Local, activationLocal, negScaleLocal, alignedN);',
]
T2_NEW = [
    '                LocalTensor<float> negScaleLocal = halfBuffer.Get<float>();',
    '                LocalTensor<float> residualLocal = reduceBuffer.Get<float>();',
    '                LocalTensor<float> scratchLocal = halfBuffer.Get<float>();',
    '                LocalTensor<uint8_t> correctionMaskLocal =',
    '                    correctionMaskBuffer.Get<uint8_t>();',
    '                const float scaleValue = rowScalePrefetch[row];',
    '                // [OPT-V48-B2] 商直接写 activationLocal；fp32Local 保留激活值',
    '                // （原实现要靠上一行那条 DataCopy 才能同时保住两份数据）。',
    '                Duplicate(negScaleLocal, scaleValue, alignedN);',
    '                PipeBarrier<PIPE_V>();',
    '                QmqDiv(activationLocal, fp32Local, negScaleLocal, alignedN);',
]

# T3：[OPT-V48-B2] floor 链的源随角色互换
T3_OLD = [
    '                Cast(floorLocal, fp32Local, RoundMode::CAST_FLOOR, alignedN);',
]
T3_NEW = [
    '                // [OPT-V48-B2] 商的宿主已换成 activationLocal。',
    '                Cast(floorLocal, activationLocal, RoundMode::CAST_FLOOR, alignedN);',
]

# T4：[OPT-V48-B2] 残差减法的源随角色互换
T4_OLD = [
    '                Sub(residualLocal, fp32Local, residualLocal, alignedN);',
]
T4_NEW = [
    '                // [OPT-V48-B2] 同上：被减数是商，现在住在 activationLocal。',
    '                Sub(residualLocal, activationLocal, residualLocal, alignedN);',
]

# T5：[OPT-V48-B2] 扫描体内两个标量入参随角色互换
T5_OLD = [
    '                                const int32_t quantized = QmqQuantizeNearHalf(',
    '                                    activationLocal.GetValue(lane),',
    '                                    scaleValue,',
    '                                    fp32Local.GetValue(lane));',
    '                                if (quantized >= 0) {',
    '                                    fp32Local.SetValue(lane, static_cast<float>(quantized));',
    '                                }',
]
T5_NEW = [
    '                                // [OPT-V48-B2] 入参随角色互换：activation（relu 后',
    '                                // 的激活值）现在住在 fp32Local，approximate（商）住在',
    '                                // activationLocal —— 两个标量逐位相同。',
    '                                const int32_t quantized = QmqQuantizeNearHalf(',
    '                                    fp32Local.GetValue(lane),',
    '                                    scaleValue,',
    '                                    activationLocal.GetValue(lane));',
    '                                if (quantized >= 0) {',
    '                                    activationLocal.SetValue(',
    '                                        lane, static_cast<float>(quantized));',
    '                                }',
]

# T6：[OPT-V48-B2] else 分支的输出缓冲同步换名
T6_OLD = [
    '                Duplicate(fp32Local, 0.0f, alignedN);',
]
T6_NEW = [
    '                // [OPT-V48-B2] 收尾段的输入已换成 activationLocal（角色互换的连带改动）。',
    '                Duplicate(activationLocal, 0.0f, alignedN);',
]

# T7：[OPT-V48-B3] int32 中转改址 + 后续一跳 Cast 随之改源
T7_OLD = [
    '            Maxs(fp32Local, fp32Local, 0.0f, alignedN);',
    '            Mins(fp32Local, fp32Local, 127.0f, alignedN);',
    '            PipeBarrier<PIPE_V>();',
    '            Cast(inputLocal, fp32Local, RoundMode::CAST_RINT, alignedN);',
    '            PipeBarrier<PIPE_V>();',
    '            SetDeqScale(static_cast<half>(1.0f));',
    '            LocalTensor<half> halfLocal = halfBuffer.Get<half>();',
    '            Cast(halfLocal, inputLocal, RoundMode::CAST_NONE, alignedN);',
]
T7_NEW = [
    '            Maxs(activationLocal, activationLocal, 0.0f, alignedN);',
    '            Mins(activationLocal, activationLocal, 127.0f, alignedN);',
    '            PipeBarrier<PIPE_V>();',
    '            // [OPT-V48-B3] int32 中转改写到 fp32Local 的 int32 视图（原为 inputLocal）：',
    '            // 角色互换后 inputLocal 已被 activationLocal 别名、且正是本 Cast 的源，',
    '            // 不能原地转写；fp32Local 此刻已完成使命，与 activationLocal 不同源、',
    '            // 不别名 ⇒ float->int32 的结果逐位相同。',
    '            LocalTensor<int32_t> quantizedInt32Local =',
    '                fp32Local.ReinterpretCast<int32_t>();',
    '            Cast(quantizedInt32Local, activationLocal, RoundMode::CAST_RINT, alignedN);',
    '            PipeBarrier<PIPE_V>();',
    '            LocalTensor<half> halfLocal = halfBuffer.Get<half>();',
    '            Cast(halfLocal, quantizedInt32Local, RoundMode::CAST_NONE, alignedN);',
]

PLAN = [
    ('V48-B4', T0_OLD, T0_NEW),
    ('V48-B2-decl', T1_OLD, T1_NEW),
    ('V48-B2-front', T2_OLD, T2_NEW),
    ('V48-B2-floor', T3_OLD, T3_NEW),
    ('V48-B2-sub', T4_OLD, T4_NEW),
    ('V48-B2-scan', T5_OLD, T5_NEW),
    ('V48-B2-else', T6_OLD, T6_NEW),
    ('V48-B3-tail', T7_OLD, T7_NEW),
]

# ---------------------------------------------------------------- 1) 基线身份
text0, L0 = load(BASE)
sha16 = hashlib.sha256(text0.encode('utf-8')).hexdigest()[:16]
print('基线 sha256(16) =', sha16, '(期望 %s)' % BASE_SHA16)
assert sha16 == BASE_SHA16, 'A1 基线身份不符 (sha)'
assert len(L0) == BASE_ELEMS, 'A1 元素数 %d' % len(L0)
assert L0[-1] == '', 'A1 末尾不是空元素'
assert '\r\n' in text0 and '\n' not in text0.replace('\r\n', ''), 'A1 非 CRLF'
print('A1 通过：基线身份一致，%d 元素，纯 CRLF' % len(L0))

# ---------------------------------------------------------------- 2) 横幅
OLD_BANNER = L0[1:26]
assert OLD_BANNER[0].startswith('// QuantMatmulReluQuant —— v45'), 'A9 横幅起点不符'
assert OLD_BANNER[24].startswith('// ——— 以下为 v40 的历史版本说明'), 'A9 横幅终点不符'

NEW_BANNER = [
    '// QuantMatmulReluQuant —— v48（行体链减负：去一层 UB->UB 搬运 + int32 中转改址）',
    '//',
    '// ——— 本版改动摘要 ———',
    '// 基线：v45（sha256 1052cfef14de53aa，平台 15/15，今日 T 口径 23.3537 / 3706.57 µs，',
    '//       当前最高分）。v40 = 23.3287 / 3717.95 µs 为判读基准锚。',
    '//',
    '// ★ 立项依据（v47 实测的逐点解剖，见 analysis/57）：',
    '//   v47 = v40 + A + B1 + B2 实测 15/15 Pass、错误占比全 0.00%，总耗时 3675.38 µs',
    '//   （全项目最低），但总分 23.1175 < v45 的 23.3537 ⇒ 「总分 ≠ 总耗时」第三次实证。',
    '//   逐条核算：A（删行体恒等 Maxs）省 27.54 µs 但总分 −0.1523 ⇒ 负项，弃用；',
    '//   B1（纯重排）只值 +0.025 ⇒ 噪声带内；B2（合并行标量循环）掉分 ⇒ 弃用；',
    '//   C（取消 currentN>=2048 的 ReduceMin 预筛）是慢簇元凶（+748 µs / +20%）⇒ 弃用。',
    '//   结论：基线取 v45。',
    '//',
    '// ★ 本轮把「行体」当一条**串行依赖链**读（E108）：Pass-2 行体 19 条指令',
    '//   （17 条向量 + 1 条 UB->UB 搬运 + 1 条 DataCopyPad），每条都读上一条的输出',
    '//   ⇒ 删 1 条 ≈ −5.3% 行体时间。本版只拿链上真正多余的那一条，且',
    '//   **E112 扫描区（预筛分支 / Compares / 字扫描循环）一个字节都不动**。',
    '//',
    '// [OPT-V48-B2] 删掉行内 `DataCopy(activationLocal, fp32Local, alignedN)`：',
    '//   QmqDiv 直接写 activationLocal，fp32Local 反过来保管 relu 后的激活值。原实现',
    '//   要靠这次位拷贝才能同时保住 (activation, approximate) 两份数据；现在由两个',
    '//   不同源的缓冲区分开保管 ⇒ 数值逐位相同，少 1 条向量/搬运 + 1 条 PipeBarrier。',
    '//   连带改动：floor 链与残差减法的源、掩码扫描的两个标量入参、掩码命中写回、',
    '//   else 分支与收尾段的操作数 —— 全部随角色互换同步改址，值逐位不变。',
    '//   activationLocal 的声明上移到 if/else 之外，纯为满足 C++ 作用域，零开销。',
    '// [OPT-V48-B3] 末尾 int32 中转（float->int32）原写进 inputLocal；角色互换后它已被',
    '//   activationLocal 别名且正是该 Cast 的源，不能原地转写 ⇒ 改写到 fp32Local 的',
    '//   int32 视图（与 activationLocal 不同源、不别名）⇒ 结果逐位相同。',
    '// [OPT-V48-B4] `SetDeqScale(1.0f)` 是 Pass-2 的循环不变量（全文唯一一次），上提到',
    '//   行循环之前只设置一次；两层循环体内无任何一处改写 deq scale ⇒ 等价。',
    '//',
    '// ⛔ 本版**不做**近半残差链的 floor->rint 改写（原计划 B5）：残差新定义',
    '//   |d − rint(d)| = min(f, 1−f) 与旧定义 |f − 0.5| 是**互补**（和为 0.5）而非恒等，',
    '//   且预筛那行的 ReduceMin 判据会随之方向反转（hasNearHalf 几乎恒 false ⇒ 输出',
    '//   错误）。要修就得换 ReduceMax —— 而 E112 已三次实证该区微扰即跨 +17~20% 悬崖。',
    '//',
    '// 不新增任何 API、不改核函数签名、不引入运行期类型分支；',
    '// 精度红线函数（QmqQuantizeNearHalf / QmqDiv / QmqCorrectDiv）一行未动。',
    '// 回退：把三处 [OPT-V48-*] 逆向还原，即逐字节等于 v45（生成器 A11 已证）。',
    '// ——— 以下为 v45 / v40 的历史版本说明，保留备查 ———',
]

L = list(L0)
edits = []
for name, old, new in PLAN:
    hits = find_all(L, old)
    assert len(hits) == 1, 'A2 %s 整块匹配命中 %d 次（应为 1）' % (name, len(hits))
    edits.append((name, hits[0], old, new))

for name, start, old, new in sorted(edits, key=lambda e: -e[1]):
    print('A2 %-16s 整块匹配 ✓ 起始行 %d（1-based %d），替换 %d 行 -> %d 行'
          % (name, start, start + 1, len(old), len(new)))
    L = replace_at(L, start, old, new)

L = L[:1] + NEW_BANNER + L[26:]
text1 = '\r\n'.join(L)
code0 = strip_comment(L0)
code1 = strip_comment(L)

# ---------------------------------------------------------------- 3) 结构化断言
print()
print('A3 PipeBarrier<PIPE_V> %d -> %d' % (code0.count('PipeBarrier<PIPE_V>'),
                                            code1.count('PipeBarrier<PIPE_V>')))
print('A3 活代码 %d 行 -> %d 行' % (code0.count('\n'), code1.count('\n')))
print('A3 ReduceMin(          %d -> %d' % (code0.count('ReduceMin('), code1.count('ReduceMin(')))
print('A3 hasNearHalf         %d -> %d' % (code0.count('hasNearHalf'), code1.count('hasNearHalf')))
print('A3 Compares(           %d -> %d' % (code0.count('Compares('), code1.count('Compares(')))
print('A3 QmqQuantizeNearHalf(%d -> %d'
      % (code0.count('QmqQuantizeNearHalf('), code1.count('QmqQuantizeNearHalf(')))
print('A3 SetDeqScale         %d -> %d' % (code0.count('SetDeqScale'), code1.count('SetDeqScale')))
print('A3 DataCopy(activationLocal, fp32Local  %d -> %d'
      % (code0.count('DataCopy(activationLocal, fp32Local'),
         code1.count('DataCopy(activationLocal, fp32Local')))

assert code1.count('ReduceMin(') == 1 and code1.count('hasNearHalf') == 3, 'A3 预筛区被改动'
assert code1.count('if (currentN >= 2048U)') == 1, 'A3 预筛分支应保持 1'
assert code1.count('Compares(') == 1, 'A3 Compares( 应保持 1'
# CMPMODE::LT 全文 3 处：2 处在精度红线 QmqCorrectDiv 内，1 处在 Pass-2 扫描区
assert code1.count('CMPMODE::LT') == 3 and code1.count('CMPMODE::GT') == 0, 'A3 判据模式不得改动'
assert code1.count('Adds(residualLocal, residualLocal, 0.5f') == 1, 'A3 残差链不得改动'
assert code1.count('QmqQuantizeNearHalf(') == 2, 'A3 红线调用数应保持 2'
assert code1.count('DataCopy(activationLocal, fp32Local') == 0, 'A3 B2 未生效'
assert code1.count('QmqDiv(fp32Local, activationLocal') == 0, 'A3 B2 未生效(div 旧址)'
assert code1.count('QmqDiv(activationLocal, fp32Local') == 1, 'A3 B2 未生效(div 新址)'
assert code1.count('Maxs(fp32Local') == 1, 'A3 Maxs(fp32Local 应 2 -> 1'
assert code1.count('Mins(fp32Local') == 0, 'A3 Mins(fp32Local 应 1 -> 0'
assert code1.count('Maxs(activationLocal') == 1, 'A3 收尾 Maxs 未改址'
assert code1.count('Mins(activationLocal') == 1, 'A3 收尾 Mins 未改址'
assert code1.count('Duplicate(activationLocal, 0.0f') == 1, 'A3 else 分支未改址'
assert code1.count('Duplicate(fp32Local') == 0, 'A3 else 分支残留'
assert code1.count('Cast(floorLocal, activationLocal') == 1, 'A3 floor 链源未改'
assert code1.count('Sub(residualLocal, activationLocal') == 1, 'A3 残差减法源未改'
assert code1.count('LocalTensor<float> activationLocal') == 1, 'A3 activationLocal 声明应唯一'
assert code1.count('SetDeqScale') == 1, 'A3 SetDeqScale 应保持 1（只是搬了位置）'
assert code1.count('quantizedInt32Local') == 3, 'A3 B3 int32 中转未生效'
assert code1.count('fp32Local.ReinterpretCast<int32_t>()') == 1, 'A3 B3 目的地未生效'
assert code1.count('Cast(inputLocal, fp32Local') == 0, 'A3 B3 旧目的地残留'
assert code1.count('Cast(halfLocal, quantizedInt32Local') == 1, 'A3 B3 后继 Cast 未改源'
assert code1.count('const int32_t quantized = QmqQuantizeNearHalf(') == 1, 'A3 扫描调用点唯一'
print('A3 通过：扫描区判据与残差链零改动、B2/B3/B4 全部生效')

# 花括号配平
def braces(lines):
    d = 0
    for x in lines:
        for ch in x:
            if ch == '{':
                d += 1
            elif ch == '}':
                d -= 1
    return d


assert braces(L0) == 0 and braces(L) == 0, 'A5 花括号不配平'
print('A5 通过：花括号配平')
for bad in ('printf', 'std::cout', 'puts(', 'fprintf'):
    assert bad not in code1, 'A6 出现 stdout：%s' % bad
print('A6 通过：无 stdout 输出')
for bad in ('跳过', '占位', '投机', '简化计算'):
    assert bad not in text1, 'A7 出现合规禁词：%s' % bad
print('A7 通过：无合规禁词')
assert '\r\n' in text1 and '\n' not in text1.replace('\r\n', ''), 'A8 非 CRLF'
assert L[-1] == '', 'A8 末尾空元素丢失'
print('A8 通过：CRLF + 末尾空元素保持')
assert L[1].startswith('// QuantMatmulReluQuant —— v48'), 'A9 横幅未更新'
print('A9 通过：横幅 = v48')

for sig in ('QmqQuantizeNearHalf(', 'QmqDiv(', 'QmqCorrectDiv('):
    defs = [k for k, x in enumerate(L0) if sig in x and ('inline' in x or 'aicore' in x)]
    assert defs, 'A10 %s 未找到定义' % sig
    i0 = defs[0]
    j0 = i0
    while '{' not in L0[j0]:
        j0 += 1
    depth, k0 = 0, j0
    while k0 < len(L0):
        depth += L0[k0].count('{') - L0[k0].count('}')
        if depth <= 0:
            break
        k0 += 1
    same = [k for k, x in enumerate(L) if x == L0[i0]]
    assert len(same) == 1, 'A10 %s 定义行不唯一' % sig
    i1 = same[0]
    j1 = i1
    while '{' not in L[j1]:
        j1 += 1
    depth, k1 = 0, j1
    while k1 < len(L):
        depth += L[k1].count('{') - L[k1].count('}')
        if depth <= 0:
            break
        k1 += 1
    assert L[i1:k1 + 1] == L0[i0:k0 + 1], 'A10 精度红线函数 %s 被改动' % sig
    print('A10 通过：%s 区逐字节不变（%d 行）' % (sig.rstrip('('), k0 - i0 + 1))

# ---------------------------------------------------------------- 4) A14 E112 扫描区守门
print()
SCAN_ANCHORS = [
    'bool hasNearHalf = true;',
    'if (currentN >= 2048U) {',
    'LocalTensor<float> scalarMinLocal = scalarMaxBuffer.Get<float>();',
    'ReduceMin(scalarMinLocal, residualLocal, scratchLocal, alignedN);',
    'hasNearHalf = !(scalarMinLocal.GetValue(0) >= kHalfScreenTolerance);',
    'if (hasNearHalf) {',
    'Compares(correctionMaskLocal, residualLocal, kHalfScreenTolerance,',
    'CMPMODE::LT, compareCount);',
    'LocalTensor<uint64_t> candidateMask =',
    'correctionMaskLocal.ReinterpretCast<uint64_t>();',
    'for (uint32_t maskGroup = 0; maskGroup < compareCount / 64U;',
    '++maskGroup) {',
    'uint64_t bits = candidateMask.GetValue(maskGroup);',
    'while (bits != 0U) {',
    'const uint32_t bit = static_cast<uint32_t>(',
    'AscendC::GetSFFValue<1>(bits));',
    'const uint32_t lane = maskGroup * 64U + bit;',
    'if (lane < currentN) {',
    'bits &= bits - 1U;',
]
for a in SCAN_ANCHORS:
    n0 = sum(1 for x in code0.split('\n') if x == a)
    n1 = sum(1 for x in code1.split('\n') if x == a)
    assert n0 == n1 and n0 > 0, 'A14 E112 扫描区被改动：%s (%d -> %d)' % (a, n0, n1)
print('A14 通过：E112 扫描区 %d 个锚点逐行文本与计数完全不变' % len(SCAN_ANCHORS))

# ---------------------------------------------------------------- 5) A12 活代码差集
cb, cn = Counter(code0.split('\n')), Counter(code1.split('\n'))
removed, added = cb - cn, cn - cb
print()
print('A12 删除集（%d 条不同行 / 共 %d 行）：' % (len(removed), sum(removed.values())))
for k, v in sorted(removed.items()):
    print('   -X%d %s' % (v, k))
print('A12 新增集（%d 条不同行 / 共 %d 行）：' % (len(added), sum(added.values())))
for k, v in sorted(added.items()):
    print('   +X%d %s' % (v, k))

# 行级编辑脚本（E109：区分「内容变化」与「纯顺序变化」）
import difflib

lines0 = code0.split('\n')
lines1 = code1.split('\n')
sm = difflib.SequenceMatcher(None, lines0, lines1, autojunk=False)
ops = [op for op in sm.get_opcodes() if op[0] != 'equal']
nd = sum(i2 - i1 for t, i1, i2, j1, j2 in ops if t in ('delete', 'replace'))
na = sum(j2 - j1 for t, i1, i2, j1, j2 in ops if t in ('insert', 'replace'))
print('A12b 行级编辑脚本：%d 个非 equal 段（删 %d 行 / 增 %d 行）：'
      % (len(ops), nd, na))
for tag, i1, i2, j1, j2 in ops:
    print('   [%s] 活代码行 %d~%d -> %d~%d' % (tag, i1, i2 - 1, j1, j2 - 1))
    for x in lines0[i1:i2]:
        print('       - %s' % x[:88])
    for x in lines1[j1:j2]:
        print('       + %s' % x[:88])
purermove = [op for op in ops if op[0] in ('insert', 'delete')]
print('A12b 纯插入/删除段（无 replace）：%d 个' % len(purermove))

EXPECT_REMOVED = Counter({
    'Cast(floorLocal, fp32Local, RoundMode::CAST_FLOOR, alignedN);': 1,
    'Cast(halfLocal, inputLocal, RoundMode::CAST_NONE, alignedN);': 1,
    'Cast(inputLocal, fp32Local, RoundMode::CAST_RINT, alignedN);': 1,
    'DataCopy(activationLocal, fp32Local, alignedN);': 1,
    'Duplicate(fp32Local, 0.0f, alignedN);': 1,
    'LocalTensor<float> activationLocal = inputLocal.ReinterpretCast<float>();': 1,
    'Maxs(fp32Local, fp32Local, 0.0f, alignedN);': 1,
    'Mins(fp32Local, fp32Local, 127.0f, alignedN);': 1,
    'PipeBarrier<PIPE_V>();': 1,
    'QmqDiv(fp32Local, activationLocal, negScaleLocal, alignedN);': 1,
    'Sub(residualLocal, fp32Local, residualLocal, alignedN);': 1,
    'activationLocal.GetValue(lane),': 1,
    'fp32Local.GetValue(lane));': 1,
    'fp32Local.SetValue(lane, static_cast<float>(quantized));': 1,
})
EXPECT_ADDED = Counter({
    'Cast(floorLocal, activationLocal, RoundMode::CAST_FLOOR, alignedN);': 1,
    'Cast(halfLocal, quantizedInt32Local, RoundMode::CAST_NONE, alignedN);': 1,
    'Cast(quantizedInt32Local, activationLocal, RoundMode::CAST_RINT, alignedN);': 1,
    'Duplicate(activationLocal, 0.0f, alignedN);': 1,
    'LocalTensor<float> activationLocal =': 1,
    'LocalTensor<int32_t> quantizedInt32Local =': 1,
    'Maxs(activationLocal, activationLocal, 0.0f, alignedN);': 1,
    'Mins(activationLocal, activationLocal, 127.0f, alignedN);': 1,
    'QmqDiv(activationLocal, fp32Local, negScaleLocal, alignedN);': 1,
    'Sub(residualLocal, activationLocal, residualLocal, alignedN);': 1,
    'activationLocal.GetValue(lane));': 1,
    'activationLocal.SetValue(': 1,
    'fp32Local.GetValue(lane),': 1,
    'fp32Local.ReinterpretCast<int32_t>();': 1,
    'inputLocal.ReinterpretCast<float>();': 1,
    'lane, static_cast<float>(quantized));': 1,
})
assert removed == EXPECT_REMOVED, 'A12 删除集不符：%s' % (removed - EXPECT_REMOVED)
assert added == EXPECT_ADDED, 'A12 新增集不符：%s' % (added - EXPECT_ADDED)
print('A12 通过：删除集 %d 条 / 新增集 %d 条与预期逐条一致' % (len(removed), len(added)))
# 关键守门：扫描判据与残差链的文本一行都没进差集
for forbidden in ('kHalfScreenTolerance', 'ReduceMin(', 'Compares(', 'CMPMODE::',
                  'hasNearHalf', '0.5f, alignedN)', 'GetSFFValue'):
    hit = [k for k in list(removed) + list(added) if forbidden in k]
    assert not hit, 'A12 扫描区文本出现在差集中：%s' % hit
print('A12 通过：差集中不含任何扫描判据/残差链文本')

# ---------------------------------------------------------------- 6) A11 逆变换
cur = list(L)
for name, old, new in PLAN:
    hits = find_all(cur, new)
    assert len(hits) == 1, 'A11 %s 逆变换定位失败（%d 次）' % (name, len(hits))
    cur = replace_at(cur, hits[0], new, old)
assert cur[1:1 + len(NEW_BANNER)] == NEW_BANNER, 'A11 横幅区不匹配'
cur = cur[:1] + OLD_BANNER + cur[1 + len(NEW_BANNER):]
assert '\r\n'.join(cur) == text0, 'A11 逆变换未能逐字节还原基线'
print('A11 通过：逆变换逐字节还原 v45')

# ---------------------------------------------------------------- 7) 落盘
os.makedirs(OUTDIR, exist_ok=True)
with io.open(OUT, 'w', encoding='utf-8', newline='') as f:
    f.write(text1)
out_sha = hashlib.sha256(text1.encode('utf-8')).hexdigest()
print()
print('输出 %s' % OUT)
print('%d 元素（含末尾空元素）｜ %d 字节 ｜ sha256(16) = %s'
      % (len(L), len(text1.encode('utf-8')), out_sha[:16]))
