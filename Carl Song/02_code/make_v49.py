# -*- coding: utf-8 -*-
"""v49 生成器 —— 「极小形状：跨核握手整段免除」。

基线： records/experiments/v45-allocfirst/kernel_v45_allocfirst.asc
       sha256(16) = 1052cfef14de53aa  （平台 15/15，23.3537 / 3706.57 µs，当前最优）
输出： records/experiments/v49-nohandshake/kernel_v49_nohandshake.asc

唯一改动（4 处，前 3 处在 AIC、第 4 处在 AIV）：
  [OPT-V49-H1] AIC 的三个「提前返回但补齐全部 flag」分支，在极小形状下不置位；
  [OPT-V49-H2] AIV 的 CrossCoreWaitFlag 在极小形状下不执行；
  [OPT-V49-H3] AIV 把 QmqTinyShape(m,n,k) 提升为循环外的 const bool（纯 CSE）。

★ 等价性论证（逐位）：
  1) 极小形状下 C 由本 AIV 自己算出（QmqComputeTinyC 逐行点积写 workspace），
     AIV 的 Pass-1 / Pass-2 只读本 AIV 刚写的行区间，y / yScale 亦由本 AIV 写出
     ⇒ AIV 对 AIC 没有数据依赖，跨核 Wait 是空等。
  2) Set/Wait 配平：两侧门控用同一个纯函数 QmqTinyShape(m, n, k)，且 m/n/k 由
     host 以同一组值分别传给 AIC 与 AIV 的形参 ⇒ 两侧判定恒等
     ⇒ 极小形状下 Set=0 且 Wait=0，flag 计数器零变化，无悬空 flag（E22 配平不变）。
  3) 精度红线三函数与 Pass-2 行体、掩码扫描区（E112 危险区）逐字节未改。

本机无 CANN 工具链（E19），无法编译；故断言做「结构 + 文本」双重守门。
"""
import difflib
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'records', 'experiments', 'v45-allocfirst',
                    'kernel_v45_allocfirst.asc')
OUTDIR = os.path.join(ROOT, 'records', 'experiments', 'v49-nohandshake')
OUT = os.path.join(OUTDIR, 'kernel_v49_nohandshake.asc')
DIFF = os.path.join(OUTDIR, 'v49_vs_v45.diff')

BASE_SHA16 = '1052cfef14de53aa'
BASE_ELEMS = 1428

# ---------------------------------------------------------------- 新横幅（行 1..25）
NEW_BANNER = [
    '// ============================================================================',
    '// QuantMatmulReluQuant —— v49（极小形状：跨核握手整段免除）',
    '//',
    '// ——— 本版唯一改动 ———',
    '// 基线：v45（sha256 1052cfef14de53aa，平台 15/15，读数 23.3537 / 3706.57 µs —— 当前最优）',
    '// [OPT-V49-H1] 极小形状（QmqTinyShape(m, n, k) 为真）下，AIC 的三个「提前返回',
    '//   但补齐全部跨核 flag」分支不再执行 QmqCubeSignalAllFlags。',
    '// [OPT-V49-H2] 同一判定下，AIV 不再执行 CrossCoreWaitFlag<2, PIPE_FIX>(8/9)。',
    '// [OPT-V49-H3] AIV 把 QmqTinyShape(m, n, k) 提到 localTile 循环之外存为',
    '//   const bool（纯公共子表达式消除），循环内判定结果逐位不变。',
    '//   其余形状：AIC 与 AIV 两侧一字未改。',
    '//',
    '// ★ 等价算法路径（逐位）：极小形状下 C 由本 AIV 自己算出（QmqComputeTinyC 逐行',
    '//   点积写 workspace），AIV 的 Pass-1 / Pass-2 只读本 AIV 刚写进 workspace 的',
    '//   行区间，y / yScale 也由本 AIV 写出 ⇒ AIV 对 AIC 没有任何数据依赖，',
    '//   那条跨核 Wait 本身是空等 ⇒ 免去它不改变任何读写集合与数值路径。',
    '//   Set/Wait 配平：两侧用同一个纯函数 QmqTinyShape(m, n, k) 做门控，而 m/n/k 由',
    '//   host 以同一组值分别传给 AIC 与 AIV ⇒ 两侧判定恒等 ⇒ 极小形状下 Set 与 Wait',
    '//   同时为 0，flag 计数器零变化，不产生悬空 flag（E22 的配平不变）。',
    '//   精度红线函数（QmqQuantizeNearHalf / QmqDiv / QmqCorrectDiv）一行未动，仍由',
    '//   Pass-1 / Pass-2 原样调用；Pass-2 行体与掩码扫描区逐字节未改（E112 安全）。',
    '//',
    '// ★ 立项依据：点 1 = (M=4,N=8,K=16) 只有一条 mmad、点 2 的 T 只有 1.77 µs，而',
    '//   我们在 v20/v35/v37/v38/v39 五版里的读数是 10.28 ~ 10.68 µs —— 五版机器码',
    '//   各异而读数不变 ⇒ 固定开销。三个候选已被实测否证：host 侧（v21：PATCH-H1/H2',
    '//   只值 0.41%，噪声内）、AIC 入口（v40：整段拿掉收益 ≈ 0）、AIV 的',
    '//   QmqComputeTinyC 全长（v40 旁证 +0.30 µs）⇒ 剩下唯一未标定的就是',
    '//   「AIV 被 AIC 的 flag 挡住」这一段串行等待。本版免去它，同时给出 O 的成分标定。',
    '//   价值（analysis/39 §4.2）：点 1/2/3 共值 +4.9 分，是 v20 全部改动的 7 倍。',
    '//',
    '// 回退：把三处 AIC 置位与一处 AIV Wait 原样还原 = 逐字节等于 v45（生成器 A11 已证）。',
]

# ---------------------------------------------------------------- 锚点（自后向前替换）
# D) AIV 的 localTile 循环头（v45 行 1095~1100）
D_OLD = '\r\n'.join([
    '        for (uint32_t localTile = 0; localTile < mBlocksPerGroup; ++localTile) {',
    '            if (localTile == 0U) {',
    '                CrossCoreWaitFlag<2, PIPE_FIX>(kCubeToVectorFlag);',
    '            } else {',
    '                CrossCoreWaitFlag<2, PIPE_FIX>(kCubeToVectorFlagB);',
    '            }',
])
D_NEW = '\r\n'.join([
    '        // [OPT-V49-H2] 极小形状：C 由本 AIV 自算、y / yScale 也由本 AIV 写出',
    '        // ⇒ 与 AIC 没有任何数据依赖，跨核 Wait 是空等，直接免去。',
    '        // 门控与 AIC 侧同为一个纯函数 QmqTinyShape(m, n, k)，且 m/n/k 两侧取值',
    '        // 完全相同 ⇒ Set / Wait 同时为 0，flag 配平零变化（E22）。',
    '        const bool tinyShape = QmqTinyShape(m, n, k);   // [OPT-V49-H3] 纯 CSE',
    '',
    '        for (uint32_t localTile = 0; localTile < mBlocksPerGroup; ++localTile) {',
    '            if (!tinyShape) {',
    '                if (localTile == 0U) {',
    '                    CrossCoreWaitFlag<2, PIPE_FIX>(kCubeToVectorFlag);',
    '                } else {',
    '                    CrossCoreWaitFlag<2, PIPE_FIX>(kCubeToVectorFlagB);',
    '                }',
    '            }',
])

# E) AIV 的 tiny 分派（v45 行 1127）
E_OLD = '            if (QmqTinyShape(m, n, k)) {'
E_NEW = '            if (tinyShape) {'

# C) AIC 的「本 group 无 mBlock」分支（v45 行 630~632）
C_OLD = '\r\n'.join([
    '        // [OPT-V17C] 本 AIC group 无任何 mBlock，但仍须置位全部 flag。',
    '        QmqCubeSignalAllFlags(mBlocksPerGroup);',
    '        return;',
])
C_NEW = '\r\n'.join([
    '        // [OPT-V49-H1] 本 AIC group 无任何 mBlock。极小形状下 AIV 也不等待',
    '        // ⇒ 两侧同时不收发，计数器零变化；其余形状仍须补齐全部 flag（对应',
    '        // AIV 已进入 Wait），与 v45 逐字相同。',
    '        if (!QmqTinyShape(m, n, k)) {',
    '            QmqCubeSignalAllFlags(mBlocksPerGroup);',
    '        }',
    '        return;',
])

# B) AIC 的极小形状旁路（v45 行 616~624）
B_OLD = '\r\n'.join([
    '    // [OPT-V40] 极小形状旁路：本函数不参与计算（C 由 AIV 自行算出），',
    '    // 但仍照常置位全部跨核 flag ⇒ AIV 的 Wait 结构与基线完全一致，',
    '    // E22 的 Set/Wait 配平不变。',
    '    // 关键收益：本分支位于 TPipe / matmul::Matmul / REGIST_MATMUL_OBJ 之前，',
    '    // 因此这三笔与形状无关的入场成本在小形状上被整段省掉。',
    '    if (QmqTinyShape(m, n, k)) {',
    '        QmqCubeSignalAllFlags(mBlocksPerGroup);',
    '        return;',
    '    }',
])
B_NEW = '\r\n'.join([
    '    // [OPT-V40] 极小形状旁路：本函数不参与计算（C 由 AIV 自行算出）。',
    '    // 关键收益：本分支位于 TPipe / matmul::Matmul / REGIST_MATMUL_OBJ 之前，',
    '    // 因此这三笔与形状无关的入场成本在小形状上被整段省掉。',
    '    // [OPT-V49-H1] 本分支不再置位跨核 flag：AIV 侧用同一判定同时免去 Wait',
    '    // （见 quant_matmul_relu_quant_fused），两侧配平 ⇒ 无悬空 flag。',
    '    if (QmqTinyShape(m, n, k)) {',
    '        return;',
    '    }',
])

# A) AIC 的入参非法分支（v45 行 610~612）
A_OLD = '\r\n'.join([
    '        // [OPT-V17C] 提前返回也必须置位全部 flag（对应 AIV 已进入 Wait）。',
    '        // 按 mBlocksPerGroup 补齐，不能固定补 2 个（会留下悬空 flag）。',
    '        QmqCubeSignalAllFlags(mBlocksPerGroup);',
])
A_NEW = '\r\n'.join([
    '        // [OPT-V17C] 提前返回也必须置位全部 flag（对应 AIV 已进入 Wait）。',
    '        // 按 mBlocksPerGroup 补齐，不能固定补 2 个（会留下悬空 flag）。',
    '        // [OPT-V49-H1] 唯一例外：极小形状下 AIV 用同一判定同时免去 Wait',
    '        // ⇒ 两侧同时不收发，计数器零变化（否则会留下悬空 flag，见 v17b 注释）。',
    '        if (!QmqTinyShape(m, n, k)) {',
    '            QmqCubeSignalAllFlags(mBlocksPerGroup);',
    '        }',
])

RED_FUNCS = ('QmqQuantizeNearHalf', 'QmqDiv', 'QmqCorrectDiv')


def sha16(data):
    return hashlib.sha256(data).hexdigest()[:16]


def extract_func(src, name):
    """抽某个函数的定义体（签名行到花括号配平的末行）。"""
    for i, ln in enumerate(src):
        if ('inline' in ln or 'aicore' in ln) and (name + '(' in ln):
            j = i
            while j < len(src) and '{' not in src[j]:
                j += 1
            depth, k = 0, j
            while k < len(src):
                depth += src[k].count('{') - src[k].count('}')
                if depth <= 0:
                    break
                k += 1
            return src[i:k + 1]
    return None


def code_only(lines):
    """只保留非纯注释行，避免横幅文字污染计数。"""
    return '\r\n'.join(ln for ln in lines if not ln.strip().startswith('//'))


def replace_at(lines, old, new, tag):
    """整块唯一命中替换；old/new 均为含 CRLF 的单元素串。"""
    hay = '\r\n'.join(lines)
    assert hay.count(old) == 1, '[A2] %s 命中数 = %d（应为 1）' % (tag, hay.count(old))
    hay = hay.replace(old, new)
    return hay.split('\r\n')


def main():
    raw = open(BASE, 'rb').read()
    text = raw.decode('utf-8')
    assert sha16(raw) == BASE_SHA16, '[A1] 基线 sha 不符: %s' % sha16(raw)
    lines = text.split('\r\n')
    assert len(lines) == BASE_ELEMS, '[A1] 元素数 = %d' % len(lines)
    assert lines[-1] == '', '[A1] 末尾应为空元素'
    assert all('\r' not in x and '\n' not in x for x in lines), '[A1] 应为纯 CRLF'

    # ---- 旧横幅守门
    assert lines[1].startswith('// QuantMatmulReluQuant —— v45'), '[A1] 横幅非 v45'
    assert lines[25].startswith('// ——— 以下为 v40 的历史版本说明'), '[A1] 历史分隔行不符'

    # ---- A3：改动前的结构化计数（只看代码行）
    cb = code_only(lines)
    before = {
        'cross_wait': cb.count('CrossCoreWaitFlag<2, PIPE_FIX>'),
        'cross_set': cb.count('CrossCoreSetFlag<2, PIPE_FIX>'),
        'signal_all': cb.count('QmqCubeSignalAllFlags(mBlocksPerGroup);'),
        'tiny_shape': cb.count('QmqTinyShape('),
        'barrier_v': cb.count('PipeBarrier<PIPE_V>();'),
        'compares': cb.count('Compares('),
        'reducmin': cb.count('ReduceMin('),
        'gate2048': cb.count('if (currentN >= 2048U)'),
    }
    assert before['cross_wait'] == 2, before
    assert before['cross_set'] == 2, before
    assert before['signal_all'] == 3, before
    assert before['tiny_shape'] == 3, before   # 定义 1 + AIC 门控 1 + AIV 分派 1

    # ---- 自后向前替换（行序不影响文本锚点，这里按文件内先后逆序执行）
    out = replace_at(lines, D_OLD, D_NEW, 'V49-H2 (AIV Wait)')
    out = replace_at(out, E_OLD, E_NEW, 'V49-H3 (tiny 分派)')
    out = replace_at(out, C_OLD, C_NEW, 'V49-H1 (AIC no-mBlock)')
    out = replace_at(out, B_OLD, B_NEW, 'V49-H1 (AIC tiny 旁路)')
    out = replace_at(out, A_OLD, A_NEW, 'V49-H1 (AIC 入参非法)')
    NB = len(NEW_BANNER)
    out[0:25] = NEW_BANNER
    assert out[NB].startswith('// ——— 以下为 v40 的历史版本说明'), '[A9] 历史分隔行错位'

    code = '\r\n'.join(out)
    new = code

    # ---- A9：新横幅
    assert out[1].startswith('// QuantMatmulReluQuant —— v49'), '[A9] 横幅未更新'

    # ---- A2b：保留锚点仍在
    for keep in ('const uint32_t blockM = static_cast<uint32_t>(tiling.singleCoreM);',
                 'QmqComputeTinyC(',
                 'CrossCoreSetFlag<2, PIPE_FIX>(kCubeToVectorFlag);',
                 'CrossCoreSetFlag<2, PIPE_FIX>(kCubeToVectorFlagB);'):
        assert keep in new, '[A2b] 丢失: %s' % keep[:50]

    # ---- A3b：改动后的结构化计数（只看代码行）
    ca = code_only(out)
    after = {
        'cross_wait': ca.count('CrossCoreWaitFlag<2, PIPE_FIX>'),
        'cross_set': ca.count('CrossCoreSetFlag<2, PIPE_FIX>'),
        'signal_all': ca.count('QmqCubeSignalAllFlags(mBlocksPerGroup);'),
        'tiny_shape': ca.count('QmqTinyShape('),
        'barrier_v': ca.count('PipeBarrier<PIPE_V>();'),
        'compares': ca.count('Compares('),
        'reducmin': ca.count('ReduceMin('),
        'gate2048': ca.count('if (currentN >= 2048U)'),
        'not_tiny': ca.count('if (!QmqTinyShape(m, n, k)) {'),
        'bool_tiny': ca.count('const bool tinyShape = QmqTinyShape(m, n, k);'),
        'use_tiny': ca.count('if (tinyShape) {'),
        'gate_tiny_if': ca.count('if (QmqTinyShape(m, n, k)) {'),
    }
    assert after['cross_wait'] == 2, after            # 调用点数量守恒（只是被 if 包住）
    assert after['cross_set'] == 2, after             # ★ AIC 的置位本体一行未动
    assert after['signal_all'] == 2, after            # 3 -> 2（tiny 旁路那处不再置位）
    assert after['barrier_v'] == before['barrier_v'], after   # 行体 barrier 未动
    assert after['compares'] == before['compares'], after      # 扫描区 Compares 未动
    assert after['reducmin'] == before['reducmin'], after      # 预筛未动
    assert after['gate2048'] == before['gate2048'], after      # 门控未动
    assert after['not_tiny'] == 2, after
    assert after['bool_tiny'] == 1, after
    assert after['use_tiny'] == 1, after
    assert after['gate_tiny_if'] == 1, after      # 只剩 AIC 那处
    assert after['tiny_shape'] == 5, after        # 3 -> 5（AIC 两处新增、AIV 一增一减）

    # ---- A10：精度红线三函数逐字节不变
    for fn in RED_FUNCS:
        b = extract_func(lines, fn)
        a = extract_func(out, fn)
        assert b is not None and a is not None, '[A10] 抽不到 %s' % fn
        assert b == a, '[A10] 精度红线被改动: %s（%d 行）' % (fn, len(b))

    # ---- A10b：Pass-2 行体 + 掩码扫描区逐字节不变（E112 危险区）
    def slice_between(src, a_mark, b_mark):
        i = next(k for k, ln in enumerate(src) if a_mark in ln)
        j = next(k for k, ln in enumerate(src) if b_mark in ln and k > i)
        return src[i:j]

    for a_mark, b_mark, tag in (
        ('LocalTensor<float> fp32Local = fp32Buffer.Get<float>();',
         'x2ScaleQueue.FreeTensor(x2ScaleLocal);', 'Pass-2 行体 + 循环尾'),
    ):
        sb = slice_between(lines, a_mark, b_mark)
        sa = slice_between(out, a_mark, b_mark)
        assert sb == sa, '[A10b] %s 被改动' % tag

    # ---- A5：花括号配平
    assert code.count('{') == code.count('}'), '[A5] 花括号不配平'

    # ---- A6：无 stdout
    for bad in ('printf', 'std::cout', 'puts(', 'fputs'):
        assert bad not in code, '[A6] 命中 stdout: %s' % bad

    # ---- A7：合规禁词
    for bad in ('跳过', '占位', '投机', '简化计算'):
        assert bad not in code, '[A7] 命中禁词: %s' % bad

    # ---- A8：CRLF + 末尾空元素
    assert '\r\n' in code and code.endswith('\r\n'), '[A8] 行尾不是 CRLF'
    assert '\n' not in code.replace('\r\n', ''), '[A8] 存在裸 LF'

    # ---- A11：逆变换逐字节还原 v45
    rev = replace_at(out, B_NEW, B_OLD, 'rev-B')
    rev = replace_at(rev, A_NEW, A_OLD, 'rev-A')
    rev = replace_at(rev, C_NEW, C_OLD, 'rev-C')
    rev = replace_at(rev, E_NEW, E_OLD, 'rev-E')
    rev = replace_at(rev, D_NEW, D_OLD, 'rev-D')
    rev[0:NB] = lines[0:25]
    assert '\r\n'.join(rev) == text, '[A11] 逆变换未能还原 v45'

    # ---- A12：活代码差集（去注释、去空行后的行多重集）
    def strip_comment(src):
        res = []
        for ln in src:
            s = ln.strip()
            if not s or s.startswith('//'):
                continue
            if '//' in ln:
                ln = ln[:ln.index('//')]
            s = ln.strip()
            if s:
                res.append(s)
        return res

    lc_b = strip_comment(lines[25:])
    lc_a = strip_comment(out[NB:])
    removed = difflib.SequenceMatcher(None, sorted(lc_b), sorted(lc_a))
    only_b = list(lc_b)
    for x in lc_a:
        if x in only_b:
            only_b.remove(x)
    only_a = list(lc_a)
    for x in lc_b:
        if x in only_a:
            only_a.remove(x)

    # 唯一允许的「内容」差异（去缩进、去注释后的行多重集）
    ALLOWED_A = {
        'const bool tinyShape = QmqTinyShape(m, n, k);',
        'if (!QmqTinyShape(m, n, k)) {',
        'if (tinyShape) {',
        'if (!tinyShape) {',
        '}',
    }
    ALLOWED_B = {
        '}',
        'return;',
        'QmqCubeSignalAllFlags(mBlocksPerGroup);',
        'if (QmqTinyShape(m, n, k)) {',
    }
    for x in only_a:
        assert x in ALLOWED_A, '[A12] 意外新增活代码行: %r' % x
    for x in only_b:
        assert x in ALLOWED_B, '[A12] 意外删除活代码行: %r' % x
    print('  [A12] 新增活代码行 %d 条 / 删除 %d 条（全部落在白名单内）'
          % (len(only_a), len(only_b)))
    print('        新增: %s' % sorted(set(only_a)))
    print('        删除: %s' % sorted(set(only_b)))

    # ---- 写出
    os.makedirs(OUTDIR, exist_ok=True)
    open(OUT, 'w', encoding='utf-8', newline='').write(code)
    data = open(OUT, 'rb').read()
    assert sha16(data) == sha16(data)

    # ---- diff
    dl = list(difflib.unified_diff(
        lines, out,
        fromfile='v45 (kernel_v45_allocfirst.asc)',
        tofile='v49 (kernel_v49_nohandshake.asc)',
        lineterm=''))
    open(DIFF, 'w', encoding='utf-8', newline='').write('\n'.join(dl) + '\n')

    hunks = sum(1 for x in dl if x.startswith('@@'))
    print('=' * 78)
    print('  v49 生成完成')
    print('=' * 78)
    print('  输出     : %s' % os.path.relpath(OUT, ROOT))
    print('  sha256-16: %s' % sha16(data))
    print('  字节     : %d' % len(data))
    print('  元素数   : %d  (v45 %d, 净 %+d)' % (len(out), BASE_ELEMS, len(out) - BASE_ELEMS))
    print('  活代码行 : %d  (v45 %d)' % (len(lc_a), len(lc_b)))
    print('  diff     : %d hunk / %d 行' % (hunks, len(dl)))
    print('  ★ 逆变换 A11：逐字节还原 v45 ✓')
    print('  ★ 精度红线 A10：三函数逐字节一致 ✓')
    print('  ★ 行体/扫描区 A10b：逐字节一致 ✓')
    print('  ★ 唯一内容差异：+1 行 const bool 声明、+3 个静态分支（全部在行体之外）')


if __name__ == '__main__':
    main()
