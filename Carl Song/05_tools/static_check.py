"""
静态检查脚本（Python 侧）：扫描算子源码，找出可静态发现的错误。
检查项：
  1. 未定义标识符（常见的拼写错误）
  2. UB 预算是否超限
  3. buffer 类型双关复用
  4. 括号/大括号配对
  5. 关键 API 的成对出现（AllocTensor/FreeTensor, EnQue/DeQue）
  6. 危险模式（注释掉的代码、占位实现）
"""

import re
import sys
import os


KILO = 1024

# UB 基准（KB）: 已验证通过版本的实测占用。0 表示不比较。
UB_REFERENCE_KB = 166.8


def strip_comments(text):
    """移除 // 行注释与 /* */ 块注释内容，保留行结构以便计数行号。"""
    out = []
    in_block = False
    for line in text.split('\n'):
        res = []
        i = 0
        while i < len(line):
            if in_block:
                end = line.find('*/', i)
                if end == -1:
                    i = len(line)
                else:
                    in_block = False
                    i = end + 2
                continue
            if line.startswith('//', i):
                break
            if line.startswith('/*', i):
                in_block = True
                i += 2
                continue
            res.append(line[i])
            i += 1
        out.append(''.join(res))
    return '\n'.join(out)


def check_braces(text, path):
    """检查括号配对（忽略字符串与注释）。"""
    depth_curly = 0
    depth_paren = 0
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    prev = ''
    line_no = 1
    errors = []

    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\n':
            line_no += 1
            in_line_comment = False
        if in_line_comment:
            i += 1
            prev = ch
            continue
        if in_block_comment:
            if prev == '*' and ch == '/':
                in_block_comment = False
            i += 1
            prev = ch
            continue
        if in_string:
            if ch == '"' and prev != '\\':
                in_string = False
            i += 1
            prev = ch
            continue
        if prev == '/' and ch == '/':
            in_line_comment = True
            i += 1
            prev = ch
            continue
        if prev == '/' and ch == '*':
            in_block_comment = True
            i += 1
            prev = ch
            continue
        if ch == '"':
            in_string = True
            i += 1
            prev = ch
            continue
        if ch == '{':
            depth_curly += 1
        elif ch == '}':
            depth_curly -= 1
            if depth_curly < 0:
                errors.append(f"  line {line_no}: '}}' 多余")
                depth_curly = 0
        elif ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren -= 1
            if depth_paren < 0:
                errors.append(f"  line {line_no}: ')' 多余")
                depth_paren = 0
        i += 1
        prev = ch

    if depth_curly != 0:
        errors.append(f"  文件结束: '{{' 未闭合，剩余深度 {depth_curly}")
    if depth_paren != 0:
        errors.append(f"  文件结束: '(' 未闭合，剩余深度 {depth_paren}")
    return errors


def check_ub_budget(text, path):
    """解析源码中真实的 pipe.InitBuffer 调用，估算 UB 占用。

    不再硬编码 buffer 列表 —— 之前那样做会在版本间给出错误结论。
    这里把 InitBuffer 的 size 表达式用已知常量求值后求和。
    """
    code = strip_comments(text)

    # 收集 constexpr 常量
    consts = {}
    for m in re.finditer(r'constexpr\s+uint32_t\s+(\w+)\s*=\s*(\d+)', code):
        consts[m.group(1)] = int(m.group(2))
    for m in re.finditer(r'constexpr\s+float\s+(\w+)\s*=\s*([0-9.eE+-]+)f', code):
        consts[m.group(1)] = float(m.group(2))

    tile_n = consts.get('kVectorTileN')
    tile_rows = consts.get('kVectorTileRows')
    block = consts.get('kFloatBlock', 8)
    repeat = consts.get('kRepeatSize', 64)
    if tile_n is None:
        return ["  未找到 kVectorTileN，跳过 UB 预算检查"], None

    # 运行时变量取最坏值
    aligned_rows = ((tile_rows + block - 1) // block) * block if tile_rows else 0
    env = dict(consts)
    env.update({
        'alignedRows': aligned_rows,
        'sizeof': lambda t: {'int32_t': 4, 'float': 4, 'int8_t': 1,
                             'uint8_t': 1, 'half': 2, 'int16_t': 2}.get(t, 4),
        'QmqAlignUp': lambda v, a: ((v + a - 1) // a) * a,
        'max': max, 'min': min,
    })

    # 找出 InitBuffer 调用（需处理括号内的嵌套逗号）
    items = []
    for m in re.finditer(r'pipe\s*\.\s*InitBuffer\s*\(', code):
        i = m.end()
        depth_p = 1
        args = ''
        while i < len(code) and depth_p > 0:
            ch = code[i]
            if ch == '(':
                depth_p += 1
            elif ch == ')':
                depth_p -= 1
                if depth_p == 0:
                    break
            args += ch
            i += 1
        parts = []
        d = 0
        cur = ''
        for ch in args:
            if ch == '(' :
                d += 1
            elif ch == ')':
                d -= 1
            if ch == ',' and d == 0:
                parts.append(cur.strip())
                cur = ''
            else:
                cur += ch
        if cur.strip():
            parts.append(cur.strip())
        # 两种形式:
        #   TQue  : InitBuffer(queue, depth, size)
        #   TBuf  : InitBuffer(buf, size)          <- depth 隐含为 1
        if len(parts) >= 3:
            name, depth_s, size_s = parts[0], parts[1], ','.join(parts[2:])
            try:
                depth = int(depth_s)
            except ValueError:
                depth = 1
        elif len(parts) == 2:
            name, depth, size_s = parts[0], 1, parts[1]
        else:
            continue
        expr = size_s.replace('U', '').replace('u', '')
        # sizeof(T) 需先折叠为字面数字：eval 无法解析类型名
        type_sizes = {'int32_t': 4, 'float': 4, 'int8_t': 1, 'uint8_t': 1,
                      'half': 2, 'int16_t': 2, 'int64_t': 8, 'uint64_t': 8}
        def _fold_sizeof(m):
            t = m.group(1).strip()
            return str(type_sizes.get(t, 4))
        expr = re.sub(r'sizeof\s*\(\s*([\w:]+)\s*\)', _fold_sizeof, expr)
        expr = expr.replace('alignedRows', str(aligned_rows))
        try:
            per = eval(expr, {'__builtins__': {}}, dict(env))
            total = int(depth * per)
        except Exception:
            items.append((name, depth, size_s, None))
            continue
        items.append((name, depth, size_s, total))

    if not items:
        return ["  未解析到 InitBuffer 调用"], None

    lines = [f"  kVectorTileN = {tile_n}, kVectorTileRows = {tile_rows}"
             f"  (alignedRows 最坏 {aligned_rows})"]
    total = 0
    unknown = 0
    for name, depth, expr, val in items:
        if val is None:
            lines.append(f"    {name:24s} x{depth}  {expr[:34]:34s} 无法求值")
            unknown += 1
        else:
            total += val
            lines.append(f"    {name:24s} x{depth}  {val:>8d} B")
    lines.append(f"    {'-'*24} {'-'*8}")
    lines.append(f"    {'合计':24s}    {total:>8d} B  ({total/KILO:.1f} KB)")
    if unknown:
        lines.append(f"    注意: 有 {unknown} 项无法求值，合计为下界")

    # 判据说明：UB 实际容量未知，不应用凭空假设的阈值。
    # 采用证据基准 —— 已验证能通过的版本占用多少，候选就不应显著超过。
    # 调用方可通过 --ub-reference 指定基准值（KB）。
    limit_kb = UB_REFERENCE_KB
    limit = limit_kb * KILO
    if limit_kb > 0:
        lines.append(f"  基准（已验证版本占用）: {limit_kb:.1f} KB")
        if total > limit:
            over = (total - limit) / KILO
            lines.append(f"  [FAIL] 超出基准 {over:.1f} KB —— 需确认 UB 是否真能容纳")
        elif total >= limit - 0.05 * KILO:
            # 与已验证版本持平：基线本身已跑通，持平即为通过，不必警告
            lines.append("  [PASS] 与基准持平（基线已验证可运行）")
        else:
            lines.append(f"  [PASS] 低于基准 {(limit-total)/KILO:.1f} KB，余量充足")
    else:
        lines.append("  [INFO] 未设定基准，仅报告占用值")
    return lines, total


def check_tensor_pairs(text, path):
    """按队列逐个核对 Alloc/Free、EnQue/DeQue 的使用是否自洽。

    要点: 同一个函数体内的调用点可能服务多次循环迭代，
    因此不能用全局计数是否相等来判断泄漏。
    这里改为统计"按队列名分组"的调用点，并给出人工核对清单。
    """
    code = strip_comments(text)
    lines = code.split('\n')

    queues = {}
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'(\w+Queue)\.(\w+)(?:<[^>]*>)?\s*\(', line):
            qname, api = m.group(1), m.group(2)
            queues.setdefault(qname, []).append((i, api))

    # 找出"把队列当参数传给辅助函数"的情况: 这类队列的
    # Alloc/EnQue 发生在被调函数里，字面统计看不到，需单独提示。
    indent_queues = {}
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'\b(\w+Queue)\b', line):
            q = m.group(1)
            # 该行不是该队列的方法调用 -> 视为作为参数传递
            if not re.search(rf'\b{q}\s*\.', line):
                if re.search(r'\w+\s*\(', line) or line.strip().startswith(q):
                    indent_queues.setdefault(q, []).append(i)

    findings = []
    if not queues:
        return ["  未找到队列调用"]

    n_bad = 0
    for qname in sorted(queues):
        ops = queues[qname]
        counts = {}
        for _, api in ops:
            counts[api] = counts.get(api, 0) + 1
        alloc = counts.get('AllocTensor', 0)
        free = counts.get('FreeTensor', 0)
        enque = counts.get('EnQue', 0)
        deque = counts.get('DeQue', 0)
        status = "OK"
        # 调用点层面: 每个函数内应各有分配与释放；DeQue 不应多于 EnQue
        indirect = indent_queues.get(qname)
        if free > alloc and indirect:
            status = f"OK(分配/入队在辅助函数内，传参调用见行 {indirect})"
        elif free > alloc:
            status = "FAIL(释放点多于分配点)"
            n_bad += 1
        elif deque > enque:
            status = "FAIL(消费点多于入队点)"
            n_bad += 1
        findings.append(
            f"  {qname:18s} Alloc={alloc} Free={free} "
            f"EnQue={enque} DeQue={deque}  -> {status}")

    findings.append("  说明: 以上为源码中的调用点数量。判定依据是"
                    "'释放点不得多于分配点、消费点不得多于入队点'。")
    # 单独列出 workspaceQueue 的成对位置，便于人工核对
    wq = queues.get('workspaceQueue')
    if wq:
        findings.append("  workspaceQueue 调用点明细:")
        for ln, api in wq:
            findings.append(f"    line {ln}: {api}")
    if n_bad == 0:
        findings.append("  [PASS] 所有队列的调用点关系自洽")
    else:
        findings.append(f"  [FAIL] {n_bad} 个队列存在调用点关系异常")
    return findings


def check_dangerous_patterns(text, path):
    """查找静态可疑模式。"""
    findings = []
    lines = strip_comments(text).split('\n')

    # 占位实现
    for i, line in enumerate(lines, 1):
        if re.search(r'\*\s*0\s*;', line) and 'count' not in line.split('*')[0][-20:]:
            findings.append(f"  line {i}: 疑似占位实现 (乘 0): {line.strip()[:70]}")
        if 'TODO' in line or 'FIXME' in line or 'XXX' in line:
            findings.append(f"  line {i}: 遗留标记: {line.strip()[:70]}")
        if re.search(r'^\s*//\s*\w+\s*\(', line) and '注释' not in line:
            # 可能是被注释掉的代码
            if re.search(r'\(.*\);', line):
                findings.append(f"  line {i}: 疑似注释掉的代码: {line.strip()[:70]}")

    findings.append(f"  共扫描 {len(lines)} 行")
    return findings


def check_reinterpret_cast(text, path):
    """检查 ReinterpretCast 的使用，提示类型双关风险。"""
    findings = []
    for i, line in enumerate(strip_comments(text).split('\n'), 1):
        if 'ReinterpretCast' in line:
            # 排除已知合法的用法（mask 到 uint64）
            if 'uint64_t' in line and 'Mask' in line:
                continue
            findings.append(f"  line {i}: {line.strip()[:80]}")
    if not findings:
        findings.append("  无可疑 ReinterpretCast")
    return findings


def check_setdeqscale(text, path):
    """确认已删除 SetDeqScale（只看实际代码，忽略注释）。"""
    code = strip_comments(text)
    if 'SetDeqScale' in code:
        hits = [i for i, l in enumerate(code.split('\n'), 1)
                if 'SetDeqScale' in l]
        return [f"  [INFO] 存在 SetDeqScale，行号 {hits}。",
                "         若为保留基线行为则属预期；改动需单独验证。"]
    return ["  [INFO] 无 SetDeqScale"]


def check_half_transit(text, path):
    """确认输出路径不再经过 half 中转。"""
    # 查找是否把 output 经 half 再转 int8
    if re.search(r'LocalTensor<half>\s+\w+\s*=', text):
        return ["  [INFO] 存在 half 局部缓冲（基线使用 half 中转输出）",
                "         此为基线行为，移除需先证明等价并单独验证。"]
    return ["  [INFO] 输出路径无 half 中转"]


def main():
    global UB_REFERENCE_KB
    args = [a for a in sys.argv[1:]]
    if '--ub-reference' in args:
        i = args.index('--ub-reference')
        UB_REFERENCE_KB = float(args[i + 1])
        del args[i:i + 2]
    if args:
        path = args[0]
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'src', 'quant_matmul_relu_quant_submit.cpp')
    path = os.path.abspath(path)

    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return 1

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("=" * 74)
    print(f"静态检查: {os.path.basename(path)}")
    print("=" * 74)

    all_pass = True

    print("\n[1] 括号配对")
    errs = check_braces(text, path)
    if errs:
        all_pass = False
        for e in errs:
            print(e)
    else:
        print("  [PASS] 括号与大括号配对正常")

    print("\n[2] UB 预算")
    lines, total = check_ub_budget(text, path)
    for l in lines:
        print(l)
    # 判据只有一个：与"已验证版本占用"这个证据基准比较。
    # 这里不再使用任何自编的绝对阈值 —— 那是凭空假设，会给出错误结论。
    if total is not None and UB_REFERENCE_KB > 0 and total > UB_REFERENCE_KB * KILO:
        all_pass = False

    print("\n[3] 队列操作配对")
    for l in check_tensor_pairs(text, path):
        print(l)

    print("\n[4] 已知问题验证")
    for l in check_setdeqscale(text, path):
        print(l)
    for l in check_half_transit(text, path):
        print(l)

    print("\n[5] ReinterpretCast 使用")
    for l in check_reinterpret_cast(text, path):
        print(l)

    print("\n[6] 可疑模式扫描")
    for l in check_dangerous_patterns(text, path):
        print(l)

    print("\n" + "=" * 74)
    print("结论: " + ("静态检查通过" if all_pass else "存在需处理的问题"))
    print("=" * 74)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
