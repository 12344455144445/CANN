"""
精度守卫：逐函数比对两个 kernel 版本，确保"精度关键路径"未被改动。

背景
----
交接笔记明确写道：
  "接手者也应保留浮点乘法顺序、除法修正和半整数舍入校正；
   此前这些部分的'数学等价'替换曾产生精度错误。"

本工具把这条要求变成可执行的检查：把精度关键函数抽出来逐字节比对，
任何改动都必须显式声明并给出理由，否则判定为 FAIL。

用法
----
  # 比对两个版本（默认检查关键函数）
  python precision_guard.py --baseline a.asc --candidate b.asc

  # 生成某版本的关键函数指纹（用于记录）
  python precision_guard.py --fingerprint a.asc

  # 列出候选版本相对基线改动过的函数
  python precision_guard.py --baseline a.asc --candidate b.asc --list-all
"""

import argparse
import hashlib
import os
import re
import sys


# ---------------------------------------------------------------------------
# 精度关键函数：这些函数的函数体必须与基线完全一致。
# 允许的差异只有空白与注释（本工具会先剥离）。
# ---------------------------------------------------------------------------
CRITICAL_FUNCTIONS = [
    # 半整数舍入校正：rint 边界上决定最终 int8 取值，最敏感
    "QmqQuantizeNearHalf",
    # 除法修正：在浮点除法结果附近搜索更优商
    "QmqCorrectDiv",
    # 向量除法（含尾块处理方式，影响 off-by-one）
    "QmqDiv",
]

# 数值路径函数：不要求逐字节一致，但一旦改动必须人工确认是否
# 影响浮点乘法顺序。本工具会列出它们并给出 diff 摘要。
NUMERIC_PATH_FUNCTIONS = [
    "quant_matmul_relu_quant_vector",
    "quant_matmul_relu_quant_cube",
]

# 这些模式出现在"精度关键"函数里就视为违规（哪怕函数整体通过了比对，
# 也要检查是否引入了危险等价替换）。
DANGEROUS_PATTERNS = {
    "CAST_NONE_on_int_to_float": (
        r'Cast\s*\(\s*\w+\s*,\s*\w+\s*,\s*RoundMode::CAST_NONE',
        "整数转浮点使用 CAST_NONE —— 累加值超过 2^24 时会丢精度，"
        "而 numpy 参考实现用就近偶数舍入",
    ),
    "reciprocal_mul_div": (
        r'Rec\s*\(|Reciprocal\s*\(',
        "用倒数乘法替代除法 —— 实测在 655 万元素中会产生不匹配",
    ),
    "added_rounding_step": (
        r'Adds\s*\([^)]*0\.5f',
        "出现 0.5 加法 —— 可能是把 rint 替换成了 floor(x+0.5)，"
        "该替换在整数部分为奇数时必然出错",
    ),
}


def strip_comments_and_space(text):
    """移除注释与所有空白，保留语义等价的代码正文。

    这样比对时忽略格式差异，只关注实际计算内容。
    """
    out = []
    in_block = False
    i = 0
    n = len(text)
    while i < n:
        if in_block:
            end = text.find('*/', i)
            if end == -1:
                break
            i = end + 2
            in_block = False
            continue
        if text.startswith('//', i):
            end = text.find('\n', i)
            i = n if end == -1 else end
            continue
        if text.startswith('/*', i):
            in_block = True
            i += 2
            continue
        if text[i] == '"':
            end = text.find('"', i + 1)
            i = n if end == -1 else end + 1
            # 字符串字面量保留内容（可能有分隔符意义）
            continue
        ch = text[i]
        if not ch.isspace():
            out.append(ch)
        i += 1
    return ''.join(out)


def extract_functions(text):
    """从 C++ 源码中抽取 top-level 函数，返回 {name: body}。

    用大括号配平来界定函数体，能处理嵌套的 if/for/while 与 lambda。
    """
    funcs = {}
    # 匹配： 返回类型 名字 ( 参数 ) { ...
    pattern = re.compile(
        r'(?:^|\n)\s*(?:__aicore__\s+)?(?:inline\s+)?'
        r'(?:[A-Za-z_][\w:<>,\s\*&]*?\s+)?'
        r'([A-Za-z_]\w*)\s*\([^;{]*?\)\s*(?:const\s*)?\{',
        re.MULTILINE)

    for m in pattern.finditer(text):
        name = m.group(1)
        if name in ('if', 'for', 'while', 'switch', 'else', 'return'):
            continue
        brace_start = text.find('{', m.end() - 1)
        if brace_start == -1:
            continue
        depth = 0
        i = brace_start
        while i < len(text):
            c = text[i]
            if text.startswith('//', i):
                j = text.find('\n', i)
                i = len(text) if j == -1 else j
                continue
            if text.startswith('/*', i):
                j = text.find('*/', i)
                i = len(text) if j == -1 else j + 2
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[brace_start:i + 1]
        # 同名函数保留第一个（避免嵌套 lambda 覆盖）
        funcs.setdefault(name, body)
    return funcs


def code_of(funcs, name):
    body = funcs.get(name)
    if body is None:
        return None
    return strip_comments_and_space(body)


def digest(code):
    return hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]


def check_dangerous(text, label):
    """在指定文本范围内扫描危险模式。"""
    hits = []
    for key, (pat, why) in DANGEROUS_PATTERNS.items():
        for m in re.finditer(pat, text):
            line = text[:m.start()].count('\n') + 1
            hits.append(f"    [{key}] 行 {line}: {why}")
    return hits


def fingerprint(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    funcs = extract_functions(text)
    print(f"版本指纹: {os.path.basename(path)}")
    print(f"  提取到 {len(funcs)} 个函数")
    print()
    for name in CRITICAL_FUNCTIONS:
        code = code_of(funcs, name)
        if code is None:
            print(f"  {name:28s} 缺失!")
        else:
            print(f"  {name:28s} {digest(code)}  ({len(code)} 字符)")
    print()
    print("  数值路径函数:")
    for name in NUMERIC_PATH_FUNCTIONS:
        code = code_of(funcs, name)
        if code is None:
            print(f"    {name:28s} 缺失!")
        else:
            print(f"    {name:28s} {digest(code)}  ({len(code)} 字符)")


def compare(baseline_path, candidate_path, list_all):
    print("=" * 74)
    print("精度守卫检查")
    print("=" * 74)
    print(f"  基线   : {baseline_path}")
    print(f"  候选   : {candidate_path}")
    print()

    base_text = open(baseline_path, encoding='utf-8', errors='replace').read()
    cand_text = open(candidate_path, encoding='utf-8', errors='replace').read()
    base_funcs = extract_functions(base_text)
    cand_funcs = extract_functions(cand_text)

    ok = True

    # ---------- 1. 精度关键函数必须逐字节一致 ----------
    print("[1] 精度关键函数（必须与基线完全一致）")
    for name in CRITICAL_FUNCTIONS:
        b = code_of(base_funcs, name)
        c = code_of(cand_funcs, name)
        if b is None:
            print(f"  ?  {name}: 基线中未找到，跳过")
            continue
        if c is None:
            print(f"  FAIL {name}: 候选中缺失")
            ok = False
            continue
        if b == c:
            print(f"  PASS {name}  ({digest(c)})")
        else:
            print(f"  FAIL {name} 已被修改")
            print(f"       基线指纹 {digest(b)}, 候选指纹 {digest(c)}")
            ok = False
    print()

    # ---------- 2. 危险模式扫描（只看候选相对基线"新增"的） ----------
    print("[2] 危险模式扫描（仅报告候选新增的，基线已有不算）")
    code_cand = ''.join(strip_comments_and_space(v) for v in cand_funcs.values())
    code_base = ''.join(strip_comments_and_space(v) for v in base_funcs.values())

    new_hits = []
    for key, (pat, why) in DANGEROUS_PATTERNS.items():
        n_base = len(re.findall(pat, code_base))
        n_cand = len(re.findall(pat, code_cand))
        if n_cand > n_base:
            new_hits.append(
                f"    [{key}] 基线 {n_base} 处 -> 候选 {n_cand} 处 (+{n_cand - n_base})")
            new_hits.append(f"        {why}")
    if new_hits:
        for h in new_hits:
            print(h)
        print("  -> FAIL 候选引入了基线没有的危险模式")
        ok = False
    else:
        print("  PASS 未新增危险模式")
    print()

    # ---------- 3. 数值路径函数改动摘要 ----------
    print("[3] 数值路径函数（不强制一致，但需确认浮点乘法顺序未变）")
    for name in NUMERIC_PATH_FUNCTIONS:
        b = code_of(base_funcs, name)
        c = code_of(cand_funcs, name)
        if b is None or c is None:
            print(f"  ?  {name}: 一方缺失")
            continue
        if b == c:
            print(f"  PASS {name} 未改动")
        else:
            print(f"  NOTE {name} 已改动（{len(b)} -> {len(c)} 字符）")
            # 检查浮点乘法顺序相关的关键序列是否保留。
            #
            # [GUARD-EXT] 关于标量预取形式的等价识别：
            # 允许把 `Muls(..., x1ScaleLocal.GetValue(row), ...)` 改写为
            # `Muls(..., x1Prefetch[row], ...)`，前提是：
            #   (a) 同一函数内仍存在对 x1ScaleLocal 的 GetValue 预读取值循环；
            #   (b) 该预取数组确实被用作 Muls 的乘数。
            # 两个条件同时满足才放行，避免"把乘法整个删掉"也被误判为等价。
            has_x1_prefetch = bool(re.search(
                r'x1Prefetch\w*\s*\[[^\]]*\]\s*=\s*x1ScaleLocal\.GetValue', c))
            uses_x1_prefetch = bool(re.search(
                r'Muls\s*\([^;]*x1Prefetch\w*\s*\[', c))
            # 关键：必须核实"预取—使用"是成对的，且不存在被替换成常数的 Muls。
            # 只要有任意一处 Muls 的第二乘数既不是 x1ScaleLocal.GetValue、
            # 也不是 x1Prefetch* 数组，就说明乘法被改写了，必须报差异。
            muls_operands = re.findall(r'Muls\s*\(([^;]*?)\)\s*;', c)
            bad_muls = False
            for args in muls_operands:
                # 取第三个逗号分隔参数（乘数）
                parts = [p.strip() for p in args.split(',')]
                if len(parts) >= 3:
                    operand = parts[2]
                    if ('x1ScaleLocal.GetValue' not in operand
                            and 'x1Prefetch' not in operand):
                        bad_muls = True
            if bad_muls:
                has_x1_prefetch = False
            seqs = [
                ("x1Scale 乘法", r'Muls\([^;]*x1ScaleLocal\.GetValue'),
                ("x2Scale 乘法", r'Mul\([^;]*x2ScaleLocal'),
                ("ReLU", r'Maxs\([^;]*0\.0f'),
                ("CAST_RINT 转整数", r'CAST_RINT'),
                ("CAST_FLOOR", r'CAST_FLOOR'),
            ]
            for label, pat in seqs:
                in_b = bool(re.search(pat, b))
                in_c = bool(re.search(pat, c))
                if (label == "x1Scale 乘法" and in_b and not in_c
                        and has_x1_prefetch and uses_x1_prefetch):
                    # 预取形式：等价改写，放行
                    print(f"       {label:18s} 基线={in_b} 候选={in_c}"
                          f"  OK(标量预取等价形式)")
                    continue
                mark = "OK" if in_b == in_c else "**差异**"
                print(f"       {label:18s} 基线={in_b} 候选={in_c}  {mark}")
                if in_b != in_c:
                    ok = False
    print()

    # ---------- 4. 全函数改动清单 ----------
    if list_all:
        print("[4] 全部函数改动清单")
        names = sorted(set(base_funcs) | set(cand_funcs))
        for name in names:
            b = code_of(base_funcs, name)
            c = code_of(cand_funcs, name)
            if b == c:
                continue
            if b is None:
                print(f"  新增 {name}")
            elif c is None:
                print(f"  删除 {name}")
            else:
                print(f"  修改 {name}")
        print()

    print("=" * 74)
    if ok:
        print("结论: PASS —— 精度关键路径未被改动")
    else:
        print("结论: FAIL —— 存在精度风险改动，必须逐项确认")
    print("=" * 74)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline")
    ap.add_argument("--candidate")
    ap.add_argument("--fingerprint")
    ap.add_argument("--list-all", action="store_true")
    args = ap.parse_args()

    if args.fingerprint:
        fingerprint(args.fingerprint)
        return 0
    if args.baseline and args.candidate:
        return compare(args.baseline, args.candidate, args.list_all)

    ap.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
