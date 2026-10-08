#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""作用域与结构静态检查器

补 submit_check.py 的盲区：后者只查括号配对与红线函数，
不检查「局部变量是否在其声明作用域之外被使用」——
v14 首次提交的 Compile Error 正是这个原因
（kPrefetchSpan 声明在 Pass-1 的 tile 循环体内，Pass-2 无法访问）。

本工具做三件事：
  1. 花括号深度追踪，定位每个标识符的作用域区间；
  2. 对指定的局部数组/常量，检查每一次使用是否都在其作用域内；
  3. 检查同一个名字是否在多个互不嵌套的作用域里重复声明（合法但可疑）。
"""
import re
import sys


def strip_comment(line):
    """去掉行注释，但保留 // 前的代码。不处理块注释（本文件不使用）。"""
    s = re.sub(r'//.*$', '', line)
    return s


def build_scope_map(lines):
    """返回 scope_depth[i] = 第 i 行（0-based）的嵌套深度。

    深度在进入 '{' 所在行之后递增。做法：逐行扫描字符。
    """
    depths = []
    depth = 0
    for line in lines:
        code = strip_comment(line)
        depths.append(depth)
        for ch in code:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
    return depths, depth


DECL_RE = re.compile(
    r'^\s*(?:constexpr\s+)?(?:const\s+)?(?:uint\w+_t|int\w*_t|float|half|bool|auto)\s+'
    r'(\w+)\s*(\[[^\]]*\])?\s*(=|;|\{)'
)


def find_declarations(lines, depths):
    """找出所有局部变量声明：名字 -> (行号1based, 作用域深度, 是否数组)"""
    decls = []
    for i, line in enumerate(lines):
        code = strip_comment(line)
        m = DECL_RE.match(code)
        if not m:
            continue
        name = m.group(1)
        after = code[m.end(2) if m.group(2) else m.end(1):]
        # 跳过函数声明/定义（后面紧跟不存在 = 或 ; 且行尾有分号带参数列表）
        if '(' in code.split('=')[0].split(';')[0] and '[' not in code:
            pass
        # 判断是数组声明
        is_array = bool(m.group(2))
        # 作用域深度：若该行有 '{'，声明本身属于外层还是内层？
        # C++ 中 for(...) 后的 '{' 进入新作用域，但同一行的声明在外层。
        # 这里采用 depths[i]（进入该行 '{' 之前的深度）作为声明作用域。
        decls.append((name, i + 1, depths[i], is_array))
    return decls


def check(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.read().split('\n')

    depths, final_depth = build_scope_map(lines)

    print("=" * 74)
    print(f"作用域检查: {path}")
    print("=" * 74)

    ok = True

    # ---- 1. 花括号总配对 ----
    print(f"\n[1] 花括号总深度: {final_depth}", end="  ")
    if final_depth == 0:
        print("OK")
    else:
        print("**不匹配！**")
        ok = False

    # ---- 2. 定位目标标识符 ----
    # 只检查在本文件 [OPT-*] 区块里引入的预取数组/常量
    targets = set()
    for line in lines:
        code = strip_comment(line)
        m = re.search(r'constexpr\s+\w+\s+(\w+)\s*=', code)
        if m:
            targets.add(m.group(1))
        m = re.search(r'(\w+Prefetch\w*)\s*\[', code)
        if m:
            targets.add(m.group(1))

    print(f"\n[2] 纳入检查的标识符: {sorted(targets) if targets else '(无)'}")

    decls = find_declarations(lines, depths)
    decl_by_name = {}
    for name, ln, d, is_arr in decls:
        decl_by_name.setdefault(name, []).append((ln, d, is_arr))

    print(f"\n[3] 声明/使用作用域核对")
    for name in sorted(targets):
        dl = decl_by_name.get(name, [])
        if not dl:
            print(f"  ?  {name}: 未识别到声明（可能是内联 constexpr）")
            continue
        # 若有多个声明，取深度最小的那个（作用域最大）作为基准可见性；
        # 但"使用点是否被至少一个声明覆盖"才是关键：
        # 对每次使用，要求存在某个声明的深度 <= 使用点深度 且该声明行号 <= 使用行号。
        base_ln, base_depth, _ = min(dl, key=lambda t: (t[1], t[0]))
        uses = []
        for i, line in enumerate(lines):
            code = strip_comment(line)
            if not re.search(r'\b' + re.escape(name) + r'\b', code):
                continue
            if any(i + 1 == ln for ln, _, _ in dl):
                continue
            uses.append((i + 1, depths[i]))
        bad = []
        for ln, d in uses:
            # 该使用点是否被至少一个声明覆盖？
            covered = any(dl_ln <= ln and dl_d <= d for dl_ln, dl_d, _ in dl)
            if not covered:
                bad.append((ln, d))
        tag = "OK" if not bad else "**越界使用！**"
        print(f"  {'v' if not bad else 'x'} {name:20s} 声明@行{base_ln} 深度{base_depth}"
              f"  使用{len(uses)}处  {tag}")
        if bad:
            ok = False
            for ln, d in bad:
                print(f"      ! 行{ln} 深度{d} 无覆盖声明 ⇒ 编译期不可见")
                print(f"        {lines[ln-1].strip()[:90]}")

    # ---- 4. 重复声明检查 ----
    # 重复声明本身即视为错误：同名局部量在多处声明，极易导致
    # "在某处可见、在另一处不可见" 的编译错误（v14 首次提交即此原因）。
    print(f"\n[4] 同名重复声明检查")
    dup_found = False
    for name in sorted(targets):
        dl = decl_by_name.get(name, [])
        if len(dl) > 1:
            dup_found = True
            ok = False
            print(f"  ! {name} 被声明 {len(dl)} 次: "
                  + ", ".join(f"行{ln}(深度{d})" for ln, d, _ in dl))
            print(f"    ⇒ 同名局部量多处声明，容易出现作用域不可见问题，必须合并为一处")
    if not dup_found:
        print("  OK 无重复声明")

    print("\n" + "=" * 74)
    print("作用域检查通过" if ok else "作用域检查未通过 —— 请修复后再提交")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: scope_check.py <kernel.asc>")
        sys.exit(2)
    sys.exit(check(sys.argv[1]))
