# -*- coding: utf-8 -*-
"""
sig_check.py —— 形参/实参个数一致性自检

本轮（v17c）新增了一个 kernel 形参 mBlocksPerGroup，并同时改了 fuse dkernel 的
launch 实参表。这类改动一旦漏改一处，编译期就会报错（而且我们本地无法编译）——
正是 E19 想防的事。本脚本用纯结构匹配把这类错误挡在前面：

  A) `quant_matmul_relu_quant_cube( <实参表> )` 的实参个数
     == `quant_matmul_relu_quant_cube( <形参表> )` 形参个数
  B) `quant_matmul_relu_quant_fused<<<...>>>( <实参表> )` 的实参个数
     == fused kernel 形参个数
  C) 两个 kernel（cube 辅助函数 / fused __global__）的形参名集合
     与调用点实参**没有明显错位**（逐参数名在实参表里出现，除字面量）
"""
import re
import sys


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    out = []
    for line in text.splitlines():
        # 去掉行注释（本文件里没有字符串字面量含 //，安全）
        idx = line.find("//")
        out.append(line if idx < 0 else line[:idx])
    return "\n".join(out)


def split_args(s):
    """按顶层逗号切分实参/形参表（考虑 <> () [] 嵌套）"""
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return [a for a in args if a]


def paren_body(text, open_paren_idx):
    """返回以 text[open_paren_idx]=='(' 起始的顶层括号内容"""
    depth, j = 0, open_paren_idx
    while j < len(text):
        if text[j] in "(<[":
            depth += 1
        elif text[j] in ")>]":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return text[open_paren_idx + 1:j], j


def func_params(text, name, is_global=False):
    """取普通函数/__global__ 函数的形参表（返回参数名列表）"""
    for m in re.finditer(re.escape(name) + r"\s*\(", text):
        head = text[max(0, m.start() - 300):m.start()]
        if is_global and "__global__" not in head:
            continue
        if (not is_global) and "__global__" in head:
            continue
        body, _ = paren_body(text, m.end() - 1)
        return split_args(body)
    return None


def check(path):
    raw = open(path, "r", encoding="utf-8", newline="").read()
    text = strip_comments(raw)
    ok = True

    # ---------- A) cube 辅助函数 ----------
    cube_name = "quant_matmul_relu_quant_cube"
    cube_params = func_params(text, cube_name)
    hits = list(re.finditer(re.escape(cube_name) + r"\s*\(", text))
    if len(hits) < 2:
        print("[FAIL] 找不到 %s 的调用点（只匹配到 %d 处）" % (cube_name, len(hits)))
        return False
    cube_args, _ = paren_body(text, hits[1].end() - 1)
    cube_args = split_args(cube_args)

    print("A) %s" % cube_name)
    print("   形参 %d 个: %s" % (len(cube_params),
                                 ", ".join(p.split()[-1] for p in cube_params)))
    print("   实参 %d 个: %s" % (len(cube_args), ", ".join(cube_args)))
    if len(cube_params) != len(cube_args):
        print("   [FAIL] 形参与实参个数不一致")
        ok = False
    else:
        print("   OK 个数一致")

    # ---------- B) fused kernel ----------
    fused = "quant_matmul_relu_quant_fused"
    fparams = func_params(text, fused, is_global=True)
    if fparams is None:
        print("[FAIL] 找不到 fused kernel 定义")
        return False
    m = re.search(re.escape(fused) + r"\s*<<<", text)
    if m is None:
        print("[FAIL] 找不到 fused 的 launch")
        return False
    i = text.index(">>>", m.end()) + 3
    while text[i] != "(":
        i += 1
    fargs, _ = paren_body(text, i)
    fargs = split_args(fargs)

    print("B) %s<<<...>>>" % fused)
    print("   形参 %d 个" % len(fparams))
    print("   实参 %d 个: %s" % (len(fargs), ", ".join(fargs)))
    if len(fparams) != len(fargs):
        print("   [FAIL] 形参与实参个数不一致")
        ok = False
    else:
        print("   OK 个数一致")
        print("   逐位核对:")
        for k, (p, a) in enumerate(zip(fparams, fargs)):
            pn = p.replace("*", " ").replace("&", " ").split()
            pn = pn[-1] if pn else p
            match = (pn == a) or a.isdigit() or (pn in a)
            print("     %2d  %-30s <- %-22s %s"
                  % (k, p.strip(), a, "" if match else "<-- 名字不匹配，请人工确认"))
    return ok


def main():
    paths = sys.argv[1:] or [
        "records/experiments/v17b-overlap/kernel_v17b_overlap.asc",
        "records/experiments/v17c-parallel/kernel_v17c_parallel.asc",
    ]
    all_ok = True
    for p in paths:
        print("=" * 74)
        print("签名一致性: %s" % p)
        print("=" * 74)
        if not check(p):
            all_ok = False
        print()
    print("=" * 74)
    print("结论：%s" % ("全部一致" if all_ok else "存在不一致，禁止提交"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
