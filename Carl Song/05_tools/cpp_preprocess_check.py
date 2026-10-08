#!/usr/bin/env python3
"""模拟编译器预处理：剔除注释与字符串后，重置缩进并做括号配平 + 行末续行检查。
这一步能捕捉：
  1) 注释里的花括号被误读（本工程不存在，因为注释被剔除）
  2) 行末反斜杠续行（\ 导致下一行被拼接）
  3) 三字符组 / 注释截断
"""
import re
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
lines = src.split("\n")

in_block = False
out = []
for i, raw in enumerate(lines, 1):
    line = raw
    # 处理块注释
    res = ""
    j = 0
    while j < len(line):
        if in_block:
            k = line.find("*/", j)
            if k == -1:
                j = len(line)
            else:
                in_block = False
                j = k + 2
            continue
        k2 = line.find("/*", j)
        k1 = line.find("//", j)
        if k1 != -1 and (k2 == -1 or k1 < k2):
            res += line[j:k1]
            j = len(line)
            break
        if k2 != -1:
            res += line[j:k2]
            in_block = True
            j = k2 + 2
            continue
        res += line[j:]
        break
    # 去掉字符串/字符字面量
    res = re.sub(r'"(\\.|[^"\\])*"', '""', res)
    res = re.sub(r"'(\\.|[^'\\])*'", "''", res)
    out.append((i, res.rstrip()))

# 行末反斜杠续行
print("=== 行末反斜杠续行检查 ===")
found = False
for i, t in out:
    if t.endswith("\\"):
        print(f"  行{i}: {t}")
        found = True
if not found:
    print("  无")

print()
print("=== 剔除注释后的花括号配平 ===")
depth = 0
min_d = 0
first_neg = None
for i, t in out:
    d0 = depth
    depth += t.count("{") - t.count("}")
    if depth < min_d:
        min_d = depth
        if first_neg is None:
            first_neg = i
print(f"  最终深度 = {depth}   (0 = 配平)")
print(f"  最小深度 = {min_d}")
if first_neg:
    print(f"  首次越界出现在行 {first_neg}")
    for k in range(max(1, first_neg - 5), first_neg + 3):
        print(f"    {k:4d}| {out[k-1][1]}")

print()
print("=== 圆括号配平 ===")
pdepth = 0
for i, t in out:
    pdepth += t.count("(") - t.count(")")
print(f"  最终圆括号深度 = {pdepth}  (0 = 配平)")

print()
print("=== 可疑 token 扫描 ===")
sus = ["SetDeqScale", "halfLocal", "??", "\\\n"]
for i, t in out:
    for s in sus[:2]:
        if s in t:
            print(f"  行{i}: 含 '{s}'  ->  {t.strip()[:80]}")
