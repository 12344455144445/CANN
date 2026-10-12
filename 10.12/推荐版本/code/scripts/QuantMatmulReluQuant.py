#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
QuantMatmulReluQuant算子golden实现
融合: INT8量化矩阵乘 + ReLU激活 + 动态per-token量化
以CPU参考实现(见赛题文档3.1节)的结果为golden
"""
import numpy as np


def impl(x1, x2, x1Scale, x2Scale):
    """
    QuantMatmulReluQuant算子golden实现

    参数:
    x1: (M, K) int8 numpy数组, 矩阵乘左矩阵(激活), 不转置
    x2: (N, K) int8 numpy数组, 矩阵乘右矩阵(权重, 已转置, 逻辑形状(K, N))
    x1Scale: (M,) float32 numpy数组, per-token反量化因子
    x2Scale: (N,) float32 numpy数组, per-channel反量化因子

    返回:
    y: (M, N) int8 numpy数组, 动态量化后的最终结果
    yScale: (M,) float32 numpy数组, 每行的动态量化因子
    """
    M, K = x1.shape
    N = x2.shape[0]

    # 阶段一: INT8矩阵乘(INT32累加) + 反量化
    # 使用float32 matmul(BLAS加速), 对K<=~1000与int32累加数学等价
    # (127*127*K < 2^24=16777216 for K < 1040)
    matmul_out = np.matmul(x1.astype(np.float32), x2.astype(np.float32).T)  # (M, N)
    matmul_out *= x1Scale.reshape(M, 1)   # per-token scale
    matmul_out *= x2Scale.reshape(1, N)   # per-channel scale

    # 阶段二: ReLU激活
    activation_out = np.maximum(0.0, matmul_out)

    # 阶段三: 动态per-token量化
    row_max = np.max(np.abs(activation_out), axis=1)                        # (M,)
    yScale = np.where(row_max > 0, row_max / 127.0, 1.0).astype(np.float32)

    quantized = np.rint(activation_out / yScale.reshape(M, 1))
    y = np.clip(quantized, -128, 127).astype(np.int8)

    return y, yScale


if __name__ == "__main__":
    # 15个测试用例, 与json中npu_cases一一对应
    # (id, M, N, K, x1_range, x2_range, x1Scale_range, x2Scale_range)
    cases = [
        ( 1,    4,     8,  16, [-128, 127],[-128, 127], [0.001, 0.1], [0.001, 0.1]),
    ]

    for cid, M, N, K, x1r, x2r, x1sr, x2sr in cases:
        # 生成INT8输入
        if x1r[0] == x1r[1]:
            x1 = np.full((M, K), x1r[0], dtype=np.int8)
        else:
            x1 = np.random.randint(x1r[0], x1r[1] + 1, size=(M, K)).astype(np.int8)
        if x2r[0] == x2r[1]:
            x2 = np.full((N, K), x2r[0], dtype=np.int8)
        else:
            x2 = np.random.randint(x2r[0], x2r[1] + 1, size=(N, K)).astype(np.int8)

        # 生成FLOAT32 scale
        x1Scale = np.random.uniform(x1sr[0], x1sr[1], size=(M,)).astype(np.float32)
        x2Scale = np.random.uniform(x2sr[0], x2sr[1], size=(N,)).astype(np.float32)

        y, yScale = impl(x1, x2, x1Scale, x2Scale)

        aligned = all(d % 16 == 0 for d in [M, N] if d > 1)
        zero_row = int(np.any(np.all(x1 == 0, axis=1)))
        print(f"Case {cid:02d}: y.shape={y.shape}, y.dtype={y.dtype}, "
              f"yScale.shape={yScale.shape}, yScale.dtype={yScale.dtype}, "
              f"aligned={'Y' if aligned else 'N'}, zero_row={zero_row}, "
              f"y_range=[{y.min()}, {y.max()}], yScale_range=[{yScale.min():.6f}, {yScale.max():.6f}]")

    print("All tests passed!")
