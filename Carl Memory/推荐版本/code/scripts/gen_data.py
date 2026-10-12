import numpy as np
import os
import sys

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from QuantMatmulReluQuant import impl

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


# --- Case 0 ---
np.random.seed(42 + 0)
os.makedirs("input/case0", exist_ok=True)
x1 = np.random.randint(low=int(1), high=int(20), size=(16, 32)).astype(np.int8)
x1.tofile("input/case0/x1.bin")
x2 = np.random.randint(low=int(1), high=int(20), size=(16, 32)).astype(np.int8)
x2.tofile("input/case0/x2.bin")
x1Scale = np.random.uniform(low=0.01, high=0.1, size=(16)).astype(np.float32)
x1Scale.tofile("input/case0/x1Scale.bin")
x2Scale = np.random.uniform(low=0.01, high=0.1, size=(16)).astype(np.float32)
x2Scale.tofile("input/case0/x2Scale.bin")



os.makedirs("output/golden_case0", exist_ok=True)
golden = impl(x1, x2, x1Scale, x2Scale)
if golden[0] is not None:
    golden[0].tofile("output/golden_case0/golden_y.bin")
if golden[1] is not None:
    golden[1].tofile("output/golden_case0/golden_yScale.bin")

print(f"Generated test data and golden output for 1 cases.")
