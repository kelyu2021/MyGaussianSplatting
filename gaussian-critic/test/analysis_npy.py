import numpy as np

path = "test/0001_back.npy"
d = np.load(path).astype(np.float32)

print(f"=== {path} ===")
print(f"Shape:            {d.shape}")
print(f"Dtype:            {d.dtype}")
print()

print("--- All pixels ---")
print(f"Min:              {d.min():.6f}")
print(f"Max:              {d.max():.6f}")
print(f"Mean:             {d.mean():.6f}")
print(f"Std:              {d.std():.6f}")
print()

neg   = (d  < 0).sum()
zero  = (d == 0).sum()
pos   = (d  > 0).sum()
total = d.size
print(f"Negative pixels:  {neg}  ({100*neg/total:.2f}%)")
print(f"Zero pixels:      {zero}  ({100*zero/total:.2f}%)")
print(f"Positive pixels:  {pos}  ({100*pos/total:.2f}%)")
print()

valid = d[d > 0]
print("--- Positive (valid) pixels only ---")
print(f"Count:            {valid.size}")
print(f"Min:              {valid.min():.6f}")
print(f"p1:               {np.percentile(valid,  1):.6f}")
print(f"p5:               {np.percentile(valid,  5):.6f}")
print(f"p25:              {np.percentile(valid, 25):.6f}")
print(f"Median:           {np.percentile(valid, 50):.6f}")
print(f"p75:              {np.percentile(valid, 75):.6f}")
print(f"p95:              {np.percentile(valid, 95):.6f}")
print(f"p99:              {np.percentile(valid, 99):.6f}")
print(f"Max:              {valid.max():.6f}")
print()

# Tiny near-zero outliers that would blow up inverse depth
epsilon = 0.01
tiny = (valid < epsilon).sum()
print(f"Valid pixels with depth < {epsilon} (outliers): {tiny}  ({100*tiny/valid.size:.4f}%)")
print()

inv = 1.0 / valid
print("--- Inverse depth of valid pixels ---")
print(f"Min:              {inv.min():.6f}")
print(f"p99:              {np.percentile(inv, 99):.6f}")
print(f"p99.9:            {np.percentile(inv, 99.9):.6f}")
print(f"Max:              {inv.max():.2f}  (from depth={valid.min():.2e})")
print(f"  → ratio max/p99: {inv.max() / np.percentile(inv, 99):.1f}x")
