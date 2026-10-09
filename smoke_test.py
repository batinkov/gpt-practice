#!/usr/bin/env -S uv run --env-file .env python
"""Two-stage ROCm check: does torch SEE the GPU, and can it actually RUN on it?

The device query can pass while real kernels fail with "no kernel image is
available" -- gfx1152 (Radeon 860M) is not in PyTorch's prebuilt arch list, so
we rely on HSA_OVERRIDE_GFX_VERSION to masquerade as gfx1151. Stage 2 is the
part that actually settles it.
"""

import os
import torch

print("torch        :", torch.__version__)
print("ROCm/HIP     :", torch.version.hip)
print("HSA_OVERRIDE :", os.environ.get("HSA_OVERRIDE_GFX_VERSION", "<unset>"))

# --- Stage 1: does torch see the GPU? ---
if not torch.cuda.is_available():
    raise SystemExit("FAIL (stage 1): no GPU visible to torch")

dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(dev)
print(f"device       : {props.name} ({props.gcnArchName})")
print(f"reported mem : {props.total_memory / 1024**3:.1f} GiB")
print("stage 1 OK: GPU visible\n")

# --- Stage 2: do real kernels execute? ---
a = torch.randn(2048, 2048, device=dev)
b = torch.randn(2048, 2048, device=dev)
c = a @ b
torch.cuda.synchronize()

# Verify against CPU, so we catch silently-wrong results, not just crashes.
err = (c.cpu() - (a.cpu() @ b.cpu())).abs().max().item()
print(f"matmul max err vs CPU: {err:.2e}")
assert err < 1e-2, "matmul result diverges from CPU -- bad kernels"

# A backward pass exercises a much wider set of kernels than matmul alone.
x = torch.randn(512, 512, device=dev, requires_grad=True)
loss = torch.nn.functional.gelu(x @ x.T).sum()
loss.backward()
torch.cuda.synchronize()
assert x.grad is not None and torch.isfinite(x.grad).all(), "bad gradients"

print("stage 2 OK: matmul + backward pass both execute correctly")
print("\nPASS -- ROCm acceleration works on this GPU.")
