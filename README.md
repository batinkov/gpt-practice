# gpt-practice

Practice project for training a small GPT locally, running on an **AMD integrated GPU** via ROCm.

## The important bit: `HSA_OVERRIDE_GFX_VERSION`

This machine's GPU **will not work with PyTorch without an environment variable override.**

| | |
|---|---|
| GPU | AMD Radeon 860M (Ryzen AI 7 350, "Krackan Point") |
| ISA target | `gfx1152` |
| Compute units | 8 |
| System ROCm | 7.1.1 |

PyTorch's ROCm wheels are compiled ahead-of-time for a fixed list of GPU
architectures. As of `torch 2.13.0+rocm7.1` that list is:

```
gfx900 gfx906 gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1100
gfx1101 gfx1102 gfx1103 gfx1200 gfx1201 gfx1150 gfx1151
```

**`gfx1152` is not on it.** The failure mode is nastier than a missing device:
PyTorch *does* see the GPU and reports it correctly, then **segfaults on the
first real kernel launch**. Measured without the override:

```
available: True          <- reports fine
arch     : gfx1152       <- real arch, but no kernels exist for it
EXIT CODE: 139           <- SIGSEGV as soon as a matmul runs
```

So a green `torch.cuda.is_available()` proves nothing here. Only running an
actual kernel does.

The fix: `gfx1150` (Strix Point), `gfx1151` (Strix Halo) and `gfx1152`
(Krackan Point) are all **RDNA 3.5 and share the same instruction set**. So we
tell the ROCm runtime to present this GPU as `gfx1151`, and the prebuilt
`gfx1151` kernels run on it unmodified:

```
HSA_OVERRIDE_GFX_VERSION=11.5.1
```

That value lives in `.env` (gitignored — copy `.env.example` to start).
It is **required on every run**, not a one-time setup step.

> Verified 2026-08-31: matmul matched CPU to `5.49e-04` (normal fp32
> accumulation-order difference) and a backward pass ran clean. If AMD later
> ships native `gfx1152` kernels, this override becomes unnecessary but stays
> harmless.

### If it stops working

Check the real target and try the neighbouring value:

```bash
rocminfo | grep gfx                    # confirm the GPU's actual target
HSA_OVERRIDE_GFX_VERSION=11.5.0 ...    # fall back to gfx1150
```

A failure here looks like a **segfault / exit code 139** on the first kernel
launch, or a runtime error reading **`no kernel image is available for
execution on the device`**. It will *not* show up as
`torch.cuda.is_available() == False`.

## Setup

Requires system ROCm (`rocm-runtime`, `rocm-hip`, `rocminfo`) and membership in
the `video` and `render` groups — check with `id`.

```bash
uv sync
```

Torch comes from the ROCm index configured in `pyproject.toml`, not PyPI.

## Running

Always pass the env file, so the override is applied:

```bash
uv run --env-file .env python smoke_test.py
```

`smoke_test.py` is executable and its shebang does exactly that, so from the
project root this is equivalent:

```bash
./smoke_test.py
```

`smoke_test.py` verifies the GPU in two stages, because they fail
independently: first that torch *sees* the device, then that real kernels
*execute correctly* — it diffs a GPU matmul against CPU rather than just
checking nothing crashed, since a mismatched-arch kernel can return silently
wrong numbers.

## Notes and gotchas

**Memory is not what it looks like.** The GPU reports **80 GiB**, but that is
the GTT pool — shared system RAM. Dedicated VRAM is only 512 MiB. Large models
*fit* where they wouldn't on a discrete card, but run at system-RAM bandwidth,
and 8 CUs is modest. Expect this to beat CPU, not a dedicated GPU.

**`(null): No such file or directory`**, printed twice at startup, comes from
the **rocprofiler** libraries bundled in `torch/lib` — `librocprofiler-sdk.so`
and `librocprofiler-register.so` fail their symbol lookups
(`undefined symbol: rocprofiler_configure`), then report the failure with a
null filename instead of a real one. It is a cosmetic error-handling bug in
ROCm, unrelated to `HSA_OVERRIDE_GFX_VERSION` (it appears with and without it),
and nothing to do with profiling unless you actually use it.

Setting `ROCPROFILER_REGISTER_ENABLED=0` removes one of the two; the other
comes from elsewhere in the HSA tool-loading path and is not worth chasing.
Confirm with `LD_DEBUG=libs python -c 'import torch'`.

**Don't set `explicit = true`** on the ROCm index in `pyproject.toml`. Torch
depends on `triton-rocm`, which exists *only* on the PyTorch index and never on
PyPI; `explicit` hides the index from transitive dependencies and resolution
fails with a misleading `there is no version of triton-rocm==3.7.1` — even
though that wheel is on the index.

**A slow `uv sync` is usually not hung.** uv misreports
`Range requests not supported` for `download.pytorch.org` and downloads the
whole ~5.4 GB wheel just to read its metadata during resolution, with no
progress output. At ~8 MB/s that is roughly 12 minutes. To watch actual
progress: `watch -n5 'du -sh ~/.cache/uv'`.
