# Orange Linux tinygrad/Vulkan/AMD fork notes

Date: 2026-05-01
Host: Orange Pi 6 Plus (`aarch64`, Debian/Trixie-derived Orange Pi OS)

## Local accelerator surface

- Vulkan: `Mali-G720-Immortalis`, proprietary ARM driver, Vulkan API 1.3.296.
- OpenCL: `Mali-G720-Immortalis r0p0`, OpenCL 3.0, 10 compute units, ~14.85 GiB global memory.
- tinygrad in this environment selects `Device.DEFAULT == "CL"`, so the immediate non-CPU target is the Mali OpenCL backend. Vulkan remains worth tracking through tinygrad/WebGPU/Vulkan work, but CL is the backend that starts here today.

## Forks and patches checked

### `exo-explore/exo` upstream

Cloned as `origin` and used as the base for branch `orange/linux-vulkan-amd-tinygrad`.

### `triko88/exo`, branch `feature/linux-support`

Source of upstream PR [#1660](https://github.com/exo-explore/exo/pull/1660), "feat(Linux): Tinygrad runner for LLM inference on Linux".

Merged/adapted:

- `TinygradInstance` and Linux default instance metadata.
- tinygrad model config, tokenizer helpers, generator, cache, quantization, layers, weight loader, and runner.
- Dashboard/API placement changes so Linux placement can return `TinygradInstance`.
- Unit tests for tinygrad import, cache, layers, quantization, sampling, weight loading, generation, and placement.

Conflict policy: preserved current upstream v1 API/dashboard/Rust networking behavior where PR #1660 was based on older paths. In particular, kept current `Keypair.generate()`/`to_node_id()` because the PR's `generate_ed25519()`/`to_peer_id()` does not exist in the current PyO3 binding.

### `Scottcjn/exo-cuda`, branch `tinygrad-cuda`

Referenced by upstream issue [#1039](https://github.com/exo-explore/exo/issues/1039), "NVIDIA CUDA Support via Tinygrad - Working Fork Available".

Useful ideas found:

- CUDA tinygrad runner validation on NVIDIA V100/M40.
- OpenCL/older AMD compatibility commits around `USE_FP32=1`, especially FirePro D500/older OpenCL devices that cannot reliably run fp16/bf16 kernels.

Adapted:

- Ported the `USE_FP32=1` dtype-normalization idea into `src/exo/worker/engines/tinygrad/weight_loader.py`, before tensors are transferred to the active tinygrad device.

Not merged wholesale:

- The fork is based on the old `exo/inference/...` tree and contains large unrelated Interweave/benchmark/server work, so direct merge would regress the current v1 architecture.

### `hawkymisc/exo-x`

Search result described it as a fork with "AMD iGPU support". It has no useful common history with current upstream and appears to be a broad old-tree rewrite/rebrand.

Not merged:

- No clean merge base with current upstream.
- AMD/GPU changes are packaged into an incompatible old setup/config tree rather than isolated patches for the current v1 worker/runner architecture.

### Windows/WIP forks (`KLM-corporation/exo-windows`, `peterxing/exo` PR #1289)

Checked because they mention cross-platform/GPU support, but they are Windows-oriented and do not add useful Mali/OpenCL/Vulkan/Linux tinygrad patches for this host.

## Verified locally

```bash
uv sync --no-dev
cd dashboard && npm install && npm run build
cd ..
uv run pytest src/exo/master/tests/test_placement.py \
  src/exo/worker/tests/unittests/test_runner/test_runner_import.py \
  src/exo/worker/tests/unittests/test_tinygrad -q
uv run exo --api-port 52420 --offline
curl http://127.0.0.1:52420/instance/placement?model_id=mlx-community%2FLlama-3.2-1B-Instruct-4bit
```

Results:

- Tests: `93 passed, 2 skipped, 8 deselected`.
- Server starts and serves `/node_id` and `/state`.
- Placement for `mlx-community/Llama-3.2-1B-Instruct-4bit` returns `TinygradInstance` with one runner.

## Next steps

- Try an actual small model download/load/generation on the Mali OpenCL backend.
- If CL fp16/bf16 kernels fail, retry with `USE_FP32=1`.
- Keep watching tinygrad Vulkan/WebGPU backend support; on this board OpenCL is currently the practical path.
