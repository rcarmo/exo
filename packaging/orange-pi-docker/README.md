# Orange Pi Docker service

Docker Compose service used on the Orange Pi 6 Plus to run this branch by default.

It builds from Rui's patched branch:

- repo: `https://github.com/rcarmo/exo.git`
- ref: `orange/linux-vulkan-amd-tinygrad`

Runtime choices:

- `network_mode: host` so libp2p/mDNS auto-discovery works on the LAN.
- `EXO_LIBP2P_NAMESPACE=home` for the home cluster.
- `EXO_TINYGRAD_DEVICE=WEBGPU` to use tinygrad's WebGPU runtime over Dawn/Vulkan.
- `EXO_TINYGRAD_CACHE_DEQUANTIZED_WEIGHTS=0` to avoid excessive memory pressure on this board.
- `WEBGPU_BACKEND=WGPUBackendType_Vulkan` and `VK_ICD_FILENAMES=/etc/vulkan/icd.d/mali.json` select the Mali Vulkan stack.
- Mounts the local CIX/Mali userspace and devices (`/opt/cixgpu-pro`, `/dev/mali0`, `/dev/dri`) for the Orange Pi GPU stack.

## Dawn dependency

Tinygrad's WebGPU runtime expects Dawn's `libwebgpu_dawn.so`; the Python `wgpu` package's `libwgpu_native` is not ABI-compatible.

Before deploying from this directory, download the aarch64 Dawn binary:

```bash
mkdir -p dawn
curl -fsSL \
  -o dawn/libwebgpu_dawn.so \
  https://github.com/wpmed92/pydawn/releases/download/v0.3.0/libwebgpu_dawn_aarch64.so
```

This is mounted read-only into the container at `/opt/dawn/libwebgpu_dawn.so`.

## Deploy/update on the host

```bash
cd /home/agent/services/exo
docker compose build exo
docker compose up -d --force-recreate
```

## Validation

```bash
docker exec exo-cpu bash -lc 'cd /home/exo/exo && uv run python - <<"PY"
from tinygrad.device import Device
from tinygrad.tensor import Tensor
print(Device["WEBGPU"])
print((Tensor([1,2,3], device="WEBGPU") + 1).realize().numpy())
PY'
```

Expected tensor output:

```text
[2 3 4]
```

On 2026-05-01, `mlx-community/Llama-3.2-3B-Instruct-8bit` loaded successfully via `TinygradInstance` + `WEBGPU` and returned a 1-token response in ~10s, and an 8-token response in ~16s.
