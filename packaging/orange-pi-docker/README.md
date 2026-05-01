# Orange Pi Docker service

Docker Compose service used on the Orange Pi 6 Plus to run this branch by default.

It builds from Rui's patched branch:

- repo: `https://github.com/rcarmo/exo.git`
- ref: `orange/linux-vulkan-amd-tinygrad`

Runtime choices:

- `network_mode: host` so libp2p/mDNS auto-discovery works on the LAN.
- `EXO_LIBP2P_NAMESPACE=home` for the home cluster.
- Mounts the local CIX/Mali userspace and devices (`/opt/cixgpu-pro`, `/dev/mali0`, `/dev/dri`) for the Orange Pi GPU stack.

Deploy/update on the host:

```bash
cd /home/agent/services/exo
docker compose build exo
docker compose up -d --force-recreate
```

Current caveat: tinygrad defaults to CPU in the Fedora container until its OpenCL binding/runtime issue is fixed for this Mali stack.
