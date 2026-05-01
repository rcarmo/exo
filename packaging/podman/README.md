# Podman

This setup runs exo as a Linux tinygrad worker under Podman.

The image uses the base Linux dependency set instead of the `cpu` extra so it
does not install or start the Linux MLX runtime. By default the compose file
sets `EXO_TINYGRAD_DEVICE=CPU`, which forces the tinygrad runner onto plain CPU.
Set `EXO_TINYGRAD_DEVICE=CL` plus `OPENCL_PATH=/path/to/libOpenCL.so` to run
through tinygrad's OpenCL device. `EXO_TINYGRAD_DEVICE=WEBGPU` is accepted, but
requires a compatible Dawn `libwebgpu_dawn.so` exposed via `WEBGPU_PATH`; the
Python `wgpu` package's `libwgpu_native` is not ABI-compatible with tinygrad's
Dawn bindings. Set `EXO_TINYGRAD_DEVICE=VULKAN`, or use the Vulkan override file
below, to run through tinygrad's Vulkan device when the installed tinygrad build
provides one.

Network discovery requires host networking because exo's libp2p layer uses local
network discovery and listens on the host interface. The compose file sets:

```yaml
network_mode: host
```

The container also pins libp2p to TCP port `52416` by default so the host
firewall can be configured predictably. The API continues to listen on `52415`.
Set `EXO_LIBP2P_NAMESPACE` to match the other nodes in the cluster; the compose
file defaults it to `home`.

On Fedora Silverblue, allow mDNS and the pinned libp2p port on the active zone:

```bash
sudo firewall-cmd --permanent --add-service=mdns
sudo firewall-cmd --permanent --add-port=52416/tcp
sudo firewall-cmd --reload
```

Start it from this directory:

```bash
podman compose up -d --build
```

To run the tinygrad worker through Vulkan, the container needs access to the
host DRM device and a tinygrad runtime that includes Vulkan support:

```bash
podman compose -f compose.yaml -f compose.vulkan.yaml up -d --build
```

The Vulkan override sets `EXO_TINYGRAD_DEVICE=VULKAN` and mounts `/dev/dri` into
the container.
