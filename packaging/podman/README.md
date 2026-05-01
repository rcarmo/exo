# Podman

This setup runs exo as a Linux CPU node under Podman.

Network discovery requires host networking because exo's libp2p layer uses local
network discovery and listens on the host interface. The compose file sets:

```yaml
network_mode: host
```

The container also pins libp2p to TCP port `52416` by default so the host
firewall can be configured predictably. The API continues to listen on
`52415`.

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
