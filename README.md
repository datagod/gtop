# gtop++

GPU-first system monitor forked from [btop++](https://github.com/aristocratos/btop).

**gtop++** keeps btop++'s terminal UI and extends it toward detailed GPU workload
monitoring: per-GPU stats today, per-process VRAM/SM%, workload categories, and
load-balance hints in upcoming releases.

The installed binary and config paths use **`gtop`** (e.g. run `gtop`, config in
`~/.config/gtop/gtop.conf`).

## Quick start

```bash
git clone https://github.com/datagod/gtop.git
cd gtop
make GPU_SUPPORT=true
sudo make install
sudo make setcap   # recommended for Intel GPU + /proc access
gtop
```

Requires GCC 14+ or Clang 19+ on Linux. GPU monitoring needs dynamic linking
(NVML for NVIDIA, ROCm SMI for AMD, sysfs for Intel).

## GPU-first layout

On first run (no existing config), gtop++ shows all detected GPU boxes plus a
compact CPU, memory, and process view. Press `p` to cycle presets; use the menu
to toggle boxes.

## Upstream

To pull security fixes from btop++:

```bash
git fetch upstream
git merge upstream/main
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).