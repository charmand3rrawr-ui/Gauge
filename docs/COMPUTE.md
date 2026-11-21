# Compute: sensing GPUs and using the machines you own

## Sensing GPUs

`gauge compute scan` lists the CPU and every GPU it can find, from NVIDIA, AMD or Intel, using
the tools that exist on the machine (nvidia-smi, rocm-smi, Windows CIM, lspci), then checks
which numpy-compatible Python backend could drive each one:

| vendor | backend GaugeLM can use | install |
| --- | --- | --- |
| NVIDIA | CuPy on CUDA | `pip install cupy-cuda12x` (CUDA 11 cards: `cupy-cuda11x`) |
| AMD | CuPy on ROCm (Linux / WSL2) | `pip install cupy-rocm-6-0` |
| Intel, Apple | none yet | reported; the CPU path is used |

Hardware without its backend is listed with the exact install command. `--apply` saves the best
device into `native.device`. `gauge compute watch` polls until a usable GPU appears, for example
an external GPU plugged in, and switches to it. The daemon does the same scan every cycle
(`compute.auto_device`), so GaugeLM moves to a GPU as soon as one is there.

## Using your other machines

Gauge will use the other computers you own, and only those. On each of them:

```powershell
gauge compute token                       # once, on the main machine: makes a shared secret
gauge worker --token <secret> --name "office-pc"   # on each other machine you own
```

A worker is a small HTTP server (standard library) that, behind the token, reports its devices,
serves its model (GaugeLM or a local Ollama) and runs training shards. It announces itself on
the LAN by UDP broadcast. On the main machine:

```powershell
gauge compute discover        # listen for announcements; register what answers with the right token
gauge compute add http://192.168.1.20:47632   # or register by URL (other network segment, VPN)
gauge compute list
```

Discovery only *listens*. Gauge never scans address ranges, never probes ports, never touches a
machine where you have not started `gauge worker`, and never signs up for cloud resources.
Announcements carry a hash of the token, not the token; every request must carry the token
itself. Treat the token like a password and run workers only on networks you trust.

### What the pool is used for

- **Generation.** When `compute.pool_generation` is on and workers serve a model, the pipeline's
  prose calls are spread over them round-robin, falling back to the local model if a worker fails.
  A machine with a big GPU running Ollama can serve the writer while a laptop runs the pipeline.
- **Training GaugeLM.** `gauge native train --distributed` and the daemon run **model-averaging
  rounds**: every live worker and this machine start from the same weights, train on their own
  shard of the corpus for K steps, and the weights are averaged. Simple, tolerant of slow or lost
  workers (a failed worker is just left out of the average), and a 4M-parameter model is a 16 MB
  payload on a LAN. Larger presets want a gigabit network.

## Commands

```
gauge compute scan [--apply]      gauge compute watch [--interval 30] [--max-minutes N]
gauge compute token               gauge compute discover [--seconds 3]
gauge compute add URL [--name]    gauge compute list        gauge compute remove URL
gauge worker --token T [--name N] [--port 47632] [--no-llm] [--no-announce]
gauge native train --distributed --rounds 3
```

Settings under `compute`: `auto_device`, `token`, `announce_port` (47631), `worker_port` (47632),
`discover_seconds`, `distributed_training`, `pool_generation`.
