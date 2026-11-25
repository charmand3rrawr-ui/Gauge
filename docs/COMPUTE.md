# Compute: sensing GPUs and using the machines you own

## Sensing GPUs

`gauge compute scan` lists the CPU and every GPU it can find, from NVIDIA, AMD or Intel, using
the tools that exist on the machine (nvidia-smi, rocm-smi, Windows CIM, lspci), then checks
which numpy-compatible Python backend could drive each one:

| vendor | backend GaugeLM can use | install |
| --- | --- | --- |
| NVIDIA | CuPy on CUDA | `gauge compute setup-gpu` (installs `cupy-cuda12x[ctk]`) |
| AMD | CuPy on ROCm (Linux / WSL2) | `pip install cupy-rocm-6-0` |
| Intel, Apple | none yet | reported; the CPU path is used |

Hardware without its backend is listed with the exact install command. `--apply` saves the best
device into `native.device`. `gauge compute watch` polls until a usable GPU appears, for example
an external GPU plugged in, and switches to it. The daemon does the same scan every cycle
(`compute.auto_device`), so GaugeLM moves to a GPU as soon as one is there.

## The device switch: choosing which devices Gauge uses

Sensing a graphics card is not the same as using it. Two different things run on your computer:

* **The downloaded model** runs inside the AI engine (Ollama), which finds and uses a graphics card by
  itself. Nothing needs setting up.
* **GaugeLM**, Gauge's own model, runs inside Gauge. It can only use a graphics card once the library
  that drives it is installed (CuPy, for NVIDIA cards). The installer now does this when it finds an
  NVIDIA card, the updater does it once for an existing installation, and
  `gauge compute setup-gpu` (or **Set up the graphics card** in the Studio's Switches) does it on
  demand. It installs `cupy-cuda12x[ctk]`, which brings the CUDA components with it, so only the
  card's normal NVIDIA driver is needed. It is a download of several hundred megabytes.

The switch says which sensed devices Gauge may use, one or several:

```powershell
.\gauge.bat compute devices            # what this computer has, what is on, what each needs
.\gauge.bat compute use gpu0           # one graphics card
.\gauge.bat compute use gpu0 gpu1      # two: they share every training step
.\gauge.bat compute use gpu0 cpu       # a card and the processor
.\gauge.bat compute use cpu            # the processor only
.\gauge.bat compute use auto           # let Gauge pick (the default)
```

In the Studio, **Switches** has a **Devices** section at the top with a switch for each device and an
**Automatic** switch. More than one can be on; at least one stays on.

| choice | GaugeLM | the downloaded model |
| --- | --- | --- |
| automatic | the largest graphics card that is ready, else the processor | the engine decides |
| one graphics card | that card only (it is the only one Gauge's process can see) | may use a graphics card; which one is the engine's decision |
| several graphics cards | each holds a copy of the model and works through its own batch at the same time; the gradients are averaged and the weights copied back | as above |
| a card and the processor | the processor is timed on the first step and rested if it is more than four times slower than the card, which it normally is | as above |
| the processor only | the processor | kept on the processor (`num_gpu: 0`) |

Several devices sharing a step give the same result as one device on a batch that many times larger
(checked to rounding in the tests), in about the time of one batch: more text learned from per step.
A card that fails mid-run is dropped and training carries on. The capacity limit applies to each card
separately. Changing which cards are chosen after GaugeLM has already used one needs Gauge to be
closed and reopened, because a program's view of its graphics cards is fixed once the library starts.

**Not run on real hardware.** The machine this was built on has no graphics card, so the choosing,
the sharing of steps (with several processor copies standing in for cards) and the setup command are
tested, but the graphics-card paths themselves have never executed. AMD cards are listed and can be
chosen once CuPy for ROCm is installed by hand; Intel and Apple graphics are listed but cannot be used
by GaugeLM.

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

## The capacity limit

`compute.capacity_percent` (default **20**) is the most of each device Gauge may use: this computer and
every linked worker. `gauge compute limit` shows what that means here in numbers; `gauge compute limit 50`
sets it; 100 removes it. `GAUGE_CAPACITY_PERCENT` overrides it for one run.

- **Processor, Gauge's own process (firm).** At start-up Gauge pins itself to `floor(share x cores)` cores
  (at least one), caps its math libraries to the same number of threads, and lowers its priority. Child
  programs inherit the pin. Where one core is already more than the share (fewer than five cores at 20%)
  the governor measures the processor time really used and rests until the average is at the limit.
  Measured on a 4-core machine training GaugeLM: 63-70% of the machine without a limit, 20.2% at 20%,
  10.7% at 10%.
- **The AI engine (firm for threads, averaged for a graphics card).** Ollama is a separate program, so it
  cannot be pinned. Each request carries `num_thread`, and calls are made one at a time. A graphics card
  runs a request flat out, so requests are spaced: after one that took *t* seconds the next waits
  *t x (1/share - 1)*. A single call is never delayed; a steady stream averages the share.
- **GaugeLM on a graphics card.** The CuPy memory pool is limited to the share.
- **Linked devices.** A worker reads its own `compute.capacity_percent` and applies all of the above to
  itself; it reports the value in `/health`. The coordinator keeps a governor per worker and sends work to
  the one that has rested longest, so a worker running an older Gauge is not overused either.

What it cannot do: shrink a model. A loaded model occupies its full size. `gauge compute limit` prints the
share of memory and whether the configured model fits inside it. macOS has no core pinning; thread caps,
priority and pacing still apply there.

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
