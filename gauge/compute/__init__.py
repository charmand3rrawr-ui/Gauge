"""Compute: sense this machine's GPUs and use the other machines you own."""
from .detect import Device, best_device, scan, summary, watch
from .pool import DistributedTrainer, Pool, PoolLLM, WorkerClient, discover
from .worker import Worker

__all__ = ["Device", "best_device", "scan", "summary", "watch", "DistributedTrainer", "Pool", "PoolLLM", "WorkerClient", "discover", "Worker"]
