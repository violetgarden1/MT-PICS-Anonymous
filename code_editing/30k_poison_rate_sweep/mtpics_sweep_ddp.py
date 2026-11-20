"""Single-node torchrun helpers for AutoDL; no CUDA initialization at import time."""
from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
import signal
import subprocess
import sys


def topology(num_gpus, environment=None):
    env = os.environ if environment is None else environment
    world = int(env.get("WORLD_SIZE", "1"))
    rank = int(env.get("RANK", "0"))
    local = int(env.get("LOCAL_RANK", "0"))
    local_world = int(env.get("LOCAL_WORLD_SIZE", str(world)))
    if world != num_gpus or local_world != world or not 0 <= rank < world or local != rank:
        raise ValueError("Use one AutoDL instance: torchrun worker count must equal --num_gpus")
    return {"world_size": world, "rank": rank, "local_rank": local}


def launch_command(script, arguments, num_gpus):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={num_gpus}", "--max-restarts=0", str(Path(script).resolve()), *arguments]


def launch(script, arguments, num_gpus):
    """Stream child output into Jupyter and stop this job's worker group on interrupt."""
    command = launch_command(script, arguments, num_gpus)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    options = {"start_new_session": True} if os.name == "posix" else {}
    print(f"Starting {num_gpus} GPU worker(s) with the current Python: {sys.executable}", flush=True)
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1, **options)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        return process.wait()
    except KeyboardInterrupt:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()


def initialize(torch, num_gpus, check_only=False):
    state = topology(num_gpus)
    local = state["local_rank"]
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        raise RuntimeError(f"Need {num_gpus} visible CUDA GPUs in this AutoDL instance; check nvidia-smi")
    torch.cuda.set_device(local)
    devices = [torch.cuda.get_device_properties(index) for index in range(num_gpus)]
    if any(device.total_memory < 20 * 1024**3 for device in devices):
        raise RuntimeError("Each selected GPU must have at least 20 GiB; intended configuration is 2 x 24GB")
    # Compatibility choice for RTX 40-series PCIe instances, not an AutoDL-wide requirement.
    # Respect explicit user settings; retain normal NCCL selection on other hardware.
    if num_gpus > 1 and any("RTX 40" in device.name for device in devices):
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=180 if check_only else 1800))
    state["devices"] = [{"name": device.name, "total_memory_bytes": device.total_memory} for device in devices]
    state["nccl_environment"] = {key: os.environ.get(key) for key in ("NCCL_P2P_DISABLE", "NCCL_IB_DISABLE")}
    return state


def primary_call(torch, callback):
    """One filesystem writer; propagate its result or error to every worker."""
    distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    if not distributed:
        return callback()
    payload = [None]
    if torch.distributed.get_rank() == 0:
        try:
            payload[0] = {"ok": True, "value": callback()}
        except Exception as error:
            payload[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    torch.distributed.broadcast_object_list(payload, src=0)
    if not payload[0]["ok"]:
        raise RuntimeError(payload[0]["error"])
    return payload[0]["value"]


def check_collective(torch, state):
    value = torch.tensor(float(state["rank"] + 1), device=f"cuda:{state['local_rank']}")
    torch.distributed.all_reduce(value)
    expected = state["world_size"] * (state["world_size"] + 1) / 2
    if value.item() != expected:
        raise RuntimeError("NCCL all_reduce result is incorrect")
    torch.cuda.synchronize()
    if state["rank"] == 0:
        print(f"GPU/NCCL check passed: world_size={state['world_size']}, all_reduce={value.item()}", flush=True)
        print(state["devices"], state["nccl_environment"], flush=True)
