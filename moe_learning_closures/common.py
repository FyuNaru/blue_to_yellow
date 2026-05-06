import os
from typing import Tuple

import torch


def npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def get_device() -> torch.device:
    if npu_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.npu.set_device(local_rank)
        return torch.device(f"npu:{local_rank}")
    return torch.device("cpu")


def get_backend() -> str:
    return "hccl" if npu_available() else "gloo"


def distributed_env() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def print_rank0(message: str) -> None:
    rank, _, _ = distributed_env()
    if rank == 0:
        print(message)
