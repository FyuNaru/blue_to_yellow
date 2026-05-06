from typing import List

import torch
import torch.distributed as dist

from common import distributed_env, get_backend, get_device


def list_all_to_all(send_matrix: List[List[List[str]]]) -> List[List[str]]:
    world_size = len(send_matrix)
    received = []
    for dst in range(world_size):
        rank_received = []
        for src in range(world_size):
            rank_received.extend(send_matrix[src][dst])
        received.append(rank_received)
    return received


def run_list_demo():
    send_matrix = [
        [["r0_to_r0_a"], ["r0_to_r1_a", "r0_to_r1_b"]],
        [["r1_to_r0_a", "r1_to_r0_b", "r1_to_r0_c"], ["r1_to_r1_a"]],
    ]
    received = list_all_to_all(send_matrix)
    print("list simulation:")
    for rank, tokens in enumerate(received):
        print(f"  rank{rank} receives: {tokens}")


def run_distributed_demo():
    rank, _, world_size = distributed_env()
    device = get_device()
    dist.init_process_group(backend=get_backend())

    # Rank r sends r + dst + 1 scalar tokens to each destination dst.
    input_split_sizes = [rank + dst + 1 for dst in range(world_size)]
    all_input_splits = [None for _ in range(world_size)]
    dist.all_gather_object(all_input_splits, input_split_sizes)
    output_split_sizes = [all_input_splits[src][rank] for src in range(world_size)]

    payload = []
    for dst, count in enumerate(input_split_sizes):
        payload.extend([rank * 100 + dst] * count)

    input_tensor = torch.tensor(payload, dtype=torch.float32, device=device)
    output_tensor = torch.empty(sum(output_split_sizes), dtype=torch.float32, device=device)

    dist.all_to_all_single(
        output_tensor,
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=dist.group.WORLD,
    )

    print(
        f"rank={rank} input_split={input_split_sizes} output_split={output_split_sizes} "
        f"received={output_tensor.cpu().tolist()}"
    )
    dist.destroy_process_group()


def main():
    _, _, world_size = distributed_env()
    if world_size == 1:
        run_list_demo()
        print("\nRun with torchrun for real all_to_all_single, for example:")
        print("  torchrun --nproc_per_node 2 03_all_to_all.py")
        return
    run_distributed_demo()


if __name__ == "__main__":
    main()
