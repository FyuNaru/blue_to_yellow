import torch
import torch.distributed as dist

from common import distributed_env, get_backend, get_device


def local_router(token_ids: torch.Tensor, world_size: int) -> torch.Tensor:
    # One expert per rank. Route token k to expert/rank k % world_size.
    return (token_ids.long() % world_size).long()


def permute_by_destination(hidden_states: torch.Tensor, dst_ranks: torch.Tensor, world_size: int):
    sorted_dst, order = torch.sort(dst_ranks)
    permuted = hidden_states.index_select(0, order)
    input_split_sizes = torch.bincount(sorted_dst, minlength=world_size).cpu().tolist()
    return permuted, order, input_split_sizes


def unpermute_by_order(permuted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    restored = torch.empty_like(permuted)
    restored.index_copy_(0, order, permuted)
    return restored


def all_to_all_v(tensor: torch.Tensor, input_split_sizes, output_split_sizes):
    output = torch.empty(
        [sum(output_split_sizes)] + list(tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    return output


def main():
    rank, _, world_size = distributed_env()
    device = get_device()

    if world_size == 1:
        print("Run with torchrun, for example: torchrun --nproc_per_node 2 04_moe_alltoall_pipeline.py")
        return

    dist.init_process_group(backend=get_backend())

    tokens_per_rank = 4
    hidden_size = 3
    token_ids = torch.arange(
        rank * tokens_per_rank,
        (rank + 1) * tokens_per_rank,
        device=device,
    )
    hidden_states = torch.stack(
        [
            token_ids.float(),
            token_ids.float() + 0.1,
            token_ids.float() + 0.2,
        ],
        dim=1,
    )

    dst_ranks = local_router(token_ids, world_size)
    permuted, order, send_splits = permute_by_destination(hidden_states, dst_ranks, world_size)

    all_send_splits = [None for _ in range(world_size)]
    dist.all_gather_object(all_send_splits, send_splits)
    recv_splits = [all_send_splits[src][rank] for src in range(world_size)]

    expert_input = all_to_all_v(permuted, send_splits, recv_splits)

    # Local expert computation. Each rank owns one toy expert.
    expert_output = expert_input + (rank + 1) * 1000.0

    # Send outputs back to the original ranks. The reverse communication swaps send/recv split roles.
    returned_permuted = all_to_all_v(expert_output, recv_splits, send_splits)
    restored = unpermute_by_order(returned_permuted, order)

    expected = hidden_states + (dst_ranks.float().unsqueeze(1) + 1.0) * 1000.0
    torch.testing.assert_close(restored, expected)

    print(
        f"rank={rank} token_ids={token_ids.cpu().tolist()} dst={dst_ranks.cpu().tolist()} "
        f"send={send_splits} recv={recv_splits}"
    )
    print(f"rank={rank} restored:\n{restored.cpu()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
