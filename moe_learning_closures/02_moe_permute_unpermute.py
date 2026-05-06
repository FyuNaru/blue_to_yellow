from typing import Tuple

import torch

from common import get_device


def top1_permute(hidden_states: torch.Tensor, expert_ids: torch.Tensor, num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_expert_ids, order = torch.sort(expert_ids)
    permuted = hidden_states.index_select(0, order)
    tokens_per_expert = torch.bincount(sorted_expert_ids, minlength=num_experts)
    return permuted, order, tokens_per_expert


def top1_unpermute(permuted: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    restored = torch.empty_like(permuted)
    restored.index_copy_(0, order, permuted)
    return restored


def main():
    device = get_device()
    hidden_states = torch.arange(8 * 3, device=device, dtype=torch.float32).view(8, 3)
    expert_ids = torch.tensor([2, 0, 1, 2, 3, 1, 0, 2], device=device)
    num_experts = 4

    permuted, order, tokens_per_expert = top1_permute(hidden_states, expert_ids, num_experts)
    restored = top1_unpermute(permuted, order)

    print(f"device: {device}")
    print(f"expert_ids: {expert_ids.cpu().tolist()}")
    print(f"order by expert: {order.cpu().tolist()}")
    print(f"tokens_per_expert: {tokens_per_expert.cpu().tolist()}")
    print("original hidden_states:")
    print(hidden_states.cpu())
    print("permuted hidden_states:")
    print(permuted.cpu())
    print("restored hidden_states:")
    print(restored.cpu())

    torch.testing.assert_close(restored, hidden_states)


if __name__ == "__main__":
    main()
