from typing import Callable, Literal
from torch import Tensor
import pandas as pd
import numpy as np
import torch

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    # put into df
    rewards_df = pd.DataFrame.from_records([
        reward_fn(response, gt)
        for response, gt in
        zip(rollout_responses, repeated_ground_truths)
    ])[['reward']]

    # assign groups
    num_groups = int(len(rewards_df) / group_size)
    rewards_df['group'] = np.concatenate([
        np.repeat(g, group_size) for g in range(num_groups)
    ])

    # compute mean and standard per group
    grouped_rewards = rewards_df.groupby('group', as_index=False)
    mean_rewards = grouped_rewards.mean().rename(columns={'reward': 'mean'})
    std_rewards = grouped_rewards.std().rename(columns={'reward': 'std'})

    # normalize
    rewards_df = rewards_df.merge(mean_rewards, on='group', how='left')
    rewards_df = rewards_df.merge(std_rewards, on='group', how='left')
    rewards_df['advantage'] = rewards_df['reward'] - rewards_df['mean']
    if normalize_by_std:
        rewards_df['advantage'] /= (rewards_df['std'] + advantage_eps)
    return Tensor(rewards_df['advantage']), Tensor(rewards_df['reward']), {}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    return - raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    p = torch.exp(policy_log_probs) / torch.exp(old_log_probs)
    pA = advantages * p
    p_clip = torch.clip(
        p, 1 - cliprange, 1 + cliprange
    )
    p_clipA = advantages * p_clip
    return -torch.minimum(pA, p_clipA), {}



def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        ), {}

    if loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs
        ), {}

    if loss_type == "grpo_clip":
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )


def masked_mean(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    tensor_summed = torch.sum(tensor * mask, dim=dim)
    mask_summed = torch.sum(mask, dim=dim)
    return tensor_summed / mask_summed


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None= None,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_token_loss, metadata = compute_policy_gradient_loss(
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        cliprange=cliprange,
        old_log_probs=old_log_probs,
    )
    loss = torch.mean(masked_mean(per_token_loss, response_mask, dim=1)) / gradient_accumulation_steps
    loss.backward()
    return loss, metadata