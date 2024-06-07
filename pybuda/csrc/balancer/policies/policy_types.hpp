// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace tt::balancer
{

// Update can_use_interactive_placer() when adding new policy type.
//
enum class PolicyType
{
    MaximizeTMinimizeGrid,
    MinimizeGrid,
    Random,
    NLP,
    CNN,
    Ribbon,
    SingleOpPerEpoch
};

bool can_use_interactive_placer(PolicyType policy_type);

}  // namespace tt::balancer
