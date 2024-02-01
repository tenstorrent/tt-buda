// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/post_epoch_passes.hpp"

namespace tt::placer
{

PlacerAttemptSummary run_post_epoch_passes(
    PlacerSolution &, PlacerSolution &epoch_placer_solution, PlacerHistory &history)
{
    PlacerAttemptSummary sum = history.next_attempt();
    sum.fail = (epoch_placer_solution.num_epochs == 0);

    return sum;
}

}  // namespace tt::placer
