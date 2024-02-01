// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "placer/pre_epoch_passes.hpp"

namespace tt
{
namespace placer
{

PlacerAttemptSummary run_post_epoch_passes(
    PlacerSolution &placer_solution, PlacerSolution &epoch_placer_solution, PlacerHistory &history);

}
}  // namespace tt

