// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "scheduler/scheduler.hpp"
#include "scheduler/utils.hpp"

namespace tt::scheduler
{

// Schedule the longest path through the graph, and then fill in the missing parts
Schedule run_longest_path_scheduler(const graphlib::Graph* graph);

}  // namespace tt::scheduler
