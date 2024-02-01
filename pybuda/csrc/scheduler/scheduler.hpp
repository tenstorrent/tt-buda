// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <memory>
#include <vector>
#include "scheduler/utils.hpp"

namespace tt {

// Forward Declares
namespace graphlib
{
    class Graph;
}

namespace scheduler {

/*
  ____        _          ____  _                   _
 |  _ \  __ _| |_ __ _  / ___|| |_ _ __ _   _  ___| |_ _   _ _ __ ___  ___
 | | | |/ _` | __/ _` | \___ \| __| '__| | | |/ __| __| | | | '__/ _ \/ __|
 | |_| | (_| | || (_| |  ___) | |_| |  | |_| | (__| |_| |_| | | |  __/\__ \
 |____/ \__,_|\__\__,_| |____/ \__|_|   \__,_|\___|\__|\__,_|_|  \___||___/

*/

enum SchedulerPolicy
{
    Topological,
    ModuleInputsBFS,
    LongestPath,
};

struct SchedulerConfig
{
    SchedulerPolicy policy;
    std::vector<std::vector<std::string>> scheduler_constraints;
    const std::unordered_set<const tt::graphlib::Node*>* ignored_nodes = nullptr;

    SchedulerConfig(
        SchedulerPolicy policy = SchedulerPolicy::Topological,
        const std::vector<std::vector<std::string>>& scheduler_constraints = {}) :
        policy(policy), scheduler_constraints(scheduler_constraints)
    {
    }
};

/*
     _    ____ ___
    / \  |  _ \_ _|___
   / _ \ | |_) | |/ __|
  / ___ \|  __/| |\__ \
 /_/   \_\_|  |___|___/
*/

SchedulerPolicy policy_from_string(const std::string& policy_str);

// Returns an ordered list of node names
Schedule run_scheduler(const SchedulerConfig& config, const graphlib::Graph* graph);


// Individual scheduler implementations
Schedule run_topological_scheduler(const graphlib::Graph* graph);
Schedule run_module_by_module_scheduler(const SchedulerConfig& config, const graphlib::Graph* graph);

} // end namespace scheduler
} // end namespace tt
