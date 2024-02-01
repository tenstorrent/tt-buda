// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

#include "balancer/balancer.hpp"

namespace tt
{
namespace graphlib
{
class Graph;
class Node;
}  // namespace graphlib

namespace placer
{

// Request for post-epoch pass to modify the graph of placement instructions
class PreEpochRequest
{
   public:
    virtual void ModifyGraph(graphlib::Graph *graph) = 0;
};

class InsertNopRequest : public PreEpochRequest
{
   private:
    graphlib::Node *src, *dest;  // insert nop between these two
   public:
    InsertNopRequest(graphlib::Node *src, graphlib::Node *dest) : src(src), dest(dest) {}
    virtual void ModifyGraph(graphlib::Graph *graph) override;
};

// Summary of configuration, and results of a placer attempt
struct PlacerAttemptSummary
{
    std::uint32_t epoch_index;
    std::uint32_t attempt_index;
    bool fail;

    // Requests made of pre-epoch passes
    std::vector<std::shared_ptr<PreEpochRequest>> pre_epoch_requests;
};

class PlacerHistory
{
    std::uint32_t current_epoch_index;
    std::uint32_t next_attempt_index;
    std::vector<std::vector<PlacerAttemptSummary>> attempts;

    PlacerAttemptSummary create_new_attempt_summary()
    {
        auto pas = PlacerAttemptSummary{current_epoch_index, next_attempt_index, false, {}};
        attempts[current_epoch_index].push_back(pas);
        return pas;
    }

   public:
    PlacerHistory() : current_epoch_index(0), next_attempt_index(0), attempts(1) {}

    PlacerAttemptSummary next_attempt()
    {
        auto pas = create_new_attempt_summary();
        next_attempt_index++;
        return pas;
    }
    void next_epoch()
    {
        current_epoch_index++;
        next_attempt_index = 0;
        attempts.push_back({});
    }

    std::uint32_t current_epoch() const { return current_epoch_index; }
    std::uint32_t current_attempt() const { return next_attempt_index; }
    void reset_attempts() { next_attempt_index = 0; }
};

std::unique_ptr<Graph> run_pre_epoch_passes(
    graphlib::Graph *graph, const balancer::BalancerConfig &config, PlacerHistory &history);

}  // namespace placer
}  // namespace tt
