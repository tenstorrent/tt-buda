// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <vector>
#include <unordered_set>

#include "balancer/balancer.hpp"
#include "placer/grid_placer.hpp"

// Interactive placer provides APIs for placing individual ops, reverting epochs back, checkpointing, etc.

namespace tt
{
namespace placer
{

class InteractivePlacer
{
   private:
    std::uint32_t current_epoch_index;  // global id of the current spatial epoch
    NodeEpochType current_epoch_type;
    bool valid;
    std::vector<ChipId> sorted_chip_ids;
    std::deque<ChipId> remaining_chip_ids_in_temporal_epoch;
    ChipId current_chip_id;
    std::uint32_t current_temporal_epoch_id;
    std::uint32_t current_spatial_epoch_id;
    std::unordered_set<ChipId> chips_with_mmio; // for quick lookups
    bool is_current_chip_id_mmio;

    ChipId INVALID_CHIP_ID = (ChipId)-1;

    balancer::BalancerConfig config;
    unordered_map<string, OpPlacement> name_to_op_placement;
    map<PlacerSolution::EpochId, int> epoch_id_to_chip;
    map<PlacerSolution::EpochId, unsigned int> epoch_id_to_subgraph_index;
    unordered_map<int, vector<OpPlacement>> epoch_id_to_op_placement;
    EpochIdToDeviceGrid epoch_id_to_device_grid;
    unordered_map<int, EpochInfo> epoch_id_to_epoch_info;
    std::vector<std::string> placed_ops_in_current_epoch;  // ordered list
    std::set<std::string> visited_ops_in_current_epoch;
    std::unordered_set<std::string> output_ops;

    // returns true if the op can be placed on current_chip_id
    bool can_place_op_onto_chip(const std::string &op_name, bool chip_break, std::vector<ChipId>& requested_chip_ids);

    // utility function for picking a chip id for the epoch
    void next_chip_id(bool start_temporal_epoch, bool new_temporal_epoch, std::optional<std::vector<ChipId>> requested_chip_ids);

    // pipelined placement chip id assignment
    void assign_chip_ids_for_pipelined_placement(
        std::uint32_t num_epochs, std::optional<std::unordered_set<string>> const &chip_break_ops);

    // Set up new epoch
    void init_epoch(bool start_temporal_epoch = true, bool new_temporal_epoch = true, std::optional<std::vector<ChipId>> requested_chip_ids = std::nullopt);

    std::optional<placer::DeviceGridPlacement> place_one_op(
        const std::string &op_name,
        bool enable_transpose,
        bool chip_break,
        const std::unordered_map<std::string, placer::GridShape>& to_place);

   public:
    InteractivePlacer(const graphlib::Graph *graph, const balancer::BalancerConfig &config);

    // Place single op on current epoch. Returns nullopt if it doesn't fit.
    std::optional<placer::CoordRange> place_op(
        const std::string &op_name,
        const placer::GridShape &shape,
        bool enable_transpose = false,
        bool chip_break = false);
    std::optional<placer::CoordRange> place_op(
        const std::string &op_name,
        const balancer::GridShape &shape,
        bool enable_transpose = false,
        bool chip_break = false);

    std::optional<placer::CoordRange> place_two_ops_rowwise(
        const std::string &op_name_1,
        const balancer::GridShape &shape_1,
        const std::string &op_name_2,
        const balancer::GridShape &shape_2,
        bool enable_transpose = false,
        bool chip_break = false);
    std::optional<placer::CoordRange> place_two_ops_rowwise(
        const std::string &op_name_1,
        const placer::GridShape &shape_1,
        const std::string &op_name_2,
        const placer::GridShape &shape_2,
        bool enable_transpose = false,
        bool chip_break = false);

    // Create and switch to new epoch. Returns next epoch id.
    std::uint32_t next_epoch(graphlib::NodeEpochType epoch_type);

    // Clear current epoch and start over. Returns the list of ops that were undone, in placed order.
    std::vector<std::pair<std::string, OpPlacement>> rewind_epoch_logged();

    // Clear current epoch and start over. Non-logged fast version.
    //
    void rewind_epoch();

    // Rewind current epoch to given op - i.e. place everything up to it, but not it.
    // Returns placement information about last placed op.
    //
    std::pair<std::string, OpPlacement> rewind_to(const std::string &op_name);

    std::uint32_t get_current_epoch_index() const { return current_epoch_index; }
    bool current_epoch_empty() const { return placed_ops_in_current_epoch.empty(); }
    int current_epoch_size() const { return placed_ops_in_current_epoch.size(); }
    std::vector<std::string> const &current_epoch_ops() const { return placed_ops_in_current_epoch; }

    bool op_placed(const std::string &op_name) const { return name_to_op_placement.count(op_name) > 0; }

    void insert_empty_graphs(std::uint32_t spatial_epoch_id, std::uint32_t temporal_epoch_id);

    PlacerSolution commit(
        std::optional<std::unordered_set<string>> const &chip_break_ops =
            std::nullopt);  // Commit and generate final placer solution. Puts object into invalid state.
    std::unordered_map<std::string, placer::PlacerOpOverride> &get_op_overrides()
    {
        return config.op_name_to_placer_overrides;
    }

    bool can_fit_on_single_epoch(uint32_t rows, uint32_t columns, bool allow_transpose = false) const
    {
        return (rows <= epoch_id_to_device_grid.rows and columns <= epoch_id_to_device_grid.columns) or
               (allow_transpose and config.enable_auto_transposing_placement and
                rows <= epoch_id_to_device_grid.columns and columns <= epoch_id_to_device_grid.rows and rows > columns);
    }

    const unordered_map<string, OpPlacement>& get_current_name_to_op_placement() const { return name_to_op_placement; }
};

}  // namespace placer
}  // namespace tt
