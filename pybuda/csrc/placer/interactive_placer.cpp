// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "placer/interactive_placer.hpp"

#include <optional>

#include "placer/grid_placer.hpp"
#include "placer/lower_to_placer.hpp"
#include "placer/lowering_utils.hpp"
#include "placer/placer.hpp"
#include "utils/assert.hpp"

namespace tt::placer
{

InteractivePlacer::InteractivePlacer(const graphlib::Graph *graph, const balancer::BalancerConfig &config) :
    valid(true), config(config)
{
    epoch_id_to_device_grid.rows = config.device_config.grid_size.r;  // TODO: get harvested rows
    epoch_id_to_device_grid.columns = config.device_config.grid_size.c;
    chips_with_mmio = std::unordered_set<ChipId>(
        std::begin(config.device_config.chips_with_mmio),
        std::end(config.device_config.chips_with_mmio));

    log_debug(tt::LogPlacer, "config.device_config.arch_name:{}", config.device_config.arch_name);
    log_debug(tt::LogPlacer, "config.chip_ids:{}", config.chip_ids);
    log_debug(tt::LogPlacer, "config.chips_with_mmio:{}", config.device_config.chips_with_mmio);

    TT_LOG_ASSERT(
        config.chip_ids.size() == 1 || config.device_config.is_wormhole_b0(),
        "Interactive placer for multi-chip - unsupported architecture: {}",
        config.device_config.arch_name);

    if (env_as<bool>("PYBUDA_WORMHOLE_PIPELINED_PLACER") == false
        && config.device_config.is_wormhole_b0())
    {
        // iterate over chip_ids in round-robin for WH placements, non-mmio chips first
        sorted_chip_ids = placer::lowering::apply_chip_placement_policy(config.device_config, config.chip_placement_policy, config.chip_ids);
    }
    else
    {
        // single-chip assignment for non-WH or for pipelined placement
        sorted_chip_ids.push_back(0);
    }

    log_debug(tt::LogPlacer, "sorted_chip_ids: {}", sorted_chip_ids);

    if (graph)
    {
        output_ops = placer::lowering::get_output_nodes(graph);
    }

    current_epoch_index = 0;
    current_epoch_type = NodeEpochType::Forward;
    current_temporal_epoch_id = 0;
    current_spatial_epoch_id = 0;
    init_epoch();
}

// insert empty graphs if there are unused chip_ids in the current temporal epoch
// arguments are spatial_epoch_id and temporal_epoch_id of the first empty graph to be inserted
void InteractivePlacer::insert_empty_graphs(std::uint32_t spatial_epoch_id, std::uint32_t temporal_epoch_id)
{
    while (remaining_chip_ids_in_temporal_epoch.size())
    {
        ChipId chip_id = remaining_chip_ids_in_temporal_epoch.front();
        remaining_chip_ids_in_temporal_epoch.pop_front();
        log_debug(
            tt::LogPlacer,
            "empty graph - current_epoch_index:{} temporal_epoch_id:{} spatial_epoch_id:{} chip_id:{}",
            current_epoch_index,
            temporal_epoch_id,
            spatial_epoch_id,
            chip_id);
        TT_ASSERT(spatial_epoch_id < sorted_chip_ids.size());

        epoch_id_to_chip[current_epoch_index] = chip_id;
        epoch_id_to_epoch_info[current_epoch_index] = EpochInfo{
            .global_epoch_id = current_epoch_index,
            .temporal_epoch_id = temporal_epoch_id,
            .spatial_epoch_id = spatial_epoch_id,
            .epoch_type = current_epoch_type};
        epoch_id_to_subgraph_index[current_epoch_index] = 0;
        epoch_id_to_op_placement[current_epoch_index].clear();
        current_epoch_index++;
        spatial_epoch_id++;
    }
}

// compute current_chip_id for the next spatial epoch with global spatial id=current_epoch_index
// start_temporal_epoch: true if the next epoch is the first epoch in the temporal epoch
// new_temporal_epoch: true if this is the first time we are processing this temporal epoch, i.e. it is not a rewind
void InteractivePlacer::next_chip_id(bool start_temporal_epoch, bool new_temporal_epoch, std::optional<std::vector<ChipId>> requested_chip_ids)
{
    // repopulate chip ids for the temporal epoch
    if (start_temporal_epoch)
    {
        // if we finished a temporal epoch and starting a new one with next_epoch,
        // remaining_chip_ids_in_temporal_epoch should be empty
        // but in case next_epoch is forced with an epoch_break or we rewind epochs,
        // remaining_chip_ids_in_temporal_epoch will not be empty and we should clear it
        // if remaining chip_ids are not empty (i.e. we are not rewinding) insert empty graphs for them
        if (remaining_chip_ids_in_temporal_epoch.size() && new_temporal_epoch)
        {
            // current_temporal_epoch_id is already incremented for the new temporal epoch
            insert_empty_graphs(
                epoch_id_to_epoch_info.at(current_epoch_index - 1).spatial_epoch_id +
                    1,  // last spatial_epoch_id in temporal epoch
                current_temporal_epoch_id -
                    1  // current_temporal_epoch_id is already incremented for the new temporal epoch
            );
        }
        remaining_chip_ids_in_temporal_epoch.clear();

        if (env_as<bool>("PYBUDA_PLACER_SNAKE") && (current_temporal_epoch_id % 2) == 1)
        {
            // every odd temporal epoch, iterate chip ids in reverse order
            // so that we start with the chip id we ended the previous temporal epoch
            std::copy(
                sorted_chip_ids.rbegin(),
                sorted_chip_ids.rend(),
                std::inserter(remaining_chip_ids_in_temporal_epoch, remaining_chip_ids_in_temporal_epoch.begin()));
        }
        else
        {
            std::copy(
                sorted_chip_ids.begin(),
                sorted_chip_ids.end(),
                std::inserter(remaining_chip_ids_in_temporal_epoch, remaining_chip_ids_in_temporal_epoch.begin()));
        }
    }

    // check if any of the requested chip ids is in remaining_chip_ids_in_temporal_epoch, then use it
    // otherwise get the next chip id from remaining_chip_ids_in_temporal_epoch
    ChipId requested_chip_id = INVALID_CHIP_ID;
    if(requested_chip_ids.has_value()) {
        for(auto& chip_id: requested_chip_ids.value()) {
            if(std::find(
                remaining_chip_ids_in_temporal_epoch.begin(),
                remaining_chip_ids_in_temporal_epoch.end(),
                chip_id) != remaining_chip_ids_in_temporal_epoch.end()) {
                requested_chip_id = chip_id;
                break;
            }
        }
    }

    if(requested_chip_id != INVALID_CHIP_ID) {
        current_chip_id = requested_chip_id;
        remaining_chip_ids_in_temporal_epoch.erase(
            std::remove(
                remaining_chip_ids_in_temporal_epoch.begin(),
                remaining_chip_ids_in_temporal_epoch.end(),
                requested_chip_id),
            remaining_chip_ids_in_temporal_epoch.end());
    }
    else {
        // get a chip id to use
        // the requirement for picking a chip_id is not to repeat chip_ids in a temporal epoch
        // i.e. we need to end the temporal epoch once all chip ids are used
        // a simple algorihm used here is to pop from a deque.
        // once all all last chip ids are used, a new temporal epoch will start.
        current_chip_id = remaining_chip_ids_in_temporal_epoch.front();
        remaining_chip_ids_in_temporal_epoch.pop_front();
    }

    is_current_chip_id_mmio = chips_with_mmio.count(current_chip_id);
}

// returns true if the op can be placed on current_chip_id
bool InteractivePlacer::can_place_op_onto_chip(const std::string &op_name, bool chip_break, std::vector<ChipId>& requested_chip_ids)
{
    bool output_op = config.output_queues_on_host && output_ops.find(op_name) != output_ops.end();
    if (output_op)
    {
        log_debug(tt::LogPlacer, "epoch {} contains output_op: {}", current_epoch_index, op_name);
    }
    if (chip_break)
    {
        log_debug(tt::LogPlacer, "epoch {} contains chip_break_op: {}", current_epoch_index, op_name);
    }

    // place output ops only on mmio chips
    bool skip_due_to_output_op = output_op && is_current_chip_id_mmio == false;
    // skip a spatial epoch if a chip break is requested
    bool skip_due_to_chip_break = chip_break && placed_ops_in_current_epoch.size() &&
                                  visited_ops_in_current_epoch.find(op_name) == visited_ops_in_current_epoch.end();
    ChipId override_chip_id =
        config.op_name_to_placer_overrides.find(op_name) != config.op_name_to_placer_overrides.end() &&
                config.op_name_to_placer_overrides.at(op_name).chip_id.has_value()
            ? config.op_name_to_placer_overrides.at(op_name).chip_id.value()
            : INVALID_CHIP_ID;
    // skip if op has a chip id override which is not current_chip_id
    bool skip_due_to_chip_id_override = override_chip_id != INVALID_CHIP_ID && override_chip_id != current_chip_id;
    TT_ASSERT(
        override_chip_id == INVALID_CHIP_ID || skip_due_to_output_op == false || chips_with_mmio.count(override_chip_id),
        "Op has override chip id but must be placed onto mmio chip");
    // only output ops are placed on mmio chips on Nebula+Galaxy systems
    bool skip_if_not_output = env_as<bool>("PYBUDA_NEBULA_GALAXY_PLACER") && output_op == false &&
                              is_current_chip_id_mmio == true && sorted_chip_ids.size() != 1;

    // if we are using pipelined placer, we insert an implicit epoch break on the output op
    // so that the output op should be the sole op on the epoch so it can be placed on the mmio chip
    bool skip_due_to_epoch_break_for_output_op =
        env_as<bool>("PYBUDA_NEBULA_GALAXY_PLACER") && env_as<bool>("PYBUDA_WORMHOLE_PIPELINED_PLACER") && output_op &&
        placed_ops_in_current_epoch.size() &&
        visited_ops_in_current_epoch.find(op_name) == visited_ops_in_current_epoch.end();

    // request a chip id from the chip_id assignment for the next epoch
    if(skip_due_to_chip_id_override) {
        requested_chip_ids.push_back(override_chip_id);
    }
    else if(skip_due_to_output_op) {
        // request an mmio chip id for next attempt
        std::transform(
            config.device_config.chips_with_mmio.begin(),
            config.device_config.chips_with_mmio.end(),
            std::inserter(requested_chip_ids, requested_chip_ids.begin()),
            [](int chip_id){ return (ChipId)chip_id; }
        );
    }

    return skip_due_to_output_op == false && skip_due_to_chip_break == false && skip_due_to_chip_id_override == false &&
           skip_if_not_output == false && skip_due_to_epoch_break_for_output_op == false;
}

// initialize a spatial epoch with the epoch_index (i.e. global epoch id)
void InteractivePlacer::init_epoch(bool start_temporal_epoch, bool new_temporal_epoch, std::optional<std::vector<ChipId>> requested_chip_ids)
{
    next_chip_id(start_temporal_epoch, new_temporal_epoch, requested_chip_ids);

    epoch_id_to_epoch_info[current_epoch_index] = EpochInfo{
        .global_epoch_id = current_epoch_index,
        .temporal_epoch_id = current_temporal_epoch_id,
        .spatial_epoch_id = current_spatial_epoch_id,
        .epoch_type = current_epoch_type};

    if(env_as<bool>("PYBUDA_NEBULA_GALAXY_PLACER") && is_current_chip_id_mmio)
    {
        // On Nebula+Galaxy systems, Nebula chip is mmio and it is harvested
        epoch_id_to_device_grid.initialize_device_grid(
            current_epoch_index,
            config.device_config.get_harvested_nebula_galaxy_grid().r,
            config.device_config.get_harvested_nebula_galaxy_grid().c
        );
    }
    else
    {
        epoch_id_to_device_grid.initialize_device_grid(
            current_epoch_index,
            config.device_config.grid_size.r,
            config.device_config.grid_size.c
        );
    }

    epoch_id_to_op_placement[current_epoch_index].clear();
    epoch_id_to_chip[current_epoch_index] = current_chip_id;
    epoch_id_to_subgraph_index[current_epoch_index] = 0;
    if (start_temporal_epoch)
    {
        placed_ops_in_current_epoch.clear();
        visited_ops_in_current_epoch.clear();
    }
    log_debug(
        tt::LogPlacer,
        "init_epoch - current_epoch_index:{} current_chip_id:{} current_temporal_epoch_id:{} "
        "current_spatial_epoch_id:{} remaining_chip_ids_in_temporal_epoch.size:{}",
        current_epoch_index,
        current_chip_id,
        current_temporal_epoch_id,
        current_spatial_epoch_id,
        remaining_chip_ids_in_temporal_epoch.size());
}

// Place single op on current epoch. Returns nullopt if it doesn't fit.
std::optional<placer::CoordRange> InteractivePlacer::place_op(
    const std::string &op_name, const balancer::GridShape &shape, bool enable_transpose, bool chip_break)
{
    return place_op(
        op_name, placer::GridShape({(std::uint32_t)shape.r, (std::uint32_t)shape.c}), enable_transpose, chip_break);
}

std::optional<placer::CoordRange> InteractivePlacer::place_op(
    const std::string &op_name, const placer::GridShape &shape, bool enable_transpose, bool chip_break)
{
    TT_ASSERT(valid);
    std::unordered_map<std::string, placer::GridShape> to_place;
    to_place[op_name] = shape;

    log_debug(
        tt::LogPlacer,
        "Interactive placer start for op {}, grid ({}, {})", op_name, shape.rows, shape.columns);

    std::optional<placer::DeviceGridPlacement> placement = place_one_op(
        op_name,
        config.enable_auto_transposing_placement && enable_transpose,
        chip_break,
        to_place);

    // cannot place the op on this temporal epoch
    if(!placement.has_value())
    {
        return std::nullopt;
    }

    // Placed, update structures
    placed_ops_in_current_epoch.push_back(op_name);

    auto device_grid_placement = placement.value();
    OpPlacement op_placement = OpPlacement{
        .id = 0,
        .name = op_name,
        .chip_id = current_chip_id,
        .global_epoch_id = current_epoch_index,
        .grid_transpose = device_grid_placement.grid_transpose,
        .placed_cores = device_grid_placement.placed_cores};
    name_to_op_placement[op_placement.name] = op_placement;
    epoch_id_to_op_placement[current_epoch_index].push_back(op_placement);

    placer::GridShape op_shape = shape;
    if (op_placement.grid_transpose)
    {
        op_shape = placer::GridShape(shape.columns, shape.rows);
    }

    epoch_id_to_device_grid.fill_device_grid_with_placement(
        current_epoch_index, device_grid_placement.placed_cores.start, op_shape);

    log_debug(
        tt::LogPlacer,
        "Interactive placer: op {}, grid ({}, {}) onto chip_id={}, epoch_id={}, inclusive_start: {}, exclusive_end={}",
        op_placement.name,
        op_shape.rows,
        op_shape.columns,
        op_placement.chip_id,
        op_placement.epoch_id(),
        op_placement.placed_cores.start,
        op_placement.placed_cores.end);

    return op_placement.placed_cores;
}

std::optional<placer::DeviceGridPlacement> InteractivePlacer::place_one_op(
    const std::string &op_name, bool enable_transpose, bool chip_break, const std::unordered_map<std::string, placer::GridShape>& to_place)
{
    std::optional<placer::DeviceGridPlacement> placement;

    // keep trying epochs/chip_ids for the op
    // until we either successfully place the op
    // or reach the end of the temporal epoch (i.e. fail)
    while (!placement.has_value())
    {
        std::vector<ChipId> requested_chip_ids;
        if (can_place_op_onto_chip(op_name, chip_break, requested_chip_ids))
        {
            placement = placer::place_one_op(
                op_name,
                to_place,
                epoch_id_to_device_grid.get_device_grid(current_epoch_index),
                config.op_name_to_placer_overrides,
                enable_transpose);
        }
        else
        {
            log_debug(tt::LogPlacer, "skipping place_op in epoch {}", current_epoch_index);
        }
        if (!placement.has_value())
        {
            // if no chip ids left in the temporal epoch, we cannot place the op
            if (remaining_chip_ids_in_temporal_epoch.size() == 0)
            {
                return std::nullopt;
            }

            // for whatever reason, we did not place this op on this spatial epoch, no need to consider it for chip
            // break again
            visited_ops_in_current_epoch.insert(op_name);

            // initialize the next spatial epoch within the current temporal epoch (with a new chip id) and try again
            current_spatial_epoch_id++;
            current_epoch_index++;

            // corner case:
            // 1. if we were not able to place any ops on the current_chip_id
            // 2. a chip id was requested for this op
            // 3. we will definitely place this on the requested chip id
            // put the current_chip_id back in the pool to be used for the remaining ops
            // #3 prevents deadlock: if requested_chip_id is not available,
            //    we will keep inserting current_chip_id to the pool and pop it in a loop
            if(placed_ops_in_current_epoch.size() == 0) {
                for(auto& requested_chip_id: requested_chip_ids) {
                    if(std::find(remaining_chip_ids_in_temporal_epoch.begin(),
                            remaining_chip_ids_in_temporal_epoch.end(), requested_chip_id) != remaining_chip_ids_in_temporal_epoch.end()) {
                        remaining_chip_ids_in_temporal_epoch.push_front(current_chip_id);
                        current_spatial_epoch_id--;
                        current_epoch_index--;
                    }
                }
            }

            init_epoch(false /* start_temporal_epoch */, false /* new_temporal_epoch */, requested_chip_ids);
        }

        // in Nebula+Galaxy systems, only output_ops will be placed onto mmio chips (Nebula)
        // these ops should not use rows 8&9 due to harvesting
        // TODO: this should be driven based on the config read from backend
        // Also Nebula is not necessarily the mmio chip when we have more than one nebula chips
        TT_ASSERT(
            env_as<bool>("PYBUDA_NEBULA_GALAXY_PLACER") == false || config.output_queues_on_host == false ||
            output_ops.find(op_name) == output_ops.end() || placement.has_value() == false ||
            placement.value().placed_cores.end.row <= 8);
    }

    return placement;
}

// Bind and atomically place two ops as if they were one op.
// Row dimension must match.
//
std::optional<placer::CoordRange> InteractivePlacer::place_two_ops_rowwise(
    const std::string &op_name_1,
    const balancer::GridShape &shape_1,
    const std::string &op_name_2,
    const balancer::GridShape &shape_2,
    bool enable_transpose,
    bool chip_break)
{
    return place_two_ops_rowwise(
        op_name_1,
        placer::GridShape((std::uint32_t)shape_1.r, (std::uint32_t)shape_1.c),
        op_name_2,
        placer::GridShape((std::uint32_t)shape_2.r, (std::uint32_t)shape_2.c),
        enable_transpose,
        chip_break);
}

std::optional<placer::CoordRange> InteractivePlacer::place_two_ops_rowwise(
    const std::string &op_name_1,
    const placer::GridShape &shape_1,
    const std::string &op_name_2,
    const placer::GridShape &shape_2,
    bool enable_transpose,
    bool chip_break)
{
    TT_ASSERT(valid);
    TT_ASSERT(shape_1.rows == shape_2.rows);
    std::unordered_map<std::string, placer::GridShape> to_place;
    to_place[op_name_1] =
        placer::GridShape({(std::uint32_t)shape_1.rows, (std::uint32_t)shape_1.columns + shape_2.columns});
    TT_ASSERT(can_fit_on_single_epoch(to_place[op_name_1].rows, to_place[op_name_1].columns, enable_transpose));

    std::optional<placer::DeviceGridPlacement> placement = place_one_op(
        op_name_1,
        config.enable_auto_transposing_placement && enable_transpose,
        chip_break,
        to_place);

    // cannot place the op on this temporal epoch
    if(!placement.has_value())
    {
        return std::nullopt;
    }

    // Placed, update structures. Since we placed two OPs as a single block now we need to unbind them.
    // Calculate grid bounds of both ops and uptate the structures accordingly.
    // First OP needs to have grid end updated, second OP needs to have grid start updated.
    //
    placed_ops_in_current_epoch.push_back(op_name_1);

    auto device_grid_placement = placement.value();
    auto device_grid_placement_1 = device_grid_placement;

    // Handle transpose.
    //
    placer::GridShape op_shape_1 = shape_1;
    if (device_grid_placement.grid_transpose)
    {
        op_shape_1 = placer::GridShape(shape_1.columns, shape_1.rows);
    }

    device_grid_placement_1.placed_cores.end.row = device_grid_placement.placed_cores.start.row + op_shape_1.rows;
    device_grid_placement_1.placed_cores.end.col = device_grid_placement.placed_cores.start.col + op_shape_1.columns;

    OpPlacement op_placement_1 = OpPlacement{
        .id = 0,
        .name = op_name_1,
        .chip_id = current_chip_id,
        .global_epoch_id = current_epoch_index,
        .grid_transpose = device_grid_placement_1.grid_transpose,
        .placed_cores = device_grid_placement_1.placed_cores};
    name_to_op_placement[op_placement_1.name] = op_placement_1;
    epoch_id_to_op_placement[current_epoch_index].push_back(op_placement_1);

    epoch_id_to_device_grid.fill_device_grid_with_placement(
        current_epoch_index, device_grid_placement_1.placed_cores.start, op_shape_1);

    log_debug(
        tt::LogPlacer,
        "Interactive placer: op {}, grid ({}, {}) onto chip_id={}, epoch_id={}, inclusive_start: {}, exclusive_end={}",
        op_placement_1.name,
        op_shape_1.rows,
        op_shape_1.columns,
        op_placement_1.chip_id,
        op_placement_1.epoch_id(),
        op_placement_1.placed_cores.start,
        op_placement_1.placed_cores.end);

    // Handle transpose.
    //
    placer::GridShape op_shape_2 = shape_2;
    if (device_grid_placement.grid_transpose)
    {
        op_shape_2 = placer::GridShape(shape_2.columns, shape_2.rows);
    }

    auto device_grid_placement_2 = device_grid_placement;
    device_grid_placement_2.placed_cores.start.row = device_grid_placement.placed_cores.end.row - op_shape_2.rows;
    device_grid_placement_2.placed_cores.start.col = device_grid_placement.placed_cores.end.col - op_shape_2.columns;
    placed_ops_in_current_epoch.push_back(op_name_2);

    OpPlacement op_placement_2 = OpPlacement{
        .id = 0,
        .name = op_name_2,
        .chip_id = current_chip_id,
        .global_epoch_id = current_epoch_index,
        .grid_transpose = device_grid_placement_2.grid_transpose,
        .placed_cores = device_grid_placement_2.placed_cores};
    name_to_op_placement[op_placement_2.name] = op_placement_2;
    epoch_id_to_op_placement[current_epoch_index].push_back(op_placement_2);

    epoch_id_to_device_grid.fill_device_grid_with_placement(
        current_epoch_index, device_grid_placement_2.placed_cores.start, op_shape_2);

    log_debug(
        tt::LogPlacer,
        "Interactive placer: op {}, grid ({}, {}) onto chip_id={}, epoch_id={}, inclusive_start: {}, exclusive_end={}",
        op_placement_2.name,
        op_shape_2.rows,
        op_shape_2.columns,
        op_placement_2.chip_id,
        op_placement_2.epoch_id(),
        op_placement_2.placed_cores.start,
        op_placement_2.placed_cores.end);

    return device_grid_placement.placed_cores;
}

// Create and switch to new epoch. Returns next epoch id.
std::uint32_t InteractivePlacer::next_epoch(graphlib::NodeEpochType epoch_type)
{
    TT_ASSERT(valid);
    log_debug(tt::LogPlacer, "InteractivePlacer::next_epoch");
    current_epoch_index++;
    current_temporal_epoch_id++;
    current_spatial_epoch_id = 0;
    current_epoch_type = epoch_type;
    init_epoch();
    return current_epoch_index;
}

// Clear current epoch and start over. Returns the list of ops that were undone, in placed order.
std::vector<std::pair<std::string, OpPlacement>> InteractivePlacer::rewind_epoch_logged()
{
    std::vector<std::pair<std::string, OpPlacement>> ret;

    log_debug(tt::LogPlacer, "InteractivePlacer::rewind_epoch");

    for (const std::string &name : placed_ops_in_current_epoch)
    {
        const OpPlacement &p = name_to_op_placement.at(name);
        log_trace(LogPlacer, "Unplacing: {}", name);

        ret.push_back(std::make_pair(name, p));
        name_to_op_placement.erase(name);
    }

    // rewind back to the first spatial epoch in the temporal epoch
    current_epoch_index -= current_spatial_epoch_id;
    current_spatial_epoch_id = 0;

    init_epoch(true /* start_temporal_epoch */, false /* new_temporal_epoch */);  // clear the epoch
    return ret;
}

// Clear current epoch and start over. Non-logged fast version.
//
void InteractivePlacer::rewind_epoch()
{
    for (const std::string &name : placed_ops_in_current_epoch)
    {
        name_to_op_placement.erase(name);
    }

    // rewind back to the first spatial epoch in the temporal epoch
    current_epoch_index -= current_spatial_epoch_id;
    current_spatial_epoch_id = 0;

    init_epoch(true /* start_temporal_epoch */, false /* new_temporal_epoch */);  // clear the epoch
}

// Rewind current epoch to given op - i.e. place everything up to it, but not it. Returns the name
// and shape of the last placed op.
std::pair<std::string, OpPlacement> InteractivePlacer::rewind_to(const std::string &op_name)
{
    std::pair<std::string, OpPlacement> last;
    last.first = "";

    log_trace(LogPlacer, "Rewind to: {}", op_name);
    auto rew = rewind_epoch_logged();

    for (const auto &p : rew)
    {
        if (p.first == op_name)
            return last;

        log_trace(LogPlacer, "Replacing: {}", p.first);

        std::unordered_map<std::string, tt::placer::PlacerOpOverride>::iterator existing_override;
        std::unordered_map<std::string, tt::placer::PlacerOpOverride>::iterator rewind_override;
        std::optional<tt::placer::PlacerOpOverride> user_override = std::nullopt;
        existing_override = get_op_overrides().find(p.first);
        if (existing_override != get_op_overrides().end())
        {
            // Save the user override, if any, so we can restore it after rewinding the op.
            //
            user_override = existing_override->second;
            get_op_overrides().erase(existing_override);
        }

        bool rewind_override_set;
        std::tie(rewind_override, rewind_override_set) = get_op_overrides().emplace(
            p.first,
            tt::placer::PlacerOpOverride(p.second.placed_cores.start, p.second.grid_transpose, p.second.chip_id));
        TT_ASSERT(rewind_override_set);

        CoordRange untransposed_shape = p.second.placed_cores;
        if (p.second.grid_transpose)
        {
            untransposed_shape.transpose();
        }

        auto pl = place_op(p.first, GridShape({untransposed_shape.size_r(), untransposed_shape.size_c()}));

        get_op_overrides().erase(rewind_override);
        if (user_override.has_value())
        {
            get_op_overrides().emplace(p.first, user_override.value());
        }

        // Re-placing in same order, on the same epoch, same grid start and size -> it should always fit.
        //
        TT_LOG_ASSERT(pl.has_value(), "Failed to re-place {} after rewinding.", p.first);
        last = p;
    }

    TT_THROW("Rewinding to op that doesn't exist");
    return last;
}

// assign consecutive epochs to the same chip until
// number of epochs per chip is reached.
// e.g.
// ep0 means epoch0
// emp means empty graph
//               chip1    chip2   chip3
// temp epoch 0:  ep0      emp     emp
// temp epoch 1:  ep1      emp     emp
// temp epoch 2:  ep2      emp     emp
// temp epoch 3:  emp      ep3     emp
// temp epoch 4:  emp      ep4     emp
// temp epoch 5:  emp      ep5     emp
// temp epoch 6:  emp      emp     ep6
// temp epoch 7:  emp      emp     ep7
// temp epoch 8:  emp      emp     ep8
void InteractivePlacer::assign_chip_ids_for_pipelined_placement(
    std::uint32_t num_epochs, std::optional<std::unordered_set<string>> const &chip_break_ops)
{
    TT_ASSERT(config.device_config.is_wormhole_b0());
    TT_ASSERT(chip_break_ops.has_value());

    log_debug(tt::LogPlacer, "Interactive placer pipelined chip id assignment for {} epochs", num_epochs);

    placed_ops_in_current_epoch.clear();
    visited_ops_in_current_epoch.clear();

    // iterate over chip_ids in round-robin for WH placements, non-mmio chips first
    sorted_chip_ids = placer::lowering::apply_chip_placement_policy(config.device_config, config.chip_placement_policy, config.chip_ids);

    std::uint32_t num_epochs_per_chip = std::ceil(float(num_epochs) / sorted_chip_ids.size());

    // expecting no chip id placement before
    TT_ASSERT(remaining_chip_ids_in_temporal_epoch.size() == 0);
    std::copy(
        sorted_chip_ids.begin(),
        sorted_chip_ids.end(),
        std::inserter(remaining_chip_ids_in_temporal_epoch, remaining_chip_ids_in_temporal_epoch.begin()));

    current_temporal_epoch_id = 0;
    current_spatial_epoch_id = 0;             // spacial_epoch_id within the temporal epoch
    std::uint32_t current_chip_id_index = 0;  // round-robin over the chip ids
    current_chip_id = sorted_chip_ids.at(current_chip_id_index % sorted_chip_ids.size());
    is_current_chip_id_mmio = chips_with_mmio.count(current_chip_id);

    std::uint32_t num_epochs_placed_on_chip = 0;

    // iterate over all the epochs already placed and assign chip ids to them
    for (std::uint32_t epoch_index = 0; epoch_index < num_epochs; epoch_index++)
    {
        bool can_place_epoch_onto_chip = false;

        // keep looking for valid chip_id
        while (!can_place_epoch_onto_chip)
        {
            can_place_epoch_onto_chip = true;
            for (auto &placement : epoch_id_to_op_placement[epoch_index])
            {
                // unused in this function
                // for pipelined assignment, the order of chip ids is fixed
                std::vector<ChipId> requested_chip_ids;
                can_place_epoch_onto_chip =
                    can_place_epoch_onto_chip &&
                    can_place_op_onto_chip(
                        placement.name, chip_break_ops.value().find(placement.name) != chip_break_ops.value().end(), requested_chip_ids);
                if (!can_place_epoch_onto_chip)
                    break;
            }

            log_debug(
                tt::LogPlacer,
                "epoch_index:{} current_epoch_index:{} current_chip_id:{} temporal_epoch_id:{} spatial_epoch_id:{} "
                "can_place_epoch_onto_chip:{} num_epochs_per_chip:{}",
                epoch_index,
                current_epoch_index,
                current_chip_id,
                current_temporal_epoch_id,
                current_spatial_epoch_id,
                can_place_epoch_onto_chip,
                num_epochs_per_chip);

            if (can_place_epoch_onto_chip)
            {
                epoch_id_to_chip[epoch_index] = current_chip_id;
                epoch_id_to_epoch_info[epoch_index].temporal_epoch_id = current_temporal_epoch_id;
                epoch_id_to_epoch_info[epoch_index].spatial_epoch_id = current_spatial_epoch_id;
                epoch_id_to_subgraph_index[epoch_index] = 0;
                for (auto &op_placement : epoch_id_to_op_placement[epoch_index])
                {
                    op_placement.chip_id = current_chip_id;
                    name_to_op_placement[op_placement.name].chip_id = current_chip_id;
                }

                // so we do not insert chip breaks for the first op on the chip
                for (auto &placement : epoch_id_to_op_placement[epoch_index])
                {
                    placed_ops_in_current_epoch.push_back(placement.name);
                }

                num_epochs_placed_on_chip++;

                // remove current_chip_id from remaining_chip_ids_in_temporal_epoch
                // TODO: unnecessarily slow code but we do this once per temporal epoch for now
                remaining_chip_ids_in_temporal_epoch.erase(
                    std::remove(
                        remaining_chip_ids_in_temporal_epoch.begin(),
                        remaining_chip_ids_in_temporal_epoch.end(),
                        current_chip_id),
                    remaining_chip_ids_in_temporal_epoch.end());

                // end temporal epoch by assigning empty graphs to all the other chips
                insert_empty_graphs(current_spatial_epoch_id + 1, current_temporal_epoch_id);
                current_temporal_epoch_id++;
                current_spatial_epoch_id = 0;

                // re-populate chip ids for the next temporal epoch
                TT_ASSERT(remaining_chip_ids_in_temporal_epoch.empty());
                std::copy(
                    sorted_chip_ids.begin(),
                    sorted_chip_ids.end(),
                    std::inserter(remaining_chip_ids_in_temporal_epoch, remaining_chip_ids_in_temporal_epoch.begin()));
            }

            // so we insert exactly one chip break on the op
            for (auto &placement : epoch_id_to_op_placement[epoch_index])
            {
                visited_ops_in_current_epoch.insert(placement.name);
            }

            // moving to next chip after successfull placement
            if (can_place_epoch_onto_chip && num_epochs_placed_on_chip == num_epochs_per_chip)
            {
                placed_ops_in_current_epoch.clear();
            }

            // advance to next chip id if we could not place the op or we reached the epoch per chip limit
            if (can_place_epoch_onto_chip == false || num_epochs_placed_on_chip == num_epochs_per_chip)
            {
                current_chip_id_index++;
                current_chip_id = sorted_chip_ids.at(current_chip_id_index % sorted_chip_ids.size());
                num_epochs_placed_on_chip = 0;
                is_current_chip_id_mmio = chips_with_mmio.count(current_chip_id);
            }
        }
    }
}

PlacerSolution InteractivePlacer::commit(std::optional<std::unordered_set<string>> const &chip_break_ops)
{
    if (epoch_id_to_op_placement.at(current_epoch_index).size() > 0)
    {
        current_epoch_index++;
    }
    std::uint32_t num_epochs = current_epoch_index;

    if (env_as<bool>("PYBUDA_WORMHOLE_PIPELINED_PLACER"))
    {
        // assign chip ids after all epochs are created because we need to know
        // how many epochs we have to balance between all the chips
        assign_chip_ids_for_pipelined_placement(num_epochs, chip_break_ops);
    }
    else
    {
        // if doing round-robin/eager chip id assignment,
        // and if the last temporal epoch has unused chip ids, insert empty graphs for them
        insert_empty_graphs(current_spatial_epoch_id + 1, current_temporal_epoch_id);
    }

    log_debug(LogPlacer, "InteractivePlacer::commit");

    PlacerSolution placer_solution = PlacerSolution{
        .name_to_op_placement = std::move(name_to_op_placement),
        .input_queue_to_grid_shape = {},
        .name_to_queue_placement = {},
        .epoch_id_to_chip = std::move(epoch_id_to_chip),
        .epoch_id_to_subgraph_index = {},
        .epoch_id_to_op_placement = std::move(epoch_id_to_op_placement),
        .epoch_id_to_device_grid = std::move(epoch_id_to_device_grid),
        .epoch_id_to_epoch_info = std::move(epoch_id_to_epoch_info),
        .num_epochs = num_epochs};

    valid = false;
    return placer_solution;
}

}  // namespace tt::placer
