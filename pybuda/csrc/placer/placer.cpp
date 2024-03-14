// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/placer.hpp"
#include "placer/utils.hpp"
#include "placer/lowering_utils.hpp"
#include "placer/dram.hpp"
#include "placer/grid_placer.hpp"
#include "third_party/json/json.hpp"

#include "graph_lib/defines.hpp"
#include "utils/logger.hpp"
#include "utils/assert.hpp"

#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include <exception>
#include <optional>

using tt::LogPlacer;
using std::max;
using std::ofstream;
using std::setw;
using std::runtime_error;

// Aliases
using NodeEpochType = tt::graphlib::NodeEpochType;
uint32_t tt::placer::OpGroupToPlace::current_op_group_id = 0;

namespace tt {
namespace placer {
    
uint32_t PlacerConfig::get_available_rows_on_device() const
{
    return device_grid.rows - harvested_rows.size();
}

uint32_t PlacerConfig::get_chip_id(const string& op_name) const
{
    if (!device_config.is_grayskull())
    {
        return 0;
    }
    return op_to_chip_id_assignment.at(op_name);
}
std::optional<uint32_t> PlacerConfig::get_chip_id_override(const string& op_name) const
{
    // check if op_name is in op_to_overrides. If so, return the chip_id from there
    if (auto it = op_to_overrides.find(op_name); it != op_to_overrides.end())
    {
        const auto& op_override = it->second;
        if (op_override.chip_id.has_value())
        {
            return op_override.chip_id.value();
        }
    }
    return {};
}

Coord Coord::operator+(const GridShape& rhs) const
{
    return {.row=this->row + rhs.rows,
            .col=this->col + rhs.columns};
}

Coord Coord::operator+(const Coord& rhs) const
{
    return {.row=this->row + rhs.row,
            .col=this->col + rhs.col};
}

Coord Coord::operator+(const CoordOffset& rhs) const
{
    return {.row=this->row + rhs.row_offset,
            .col=this->col + rhs.column_offset};
}

bool Coord::operator< (const Coord &rhs) const
{
    if (this->row == rhs.row)
    {
        return this->col < rhs.col;
    }
    return this->row < rhs.row;
}

bool Coord::operator==(const Coord &rhs) const
{
    return (row == rhs.row) && (col == rhs.col);
}

bool Coord::operator!=(const Coord &rhs) const
{
    return !(*this == rhs);
}

json Coord::to_json() const
{
    json output_json;
    output_json["row"] = this->row;
    output_json["col"] = this->col;
    return output_json;
}

std::array<std::uint32_t, 2> Coord::as_array() const
{
    return {this-> row, this->col};
}

json CoordRange::to_json() const
{
    json output_json;
    output_json["start"] = this->start.to_json();
    output_json["end"] = this->end.to_json();
    return output_json;
}

bool CoordRange::operator==(const CoordRange &rhs) const
{
    return this->start == rhs.start and this->end == rhs.end;
}
bool CoordRange::operator!=(const CoordRange &rhs) const
{
    return !(*this == rhs);
}


uint32_t OpGroupToPlace::get_next_op_group_id()
{
    return current_op_group_id++;
}

json OpPlacement::to_json() const
{
    json output_json;
    output_json["name"] = this->name;
    output_json["chip_id"] = this->chip_id;
    output_json["epoch_id"] = this->epoch_id();

    vector<json> placed_cores_json;
    placed_cores_json.push_back(this->placed_cores.to_json());

    output_json["placed_cores"] = placed_cores_json;
    return output_json;
}

bool OpPlacement::operator==(const OpPlacement& rhs) const
{
    // exclude global_id from capture
    return (
        this->name == rhs.name and
        this->chip_id == rhs.chip_id and
        this->placed_cores == rhs.placed_cores
    );
}
bool OpPlacement::operator!=(const OpPlacement& rhs) const
{
    return !(*this == rhs);
}

json QueueBufferPlacement::to_json() const
{
    json output_json;
    output_json["dram_channel"] = this->dram_channel;
    output_json["dram_address"] = this->dram_address;
    output_json["dram_channel_location"] = this->dram_channel_location.to_json();
    output_json["buffer_size"] = this->buffer_size;
    return output_json;
}

json QueueHostBufferPlacement::to_json() const
{
    json output_json;
    output_json["channel"] = this->channel;
    output_json["address"] = this->address;
    output_json["buffer_size"] = this->buffer_size;
    return output_json;
}

json QueuePlacement::to_json() const
{
    json output_json;
    output_json["name"] = this->name;
    output_json["on_host"] = this->on_host;
    output_json["chip_id"] = this->chip_id;

    vector<json> buffers_json;
    for (const QueueBufferPlacement& qb : this->dram_buffers)
    {
        buffers_json.push_back(qb.to_json());
    }

    output_json["buffers"] = buffers_json;
    return output_json;
}

json PlacerSolution::to_json() const
{
    json output_json;

    // serialize `name_to_op_placement`
    for (const auto& [name, op_placement] : this->name_to_op_placement)
    {
        output_json["name_to_op_placement"][name] = op_placement.to_json();
    }

    // serialize `name_to_queue_placement`
    for (const auto& [name, q_placement] : this->name_to_queue_placement)
    {
        output_json["name_to_queue_placement"][name] = q_placement.to_json();
    }

    // serialize `epoch_id_to_chip`
    for (const auto& [epoch_id, chip] : this->epoch_id_to_chip)
    {
        output_json["epoch_id_to_chip"][epoch_id] = chip;
    }

    output_json["num_epochs"] = this->num_epochs;

    return output_json;
}

bool PlacerSolution::is_placed(const std::string& op) const
{
    return this->name_to_op_placement.find(op) != this->name_to_op_placement.end();
}

uint32_t PlacerSolution::chip_id(const std::string& op) const {
    if (this->name_to_op_placement.find(op) != this->name_to_op_placement.end()) {
        return this->name_to_op_placement.at(op).chip_id;

    } else if (this->name_to_queue_placement.find(op) != this->name_to_queue_placement.end()) {
        return this->name_to_queue_placement.at(op).chip_id;
    }
    TT_LOG_ASSERT(false, "Error: PlacerSolution::chip_id() invoked with unassigned op/queue: {}", op);
    return 0;
}

uint32_t PlacerSolution::epoch_id(const std::string& op) const {
    int global_epoch_id = this->name_to_op_placement.at(op).epoch_id();
    return global_epoch_id;
}

uint32_t PlacerSolution::temporal_epoch_id(const std::string& op) const {
    int global_epoch_id = this->name_to_op_placement.at(op).epoch_id();
    return this->epoch_id_to_epoch_info.at(global_epoch_id).temporal_epoch_id;
}

uint32_t PlacerSolution::temporal_epoch_id(uint32_t global_epoch_id) const {
    return this->epoch_id_to_epoch_info.at(global_epoch_id).temporal_epoch_id;
}

uint32_t PlacerSolution::num_temporal_epochs() const {
    uint32_t max_epoch_id_found = 0;
    for (const auto& [epoch_id, epoch_info] : this->epoch_id_to_epoch_info) {
        max_epoch_id_found = std::max(max_epoch_id_found, epoch_info.temporal_epoch_id);
    }
    return max_epoch_id_found + 1;
}
NodeEpochType PlacerSolution::epoch_type(uint32_t global_epoch_id) const
{
    return this->epoch_id_to_epoch_info.at(global_epoch_id).epoch_type;
}

const EpochInfo& PlacerSolution::epoch_info(uint32_t global_epoch_id) const
{
    return this->epoch_id_to_epoch_info.at(global_epoch_id);
}

uint32_t PlacerSolution::num_temporal_epochs(NodeEpochType type) const
{
    std::set<uint32_t> temporal_epoch_ids;
    for (const auto& [epoch_id, epoch_info] : this->epoch_id_to_epoch_info)
    {
        if (epoch_info.epoch_type == type)
        {
            temporal_epoch_ids.insert(epoch_info.temporal_epoch_id);
        }
    }
    return temporal_epoch_ids.size();
}
    
// Merge another placer solution into this one. Destroys the original!
// Assumes that the 'other' contains new stand-alone epochs, this will likely not work 
// for partial epoch merging.
void PlacerSolution::merge(PlacerSolution &other)
{
    TT_ASSERT(is_pipelined == other.is_pipelined, "Incompatible placer solutions for merging (pipelined).");
    TT_ASSERT(epoch_id_to_device_grid.rows == other.epoch_id_to_device_grid.rows, 
            "Incompatible placer solutions for merging (grid rows).");
    TT_ASSERT(epoch_id_to_device_grid.columns == other.epoch_id_to_device_grid.columns, 
            "Incompatible placer solutions for merging (grid columns).");

    name_to_op_placement.merge(other.name_to_op_placement);
    input_queue_to_grid_shape.merge(other.input_queue_to_grid_shape);
    name_to_queue_placement.merge(other.name_to_queue_placement);
    epoch_id_to_chip.merge(other.epoch_id_to_chip);
    epoch_id_to_op_placement.merge(other.epoch_id_to_op_placement);
    epoch_id_to_epoch_info.merge(other.epoch_id_to_epoch_info);
    num_epochs += other.num_epochs;
    epoch_id_to_device_grid.epoch_id_to_device_grid.merge(other.epoch_id_to_device_grid.epoch_id_to_device_grid);
}

void EpochIdToDeviceGrid::initialize_device_grid(uint32_t candidate_epoch_id, bool clear_existing)
{
    if (clear_existing || (this->epoch_id_to_device_grid.find(candidate_epoch_id) == this->epoch_id_to_device_grid.end()))
    {
        this->epoch_id_to_device_grid[candidate_epoch_id] = device_grid::create_empty_device_grid(this->rows, this->columns);
    }
}

void EpochIdToDeviceGrid::initialize_device_grid(uint32_t candidate_epoch_id, uint32_t rows, uint32_t columns)
{
    this->epoch_id_to_device_grid[candidate_epoch_id] = device_grid::create_empty_device_grid(rows, columns);
}

void EpochIdToDeviceGrid::initialize_device_grid(uint32_t candidate_epoch_id, const DeviceGrid& device_grid)
{
    if (this->epoch_id_to_device_grid.find(candidate_epoch_id) == this->epoch_id_to_device_grid.end())
    {
        this->epoch_id_to_device_grid[candidate_epoch_id] = device_grid;
    }
}
std::optional<Coord> EpochIdToDeviceGrid::get_next_grid_coordinate(const std::string& op_name, uint32_t epoch_id, const GridShape& op_grid_shape) const
{
    DeviceGrid device_grid = this->epoch_id_to_device_grid.at(epoch_id);
    for (const auto& [constraint_name, constraint_grid] : this->op_to_constraints)
    {
        if (op_name != constraint_name)
        {
            device_grid = device_grid::superposition(device_grid, constraint_grid);
        }
    }

    return device_grid::get_next_grid_coordinate(device_grid, op_grid_shape);
}

bool EpochIdToDeviceGrid::satisfies_constraints(const std::string& op_name, const Coord& start, const GridShape& shape) const
{
    bool satisfies_constraints = true;
    for (const auto& [constraint_name, constraint_grid] : this->op_to_constraints)
    {
        if (op_name != constraint_name)
        {
            satisfies_constraints &= device_grid::can_place_on_device_grid(constraint_grid, start, shape);
        }
    }
    return satisfies_constraints;
}

bool EpochIdToDeviceGrid::can_place_on_device_grid(
    const std::string& op_name,
    int epoch_id,
    const Coord& start,
    const GridShape& shape)
{
    bool satisfies_constraints = this->satisfies_constraints(op_name, start, shape);
    this->initialize_device_grid(epoch_id);
    const DeviceGrid& device_grid = this->epoch_id_to_device_grid.at(epoch_id);
    return satisfies_constraints and device_grid::can_place_on_device_grid(device_grid, start, shape);
}

void fill_in_device_grid(
    const PlacerSolution& placer_solution,
    vector<vector<uint32_t>>& device_grid_for_epoch,
    const unordered_map<uint32_t, string> &id_to_string,
    const OpPlacement& op_placement,
    uint32_t id)
{
    const auto& cores = op_placement.placed_cores;
    for (uint32_t i = cores.start.row; i < cores.end.row; ++i) {
        for (uint32_t j = cores.start.col; j < cores.end.col; ++j) {
            if (device_grid_for_epoch.at(i).at(j) != 0) {
                uint32_t offending_id = device_grid_for_epoch[i][j];
                const string& offending_op = id_to_string.at(offending_id);
                auto oop = placer_solution.name_to_op_placement.at(offending_op);


                log_fatal("On chip {}, epoch {}, we are placing {} onto [{},{}]->[{},{}] but it overlaps with another op: {}, i:{}, j:{}, ->[{},{}]->[{},{}] ",
                        op_placement.chip_id, op_placement.epoch_id(), op_placement.name,
                        cores.start.row, cores.start.col, cores.end.row, cores.end.col,
                        offending_op,
                        i,j,
                        oop.placed_cores.start.row,oop.placed_cores.start.col, oop.placed_cores.end.row, oop.placed_cores.end.col);
            }
            device_grid_for_epoch[i][j] = id;
        }

    }
}

void fill_device_grid_with_placement(
    DeviceGrid& device_grid_for_epoch,
    const Coord& op_start,
    const GridShape& op_grid_shape)
{
    for (uint32_t i = op_start.row; i < op_start.row + op_grid_shape.rows; ++i) {
        for (uint32_t j = op_start.col; j < op_start.col + op_grid_shape.columns; ++j) {
            device_grid_for_epoch.at(i).at(j) = 1;
        }
    }
}


void EpochIdToDeviceGrid::fill_device_grid_with_placement(
    int epoch_id,
    const Coord& op_start,
    const GridShape& op_grid_shape)
{
    initialize_device_grid(epoch_id);
    device_grid::fill_device_grid_with_placement(this->epoch_id_to_device_grid.at(epoch_id), op_start, op_grid_shape);
}

/* static */ GridShape GridShape::from_array(std::array<uint32_t, 2> array) { return GridShape(array[0], array[1]); };

bool contains_harvested_row(uint32_t row_start, uint32_t num_rows_for_op, const vector<uint32_t>& harvested_rows)
{
    for (uint32_t row : harvested_rows) {
        bool contains_harvested_row = row >= row_start and row < row_start + num_rows_for_op;
        if (contains_harvested_row) {
            return false;
        }
    }
    return true;
}

bool EpochIdToDeviceGrid::contains_empty_grid(uint32_t epoch_id) {
    initialize_device_grid(epoch_id);
    if (this->epoch_id_to_device_grid.find(epoch_id) == this->epoch_id_to_device_grid.end()) {
        return true;
    }
    const auto& device_grid_for_epoch = this->epoch_id_to_device_grid.at(epoch_id);
    return device_grid::contains_empty_device_grid(device_grid_for_epoch);
}

uint32_t EpochIdToDeviceGrid::get_current_epoch_id() const
{
    int current_epoch = 0;
    for (const auto & [epoch_id, device_grid] : this->epoch_id_to_device_grid)
    {
        current_epoch = std::max(current_epoch, epoch_id);
    }
    return (uint32_t)current_epoch;
}

const DeviceGrid& EpochIdToDeviceGrid::get_device_grid(uint32_t epoch_id) const
{
    TT_ASSERT(this->epoch_id_to_device_grid.find(epoch_id) != this->epoch_id_to_device_grid.end());
    return this->epoch_id_to_device_grid.at(epoch_id);
}
void EpochIdToDeviceGrid::add_constraints(const std::unordered_map<std::string, DeviceGrid>& constraints)
{
    this->op_to_constraints = constraints;
}

void generate_placement_constraints(const PlacerConfig& config, vector<OpGroupToPlace>& placer_op_group_workload)
{
    // Encode the constraint that we need an epoch break between new epoch transitions
    // e.g. FWD->{RECOMPUTE/BWD}->OPT
    NodeEpochType prev_epoch_type = NodeEpochType::Forward;
    for (OpGroupToPlace& op_group_to_place : placer_op_group_workload)
    {
        for (const string& op_name : op_group_to_place.op_names)
        {
            if (is_forward_to_backward_epoch_transition(prev_epoch_type, config.op_to_epoch_type.at(op_name)) or
                is_backward_to_optimizer_epoch_transition(prev_epoch_type, config.op_to_epoch_type.at(op_name)))
            {
                op_group_to_place.increment_epoch = true;
                prev_epoch_type = config.op_to_epoch_type.at(op_name);
            }
        }
    }
}

void validate_placer_solution(const PlacerConfig& config, const PlacerSolution& placer_solution)
{
    for (const auto& [name, op_placement] : placer_solution.name_to_op_placement) {
        const CoordRange& coord = op_placement.placed_cores;
        if (coord.end.row > config.device_grid.rows or coord.end.col > config.device_grid.columns) {
            log_fatal("{} placed cores is out of bounds: [{},{}]->[{},{}], device_grid: [{},{}]", 
                    name,
                    coord.start.row, coord.start.col, coord.end.row, coord.end.col,
                    config.device_grid.rows, config.device_grid.columns);
        }
        if (env_as<bool>("PYBUDA_NEBULA_GALAXY_PLACER"))
        {
            if (config.output_queues_on_host and config.output_ops.find(name) != config.output_ops.end())
            {
                // TODO: get this from device config when mixed harvesting changes are fully consumed
                const uint32_t MAX_NUM_ROWS_UNHARVESTED = 8;
                // when PYBUDA_NEBULA_GALAXY_PLACER is enabled, nops are forced onto the Nebula chip which may be two
                // row harvested for now assert that the output nop is not spilling onto the last two rows
                if (coord.end.row > MAX_NUM_ROWS_UNHARVESTED)
                {
                    log_fatal(
                        "{} forced onto mmio chip may be out of bounds: [{},{}]->[{},{}], Nebula assumed harvested "
                        "grid: [{},{}]",
                        name,
                        coord.start.row,
                        coord.start.col,
                        coord.end.row,
                        coord.end.col,
                        MAX_NUM_ROWS_UNHARVESTED,
                        config.device_grid.columns);
                }
            }
        }
    }

    // within an epoch, there should not be any overlap in terms of op placement
    const int device_available_rows = config.get_available_rows_on_device();

    for (const auto& [epoch, op_placements] : placer_solution.epoch_id_to_op_placement) {
        // simple way to do the check by coloring in device grid - switch just to do simple boundary checks later
        vector<vector<uint32_t>> device_grid_for_epoch(device_available_rows, vector<uint32_t>(config.device_grid.columns));

        uint32_t start_id = 1;
        unordered_map<uint32_t, string> id_to_string;

        for (const auto& op_placement : op_placements) {
            id_to_string[start_id] = op_placement.name;
            fill_in_device_grid(placer_solution, device_grid_for_epoch, id_to_string, op_placement, start_id);
            start_id += 1;
        }
    }
}

PlacerSolution place_onto_chip(
    const PlacerConfig& config,
    PlacerWorkload& placer_op_group_workload,
    uint32_t epoch_start_id,
    std::optional<NodeEpochType> epoch_type)
{
    validate_chip_mapping(config, placer_op_group_workload);
    validate_placer_inputs(config, placer_op_group_workload);

    if (epoch_type)
    {
        if (epoch_type.value() == NodeEpochType::Forward) {
            log_debug(tt::LogPlacer, "Placing FWD ops...");
        } else if (epoch_type.value() == NodeEpochType::Backward) {
            log_debug(tt::LogPlacer, "Placing BWD ops...");
        } else if (epoch_type.value() == NodeEpochType::Optimizer) {
            log_debug(tt::LogPlacer, "Placing OPT ops...");
        }
    }
    else
    {
        generate_placement_constraints(config, placer_op_group_workload);
    }


    unordered_map<string, OpPlacement> name_to_op_placement;
    map<PlacerSolution::EpochId, int> epoch_id_to_chip;
    unordered_map<int, vector<OpPlacement>> epoch_id_to_op_placement;
    GridShape device_grid_shape(config.get_available_rows_on_device(), config.device_grid.columns);

    auto epoch_id_to_device_grid = EpochIdToDeviceGrid(device_grid_shape.rows, device_grid_shape.columns);
    unordered_map<int, EpochInfo> epoch_id_to_epoch_info;

    uint32_t max_epoch_id_assigned = epoch_start_id;
    bool placed = false;

    uint32_t current_epoch_id = epoch_start_id;
    vector<DeviceGridPlacement> device_grid_placements;

    std::vector<OpGroupToPlace> filtered_op_groups;
    std::unordered_set<uint32_t> visited_op_group_ids;
    for (const OpGroupToPlace& op_group : placer_op_group_workload)
    {
        if ((not epoch_type.has_value()) or (epoch_type and op_group.epoch_type == epoch_type.value()))
        {
            filtered_op_groups.push_back(op_group);
        }
    }

    auto placer = EpochDevicePlacer(config);
    std::vector<EpochDeviceGridPlacement> placed_epochs = placer.place(filtered_op_groups);
    std::map<std::string, uint32_t> name_to_op_group_id;
    std::map<std::string, uint32_t> name_to_chip_id;
    for (const OpGroupToPlace& op_group : placer_op_group_workload)
    {
        for (const auto& name : op_group.op_names)
        {
            name_to_op_group_id[name] = op_group.op_group_id;
            name_to_chip_id[name] = op_group.chip_id;
        }
    }


    for (uint32_t epoch_id = 0; epoch_id < placed_epochs.size(); ++epoch_id)
    {
        auto& placed_epoch = placed_epochs.at(epoch_id);
        epoch_id_to_device_grid.initialize_device_grid(current_epoch_id + epoch_id);
        vector<OpPlacement> op_placements;
        for (const auto& device_grid_placement : placed_epoch.op_placements)
        {
            op_placements.push_back(OpPlacement{
                .id = name_to_op_group_id.at(device_grid_placement.op_name),
                .name = device_grid_placement.op_name,
                .chip_id = name_to_chip_id.at(device_grid_placement.op_name), // TODO(JCHU): HACK
                .global_epoch_id = current_epoch_id + epoch_id,
                .grid_transpose = device_grid_placement.grid_transpose,
                .placed_cores = device_grid_placement.placed_cores
            });
 
            GridShape op_grid_shape = config.op_to_grid_shape.at(device_grid_placement.op_name);
            if(device_grid_placement.grid_transpose){
                op_grid_shape = GridShape(op_grid_shape.columns, op_grid_shape.rows);
            }
            epoch_id_to_device_grid.fill_device_grid_with_placement(
                current_epoch_id + epoch_id, 
                device_grid_placement.placed_cores.start,
                op_grid_shape); 
        }

        if (not op_placements.empty())
        {
            placed = true;
        }

        for (const OpPlacement& op_placement : op_placements)
        {
            const string& name = op_placement.name;
            name_to_op_placement[name] = op_placement;
            epoch_id_to_chip[op_placement.epoch_id()] = op_placement.chip_id;
            epoch_id_to_op_placement[op_placement.epoch_id()].push_back(op_placement);
            max_epoch_id_assigned = std::max(max_epoch_id_assigned, op_placement.epoch_id());

            log_debug(tt::LogPlacer, "\tPlacing {} with grid_shape ({}, {}) onto:",
                    op_placement.name, config.op_to_grid_shape.at(name).rows, config.op_to_grid_shape.at(name).columns);

            log_debug(tt::LogPlacer, "\t\t chip_id={}, epoch_id={}, inclusive_start: {}, exclusive_end={}",
                    op_placement.chip_id,
                    op_placement.epoch_id(),
                    op_placement.placed_cores.start,
                    op_placement.placed_cores.end
            );
        }
    }
    PlacerSolution placer_solution = PlacerSolution{
        .name_to_op_placement = std::move(name_to_op_placement),
        .input_queue_to_grid_shape = config.input_queue_to_grid_shape,
        .name_to_queue_placement = {},
        .epoch_id_to_chip = std::move(epoch_id_to_chip),
        .epoch_id_to_subgraph_index = {},
        .epoch_id_to_op_placement = std::move(epoch_id_to_op_placement),
        .epoch_id_to_device_grid = std::move(epoch_id_to_device_grid),
        .epoch_id_to_epoch_info = std::move(epoch_id_to_epoch_info),
        .num_epochs = placed ? (max_epoch_id_assigned - epoch_start_id) + 1 : 0,
    };

    validate_placer_solution(config, placer_solution);

    return placer_solution;
}


static std::vector<ChipId> get_chip_id_order(
    const ChipIdToPlacerWorkload& chip_to_placer_op_group_workload,
    NodeEpochType epoch_type)
{
    vector<ChipId> chip_id_order;
    for (auto& [chip_id, placer_workload] : chip_to_placer_op_group_workload) {
        chip_id_order.push_back(chip_id);
    }

    if (epoch_type == NodeEpochType::Backward) {
        std::reverse(std::begin(chip_id_order), std::end(chip_id_order));
    }
    return chip_id_order;
}

static PlacerSolution grayskull_placer(const PlacerConfig& config, const std::vector<std::string>& scheduled_ops)
{
    ChipIdToPlacerWorkload chip_to_placer_op_group_workload = lowering::generate_placer_workload(config, scheduled_ops);

    uint32_t current_epoch_id = 0;
    unordered_map<string, OpPlacement> name_to_op_placement;
    map<int, int> epoch_id_to_chip;
    unordered_map<int, vector<OpPlacement>> epoch_id_to_op_placement;
    auto e = EpochIdToDeviceGrid(config.get_available_rows_on_device(), config.device_grid.columns);
    unordered_map<int, EpochInfo> epoch_id_to_epoch_info;

    // For each chip, assign the op_group_workload to the chip and place it.
    for (auto epoch_type : {NodeEpochType::Forward, NodeEpochType::Backward, NodeEpochType::Optimizer}) {
        for (ChipId chip_id : get_chip_id_order(chip_to_placer_op_group_workload, epoch_type)) {
            auto& placer_workload = chip_to_placer_op_group_workload.at(chip_id);
            log_debug(tt::LogPlacer, "############################");
            log_debug(tt::LogPlacer, "Placing OPs onto chip_id: {}", chip_id);
            log_debug(tt::LogPlacer, "############################");

            auto chip_solution = place_onto_chip(config, placer_workload, current_epoch_id, epoch_type);

            name_to_op_placement.insert(
                chip_solution.name_to_op_placement.begin(),
                chip_solution.name_to_op_placement.end());

            epoch_id_to_op_placement.insert(
                chip_solution.epoch_id_to_op_placement.begin(),
                chip_solution.epoch_id_to_op_placement.end());

            e.epoch_id_to_device_grid.insert(
                chip_solution.epoch_id_to_device_grid.epoch_id_to_device_grid.begin(),
                chip_solution.epoch_id_to_device_grid.epoch_id_to_device_grid.end());


            for (auto epoch_id = current_epoch_id; epoch_id < current_epoch_id + chip_solution.num_epochs; ++epoch_id)
            {
                epoch_id_to_chip[epoch_id] = chip_id;
                epoch_id_to_epoch_info[epoch_id] = EpochInfo{
                    .global_epoch_id = epoch_id,
                    .temporal_epoch_id = epoch_id,
                    .spatial_epoch_id = 0,
                    .epoch_type = epoch_type,
                };
            }
            current_epoch_id += chip_solution.num_epochs;
        }
    }

    PlacerSolution placer_solution = {
        .name_to_op_placement = name_to_op_placement,
        .input_queue_to_grid_shape = config.input_queue_to_grid_shape,
        .name_to_queue_placement = {},
        .epoch_id_to_chip = epoch_id_to_chip,
        .epoch_id_to_subgraph_index = {},
        .epoch_id_to_op_placement = std::move(epoch_id_to_op_placement),
        .epoch_id_to_device_grid = std::move(e),
        .epoch_id_to_epoch_info = std::move(epoch_id_to_epoch_info),
        .num_epochs = current_epoch_id,
        .is_pipelined = true,
    };

    for (uint32_t i = 0; i < placer_solution.num_epochs; ++i) {
        if (placer_solution.epoch_id_to_op_placement.find(i) == placer_solution.epoch_id_to_op_placement.end()) {
            log_fatal(tt::LogPlacer, "Placer: Error found blank/missing epoch_id: {}", i);
        }
    }


    return placer_solution;
}

vector<vector<OpGroupToPlace>> get_placer_workload_grouped_by_chip_id(ChipIdToPlacerWorkload& chip_to_placer_op_group_workload, NodeEpochType epoch_type) {
    vector<OpGroupToPlace> placer_workload;

    for (auto& [chip_id, op_groups] : chip_to_placer_op_group_workload)
    {
        for (const auto& op_group : op_groups) {
            if (op_group.epoch_type == epoch_type)
            {
                placer_workload.push_back(op_group);
            }
        }
    }
    if (placer_workload.empty())
    {
        return {};
    }
    std::sort(placer_workload.begin(), placer_workload.end(),
            [](const OpGroupToPlace& a, const OpGroupToPlace& b) { return a.op_group_id < b.op_group_id; });

    uint32_t current_chip_id = placer_workload.at(0).chip_id;
    uint32_t previous_chip_id = current_chip_id;
    vector<vector<OpGroupToPlace>> placer_workload_grouped_by_chip_id;

    for (const OpGroupToPlace& op_group : placer_workload) {
        current_chip_id = op_group.chip_id;
        if (placer_workload_grouped_by_chip_id.empty() or previous_chip_id != current_chip_id) {
            placer_workload_grouped_by_chip_id.push_back({});
        }

        placer_workload_grouped_by_chip_id.back().push_back(op_group);
        previous_chip_id = op_group.chip_id;
    }
    if (epoch_type == NodeEpochType::Backward)
    {
        std::reverse(placer_workload_grouped_by_chip_id.begin(), placer_workload_grouped_by_chip_id.end());
    }

    return placer_workload_grouped_by_chip_id;
}

static bool can_place_epoch_onto_chip(const PlacerConfig& config, const PlacerSolution& chip_solution, uint32_t epoch_id, uint32_t proposed_chip_id)
{
    TT_ASSERT(not config.device_config.chips_with_mmio.empty(), "Expecting at least one chip with MMIO capability.");
    for (const auto &placement : chip_solution.epoch_id_to_op_placement.at(epoch_id))
    {
        if (config.output_queues_on_host and config.output_ops.find(placement.name) != config.output_ops.end() and
            std::find(config.device_config.chips_with_mmio.begin(), config.device_config.chips_with_mmio.end(), proposed_chip_id) == config.device_config.chips_with_mmio.end())
        {
            log_debug(tt::LogPlacer, "output op {} not on MMIO chip", placement.name);
            return false;
        }
        if (config.op_to_chip_id_assignment.find(placement.name) != config.op_to_chip_id_assignment.end() and
            config.op_to_chip_id_assignment.at(placement.name) != proposed_chip_id)
        {
            log_debug(tt::LogPlacer, "op {} assigned a chip id {} that is not the proposed chip id {}", placement.name, config.op_to_chip_id_assignment.at(placement.name), proposed_chip_id);
            return false;
        }
        if (auto maybe_chip_id_override = config.get_chip_id_override(placement.name); maybe_chip_id_override)
        {
            if (maybe_chip_id_override.value() != proposed_chip_id)
            {
                log_debug(tt::LogPlacer, "op {} has an override chip id {} that is not the proposed chip id {}", placement.name, maybe_chip_id_override.value(), proposed_chip_id);
                return false;
            }
        }
        // TODO: generalize to chips to avoid in case there are multichip chips on shelf (e.g. Nebula x2)
        if (env_as<bool>("PYBUDA_NEBULA_GALAXY_PLACER"))
        {
            if (std::find(config.device_config.chips_with_mmio.begin(), config.device_config.chips_with_mmio.end(), proposed_chip_id) !=
                config.device_config.chips_with_mmio.end())
            {
                if (!config.output_queues_on_host)
                {
                    return false;
                }
                else if (
                    config.output_queues_on_host and config.output_ops.find(placement.name) == config.output_ops.end())
                {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool validate_epoch_placement(const PlacerConfig& config, const PlacerSolution& chip_solution, uint32_t epoch_id)
{
    TT_ASSERT(not config.device_config.chips_with_mmio.empty(), "Expecting at least one chip with MMIO capability.");
    std::unordered_set<uint32_t> user_assigned_chip_ids;
    std::unordered_map<std::string, uint32_t> op_to_constraint;
    for (const auto& placement : chip_solution.epoch_id_to_op_placement.at(epoch_id))
    {
        if (auto maybe_chip_id_override = config.get_chip_id_override(placement.name); maybe_chip_id_override)
        {
            user_assigned_chip_ids.insert(maybe_chip_id_override.value());
            op_to_constraint[placement.name] = maybe_chip_id_override.value();
        }
        if (config.op_to_chip_id_assignment.find(placement.name) != config.op_to_chip_id_assignment.end())
        {
            user_assigned_chip_ids.insert(config.op_to_chip_id_assignment.at(placement.name));
            op_to_constraint[placement.name] = config.op_to_chip_id_assignment.at(placement.name);
        }
    }
    if (user_assigned_chip_ids.size() > 1)
    {
        log_fatal("Placer: Error, epoch {} has ops assigned to multiple chips: {}", epoch_id, op_to_constraint);
        return false;
    }
    else if (user_assigned_chip_ids.size() == 1)
    {
        uint32_t user_assigned_chip_id = *user_assigned_chip_ids.begin();

        // check that that is no conflict with constraints configured and the output op
        for (const auto& placement : chip_solution.epoch_id_to_op_placement.at(epoch_id))
        {
            if (config.output_queues_on_host and config.output_ops.find(placement.name) != config.output_ops.end() and
                std::find(config.device_config.chips_with_mmio.begin(), config.device_config.chips_with_mmio.end(), user_assigned_chip_id) ==
                    config.device_config.chips_with_mmio.end())
            {
                log_fatal(
                    "Placer: User has defined constraints on the ops: {} but there is an output op: {} on the same "
                    "epoch that must be assigned to an MMIO capable chip",
                    op_to_constraint,
                    placement.name);
                return false;
            }
        }
    }
    else
    {
        // no user assigned constraints so we are done
        return true;
    }

    return true;
}

static std::tuple<bool, std::uint32_t, std::uint32_t, std::uint32_t>
advance_epoch(
    const std::vector<std::uint32_t>& chip_ids,
    bool placing_forward,
    bool is_fwd_chip_direction,
    std::uint32_t current_chip_index, 
    std::uint32_t current_temporal_epoch_id,
    std::uint32_t current_spatial_epoch_id)
{
    std::uint32_t next_chip_index = current_chip_index;
    std::uint32_t next_temporal_epoch_id = current_temporal_epoch_id;
    std::uint32_t next_spatial_epoch_id = current_spatial_epoch_id + 1;
    
    // Snake the chip assignments so it's likely the first spatial epoch of a new temporal epoch 
    // reads activations from its own DRAM
    if (env_as<bool>("PYBUDA_PLACER_SNAKE"))
    {
        if (placing_forward) {
            if (next_chip_index == (chip_ids.size() - 1)) {
                placing_forward = false;
                next_spatial_epoch_id = 0;
                next_temporal_epoch_id++;
            }
            else {
                next_chip_index++;
            }
        } else {
            if (next_chip_index == 0) {
                placing_forward = true;
                next_spatial_epoch_id = 0;
                next_temporal_epoch_id++;
            }
            else {
                next_chip_index--;
            }
        }
    }
    else {
        bool wrap;

        if (is_fwd_chip_direction) {
            next_chip_index++;
            wrap = (next_chip_index >= chip_ids.size());
        } else {
            wrap = (next_chip_index == 0);
            if (!wrap) next_chip_index--;
        }
        if (wrap) {
            next_chip_index = is_fwd_chip_direction ? 0 : chip_ids.size() - 1;
            next_spatial_epoch_id = 0;
            next_temporal_epoch_id++;
        }
    }
    return {placing_forward, next_chip_index, next_temporal_epoch_id, next_spatial_epoch_id};
}

static PlacerSolution wormhole_placer(const PlacerConfig& config, const std::vector<std::string>& scheduled_ops)
{
    log_debug(LogPlacer, "schedule {}", scheduled_ops);

    unordered_map<string, OpPlacement> name_to_op_placement;
    map<int, int> epoch_id_to_chip;
    unordered_map<int, vector<OpPlacement>> epoch_id_to_op_placement;
    auto e = EpochIdToDeviceGrid(config.get_available_rows_on_device(), config.device_grid.columns);
    unordered_map<int, NodeEpochType> epoch_id_to_type;
    unordered_map<int, EpochInfo> epoch_id_to_epoch_info;

    std::vector<OpGroupToPlace> placer_workload = lowering::generate_wormhole_placer_workload(config, scheduled_ops);

    uint32_t current_epoch_id = 0;
    uint32_t current_temporal_epoch_id = 0;

    log_debug(LogPlacer, "WH Fracturing Constraints: {}", config.op_to_chip_id_assignment);

    for (const auto& [op, override] : config.op_to_overrides)
    {
        if (override.chip_id.has_value())
        {
            log_debug(LogPlacer, "WH Override: {}, {}", op, override.chip_id.value());
        }
    }

    std::vector<uint32_t> chip_ids = lowering::apply_chip_placement_policy(config.device_config, config.chip_placement_policy, config.chip_ids);

    for (auto epoch_type : {NodeEpochType::Forward, NodeEpochType::Backward, NodeEpochType::Optimizer})
    {
        uint32_t starting_epoch_id = current_epoch_id;
        auto chip_solution = place_onto_chip(config, placer_workload, current_epoch_id, epoch_type);
        name_to_op_placement.insert(
            chip_solution.name_to_op_placement.begin(),
            chip_solution.name_to_op_placement.end());

        current_epoch_id += chip_solution.num_epochs;

        // Everything's placed on one chip, but we need to split across available chips
        bool is_fwd_chip_direction = epoch_type == NodeEpochType::Forward or epoch_type == NodeEpochType::Optimizer;
        std::uint32_t current_chip_index = is_fwd_chip_direction ? 0 : chip_ids.size() - 1;
        std::uint32_t current_spatial_epoch_id = 0;
        bool placing_forward = true;
        bool enable_pipelined_placement = env_as<bool>("PYBUDA_WORMHOLE_PIPELINED_PLACER");

        std::uint32_t num_epochs_placed_on_chip = 0;
        for (std::uint32_t epoch = starting_epoch_id; epoch < current_epoch_id; epoch++)
        {
            validate_epoch_placement(config, chip_solution, epoch);

            bool valid_chip_assignment = true;
            // With snaking chip assignment we need one more attempt to account for transition between directions
            for (std::size_t attempt = 0; attempt < (2*chip_ids.size()+1); ++attempt)
            {
                std::uint32_t current_chip_id = chip_ids[current_chip_index];
                valid_chip_assignment = can_place_epoch_onto_chip(config, chip_solution, epoch, current_chip_id);

                if (valid_chip_assignment)
                {
                    for (auto &placement : chip_solution.epoch_id_to_op_placement.at(epoch))
                    {
                        placement.chip_id = current_chip_id;
                        name_to_op_placement[placement.name].chip_id = current_chip_id;
                    }

                    epoch_id_to_op_placement.insert(
                        chip_solution.epoch_id_to_op_placement.begin(),
                        chip_solution.epoch_id_to_op_placement.end());

                    e.epoch_id_to_device_grid.insert(
                        chip_solution.epoch_id_to_device_grid.epoch_id_to_device_grid.begin(),
                        chip_solution.epoch_id_to_device_grid.epoch_id_to_device_grid.end());

                    epoch_id_to_chip[epoch] = current_chip_id;
                    epoch_id_to_type[epoch] = epoch_type;

                    epoch_id_to_epoch_info[epoch] = EpochInfo{
                        .global_epoch_id = (uint32_t)epoch,
                        .temporal_epoch_id = (uint32_t)current_temporal_epoch_id,
                        .spatial_epoch_id = (uint32_t)(current_spatial_epoch_id % chip_ids.size()),
                        .epoch_type = epoch_type
                    };
                    num_epochs_placed_on_chip++;
                }

                if (enable_pipelined_placement)
                {
                    std::uint32_t num_epochs_per_chip = std::ceil(float(chip_solution.num_epochs) / chip_ids.size());
                    if (not valid_chip_assignment or (num_epochs_placed_on_chip >= num_epochs_per_chip))
                    {
                        std::tie(placing_forward, current_chip_index, current_temporal_epoch_id, current_spatial_epoch_id) = 
                            advance_epoch(chip_ids, placing_forward, is_fwd_chip_direction, current_chip_index, current_temporal_epoch_id, current_spatial_epoch_id);
                        num_epochs_placed_on_chip = 0;
                    }
                }
                else
                {
                    std::tie(placing_forward, current_chip_index, current_temporal_epoch_id, current_spatial_epoch_id) = 
                        advance_epoch(chip_ids, placing_forward, is_fwd_chip_direction, current_chip_index, current_temporal_epoch_id, current_spatial_epoch_id);

                }

                if (valid_chip_assignment)
                {
                    break;
                }
            }
            TT_LOG_ASSERT(valid_chip_assignment, "Invalid chip assignment for temporal epoch {} {}, spatial epoch {}, chip {}", epoch_type, current_temporal_epoch_id, current_spatial_epoch_id, current_chip_index);
        }
        log_debug(tt::LogPlacer, "Placing {} epochs onto chip_id: {}", chip_solution.num_epochs, 0);
    }


    int current_temporal_epoch = 0;
    map<int, int> chip_id_to_spatial_epoch_index;
    for (uint32_t i = 0; i < config.chip_ids.size(); ++i) {
        chip_id_to_spatial_epoch_index[config.chip_ids[i]] = i;
    }

    map<int, int> chip_to_current_temporal_epoch;
    map<int, std::set<int>> temporal_epoch_id_to_spatial_epochs;
    map<int, NodeEpochType> temporal_epoch_id_to_epoch_type;

    log_debug(tt::LogPlacer, "## Wormhole Placement Summary ##");
    NodeEpochType prev_epoch_type = NodeEpochType::Forward;
    for (const auto& [epoch_id, chip_id] : epoch_id_to_chip) {
        if (chip_to_current_temporal_epoch.find(chip_id) != chip_to_current_temporal_epoch.end()) {
            int last_recorded_temporal_epoch = chip_to_current_temporal_epoch[chip_id];
            current_temporal_epoch = std::max(current_temporal_epoch, last_recorded_temporal_epoch + 1);
        }
        bool is_new_temporal_epoch_requested = false;
        for (const auto &placement : epoch_id_to_op_placement.at(epoch_id))
        {
            if (config.ops_tagged_for_temporal_epoch_break.find(placement.name) != config.ops_tagged_for_temporal_epoch_break.end())
            {
                is_new_temporal_epoch_requested = true;
            }
        }
        if (epoch_id_to_type[epoch_id] != prev_epoch_type)
        {
            is_new_temporal_epoch_requested = true;
        }
        prev_epoch_type = epoch_id_to_type[epoch_id];

        if (is_new_temporal_epoch_requested and temporal_epoch_id_to_spatial_epochs[current_temporal_epoch].size() > 0)
        {
            // Since there are already op-placements on this current temporal epoch,
            // and a new temporal epoch is requested, we'll just increment
            current_temporal_epoch += 1;
        }

        int current_spatial_epoch = chip_id_to_spatial_epoch_index.at(chip_id);
        epoch_id_to_epoch_info[epoch_id] = EpochInfo{
            .global_epoch_id = (uint32_t)epoch_id,
            .temporal_epoch_id = (uint32_t)current_temporal_epoch,
            .spatial_epoch_id = (uint32_t)current_spatial_epoch,
            .epoch_type = epoch_id_to_type.at(epoch_id)
        };
        log_debug(tt::LogPlacer, "Epoch: {}, Chip: {}, Temporal Epoch: {}, Spatial Epoch: {}",
                epoch_id, chip_id, current_temporal_epoch, current_spatial_epoch);

        chip_to_current_temporal_epoch[chip_id] = current_temporal_epoch;
        temporal_epoch_id_to_spatial_epochs[current_temporal_epoch].insert(current_spatial_epoch);
        temporal_epoch_id_to_epoch_type[current_temporal_epoch] = epoch_id_to_type[epoch_id];
    }

    for (std::uint32_t temporal_epoch_id = 0; temporal_epoch_id < temporal_epoch_id_to_epoch_type.size(); ++temporal_epoch_id)
    {
        const auto& spatial_epochs = temporal_epoch_id_to_spatial_epochs[temporal_epoch_id];
        for (uint32_t spatial_epoch_index = 0; spatial_epoch_index < config.chip_ids.size(); ++spatial_epoch_index)
        {
            if (spatial_epochs.find(spatial_epoch_index) == spatial_epochs.end())
            {
                // need to insert empty epochs
                // NB: assume temporal epoch should share the same epoch-type
                int global_epoch_id = current_epoch_id++;

                epoch_id_to_chip[global_epoch_id] = config.chip_ids.at(spatial_epoch_index);
                epoch_id_to_op_placement[global_epoch_id] = {};
                e.epoch_id_to_device_grid[global_epoch_id] = {};
                epoch_id_to_epoch_info[global_epoch_id] = EpochInfo{
                    .global_epoch_id = (uint32_t)global_epoch_id,
                    .temporal_epoch_id = (uint32_t)temporal_epoch_id,
                    .spatial_epoch_id = (uint32_t)spatial_epoch_index,
                    .epoch_type = temporal_epoch_id_to_epoch_type.at(temporal_epoch_id)
                };

                log_debug(tt::LogPlacer, "Inserting Empty Epoch: {}, Chip: {}, Temporal Epoch: {}, Spatial Epoch: {}",
                        global_epoch_id, config.chip_ids.at(spatial_epoch_index), temporal_epoch_id, spatial_epoch_index);
            }
        }
    }

    // if user has defined manual configuration for `place_on_new_chip`, the multichip
    // wormhole placement is configured to be pipelined
    bool is_pipelined = not config.ops_tagged_for_chip_id_break.empty();

    PlacerSolution placer_solution = {
        .name_to_op_placement = name_to_op_placement,
        .input_queue_to_grid_shape = config.input_queue_to_grid_shape,
        .name_to_queue_placement = {},
        .epoch_id_to_chip = epoch_id_to_chip,
        .epoch_id_to_subgraph_index = {},
        .epoch_id_to_op_placement = std::move(epoch_id_to_op_placement),
        .epoch_id_to_device_grid = std::move(e),
        .epoch_id_to_epoch_info = std::move(epoch_id_to_epoch_info),
        .num_epochs = current_epoch_id,
        .is_pipelined = is_pipelined,
    };

    return placer_solution;
}


PlacerSolution galaxy_placer(const PlacerConfig &config, const std::vector<std::string> &scheduled_ops)
{

    // Group ops into fwd, bwd, grad, opt
    std::unordered_map<std::string, std::vector<std::string>> op_megagroup;

    bool split_grad = not env_as<bool>("PYBUDA_GALAXY_PLACER_COMBINE_GRAD");
    bool split_recompute = not env_as<bool>("PYBUDA_GALAXY_PLACER_COMBINE_RECOMPUTE");

    for (auto op_name : scheduled_ops)
    {
        NodeEpochType epoch_type = config.op_to_epoch_type.at(op_name);
        bool is_gradient_op = config.op_to_grad_op.at(op_name);
        //bool is_recompute_op = config.op_to_recompute_op.at(op_name);
        // TODO
        bool is_recompute_op = false;

        if (epoch_type == NodeEpochType::Forward)
            op_megagroup["fwd"].push_back(op_name);
        else if (epoch_type == NodeEpochType::Optimizer)
            op_megagroup["opt"].push_back(op_name);
        else {
            // bwd
            if (split_recompute && is_recompute_op)
                op_megagroup["rcmp"].push_back(op_name);
            else if (split_grad && is_gradient_op) 
                op_megagroup["grad"].push_back(op_name);
            else 
                op_megagroup["bwd"].push_back(op_name);
        }
    }

    std::unordered_map<string, OpPlacement> name_to_op_placement;
    std::map<int, int> epoch_id_to_chip;
    std::unordered_map<int, vector<OpPlacement>> epoch_id_to_op_placement;
    auto e = EpochIdToDeviceGrid(config.get_available_rows_on_device(), config.device_grid.columns);
    std::unordered_map<int, EpochInfo> epoch_id_to_epoch_info;

    std::uint32_t current_epoch_id = 0;
    std::uint32_t current_temporal_epoch_id = 0;

    for (auto &type : std::vector<std::string>{"fwd", "rcmp", "bwd", "grad", "opt"})
    {
        if (op_megagroup[type].size() == 0)
            continue;

        bool chip_direction = (type == "fwd") || (type == "rcmp"); // incrementing chip IDs
        
        std::uint32_t current_spatial_epoch_id = 0;
        std::vector<OpGroupToPlace> placer_workload = lowering::generate_simple_placer_workload(config, op_megagroup[type]);
        std::uint32_t starting_epoch_id = current_epoch_id;
        auto chip_solution = place_onto_chip(config, placer_workload, current_epoch_id);
        current_epoch_id += chip_solution.num_epochs;

        // Everything's placed on one chip, but we need to split across available chips
        std::uint32_t current_chip_index = chip_direction ? 0 : config.chip_ids.size() - 1;
        for (std::uint32_t epoch = starting_epoch_id; epoch < current_epoch_id; epoch++)
        {
            std::uint32_t current_chip_id = config.chip_ids[current_chip_index];
            for (auto &placement : chip_solution.epoch_id_to_op_placement.at(epoch))
                placement.chip_id = current_chip_id;

            name_to_op_placement.insert(
                chip_solution.name_to_op_placement.begin(),
                chip_solution.name_to_op_placement.end());

            for (auto &placement : chip_solution.epoch_id_to_op_placement.at(epoch))
                name_to_op_placement[placement.name].chip_id = current_chip_id;

            epoch_id_to_op_placement.insert(
                chip_solution.epoch_id_to_op_placement.begin(),
                chip_solution.epoch_id_to_op_placement.end());

            e.epoch_id_to_device_grid.insert(
                chip_solution.epoch_id_to_device_grid.epoch_id_to_device_grid.begin(),
                chip_solution.epoch_id_to_device_grid.epoch_id_to_device_grid.end());

            epoch_id_to_chip[epoch] = current_chip_id;
            NodeEpochType epoch_type = (type == "fwd") ? NodeEpochType::Forward :
                                       (type == "opt") ? NodeEpochType::Optimizer : 
                                                         NodeEpochType::Backward;

            epoch_id_to_epoch_info[epoch] = EpochInfo{
                .global_epoch_id = (uint32_t)epoch,
                .temporal_epoch_id = (uint32_t)current_temporal_epoch_id,
                .spatial_epoch_id = (uint32_t)(current_spatial_epoch_id % config.chip_ids.size()),
                .epoch_type = epoch_type
            };

            if (epoch < current_epoch_id - 1) {
                current_spatial_epoch_id++;

                bool wrap;

                if (chip_direction) {
                    current_chip_index++;
                    wrap = (current_chip_index >= config.chip_ids.size());
                } else {
                    wrap = (current_chip_index == 0);
                    if (!wrap) current_chip_index--;
                }
                if (wrap) {
                    current_chip_index = chip_direction ? 0 : config.chip_ids.size() - 1;
                    current_spatial_epoch_id = 0;
                    current_temporal_epoch_id ++;
                }
            }
        }

        current_temporal_epoch_id++;
    }

    PlacerSolution placer_solution = {
        .name_to_op_placement = name_to_op_placement,
        .input_queue_to_grid_shape = config.input_queue_to_grid_shape,
        .name_to_queue_placement = {},
        .epoch_id_to_chip = epoch_id_to_chip,
        .epoch_id_to_subgraph_index = {},
        .epoch_id_to_op_placement = std::move(epoch_id_to_op_placement),
        .epoch_id_to_device_grid = std::move(e),
        .epoch_id_to_epoch_info = std::move(epoch_id_to_epoch_info),
        .num_epochs = current_epoch_id,
        .is_pipelined = false,
    };

    return placer_solution;
}


PlacerSolution placer(const PlacerConfig& config, const vector<string>& scheduled_ops)
{
    lowering::validate_placer_config(config);

    // TODO: expose as config... for now, quick testing through env variable
    if (env_as<bool>("PYBUDA_GALAXY_PLACER"))
    {
        return galaxy_placer(config, scheduled_ops);
    }

    if (config.device_config.is_grayskull()) {
        return grayskull_placer(config, scheduled_ops);
    } else {
        TT_ASSERT((config.device_config.is_wormhole() || config.device_config.is_wormhole_b0()), "Placer Failed: Unknown device arch name.");
        return wormhole_placer(config, scheduled_ops);
    }
}

void place_on_new_epoch(PlacerConfig& config, const string& op_name)
{
    config.ops_tagged_for_epoch_break.insert(op_name);
}

void place_on_new_chip(PlacerConfig& config, const string& op_name)
{
    config.ops_tagged_for_chip_id_break.insert(op_name);
}


void dump_placer_solution_json_to_file(const PlacerSolution& solution)
{
    json placer_solution_json = solution.to_json();

    const string DEFAULT_FILEPATH = "placement.json";
    ofstream o(DEFAULT_FILEPATH);
    o << setw(4) << placer_solution_json;
    o.close();

}

std::ostream& operator<<(std::ostream& os, const Coord& coord)
{
    os << "Coord{";
    os << ".row= " << coord.row << ", ";
    os << ".col= " << coord.col << ", ";
    os << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const CoordRange& coord_range)
{
    os << "CoordRange{";
    os << ".start= " << coord_range.start << ", ";
    os << ".end= " << coord_range.end << ", ";
    os << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const PlacerOpOverride& override)
{
    os << "PlacerOpOverride{";
    os << ".transpose= " << override.transpose_op << ", ";
    if (override.grid_start.has_value())
    {
        os << ".grid_start= " << override.grid_start.value() << ", ";
    }
    else
    {
        os << ".grid_start= " << "None" << ", ";
    }
    if (override.chip_id.has_value())
    {
        os << ".chip_id= " << override.chip_id.value() << ", ";
    }
    else
    {
        os << ".chip_id= " << "None" << ", ";
    }
    os << ".temporal_epoch_break= " << (override.temporal_epoch_break ? "true" : "false") << ", ";

    os << "}";
    return os;
}

std::unordered_map<std::string, placer::PlacerOpOverride> match_op_names_to_placer_overrides(
    graphlib::Graph* graph,
    std::vector<std::pair<std::variant<std::string, graphlib::query::NodePredicate>, placer::PlacerOpOverride>> const&
        predicates_to_overrides)
{
    std::unordered_map<std::string, placer::PlacerOpOverride> op_names_to_placer_overrides;
    auto is_op_node = graphlib::query::predicate_op_node_type();
    for (auto const& [string_or_predicate, override] : predicates_to_overrides)
    {
        if (std::string const* s = std::get_if<std::string>(&string_or_predicate))
        {
            op_names_to_placer_overrides[*s] = override;
        }
        else if (graphlib::query::NodePredicate const* p = std::get_if<graphlib::query::NodePredicate>(&string_or_predicate))
        {
            for (graphlib::Node* node : graphlib::query::filter_nodes(graph, *p & is_op_node))
            {
                if (op_names_to_placer_overrides.find(node->name()) != op_names_to_placer_overrides.end())
                  log_fatal("Overlapping placer override predicates for node: {}", node->name());
                op_names_to_placer_overrides[node->name()] = override;
            }
        }
    }
    return op_names_to_placer_overrides;
}

std::vector<std::vector<std::string>> match_op_names_to_breaks(
    graphlib::Graph* graph, const PredicatesToBreaks& predicates_to_breaks)
{
    std::vector<std::vector<std::string>> op_names_to_breaks;
    op_names_to_breaks.reserve(predicates_to_breaks.size());
    auto is_op_node = graphlib::query::predicate_op_node_type();
    for (auto const& outer : predicates_to_breaks)
    {
        if (auto* p = std::get_if<graphlib::query::NodePredicate>(&outer))
        {
            for (graphlib::Node* node : graphlib::query::filter_nodes(graph, *p & is_op_node))
            {
                op_names_to_breaks.push_back({node->name()});
            }
        }
        else if (
            auto* sublist = std::get_if<std::vector<std::variant<std::string, graphlib::query::NodePredicate>>>(&outer))
        {
            op_names_to_breaks.emplace_back();
            auto& back = op_names_to_breaks.back();
            back.reserve(sublist->size());
            for (auto const& elem : *sublist)
            {
                if (std::string const* s = std::get_if<std::string>(&elem))
                {
                    back.push_back(*s);
                }
                else if (graphlib::query::NodePredicate const* p = std::get_if<graphlib::query::NodePredicate>(&elem))
                {
                    for (graphlib::Node* node : graphlib::query::filter_nodes(graph, *p & is_op_node))
                    {
                        back.push_back(node->name());
                    }
                }
            }
        }
    }
    return op_names_to_breaks;
}

}  // namespace placer
}  // namespace tt
