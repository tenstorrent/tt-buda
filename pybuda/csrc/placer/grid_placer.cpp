// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/grid_placer.hpp"

#include "placer/utils.hpp"
#include "placer/lowering_utils.hpp"
#include "placer/exceptions.hpp"

#include "graph_lib/defines.hpp"
#include "utils/logger.hpp"
#include "utils/assert.hpp"

#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <exception>

using tt::LogPlacer;

namespace tt {
namespace placer {

namespace device_grid
{
DeviceGrid create_empty_device_grid(uint32_t rows, uint32_t columns)
{
    return vector<vector<uint32_t>>(rows, vector<uint32_t>(columns, 0));
}
DeviceGrid superposition(const DeviceGrid& a, const DeviceGrid& b)
{
    const uint32_t a_rows = a.size();
    const uint32_t a_cols = a.at(0).size(); 
    const uint32_t b_rows = a.size();
    const uint32_t b_cols = a.at(0).size(); 
    TT_ASSERT(a_rows == b_rows);
    TT_ASSERT(a_cols == b_cols);

    DeviceGrid new_device_grid = create_empty_device_grid(a_rows, a_cols);
    for (uint32_t i = 0; i < a_rows; ++i) {
        for (uint32_t j = 0; j < a_cols; ++j) {
            if (a[i][j])
            {
                new_device_grid[i][j] = a[i][j];
            }
            else if (b[i][j])
            {
                new_device_grid[i][j] = b[i][j];
            }
            else
            {
                new_device_grid[i][j] = 0;
            }
        }
    }
    return new_device_grid;
}

bool can_place_on_device_grid(const DeviceGrid& device_grid, const Coord& start, const GridShape& shape)
{
    const uint32_t available_rows_on_grid = device_grid.size();
    const uint32_t available_columns_on_grid = device_grid.at(0).size(); 

    if (start.row + shape.rows > available_rows_on_grid)
    {
        return false;
    }
    else if (start.col + shape.columns > available_columns_on_grid)
    {
        return false;
    }

    for (uint32_t i = start.row; i < start.row + shape.rows; ++i) {
        for (uint32_t j = start.col; j < start.col + shape.columns; ++j) {
            if (device_grid.at(i).at(j) != 0) {
                return false;
            }
        }
    }
    return true;
}

bool contains_empty_device_grid(const DeviceGrid& device_grid)
{
    if (device_grid.empty())
    {
        return true;
    }
    if (device_grid[0].empty())
    {
        return true;
    }

    for (uint32_t i = 0; i < device_grid.size(); ++i) {
        for (uint32_t j = 0; j < device_grid.at(0).size(); ++j) {
            if (device_grid[i][j] != 0) {
                return false;
            }
        }
    }
    return true;
}

void fill_device_grid_with_placement(DeviceGrid& device_grid, const Coord& op_start, const GridShape& op_grid_shape)
{ 
    for (uint32_t i = op_start.row; i < op_start.row + op_grid_shape.rows; ++i) {
        for (uint32_t j = op_start.col; j < op_start.col + op_grid_shape.columns; ++j) {
            device_grid.at(i).at(j) = 1;
        }
    }
}

void print_device_grid(const DeviceGrid& device_grid)
{ 
    for (uint32_t i = 0; i < device_grid.size(); ++i) {
        for (uint32_t j = 0; j < device_grid.at(0).size(); ++j) {
            std::cout << " " << device_grid.at(i).at(j);
        }
        std::cout << std::endl;
    }
}
std::optional<Coord> get_first_free_location(const DeviceGrid& device_grid, const GridShape& op_grid_shape)
{
    const uint32_t rows = device_grid.size();
    const uint32_t columns = device_grid.at(0).size(); 
    for (uint32_t i = 0; i < rows; ++i)
    { 
        for (uint32_t j = 0; j < columns; ++j)
        { 
            if (i + op_grid_shape.rows > rows or j + op_grid_shape.columns > columns)
            {
                continue;
            } 
             
            bool is_valid = true;
            for (uint32_t op_i = i; op_i < i + op_grid_shape.rows; ++ op_i)
            {
                for (uint32_t op_j = j; op_j < j + op_grid_shape.columns; ++ op_j)
                {
                    if (device_grid[op_i][op_j] != 0)
                    { 
                        is_valid = false;  
                        break;
                    }
                }
            }
            if (is_valid)
            {  
                return Coord{.row = i, .col = j};
            }
        } 
    }
    return std::nullopt;
}

std::optional<Coord> get_next_grid_coordinate(const DeviceGrid& device_grid, const GridShape& op_grid_shape)
{
    return get_first_free_location(device_grid, op_grid_shape);
}
} // namespace device_grid

enum class PlacerState {
    PLACE_WITH_RELATIVE_OFFSETS,
    INCREMENT_EPOCH_AND_PLACE_WITH_RELATIVE_OFFSETS,
    ALLOW_EPOCH_INCREMENTS,
};

struct DeviceGridConfig {
    bool enable_relative_offsets;
    bool increment_epoch;
    bool allow_increment;
};

uint32_t get_active_placement_row_height(const DeviceGrid& device_grid, const std::optional<Coord>& candidate)
{ 
    if (not candidate.has_value())
    {
        return 0;
    }

    const uint32_t device_grid_rows = device_grid.size(); 
    const uint32_t device_grid_columns = device_grid[0].size();
    bool device_R_larger_than_C = (device_grid_rows > device_grid_columns);
    uint32_t row_height = 0; 
    uint32_t i = candidate.value().row;
    uint32_t j = candidate.value().col;
    
    for (int op_j = j-1; i+row_height < device_grid_rows && op_j >= 0; --op_j) 
    {
        if(device_grid[i+row_height][op_j] != 0)
        { 
            for (uint32_t op_i = i+row_height; op_i < device_grid_rows; ++op_i) 
            {
                if (device_grid[op_i][op_j] == 0) 
                {
                    break;
                }
                row_height++;
            }
            break; 
        }

    }
        
    return (row_height == 0 and device_R_larger_than_C) ? (device_grid_rows - i) : row_height;
}


DeviceGridConfig get_device_grid_config_from_strategy(
    PlacerState grid_placer_strategy, bool induce_epoch_increment, bool allow_increment=false)
{
    bool enable_relative_offsets = env_as<bool>("PYBUDA_TRIPLET_PLACEMENT");
    switch(grid_placer_strategy)
    {
        case PlacerState::PLACE_WITH_RELATIVE_OFFSETS:
        {
            return DeviceGridConfig{
                .enable_relative_offsets=enable_relative_offsets,
                .increment_epoch=induce_epoch_increment,
                .allow_increment=allow_increment,
            };
        }
        case PlacerState::INCREMENT_EPOCH_AND_PLACE_WITH_RELATIVE_OFFSETS:
        {
            return DeviceGridConfig{
                .enable_relative_offsets=enable_relative_offsets,
                .increment_epoch=true,
                .allow_increment=false,
            };
        }
        case PlacerState::ALLOW_EPOCH_INCREMENTS:
        {
            return DeviceGridConfig{
                .enable_relative_offsets=false,
                .increment_epoch=true,
                .allow_increment=true,
            };
        }
    }
    TT_ASSERT("Failed to configure device_grid_config.");
    return DeviceGridConfig{};
}

bool should_try_auto_transpose_op(
    const DeviceGrid& device_grid, 
    const GridShape& op_grid_shape, 
    const bool enable_auto_transposing, 
    const bool manually_transpose_this_op)
{
    const uint32_t device_grid_r = device_grid.size();
    const uint32_t device_grid_c = device_grid[0].size();
    bool is_transposable = op_grid_shape.columns <= device_grid_r and op_grid_shape.rows <= device_grid_c;
    TT_LOG_ASSERT((not manually_transpose_this_op) or (manually_transpose_this_op and is_transposable), 
                  "Manually passed op is not transposable, op-grid-shape: {}x{}, device-grid: {}x{}", 
                  op_grid_shape.rows, 
                  op_grid_shape.columns, 
                  device_grid_r, 
                  device_grid_c);

    // check for auto-transpose 
    bool try_auto_transpose = is_transposable and enable_auto_transposing and (op_grid_shape.rows > op_grid_shape.columns);  
    return try_auto_transpose and not manually_transpose_this_op;
}

bool apply_auto_transpose(
    const bool try_auto_transpose, 
    const DeviceGrid& device_grid,
    const std::optional<Coord>& coordinate_to_try,
    const uint32_t op_grid_R)
{    
    return try_auto_transpose and (not coordinate_to_try.has_value() or get_active_placement_row_height(device_grid, coordinate_to_try) < op_grid_R);
}

std::tuple<Coord, uint32_t, bool> get_placed_coordinate(
    const string& op_name,
    const GridShape& op_grid_shape,
    EpochIdToDeviceGrid& e,
    uint32_t candidate_epoch_id,
    std::optional<Coord> coordinate_to_try,
    bool allow_increment,
    const bool enable_auto_transposing,
    const std::unordered_map<std::string, PlacerOpOverride> &op_to_overrides)
{
    TT_ASSERT(coordinate_to_try.has_value());
    GridShape op_grid_shape_local(op_grid_shape.rows, op_grid_shape.columns);

    e.initialize_device_grid(candidate_epoch_id);

    std::optional<PlacerOpOverride> op_override = std::nullopt;
    if (op_to_overrides.find(op_name) != op_to_overrides.end())
    {
        op_override = op_to_overrides.at(op_name);
    }
    bool manually_transpose_this_op = op_override.has_value() ? op_override.value().transpose_op : false;

    bool try_auto_transpose = should_try_auto_transpose_op(
        e.epoch_id_to_device_grid.at(candidate_epoch_id), 
        op_grid_shape, 
        enable_auto_transposing, 
        manually_transpose_this_op); 

    if (manually_transpose_this_op)
    {
        op_grid_shape_local = GridShape(op_grid_shape.columns, op_grid_shape.rows);
    }

    // Try conditional after transposing
    if (op_override.has_value() and op_override.value().grid_start.has_value())
    {
        const auto& user_grid_start = op_override.value().grid_start.value();
        bool can_place_with_user_override = e.can_place_on_device_grid(op_name, candidate_epoch_id, user_grid_start, op_grid_shape_local);
        if (can_place_with_user_override)
        {
            log_debug(LogPlacer, "{} has an op override is now placed at: {}", op_name, op_override.value().grid_start.value());
            coordinate_to_try = op_override.value().grid_start;
        }
        else if (not e.satisfies_constraints(op_name, user_grid_start, op_grid_shape_local))
        {
            const Coord& user_grid_start = op_override.value().grid_start.value();

            for (const auto& [constraint_name, constraint_grid] : e.op_to_constraints)
            {
                if (op_name != constraint_name and not device_grid::can_place_on_device_grid(constraint_grid, user_grid_start, op_grid_shape_local))
                {
                    throw FailToSatisfyConflictingConstraint(
                        fmt::format("OpPlacement for {} to start at {} it conflicts with the constraint placed at {} : {}.",
                            op_name,
                            op_override.value().grid_start.value(),
                            constraint_name,
                            op_to_overrides.at(constraint_name).grid_start.value()
                        )
                    );
                }
            }
        }
        else
        {
            throw FailToSatisfyPlacementConstraint(
                fmt::format("User has specified an override of the OpPlacement for {} to start at {} but it is not valid.",
                    op_name,
                    op_override.value().grid_start.value()
                )
            );
        }
    }
 
    while (not e.can_place_on_device_grid(op_name, candidate_epoch_id, coordinate_to_try.value(), op_grid_shape_local))
    {  
        const DeviceGrid& device_grid = e.epoch_id_to_device_grid.at(candidate_epoch_id);
        coordinate_to_try = e.get_next_grid_coordinate(op_name, candidate_epoch_id, op_grid_shape_local);
        if (allow_increment and not coordinate_to_try.has_value())
        {
            candidate_epoch_id += 1;
            e.initialize_device_grid(candidate_epoch_id);
            coordinate_to_try = Coord{.row = 0, .col = 0};
        } 
        else if (apply_auto_transpose(try_auto_transpose, device_grid, coordinate_to_try, op_grid_shape.rows))
        {
            std::optional<Coord> coord_T = e.get_next_grid_coordinate(
                op_name, candidate_epoch_id, GridShape(op_grid_shape_local.columns, op_grid_shape_local.rows));
            if (coord_T.has_value() and (not coordinate_to_try.has_value() or coord_T.value() < coordinate_to_try.value() || coord_T.value() == coordinate_to_try.value()))
            {
                op_grid_shape_local = GridShape(op_grid_shape.columns, op_grid_shape.rows);
                coordinate_to_try = Coord{.row = coord_T.value().row, .col = coord_T.value().col};
            } 
        }

        if (not coordinate_to_try.has_value())
        {
            throw FailToPlaceOnCurrentEpoch("ran out of valid placements");
        }
    }
    TT_ASSERT(coordinate_to_try.has_value());
    bool is_transposed = (op_grid_shape.rows != op_grid_shape_local.rows);  
    return {coordinate_to_try.value(), candidate_epoch_id, is_transposed};
}

std::tuple<vector<DeviceGridPlacement>, Coord> place_on_grid_helper(
    const vector<string>& op_names,
    const unordered_map<string, GridShape>& op_to_grid_shape,
    const std::unordered_map<std::string, PlacerOpOverride> &op_to_overrides,
    const bool enable_auto_transposing,
    const DeviceGridConfig config,
    const DeviceGrid& device_grid,
    std::optional<Coord> starting_coordinate,
    bool return_after_one_epoch = false,
    const unordered_map<string, CoordOffset>& op_name_to_relative_offset_from_first_op = {},
    const std::unordered_map<std::string, DeviceGrid> constraints = {})
{
    uint32_t current_epoch_id = 0;
    if (not starting_coordinate.has_value())
    {
        starting_coordinate = Coord{.row=0, .col=0};
    }

    auto e_copy = EpochIdToDeviceGrid(device_grid.size(), device_grid.at(0).size());
    e_copy.initialize_device_grid(current_epoch_id, device_grid);
    e_copy.add_constraints(constraints);

    if (config.increment_epoch and not e_copy.contains_empty_grid(current_epoch_id))
    {
        current_epoch_id += 1;
        starting_coordinate = {.row = 0, .col = 0};
    }

    Coord placed_coordinate = starting_coordinate.value();
    uint32_t candidate_epoch_id = current_epoch_id;

    vector<DeviceGridPlacement> op_placements;
    std::optional<Coord> first_placement_start = std::nullopt; 
    bool is_op_transpose_enabled = enable_auto_transposing or op_to_overrides.size() > 0;
 
    for (const auto& op_name : op_names)
    {
        Coord previous_coordinate = placed_coordinate; 
        GridShape op_grid_shape = op_to_grid_shape.at(op_name); 

        // NB: relative offsets are not applied when op is transposed
        bool has_relative_offset = op_name_to_relative_offset_from_first_op.find(op_name) != op_name_to_relative_offset_from_first_op.end();
        if (config.enable_relative_offsets and has_relative_offset and first_placement_start.has_value() and not is_op_transpose_enabled)  
        {
            const CoordOffset& offset = op_name_to_relative_offset_from_first_op.at(op_name);
            placed_coordinate = first_placement_start.value() + offset;
        }

        bool op_transposed = false; 
        std::tie(placed_coordinate, candidate_epoch_id, op_transposed) = get_placed_coordinate(
            op_name,
            op_grid_shape,
            e_copy,
            candidate_epoch_id,
            placed_coordinate,
            config.allow_increment,
            enable_auto_transposing,
            op_to_overrides
        ); 
 
        if (return_after_one_epoch and candidate_epoch_id != 0)
        {
            return {op_placements, previous_coordinate};
        }
        
        if (op_transposed)
        {
            op_grid_shape = GridShape(op_grid_shape.columns, op_grid_shape.rows);
        }
        previous_coordinate = placed_coordinate;  
        e_copy.fill_device_grid_with_placement(candidate_epoch_id, placed_coordinate, op_grid_shape);

        op_placements.push_back(DeviceGridPlacement{
            .op_name = op_name,
            .device_grid_index = candidate_epoch_id,
            .placed_cores = CoordRange{.start = placed_coordinate, .end = placed_coordinate + op_grid_shape},
            .grid_transpose = op_transposed,
        });

        if (not first_placement_start.has_value())
        {
            first_placement_start = placed_coordinate;
        } 
    }
    return {op_placements, placed_coordinate};
}

std::string to_string(PlacerState state)
{
    switch (state)
    {
        case PlacerState::PLACE_WITH_RELATIVE_OFFSETS:
        {
            return "PLACE_WITH_RELATIVE_OFFSETS";
        }
        case PlacerState::INCREMENT_EPOCH_AND_PLACE_WITH_RELATIVE_OFFSETS:
        {
            return "INCREMENT_EPOCH_AND_PLACE_WITH_RELATIVE_OFFSETS";
        }
        case PlacerState::ALLOW_EPOCH_INCREMENTS:
        {
            return "ALLOW_EPOCH_INCREMENTS";
        }
    }
    TT_ASSERT("PlacerState with undefined string conversion.");
    return "";
}

std::optional<DeviceGridPlacement> place_one_op(
    const string op,
    const unordered_map<string, GridShape>& op_to_grid_shape,
    const DeviceGrid& device_grid,
    const std::unordered_map<std::string, PlacerOpOverride> &op_to_overrides,
    const bool enable_auto_transposing,
    std::optional<Coord> starting_coordinate)
{
    try
    {
        std::vector<std::string> ops = {op};
        auto [device_grid_placements, last_placed] = place_on_grid_helper(
            ops,
            op_to_grid_shape,
            op_to_overrides, 
            enable_auto_transposing,
            get_device_grid_config_from_strategy(PlacerState::PLACE_WITH_RELATIVE_OFFSETS, false),
            device_grid,
            starting_coordinate,
            true /* return_after_one_epoch */
        );
        TT_ASSERT(device_grid_placements.size() == 1);
        return device_grid_placements.at(0);
    }
    catch (const FailToPlaceOnCurrentEpoch& e)
    {
        // can't place on current epoch, return back to user
    }
    catch (...)
    {
        TT_ASSERT("place_one_op: caught unhandled exception");
    }
    return std::nullopt;
}

std::tuple<vector<DeviceGridPlacement>, Coord> place_on_grid(
    const OpGroupToPlace& op_group_to_place,
    const unordered_map<string, GridShape>& op_to_grid_shape,
    const DeviceGrid& device_grid,
    const std::unordered_map<std::string, PlacerOpOverride> &op_to_overrides,
    const bool enable_auto_transposing,
    std::optional<Coord> starting_coordinate)
{
    std::vector<PlacerState> grid_placer_strategies = {
        PlacerState::PLACE_WITH_RELATIVE_OFFSETS,
        PlacerState::INCREMENT_EPOCH_AND_PLACE_WITH_RELATIVE_OFFSETS,
        PlacerState::ALLOW_EPOCH_INCREMENTS,
    };

    for (PlacerState grid_placer_strategy : grid_placer_strategies)
    {
        try
        {
            log_trace(LogPlacer, "Placing with strategy: {}", to_string(grid_placer_strategy)); 
            DeviceGridConfig grid_placer_config = get_device_grid_config_from_strategy(grid_placer_strategy, op_group_to_place.increment_epoch);
            return place_on_grid_helper(
                op_group_to_place.op_names,
                op_to_grid_shape, 
                op_to_overrides,
                enable_auto_transposing,
                grid_placer_config,
                device_grid,
                starting_coordinate,
                false /* return_after_one_epoch */,
                op_group_to_place.op_name_to_relative_offset_from_first_op
            );
        }
        catch (const FailToPlaceOnCurrentEpoch& e)
        {
            // can't place on current epoch, switch to next grid strategy
        }
        catch (...)
        {
            TT_ASSERT("place_on_grid: caught unhandled exception");
        }
    }
    log_fatal("All place_on_grid(..) strategies have failed.");
    return {};
}

vector<EpochDeviceGridPlacement> place_onto_device_grids(
    const GridShape& device_grid_shape,
    const vector<OpGroupToPlace>& op_groups_to_place,
    const unordered_map<string, GridShape>& op_to_grid_shape,
    const std::unordered_map<std::string, PlacerOpOverride> &op_to_overrides,
    const bool enable_auto_transposing)
{
    vector<EpochDeviceGridPlacement> epoch_device_grid_placements;
    unordered_map<uint32_t, vector<DeviceGridPlacement>> epoch_id_to_device_grid_placement;
    auto epoch_id_to_device_grid = EpochIdToDeviceGrid(device_grid_shape.rows, device_grid_shape.columns);

    uint32_t current_epoch_id = 0;
    std::optional<Coord> current_coordinate = std::nullopt;

    for (const OpGroupToPlace& op_group : op_groups_to_place)
    {
        vector<DeviceGridPlacement> device_grid_placements;
        epoch_id_to_device_grid.initialize_device_grid(current_epoch_id);
        std::tie(device_grid_placements, current_coordinate) = place_on_grid(
            op_group,
            op_to_grid_shape,
            epoch_id_to_device_grid.get_device_grid(current_epoch_id),
            op_to_overrides,
            enable_auto_transposing,
            current_coordinate
        );

        for (const DeviceGridPlacement& device_grid_placement : device_grid_placements)
        {

            uint32_t device_index = current_epoch_id + device_grid_placement.device_grid_index;
            epoch_id_to_device_grid.fill_device_grid_with_placement(
                device_index, 
                device_grid_placement.placed_cores.start,
                op_to_grid_shape.at(device_grid_placement.op_name));

            epoch_id_to_device_grid_placement[device_index].push_back(device_grid_placement);
        }
        current_epoch_id = epoch_id_to_device_grid.get_current_epoch_id(); 
    }

    for (uint32_t epoch_id = 0; epoch_id < epoch_id_to_device_grid.epoch_id_to_device_grid.size(); ++epoch_id)
    {
        epoch_device_grid_placements.emplace_back(
            epoch_id_to_device_grid.epoch_id_to_device_grid.at(epoch_id), epoch_id_to_device_grid_placement.at(epoch_id));
        
    }

    return epoch_device_grid_placements;
}

void EpochDevicePlacer::enqueue_workload(const vector<OpGroupToPlace>& op_groups)
{
    for (const auto& op_group : op_groups)
    {
        this->remaining_op_groups.push_back(op_group);
    }
}

std::vector<DeviceGridPlacement> EpochDevicePlacer::place_on_grid(
    const OpGroupToPlace& op_group_to_place)
{
    auto [device_grid_placements, _] = place_on_grid_helper(
        op_group_to_place.op_names,
        this->config.op_to_grid_shape,
        this->config.op_to_overrides, 
        this->config.enable_auto_transposing_placement,
        get_device_grid_config_from_strategy(PlacerState::PLACE_WITH_RELATIVE_OFFSETS, false),
        this->active_device_grid,
        std::nullopt,
        true /* return_after_one_epoch */,
        op_group_to_place.op_name_to_relative_offset_from_first_op,
        this->op_to_device_grid_constraint
    );

    for (const auto& op_device_placement : device_grid_placements)
    {
        auto op_grid_shape  = this->config.op_to_grid_shape.at(op_device_placement.op_name);
        if (op_device_placement.grid_transpose){
            op_grid_shape = GridShape(op_grid_shape.columns, op_grid_shape.rows);
        }

        device_grid::fill_device_grid_with_placement(
            this->active_device_grid, 
            op_device_placement.placed_cores.start,
            op_grid_shape);
    }
    return device_grid_placements;
}
void EpochDevicePlacer::clear_state()
{
    GridShape device_grid_shape(config.get_available_rows_on_device(), config.device_grid.columns);
    this->active_device_grid = device_grid::create_empty_device_grid(device_grid_shape.rows, device_grid_shape.columns);
    while (not this->placed_op_groups.empty()) { this->placed_op_groups.pop_front(); }
    this->active_op_placements.clear();
    this->op_to_device_grid_constraint.clear();
}

EpochDeviceGridPlacement EpochDevicePlacer::complete_epoch()
{
    // either we have to place this on a new epoch OR 
    // apply constraint and then replace the previous ops
    auto complete_epoch = EpochDeviceGridPlacement(this->active_device_grid, this->active_op_placements);
    
    // Reset state
    this->clear_state();

    //device_grid::print_device_grid(complete_epoch.device_grid);
    return complete_epoch;
}

std::optional<EpochDeviceGridPlacement> EpochDevicePlacer::get_next_epoch()
{
    std::optional<EpochDeviceGridPlacement> active_epoch_placement = std::nullopt;
    GridShape device_grid_shape(config.get_available_rows_on_device(), config.device_grid.columns);
    this->active_device_grid = device_grid::create_empty_device_grid(device_grid_shape.rows, device_grid_shape.columns);

    while (not this->remaining_op_groups.empty() or not this->op_groups_to_place_again.empty())
    {
        // try to place current op-group, if not possible

        const auto& op_group_to_place = this->op_groups_to_place_again.empty() ?
            this->remaining_op_groups.front() : this->op_groups_to_place_again.front();

        bool contains_single_op_in_group = op_group_to_place.op_names.size() == 1;
        TT_ASSERT(op_group_to_place.op_names.size() >= 1);
        if (op_group_to_place.op_names.size() > 1)
        {
            // We don't support constraints applied on an OpGroup with multiple ops
            for (const auto& op : op_group_to_place.op_names)
            {
                TT_ASSERT(this->op_to_device_grid_constraint.find(op) == this->op_to_device_grid_constraint.end());
            }
        }
        const auto& op = op_group_to_place.op_names.at(0);

        if (op_group_to_place.increment_epoch and not this->active_op_placements.empty())
        {
            return this->complete_epoch();
        }

        try
        {
            log_debug(LogPlacer, "trying to place op_group: {}", op_group_to_place.op_names);
            auto op_device_placements = this->place_on_grid(op_group_to_place);
            if (contains_single_op_in_group and this->op_to_device_grid_constraint.find(op) != this->op_to_device_grid_constraint.end())
            {
                TT_ASSERT(this->op_groups_to_place_again.empty());
                log_trace(LogPlacer, "erasing constraint for: {}", op);
                this->op_to_device_grid_constraint.erase(op);
            }

            for (const auto& op_device_placement : op_device_placements)
            {
                this->active_op_placements.push_back(op_device_placement);
            }
            this->placed_op_groups.push_back(op_group_to_place);

            if (this->op_groups_to_place_again.empty())
            {
                this->remaining_op_groups.pop_front();
            }
            else
            {
                this->op_groups_to_place_again.pop_front();
            }

        }
        catch (const FailToPlaceOnCurrentEpoch& e)
        {
            return this->complete_epoch();
        }
        catch (const FailToSatisfyPlacementConstraint& e)
        {
            // Replay epoch placement with the constraint
            log_debug(LogPlacer, "failing to place {} because of existing constraints. adding constraint", op);
            auto constraint_grid = device_grid::create_empty_device_grid(device_grid_shape.rows, device_grid_shape.columns);
            TT_ASSERT(config.op_to_overrides.find(op) != config.op_to_overrides.end());
            const auto& op_override = config.op_to_overrides.at(op);
            auto op_grid_shape  = this->config.op_to_grid_shape.at(op);
            if (op_override.transpose_op){
                op_grid_shape = GridShape(op_grid_shape.columns, op_grid_shape.rows);
            }
            device_grid::fill_device_grid_with_placement(constraint_grid, op_override.grid_start.value(), op_grid_shape);

            for (const auto& active_placement : this->active_op_placements)
            {
                if (this->config.op_to_overrides.find(active_placement.op_name) != this->config.op_to_overrides.end())
                {
                    const auto& existing_op_override = config.op_to_overrides.at(active_placement.op_name);
                    auto existing_op_grid_shape  = this->config.op_to_grid_shape.at(active_placement.op_name);
                    if (existing_op_override.transpose_op){
                        existing_op_grid_shape = GridShape(existing_op_grid_shape.columns, existing_op_grid_shape.rows);
                    }

                    if (not device_grid::can_place_on_device_grid(constraint_grid, existing_op_override.grid_start.value(), existing_op_grid_shape))
                    {
                        log_debug(LogPlacer, "Placer: Completing epoch because there's an op in the current epoch that conflicts with constraint. {}", op);
                        return this->complete_epoch();
                    }
                }
            }

            this->op_to_device_grid_constraint[op] = constraint_grid;

            while (not this->placed_op_groups.empty()) { 
                const auto& old_op_group = this->placed_op_groups.back(); 
                this->op_groups_to_place_again.push_front(old_op_group);
                this->placed_op_groups.pop_back(); 
            }
            this->active_op_placements.clear();
            this->active_device_grid = device_grid::create_empty_device_grid(device_grid_shape.rows, device_grid_shape.columns);
        }
        catch (const FailToSatisfyConflictingConstraint& e)
        {
            log_fatal("Caught FailToSatisfyConflictingConstraint: {}", e.what());
        }
    }
    if (not this->active_op_placements.empty())
    {
        return this->complete_epoch();
    }
    return active_epoch_placement;
}

std::vector<EpochDeviceGridPlacement> EpochDevicePlacer::place(const vector<OpGroupToPlace>& op_groups)
{
    log_debug(LogPlacer, "ops tagged for chip_break: {}", this->config.ops_tagged_for_chip_id_break);
    log_debug(LogPlacer, "ops tagged for epoch_break: {}", this->config.ops_tagged_for_epoch_break);

    log_debug(LogPlacer, "ops tagged for override:");
    for (const auto& [op, override] : this->config.op_to_overrides)
    {
        log_debug(LogPlacer,
            "ops tagged for override: {}, override={}", op, override
        );
    }

    std::vector<EpochDeviceGridPlacement> epochs;
    this->enqueue_workload(op_groups);

    for (auto epoch_placement = get_next_epoch(); epoch_placement.has_value();)
    {
        epochs.push_back(epoch_placement.value());
        epoch_placement = get_next_epoch();
    }
    return epochs;
}

} // namespace placer
} // namespace tt
