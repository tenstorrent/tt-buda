// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "placer/placer.hpp"
#include <deque>
#include <queue>

namespace tt::placer {

// In the future, refactor this into a class..
// if DeviceGrid[i][j] == 0, then we consider the core at location (i,j) to be free/available
using DeviceGrid = vector<vector<uint32_t>>;

// Functions on Device Grid
namespace device_grid
{
DeviceGrid create_empty_device_grid(uint32_t rows, uint32_t columns);
DeviceGrid superposition(const DeviceGrid& a, const DeviceGrid& b);
bool can_place_on_device_grid(const DeviceGrid& device_grid, const Coord& start, const GridShape& shape);
bool contains_empty_device_grid(const DeviceGrid& device_grid);
void fill_device_grid_with_placement(DeviceGrid& device_grid, const Coord& op_start, const GridShape& op_grid_shape);
void print_device_grid(const DeviceGrid& device_grid);
std::optional<Coord> get_next_grid_coordinate(const DeviceGrid& device_grid, const GridShape& op_grid_shape);
} // namespace device_grid

struct DeviceGridPlacement {
    std::string op_name;
    uint32_t device_grid_index;
    CoordRange placed_cores;
    bool grid_transpose;
};

struct EpochDeviceGridPlacement
{
    DeviceGrid device_grid;
    vector<DeviceGridPlacement> op_placements;

    EpochDeviceGridPlacement(DeviceGrid&& device_grid) : device_grid(std::move(device_grid)) {}
    EpochDeviceGridPlacement(const DeviceGrid& device_grid) : device_grid(device_grid) {}
    EpochDeviceGridPlacement(const DeviceGrid& device_grid, const vector<DeviceGridPlacement>& op_placements)
        : device_grid(device_grid), op_placements(op_placements) {}
};

// Notes:
// op_to_grid_shape := op to grid_shapes that we need to place
// device_grid := device_grid containing current view of placed ops
// [[optional]] starting_coordinate := coordinate from where to start placing
//
// Returns the grid coordinate location of the last placed op.
std::tuple<vector<DeviceGridPlacement>, Coord> place_on_grid(
    const OpGroupToPlace& op_group_to_place,
    const unordered_map<std::string, GridShape>& op_to_grid_shape,
    const DeviceGrid& device_grid,
    const std::unordered_map<std::string, PlacerOpOverride> &op_to_overrides,
    const bool enable_auto_transposing_placement,
    std::optional<Coord> starting_coordinate = std::nullopt);

// Grid-placer API that attempts to place an op in the current epoch. It never
// moves to a new epoch on its own.
std::optional<DeviceGridPlacement> place_one_op(
    const string op,
    const unordered_map<string, GridShape>& op_to_grid_shape,
    const DeviceGrid& device_grid,
    const std::unordered_map<std::string, PlacerOpOverride> &op_to_overrides,
    const bool enable_auto_transposing_placement,
    std::optional<Coord> starting_coordinate = std::nullopt);


// Constraint-based iterative grid-placer:
// Given {op_groups, constraints on grid-location of ops}, just return 
// a fully-populated device-grid epoch.
class EpochDevicePlacer
{
    const PlacerConfig& config;
    std::deque<OpGroupToPlace> op_groups_to_place_again;
    std::deque<OpGroupToPlace> remaining_op_groups;

    std::deque<OpGroupToPlace> placed_op_groups;
    std::vector<DeviceGridPlacement> active_op_placements;
    DeviceGrid active_device_grid;

    std::unordered_map<std::string, DeviceGrid> op_to_device_grid_constraint;

    void clear_state();
    EpochDeviceGridPlacement complete_epoch();
    std::vector<DeviceGridPlacement> place_on_grid(const OpGroupToPlace& op_group_to_place);
    void enqueue_workload(const vector<OpGroupToPlace>& op_groups);

  public:
    EpochDevicePlacer(const PlacerConfig& config) : config(config) {}
    std::optional<EpochDeviceGridPlacement> get_next_epoch();
    std::vector<EpochDeviceGridPlacement> place(const vector<OpGroupToPlace>& op_groups);
};

} // end namespace tt::placer
