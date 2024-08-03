// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend_api/device_config.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/query.hpp"
#include "third_party/json/json_fwd.hpp"

using NodeEpochType = tt::graphlib::NodeEpochType;

using std::map;
using std::set;
using std::string;
using std::uint32_t;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using json = nlohmann::json;

namespace tt {
namespace placer {


/*
  ____        _          ____  _                   _
 |  _ \  __ _| |_ __ _  / ___|| |_ _ __ _   _  ___| |_ _   _ _ __ ___  ___
 | | | |/ _` | __/ _` | \___ \| __| '__| | | |/ __| __| | | | '__/ _ \/ __|
 | |_| | (_| | || (_| |  ___) | |_| |  | |_| | (__| |_| |_| | | |  __/\__ \
 |____/ \__,_|\__\__,_| |____/ \__|_|   \__,_|\___|\__|\__,_|_|  \___||___/

*/

struct CoordOffset
{
    uint32_t row_offset;
    uint32_t column_offset;
};

struct GridShape
{
    uint32_t rows = 0;
    uint32_t columns = 0;

    GridShape() = default;
    GridShape(uint32_t rows, uint32_t columns) : rows(rows), columns(columns) {};
    std::uint32_t volume() const { return rows * columns; }
    GridShape transposed() const { return GridShape(columns, rows); }

    static GridShape from_array(std::array<uint32_t, 2> array);
};

struct Coord
{
    uint32_t row = 0;
    uint32_t col = 0;
    Coord operator+(const GridShape &rhs) const;
    Coord operator+(const CoordOffset &rhs) const;
    Coord operator+(const Coord &rhs) const;
    bool operator< (const Coord &rhs) const;
    bool operator== (const Coord &rhs) const;
    bool operator!= (const Coord &rhs) const;

    json to_json() const;
    std::array<uint32_t, 2> as_array() const;
};

struct CoordRange
{
    // Contiguous range of core coordinates from top-left(start) to bottom-right(end)
    Coord start; // inclusive
    Coord end; // exclusive

    uint32_t size_r() const { return end.row - start.row; }
    uint32_t size_c() const { return end.col - start.col; }
    bool operator==(const CoordRange &rhs) const;
    bool operator!=(const CoordRange &rhs) const;
    json to_json() const;

    void transpose()
    {
        uint32_t r = size_r();
        end.row = start.row + size_c();
        end.col = start.col + r;
    }
};

struct PlacerOpOverride
{
    std::optional<Coord> grid_start = std::nullopt;
    bool transpose_op = false;
    std::optional<uint32_t> chip_id = std::nullopt;
    bool temporal_epoch_break = false;

    PlacerOpOverride() = default;
    PlacerOpOverride(
        std::optional<std::array<uint32_t, 2>> start,
        bool transpose_op,
        std::optional<uint32_t> chip_id,
        bool temporal_epoch_break = false) :
        transpose_op(transpose_op), chip_id(chip_id), temporal_epoch_break(temporal_epoch_break)
    {
        if (start.has_value())
        {
            std::array<uint32_t, 2> start_array = start.value();
            this->grid_start = Coord{.row = start_array[0], .col = start_array[1]};
        }
    }
    PlacerOpOverride(
        std::optional<Coord> start,
        bool transpose_op,
        std::optional<uint32_t> chip_id,
        bool temporal_epoch_break = false) :
        grid_start(start), transpose_op(transpose_op), chip_id(chip_id), temporal_epoch_break(temporal_epoch_break)
    {
    }

    static PlacerOpOverride force_op_transpose()
    {
        std::optional<Coord> start = std::nullopt;
        return PlacerOpOverride(start, true, std::nullopt, false);
    }

    static PlacerOpOverride override_chip_id(int chip_id)
    {
        std::optional<Coord> start = std::nullopt;
        return PlacerOpOverride(start, false, chip_id, false);
    }

    bool operator==(const PlacerOpOverride& rhs) const
    {
        return (grid_start == rhs.grid_start) && (transpose_op == rhs.transpose_op) && (chip_id == rhs.chip_id) &&
               (temporal_epoch_break == rhs.temporal_epoch_break);
    }
};

enum PlacementStrategy
{
    // PlacementStrategy controls how we place a sequence of ops.

    // Extend and implement for different placement strategies
    LeftToRight = 0, // Place left-to-right on each new row
};
enum PlacementScheduleOrder
{
    // PlacementSchedule controls the sequence order of the ops we place.
    // By default, we place based on topological ordering of the nodes
    Topological,
};

enum class ChipPlacementPolicy;

struct PlacerConfig
{
    // Arch config
    std::vector<std::uint32_t> chip_ids;
    tt::placer::ChipPlacementPolicy chip_placement_policy;
    const DeviceConfig& device_config;
    GridShape device_grid;
    bool contains_recompute = false;
    bool output_queues_on_host = true;

    // a list of row_indices (range defined by logical coordinates), defining the harvested rows
    // in other words, placer should skip placing ops on these rows.
    vector<uint32_t> harvested_rows = {};

    // Placer config toggling strategies/behaviors of different automatic placements
    PlacementStrategy strategy = PlacementStrategy::LeftToRight;

    unordered_map<string, GridShape> op_to_grid_shape;
    unordered_map<string, GridShape> input_queue_to_grid_shape;

    // Capture any user or op-specific config for placement
    // like chip-breaks or epoch-breaks
    unordered_map<string, NodeEpochType> op_to_epoch_type;
    unordered_map<string, bool> op_to_grad_op; // set for gradient accumulation ops
    unordered_map<string, bool> op_to_recompute_op;

    // captures any user-configuration for chip-breaking
    unordered_set<string> ops_tagged_for_chip_id_break;
    unordered_set<string> ops_tagged_for_epoch_break;
    unordered_set<string> ops_tagged_for_temporal_epoch_break; // WH and legacy-placer specific

    unordered_map<string, vector<string>> fwd_to_bwd_nodes;
    unordered_map<string, map<int, vector<string>>> fwd_to_opt_nodes;
    unordered_set<string> output_ops = {};
    unordered_map<string, uint32_t> op_to_chip_id_assignment;
    unordered_map<string, PlacerOpOverride> op_to_overrides;

    bool enable_auto_transposing_placement = false;

    // methods
    uint32_t get_available_rows_on_device() const;
    uint32_t get_chip_id(const string& op_name) const;
    std::optional<uint32_t> get_chip_id_override(const string& op_name) const;
};

struct PlacerConfigUpdate
{
    unordered_map<string, uint32_t> op_to_chip_id_assignment;
    vector<vector<string>> op_names_to_chip_break;
    vector<vector<string>> op_names_to_epoch_break;

    PlacerConfigUpdate(
        const unordered_map<string, uint32_t>& op_to_chip_id_assignment,
        const vector<vector<string>>& op_names_to_chip_break,
        const vector<vector<string>>& op_names_to_epoch_break) :
        op_to_chip_id_assignment(op_to_chip_id_assignment),
        op_names_to_chip_break(op_names_to_chip_break),
        op_names_to_epoch_break(op_names_to_epoch_break)
    {
    }
};


// The struct capturing the decision made by the placer for how to place an op.
// This struct defines the atomic unit of work for the Placer.
//
// This captures one or more ops to be placed TOGETHER in the same epoch/chip.
// This simplifies things so placer only needs to worry about placing one OpGroupToPlace at a time,
// instead of doing look-aheads to make sure we're still conforming to constraints
//
// Consider the following cases:
//   1. tilize/untilize unaries needing to be placed with its producer op
//   2. any user-defined groupings for the op (user: "I want to place ops {A, B, C} in the same epoch")
//   3. triplet placement
struct OpGroupToPlace
{
    static uint32_t current_op_group_id; // assigned based on placement order

    uint32_t op_group_id; // assigned based on placement order
    vector<string> op_names;
    unordered_map<string, CoordOffset> op_name_to_relative_offset_from_first_op;
    uint32_t chip_id = 0;
    bool increment_epoch = false;

    NodeEpochType epoch_type = NodeEpochType::Forward;
    static uint32_t get_next_op_group_id();
};

struct EpochInfo
{
    uint32_t global_epoch_id; // globally unique across time/space/chip
    uint32_t temporal_epoch_id; // epoch timestep where multiple spatially arranged chips may be executing concurrently
    uint32_t spatial_epoch_id; // within a temporal_epoch_id, the linearized id defining the spatial index. for grayskull, this is always zero.

    NodeEpochType epoch_type;
};

inline bool operator<(const EpochInfo& lhs, const EpochInfo& rhs)
{
    //return lhs.global_epoch_id < rhs.global_epoch_id;
    if (lhs.temporal_epoch_id == rhs.temporal_epoch_id) {
        return lhs.spatial_epoch_id < rhs.spatial_epoch_id;
    }
    return lhs.temporal_epoch_id < rhs.temporal_epoch_id;
}

// The struct capturing the decision made by the placer for how to place an op.
struct OpPlacement
{
    uint32_t id = 0;
    string name;
    uint32_t chip_id;
    uint32_t global_epoch_id; // globally unique across time/space/chip
    bool grid_transpose;

    // Future: For initial implementation, no fracturing support. `placed_cores` will only
    // have a single element in the vector.
    CoordRange placed_cores;

    // methods
    uint32_t epoch_id() const { return global_epoch_id; }
    bool operator==(const OpPlacement& rhs) const;
    bool operator!=(const OpPlacement& rhs) const;
    json to_json() const;
};

// Placement information for a single buffer in DRAM queue, placed on one dram channel
struct QueueBufferPlacement
{
    uint32_t dram_channel;
    size_t dram_address;

    // Not strictly needed to set placement, but convenient to have here
    Coord dram_channel_location;
    size_t buffer_size;
    bool allocated_in_p2p_region;

    // methods
    json to_json() const;
};

struct QueueHostBufferPlacement
{
    uint32_t channel;
    size_t address;
    size_t buffer_size;

    // methods
    json to_json() const;
};

// Placement information for a DRAM queue, split over some number of channels
struct QueuePlacement
{
    string name;
    string input_name;
    GridShape grid_shape;
    bool on_host;
    uint32_t chip_id;
    std::vector<QueueBufferPlacement> dram_buffers;
    std::vector<QueueHostBufferPlacement> host_buffers;
    bool read_only = false;
    bool write_only = false;
    int write_stride = -1;

    // If dynamic, this indicates when queue will be allocated/deallocated
    int epoch_allocate = -1;
    int epoch_deallocate = -1;

    // methods
    json to_json() const;
    uint32_t queue_size_bytes();
    // returns true if queue is static
    bool is_static() const { return epoch_allocate == -1 && epoch_deallocate == -1; }
};

// The final returned struct out of the Placer module will have fully populated attributes
using DeviceGrid = vector<vector<uint32_t>>;

struct EpochIdToDeviceGrid
{
    uint32_t rows = 0;
    uint32_t columns = 0;
    unordered_map<int, DeviceGrid> epoch_id_to_device_grid;
    unordered_map<std::string, DeviceGrid> op_to_constraints;

    EpochIdToDeviceGrid() : rows(0), columns(0) {}
    EpochIdToDeviceGrid(uint32_t rows, uint32_t columns) : rows(rows), columns(columns) {}
    EpochIdToDeviceGrid(const std::pair<int, int>& grid_pair) : rows(grid_pair.first), columns(grid_pair.second) {}

    void initialize_device_grid(uint32_t epoch_id, bool clear_existing = false);
    void initialize_device_grid(uint32_t candidate_epoch_id, uint32_t rows, uint32_t columns);
    void initialize_device_grid(uint32_t epoch_id, const DeviceGrid& device_grid);
    bool contains_empty_grid(uint32_t epoch_id) ;
    bool satisfies_constraints(const std::string& op_name, const Coord& start, const GridShape& shape) const;
    bool can_place_on_device_grid(const std::string& op_name, int epoch_id, const Coord& start, const GridShape& shape);
    void fill_device_grid_with_placement(int epoch_id, const Coord& op_start, const GridShape& op_grid_shape);
    uint32_t get_current_epoch_id() const;
    const DeviceGrid& get_device_grid(uint32_t epoch_id) const;
    std::optional<Coord> get_next_grid_coordinate(const std::string& op_name, uint32_t epoch_id, const GridShape& op_grid_shape) const;

    void add_constraints(const std::unordered_map<std::string, DeviceGrid>& constraints);
};
struct PlacerSolution
{
    using EpochId = int;
    unordered_map<string, OpPlacement> name_to_op_placement;
    unordered_map<string, GridShape> input_queue_to_grid_shape;
    unordered_map<string, QueuePlacement> name_to_queue_placement;
    map<EpochId, int> epoch_id_to_chip;
    map<EpochId, unsigned int> epoch_id_to_subgraph_index;
    unordered_map<int, vector<OpPlacement>> epoch_id_to_op_placement;
    EpochIdToDeviceGrid epoch_id_to_device_grid;
    unordered_map<int, EpochInfo> epoch_id_to_epoch_info;
    uint32_t num_epochs = 0;
    bool is_pipelined = true;
    bool fork_join_buffered = false;

    // methods
    json to_json() const;

    uint32_t chip_id(const std::string& op_name) const;

    // Globally unique across chips
    uint32_t epoch_id(const std::string& op_name) const;

    const EpochInfo& epoch_info(uint32_t global_epoch_id) const;

    // These methods are really only relevant for wormhole.
    // For grayskull, temporal_epoch_id == epoch_id
    uint32_t temporal_epoch_id(const std::string& op_name) const;
    uint32_t temporal_epoch_id(uint32_t global_epoch_id) const;
    uint32_t num_temporal_epochs() const;
    uint32_t num_temporal_epochs(NodeEpochType type) const;
    NodeEpochType epoch_type(uint32_t global_epoch_id) const;

    void merge(PlacerSolution &other);
    bool is_placed(const std::string& op_name) const;

};

/*
  ____  _                          _    ____ ___
 |  _ \| | __ _  ___ ___ _ __     / \  |  _ \_ _|___
 | |_) | |/ _` |/ __/ _ \ '__|   / _ \ | |_) | |/ __|
 |  __/| | (_| | (_|  __/ |     / ___ \|  __/| |\__ \
 |_|   |_|\__,_|\___\___|_|    /_/   \_\_|  |___|___/
*/

// Placer Manipulation APIs: Convenience methods to update PlacerConfig with user-constraints
void place_on_new_epoch(PlacerConfig& config, const string& op_name);
void place_on_new_chip(PlacerConfig& config, const string& op_name);
void dump_placer_solution_json_to_file(const PlacerSolution& solution);


// *Main Entrypoints* from placer lowering
//
// Intentionally not introducing gstate or tt_graph into these APIs for the placer-module
// to decouple from all that state.
// Placer just receives a schedule and config and is responsible for generating op placements
//
// Given a list of scheduled ops, and the PlacerConfig::PlacementStrategy, iterate through
// the list placing each op one at a time.

using ChipId = uint32_t;
using PlacerWorkload = vector<OpGroupToPlace>;
using ChipIdToPlacerWorkload = map<ChipId, PlacerWorkload>;

PlacerSolution place_onto_chip(
    const PlacerConfig& config,
    PlacerWorkload& placer_op_group_workload,
    uint32_t epoch_start_id = 0,
    std::optional<NodeEpochType> epoch_type = std::nullopt);

PlacerSolution placer(const PlacerConfig& config, const vector<string>& scheduled_ops);

std::ostream& operator<<(std::ostream& os, const Coord& coord);
std::ostream& operator<<(std::ostream& os, const CoordRange& coord_range);
std::ostream& operator<<(std::ostream& os, const PlacerOpOverride& override);

// Expand predicates into a map of all matched node names
std::unordered_map<std::string, placer::PlacerOpOverride> match_op_names_to_placer_overrides(
    graphlib::Graph* graph,
    std::vector<std::pair<std::variant<std::string, graphlib::query::NodePredicate>, placer::PlacerOpOverride>> const&
        predicates_to_overrides);

using PredicatesToBreaks = std::vector<std::variant<
    std::vector<std::variant<std::string, graphlib::query::NodePredicate>>,
    graphlib::query::NodePredicate>>;

// Expand predicates into a list of all matched node names
std::vector<std::vector<std::string>> match_op_names_to_breaks(
    graphlib::Graph* graph, const PredicatesToBreaks& predicates_to_breaks);
}  // end namespace placer
} // end namespace tt
