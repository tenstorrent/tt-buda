// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backend_api/device_config.hpp"
#include "balancer/types.hpp"
#include "graph_lib/node_types.hpp"
#include "passes_utils.hpp"
#include "passes/amp.hpp"
namespace tt
{

struct DeviceConfig;

// Attributes that can be specified on sub op level.
// TODO: Instead of hardcoding these values here get them from backend api. 
inline static unordered_map<std::string, std::unordered_set<std::string>> SubOpAttr = {
    {"*", std::unordered_set<std::string>{"m_k", "u_kt", "vector", "relu_en", "relu_threshold", "relu_mode"}},
    {"reduce", std::unordered_set<std::string>{"dim", "type"}},
    {"dropout", std::unordered_set<std::string>{"p", "seed"}},
    {"lrelu", std::unordered_set<std::string>{"slope"}},
    {"power", std::unordered_set<std::string>{"exp"}},
};

// Main entry
void fuse_ops(
    graphlib::Graph *graph,
    const DeviceConfig &device_config,
    const std::vector<std::vector<std::string>> &op_names_to_chip_break,
    const std::vector<std::vector<std::string>> &op_names_to_epoch_break,
    const std::vector<std::string> &op_names_dont_fuse,
    const std::vector<std::string> &op_names_manual_fuse,
    std::vector<tt::passes::AMPNodeProperties> &amp_properties);

// Op input type / id
struct FusedSubOpInput
{
    enum InputType
    {
        INPUT,
        INTERMED,
        DEST
    } type;

    std::uint32_t index;  // either input index, or buffer index
    std::pair<std::uint32_t, std::uint32_t> broadcast = {0, 0};
    std::pair<bool, bool> tile_broadcast = {false, false};

    bool has_broadcast() const { return broadcast.second != 0; }
    bool has_tile_broadcast() const { return tile_broadcast.first || tile_broadcast.second; }
};

// Op within fused op
struct FusedSubOp
{
    std::string name;
    graphlib::OpType op_type;
    balancer::OpShape op_shape;

    // Inputs are either other ops or intermediate buffers
    std::vector<FusedSubOpInput> inputs;

    enum OutputType
    {
        OUTPUT,
        DEST,
        INTERMED
    } output_type;
    std::uint32_t output_buffer;  // only valid for INTERMED type
    DataFormat output_df;

    std::unordered_map<std::string, std::uint32_t> attrs;
    std::vector<std::uint32_t> popped_buffers;
    std::vector<std::uint32_t> popped_last_buffers;

    std::pair<std::uint32_t, std::uint32_t> get_mblock_for_ublock(
        const std::pair<std::uint32_t, std::uint32_t> ublock, const std::pair<std::uint32_t, std::uint32_t> grid) const;

    BudaOpAttrs get_sub_op_buda_attr() const;
};

// Single sequential scheduled of fused ops to run through
struct FusedSchedule
{
    std::vector<FusedSubOp> ops;
};

class FusionGroup;
using FusionGroupP = std::shared_ptr<FusionGroup>;
class FusedOp;
using FusedOpP = std::shared_ptr<FusedOp>;
using InputMapping = std::unordered_map<graphlib::Node *, std::unordered_map<std::uint32_t, std::uint32_t>>;
using BudaOpNode = graphlib::BudaOpNode;

// Represet information about the new op in the graph, made from the ops in fusion group
class FusedOp
{
   public:
    static constexpr int kMaxNumDRAMInputs = 8;
    static constexpr int kMaxNumInputs = 16;
    static constexpr int kMaxNumConnections = kMaxNumInputs + 1;  // +1 for at least 1 output connection

   private:
    FusionGroupP group;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-private-field"
    BudaOpNode *node;  // fused node

    // Ordered list of inputs - pair of nodes which inputs are connected to, and their operand index
    InputMapping inputs;

    // Output op within fused nodes
    BudaOpNode *output_op;
#pragma GCC diagnostic pop

    // Schedules to execute the op
    std::vector<FusedSchedule> schedules;

    bool has_matmul_ = false;
    bool has_reduce_;
    bool has_broadcast_c_ = false;
    // Reduce dim - only one for all reduces allowed
    std::uint32_t reduce_dim_;

   public:
    // Construct a new fused op, fusing itself into the graph
    FusedOp(
        FusionGroupP group,
        BudaOpNode *node,
        InputMapping inputs,
        BudaOpNode *output_op,
        std::vector<FusedSchedule> schedules);

    std::shared_ptr<FusedOp> clone(BudaOpNode *node);

    const std::vector<FusedSchedule> &get_schedules() const { return schedules; }
    std::vector<FusedSchedule> &get_schedules() { return schedules; }
    std::uint32_t id() const;
    std::uint32_t get_input_count() const;
    BudaOpAttrs get_operation_attr();

    bool has_matmul_op() const { return has_matmul_; }
    bool has_reduce_op() const { return has_reduce_; }
    bool has_broadcast_c() const {return has_broadcast_c_; }

    std::uint32_t get_reduce_dim() const
    {
        TT_ASSERT(has_reduce_op());
        return reduce_dim_;
    }
};

}  // namespace tt
