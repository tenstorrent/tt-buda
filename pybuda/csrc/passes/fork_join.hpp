// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <set>
#include <string>

#include "balancer/types.hpp"
#include "graph_lib/defines.hpp"
#include "utils/hash_combine.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"

namespace tt
{

namespace graphlib
{
class Graph;
class Node;
}  // namespace graphlib

// Instruct pre-placer to insert a NOP between src/dest nodes
// Further information on iteration attempt, etc. can be added in the future to augment this
enum class InstructionType : std::uint8_t
{
    NopInstruction,
    QueueInstruction
};

using InsInstructionUniqueId = std::tuple<std::string, std::string, std::uint32_t, std::uint32_t, bool, bool>;

struct InsInstructionUniqueIdHash : public std::unary_function<InsInstructionUniqueId, std::size_t>
{
    std::size_t operator()(const InsInstructionUniqueId &instr) const
    {
        std::size_t seed = 0;
        tt::hash_combine(seed, static_cast<std::size_t>(std::hash<std::string>{}(std::get<0>(instr))));
        tt::hash_combine(seed, static_cast<std::size_t>(std::hash<std::string>{}(std::get<1>(instr))));
        tt::hash_combine(seed, static_cast<std::size_t>(std::get<2>(instr)));
        tt::hash_combine(seed, static_cast<std::size_t>(std::get<3>(instr)));
        tt::hash_combine(seed, static_cast<std::size_t>(std::get<4>(instr)));
        tt::hash_combine(seed, static_cast<std::size_t>(std::get<5>(instr)));
        return seed;
    }
};

using ForkJoinId = std::pair<graphlib::NodeId, graphlib::NodeId>;
using ForkJoin = std::pair<std::vector<graphlib::Node *>, std::vector<graphlib::Node *>>;

// information on buffered fork-join
struct FJBufferingInfo
{
    graphlib::Node *join; /* join node ptr */
    std::uint32_t req;    /* required buffering */
    std::uint32_t avail;  /* available buffering */
    const ForkJoin *fj;   /* pointer to buffered fork-join */

    FJBufferingInfo(graphlib::Node *join, std::uint32_t req, std::uint32_t avail, const ForkJoin *fj) :
        join(join), req(req), avail(avail), fj(fj)
    {
    }
};

struct ForkJoinIdHash : public std::unary_function<ForkJoinId, std::size_t>
{
    std::size_t operator()(const ForkJoinId &fj_id) const
    {
        std::size_t seed = 0;
        tt::hash_combine(seed, static_cast<std::size_t>(std::hash<graphlib::NodeId>{}(fj_id.first)));
        tt::hash_combine(seed, static_cast<std::size_t>(std::hash<graphlib::NodeId>{}(fj_id.second)));
        return seed;
    }
};

struct InsertionInstruction
{
    /*
    This is base class for insertion instructions. From this we derive NopInsertionInstruction and
    QueueInsertionInstruction classes.
    */
   public:
    InstructionType instr_type;
    std::string src, dest;
    bool hoist_tms;                         // whether to hoist tms to the input to the new nop
    std::optional<std::uint32_t> input_id;  // input id into dest; if nullopt, use input_id from original edge
    std::optional<std::uint32_t> fork_id;   // index of output from src; if nullopt, use fork_id from original edge
    bool user_defined;                      // whether these requested NOPs were user-defined
    bool is_fj_buffering;                   // whether this insertion instruction is generated for FJ buffering

    InsertionInstruction(
        InstructionType instr_type,
        const std::string &src,
        const std::string &dest,
        bool hoist_tms,
        std::optional<std::uint32_t> input_id = std::nullopt,
        std::optional<std::uint32_t> fork_id = std::nullopt,
        bool user_defined = false,
        bool is_fj_buffering = false) :
        instr_type(instr_type),
        src(src),
        dest(dest),
        hoist_tms(hoist_tms),
        input_id(input_id),
        fork_id(fork_id),
        user_defined(user_defined),
        is_fj_buffering(is_fj_buffering)
    {
    }

    virtual ~InsertionInstruction() = default;

    InsInstructionUniqueId unique_id() const
    {
        return create_unique_id(
            this->src,
            this->dest,
            this->input_id.value_or(-1),
            this->fork_id.value_or(-1),
            this->user_defined,
            this->is_fj_buffering);
    }

    std::pair<graphlib::Node *, graphlib::Node *> is_instruction_still_valid(graphlib::Graph *graph);

    virtual void insert(graphlib::Graph *graph) = 0;

    virtual std::string to_string() const
    {
        return "src: " + src + ", dest: " + dest + ", hoist_tms: " + std::to_string(hoist_tms) +
               ", input_id: " + (input_id.has_value() ? std::to_string(input_id.value()) : "nullopt") +
               ", fork_id: " + (fork_id.has_value() ? std::to_string(fork_id.value()) : "nullopt") +
               ", user_defined: " + std::to_string(user_defined);
    }

    static InsInstructionUniqueId create_unique_id(
        const std::string &src,
        const std::string &dest,
        std::optional<std::uint32_t> input_id,
        std::optional<std::uint32_t> fork_id,
        bool user_defined,
        bool is_fj_buffering)
    {
        return std::make_tuple(src, dest, input_id.value_or(-1), fork_id.value_or(-1), user_defined, is_fj_buffering);
    }
};

std::ostream &operator<<(std::ostream &out, const InsertionInstruction *ins);

using InsertionInstructionMap =
    tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>;

struct PyInsertionInstruction : public InsertionInstruction
{
   public:
    /* Inherit the constructors */
    using InsertionInstruction::InsertionInstruction;

    void insert(graphlib::Graph *graph) override;
};

struct NopInsertionInstruction : public InsertionInstruction
{
   public:
    std::uint32_t nop_count;  // number of nops to insert
    bool mergeable;           // whether to merge user-defined NOPs with the same src
    bool daisy_chain;         // change the behaviour for merging nops with src->multiple consumers
    bool request_merge;       // enable to invoke the API call to perform the daisy-chain/merge

    NopInsertionInstruction(
        const std::string &src,
        const std::string &dest,
        bool hoist_tms,
        std::uint32_t nop_count,
        std::optional<std::uint32_t> input_id = std::nullopt,
        std::optional<std::uint32_t> fork_id = std::nullopt,
        bool user_defined = false,
        bool mergeable = false,
        bool daisy_chain = false,
        bool request_merge = false,
        bool is_fj_buffering = false) :
        InsertionInstruction(
            InstructionType::NopInstruction, src, dest, hoist_tms, input_id, fork_id, user_defined, is_fj_buffering),
        nop_count(nop_count),
        mergeable(mergeable),
        daisy_chain(daisy_chain),
        request_merge(request_merge)
    {
    }

    void insert(graphlib::Graph *graph) override;
    void set_nop_count(int nop_count) { this->nop_count = nop_count; };

    std::string to_string() const override
    {
        return InsertionInstruction::to_string() + ", nop_count: " + std::to_string(nop_count) +
               ", mergeable: " + std::to_string(mergeable) + ", daisy_chain: " + std::to_string(daisy_chain) +
               ", request_merge: " + std::to_string(request_merge) +
               ", is_fj_buffering: " + std::to_string(is_fj_buffering);
    }
};

struct QueueInsertionInstruction : public InsertionInstruction
{
   public:
    int num_entries;
    int queue_size;  // in bytes

    QueueInsertionInstruction(
        const std::string &src,
        const std::string &dest,
        bool hoist_tms,
        int num_entries,
        std::uint32_t queue_size,
        std::optional<std::uint32_t> input_id = std::nullopt,
        std::optional<std::uint32_t> fork_id = std::nullopt,
        bool user_defined = false,
        bool is_fj_buffering = false) :
        InsertionInstruction(
            InstructionType::QueueInstruction, src, dest, hoist_tms, input_id, fork_id, user_defined, is_fj_buffering),
        num_entries(num_entries),
        queue_size(queue_size)
    {
    }

    void insert(graphlib::Graph *graph) override;
    void set_num_entries(int num_entries) { this->num_entries = num_entries; };

    std::string to_string() const override
    {
        return InsertionInstruction::to_string() + ", num_entries: " + std::to_string(num_entries) +
               ", queue_size: " + std::to_string(queue_size);
    }
};

struct FJBufferingResult
{
    // Instructions generated for fork-join buffering.
    InsertionInstructionMap instructions;
    // All fork-joins which were buffered with either nop or queue instructions.
    std::vector<ForkJoin> fjs_buffered_with_instr;
};

// Insert buffers to match short/long forks
FJBufferingResult insert_fork_join_buffering(
    graphlib::Graph *graph,
    balancer::OpModelMap *op_models_post_placer,
    balancer::OpModels *op_models,
    const std::uint32_t usable_l1_size,
    const InsertionInstructionMap &previous_ins_instructions,
    std::function<int(const tt::balancer::OpModel &)> buffering_factor = [](const tt::balancer::OpModel &)
    { return 1; });

void upsize_dram_input(graphlib::Graph *graph, balancer::OpModelMap &op_models, const std::uint32_t usable_l1_size);

// Checking if two maps of instructions are equal
std::tuple<bool, int, int> is_subset_of_instructions(
    const InsertionInstructionMap &instructions, const InsertionInstructionMap &previous_instructions);

class FJGraph
{
    std::vector<std::uint32_t>
        fj_ids;  // fj_id is actually index of the in vector of fork-joins that is given to the constructor
    std::vector<ForkJoin> fork_joins;
    std::vector<const ForkJoin *> topo_sort_fjs;
    std::vector<std::size_t> topo_sort_fj_indices;
    std::vector<std::set<std::uint32_t>> adjacency_vector;  // indices in this vector pertain to indices in vector of
                                                            // fork-joins that ptr_to_fjs points to.

    // buffered_fjs map contains information about fork-joins that are already buffered. Key to map is fork node id, and
    // value is tuple of: join node id, required buffering, available buffering, and pointer to buffered fork-join
    // itself.
    std::unordered_map<graphlib::NodeId, std::vector<FJBufferingInfo>> buffered_fjs;
    std::unordered_map<const ForkJoin *, const ForkJoin *> parent_fj_map;
    std::vector<const ForkJoin *> fjs_buffered_with_instr;

   public:
    FJGraph(graphlib::Graph *graph);

    void add_edge(std::uint32_t src, std::uint32_t dest);

    void topological_sort();

    void create_parents_map();

    FJBufferingInfo find_sub_fork_join_from_node(
        const ForkJoin &fj, const std::vector<graphlib::Node *> &path, graphlib::Node *fork);

    void update_buffered_fj_map(const ForkJoin &fj, FJBufferingInfo fj_buff_info);

    // getters
    std::unordered_map<const ForkJoin *, const ForkJoin *> &get_parent_fj_map() { return parent_fj_map; }

    std::vector<const ForkJoin *> get_topo_sorted_fjs() { return topo_sort_fjs; }
    std::vector<ForkJoin> &get_fjs() { return fork_joins; }
    std::vector<const ForkJoin *> &get_fjs_buffered_with_instr() { return fjs_buffered_with_instr; }

    // setters

    // add buffered fork_join info to map of buffered fork-joins
    void add_elem_to_buffered_fjs(graphlib::NodeId fork_id, FJBufferingInfo buff_fj_info);
    // erase element with the key fork_id and index idx from the map
    void erase_elem_from_buffered_fjs(graphlib::NodeId fork_id, std::size_t idx);

    void add_fj_buffered_with_instr(const ForkJoin *fj) { fjs_buffered_with_instr.push_back(fj); }
};

}  // namespace tt
