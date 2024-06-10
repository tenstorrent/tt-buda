// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <type_traits>

#include "defines.hpp"
#include "gtest/gtest.h"
#include "node_types.hpp"
#include "passes/fork_join.cpp"
#include "passes/fork_join.hpp"
#include "test/common.hpp"
#include "utils.hpp"

using namespace tt;

namespace tt::test {

struct SimpleForkJoin : public BudaGraphTest
{
protected:

    virtual std::vector<OpType*> create_graph() override
    {
        auto in0 = create_activation(1, 1, 32, 32);
        auto in1 = create_activation(1, 1, 32, 32);
        auto in2 = create_activation(1, 1, 32, 32);
        auto in3 = create_activation(1, 1, 32, 32);

        auto main_fork = create_op("add", {in0, in1});
        
        // Long path main fork (main_fork->matmul->matmul->matmul->main_join (add))
        auto matmul1 = create_op("matmul", {main_fork, in2});
        auto matmul2 = create_op("matmul", {matmul1, in3});
        auto matmul3 = create_op("matmul", {matmul2, in3});

        // Short path is a direct link from main_fork to main_join
        auto main_join = create_op("add", {matmul3, main_fork});

        return {main_join};
    }

public:

    FJGraph get_fj_graph()
    {
        Graph* graph = get_graph();
        FJGraph fj_graph = FJGraph(graph);

        EXPECT_EQ(fj_graph.get_fjs().size(), 1);
        return fj_graph;
    }
};

Edge get_edge_between(std::string src_name, std::string dest_name, Graph *graph)
{
    auto src = graph->get_node_by_name(src_name);
    auto dest = graph->get_node_by_name(dest_name);
    std::vector<Edge> edges = graph->get_edges(src, dest);
    EXPECT_EQ(edges.size(), 1);

    return edges[0];
}

template<typename InstrType, typename = std::enable_if_t<std::is_base_of<InsertionInstruction, InstrType>::value>>
std::shared_ptr<InstrType> create_instruction(
    const std::string &src_name,
    const std::string &dest_name,
    uint32_t input_id,
    uint32_t fork_id,
    bool user_defined,
    bool is_fj_buffering
)
{
    static_assert(std::is_same<InstrType, NopInsertionInstruction>::value || std::is_same<InstrType, QueueInsertionInstruction>::value);

    if constexpr (std::is_same<InstrType, NopInsertionInstruction>::value)
    {
        return std::make_shared<NopInsertionInstruction>(
            src_name,
            dest_name,
            false, // hoist_tms
            1, // nop_count
            input_id,
            fork_id,
            user_defined,
            false, // mergeable
            false, // daisy_chain
            false, // request_merge
            is_fj_buffering
        );
    }
    else if constexpr (std::is_same<InstrType, QueueInsertionInstruction>::value)
    {
        return std::make_shared<QueueInsertionInstruction>(
            src_name,
            dest_name,
            false, // hoist_tms
            32, // num_entries
            64, // queue_size
            input_id,
            fork_id,
            user_defined,
            is_fj_buffering
        );
    }

    return nullptr;
}

TEST_F(SimpleForkJoin, MergeNopAndQueueInstr)
{
    Graph *graph = get_graph();
    FJGraph fj_graph = get_fj_graph();
    ForkJoin fj = fj_graph.get_fjs()[0];

    // Take two nodes on the first path and create queue and NOP instructions for the edge between them.
    std::string src_name = fj.first[0]->name();
    std::string dest_name = fj.first.back()->name();
    Edge edge = get_edge_between(src_name, dest_name, graph);

    auto nop_instruction = create_instruction<NopInsertionInstruction>(src_name, dest_name, edge.consumer_input_port_id, edge.producer_output_port_id, false, true);

    auto queue_instruction = create_instruction<QueueInsertionInstruction>(src_name, dest_name, edge.consumer_input_port_id, edge.producer_output_port_id, false, true);

    InsertionInstructionMap prev_instructions;
    prev_instructions.insert({queue_instruction->unique_id(), queue_instruction});

    InsertionInstructionMap new_instructions;
    new_instructions.insert({nop_instruction->unique_id(), nop_instruction});

    InsertionInstructionMap combined_instructions = merge_with_prev_instr(graph, new_instructions, prev_instructions);
    EXPECT_EQ(combined_instructions.size(), 1) << "Expected only one instruction in resulting map. Queue and the NOP should have been merged.";
    EXPECT_EQ(combined_instructions.at(nop_instruction->unique_id())->instr_type, InstructionType::QueueInstruction);
}

TEST_F(SimpleForkJoin, MergeTwoNopInstructions)
{
    Graph *graph = get_graph();
    FJGraph fj_graph = get_fj_graph();
    ForkJoin fj = fj_graph.get_fjs()[0];

    std::string src_name = fj.first[0]->name();
    std::string dest_name = fj.first.back()->name();
    Edge edge = get_edge_between(src_name, dest_name, graph);

    auto nop_instruction = create_instruction<NopInsertionInstruction>(src_name, dest_name, edge.consumer_input_port_id, edge.producer_output_port_id, false, true);
    auto nop_instruction_2 = create_instruction<NopInsertionInstruction>(src_name, dest_name, edge.consumer_input_port_id, edge.producer_output_port_id, false, true);
    nop_instruction_2->nop_count = 10;

    InsertionInstructionMap prev_instructions;
    prev_instructions.insert({nop_instruction->unique_id(), nop_instruction});

    InsertionInstructionMap new_instructions;
    new_instructions.insert({nop_instruction_2->unique_id(), nop_instruction_2});

    InsertionInstructionMap combined_instructions = merge_with_prev_instr(graph, new_instructions, prev_instructions);
    EXPECT_EQ(combined_instructions.size(), 1) << "Expected only one instruction in resulting map. NOPs should have been merged.";
    EXPECT_EQ(combined_instructions.at(nop_instruction->unique_id())->instr_type, InstructionType::NopInstruction);
    EXPECT_EQ(1 + 10, static_cast<NopInsertionInstruction*>(combined_instructions.at(nop_instruction->unique_id()).get())->nop_count);

}

TEST_F(SimpleForkJoin, TestRecoverOriginalInstruction)
{
    Graph *graph = get_graph();
    FJGraph fj_graph = get_fj_graph();
    ForkJoin fj = fj_graph.get_fjs()[0];

    Node* fork_node = fj.first[0];
    std::string src_name = fork_node->name();
    std::string dest_name = fj.first.back()->name();
    Edge edge = get_edge_between(src_name, dest_name, graph);

    auto nop_instruction = create_instruction<NopInsertionInstruction>(src_name, dest_name, edge.consumer_input_port_id, edge.producer_output_port_id, false, true);

    InsertionInstructionMap prev_instructions;
    prev_instructions.insert({nop_instruction->unique_id(), nop_instruction});

    // Add the NOP to the graph.
    nop_instruction->insert(graph);

    // Now create a Queue instruction for a new NOP between original source and the previously added NOP.
    std::string buffer_name{};
    for (const auto& user : graph->users(fork_node))
    {
        if (user->as<graphlib::BudaOpNode>()->is_buffering_op())
        {
            buffer_name = user->name();
            break;
        }
    }

    edge = get_edge_between(src_name, buffer_name, graph);

    auto queue_instruction = create_instruction<QueueInsertionInstruction>(src_name, buffer_name, edge.consumer_input_port_id, edge.producer_output_port_id, false, true);

    InsertionInstructionMap new_instructions;
    new_instructions.insert({queue_instruction->unique_id(), queue_instruction});

    // Merge should recover the original instruction and merge the two instructions into one.
    InsertionInstructionMap combined_instructions = merge_with_prev_instr(graph, new_instructions, prev_instructions);
    EXPECT_EQ(combined_instructions.size(), 1) << "Expected only one instruction in resulting map. Queue and the NOP should have been merged.";
    EXPECT_EQ(combined_instructions.begin()->second->instr_type, InstructionType::QueueInstruction);

    // Merged instruction should have the same unique_id as the original NOP instruction.
    EXPECT_EQ(combined_instructions.begin()->second->unique_id(), nop_instruction->unique_id());

    // Check the same with NOP instruction.
    auto nop_instruction_2 = create_instruction<NopInsertionInstruction>(src_name, buffer_name, edge.consumer_input_port_id, edge.producer_output_port_id, false, true);
    new_instructions.clear();
    new_instructions.insert({nop_instruction_2->unique_id(), nop_instruction_2});

    combined_instructions = merge_with_prev_instr(graph, new_instructions, prev_instructions);
    EXPECT_EQ(combined_instructions.size(), 1) << "Expected only one instruction in resulting map. NOPs should have been merged.";
    EXPECT_EQ(combined_instructions.begin()->second->instr_type, InstructionType::NopInstruction);
    auto resulting_instruction = static_cast<NopInsertionInstruction*>(combined_instructions.begin()->second.get());

    // Merged instruction should have the same unique_id as the original NOP instruction.
    // Additionally, the nop count should be updated.
    EXPECT_EQ(resulting_instruction->unique_id(), nop_instruction->unique_id());
    EXPECT_EQ(resulting_instruction->nop_count, nop_instruction->nop_count + nop_instruction_2->nop_count);
}

// Modify the given field of the new instruction and confirm the unique id changes (compared to the old one).
template <bool should_be_same, typename T, typename V, typename InstrType>
void modify_field_and_check(T& field, V new_value, const std::shared_ptr<InstrType>& instruction, std::shared_ptr<InstrType>& new_instruction)
{
    static_assert(std::is_base_of<InsertionInstruction, InstrType>::value);
    static_assert(std::is_same<InstrType, NopInsertionInstruction>::value || std::is_same<InstrType, QueueInsertionInstruction>::value);

    T old_value = field;
    field = new_value;

    if constexpr (should_be_same)
    {
        EXPECT_EQ(instruction->unique_id(), new_instruction->unique_id());
    }
    else
    {
        EXPECT_NE(instruction->unique_id(), new_instruction->unique_id());
    }

    field = old_value;
}

TEST_F(SimpleForkJoin, TestUniqueIdSanity)
{
    const std::string src_name = "src";
    const std::string dest_name = "dest";
    const uint32_t input_id = 1;
    const uint32_t fork_id = 0;
    const bool user_defined = false;
    const bool is_fj_buffering = true;

    // You have modified fields in unique id struct. You should be aware that this structure dictates
    // how we differentiate between instructions (in instruction maps). The two instructions with the 
    // same unique id (key) should not and won't coexist. For e.g., if we have two instructions (one 
    // nop and one for a queue) on the same edge, we shouldn't insert them both, so to guard against
    // that, we use the unique id as the key for the instruction map (they will have the same id).
    //
    // Bear in mind that the insertion instructions are used in both user and internal space (fork-join buffering).
    static_assert(std::tuple_size_v<InsInstructionUniqueId> == 6);

    {
        auto nop_instruction = create_instruction<NopInsertionInstruction>(src_name, dest_name, input_id, fork_id, user_defined, is_fj_buffering);
        auto queue_instruction = create_instruction<QueueInsertionInstruction>(src_name, dest_name, input_id, fork_id, user_defined, is_fj_buffering);

        // We have created two instructions with the same parameters. Even though they are different types, they should have the same unique id.
        // This is because they go to the same map of instructions - that way we can detect "overlapping" instructions.
        EXPECT_EQ(nop_instruction->unique_id(), queue_instruction->unique_id());
    }

    {
        auto nop_instruction = create_instruction<NopInsertionInstruction>(src_name, dest_name, input_id, fork_id, false, true);
        auto new_instruction = std::make_shared<NopInsertionInstruction>(*nop_instruction.get());

        // Modify nop-specific fields to make different instructions.
        // The unique ids should NOT change.
        modify_field_and_check<true>(new_instruction->nop_count, new_instruction->nop_count + 1, nop_instruction, new_instruction);
        modify_field_and_check<true>(new_instruction->daisy_chain, !new_instruction->daisy_chain, nop_instruction, new_instruction);
        modify_field_and_check<true>(new_instruction->request_merge, !new_instruction->request_merge, nop_instruction, new_instruction);
        modify_field_and_check<true>(new_instruction->mergeable, !new_instruction->mergeable, nop_instruction, new_instruction);
        modify_field_and_check<true>(new_instruction->hoist_tms, !new_instruction->hoist_tms, nop_instruction, new_instruction);


        // Modify shared fields (of the base class) to make different instructions.
        // The unique ids should also change.
        new_instruction = std::make_shared<NopInsertionInstruction>(*nop_instruction.get());
        modify_field_and_check<false>(new_instruction->src, "new_" + src_name, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->dest, "new_" + dest_name, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->input_id, input_id + 1, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->fork_id, fork_id + 1, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->user_defined, !user_defined, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->is_fj_buffering, !is_fj_buffering, nop_instruction, new_instruction);
    }

    {
        auto nop_instruction = create_instruction<QueueInsertionInstruction>(src_name, dest_name, input_id, fork_id, false, true);
        auto new_instruction = std::make_shared<QueueInsertionInstruction>(*nop_instruction.get());

        // Modify queue-specific fields to make different instructions.
        // The unique ids should NOT change.
        modify_field_and_check<true>(new_instruction->num_entries, new_instruction->num_entries + 1, nop_instruction, new_instruction);
        modify_field_and_check<true>(new_instruction->queue_size, new_instruction->queue_size + 1, nop_instruction, new_instruction);

        // Modify shared fields (of the base class) to make different instructions.
        // The unique ids should also change.
        new_instruction = std::make_shared<QueueInsertionInstruction>(*nop_instruction.get());
        modify_field_and_check<false>(new_instruction->src, "new_" + src_name, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->dest, "new_" + dest_name, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->input_id, input_id + 1, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->fork_id, fork_id + 1, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->user_defined, !user_defined, nop_instruction, new_instruction);
        modify_field_and_check<false>(new_instruction->is_fj_buffering, !is_fj_buffering, nop_instruction, new_instruction);
    }
}

TEST_F(SimpleForkJoin, TestMergeableNops)
{
    Graph *graph = get_graph();
    FJGraph fj_graph = get_fj_graph();
    ForkJoin fj = fj_graph.get_fjs()[0];

    std::string src_name = fj.first.front()->name();
    std::string dest_1_name = fj.first[1]->name();
    std::string dest_2_name = fj.second[1]->name();

    Edge edge_1 = get_edge_between(src_name, dest_1_name, graph);
    auto nop_instruction = create_instruction<NopInsertionInstruction>(src_name, dest_1_name, edge_1.consumer_input_port_id, edge_1.producer_output_port_id, false, true);
    nop_instruction->mergeable = true;

    Edge edge_2 = get_edge_between(src_name, dest_2_name, graph);
    auto nop_instruction_2 = create_instruction<NopInsertionInstruction>(src_name, dest_2_name, edge_2.consumer_input_port_id, edge_2.producer_output_port_id, false, true);
    nop_instruction_2->mergeable = true;
    nop_instruction_2->request_merge = true;

    // NOTE: the ordering of the instructions matters due to the fact
    // that we will merge the nops after we encounter the NOP instruction
    // which has request_merge flag set. The map we use for storing the instructions
    // has a property that the order in which items are inserted is preserved (i.e.
    // it is not 'ordered' in the usual way - by keys).
    //
    // So if we would insert the second instruction first, we would end up with
    // two NOPs in the graph.
    InsertionInstructionMap instructions;
    instructions.insert({nop_instruction->unique_id(), nop_instruction});
    instructions.insert({nop_instruction_2->unique_id(), nop_instruction_2});

    for (const auto& [id, instruction] : instructions)
    {
        instruction->insert(graph);
    }

    uint32_t nop_count = 0;
    for (const auto node : graph->nodes_by_type(graphlib::kBudaOp))
    {
        nop_count += node->as<graphlib::BudaOpNode>()->is_buffering_op();
    }

    // The NOPs should have been merged into one.
    EXPECT_EQ(nop_count, 1);
}

struct AddToMatmul : public BudaGraphTest
{
protected:

    virtual std::vector<OpType*> create_graph() override
    {
        auto in0 = create_activation(1, 1, 32, 32);
        auto in1 = create_activation(1, 1, 32, 32);

        auto add = create_op("add", {in0, in1});
        auto matmul = create_op("matmul", {add, add});

        return {matmul};
    }
};

TEST_F(AddToMatmul, TestInsertNOPMultiEdge)
{
    Graph *graph = get_graph();
    Node* src_node = graph->get_node_by_name("add0");

    Node* matmul = nullptr;
    for (auto& user : graph->users(src_node))
    {
        if (user->as<graphlib::BudaOpNode>()->is_matmul())
        {
            matmul = user;
        }
    }

    auto nop_instr = create_instruction<NopInsertionInstruction>("add0", "matmul0", 0, 0, false, true);

    nop_instr->insert(graph);

    auto edges = graph->get_edges(src_node, matmul);
    EXPECT_EQ(edges.size(), 0);

    auto nop = graph->users(src_node)[0];

    EXPECT_EQ(graph->get_edges(src_node, nop).size(), 1);
    EXPECT_EQ(graph->get_edges(nop, matmul).size(), 2);
}

} // namespace tt::test
