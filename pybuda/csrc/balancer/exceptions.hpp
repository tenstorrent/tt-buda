// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <exception>
#include <graph_lib/node.hpp>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>
namespace tt::balancer
{

// New failure reason can be defined after initial NoFailure value but prior to MaxFailureReason.
//
#define OpModelFailureReasons                                              \
    X(NoFailure, "No Failure - Valid OpModel")                             \
    X(IllegalStackForGrid, "Illegal stack for grid dimension")             \
    X(UserAccessPreventsStreaming, "User access prevents streaming")       \
    X(IllegalStreaming, "Illegal streaming")                               \
    X(L1UsageOverMaxLimit, "L1 usage > L1 Max")                            \
    X(ExceededDramChannelCapacity, "Exceeded DRAM channel capacity")       \
    X(InputBufferAllocationFailure, "Failed to allocate input buffers")    \
    X(PaddingConstraintsNotSatisfied, "Padding constraints not satisfied") \
    X(IllegalSparseMatmul, "Illegal sparse matmul")                        \
    X(MaxFailureReason, "")

#define X(a, b) a,
enum OpModelFailureReason
{
    OpModelFailureReasons
};
#undef X

#define X(a, b) b,
static std::string OpModelFailureReasonMessages[] = {OpModelFailureReasons};
#undef X

class BudaOpNodeLegalizerFailureInfo
{
   private:
    std::uint32_t opModelFailureCountByType[MaxFailureReason];

   public:
    BudaOpNodeLegalizerFailureInfo() : opModelFailureCountByType() {}
    void recordOpModelFailure(OpModelFailureReason failureReason)
    {
        TT_ASSERT(failureReason < MaxFailureReason, "Invalid failure reason.");
        opModelFailureCountByType[failureReason]++;
    }

    std::uint32_t getOpModelFailureCountByType(OpModelFailureReason failureReason) const
    {
        return opModelFailureCountByType[failureReason];
    }

    std::string toString() const
    {
        std::string result = "Op model failure counts by type: \n";

        for (int i = NoFailure; i < MaxFailureReason; i++)
        {
            result += OpModelFailureReasonMessages[i] + ": " + std::to_string(opModelFailureCountByType[i]) + "\n";
        }

        return result;
    }
};

struct BalancerError : public std::exception
{
    struct NodeExceedsMaxOpForks
    {
        std::int64_t max_forks;
        std::int64_t node_id;

        bool specific_node() const { return node_id != -1; }

        NodeExceedsMaxOpForks(std::int64_t max_forks, std::int64_t node_id = -1) :
            max_forks(max_forks), node_id(node_id)
        {
        }
    };

    struct InputBroadcastExceedsMaxGridForks
    {
        std::int64_t input_node_id;

        InputBroadcastExceedsMaxGridForks(std::int64_t input_node_id) : input_node_id(input_node_id) {}
    };

    struct DRAMWriterNOPNeeded
    {
        const std::string src;
        bool transpose;

        DRAMWriterNOPNeeded(const std::string& src, bool transpose) : src(src), transpose(transpose) {}
    };

    struct NoValidGrid
    {
        std::unordered_map<graphlib::Node*, const BudaOpNodeLegalizerFailureInfo> nodes_without_legal_op_model;

        NoValidGrid(
            std::unordered_map<graphlib::Node*, const BudaOpNodeLegalizerFailureInfo>&& nodes_without_legal_op_model) :
            nodes_without_legal_op_model(std::move(nodes_without_legal_op_model))
        {
        }
    };

    // Generic unrecoverable error
    struct Fatal
    {
        std::string message;
        Fatal(const std::string& message) : message(message) {}
    };

    using Type = std::variant<
        std::monostate,
        NodeExceedsMaxOpForks,
        InputBroadcastExceedsMaxGridForks,
        DRAMWriterNOPNeeded,
        NoValidGrid,
        Fatal>;

    std::string message;
    Type type;

    BalancerError(std::string const& message, Type type = std::monostate{}) : message(message), type(type) {}
    virtual char const* what() const noexcept override { return message.c_str(); }
};
}  // namespace tt::balancer
