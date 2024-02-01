// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/placer_buda_passes.hpp"
#include "test/common.hpp"

namespace tt::test
{
struct InsertQueues : public BudaGraphTest, public testing::WithParamInterface<int>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto num_forks = GetParam();
        auto act = create_activation(shape(1, 1, 32, 32));
        gelu0 = create_op("gelu", {act});
        for (int i = 0; i < num_forks; ++i)
        {
            forks.push_back(create_op("gelu", {gelu0}));
            forks_as_nodes.push_back(forks.back());
        }
        return forks;
    }

    OpType* gelu0;
    std::vector<OpType*> forks;
    std::vector<graphlib::Node*> forks_as_nodes;
};

TEST_P(InsertQueues, insert_queues)
{
    graphlib::Graph* graph = get_graph();

    balancer::BudaOpNodeLegalizerFailureInfo info;
    info.recordOpModelFailure(balancer::OpModelFailureReason::UserAccessPreventsStreaming);
    std::unordered_map<graphlib::Node*, const balancer::BudaOpNodeLegalizerFailureInfo> nodes_without_legal_op_model = {
        {gelu0, info},
    };
    passes::insert_queues(graph, nodes_without_legal_op_model);
    auto users = graph->data_users(gelu0);
    ASSERT_EQ(users.size(), 1);
    auto *queue = dynamic_cast<graphlib::QueueNode*>(users.front());
    ASSERT_NE(queue, nullptr);
    auto queue_users = graph->data_users(queue);
    std::sort(queue_users.begin(), queue_users.end());
    std::sort(forks_as_nodes.begin(), forks_as_nodes.end());
    ASSERT_EQ(queue_users, forks_as_nodes);
}

INSTANTIATE_TEST_SUITE_P(InsertQueues, InsertQueues, testing::Values(1, 2, 3));
}  // namespace tt::test
