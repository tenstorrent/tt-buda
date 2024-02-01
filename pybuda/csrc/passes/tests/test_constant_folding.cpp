// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/constant_folding.hpp"
#include "test/common.hpp"

namespace tt::test
{
struct ConstantFoldMultiply : public PybudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        std::uint32_t m = 784;
        std::uint32_t k = 64;
        std::uint32_t n = 128;

        auto act = create_activation(shape(1, 1, m, k));
        auto weights = create_parameter(shape(1, 1, k, n));

        auto bn_constant = create_constant(shape(1, 1, 1, n));
        auto bias = create_constant(shape(1, 1, 1, n));

        auto matmul = create_op("matmul", {act, weights});
        auto narrow0 = create_op("narrow", {matmul}, -2, 0, (int)m, (int)m);
        auto narrow1 = create_op("narrow", {narrow0}, -1, 0, (int)n, (int)n);
        auto multiply = create_op("multiply", {bn_constant, narrow1});
        auto add = create_op("add", {bias, multiply});

        multiply_name = multiply->name();
        add_name = add->name();
        narrow_name = narrow0->name();

        return {add};
    }

    std::string multiply_name;
    std::string add_name;
    std::string narrow_name;
};

TEST_F(ConstantFoldMultiply, constant_fold_multiply)
{
    graphlib::Graph* graph = get_graph();

    passes::constant_folding(graph);

    auto nodes = graphlib::topological_sort(*graph);

    // Expect the multiply was erased
    EXPECT_TRUE(std::none_of(nodes.begin(), nodes.end(), [this](auto* n) { return n->name() == this->multiply_name; }));

    // Expect the add to be hoisted ahead of narrow
    auto add_position =
        std::find_if(nodes.begin(), nodes.end(), [this](auto* n) { return n->name() == this->add_name; });
    ASSERT_NE(add_position, nodes.end());
    auto narrow_position =
        std::find_if(nodes.begin(), nodes.end(), [this](auto* n) { return n->name() == this->narrow_name; });
    ASSERT_NE(narrow_position, nodes.end());
    EXPECT_LT(add_position, narrow_position);
}

struct ConstantFoldMultiplyThroughAdd : public PybudaGraphTest, public testing::WithParamInterface<std::pair<int, bool>>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto [levels, nop_mixin] = GetParam();
        int num_matmuls = 1 << levels;

        std::uint32_t m = 784;
        std::uint32_t k = 64;
        std::uint32_t n = 128;

        std::vector<graphlib::InputNode*> acts;
        std::vector<graphlib::InputNode*> weights;

        for (int i = 0; i < num_matmuls; ++i) acts.push_back(create_activation(shape(1, 1, m, k)));
        for (int i = 0; i < num_matmuls; ++i) weights.push_back(create_parameter(shape(1, 1, k, n)));

        auto bn_constant = create_constant(shape(1, 1, 1, n));
        auto bias = create_constant(shape(1, 1, 1, n));

        std::vector<OpType*> matmuls;
        for (int i = 0; i < num_matmuls; ++i) matmuls.push_back(create_op("matmul", {acts[i], weights[i]}));

        TT_ASSERT(matmuls.size() % 2 == 0);
        while (matmuls.size() > 1)
        {
            for (int i = 0; i < (int)(matmuls.size() / 2); ++i)
            {
                matmuls[i] = create_op("add", {matmuls[i * 2], matmuls[i * 2 + 1]});
                if (nop_mixin)
                    matmuls[i] = create_op("nop", {matmuls[i]});
            }
            matmuls.resize(matmuls.size() / 2);
        }

        TT_ASSERT(matmuls.size() == 1);
        auto multiply = create_op("multiply", {bn_constant, matmuls.back()});
        auto add = create_op("add", {bias, multiply});

        multiply_name = multiply->name();

        return {add};
    }

    std::string multiply_name;
};

TEST_P(ConstantFoldMultiplyThroughAdd, constant_fold_multiply_through_add)
{
    graphlib::Graph* graph = get_graph();

    passes::constant_folding(graph);

    auto nodes = graphlib::topological_sort(*graph);

    // Expect the multiply was erased
    EXPECT_TRUE(std::none_of(nodes.begin(), nodes.end(), [this](auto* n) { return n->name() == this->multiply_name; }));
}
INSTANTIATE_TEST_SUITE_P(
    ConstantFoldMultiplyThroughAdd,
    ConstantFoldMultiplyThroughAdd,
    testing::Values(
        std::make_pair(1, false),
        std::make_pair(2, false),
        std::make_pair(3, false),
        std::make_pair(1, true),
        std::make_pair(2, true),
        std::make_pair(3, true)));

struct ConstantFoldBackToBack : public PybudaGraphTest, public testing::WithParamInterface<std::string>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto op_type = GetParam();

        auto act = create_activation(shape(1, 1, 32, 32));
        auto constant0 = create_constant(shape(1, 1, 32, 32));
        auto constant1 = create_constant(shape(1, 1, 32, 32));

        auto multiply0 = create_op(op_type, {act, constant0});
        auto multiply1 = create_op(op_type, {multiply0, constant1});

        erased_op_name = multiply1->name();

        return {multiply1};
    }

    std::string erased_op_name;
};

TEST_P(ConstantFoldBackToBack, constant_fold_back_to_back)
{
    graphlib::Graph* graph = get_graph();

    passes::constant_folding(graph);

    auto nodes = graphlib::topological_sort(*graph);

    // Expect the multiply was erased
    EXPECT_TRUE(
        std::none_of(nodes.begin(), nodes.end(), [this](auto* n) { return n->name() == this->erased_op_name; }));
}
INSTANTIATE_TEST_SUITE_P(BinaryOps, ConstantFoldBackToBack, testing::Values("add", "multiply"));

}  // namespace tt
