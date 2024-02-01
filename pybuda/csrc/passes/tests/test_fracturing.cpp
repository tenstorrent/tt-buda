// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/fracture.hpp"
#include "test/common.hpp"

namespace tt::test
{

template <typename T>
static int count_nodes(
    std::vector<graphlib::Node*> const& nodes, std::string const& name_filter, std::string const& op_type = "")
{
    return static_cast<int>(std::count_if(
        nodes.begin(),
        nodes.end(),
        [&name_filter, &op_type](auto* n)
        {
            auto* t = dynamic_cast<T*>(n);
            return t and t->name().find(name_filter) != std::string::npos and
                   (op_type.empty() or (t->template as<graphlib::OpNode>()->op_name() == op_type));
        }));
}

static bool fully_connected(graphlib::Graph* graph)
{
    auto nodes = graphlib::topological_sort(*graph);
    for (auto* node : nodes)
    {
        auto operands = graph->data_operands(node);
        auto users = graph->data_users(node);
        if (node->node_type() == graphlib::kInput)
        {
            if (users.empty())
                return false;
        }
        else if (node->node_type() == graphlib::kOutput)
        {
            if (operands.empty())
                return false;
        }
        else
        {
            if (operands.empty() or users.empty())
                return false;
        }
    }

    return true;
}

struct FractureFF : public PybudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        std::uint32_t seq_len = 128;
        std::uint32_t embed = 384;
        std::uint32_t hidden = 512;

        auto act = create_activation(shape(1, 1, seq_len, embed));
        auto Win = create_parameter(shape(1, 1, embed, hidden));
        auto Wout = create_parameter(shape(1, 1, hidden, embed));

        auto e0 = create_op("matmul", {act, Win});
        auto gelu = create_op("gelu", {e0});
        auto e1 = create_op("matmul", {gelu, Wout});

        Win_name = Win->name();
        Wout_name = Wout->name();
        e0_name = e0->name();
        gelu_name = gelu->name();
        e1_name = e1->name();

        return {e1};
    }

    std::string Win_name;
    std::string Wout_name;
    std::string e0_name;
    std::string e1_name;
    std::string gelu_name;
};

TEST_F(FractureFF, 1d_weight_stationary)
{
    graphlib::Graph* graph = get_graph();

    passes::fracture(
        graph,
        {{{
              {Win_name, {-1}, {2}},
              {Wout_name, {-2}, {2}},
              {e0_name, {}, {}},
              {gelu_name, {}, {}},
              {e1_name, {}, {}},
          },
          {}}});

    auto nodes = graphlib::topological_sort(*graph);

    EXPECT_TRUE(fully_connected(graph));
    EXPECT_EQ(count_nodes<graphlib::InputNode>(nodes, Win_name), 2);
    EXPECT_EQ(count_nodes<graphlib::InputNode>(nodes, Wout_name), 2);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, e0_name, "matmul"), 2);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, gelu_name, "gelu"), 2);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, e1_name, "matmul"), 2);
}

TEST_F(FractureFF, 2d_weight_stationary)
{
    graphlib::Graph* graph = get_graph();

    passes::fracture(
        graph,
        {{{
              {Win_name, {-2, -1}, {2, 4}},
              {Wout_name, {-2, -1}, {4, 3}},
              {e0_name, {}, {}},
              {gelu_name, {}, {}},
              {e1_name, {}, {}},
          },
          {}}});

    auto nodes = graphlib::topological_sort(*graph);

    EXPECT_TRUE(fully_connected(graph));
    EXPECT_EQ(count_nodes<graphlib::InputNode>(nodes, Win_name), 2 * 4);
    EXPECT_EQ(count_nodes<graphlib::InputNode>(nodes, Wout_name), 4 * 3);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, e0_name, "matmul"), 2 * 4);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, gelu_name, "gelu"), 4);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, e1_name, "matmul"), 4 * 3);
}

TEST_F(FractureFF, 2d_weight_stationary_inferred)
{
    graphlib::Graph* graph = get_graph();

    passes::fracture(
        graph,
        {{{
              {Win_name, {-2, -1}, {2, 4}},
              {Wout_name, {-2, -1}, {4, 3}},
          },
          {}}});

    auto nodes = graphlib::topological_sort(*graph);

    EXPECT_TRUE(fully_connected(graph));
    EXPECT_EQ(count_nodes<graphlib::InputNode>(nodes, Win_name), 2 * 4);
    EXPECT_EQ(count_nodes<graphlib::InputNode>(nodes, Wout_name), 4 * 3);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, e0_name, "matmul"), 2 * 4);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, gelu_name, "gelu"), 4);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, e1_name, "matmul"), 4 * 3);
}

TEST_F(FractureFF, 2d_weight_stationary_mixed_factors)
{
    graphlib::Graph* graph = get_graph();

    passes::fracture(
        graph,
        {{{
              {Win_name, {-2, -1}, {2, 4}},
              {Wout_name, {-2, -1}, {2, 2}},
          },
          {}}});

    auto nodes = graphlib::topological_sort(*graph);

    EXPECT_TRUE(fully_connected(graph));
    EXPECT_EQ(count_nodes<graphlib::InputNode>(nodes, Win_name), 2 * 4);
    EXPECT_EQ(count_nodes<graphlib::InputNode>(nodes, Wout_name), 2 * 2);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, e0_name, "matmul"), 2 * 4);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, gelu_name, "gelu"), 4);
    EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, e1_name, "matmul"), 2 * 2);
}

struct FractureForkJoin : public PybudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        std::uint32_t seq_len = 128;
        std::uint32_t embed = 384;
        std::uint32_t hidden = 512;

        auto act = create_activation(shape(1, 1, seq_len, embed));
        auto y = create_activation(shape(1, 16, seq_len, 32));
        auto Win = create_parameter(shape(1, 1, embed, hidden));

        auto e0 = create_op("matmul", {act, Win});
        auto hslice = create_op("hslice", {e0}, {16});
        auto mul = create_op("multiply", {hslice, y});
        auto gelu0 = create_op("gelu", {mul});
        auto gelu1 = create_op("gelu", {hslice});
        auto join = create_op("multiply", {gelu0, gelu1});

        e0_name = e0->name();
        join_name = join->name();

        return {join};
    }

    std::string e0_name;
    std::string join_name;
};

TEST_F(FractureForkJoin, fracture_fork_join)
{
    graphlib::Graph* graph = get_graph();

    passes::fracture(
        graph,
        {{{
              {e0_name, {-2}, {2}},
              {join_name, {-2}, {2}},
          },
          {}}});

    auto nodes = graphlib::topological_sort(*graph);

    for (auto* node : nodes)
    {
        graphlib::OpNode* op = dynamic_cast<graphlib::OpNode*>(node);
        if (not op)
            continue;

        // Enforce that all of the gathers got wired up by ensuring that all ops have the expected number of inputs
        if (op->op_name() == "matmul" or op->op_name() == "multiply" or op->op_name() == "concatenate")
            EXPECT_EQ(graph->operands(op).size(), 2);
        else if (op->op_name() == "gelu" or op->op_name() == "hslice" or op->op_name() == "select")
            EXPECT_EQ(graph->operands(op).size(), 1);
        else
            FAIL();
    }
}

struct FractureDimSwitch : public PybudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        std::uint32_t seq_len = 128;
        std::uint32_t embed = 384;
        std::uint32_t hidden = 512;

        auto act = create_activation(shape(1, 1, seq_len, embed));
        auto Win = create_parameter(shape(1, 1, embed, hidden));

        auto gelu0 = create_op("gelu", {act});
        auto ff1 = create_op("matmul", {gelu0, Win});

        gelu0_name = gelu0->name();
        ff1_name = ff1->name();

        return {ff1};
    }

    std::string gelu0_name;
    std::string ff1_name;
};

TEST_F(FractureDimSwitch, dim_switch)
{
    graphlib::Graph* graph = get_graph();

    passes::fracture(
        graph,
        {{{
              {gelu0_name, {-2, -1}, {2, 1}},
              {ff1_name, {-2, -1}, {1, 4}},
          },
          {}}});

    EXPECT_TRUE(fully_connected(graph));
}

struct FractureSimpleMixedFactors : public PybudaGraphTest, public testing::WithParamInterface<std::pair<int, int>>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        std::uint32_t r = 128;
        std::uint32_t c = 128;

        auto act = create_activation(shape(1, 1, r, c));

        auto gelu0 = create_op("gelu", {act});
        auto gelu1 = create_op("gelu", {gelu0});

        gelu0_name = gelu0->name();
        gelu1_name = gelu1->name();

        return {gelu1};
    }

    std::string gelu0_name;
    std::string gelu1_name;
};

TEST_P(FractureSimpleMixedFactors, simple_mixed_factors)
{
    graphlib::Graph* graph = get_graph();
    auto [f0, f1] = GetParam();

    passes::fracture(
        graph,
        {{{
              {gelu0_name, {-1}, {f0}},
              {gelu1_name, {-1}, {f1}},
          },
          {}}});

    auto nodes = graphlib::topological_sort(*graph);
    EXPECT_TRUE(fully_connected(graph));
}

INSTANTIATE_TEST_SUITE_P(
    FractureSimpleMixedFactors,
    FractureSimpleMixedFactors,
    testing::Values(
        std::make_pair<int>(1, 2),
        std::make_pair<int>(1, 4),
        std::make_pair<int>(2, 4),
        std::make_pair<int>(4, 2),
        std::make_pair<int>(4, 1)));

struct FractureLayers : public PybudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        std::uint32_t seq_len = 128;
        std::uint32_t embed = 384;
        std::uint32_t hidden = 512;

        auto act = create_activation(shape(1, 1, seq_len, embed));

        OpType* out = nullptr;
        for (int layer = 0; layer < num_layers; ++layer)
        {
            auto Win = create_parameter(shape(1, 1, embed, hidden));
            auto Wout = create_parameter(shape(1, 1, hidden, embed));
            auto layer_name = "layer." + std::to_string(layer) + ".";

            auto e0 = create_op(layer_name + "e0", "matmul", {out ? (graphlib::Node*)out : (graphlib::Node*)act, Win});
            auto gelu = create_op(layer_name + "gelu", "gelu", {e0});
            auto e1 = create_op(layer_name + "e1", "matmul", {gelu, Wout});

            out = e1;
        }

        return {out};
    }

    int fracture_factor(int) const { return 2; }

    std::vector<OpType*> layer0;
    int num_layers = 3;
};

TEST_F(FractureLayers, regex_layers)
{
    graphlib::Graph* graph = get_graph();
    passes::FractureGroups groups;

    for (int layer = 0; layer < num_layers; ++layer)
    {
        groups.push_back(std::make_tuple<passes::FractureGroup, passes::FractureChipIds>(
            {
                {"layer." + std::to_string(layer) + ".*", {-1}, {fracture_factor(layer)}},
            },
            {}));
    }

    passes::fracture(graph, groups);

    auto nodes = graphlib::topological_sort(*graph);

    EXPECT_TRUE(fully_connected(graph));

    for (int layer = 0; layer < num_layers; ++layer)
    {
        EXPECT_EQ(count_nodes<graphlib::OpNode>(nodes, "layer." + std::to_string(layer)), 12);
    }
}
}  // namespace tt::test
