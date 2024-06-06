// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <tuple>

#include "balancer/balancer_cache_collection.hpp"
#include "balancer/balancer_utils.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "graph_lib/node_types.hpp"

namespace tt::graphlib
{
class Graph;
class Node;
class Shape;
}  // namespace tt::graphlib

namespace tt::balancer
{
struct BalancerConfig;
}

namespace tt::padding_placer
{

struct Padding
{
    // Padding record for preserving the most important information during the padding pass.
    // For each node in the graph we can have the particular padding structure.
    // So, here we should preserve only things specific for particular node, not something common for all nodes.

    // Original shape of the node.
    tt::graphlib::Shape orig_shape;

    // Sparse matmul R dimension attribute.
    int sparse_r_attr = 0;

    // Buda.
    std::uint32_t pad_lhs_rt = 0;
    std::uint32_t pad_lhs_ct = 0;
    std::uint32_t pad_rhs_ct = 0;

    // If we have added nop input edge of the node with NodeId map will hold true for that NodeId.
    std::unordered_set<tt::graphlib::NodeId> added_nop;
};

// Padding criterion says how we want to compute
// pad number for a given tensor
enum PaddingCriterion
{
    // These are criterions for all ops except sparse matmul
    PRIME_NUMBER = 0,
    POWER_OF_TWO = 1,
    MULTIPLE_OF_TILE = 2,
    PRIME_TILE = 3,
    MULTIPLE_12_OF_TILE = 4,
    MULTIPLE_10_OF_TILE = 5,
    BIGGEST_FACTOR_PRIME_10 = 6,
    BIGGEST_FACTOR_PRIME_10_INCREMENT = 7,

    // These are criterions specific for sparse matmul ops
    SPARSE_MATMUL_BASIC = 20,
    SPARSE_MATMUL_FACTORS = 21,
};

enum PaddingDimension
{
    R = 0,
    C = 1,
    Z = 2,
    W = 3
};

enum PaddingOperation
{
    ALL = 0,
    CONVOLUTION = 1,
    POOLING = 2,
    ELEMENT_WISE = 3,
    MATMUL = 4,
    REDUCE = 5,
    NN = 6,
    TM = 7,
};

bool pad_pass_placer(
    tt::graphlib::Graph *,
    // This parameter represents nodes that should be padded
    const std::unordered_map<tt::graphlib::Node *, const balancer::BudaOpNodeLegalizerFailureInfo> &,
    const tt::balancer::BalancerConfig &,
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection);

std::unordered_map<tt::graphlib::Node *, const tt::balancer::BudaOpNodeLegalizerFailureInfo> check_node_legality(
    tt::graphlib::Graph *,
    tt::graphlib::Node *,
    const tt::balancer::BalancerConfig &,
    std::shared_ptr<balancer::BalancerCacheCollection>);

void remove_padding(tt::graphlib::Graph *, tt::graphlib::Node *, const Padding &);

void reset_smm(tt::graphlib::Graph *, tt::graphlib::Node *, const Padding &);

void remove_pad(tt::graphlib::Graph *, tt::graphlib::Node *, const Padding &);

void reset_bias_matmul_input(Graph *, Node *, const Padding &);

void remove_unpad(tt::graphlib::Graph *, tt::graphlib::Node * /* , Padding &padding */);

void remove_input_padding_nop(Graph *, Node *);

void remove_pad_tm(Graph *, Node *);

void remove_buda_unpad(tt::graphlib::Graph *, tt::graphlib::Node *);

bool pad_node(tt::graphlib::Graph *, tt::graphlib::Node *, Padding &, bool add_nop_on_input_edge = false);

bool pad_eltwise(tt::graphlib::Graph *, tt::graphlib::Node *, Padding &, bool add_nop_on_input_edge = false);

bool pad_matmul(tt::graphlib::Graph *, tt::graphlib::Node *, Padding &, bool add_nop_on_input_edge = false);

bool pad_smm(tt::graphlib::Graph *, tt::graphlib::Node *, Padding &, bool add_nop_on_input_edge = false);

// TODO: In Progress.
// bool pad_splice(tt::graphlib::Graph *, tt::graphlib::Node *);

// TODO: In Progress.
// bool pad_fused_op(tt::graphlib::Graph *, tt::graphlib::Node *);

// TODO: In Progress
// void remove_redundant_pad(tt::graphlib::Graph *);

// TODO: In Progress
// void remove_redudant_pad_bfs(tt::graphlib::Graph *);

// TODO: In Progress
// void remove_redudant_pad_dfs(tt::graphlib::Graph *);

void insert_pad_smm(tt::graphlib::Node *, std::uint32_t, std::uint32_t);

void insert_pad_buda(tt::graphlib::Graph *, tt::graphlib::Edge, std::uint32_t, std::uint32_t, float, bool insert_nop = false);

void insert_unpad_buda(
    tt::graphlib::Graph *,
    tt::graphlib::Node *,
    tt::graphlib::Edge,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t orig_r = 0,
    std::uint32_t orig_c = 0);

void set_padded_node_out_shape(Node *, Padding &);

void insert_unpad(tt::graphlib::Graph *, tt::graphlib::Node *, tt::graphlib::Edge, Padding &, bool);

std::vector<Node*> insert_queue(tt::graphlib::Graph *, tt::graphlib::Node *);

tt::graphlib::BudaOpNode *create_op(
    tt::graphlib::Graph *,
    tt::graphlib::Node *,
    tt::graphlib::Shape,
    std::vector<tt::graphlib::OpType::Attr>,
    std::string,
    std::string);

tt::graphlib::BudaOpNode *create_nop(tt::graphlib::Graph *, tt::graphlib::Node *, std::string);

bool check_op_type(std::string, tt::padding_placer::PaddingOperation);

bool is_irregular(tt::graphlib::Graph *, tt::graphlib::Node *, Padding &, PaddingCriterion);

bool is_irregular_element_wise(tt::graphlib::Node *, Padding &, PaddingCriterion);

bool is_irregular_matmul(tt::graphlib::Graph *, tt::graphlib::Node *, Padding &, PaddingCriterion);

bool is_irregular_smm(tt::graphlib::Graph *, tt::graphlib::Node *, Padding &, PaddingCriterion);

bool is_irregular(std::uint32_t, PaddingCriterion);

bool is_sparse_irregular(std::uint32_t, std::uint32_t, PaddingCriterion);

bool is_sparse_irregular_tiles(std::uint32_t, std::uint32_t);

bool is_prime(std::uint32_t);

bool is_tile_prime(std::uint32_t);

bool is_power_of_2(std::uint32_t);

bool is_multiple_12_of_tile(std::uint32_t);

bool is_multiple_10_of_tile(std::uint32_t);

bool is_biggest_factor_prime(std::uint32_t, std::uint32_t);

std::vector<std::uint32_t> prime_factorize(std::uint32_t);

std::uint32_t round_power_of_2(std::uint32_t);

std::uint32_t round_tile_prime(std::uint32_t);

std::uint32_t round_multiple_12_of_tile(std::uint32_t);

std::uint32_t round_multiple_10_of_tile(std::uint32_t);

std::uint32_t round_biggest_factor_prime(std::uint32_t, std::uint32_t);

std::vector<std::uint32_t> round_biggest_factor_prime_inner(std::uint32_t, std::uint32_t);

std::uint32_t increment_until_valid(std::uint32_t, PaddingCriterion);

void compute_pad_eltwise(tt::graphlib::Node *, Padding &, PaddingCriterion);

void compute_pad_matmul(tt::graphlib::Graph *, tt::graphlib::Node *, Padding &, PaddingCriterion);

void compute_pad_smm(tt::graphlib::Graph *, tt::graphlib::Node *, Padding &, PaddingCriterion);

std::uint32_t compute_pad(std::uint32_t, PaddingCriterion);

std::uint32_t compute_sparse_pad(std::uint32_t, std::uint32_t, PaddingCriterion);

std::uint32_t compute_sparse_pad_basic(std::uint32_t, std::uint32_t);

std::uint32_t compute_sparse_pad_factors(std::uint32_t, std::uint32_t);

bool check_shape_dims(tt::graphlib::Shape);

bool check_shape_size(tt::graphlib::Shape);

bool check_shape_ones(tt::graphlib::Shape);

void reset_broadcast_tm_on_edge(Graph *, Edge , std::uint32_t , std::uint32_t );

std::pair<bool,bool> update_broadcast_op_with_pad(graphlib::Graph *, graphlib::Edge, std::uint32_t, std::uint32_t, bool increase = true);

void update_bias_tms(graphlib::Graph *, graphlib::Edge, std::uint32_t, std::uint32_t);

std::string convert_pad_op(PaddingOperation);

std::uint32_t get_tiles_num(std::uint32_t);

bool change_result(tt::graphlib::Node *);

std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> extract_dimensions_smm(
    tt::graphlib::Graph *, tt::graphlib::Node *);

std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> extract_dimensions_matmul(
    tt::graphlib::Graph *, tt::graphlib::Node *);

std::tuple<std::uint32_t, std::uint32_t> extract_dimensions_eltwise(tt::graphlib::Node *);

// void update_padding_matmul();

// void update_padding_eltwise();

}  // namespace tt::padding_placer
