// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once


#include "passes/passes_utils.hpp"


#include <string>
#include <unordered_map>
namespace tt::passes
{
    using OpType = tt::graphlib::OpType;
    struct OpTypeItem {
        std::string op_name;
        std::vector<OpType::Attr> attrs;
        bool check_attrs;
        OpTypeItem(OpType const& op_type, bool check_attrs) :
            op_name(op_type.op),
            attrs(
                op_type.op == "transpose" ? std::vector<OpType::Attr>(
                                                {op_type.get_attr_as<int>("dim0"),
                                                 op_type.get_attr_as<int>("dim1"),
                                                 op_type.get_attr_as<int>("z_dim_slice")})
                                          : op_type.attr),
            check_attrs(check_attrs)
        {
        }

        OpTypeItem(std::string const& op_name, std::vector<OpType::Attr> attrs, bool check_attrs) :
            op_name(op_name), attrs(attrs), check_attrs(check_attrs)
        {
        }

        OpType as_op_type() const
        {
            return op_name == "transpose" ? graphlib::OpType(
                                                op_name,
                                                {},
                                                {},
                                                {{"dim0", std::get<int>(attrs[0])},
                                                 {"dim1", std::get<int>(attrs[1])},
                                                 {"z_dim_slice", std::get<int>(attrs[1])}})
                                          : graphlib::OpType(op_name, attrs);
        }
    };

    using TMPattern = std::vector<OpTypeItem>;
    using TMPatternPairs = std::vector<std::pair<TMPattern, std::vector<TMPattern>>>;

    // PreDefine TM sequence pattern
    static TMPattern pattern_0 = {
        OpTypeItem("vslice", {}, false),
        OpTypeItem("transpose", {-3, -1, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
    };

    static TMPattern replace_0 = {
        OpTypeItem("transpose", {-2, -1, -1}, true),
    };


    static TMPattern pattern_1 = {
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-3, -1, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
    };

    static TMPattern replace_1 = {
        OpTypeItem("transpose", {-2, -1, -1}, true),
    };

    static TMPattern pattern_2 = {
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-3, -2, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
    };

    static TMPattern replace_2_0 = {
        OpTypeItem("reshape", {1, 2166, 21}, false),
    };

    static TMPattern replace_2_1 = {
        OpTypeItem("reshape", {1, 600, 21}, false),
    };

    static TMPattern replace_2_2 = {
        OpTypeItem("reshape", {1, 150, 21}, false),
    };

    static TMPattern replace_2_3 = {
        OpTypeItem("reshape", {1, 54, 21}, false),
    };

    static TMPattern replace_2_4 = {
        OpTypeItem("reshape", {1, 24, 21}, false),
    };

    static TMPattern replace_2_5 = {
        OpTypeItem("reshape", {1, 6, 21}, false),
    };

    static TMPattern replace_2_6 = {
        OpTypeItem("reshape", {1, 2166, 4}, false),
    };
    
    static TMPattern replace_2_7 = {
        OpTypeItem("reshape", {1, 600, 4}, false),
    };

    static TMPattern replace_2_8 = {
        OpTypeItem("reshape", {1, 150, 4}, false),
    };

    static TMPattern replace_2_9 = {
        OpTypeItem("reshape", {1, 54, 4}, false),
    };

    static TMPattern replace_2_10 = {
        OpTypeItem("reshape", {1, 24, 4}, false),
    };

    static TMPattern replace_2_11 = {
        OpTypeItem("reshape", {1, 6, 4}, false),
    };

    static TMPattern replace_2_12 = {
        OpTypeItem("reshape", {1, 384, 12}, false),
    };    

    static TMPattern replace_2_13 = {
        OpTypeItem("reshape", {1, 512, 12}, false),
    };

    static TMPattern replace_2_14 = {
        OpTypeItem("reshape", {1, 384, 1}, false),
    };

    static TMPattern replace_2_15 = {
        OpTypeItem("reshape", {1, 512, 1}, false),
    };

    static TMPattern pattern_3 = {
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-4, -2, -1}, true),
        OpTypeItem("transpose", {-3, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
    };

    static TMPattern replace_3_0 = {
        OpTypeItem("reshape", {1, 90000, 91}, false),
    };

    static TMPattern replace_3_1 = {
        OpTypeItem("reshape", {1, 22500, 91}, false),
    };

    static TMPattern replace_3_2 = {
        OpTypeItem("reshape", {1, 5625, 91}, false),
    };

    static TMPattern replace_3_3 = {
        OpTypeItem("reshape", {1, 1521, 91}, false),
    };

    static TMPattern replace_3_4 = {
        OpTypeItem("reshape", {1, 441, 91}, false),
    };

    static TMPattern replace_3_5 = {
        OpTypeItem("reshape", {1, 90000, 4}, false),
    };

    static TMPattern replace_3_6 = {
        OpTypeItem("reshape", {1, 22500, 4}, false),
    };

    static TMPattern replace_3_7 = {
        OpTypeItem("reshape", {1, 5625, 4}, false),
    };

    static TMPattern replace_3_8 = {
        OpTypeItem("reshape", {1, 1521, 4}, false),
    };

    static TMPattern replace_3_9 = {
        OpTypeItem("reshape", {1, 441, 4}, false),
    };

    static TMPattern pattern_4 = {
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-3, -2, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
    };

    static TMPattern replace_4_0 = {
        OpTypeItem("reshape", {1, 2, 2, 720}, false),
    };

    static TMPattern replace_4_1 = {
        OpTypeItem("reshape", {1, 14, 14, 36}, false),
    };

    static TMPattern replace_4_2 = {
        OpTypeItem("reshape", {1, 7, 7, 720}, false),
    };

    static TMPattern replace_4_3 = {
        OpTypeItem("reshape", {1, 7, 7, 36}, false),
    };
    
    static TMPattern replace_4_4 = {
        OpTypeItem("reshape", {1, 4, 4, 720}, false),
    };

    static TMPattern replace_4_5 = {
        OpTypeItem("reshape", {1, 4, 4, 36}, false),
    };

    static TMPattern replace_4_6 = {
        OpTypeItem("reshape", {1, 14, 14, 720}, false),
    };

    static TMPattern replace_4_7 = {
        OpTypeItem("reshape", {1, 2, 2, 36}, false),
    };

    static TMPattern replace_4_8 = {
        OpTypeItem("reshape", {1, 28, 28, 720}, false),
    };

    static TMPattern replace_4_9 = {
        OpTypeItem("reshape", {1, 28, 28, 36}, false),
    };

    static TMPattern replace_4_10 = {
        OpTypeItem("reshape", {1, 56, 56, 96}, false),
    };

    static TMPattern replace_4_11 = {
        OpTypeItem("reshape", {1, 14, 14, 384}, false),
    };

    static TMPattern replace_4_12 = {
        OpTypeItem("reshape", {1, 7, 7, 768}, false),
    };

    static TMPattern replace_4_13 = {
        OpTypeItem("reshape", {1, 28, 28, 192}, false),
    };

    static TMPattern pattern_5 = {
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-3, -1, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("transpose", {-3, -2, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
    };

    static TMPattern replace_5_0 = {
        OpTypeItem("reshape", {1, 56, 56, 96}, false),
    };

    static TMPattern pattern_6 = {
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-3, -1, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-2, -1, -1}, true),
    };

    static TMPattern replace_6_0 = {
        OpTypeItem("reshape", {1, 1, 3136, 96}, false),
    };

    static TMPattern replace_6_1 = {
        OpTypeItem("reshape", {1, 1, 784, 192}, false),
    };

    static TMPattern replace_6_2 = {
        OpTypeItem("reshape", {1, 1, 196, 384}, false),
    };

    static TMPattern pattern_7 {
        OpTypeItem("transpose", {-3, -1, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-2, -1, -1}, true),
    };

    static TMPattern replace_7_0 = {
        OpTypeItem("reshape", {1, 1, 3136, 96}, false),
    };

    static TMPattern replace_7_1 = {
        OpTypeItem("reshape", {1, 1, 784, 192}, false),
    };

    static TMPattern replace_7_2 = {
        OpTypeItem("reshape", {1, 1, 196, 384}, false),
    };

    static TMPattern pattern_8 {
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-2, -1, -1}, true),
    };

    static TMPattern replace_8_0 = {
        OpTypeItem("reshape", {1, 1, 784, 192}, false),
    };

    static TMPattern pattern_9 = {
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-3, -1, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-1, -2, -1}, true),
    };

    static TMPattern replace_9_0 = {
        OpTypeItem("reshape", {1, 1, 196, 384}, false),
    };

    static TMPattern pattern_10 = {
        OpTypeItem("reshape", {}, false),
        OpTypeItem("transpose", {-3, -1, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("transpose", {-2, -1, -1}, true),
        OpTypeItem("transpose", {-3, -1, -1}, true),
    };

    static TMPattern replace_10_0 = {
        OpTypeItem("reshape", {1, 14, 14, 384}, false),
    };
    
    static TMPatternPairs pattern_map = {
        {pattern_0, {replace_0}},
        {pattern_1, {replace_1}},
        {pattern_2, {
            replace_2_0,
            replace_2_1,
            replace_2_2,
            replace_2_3,
            replace_2_4,
            replace_2_5,
            replace_2_6,
            replace_2_7,
            replace_2_8,
            replace_2_9,
            replace_2_10,
            replace_2_11,
            replace_2_12,
            replace_2_13,
            replace_2_14,
            replace_2_15,
        }},
        {pattern_3, {
            replace_3_0,
            replace_3_1,
            replace_3_2,
            replace_3_3,
            replace_3_4,
            replace_3_5,
            replace_3_6,
            replace_3_7,
            replace_3_8,
            replace_3_9,
        }},
        {pattern_4, {
            replace_4_0,
            replace_4_1,
            replace_4_2,
            replace_4_3,
            replace_4_4,
            replace_4_5,
            replace_4_6,
            replace_4_7,
            replace_4_8,
            replace_4_9,
            replace_4_10,
            replace_4_11,
            replace_4_12,
            replace_4_13,
        }},
        {pattern_5, {replace_5_0}},
        {pattern_6, {
            replace_6_0,
            replace_6_1,
            replace_6_2,
        }},
        {pattern_7, {
            replace_7_0,
            replace_7_1,
            replace_7_2,
        }},
        {pattern_8, {replace_8_0}},
        {pattern_9, {
            replace_9_0,
        }},
        {pattern_10, {
            replace_10_0,
        }},
    };

    bool fuse_tm_sequences(tt::graphlib::Graph* graph, TMPatternPairs& pattern_map_ = pattern_map);
}
