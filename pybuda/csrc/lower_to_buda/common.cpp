// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "lower_to_buda/common.hpp"
#include "utils/assert.hpp"

namespace tt {

static bool contains(std::string const &str, std::string const &substr) {
    return str.find(substr) != std::string::npos;
}

std::ostream &operator<<(std::ostream &os, DramLoc const &dram_loc)
{
    os << "[" << dram_loc.channel << ", 0x" << std::hex << dram_loc.address << std::dec << "]";
    return os;
}

inline void to_netlist(std::ostream &os, int i) { os << i; }

inline void to_netlist(std::ostream &os, bool b) { os << (b ? "true" : "false"); }

inline void to_netlist(std::ostream &os, std::string const &s) { os << s; }

inline void to_netlist(std::ostream &os, float f) { os << std::scientific << f << std::defaultfloat; }

void to_netlist(std::ostream &os, DramLoc const &dram_loc) { os << dram_loc; }

template <typename K, typename V>
inline void to_netlist(std::ostream &os, std::unordered_map<K, V> const &map);
template <typename T>
inline void to_netlist(std::ostream &os, std::vector<T> const &vec);
template <typename... Ts>
inline void to_netlist(std::ostream &os, std::tuple<Ts...> const &tuple);
template <typename... Ts>
inline void to_netlist(std::ostream &os, std::variant<Ts...> const &variant);

template <typename T>
inline void to_netlist(std::ostream &os, std::vector<T> const &vec)
{
    bool first = true;
    os << "[";
    for (T const &i : vec)
    {
        if (not first)
            os << ", ";
        to_netlist(os, i);
        first = false;
    }
    os << "]";
}

template <typename K, typename V>
inline void to_netlist(std::ostream &os, std::unordered_map<K, V> const &map)
{
    bool first = true;
    os << "{";
    for (auto const &[k, v] : map)
    {
        if (not first)
            os << ", ";
        os << k << ": " << v;
        first = false;
    }
    os << "}";
}

template <typename Tuple, size_t... I>
inline void to_netlist_tuple_helper(std::ostream &os, Tuple const &tuple, std::index_sequence<I...>)
{
    os << "[";
    (..., (os << (I == 0 ? "" : ", ") << std::get<I>(tuple)));
    os << "]";
}

template <typename... Ts>
inline void to_netlist(std::ostream &os, std::tuple<Ts...> const &tuple)
{
    to_netlist_tuple_helper(os, tuple, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename... Ts>
inline void to_netlist(std::ostream &os, std::variant<Ts...> const &variant)
{
    std::visit([&os](auto &&value) { to_netlist(os, value); }, variant);
}

std::ostream &operator<<(std::ostream &os, const BudaName &name)
{
    bool needs_quotes = contains(name.name, " ") or contains(name.name, "/");
    if (needs_quotes)
        os << "\"";
    os << name.name;
    if (needs_quotes)
        os << "\"";
    return os;
}

std::ostream &operator<<(std::ostream &os, const BudaBlocks &bb) {
    TT_ASSERT(bb.z > 0);
    TT_ASSERT(bb.mblock_m > 0);
    TT_ASSERT(bb.mblock_n > 0);
    TT_ASSERT(bb.ublock_rt > 0);
    TT_ASSERT(bb.ublock_ct > 0);
    os << "t: " << bb.z << ", ";
    os << "mblock: [" << bb.mblock_m << ", " << bb.mblock_n << "], ";
    os << "ublock: [" << bb.ublock_rt << ", " << bb.ublock_ct << "]";
    return os;
}

std::ostream &operator<<(std::ostream &os, const BudaOpAttr &attr)
{
    to_netlist(os, attr);
    return os;
}

std::ostream& operator<<(std::ostream &os, const DataFormat &format) {
    switch (format) {
        case DataFormat::Bfp2: os << "Bfp2"; break;
        case DataFormat::Bfp2_b: os << "Bfp2_b"; break;
        case DataFormat::Bfp4: os << "Bfp4"; break;
        case DataFormat::Bfp4_b: os << "Bfp4_b"; break;
        case DataFormat::Bfp8: os << "Bfp8"; break;
        case DataFormat::Bfp8_b: os << "Bfp8_b"; break;
        case DataFormat::Float16: os << "Float16"; break;
        case DataFormat::Float16_b: os << "Float16_b"; break;
        case DataFormat::Float32: os << "Float32"; break;
        case DataFormat::Int8: os << "Int8"; break;
        case DataFormat::Int32: os << "Int32"; break;
        case DataFormat::Lf8: os << "Lf8"; break;
        case DataFormat::UInt16: os << "UInt16"; break;
        case DataFormat::RawUInt8: os << "RawUInt8"; break;
        case DataFormat::RawUInt16: os << "RawUInt16"; break;
        case DataFormat::RawUInt32: os << "RawUInt32"; break;
        case DataFormat::Invalid: os << "Invalid"; break;
        default: throw std::invalid_argument("Unknown format");
    }
    return os;
}

std::ostream& operator<<(std::ostream &os, const MathFidelity &fidelity) {
    switch (fidelity) {
        case MathFidelity::LoFi: os << "LoFi"; break;
        case MathFidelity::HiFi2: os << "HiFi2"; break;
        case MathFidelity::HiFi3: os << "HiFi3"; break;
        case MathFidelity::HiFi4: os << "HiFi4"; break;
        case MathFidelity::Invalid: os << "Invalid"; break;
        default: throw std::invalid_argument("Unknown fidelity");
    }
    return os;
}

MathFidelity string_to_math_fidelity(const std::string& fidelity_string)
{
    const std::unordered_map<std::string, MathFidelity> string_to_fidelity = {
        {"LoFi", MathFidelity::LoFi},
        {"HiFi2", MathFidelity::HiFi2},
        {"HiFi3", MathFidelity::HiFi3},
        {"HiFi4", MathFidelity::HiFi4}
    };
    auto it = string_to_fidelity.find(fidelity_string);
    TT_LOG_ASSERT(it != string_to_fidelity.end(),
        "Error: Cannot find {} in string_to_math_fidelity lookup.", fidelity_string);
    return it->second;
}

std::uint32_t data_format_byte_size(DataFormat df, int elements)
{
    switch (df) {
        case DataFormat::Float32: return 4 * elements;
        case DataFormat::UInt16:
        case DataFormat::Float16_b:
        case DataFormat::Float16: return 2 * elements;
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp8: return (elements + elements/16); 
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp4: return (elements / 2 + elements / 16);
        case DataFormat::Bfp2_b:
        case DataFormat::Bfp2: return (elements / 4 + elements / 16);
        case DataFormat::Lf8:
        case DataFormat::Int8: return elements;
        case DataFormat::Int32: return 4 * elements;
        case DataFormat::RawUInt8: return elements;
        case DataFormat::RawUInt16: return 2 * elements;
        case DataFormat::RawUInt32: return 4 * elements;
        case DataFormat::Invalid: return 0;
    }
    throw std::runtime_error("Invalid format");

}

MathFidelity string_to_data_format(const std::string& fidelity_string)
{
    const std::unordered_map<std::string, MathFidelity> string_to_fidelity = {
        {"LoFi", MathFidelity::LoFi},
        {"HiFi2", MathFidelity::HiFi2},
        {"HiFi3", MathFidelity::HiFi3},
        {"HiFi4", MathFidelity::HiFi4}
    };
    auto it = string_to_fidelity.find(fidelity_string);
    TT_LOG_ASSERT(it != string_to_fidelity.end(),
        "Error: Cannot find {} in string_to_math_fidelity lookup.", fidelity_string);
    return it->second;
}

}

