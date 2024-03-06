// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tt {

struct BudaBlocks {
    int z;
    int ublock_rt, ublock_ct;
    int mblock_m, mblock_n;
};

struct BudaName
{
    std::string name;
    BudaName(std::string const &name) : name(name) {}
};

struct DramLoc
{
    std::uint32_t channel;
    std::uint32_t address;

    DramLoc(std::uint32_t channel, std::uint32_t address) : channel(channel), address(address) {}
    DramLoc(std::pair<std::uint32_t, std::uint32_t> const &p) : DramLoc(p.first, p.second) {}
    bool operator==(DramLoc const &o) const { return channel == o.channel and address == o.address; }
    bool operator==(std::pair<std::uint32_t, std::uint32_t> const &p) const
    {
        return channel == p.first and address == p.second;
    }
};

using BudaOpAttrQueueDramLocs = std::vector<DramLoc>;

using BudaKernelBroadcastInputs = std::unordered_map<std::string, int>;

enum class BudaQueueLayout
{
    Tilized,
    Flat
};

using BudaOpAttr = ::std::variant<
    std::string,
    bool,
    int,
    float,
    std::tuple<int, int, int>,
    std::vector<int>,
    std::vector<std::tuple<int, int, int>>,
    std::vector<std::tuple<int, int, int, int>>,
    BudaOpAttrQueueDramLocs,
    BudaKernelBroadcastInputs>;
using BudaOpAttrs = ::std::map<std::string, BudaOpAttr>;

std::ostream &operator<<(std::ostream &os, const BudaName &name);
std::ostream &operator<<(std::ostream &os, const BudaBlocks &bb);
std::ostream &operator<<(std::ostream &os, const DramLoc &attr);
std::ostream &operator<<(std::ostream &os, const BudaOpAttr &attr);

enum class ExpPrecision : uint8_t
{
  A = 0,
  B = 1,
};

/**
 * @brief Tile dimension enum used to pass variable tile sizes across the SW+HW stack.
 * Only specific dimensions are valid. Please check the enum definition for all valid dimensions.
 */ 
enum class TileDim : std::uint8_t {
    Dim32x32 = 0,
    Dim16x32 = 1,
    Dim32x16 = 2,
    Dim8x32 = 3,
    Dim4x32 = 4,
    Dim2x32 = 5,
    Dim1x32 = 6,
    Default   = Dim32x32,
    Invalid   = 0xff,
};

enum class DataFormat : std::uint8_t
{
    Float32   = 0,
    Float16   = 1,
    Bfp8      = 2,
    Bfp4      = 3,
    Bfp2      = 11,
    Float16_b = 5,
    Bfp8_b    = 6,
    Bfp4_b    = 7,
    Bfp2_b    = 15,
    Lf8       = 10,
    UInt16    = 12,
    Int8      = 14,
    Int32      = 8,
    RawUInt8  = 0xf0,
    RawUInt16 = 0xf1,
    RawUInt32 = 0xf2,
    Invalid   = 0xff
};

enum class MathFidelity : uint8_t
{
    LoFi          = 0,
    HiFi2         = 2,
    HiFi3         = 3,
    HiFi4         = 4,
    Invalid       = 0xff,
};

std::uint32_t data_format_byte_size(DataFormat df, int elements = 1);

inline bool is_integer_data_format(DataFormat df)
{
    switch (df)
    {
        case DataFormat::Lf8:
        case DataFormat::UInt16:
        case DataFormat::Int8:
        case DataFormat::Int32:
        case DataFormat::RawUInt8:
        case DataFormat::RawUInt16:
        case DataFormat::RawUInt32: return true;
        default: return false;
    }
}

inline bool is_valid_accumulate_df(DataFormat df)
{
    switch (df)
    {
        case DataFormat::Float32:   // fallthrough
        case DataFormat::Float16_b: // fallthrough
        case DataFormat::Float16:   // fallthrough
        case DataFormat::Int32: return true;
        default: return false;
    }
}

inline std::uint32_t get_num_fidelity_phases(MathFidelity fidelity)
{
    switch (fidelity)
    {
        case MathFidelity::LoFi: return 1;
        case MathFidelity::HiFi2: return 2;
        case MathFidelity::HiFi3: return 3;
        case MathFidelity::HiFi4: return 4;
        default: return 0;
    }
}

inline MathFidelity get_math_fidelity(std::uint32_t num_fidelity_phases)
{
    switch (num_fidelity_phases)
    {
        case 1: return MathFidelity::LoFi;
        case 2: return MathFidelity::HiFi2;
        case 3: return MathFidelity::HiFi3;
        case 4: return MathFidelity::HiFi4;
        default: return MathFidelity::Invalid;
    }
}

inline bool is_b_data_format(DataFormat df)
{
    switch (df)
    {
        case DataFormat::Float32:
        case DataFormat::Float16_b:
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp2_b: return true;
        default: return false;
    }
}

inline bool is_a_data_format(DataFormat df)
{
    switch(df){
        case DataFormat::Float16:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2: return true;
        default: return false;
    }
}

inline DataFormat to_a_data_format(DataFormat df)
{
    switch (df)
    {
        case DataFormat::Float16_b: return DataFormat::Float16;
        case DataFormat::Bfp8_b: return DataFormat::Bfp8;
        case DataFormat::Bfp4_b: return DataFormat::Bfp4;
        case DataFormat::Bfp2_b: return DataFormat::Bfp2;
        default: return df;
    }
}

inline DataFormat to_b_data_format(DataFormat df)
{
    switch (df)
    {
        case DataFormat::Float16: return DataFormat::Float16_b;
        case DataFormat::Bfp8: return DataFormat::Bfp8_b;
        case DataFormat::Bfp4: return DataFormat::Bfp4_b;
        case DataFormat::Bfp2: return DataFormat::Bfp2_b;
        default: return df;
    }
}

inline int get_precision_bits(DataFormat df)
{
    switch (df)
    {
        case DataFormat::Float32: return 32;
        case DataFormat::Float16: return 16;
        case DataFormat::Bfp8: return 8;
        case DataFormat::Bfp4: return 4;
        case DataFormat::Bfp2: return 2;
        case DataFormat::Float16_b: return 16;
        case DataFormat::Bfp8_b: return 8;
        case DataFormat::Bfp4_b: return 4;
        case DataFormat::Bfp2_b: return 2;
        default: return 0;
    }
}

inline DataFormat preserve_lower_precision_cast(DataFormat from, DataFormat to)
{
    if (get_precision_bits(from) >= get_precision_bits(to))
        return to;

    return is_b_data_format(to) ? to_b_data_format(from) : to_a_data_format(from);
}

std::ostream &operator<<(std::ostream &os, DataFormat const &df);
std::ostream &operator<<(std::ostream &os, MathFidelity const &mf);

struct PytorchTensorDesc
{
    const void* ptr;
    std::uint32_t itemsize;
    DataFormat format;
    std::array<std::uint32_t, 4> shape;   // outer-most dimension first
    std::array<std::uint32_t, 4> strides; // outer-most dimension first, in bytes
};

}  // namespace tt
