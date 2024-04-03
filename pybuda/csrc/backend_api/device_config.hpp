// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arch_type.hpp"
#include "utils/assert.hpp"
#include "utils/env.hpp"
#include "utils/logger.hpp"

namespace tt
{
struct DeviceGrid
{
    int r;
    int c;

    DeviceGrid(int r, int c) : r(r), c(c) {}
    DeviceGrid(std::pair<int, int> p) : r(p.first), c(p.second) {}
};

struct CoreCoord
{
    int x;
    int y;

    CoreCoord(int x, int y) : x(x), y(y) {}
};

struct EthCoord
{
    int x;
    int y;
    int rack;
    int shelf;

    EthCoord(int x, int y, int rack, int shelf) : x(x), y(y), rack(rack), shelf(shelf) {}
    bool operator<(const struct EthCoord& rhs) const
    {
        return (shelf < rhs.shelf) || (shelf == rhs.shelf && rack < rhs.rack) ||
               (shelf == rhs.shelf && rack == rhs.rack && y < rhs.y) ||
               (shelf == rhs.shelf && rack == rhs.rack && y == rhs.y && x < rhs.x);
    }
};

struct DeviceConfig
{
    std::string arch_name;
    ARCH arch;
    std::string device_yaml;
    std::string cluster_config_yaml;
    std::string runtime_params_yaml;
    std::string backend_type;
    bool store_backend_db_to_yaml;
    DeviceGrid grid_size;
    std::vector<std::uint32_t> chip_ids;
    std::vector<int> chips_with_mmio;
    std::unordered_map<std::uint32_t, EthCoord> chip_locations;
    std::map<EthCoord, std::uint32_t> chip_coord_to_chip_id;
    std::map<std::uint32_t, std::vector<std::uint32_t>> shelves_chip_ids;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t>>> ethernet_connections;
    // chip_to_chip_connections[chip_a][chip_b] = std::set< channel_a > where channel_a is channel on chip_a to
    // chip_b
    std::map<std::uint32_t, std::map<std::uint32_t, std::set<std::uint32_t>>> chip_to_chip_connections;
    std::vector<std::uint32_t> galaxy_shelves;
    static const std::uint32_t GALAXY_GRID_X = 4;
    static const std::uint32_t GALAXY_GRID_Y = 8;
    static const std::uint32_t GALAXY_CHIP_CONNECTIONS = 4;
    
    // Temporal constants used for blackhole onboarding
    static const std::string wormhole_b0_string;

    std::unordered_map<std::string, std::string> cached_system_level_params;

    DeviceConfig(
        std::string arch_name,
        std::string device_yaml,
        std::string cluster_config_yaml,
        std::string runtime_params_yaml,
        std::string backend_type,
        bool store_backend_db_to_yaml,
        bool skip_backend_queries = true) :
        arch_name(arch_name),
        device_yaml(device_yaml),
        cluster_config_yaml(cluster_config_yaml),
        runtime_params_yaml(runtime_params_yaml),
        backend_type(backend_type),
        store_backend_db_to_yaml(store_backend_db_to_yaml),
        grid_size(get<DeviceGrid>("t6-grid_size", false))
    {
        arch = to_arch_type(arch_name);

        // Constructor - used only by unittesting.
        if (skip_backend_queries)
            return;

        // Get backend related parameters
        if (this->is_wormhole_b0())
        {
            // Load and cache system-level params if needed
            if (this->backend_type == "silicon")
                this->load_system_level_params();

            // If cluster descriptor is not provided, get it from backend
            if (this->cluster_config_yaml.empty() && this->backend_type == "silicon")
            {
                try
                {
                    this->cluster_config_yaml = this->get_cluster_descriptor();
                }
                catch (const std::exception& e)
                {
                    log_fatal(
                        "Failed to get cluster descriptor from backend. User is attempting to get cluster "
                        "descriptor "
                        "from a machine without a WH-silicon device or not providing runtime_params.yaml for "
                        "offline "
                        "compilation.");
                }
            }

            if (!this->cluster_config_yaml.empty())
            {
                //  Set multichip configs if Silicon generated cluster config or user provided one
                this->chips_with_mmio = this->get_chips_with_mmio();
                this->chip_locations = this->get_chip_locations();
                for (auto& chip_id_to_coord : chip_locations)
                {
                    std::uint32_t chip_id = chip_id_to_coord.first;
                    const EthCoord& coord = chip_id_to_coord.second;
                    TT_ASSERT(chip_coord_to_chip_id.find(coord) == chip_coord_to_chip_id.end());
                    chip_coord_to_chip_id[coord] = chip_id;
                    shelves_chip_ids[coord.shelf].push_back(chip_id);
                }
                this->ethernet_connections = this->get_ethernet_connections();
                populate_chip_to_chip_connections();
                find_galaxy_shelves();
            }
            else
            {
                // Place MMIO access on chip 0 for non silicon runs
                this->chips_with_mmio = {0};
            }
        }
        else
        {
            // Place MMIO access on chip 0 for non WH runs
            this->chips_with_mmio = {0};
        }
        this->grid_size = get<DeviceGrid>("t6-grid_size", false);
        TT_ASSERT(this->grid_size.r != 0 && this->grid_size.c != 0);
    }

    DeviceConfig(
        std::string arch_name,
        std::string device_yaml,
        std::string cluster_config_yaml,
        std::string runtime_params_yaml,
        std::string backend_type,
        bool store_backend_db_to_yaml,
        const std::vector<std::uint32_t>& chip_ids) :
        DeviceConfig(
            arch_name,
            device_yaml,
            cluster_config_yaml,
            runtime_params_yaml,
            backend_type,
            store_backend_db_to_yaml,
            false)
    {
        this->chip_ids = chip_ids;
    }

    DeviceConfig(
        std::string arch_name,
        std::string device_yaml,
        std::string cluster_config_yaml,
        std::string runtime_params_yaml,
        std::string backend_type,
        bool store_backend_db_to_yaml,
        const std::vector<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>>& chip_ids) :
        DeviceConfig(
            arch_name,
            device_yaml,
            cluster_config_yaml,
            runtime_params_yaml,
            backend_type,
            store_backend_db_to_yaml,
            false)
    {
        for (auto& chip_coord : chip_ids)
        {
            EthCoord eth_coord(
                std::get<0>(chip_coord), std::get<1>(chip_coord), std::get<2>(chip_coord), std::get<3>(chip_coord));
            if (this->chip_coord_to_chip_id.find(eth_coord) == this->chip_coord_to_chip_id.end())
            {
                log_fatal(
                    "Chip with coordinates ({}, {}, {}, {}) cannot be found",
                    eth_coord.x,
                    eth_coord.y,
                    eth_coord.rack,
                    eth_coord.shelf);
            }

            this->chip_ids.push_back(this->chip_coord_to_chip_id.at(eth_coord));
        }
    }

    // Get if the device is a blackhole
    inline bool is_blackhole() const { return arch == ARCH::BLACKHOLE; }

    // Get if the device is a wormhole_b0
    // During the onboarding process of the blackhole architecture,
    // we temporarily treat it as equivalent to the Wormhole_b0 architecture.
    inline bool is_wormhole_b0() const { return arch == ARCH::WORMHOLE_B0 || is_blackhole(); }
    
    // Get if the device is a grayskull
    inline bool is_grayskull() const { return arch == ARCH::GRAYSKULL; }

    // This is a temporary workaround to handle the estimation calculation for the blackhole architecture.
    // Since there is currently no implemented estimate for blackhole, we are reusing the estimate for wormhole_b0.
    const std::string& get_arch_name_for_perf_estimates() const 
    {
        if (is_blackhole())
            return wormhole_b0_string;
        return arch_name;
    }

    template <typename T>
    T get(std::string const &param, const bool system_level_command) const;
    void load_system_level_params();
    std::vector<std::uint32_t> get_harvested_cfg() const;

    std::size_t get_dst_size() const { return get<std::size_t>("t6-dst_size", false); }
    std::size_t get_clock_freq() const
    {
        return 1000000000;  // tenstorrent/budabackend#1912
    }
    std::uint32_t get_host_memory_num_channels() const
    {
        return get<std::uint32_t>("sysmem-host_region_num_channels", false);
    }
    std::uint32_t get_host_memory_channel_start_address() const
    {
        return get<std::uint32_t>("sysmem-host_region_range_start", false);
    }
    std::uint32_t get_host_memory_channel_size(uint8_t mem_channel) const
    {
        std::string suffix = "_chan" + std::to_string(mem_channel);
        return get<std::uint32_t>("sysmem-host_region_range_size" + suffix, false);
    }

    std::uint32_t get_host_mmio_range_offset() const { return get<std::uint32_t>("dram-host_mmio_range_start", false); }
    std::uint32_t get_host_mmio_range_size() const { return get<std::uint32_t>("dram-host_mmio_range_size", false); }
    std::uint32_t get_p2p_offset() const { return get<std::uint32_t>("dram-p2p_range_start", false); }
    std::uint32_t get_p2p_size() const { return get<std::uint32_t>("dram-p2p_range_size", false); }
    std::size_t get_l1_size() const { return get<std::size_t>("t6-l1_size", false); }
    std::size_t get_overlay_blob_extra_size() const
    {
        static size_t overlay_blob_extra_size = env_as<size_t>("TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE", 0);
        return overlay_blob_extra_size;
    }
    std::size_t get_l1_backend_reserved_size() const
    {
        // BBE will account for extra blob size (TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE) in the reserved size
        //
        auto reserved_size = get<std::size_t>("t6-l1_backend_reserved_size", false);
        static auto extra_l1_margin = env_as<int>("PYBUDA_EXTRA_L1_MARGIN");
        if (reserved_size < (std::uint32_t)extra_l1_margin)
            return 0;

        return reserved_size - extra_l1_margin;
    }
    std::size_t get_l1_usable_size() const { return get_l1_size() - get_l1_backend_reserved_size(); }
    std::size_t get_l1_dram_io_backend_reserved_size() const
    {
        // Get this number from DB query:
        // tenstorrent/budabackend#1979
        return 100 * 1024;
    }
    std::size_t get_noc_bandwidth_bytes_per_cycle() const
    {
        return 32;  // tenstorrent/budabackend#1912
    }
    std::uint32_t get_dram_num_channels() const { return get<std::uint32_t>("dram-num_channels", false); }
    std::uint32_t get_dram_num_subchannels() const
    {
        // TODO - get from backend, but backend needs to add it
        return is_grayskull() ? 1 : 3;
    }
    std::uint32_t get_dram_channel_capacity() const { return get<std::uint32_t>("dram-channel_capacity", false); }
    std::size_t get_dram_bandwidth_per_block_theoretical() const
    {
        return get<std::size_t>("dram-bandwidth_per_block_theoretical", false);
    }
    std::size_t get_dram_bandwidth_per_block_measured() const
    {
        return get<std::size_t>("dram-bandwidth_per_block_measured", false);
    }
    std::size_t get_dram_bandwidth_bytes_per_cycle() const
    {
        return get_dram_bandwidth_per_block_measured() / get_clock_freq();
    }
    std::uint32_t get_dram_backend_reserved(std::uint32_t channel) const
    {
        return get<std::uint32_t>("dram-backend_reserved_chan" + std::to_string(channel), false);
    }
    std::uint32_t get_dram_backend_reserved_max() const
    {
        return get<std::uint32_t>("dram-backend_reserved_max", false);
    }
    CoreCoord get_dram_core_coord(std::uint32_t channel, std::uint32_t subchannel) const
    {
        if (is_grayskull())
        {
            return get<CoreCoord>("dram-core_xy_chan" + std::to_string(channel), false);
        }
        return get<CoreCoord>(
            "dram-core_xy_chan" + std::to_string(channel) + "_subchan" + std::to_string(subchannel), false);
    }
    std::string get_cluster_descriptor() const { return get<std::string>("device-cluster_descriptor", true); }
    std::vector<int> get_chips_with_mmio() const { return get<std::vector<int>>("device-chips_with_mmio", true); }
    std::uint32_t get_number_of_chips() const { return get<std::uint32_t>("device-number_of_chips", true); }
    std::unordered_map<std::uint32_t, EthCoord> get_chip_locations() const
    {
        return get<std::unordered_map<std::uint32_t, EthCoord>>("device-chip_locations", true);
    }
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t>>>
    get_ethernet_connections() const
    {
        return get<std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t>>>>(
            "device-ethernet_connections", true);
    }

    void populate_chip_to_chip_connections()
    {
        for (const auto& [chip_a, channels] : ethernet_connections)
        {
            for (const auto& [channel_a, chip_b_channel] : channels)
            {
                std::uint32_t chip_b = std::get<0>(chip_b_channel);
                std::uint32_t channel_b = std::get<1>(chip_b_channel);
                chip_to_chip_connections[chip_a][chip_b].insert(channel_a);
                chip_to_chip_connections[chip_b][chip_a].insert(channel_b);
            }
        }
    }

    // compute which shelves have 4x8 chip coordinates and verify the galaxy config
    // we can have Galaxy, Nebula+Galaxy, Nebula+Galaxy+Galaxy, etc.
    void find_galaxy_shelves()
    {
        for (auto it = shelves_chip_ids.begin(); it != shelves_chip_ids.end(); it++)
        {
            if (it->second.size() == (GALAXY_GRID_X * GALAXY_GRID_Y))
            {
                galaxy_shelves.push_back(it->first);
            }
        }

        // verification:
        // 1. galaxy grid must be 8x4
        // 2. there must be exactly 4 eth connections between each galaxy chip
        for (auto shelf_id : galaxy_shelves)
        {
            std::map<uint32_t, std::map<uint32_t, uint32_t>> xy_coord_to_chip_id;
            for (auto it = chip_coord_to_chip_id.begin(); it != chip_coord_to_chip_id.end(); it++)
            {
                EthCoord coord = it->first;
                std::uint32_t chip_id = it->second;
                if (coord.shelf == (int)shelf_id)
                {
                    xy_coord_to_chip_id[coord.x][coord.y] = chip_id;
                }
            }
            for (unsigned x = 0; x < GALAXY_GRID_X; x++)
            {
                TT_ASSERT(xy_coord_to_chip_id.find(x) != xy_coord_to_chip_id.end());
                for (unsigned y = 0; y < GALAXY_GRID_Y; y++)
                {
                    TT_ASSERT(xy_coord_to_chip_id[x].find(y) != xy_coord_to_chip_id[x].end());
                }
            }

            for (unsigned x = 0; x < GALAXY_GRID_X; x++)
            {
                for (unsigned y = 0; y < GALAXY_GRID_Y; y++)
                {
                    std::uint32_t src_chip = xy_coord_to_chip_id[x][y];
                    if (x < (GALAXY_GRID_X - 1))
                    {
                        std::uint32_t right_chip = xy_coord_to_chip_id[x + 1][y];
                        TT_ASSERT(
                            env_as<bool>("SKIP_GALAXY_MODULES_LINK_CHECK") ||
                            chip_to_chip_connections[src_chip][right_chip].size() == GALAXY_CHIP_CONNECTIONS);
                    }
                    if (y < (GALAXY_GRID_Y - 1))
                    {
                        std::uint32_t down_chip = xy_coord_to_chip_id[x][y + 1];
                        TT_ASSERT(
                            env_as<bool>("SKIP_GALAXY_MODULES_LINK_CHECK") ||
                            chip_to_chip_connections[src_chip][down_chip].size() == GALAXY_CHIP_CONNECTIONS);
                    }
                }
            }
        }
    }

    DeviceGrid get_harvested_nebula_galaxy_grid() const { return DeviceGrid(8, 8); }

    std::vector<std::uint32_t> get_harvested_rows(std::uint32_t mask) const
    {
        std::vector<std::uint32_t> ret;
        for (std::uint32_t i = 0; i < 32; i++)
        {
            if (mask & (1 << i))
                ret.push_back(i);
        }
        return ret;
    }

    bool supports_fp32_accumulation() const { return not is_grayskull(); }
    bool supports_stochastic_rounding() const { return is_wormhole_b0(); }
    std::unordered_map<std::uint32_t, std::set<std::uint32_t>> get_chip_connections() const
    {
        // Assume bidirectional connection for now, and no weights assigned to the connections
        // <chip_a <channel_a, <chip_b, channel_b>>> eth_connections
        std::unordered_map<std::uint32_t, std::set<std::uint32_t>> chip_connections;
        for (const auto& [chip_a, channels] : ethernet_connections)
        {
            for (const auto& [channel_a, chip_b_channel] : channels)
            {
                if (chip_connections.find(chip_a) == chip_connections.end())
                {
                    chip_connections.insert({chip_a, {std::get<0>(chip_b_channel)}});
                }
                else
                {
                    chip_connections.at(chip_a).insert(std::get<0>(chip_b_channel));
                }
            }
        }
        return chip_connections;
    }
};

inline std::ostream& operator<<(std::ostream& os, DeviceConfig const& device_config)
{
    auto indent = "  ";
    os << "DeviceConfig {" << std::endl;
    os << indent << ".arch_name = " << device_config.arch_name << "," << std::endl;
    os << indent << ".device_yaml = " << device_config.device_yaml << "," << std::endl;
    os << indent << ".cluster_config_yaml = " << device_config.cluster_config_yaml << "," << std::endl;
    os << indent << ".runtime_params_yaml = " << device_config.runtime_params_yaml << "," << std::endl;
    os << indent << ".grid_size = {" << device_config.grid_size.r << ", " << device_config.grid_size.c << "}"
       << "," << std::endl;
    os << indent << ".get_dst_size = " << device_config.get_dst_size() << "," << std::endl;
    os << indent << ".get_l1_size = " << device_config.get_l1_size() << "," << std::endl;
    os << indent << ".get_l1_backend_reserved_size = " << device_config.get_l1_backend_reserved_size() << ","
       << std::endl;
    os << indent << ".get_l1_usable_size = " << device_config.get_l1_usable_size() << "," << std::endl;
    os << indent << ".get_dram_num_channels = " << device_config.get_dram_num_channels() << "," << std::endl;
    os << indent << ".get_dram_channel_capacity = " << device_config.get_dram_channel_capacity() << "," << std::endl;
    os << indent << ".supports_fp32_accumulation = " << device_config.supports_fp32_accumulation() << "," << std::endl;
    os << indent << ".supports_stochastic_rounding = " << device_config.supports_stochastic_rounding() << ","
       << std::endl;
    os << indent << ".chips_with_mmio = {";
    for (int chip_id : device_config.chips_with_mmio) os << chip_id << ", ";
    os << "}"
       << "," << std::endl;
    os << "}";
    return os;
}

}  // namespace tt
