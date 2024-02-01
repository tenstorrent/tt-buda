// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "utils/assert.hpp"

#include <numeric>
#include <string>
#include <vector>


namespace tt::utils {

class PlacementPrinter {
public:
    enum class DeviceType {
        Grayskull,
        Wormhole
    };

    PlacementPrinter(DeviceType device, uint nodeEpochTypesCnt, std::vector<uint> epochsPerEpochType, uint chipCount)
    {
        this->device = device;
        this->nodeEpochTypesCnt = nodeEpochTypesCnt;  // Usually size=1 for just fwd or size=3 for fwd/bwd/opt
        this->epochsPerEpochType = epochsPerEpochType;
        this->chipCount = chipCount;  // How many chips in total on host

        switch (device)
        {
            case DeviceType::Grayskull:
                this->height = 10;
                this->width = 12;
                break;
            case DeviceType::Wormhole:
                this->height = 10;
                this->width = 8;
                break;
            default:
                TT_ASSERT(false, "Unsupported DeviceType");
        }

        this->totalEpochsCount = std::accumulate(epochsPerEpochType.begin(), epochsPerEpochType.end(), 0);

        this->linearMap = std::vector<int>(totalEpochsCount * chipCount * this->height * this->width);
    };

    PlacementPrinter() = delete;

    void fillRectangle(uint epoch, uint chip, uint top, uint left, uint bottom, uint right, int val);
    std::string generatePlacementString();

private:
    DeviceType device;
    uint nodeEpochTypesCnt;
    std::vector<uint> epochsPerEpochType;
    uint chipCount;

    uint totalEpochsCount;  // stored for ease of use

    uint height;
    uint width;

    // [total_epochs, chip, h, w] where total_epochs = sum(x for x in epochsPerEpochType)
    std::vector<int> linearMap;
};

}  // namespace utils
