// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placement_printer.hpp"

#include "utils/assert.hpp"

#include <algorithm>


namespace tt::utils {

int getNumDigits(int num)
{
    int digits = 0;

    do
    {
        digits++;
        num /= 10;
    } while (num);

    return digits;
}

std::string spacess(int num) {
    std::string spaces_string;

    for (int i = 0; i < num; i++)
    {
        spaces_string.append(" ");
    }

    return spaces_string;
}

void PlacementPrinter::fillRectangle(uint epoch, uint chip, uint top, uint left, uint bottom, uint right, int val)
{
    // TODO: Validate values
    for (uint h = top; h < bottom; h++)
    {
        // TODO: Validate values
        for (uint w = left; w < right; w++)
        {
            // TODO: Validate values
            int idx =
                epoch * (this->chipCount * this->height * this->width) +
                chip * (this->height * this->width) +
                h * this->width +
                w;

            TT_ASSERT(this->linearMap[idx] == 0, "Overwrite happened! Writing " + std::to_string(val) + " over " +
                std::to_string(this->linearMap[idx] - 1));

            // Save values + 1, in order to keep 0 as a viable op id, but then print a non-zero char for empty cores
            this->linearMap[idx] = val + 1;
        }
    }
    return;
}

std::string PlacementPrinter::generatePlacementString()
{
    int maxDigitsSize = 0;
    for (size_t i = 0; i < this->linearMap.size(); i++)
    {
        maxDigitsSize = std::max(maxDigitsSize, getNumDigits(linearMap[i]));
    }

    std::stringstream out;

    for (uint e = 0; e < this->totalEpochsCount; e++)
    {
        out << "Epoch " + std::to_string(e) + ":\n";
        int epochType = 0;
        while (int(e) - std::accumulate(this->epochsPerEpochType.begin(), this->epochsPerEpochType.begin() + epochType + 1, 0) >= 0)
        {
            epochType++;
        }
        std::string epochTypeStr;
        switch (epochType)
        {
            case 0:
                epochTypeStr = "fwd";
                break;
            case 1:
                epochTypeStr = "bwd";
                break;
            case 2:
                epochTypeStr = "opt";
                break;
            default:
                TT_ASSERT(false, "Invalid epoch type value: " + std::to_string(epochType));
        }
        out << "Epoch type: " + epochTypeStr + "\n";
        for (uint h = 0; h < this->height; h++)
        {
            for (uint c = 0; c < this->chipCount; c++)
            {
                for (uint w = 0; w < this->width; w++)
                {
                    int idx =
                        e * (this->chipCount * this->height * this->width) +
                        c * (this->height * this->width) +
                        h * this->width +
                        w;
                    int val = linearMap[idx] - 1;  // Reduce by 1 to get the original value

                    // If new chip started, add buffer inbetween chips
                    if (w == 0 and c > 0)
                    {
                        out << spacess(maxDigitsSize + 2);
                    }

                    out << spacess(1 + (maxDigitsSize - getNumDigits(val)));

                    if (val == -1)
                    {
                        out << "*";
                    }
                    else
                    {
                        out << std::to_string(val);
                    }
                }
            }

            out << "\n";
        }
        out << "\n\n";
    }

    return out.str();
}

}  // namespace tt::utils