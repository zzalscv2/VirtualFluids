//=======================================================================================
// ____          ____    __    ______     __________   __      __       __        __
// \    \       |    |  |  |  |   _   \  |___    ___| |  |    |  |     /  \      |  |
//  \    \      |    |  |  |  |  |_)   |     |  |     |  |    |  |    /    \     |  |
//   \    \     |    |  |  |  |   _   /      |  |     |  |    |  |   /  /\  \    |  |
//    \    \    |    |  |  |  |  | \  \      |  |     |   \__/   |  /  ____  \   |  |____
//     \    \   |    |  |__|  |__|  \__\     |__|      \________/  /__/    \__\  |_______|
//      \    \  |    |   ________________________________________________________________
//       \    \ |    |  |  ______________________________________________________________|
//        \    \|    |  |  |         __          __     __     __     ______      _______
//         \         |  |  |_____   |  |        |  |   |  |   |  |   |   _  \    /  _____)
//          \        |  |   _____|  |  |        |  |   |  |   |  |   |  | \  \   \_______
//           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  |
//            \ _____|  |__|        |________|   \_______/    |__|   |______/    (_______/
//
//  This file is part of VirtualFluids. VirtualFluids is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  VirtualFluids is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
//  for more details.
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//  SPDX-FileCopyrightText: Copyright © VirtualFluids Project contributors, see AUTHORS.md in root folder
//
//! \addtogroup gpu_grid_tests grid
//! \ingroup gpu_GridGenerator_tests GridGenerator
//! \{
//=======================================================================================
#include "grid/NodeValues.h"
#include <gmock/gmock.h>

#include <gpu/GridGenerator/geometries/BoundingBox/BoundingBox.h>
#include <gpu/GridGenerator/grid/GridBuilder/CommunicationNodeFinder.h>
#include <gpu/GridGenerator/grid/GridImp.h>
#include <gpu/GridGenerator/utilities/communication.h>

namespace
{

class GridImpStub : public GridImp
{
public:
    GridImpStub(SPtr<Object> object, real startX, real startY, real startZ, real endX, real endY, real endZ, real delta,
                uint level)
        : GridImp(std::move(object), startX, startY, startZ, endX, endY, endZ, delta,
                  DistributionHelper::getDistribution("D3Q27"), level)
    {
    }

    char getFieldEntry(uint /*matrixIndex*/) const override
    {
        return vf::gpu::FLUID;
    }
};

class CommunicationNodeFinderTest : public testing::TestWithParam<CommunicationDirections::CommunicationDirection>
{
public:
    real minCoordinate = 10;
    real maxCoordinate = 16;
    real delta = 2;
    uint numberOfLevels = 1;
    uint level = 0;
    bool doShift = false;

    SPtr<Grid> grid = std::make_shared<GridImpStub>(nullptr, minCoordinate, minCoordinate, minCoordinate, maxCoordinate,
                                                    maxCoordinate, maxCoordinate, delta, level);
    const std::vector<SPtr<Grid>> grids = { grid };

    BoundingBox subDomainBox =
        BoundingBox(minCoordinate + 0.5 * delta, maxCoordinate - 0.5 * delta, minCoordinate + 0.5 * delta,
                    maxCoordinate - 0.5 * delta, minCoordinate + 0.5 * delta, maxCoordinate - 0.5 * delta);
    CommunicationNodeFinder sut = CommunicationNodeFinder(numberOfLevels);
};

TEST_P(CommunicationNodeFinderTest, findCommunicationIndices_worksForDirection)
{
    auto direction = GetParam();

    std::vector<uint> receiveExpected;
    std::vector<uint> sendExpected;
    switch (direction) {
        case CommunicationDirections::MX: {
            sendExpected = { 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61 };
            receiveExpected = { 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60 };
            break;
        }
        case CommunicationDirections::PX: {
            sendExpected = { 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62 };
            receiveExpected = { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 };
            break;
        }
        case CommunicationDirections::MY: {
            sendExpected = { 4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55 };
            receiveExpected = { 0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51 };
            break;
        }
        case CommunicationDirections::PY: {
            sendExpected = { 8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59 };
            receiveExpected = { 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63 };
            break;
        }
        case CommunicationDirections::MZ: {
            sendExpected = { 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
            receiveExpected = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
            break;
        }
        case CommunicationDirections::PZ: {
            sendExpected = { 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 };
            receiveExpected = { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 };
            break;
        }
        default: {
            EXPECT_TRUE(false) << "unknown direction: " << direction;
            return;
        }
    }

    sut.findCommunicationIndices(direction, subDomainBox, doShift, grids);

    EXPECT_THAT(sut.getCommunicationIndices().at(level).at(direction).sendIndices, testing::Eq(sendExpected));
    EXPECT_THAT(sut.getCommunicationIndices().at(level).at(direction).receiveIndices, testing::Eq(receiveExpected));
}

INSTANTIATE_TEST_SUITE_P(CommunicationNodeFinder_DirectionalTests, CommunicationNodeFinderTest,
                         testing::Values(CommunicationDirections::MX, CommunicationDirections::PX,
                                         CommunicationDirections::MY, CommunicationDirections::PY,
                                         CommunicationDirections::MZ, CommunicationDirections::PZ));

} // namespace
