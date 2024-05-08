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
//! \addtogroup gpu_grid grid
//! \ingroup gpu_GridGenerator GridGenerator
//! \{
//! \author Soeren Peters, Stephan Lenz, Martin Schoenherr, Anna Wellmann
//=======================================================================================
#ifndef COMMUNICATION_NODE_FINDER_H
#define COMMUNICATION_NODE_FINDER_H

#include <vector>
#include <array>

#include <basics/DataTypes.h>
#include <basics/PointerDefinitions.h>

class BoundingBox;
class Grid;

struct CommunicationIndicesOfDirection
{
    std::vector<uint> sendIndices;
    std::vector<uint> receiveIndices;
};

// There are six directions per grid level. We store the CommunicationIndices for all six directions in this array.
using CommunicationIndicesForLevel = std::array<CommunicationIndicesOfDirection, 6>;

class CommunicationNodeFinder
{
public:
    CommunicationNodeFinder(uint numberOfLevels);

    void findCommunicationIndices(int direction, SPtr<BoundingBox> subDomainBox, bool doShift, const Grid* grid);

    uint getNumberOfSendNodes(uint level, int direction) const;
    uint getNumberOfReceiveNodes(uint level, int direction) const;
    uint getSendIndex(uint level, int direction, uint index) const;
    uint getReceiveIndex(uint level, int direction, uint index) const;
    void getSendIndices(int* sendIndices, int direction, int level, const Grid* grid) const;
    void getReceiveIndices(int* receiveIndices, int direction, int level, const Grid* grid) const;

    uint getNumberOfSendIndices(uint level, int direction) const;
    uint getNumberOfReceiveIndices(uint level, int direction) const;

    // // needed for CUDA Streams
    // virtual void findFluidNodeIndices(bool onlyBulk);
    // virtual uint getNumberOfFluidNodes() const;
    // virtual void getFluidNodeIndices(uint* fluidNodeIndices) const;
    // virtual void findFluidNodeIndicesBorder();
    // virtual uint getNumberOfFluidNodesBorder() const;
    // virtual void getFluidNodeIndicesBorder(uint* fluidNodeIndicesBorder) const;

    const std::vector<CommunicationIndicesForLevel>& getCommunicationIndices() const;

private:
    void findCommunicationIndex(uint index, real coordinate, real limit, int direction, real delta);
    void findCommunicationIndexForLevel( uint index, real coordinate, real limit, int direction, real delta, CommunicationIndicesForLevel& communicationIndicesForLevel);

    bool isSendNode(uint level, int index) const;
    bool isReceiveNode(uint level, int index) const;

    void repairCommunicationIndicesForLevel(int direction, CommunicationIndicesForLevel& communicationIndicesForLevel);

    std::vector<CommunicationIndicesForLevel> communicationIndices;
};

#endif

//! \}
