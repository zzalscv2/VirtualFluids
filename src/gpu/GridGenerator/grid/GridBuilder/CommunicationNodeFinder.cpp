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

#include "CommunicationNodeFinder.h"
#include "grid/Grid.h"
#include "grid/NodeValues.h"
#include "geometries/BoundingBox/BoundingBox.h"
#include "utilities/communication.h"

using namespace vf::gpu;

CommunicationNodeFinder::CommunicationNodeFinder(uint numberOfLevels)
{
    for (uint i = 0; i < numberOfLevels; i++)
        communicationIndices.emplace_back();
}

void CommunicationNodeFinder::findCommunicationIndices(int direction, SPtr<BoundingBox> subDomainBox, bool doShift, const Grid* grid)
{
    for( uint index = 0; index < grid->getSize(); index++ ){

        int shiftedIndex = doShift ? grid->getShiftedCommunicationIndex(index, direction) : index;
        
        const char fieldEntry = grid->getFieldEntry(shiftedIndex);
        if( fieldEntry == INVALID_OUT_OF_GRID ||
            fieldEntry == INVALID_SOLID ||
            fieldEntry == INVALID_COARSE_UNDER_FINE ||
            fieldEntry == STOPPER_OUT_OF_GRID ||
            fieldEntry == STOPPER_COARSE_UNDER_FINE ||
            fieldEntry == STOPPER_OUT_OF_GRID_BOUNDARY ||
            fieldEntry == STOPPER_SOLID ) continue;

        real x, y, z;
        grid->transIndexToCoords(shiftedIndex, x, y, z);

        switch(direction)
        {
            case CommunicationDirections::MX: findCommunicationIndex( shiftedIndex, x, subDomainBox->minX, direction, grid->getDelta()); break;
            case CommunicationDirections::PX: findCommunicationIndex( shiftedIndex, x, subDomainBox->maxX, direction, grid->getDelta()); break;
            case CommunicationDirections::MY: findCommunicationIndex( shiftedIndex, y, subDomainBox->minY, direction, grid->getDelta()); break;
            case CommunicationDirections::PY: findCommunicationIndex( shiftedIndex, y, subDomainBox->maxY, direction, grid->getDelta()); break;
            case CommunicationDirections::MZ: findCommunicationIndex( shiftedIndex, z, subDomainBox->minZ, direction, grid->getDelta()); break;
            case CommunicationDirections::PZ: findCommunicationIndex( shiftedIndex, z, subDomainBox->maxZ, direction, grid->getDelta()); break;
        }
    }
}

void CommunicationNodeFinder::findCommunicationIndex(uint index, real coordinate, real limit, int direction, real delta)
{
    for (auto& communicationIndicesForLevel : communicationIndices)
        findCommunicationIndexForLevel(index, coordinate, limit, direction, delta, communicationIndicesForLevel);
}

void CommunicationNodeFinder::findCommunicationIndexForLevel(uint index, real coordinate, real limit, int direction,
                                                             real delta,
                                                             CommunicationIndicesForLevel& communicationIndicesForLevel)
{
    // negative direction get a negative sign
    real s = (direction % 2 == 0) ? (-1.0) : (1.0);

    if (std::abs(coordinate - (limit + s * 0.5 * delta)) < 0.1 * delta)
        communicationIndicesForLevel[direction].receiveIndices.push_back(index);

    if (std::abs(coordinate - (limit - s * 0.5 * delta)) < 0.1 * delta)
        communicationIndicesForLevel[direction].sendIndices.push_back(index);
}

bool CommunicationNodeFinder::isSendNode(uint level, int index) const
{
    bool isSendNode = false;
    for (const CommunicationIndicesOfDirection& communicationIndicesForDirection: this->communicationIndices[level])
        if (std::find(communicationIndicesForDirection.sendIndices.begin(),
                      communicationIndicesForDirection.sendIndices.end(), index) != communicationIndicesForDirection.sendIndices.end())
            isSendNode = true;
    return isSendNode;
}

bool CommunicationNodeFinder::isReceiveNode(uint level, int index) const
{
    bool isReceiveNode = false;
    for (const CommunicationIndicesOfDirection& communicationIndicesForDirection: this->communicationIndices[level])
        if (std::find(communicationIndicesForDirection.receiveIndices.begin(),
                      communicationIndicesForDirection.receiveIndices.end(),
                      index) != communicationIndicesForDirection.receiveIndices.end())
            isReceiveNode = true;
    return isReceiveNode;
}

void CommunicationNodeFinder::getSendIndices(int * sendIndices, int direction, int level, const Grid* grid) const
{
    for( uint i = 0; i < getNumberOfSendIndices(direction, level); i++ )
    {
        sendIndices[i] = grid->getSparseIndex(getSendIndex(level, direction, i)) + 1;
    }
}

void CommunicationNodeFinder::getReceiveIndices(int * receiveIndices, int direction, int level, const Grid* grid) const
{
    for( uint i = 0; i < getNumberOfReceiveIndices(direction, level); i++ )
    {
        receiveIndices[i] = grid->getSparseIndex(getReceiveIndex(level, direction, i) ) + 1;
    }
}

uint CommunicationNodeFinder::getNumberOfSendIndices(uint level, int direction) const
{
    return getNumberOfSendNodes(level, direction);
}

uint CommunicationNodeFinder::getNumberOfReceiveIndices(uint level, int direction) const
{
    return getNumberOfReceiveNodes(level, direction);
}

uint CommunicationNodeFinder::getNumberOfSendNodes(uint level, int direction) const
{
    return (uint)communicationIndices[level][direction].sendIndices.size();
}

uint CommunicationNodeFinder::getNumberOfReceiveNodes(uint level, int direction) const
{
    return (uint)communicationIndices[level][direction].receiveIndices.size();
}

uint CommunicationNodeFinder::getSendIndex(uint level, int direction, uint index) const
{
    return communicationIndices[level][direction].sendIndices[index];
}

uint CommunicationNodeFinder::getReceiveIndex(uint level, int direction, uint index) const
{
    return communicationIndices[level][direction].receiveIndices[index];
}

void CommunicationNodeFinder::repairCommunicationIndicesForLevel(int direction,
                                                                 CommunicationIndicesForLevel& communicationIndicesForLevel)
{
    communicationIndicesForLevel[direction].sendIndices.insert(
        communicationIndicesForLevel[direction].sendIndices.end(),
        communicationIndicesForLevel[direction + 1].sendIndices.begin(),
        communicationIndicesForLevel[direction + 1].sendIndices.end());

    communicationIndicesForLevel[direction + 1].receiveIndices.insert(
        communicationIndicesForLevel[direction + 1].receiveIndices.end(),
        communicationIndicesForLevel[direction].receiveIndices.begin(),
        communicationIndicesForLevel[direction].receiveIndices.end());

    communicationIndicesForLevel[direction].receiveIndices = communicationIndicesForLevel[direction + 1].receiveIndices;

    VF_LOG_INFO("size send {}", (int)communicationIndicesForLevel[direction].sendIndices.size());
    VF_LOG_INFO("recv send {}", (int)communicationIndicesForLevel[direction].receiveIndices.size());
}

const std::vector<CommunicationIndicesForLevel>& CommunicationNodeFinder::getCommunicationIndices() const
{
    return communicationIndices;
}

//! \}
