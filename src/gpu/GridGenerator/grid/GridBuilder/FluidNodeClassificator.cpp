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
//! \{
//=======================================================================================

#include "FluidNodeClassificator.h"

#include "grid/FluidNodeTagger.h"
#include "grid/Grid.h"

FluidNodeClassificator::FluidNodeClassificator(uint numberOfLevels)
{
    fluidNodeTaggers.resize(numberOfLevels);
}

void FluidNodeClassificator::findFluidNodes(bool splitDomain, std::vector<std::shared_ptr<Grid>>& grids,
                                            const std::vector<CommunicationIndicesForLevel>& communicationIndices)
{
    VF_LOG_TRACE("Start findFluidNodes()");
    for (uint level = 0; level < grids.size(); level++)
        fluidNodeTaggers[level]->findFluidNodeIndices(splitDomain, grids[level].get(), communicationIndices[level]);
    VF_LOG_TRACE("Done findFluidNodes()");
}

void FluidNodeClassificator::getFluidNodeIndices(uint* fluidNodeIndices, const int level) const
{
    fluidNodeTaggers[level]->getFluidNodeIndices(fluidNodeIndices);
}

void FluidNodeClassificator::getFluidNodeIndicesBorder(uint* fluidNodeIndices, const int level) const
{
    fluidNodeTaggers[level]->getFluidNodeIndicesBorder(fluidNodeIndices);
}

uint FluidNodeClassificator::getNumberOfFluidNodes(unsigned int level) const
{
    return fluidNodeTaggers[level]->getNumberOfFluidNodes();
}

uint FluidNodeClassificator::getNumberOfFluidNodesBorder(unsigned int level) const
{
    return fluidNodeTaggers[level]->getNumberOfFluidNodesBorder();
}

void FluidNodeClassificator::addFluidNodeIndicesMacroVars(const std::vector<uint>& fluidNodeIndicesMacroVars, uint level)
{
    fluidNodeTaggers[level]->addFluidNodeIndicesMacroVars(fluidNodeIndicesMacroVars);
}

void FluidNodeClassificator::addFluidNodeIndicesApplyBodyForce(const std::vector<uint>& fluidNodeIndicesApplyBodyForce,
                                                               uint level)
{

    fluidNodeTaggers[level]->addFluidNodeIndicesApplyBodyForce(fluidNodeIndicesApplyBodyForce);
}

void FluidNodeClassificator::addFluidNodeIndicesAllFeatures(const std::vector<uint>& fluidNodeIndicesAllFeatures, uint level)
{
    fluidNodeTaggers[level]->addFluidNodeIndicesAllFeatures(fluidNodeIndicesAllFeatures);
}

void FluidNodeClassificator::sortFluidNodeIndicesMacroVars(uint level)
{
    fluidNodeTaggers[level]->sortFluidNodeIndicesMacroVars();
}

void FluidNodeClassificator::sortFluidNodeIndicesApplyBodyForce(uint level)
{
    fluidNodeTaggers[level]->sortFluidNodeIndicesApplyBodyForce();
}

void FluidNodeClassificator::sortFluidNodeIndicesAllFeatures(uint level)
{
    fluidNodeTaggers[level]->sortFluidNodeIndicesAllFeatures();
}

uint FluidNodeClassificator::getNumberOfFluidNodesMacroVars(unsigned int level) const
{
    return fluidNodeTaggers[level]->getNumberOfFluidNodeIndicesMacroVars();
}

void FluidNodeClassificator::getFluidNodeIndicesMacroVars(uint* fluidNodeIndicesMacroVars, const int level) const
{
    fluidNodeTaggers[level]->getFluidNodeIndicesMacroVars(fluidNodeIndicesMacroVars);
}

uint FluidNodeClassificator::getNumberOfFluidNodesApplyBodyForce(unsigned int level) const
{
    return fluidNodeTaggers[level]->getNumberOfFluidNodeIndicesApplyBodyForce();
}

void FluidNodeClassificator::getFluidNodeIndicesApplyBodyForce(uint* fluidNodeIndicesApplyBodyForce, const int level) const
{
    fluidNodeTaggers[level]->getFluidNodeIndicesApplyBodyForce(fluidNodeIndicesApplyBodyForce);
}

uint FluidNodeClassificator::getNumberOfFluidNodesAllFeatures(unsigned int level) const
{
    return fluidNodeTaggers[level]->getNumberOfFluidNodeIndicesAllFeatures();
}

void FluidNodeClassificator::getFluidNodeIndicesAllFeatures(uint* fluidNodeIndicesAllFeatures, const int level) const
{
    fluidNodeTaggers[level]->getFluidNodeIndicesAllFeatures(fluidNodeIndicesAllFeatures);
}

bool FluidNodeClassificator::isSparseIndexInFluidNodeIndicesBorder(uint sparseIndex, uint level) const
{
    return fluidNodeTaggers[level]->isSparseIndexInFluidNodeIndicesBorder(sparseIndex);
}

//! }