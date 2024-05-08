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
#ifndef FLUID_NODE_TAG_BUILDER_H
#define FLUID_NODE_TAG_BUILDER_H

#include <vector>
#include <basics/DataTypes.h>
#include <basics/PointerDefinitions.h>

#include "grid/GridBuilder/CommunicationNodeFinder.h"

class Grid;
class FluidNodeTagger;

class FluidNodeClassificator
{
public:
    FluidNodeClassificator(uint numberOfLevels);

    // findFluidNodes is needed for CUDA Streams MultiGPU (Communication Hiding)
    void findFluidNodes(bool splitDomain, std::vector<std::shared_ptr<Grid>>& grids,
                        const std::vector<CommunicationIndicesForLevel>& communicationIndices);
    void getFluidNodeIndices(uint* fluidNodeIndices, int level) const;
    void getFluidNodeIndicesBorder(uint* fluidNodeIndices, int level) const;
    uint getNumberOfFluidNodes(unsigned int level) const;
    uint getNumberOfFluidNodesBorder(unsigned int level) const;

    void addFluidNodeIndicesMacroVars(const std::vector<uint>& fluidNodeIndicesMacroVars, uint level);
    void addFluidNodeIndicesApplyBodyForce(const std::vector<uint>& fluidNodeIndicesApplyBodyForce, uint level);
    void addFluidNodeIndicesAllFeatures(const std::vector<uint>& fluidNodeIndicesAllFeatures, uint level);

    void sortFluidNodeIndicesMacroVars(uint level);
    void sortFluidNodeIndicesApplyBodyForce(uint level);
    void sortFluidNodeIndicesAllFeatures(uint level);

    uint getNumberOfFluidNodesMacroVars(unsigned int level) const;
    void getFluidNodeIndicesMacroVars(uint* fluidNodeIndicesMacroVars, int level) const;
    uint getNumberOfFluidNodesApplyBodyForce(unsigned int level) const;
    void getFluidNodeIndicesApplyBodyForce(uint* fluidNodeIndicesApplyBodyForce, int level) const;
    uint getNumberOfFluidNodesAllFeatures(unsigned int level) const;
    void getFluidNodeIndicesAllFeatures(uint* fluidNodeIndicesAllFeatures, int level) const;

    bool isSparseIndexInFluidNodeIndicesBorder(uint sparseIndex, uint level) const;

private:
    // one FluidNodeTagger per grid level
    std::vector<UPtr<FluidNodeTagger>> fluidNodeTaggers;
};

#endif

//! }