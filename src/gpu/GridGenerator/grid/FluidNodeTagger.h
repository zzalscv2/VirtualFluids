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
#ifndef FLUID_NODE_TAGGER_H
#define FLUID_NODE_TAGGER_H

#include <array>
#include <vector>

#include "grid/GridBuilder/CommunicationNodeFinder.h"
#include <basics/DataTypes.h>

class Grid;

class FluidNodeTagger
{

public:
    void findFluidNodeIndices(bool splitDomain, const Grid* grid,
                              const CommunicationIndicesForLevel& communicationIndices);

    uint getNumberOfFluidNodes() const;
    void getFluidNodeIndices(uint* fluidNodeIndices) const;

    uint getNumberOfFluidNodesBorder() const;
    void getFluidNodeIndicesBorder(uint* fluidNodeIndicesBorder) const;

    bool isSparseIndexInFluidNodeIndicesBorder(uint sparseIndex) const;

    void addFluidNodeIndicesMacroVars(std::vector<uint> _fluidNodeIndicesMacroVars);
    void addFluidNodeIndicesApplyBodyForce(std::vector<uint> _fluidNodeIndicesApplyBodyForce);
    void addFluidNodeIndicesAllFeatures(std::vector<uint> _fluidNodeIndicesAllFeatures);

    void sortFluidNodeIndicesMacroVars();
    void sortFluidNodeIndicesApplyBodyForce();
    void sortFluidNodeIndicesAllFeatures();

    uint getNumberOfFluidNodeIndicesMacroVars() const;
    uint getNumberOfFluidNodeIndicesApplyBodyForce() const;
    uint getNumberOfFluidNodeIndicesAllFeatures() const;

    void getFluidNodeIndicesMacroVars(uint* fluidNodeIndicesMacroVars) const;
    void getFluidNodeIndicesApplyBodyForce(uint* fluidNodeIndicesApplyBodyForce) const;
    void getFluidNodeIndicesAllFeatures(uint* fluidNodeIndicesAllFeatures) const;

private:
    void findFluidNodeIndicesBorder(const Grid* grid, const CommunicationIndicesForLevel& communicationIndices);

    std::vector<uint> fluidNodeIndices;                 // run on CollisionTemplate::Default
    std::vector<uint> fluidNodeIndicesBorder;           // run on subdomain border nodes (CollisionTemplate::SubDomainBorder)
    std::vector<uint> fluidNodeIndicesMacroVars;        // run on CollisionTemplate::MacroVars
    std::vector<uint> fluidNodeIndicesApplyBodyForce;   // run on CollisionTemplate::ApplyBodyForce
    std::vector<uint> fluidNodeIndicesAllFeatures;      // run on CollisionTemplate::AllFeatures
};

#endif

//! \}
