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
//! \author Soeren Peters, Stephan Lenz, Martin Schoenherr
//=======================================================================================

#include "FluidNodeTagger.h"

#include <algorithm>

#include "grid/Grid.h"
#include "grid/Field.h"
#include "grid/GridBuilder/CommunicationNodeFinder.h"

void FluidNodeTagger::findFluidNodeIndices(bool splitDomain, const Grid* grid, const CommunicationIndicesForLevel& communicationIndices)
{
    // find sparse index of all fluid nodes
    this->fluidNodeIndices.clear();
    for (uint index = 0; index < grid->getSize(); index++) {
        int sparseIndex = grid->getSparseIndex(index);
        if (sparseIndex == -1)
            continue;
        if (Field::isFluidType(grid->getFieldEntry(index)))
            this->fluidNodeIndices.push_back((uint)sparseIndex +
                                             1); // + 1 for numbering shift between GridGenerator and VF_GPU
    }

    // If splitDomain: find fluidNodeIndicesBorder and remove all indices in fluidNodeIndicesBorder from fluidNodeIndices
    if (splitDomain) {
        findFluidNodeIndicesBorder(grid, communicationIndices);
        std::sort(this->fluidNodeIndices.begin(), this->fluidNodeIndices.end());
        auto iterator = std::set_difference(this->fluidNodeIndices.begin(), this->fluidNodeIndices.end(),
                                            this->fluidNodeIndicesBorder.begin(), this->fluidNodeIndicesBorder.end(),
                                            this->fluidNodeIndices.begin());
        this->fluidNodeIndices.resize(iterator - this->fluidNodeIndices.begin());
    }
}

void FluidNodeTagger::findFluidNodeIndicesBorder(const Grid* grid, const CommunicationIndicesForLevel& communicationIndices) {
    this->fluidNodeIndicesBorder.clear();

    // resize fluidNodeIndicesBorder (for better performance in copy operation)
    size_t newSize = 0;
    for (const CommunicationIndicesOfDirection& communicationIndexForDirection : communicationIndices)
        newSize += communicationIndexForDirection.sendIndices.size();
    this->fluidNodeIndicesBorder.reserve(newSize);

    // copy all send indices to fluidNodeIndicesBorder
    for (const CommunicationIndicesOfDirection& communicationIndexForDirection : communicationIndices)
        std::copy(communicationIndexForDirection.sendIndices.begin(), communicationIndexForDirection.sendIndices.end(), std::back_inserter(this->fluidNodeIndicesBorder));

    // remove duplicate elements
    std::sort(this->fluidNodeIndicesBorder.begin(), this->fluidNodeIndicesBorder.end());
    this->fluidNodeIndicesBorder.erase(std::unique(this->fluidNodeIndicesBorder.begin(), this->fluidNodeIndicesBorder.end()),
                                       this->fluidNodeIndicesBorder.end());

    // + 1 for numbering shift between GridGenerator and VF_GPU
    for (size_t i = 0; i < this->fluidNodeIndicesBorder.size(); i++)
        this->fluidNodeIndicesBorder[i] = grid->getSparseIndex(this->fluidNodeIndicesBorder[i]) + 1;
}

bool FluidNodeTagger::isSparseIndexInFluidNodeIndicesBorder(uint sparseIndex) const
{
    return std::find(this->fluidNodeIndicesBorder.begin(), this->fluidNodeIndicesBorder.end(), sparseIndex) !=
           this->fluidNodeIndicesBorder.end();
}

uint FluidNodeTagger::getNumberOfFluidNodes() const
{
    return (uint)this->fluidNodeIndices.size();
}

void FluidNodeTagger::getFluidNodeIndices(uint* fluidNodeIndices) const
{
    for (uint nodeNumber = 0; nodeNumber < (uint)this->fluidNodeIndices.size(); nodeNumber++)
        fluidNodeIndices[nodeNumber] = this->fluidNodeIndices[nodeNumber];
}

uint FluidNodeTagger::getNumberOfFluidNodesBorder() const
{
    return (uint)this->fluidNodeIndicesBorder.size();
}

void FluidNodeTagger::getFluidNodeIndicesBorder(uint* fluidNodeIndicesBorder) const
{
    for (uint nodeNumber = 0; nodeNumber < (uint)this->fluidNodeIndicesBorder.size(); nodeNumber++)
        fluidNodeIndicesBorder[nodeNumber] = this->fluidNodeIndicesBorder[nodeNumber];
}

void FluidNodeTagger::addFluidNodeIndicesMacroVars(std::vector<uint> _fluidNodeIndicesMacroVars)
{
    size_t newSize = this->fluidNodeIndicesMacroVars.size() + _fluidNodeIndicesMacroVars.size();
    this->fluidNodeIndicesMacroVars.reserve(newSize);
    std::copy(_fluidNodeIndicesMacroVars.begin(), _fluidNodeIndicesMacroVars.end(),
              std::back_inserter(this->fluidNodeIndicesMacroVars));
}

void FluidNodeTagger::addFluidNodeIndicesApplyBodyForce(std::vector<uint> _fluidNodeIndicesApplyBodyForce)
{

    size_t newSize = this->fluidNodeIndicesApplyBodyForce.size() + _fluidNodeIndicesApplyBodyForce.size();
    this->fluidNodeIndicesApplyBodyForce.reserve(newSize);
    std::copy(_fluidNodeIndicesApplyBodyForce.begin(), _fluidNodeIndicesApplyBodyForce.end(),
              std::back_inserter(this->fluidNodeIndicesApplyBodyForce));
}

void FluidNodeTagger::addFluidNodeIndicesAllFeatures(std::vector<uint> _fluidNodeIndicesAllFeatures)
{

    size_t newSize = this->fluidNodeIndicesAllFeatures.size() + _fluidNodeIndicesAllFeatures.size();
    this->fluidNodeIndicesAllFeatures.reserve(newSize);
    std::copy(_fluidNodeIndicesAllFeatures.begin(), _fluidNodeIndicesAllFeatures.end(),
              std::back_inserter(this->fluidNodeIndicesAllFeatures));
}

void FluidNodeTagger::sortFluidNodeIndicesMacroVars()
{
    if(this->fluidNodeIndicesMacroVars.size()>0)
    {
        std::sort(this->fluidNodeIndicesMacroVars.begin(), this->fluidNodeIndicesMacroVars.end());
        // Remove duplicates
        this->fluidNodeIndicesMacroVars.erase( std::unique( this->fluidNodeIndicesMacroVars.begin(), this->fluidNodeIndicesMacroVars.end() ), this->fluidNodeIndicesMacroVars.end() );

         // Remove indices of fluidNodeIndicesAllFeatures from fluidNodeIndicesMacroVars
        if(this->fluidNodeIndicesAllFeatures.size()>0)
        {
            this->fluidNodeIndicesMacroVars.erase(   std::remove_if(   this->fluidNodeIndicesMacroVars.begin(), this->fluidNodeIndicesMacroVars.end(),
                                                    [&](auto x){return binary_search(fluidNodeIndicesAllFeatures.begin(),fluidNodeIndicesAllFeatures.end(),x);} ),
                                                    this->fluidNodeIndicesMacroVars.end() );
        }

        // Remove all indices in fluidNodeIndicesBorder from fluidNodeIndicesApplyBodyForce
        if(this->fluidNodeIndicesBorder.size()>0)
        {
            this->fluidNodeIndicesMacroVars.erase(  std::remove_if(   this->fluidNodeIndicesMacroVars.begin(), this->fluidNodeIndicesMacroVars.end(),
                                                    [&](auto x){return binary_search(fluidNodeIndicesBorder.begin(),fluidNodeIndicesBorder.end(),x);} ),
                                                    this->fluidNodeIndicesMacroVars.end() );
        }

        // Remove indices of fluidNodeIndicesMacroVars from fluidNodeIndices
        this->fluidNodeIndices.erase(   std::remove_if(   this->fluidNodeIndices.begin(), this->fluidNodeIndices.end(),
                                                        [&](auto x){return binary_search(fluidNodeIndicesMacroVars.begin(),fluidNodeIndicesMacroVars.end(),x);} ),
                                        this->fluidNodeIndices.end() );
    }
}

void FluidNodeTagger::sortFluidNodeIndicesApplyBodyForce()
{
    if(this->fluidNodeIndicesApplyBodyForce.size()>0)
    {
        std::sort(this->fluidNodeIndicesApplyBodyForce.begin(), this->fluidNodeIndicesApplyBodyForce.end());
        // Remove duplicates
        this->fluidNodeIndicesApplyBodyForce.erase( std::unique( this->fluidNodeIndicesApplyBodyForce.begin(), this->fluidNodeIndicesApplyBodyForce.end() ), this->fluidNodeIndicesApplyBodyForce.end() );

         // Remove indices of fluidNodeIndicesAllFeatures from fluidNodeIndicesApplyBodyForce
        if(this->fluidNodeIndicesAllFeatures.size()>0)
        {
            this->fluidNodeIndicesApplyBodyForce.erase( std::remove_if(   this->fluidNodeIndicesApplyBodyForce.begin(), this->fluidNodeIndicesApplyBodyForce.end(),
                                                        [&](auto x){return binary_search(fluidNodeIndicesAllFeatures.begin(),fluidNodeIndicesAllFeatures.end(),x);} ),
                                                        this->fluidNodeIndicesApplyBodyForce.end() );
        }

        // Remove all indices in fluidNodeIndicesBorder from fluidNodeIndicesApplyBodyForce
        if(this->fluidNodeIndicesBorder.size()>0)
        {
            this->fluidNodeIndicesApplyBodyForce.erase( std::remove_if(   this->fluidNodeIndicesApplyBodyForce.begin(), this->fluidNodeIndicesApplyBodyForce.end(),
                                                        [&](auto x){return binary_search(fluidNodeIndicesBorder.begin(),fluidNodeIndicesBorder.end(),x);} ),
                                                        this->fluidNodeIndicesApplyBodyForce.end() );
        }

        // Remove indices of fluidNodeIndicesMacroVars from fluidNodeIndices
        this->fluidNodeIndices.erase(   std::remove_if(   this->fluidNodeIndices.begin(), this->fluidNodeIndices.end(),
                                        [&](auto x){return binary_search(fluidNodeIndicesApplyBodyForce.begin(),fluidNodeIndicesApplyBodyForce.end(),x);} ),
                                        this->fluidNodeIndices.end() );
    }
}

void FluidNodeTagger::sortFluidNodeIndicesAllFeatures()
{
    if(this->fluidNodeIndicesAllFeatures.size()>0)
    {
        std::sort(this->fluidNodeIndicesAllFeatures.begin(), this->fluidNodeIndicesAllFeatures.end());
        // Remove duplicates
        this->fluidNodeIndicesAllFeatures.erase( std::unique( this->fluidNodeIndicesAllFeatures.begin(), this->fluidNodeIndicesAllFeatures.end() ), this->fluidNodeIndicesAllFeatures.end() );

        // Remove all indices in fluidNodeIndicesBorder from fluidNodeIndicesAllFeatures
        if(this->fluidNodeIndicesBorder.size()>0)
        {
            this->fluidNodeIndicesAllFeatures.erase(    std::remove_if(   this->fluidNodeIndicesAllFeatures.begin(), this->fluidNodeIndicesAllFeatures.end(),
                                                        [&](auto x){return binary_search(fluidNodeIndicesBorder.begin(),fluidNodeIndicesBorder.end(),x);} ),
                                                        this->fluidNodeIndicesAllFeatures.end() );
        }

        // Remove indices of fluidNodeIndicesAllFeatures from fluidNodeIndices
        this->fluidNodeIndices.erase(   std::remove_if(   this->fluidNodeIndices.begin(), this->fluidNodeIndices.end(),
                                                        [&](auto x){return binary_search(fluidNodeIndicesAllFeatures.begin(),fluidNodeIndicesAllFeatures.end(),x);} ),
                                        this->fluidNodeIndices.end() );
    }
}


uint FluidNodeTagger::getNumberOfFluidNodeIndicesMacroVars() const {
    return (uint)this->fluidNodeIndicesMacroVars.size();
}

uint FluidNodeTagger::getNumberOfFluidNodeIndicesApplyBodyForce() const {
    return (uint)this->fluidNodeIndicesApplyBodyForce.size();
}

uint FluidNodeTagger::getNumberOfFluidNodeIndicesAllFeatures() const {
    return (uint)this->fluidNodeIndicesAllFeatures.size();
}

void FluidNodeTagger::getFluidNodeIndicesMacroVars(uint* _fluidNodeIndicesMacroVars) const
{
    std::copy(fluidNodeIndicesMacroVars.begin(), fluidNodeIndicesMacroVars.end(), _fluidNodeIndicesMacroVars);
}

void FluidNodeTagger::getFluidNodeIndicesApplyBodyForce(uint* _fluidNodeIndicesApplyBodyForce) const
{
    std::copy(fluidNodeIndicesApplyBodyForce.begin(), fluidNodeIndicesApplyBodyForce.end(), _fluidNodeIndicesApplyBodyForce);
}

void FluidNodeTagger::getFluidNodeIndicesAllFeatures(uint* _fluidNodeIndicesAllFeatures) const
{
    std::copy(fluidNodeIndicesAllFeatures.begin(), fluidNodeIndicesAllFeatures.end(), _fluidNodeIndicesAllFeatures);
}
//! \}
