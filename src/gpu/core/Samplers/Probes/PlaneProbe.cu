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
//! \addtogroup gpu_PreCollisionInteractor PreCollisionInteractor
//! \ingroup gpu_core core
//! \{
#include "PlaneProbe.h"
#include "Probe.h"

#include <stdexcept>

#include <cuda_helper/CudaGrid.h>

#include <basics/DataTypes.h>
#include <basics/constants/NumericConstants.h>

#include "Cuda/CudaMemoryManager.h"
#include "DataStructureInitializer/GridProvider.h"
#include "Parameter/Parameter.h"

using namespace vf::basics::constant;

bool PlaneProbe::isAvailableStatistic(Statistic variable)
{
    switch (variable) {
        case Statistic::Instantaneous:
        case Statistic::Means:
        case Statistic::Variances:
            return true;
        default:
            return false;
    }
}

std::vector<PostProcessingVariable> PlaneProbe::getPostProcessingVariables(Statistic statistic)
{
    std::vector<PostProcessingVariable> postProcessingVariables;
    switch (statistic) {
        case Statistic::Instantaneous:
            postProcessingVariables.emplace_back("vx", this->velocityRatio);
            postProcessingVariables.emplace_back("vy", this->velocityRatio);
            postProcessingVariables.emplace_back("vz", this->velocityRatio);
            postProcessingVariables.emplace_back("rho", this->densityRatio);
            break;
        case Statistic::Means:
            postProcessingVariables.emplace_back("vx_mean", this->velocityRatio);
            postProcessingVariables.emplace_back("vy_mean", this->velocityRatio);
            postProcessingVariables.emplace_back("vz_mean", this->velocityRatio);
            postProcessingVariables.emplace_back("rho_mean", this->densityRatio);
            break;
        case Statistic::Variances:
            postProcessingVariables.emplace_back("vx_var", this->stressRatio);
            postProcessingVariables.emplace_back("vy_var", this->stressRatio);
            postProcessingVariables.emplace_back("vz_var", this->stressRatio);
            postProcessingVariables.emplace_back("rho_var", this->densityRatio);
            break;

        default:
            throw std::runtime_error("PlaneProbe::getPostProcessingVariables: Statistic unavailable!");
            break;
    }
    return postProcessingVariables;
}

void PlaneProbe::findPoints(std::vector<int>& probeIndices, std::vector<real>& distancesX, std::vector<real>& distancesY,
                            std::vector<real>& distancesZ, std::vector<real>& pointCoordsX, std::vector<real>& pointCoordsY,
                            std::vector<real>& pointCoordsZ, int level)
{
    const real* coordinateX = para->getParH(level)->coordinateX;
    const real* coordinateY = para->getParH(level)->coordinateY;
    const real* coordinateZ = para->getParH(level)->coordinateZ;
    const real deltaX = coordinateX[para->getParH(level)->neighborX[1]] - coordinateX[1];
    for (size_t pos = 1; pos < para->getParH(level)->numberOfNodes; pos++) {
        const real pointCoordX = para->getParH(level)->coordinateX[pos];
        const real pointCoordY = para->getParH(level)->coordinateY[pos];
        const real pointCoordZ = para->getParH(level)->coordinateZ[pos];
        const real distanceX = pointCoordX - this->posX;
        const real distanceY = pointCoordY - this->posY;
        const real distanceZ = pointCoordZ - this->posZ;

        if (distanceX <= this->deltaX && distanceY <= this->deltaY && distanceZ <= this->deltaZ && distanceX >= c0o1 &&
            distanceY >= c0o1 && distanceZ >= c0o1) {
            probeIndices.push_back((int)pos);
            distancesX.push_back(distanceX / deltaX);
            distancesY.push_back(distanceY / deltaX);
            distancesZ.push_back(distanceZ / deltaX);
            pointCoordsX.push_back(pointCoordX);
            pointCoordsY.push_back(pointCoordY);
            pointCoordsZ.push_back(pointCoordZ);
        }
    }
}

void PlaneProbe::calculateQuantities(SPtr<ProbeStruct> probeStruct, uint t, int level)
{
    const GridParams gridParams = getGridParams(probeStruct.get(), para->getParD(level).get());
    const ProbeArray probeArray = getProbeArray(probeStruct.get());
    vf::cuda::CudaGrid grid = vf::cuda::CudaGrid(para->getParH(level)->numberofthreads, probeStruct->nPoints);
    calculateQuantitiesKernel<<<grid.grid, grid.threads>>>(probeStruct->nTimesteps, gridParams, probeArray);
}

void PlaneProbe::getTaggedFluidNodes(GridProvider* gridProvider)
{
    for (int level = 0; level <= para->getMaxLevel(); level++) {
        SPtr<ProbeStruct> probeStruct = this->getProbeStruct(level);
        std::vector<uint> probeIndices(probeStruct->pointIndicesH, probeStruct->pointIndicesH + probeStruct->nIndices);
        gridProvider->tagFluidNodeIndices(probeIndices, CollisionTemplate::WriteMacroVars, level);
    }
}
//! \}
