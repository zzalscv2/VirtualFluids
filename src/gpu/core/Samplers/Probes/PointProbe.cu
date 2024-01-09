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
#include "PointProbe.h"
#include "Probe.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cuda_helper/CudaGrid.h>

#include <basics/constants/NumericConstants.h>

#include "Cuda/CudaMemoryManager.h"
#include "DataStructureInitializer/GridProvider.h"
#include "Parameter/Parameter.h"

bool PointProbe::isAvailableStatistic(Statistic variable)
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

std::vector<PostProcessingVariable> PointProbe::getPostProcessingVariables(Statistic statistic)
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
            throw std::runtime_error("PointProbe::getPostProcessingVariables: Statistic unavailable!");
            break;
    }
    return postProcessingVariables;
}

void PointProbe::findPoints(std::vector<int>& probeIndices, std::vector<real>& distancesX, std::vector<real>& distancesY,
                            std::vector<real>& distancesZ, std::vector<real>& pointCoordsX, std::vector<real>& pointCoordsY,
                            std::vector<real>& pointCoordsZ, int level)
{
    const real* coordinateX = para->getParH(level)->coordinateX;
    const real* coordinateY = para->getParH(level)->coordinateY;
    const real* coordinateZ = para->getParH(level)->coordinateZ;
    const real deltaX = para->getScaledLengthRatio(level);
    for (size_t pos = 1; pos < para->getParH(level)->numberOfNodes; pos++) {
        for (uint point = 0; point < this->pointCoordsX.size(); point++) {
            const real pointCoordX = this->pointCoordsX[point];
            const real pointCoordY = this->pointCoordsY[point];
            const real pointCoordZ = this->pointCoordsZ[point];
            const real distX = pointCoordX - coordinateX[pos];
            const real distY = pointCoordY - coordinateY[pos];
            const real distZ = pointCoordZ - coordinateZ[pos];
            if (distX <= deltaX && distY <= deltaX && distZ <= deltaX && distX > c0o1 && distY > c0o1 && distZ > c0o1) {
                probeIndices.push_back((int)pos);
                distancesX.push_back(distX / deltaX);
                distancesY.push_back(distY / deltaX);
                distancesZ.push_back(distZ / deltaX);
                pointCoordsX.push_back(pointCoordX);
                pointCoordsY.push_back(pointCoordY);
                pointCoordsZ.push_back(pointCoordZ);
            }
        }
    }
}

void PointProbe::calculateQuantities(SPtr<ProbeStruct> probeStruct, uint t, int level)
{
    vf::cuda::CudaGrid grid = vf::cuda::CudaGrid(para->getParH(level)->numberofthreads, probeStruct->nPoints);

    int oldTimestepInTimeseries = this->outputTimeSeries ? calcOldTimestep(probeStruct->timestepInTimeseries, probeStruct->lastTimestepInOldTimeseries) : 0;
    int currentTimestep = this->outputTimeSeries ? probeStruct->timestepInTimeseries : 0;

    interpAndCalcQuantitiesKernel<<<grid.grid, grid.threads>>>(
        probeStruct->pointIndicesD, probeStruct->nPoints, oldTimestepInTimeseries, currentTimestep,
        probeStruct->timestepInTimeAverage, probeStruct->nTimesteps, probeStruct->distXD, probeStruct->distYD,
        probeStruct->distZD, para->getParD(level)->velocityX, para->getParD(level)->velocityY,
        para->getParD(level)->velocityZ, para->getParD(level)->rho, para->getParD(level)->neighborX,
        para->getParD(level)->neighborY, para->getParD(level)->neighborZ, probeStruct->quantitiesD,
        probeStruct->arrayOffsetsD, probeStruct->quantitiesArrayD);
}

void PointProbe::addProbePoint(real pointCoordX, real pointCoordY, real pointCoordZ)
{
    this->pointCoordsX.push_back(pointCoordX);
    this->pointCoordsY.push_back(pointCoordY);
    this->pointCoordsZ.push_back(pointCoordZ);
}

void PointProbe::addProbePointsFromList(std::vector<real>& pointCoordsX, std::vector<real>& pointCoordsY,
                                        std::vector<real>& pointCoordsZ)
{
    if ((pointCoordsX.size() != pointCoordsY.size()) && (pointCoordsY.size() != pointCoordsZ.size()))
        throw std::runtime_error("Probe::addProbePointsFromList(): point lists have different lengths!");

    this->pointCoordsX.insert(this->pointCoordsX.end(), pointCoordsX.begin(), pointCoordsX.end());
    this->pointCoordsY.insert(this->pointCoordsY.end(), pointCoordsY.begin(), pointCoordsY.end());
    this->pointCoordsZ.insert(this->pointCoordsZ.end(), pointCoordsZ.begin(), pointCoordsZ.end());
    printf("Added list of %u  points \n", uint(pointCoordsX.size()));
}

void PointProbe::getTaggedFluidNodes(GridProvider* gridProvider)
{
    for (int level = 0; level <= para->getMaxLevel(); level++) {
        SPtr<ProbeStruct> probeStruct = this->getProbeStruct(level);
        std::vector<uint> probeIndices(probeStruct->pointIndicesH, probeStruct->pointIndicesH + probeStruct->nIndices);
        gridProvider->tagFluidNodeIndices(probeIndices, CollisionTemplate::WriteMacroVars, level);
    }
}

//! \}
