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
#include "Probe.h"
#include "WallModelProbe.h"

#include <stdexcept>

#include <helper_cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <basics/DataTypes.h>

#include "Cuda/CudaMemoryManager.h"
#include "DataStructureInitializer/GridProvider.h"
#include "Parameter/Parameter.h"
#include "basics/constants/NumericConstants.h"

using namespace vf::basics::constant;
using valueIterator = thrust::device_vector<real>::iterator;
using indexIterator = thrust::device_vector<uint>::iterator;
using permuationIterator = thrust::permutation_iterator<valueIterator, indexIterator>;
///////////////////////////////////////////////////////////////////////////////////
bool WallModelProbe::isAvailableStatistic(Statistic variable)
{
    switch (variable) {
        case Statistic::SpatialMeans:
        case Statistic::SpatioTemporalMeans:
            return true;
        default:
            return false;
    }
}

///////////////////////////////////////////////////////////////////////////////////

std::vector<PostProcessingVariable> WallModelProbe::getPostProcessingVariables(Statistic statistic)
{
    std::vector<PostProcessingVariable> postProcessingVariables;
    switch (statistic) {
        case Statistic::SpatialMeans:
            postProcessingVariables.emplace_back("vx_el_spatMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vy_el_spatMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vz_el_spatMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vx1_spatMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vy1_spatMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vz1_spatMean", this->velocityRatio);
            postProcessingVariables.emplace_back("u_star_spatMean", this->velocityRatio);
            postProcessingVariables.emplace_back("Fx_spatMean", this->outputStress ? this->stressRatio : this->forceRatio);
            postProcessingVariables.emplace_back("Fy_spatMean", this->outputStress ? this->stressRatio : this->forceRatio);
            postProcessingVariables.emplace_back("Fz_spatMean", this->outputStress ? this->stressRatio : this->forceRatio);
            if (this->evaluatePressureGradient) {
                postProcessingVariables.emplace_back("dpdx_spatMean", this->forceRatio);
                postProcessingVariables.emplace_back("dpdy_spatMean", this->forceRatio);
                postProcessingVariables.emplace_back("dpdz_spatMean", this->forceRatio);
            }
            break;
        case Statistic::SpatioTemporalMeans:
            postProcessingVariables.emplace_back("vx_el_spatTmpMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vy_el_spatTmpMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vz_el_spatTmpMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vx1_spatTmpMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vy1_spatTmpMean", this->velocityRatio);
            postProcessingVariables.emplace_back("vz1_spatTmpMean", this->velocityRatio);
            postProcessingVariables.emplace_back("u_star_spatTmpMean", this->velocityRatio);
            postProcessingVariables.emplace_back("Fx_spatTmpMean",
                                                 this->outputStress ? this->stressRatio : this->forceRatio);
            postProcessingVariables.emplace_back("Fy_spatTmpMean",
                                                 this->outputStress ? this->stressRatio : this->forceRatio);
            postProcessingVariables.emplace_back("Fz_spatTmpMean",
                                                 this->outputStress ? this->stressRatio : this->forceRatio);
            if (this->evaluatePressureGradient) {
                postProcessingVariables.emplace_back("dpdx_spatTmpMean", this->forceRatio);
                postProcessingVariables.emplace_back("dpdy_spatTmpMean", this->forceRatio);
                postProcessingVariables.emplace_back("dpdz_spatTmpMean", this->forceRatio);
            }
            break;

        default:
            throw std::runtime_error("WallModelProbe::getPostProcessingVariables: Statistic unavailable!");
            break;
    }
    return postProcessingVariables;
}

///////////////////////////////////////////////////////////////////////////////////

void WallModelProbe::findPoints(std::vector<int>& probeIndices_level, std::vector<real>& distX_level,
                                std::vector<real>& distY_level, std::vector<real>& distZ_level,
                                std::vector<real>& pointCoordsX_level, std::vector<real>& pointCoordsY_level,
                                std::vector<real>& pointCoordsZ_level, int level)
{
    if (!para->getHasWallModelMonitor())
        throw std::runtime_error("WallModelProbe::findPoints(): !para->getHasWallModelMonitor() !");

    pointCoordsX_level.push_back(0);
    pointCoordsY_level.push_back(0);
    pointCoordsZ_level.push_back(0);

    if (this->evaluatePressureGradient) {
        if (!para->getIsBodyForce())
            throw std::runtime_error("WallModelProbe::findPoints(): bodyforce not allocated!");
        // Find all fluid nodes
        for (size_t pos = 1; pos < para->getParH(level)->numberOfNodes; pos++) {
            if (GEO_FLUID == para->getParH(level)->typeOfGridNode[pos]) {
                probeIndices_level.push_back((int)pos);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////

template <typename T>
T spatial_mean(T* device_pointer, uint numberOfPoints)
{
    thrust::device_ptr<T> thrust_pointer = thrust::device_pointer_cast(device_pointer);
    return thrust::reduce(thrust_pointer, thrust_pointer + numberOfPoints) / real(numberOfPoints);
}

template <typename T>
T index_based_spatial_mean(T* device_pointer, thrust::device_ptr<uint> indeces_ptr, uint numberOfIndeces)
{
    thrust::device_ptr<T> thrust_pointer = thrust::device_pointer_cast(device_pointer);

    permuationIterator iter_begin(thrust_pointer, indeces_ptr);
    permuationIterator iter_end(thrust_pointer, indeces_ptr + numberOfIndeces);
    return thrust::reduce(iter_begin, iter_end) / real(numberOfIndeces);
}

template <typename T>
T compute_and_save_mean(T* device_pointer, uint numberOfPoints, T* quantitiesArray, uint timestep, uint numberOfTimesteps,
                        uint indexOfArray)
{
    T mean = spatial_mean(device_pointer, numberOfPoints);
    quantitiesArray[calcArrayIndex(timestep, numberOfTimesteps, indexOfArray)] = mean;
    return mean;
}

template <typename T>
T compute_and_save_index_based_mean(T* device_pointer, thrust::device_ptr<uint> indeces_ptr, uint numberOfIndices,
                                    T* quantitiesArray, uint timestep, uint numberOfTimesteps, uint indexOfArray)
{
    const T mean = index_based_spatial_mean(device_pointer, indeces_ptr, numberOfIndices);
    quantitiesArray[calcArrayIndex(timestep, numberOfTimesteps, indexOfArray)] = mean;
    return mean;
}

template <typename T>
void temporal_average(T* quantitiesArray, T currentValue, uint currentTimestep, uint numberOfTimesteps, uint oldTimestep,
                      uint indexOfArray, real invNumberOfAverages)
{
    const T oldMean = quantitiesArray[calcArrayIndex(oldTimestep, numberOfTimesteps, indexOfArray)];
    quantitiesArray[calcArrayIndex(currentTimestep, numberOfTimesteps, indexOfArray)] =
        oldMean + (currentValue - oldMean) * invNumberOfAverages;
}

void WallModelProbe::calculateQuantities(SPtr<ProbeStruct> probeStruct, uint t, int level)
{
    const bool doTemporalAveraging = (t > this->getTStartTmpAveraging());
    const uint numberOfStressBCPoints = para->getParD(level)->stressBC.numberOfBCnodes;
    if (numberOfStressBCPoints < 1)
        return; // Skipping levels without StressBC
    const uint timestep = probeStruct->timestepInTimeseries;
    const real inverseNumberOfAveragedValues = c1o1 / real(probeStruct->numberOfAveragedValues + 1);
    const uint oldTimestep = calcOldTimestep(timestep, probeStruct->lastTimestepInOldTimeseries);

    thrust::device_ptr<uint> indices_thrust = thrust::device_pointer_cast(probeStruct->pointIndicesD);

    if (probeStruct->quantitiesH[int(Statistic::SpatialMeans)]) {
        const uint arrOff = probeStruct->arrayOffsetsH[int(Statistic::SpatialMeans)];
        // Compute the instantaneous spatial means of the velocity moments
        const real velocityXExchangeLocationSpatialMean =
            compute_and_save_mean(para->getParD(level)->stressBC.Vx, numberOfStressBCPoints, probeStruct->quantitiesArrayH,
                                  timestep, probeStruct->nTimesteps, arrOff + 0);
        const real velocityYExchangeLocationSpatialMean =
            compute_and_save_mean(para->getParD(level)->stressBC.Vy, numberOfStressBCPoints, probeStruct->quantitiesArrayH,
                                  timestep, probeStruct->nTimesteps, arrOff + 1);
        const real velocityZExchangeLocationSpatialMean =
            compute_and_save_mean(para->getParD(level)->stressBC.Vz, numberOfStressBCPoints, probeStruct->quantitiesArrayH,
                                  timestep, probeStruct->nTimesteps, arrOff + 2);
        const real velocityXFirstNodeSpatialMean =
            compute_and_save_mean(para->getParD(level)->stressBC.Vx1, numberOfStressBCPoints, probeStruct->quantitiesArrayH,
                                  timestep, probeStruct->nTimesteps, arrOff + 3);
        const real velocityYFirstNodeSpatialMean =
            compute_and_save_mean(para->getParD(level)->stressBC.Vy1, numberOfStressBCPoints, probeStruct->quantitiesArrayH,
                                  timestep, probeStruct->nTimesteps, arrOff + 4);
        const real velocityZFirstNodeSpatialMean =
            compute_and_save_mean(para->getParD(level)->stressBC.Vz1, numberOfStressBCPoints, probeStruct->quantitiesArrayH,
                                  timestep, probeStruct->nTimesteps, arrOff + 5);
        const real frictionVelocitySpatialMean =
            compute_and_save_mean(para->getParD(level)->wallModel.u_star, numberOfStressBCPoints,
                                  probeStruct->quantitiesArrayH, timestep, probeStruct->nTimesteps, arrOff + 6);
        const real forceXSpatialMean =
            compute_and_save_mean(para->getParD(level)->wallModel.Fx, numberOfStressBCPoints, probeStruct->quantitiesArrayH,
                                  timestep, probeStruct->nTimesteps, arrOff + 7);
        const real forceYSpatialMean =
            compute_and_save_mean(para->getParD(level)->wallModel.Fy, numberOfStressBCPoints, probeStruct->quantitiesArrayH,
                                  timestep, probeStruct->nTimesteps, arrOff + 8);
        const real forceZSpatialMean =
            compute_and_save_mean(para->getParD(level)->wallModel.Fz, numberOfStressBCPoints, probeStruct->quantitiesArrayH,
                                  timestep, probeStruct->nTimesteps, arrOff + 9);

        real pressureGradientXSpatialMean;
        real pressureGradientYSpatialMean;
        real pressureGradientZSpatialMean;
        if (this->evaluatePressureGradient) {
            pressureGradientXSpatialMean = compute_and_save_index_based_mean(
                para->getParD(level)->forceX_SP, indices_thrust, probeStruct->nIndices, probeStruct->quantitiesArrayH,
                timestep, probeStruct->nTimesteps, arrOff + 10);
            pressureGradientYSpatialMean = compute_and_save_index_based_mean(
                para->getParD(level)->forceY_SP, indices_thrust, probeStruct->nIndices, probeStruct->quantitiesArrayH,
                timestep, probeStruct->nTimesteps, arrOff + 11);
            pressureGradientZSpatialMean = compute_and_save_index_based_mean(
                para->getParD(level)->forceZ_SP, indices_thrust, probeStruct->nIndices, probeStruct->quantitiesArrayH,
                timestep, probeStruct->nTimesteps, arrOff + 12);
        }

        if (probeStruct->quantitiesH[int(Statistic::SpatioTemporalMeans)] && doTemporalAveraging) {
            const uint arrOff2 = probeStruct->arrayOffsetsH[int(Statistic::SpatioTemporalMeans)];
            temporal_average(probeStruct->quantitiesArrayH, velocityXExchangeLocationSpatialMean, timestep,
                             probeStruct->nTimesteps, oldTimestep, arrOff2 + 0, inverseNumberOfAveragedValues);
            temporal_average(probeStruct->quantitiesArrayH, velocityYExchangeLocationSpatialMean, timestep,
                             probeStruct->nTimesteps, oldTimestep, arrOff2 + 1, inverseNumberOfAveragedValues);
            temporal_average(probeStruct->quantitiesArrayH, velocityYExchangeLocationSpatialMean, timestep,
                             probeStruct->nTimesteps, oldTimestep, arrOff2 + 2, inverseNumberOfAveragedValues);
            temporal_average(probeStruct->quantitiesArrayH, velocityXFirstNodeSpatialMean, timestep, probeStruct->nTimesteps,
                             oldTimestep, arrOff2 + 3, inverseNumberOfAveragedValues);
            temporal_average(probeStruct->quantitiesArrayH, velocityYFirstNodeSpatialMean, timestep, probeStruct->nTimesteps,
                             oldTimestep, arrOff2 + 4, inverseNumberOfAveragedValues);
            temporal_average(probeStruct->quantitiesArrayH, velocityZFirstNodeSpatialMean, timestep, probeStruct->nTimesteps,
                             oldTimestep, arrOff2 + 5, inverseNumberOfAveragedValues);
            temporal_average(probeStruct->quantitiesArrayH, frictionVelocitySpatialMean, timestep, probeStruct->nTimesteps,
                             oldTimestep, arrOff2 + 6, inverseNumberOfAveragedValues);
            temporal_average(probeStruct->quantitiesArrayH, forceXSpatialMean, timestep, probeStruct->nTimesteps,
                             oldTimestep, arrOff2 + 7, inverseNumberOfAveragedValues);
            temporal_average(probeStruct->quantitiesArrayH, forceYSpatialMean, timestep, probeStruct->nTimesteps,
                             oldTimestep, arrOff2 + 8, inverseNumberOfAveragedValues);
            temporal_average(probeStruct->quantitiesArrayH, forceZSpatialMean, timestep, probeStruct->nTimesteps,
                             oldTimestep, arrOff2 + 9, inverseNumberOfAveragedValues);

            if (this->evaluatePressureGradient) {
                temporal_average(probeStruct->quantitiesArrayH, pressureGradientXSpatialMean, timestep,
                                 probeStruct->nTimesteps, oldTimestep, arrOff2 + 10, inverseNumberOfAveragedValues);
                temporal_average(probeStruct->quantitiesArrayH, pressureGradientYSpatialMean, timestep,
                                 probeStruct->nTimesteps, oldTimestep, arrOff2 + 11, inverseNumberOfAveragedValues);
                temporal_average(probeStruct->quantitiesArrayH, pressureGradientZSpatialMean, timestep,
                                 probeStruct->nTimesteps, oldTimestep, arrOff2 + 12, inverseNumberOfAveragedValues);
            }
        }
    }
    getLastCudaError("WallModelProbe::calculateQuantities execution failed");
}

uint WallModelProbe::getNumberOfTimestepsInTimeseries(int level)
{
    return this->tBetweenWriting * exp2(level) / this->tBetweenAverages + 1;
}

//! \}
