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
//! \author Henry Korb, Henrik Asmuth
//=======================================================================================

#include "Probe.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <basics/DataTypes.h>
#include <basics/StringUtilities/StringUtil.h>
#include <basics/constants/NumericConstants.h>

#include "gpu/core/Cuda/CudaMemoryManager.h"
#include "gpu/core/Output/FilePartCalculator.h"
#include "gpu/core/Parameter/Parameter.h"
#include "gpu/core/Utilities/GeometryUtils.h"
#include "gpu/core/Utilities/KernelUtilities.h"

using namespace vf::basics::constant;

__host__ __device__ int calcArrayIndex(int node, int nNodes, int timestep, int nTimesteps, int array)
{
    return node + nNodes * (timestep + nTimesteps * array);
}

__host__ __device__ int calcArrayIndex(int node, int nNodes, int array)
{
    return node + nNodes * array;
}

uint calcOldTimestep(uint currentTimestep, uint lastTimestepInOldSeries)
{
    return currentTimestep > 0 ? currentTimestep - 1 : lastTimestepInOldSeries;
}

__host__ __device__ real computeMean(real oldMean, real newValue, real inverseCount)
{
    return oldMean + (newValue - oldMean) * inverseCount;
}

__host__ __device__ real computeAndSaveMean(real* quantityArray, real oldValue, uint index, real currentValue, real invCount)
{
    const real newValue = computeMean(oldValue, currentValue, invCount);
    quantityArray[index] = newValue;
    return newValue;
}

__host__ __device__ real computeVariance(real oldVariance, real oldMean, real newMean, real currentValue,
                                         uint numberOfAveragedValues, real inverseCount)
{
    return numberOfAveragedValues * oldVariance + (currentValue - oldMean) * (currentValue - newMean) * inverseCount;
}

__host__ __device__ real computeAndSaveVariance(real* quantityArray, real oldVariance, uint indexNew, real currentValue,
                                                real oldMean, real newMean, uint numberOfAveragedValues, real inverseCount)
{
    const real newVariance =
        computeVariance(oldVariance, oldMean, newMean, currentValue, numberOfAveragedValues, inverseCount);
    quantityArray[indexNew] = newVariance;
    return newVariance;
}

__device__ void calculatePointwiseQuantities(uint numberOfAveragedValues, ProbeArray array, uint node, real velocityX,
                                             real velocityY, real velocityZ, real density)
{
    //"https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    // also has extensions for higher order and covariances
    const real invCount = c1o1 / real(numberOfAveragedValues + 1);
    const uint nPoints = array.numberOfPoints;

    if (array.statistics[int(Statistic::Instantaneous)]) {
        const uint arrayOffset = array.offsets[int(Statistic::Instantaneous)];
        array.data[calcArrayIndex(node, nPoints, arrayOffset + 0)] = velocityX;
        array.data[calcArrayIndex(node, nPoints, arrayOffset + 1)] = velocityY;
        array.data[calcArrayIndex(node, nPoints, arrayOffset + 2)] = velocityZ;
        array.data[calcArrayIndex(node, nPoints, arrayOffset + 3)] = density;
    }

    if (array.statistics[int(Statistic::Means)]) {

        real vxMeanOld, vxMeanNew, vyMeanOld, vyMeanNew, vzMeanOld, vzMeanNew, rhoMeanOld, rhoMeanNew;
        {
            const uint arrayOffset = array.offsets[int(Statistic::Means)];
            const uint indexVx = calcArrayIndex(node, nPoints, arrayOffset + 0);
            const uint indexVy = calcArrayIndex(node, nPoints, arrayOffset + 1);
            const uint indexVz = calcArrayIndex(node, nPoints, arrayOffset + 2);
            const uint indexRho = calcArrayIndex(node, nPoints, arrayOffset + 3);

            vxMeanOld = array.data[indexVx];
            vyMeanOld = array.data[indexVy];
            vzMeanOld = array.data[indexVz];
            rhoMeanOld = array.data[indexRho];

            vxMeanNew = computeAndSaveMean(array.data, vxMeanOld, indexVx, velocityX, invCount);
            vyMeanNew = computeAndSaveMean(array.data, vyMeanOld, indexVy, velocityY, invCount);
            vzMeanNew = computeAndSaveMean(array.data, vzMeanOld, indexVz, velocityZ, invCount);
            rhoMeanNew = computeAndSaveMean(array.data, rhoMeanOld, indexRho, density, invCount);
        }

        if (array.statistics[int(Statistic::Variances)]) {
            const uint arrayOffset = array.offsets[int(Statistic::Variances)];
            const uint indexVx = calcArrayIndex(node, nPoints, arrayOffset + 0);
            const uint indexVy = calcArrayIndex(node, nPoints, arrayOffset + 1);
            const uint indexVz = calcArrayIndex(node, nPoints, arrayOffset + 2);
            const uint indexRho = calcArrayIndex(node, nPoints, arrayOffset + 3);

            const real vxVarianceOld = array.data[indexVx];
            const real vyVarianceOld = array.data[indexVy];
            const real vzVarianceOld = array.data[indexVz];
            const real rhoVarianceOld = array.data[indexRho];

            computeAndSaveVariance(array.data, vxVarianceOld, indexVx, velocityX, vxMeanOld, vxMeanNew,
                                   numberOfAveragedValues, invCount);
            computeAndSaveVariance(array.data, vyVarianceOld, indexVy, velocityY, vyMeanOld, vyMeanNew,
                                   numberOfAveragedValues, invCount);
            computeAndSaveVariance(array.data, vzVarianceOld, indexVz, velocityZ, vzMeanOld, vzMeanNew,
                                   numberOfAveragedValues, invCount);
            computeAndSaveVariance(array.data, rhoVarianceOld, indexRho, density, rhoMeanOld, rhoMeanNew,
                                   numberOfAveragedValues, invCount);
        }
    }
}

__device__ void calculatePointwiseQuantitiesInTimeseries(uint numberOfAveragedValues, TimeseriesParams timeseriesParams,
                                                         ProbeArray array, uint node, real vx, real vy, real vz, real rho)
{
    const uint currentTimestep = timeseriesParams.lastTimestep + 1;
    const uint nTimesteps = timeseriesParams.numberOfTimesteps;
    const uint lastTimestep = timeseriesParams.lastTimestep;
    const real invCount = c1o1 / real(numberOfAveragedValues + 1);
    const real nPoints = array.numberOfPoints;

    if (array.statistics[int(Statistic::Instantaneous)]) {
        const uint arrayOffset = array.offsets[int(Statistic::Instantaneous)];
        array.data[calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 0)] = vx;
        array.data[calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 1)] = vy;
        array.data[calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 2)] = vz;
        array.data[calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 3)] = rho;
    }

    if (array.statistics[int(Statistic::Means)]) {

        real vxMeanOld, vxMeanNew, vyMeanOld, vyMeanNew, vzMeanOld, vzMeanNew, rhoMeanOld, rhoMeanNew;
        {
            const uint arrayOffset = array.offsets[int(Statistic::Means)];
            const uint indexVx = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 0);
            const uint indexVy = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 1);
            const uint indexVz = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 2);
            const uint indexRho = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 3);

            vxMeanOld = array.data[calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, arrayOffset + 0)];
            vyMeanOld = array.data[calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, arrayOffset + 1)];
            vzMeanOld = array.data[calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, arrayOffset + 2)];
            rhoMeanOld = array.data[calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, arrayOffset + 3)];

            vxMeanNew = computeAndSaveMean(array.data, vxMeanOld, indexVx, vx, invCount);
            vyMeanNew = computeAndSaveMean(array.data, vyMeanOld, indexVy, vy, invCount);
            vzMeanNew = computeAndSaveMean(array.data, vzMeanOld, indexVz, vz, invCount);
            rhoMeanNew = computeAndSaveMean(array.data, rhoMeanOld, indexRho, rho, invCount);
        }

        if (array.statistics[int(Statistic::Variances)]) {
            const uint arrayOffset = array.offsets[int(Statistic::Variances)];
            const uint indexVx = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 0);
            const uint indexVy = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 1);
            const uint indexVz = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 2);
            const uint indexRho = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, arrayOffset + 3);

            const real vxVarianceOld = array.data[calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, arrayOffset + 0)];
            const real vyVarianceOld = array.data[calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, arrayOffset + 1)];
            const real vzVarianceOld = array.data[calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, arrayOffset + 2)];
            const real rhoVarianceOld = array.data[calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, arrayOffset + 3)];

            computeAndSaveVariance(array.data, vxVarianceOld, indexVx, vx, vxMeanOld, vxMeanNew, numberOfAveragedValues,
                                   invCount);
            computeAndSaveVariance(array.data, vyVarianceOld, indexVy, vy, vyMeanOld, vyMeanNew, numberOfAveragedValues,
                                   invCount);
            computeAndSaveVariance(array.data, vzVarianceOld, indexVz, vz, vzMeanOld, vzMeanNew, numberOfAveragedValues,
                                   invCount);
            computeAndSaveVariance(array.data, rhoVarianceOld, indexRho, rho, rhoMeanOld, rhoMeanNew, numberOfAveragedValues,
                                   invCount);
        }
    }
}

__global__ void calculateQuantitiesKernel(uint numberOfAveragedValues, GridParams gridParams, ProbeArray array)
{
    const uint nodeIndex = vf::gpu::getNodeIndex();

    if (nodeIndex >= array.numberOfPoints)
        return;

    const uint gridNodeIndex = gridParams.gridNodeIndices[nodeIndex];

    calculatePointwiseQuantities(numberOfAveragedValues, array, nodeIndex, gridParams.velocityX[gridNodeIndex],
                                 gridParams.velocityY[gridNodeIndex], gridParams.velocityZ[gridNodeIndex],
                                 gridParams.density[gridNodeIndex]);
}

__global__ void interpolateAndCalculateQuantitiesKernel(uint numberOfAveragedValues, GridParams gridParams, ProbeArray array,
                                                        InterpolationParams interpolationParams)
{
    const uint node = vf::gpu::getNodeIndex();

    if (node >= array.numberOfPoints)
        return;

    const uint k_MMM = gridParams.gridNodeIndices[node];

    uint k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP;
    getNeighborIndicesOfBSW(k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, interpolationParams.neighborX,
                            interpolationParams.neighborY, interpolationParams.neighborZ);

    const real dXM = interpolationParams.distanceX[node];
    const real dYM = interpolationParams.distanceY[node];
    const real dZM = interpolationParams.distanceZ[node];

    const real velocityX =
        trilinearInterpolation(dXM, dYM, dZM, k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, gridParams.velocityX);
    const real velocityY =
        trilinearInterpolation(dXM, dYM, dZM, k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, gridParams.velocityY);
    const real velocityZ =
        trilinearInterpolation(dXM, dYM, dZM, k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, gridParams.velocityZ);
    const real density =
        trilinearInterpolation(dXM, dYM, dZM, k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, gridParams.density);

    calculatePointwiseQuantities(numberOfAveragedValues, array, node, velocityX, velocityY, velocityZ, density);
}

__global__ void calculateQuantitiesKernelInTimeseries(uint numberOfAveragedValues, GridParams gridParams, ProbeArray array,
                                                      TimeseriesParams timeseriesParams)
{
    const uint nodeIndex = vf::gpu::getNodeIndex();

    if (nodeIndex >= array.numberOfPoints)
        return;

    const uint gridNodeIndex = gridParams.gridNodeIndices[nodeIndex];

    calculatePointwiseQuantitiesInTimeseries(numberOfAveragedValues, timeseriesParams, array, nodeIndex,
                                             gridParams.velocityX[gridNodeIndex], gridParams.velocityY[gridNodeIndex],
                                             gridParams.velocityZ[gridNodeIndex], gridParams.density[gridNodeIndex]);
}

__global__ void interpolateAndCalculateQuantitiesInTimeseriesKernel(uint numberOfAveragedValues, GridParams gridParams,
                                                                    ProbeArray array,
                                                                    InterpolationParams interpolationParams,
                                                                    TimeseriesParams timeseriesParams)
{
    const uint node = vf::gpu::getNodeIndex();

    if (node >= array.numberOfPoints)
        return;

    const uint k_MMM = gridParams.gridNodeIndices[node];

    uint k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP;
    getNeighborIndicesOfBSW(k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, interpolationParams.neighborX,
                            interpolationParams.neighborY, interpolationParams.neighborZ);

    const real dXM = interpolationParams.distanceX[node];
    const real dYM = interpolationParams.distanceY[node];
    const real dZM = interpolationParams.distanceZ[node];

    const real velocityX =
        trilinearInterpolation(dXM, dYM, dZM, k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, gridParams.velocityX);
    const real velocityY =
        trilinearInterpolation(dXM, dYM, dZM, k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, gridParams.velocityY);
    const real velocityZ =
        trilinearInterpolation(dXM, dYM, dZM, k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, gridParams.velocityZ);
    const real density =
        trilinearInterpolation(dXM, dYM, dZM, k_MMM, k_PMM, k_MPM, k_MMP, k_PPM, k_PMP, k_MPP, k_PPP, gridParams.density);

    calculatePointwiseQuantitiesInTimeseries(numberOfAveragedValues, timeseriesParams, array, node, velocityX, velocityY,
                                             velocityZ, density);
}

bool Probe::getHasDeviceQuantityArray()
{
    return this->hasDeviceQuantityArray;
}

real Probe::getNondimensionalConversionFactor(int level)
{
    return c1o1;
}

void Probe::init()
{
    this->velocityRatio = [this](int level) { return para->getScaledVelocityRatio(level); };
    this->densityRatio = [this](int level) { return para->getScaledDensityRatio(level); };
    this->forceRatio = [this](int level) { return para->getScaledForceRatio(level); };
    this->stressRatio = [this](int level) { return para->getScaledStressRatio(level); };
    this->viscosityRatio = [this](int level) { return para->getScaledViscosityRatio(level); };
    this->nondimensional = [](int level) { return c1o1; };

    probeParams.resize(para->getMaxLevel() + 1);

    for (int level = 0; level <= para->getMaxLevel(); level++) {
        std::vector<int> probeIndices;
        std::vector<real> distX;
        std::vector<real> distY;
        std::vector<real> distZ;
        std::vector<real> pointCoordsX;
        std::vector<real> pointCoordsY;
        std::vector<real> pointCoordsZ;

        this->findPoints(probeIndices, distX, distY, distZ, pointCoordsX, pointCoordsY, pointCoordsZ, level);

        this->addProbeStruct(probeIndices, distX, distY, distZ, pointCoordsX, pointCoordsY, pointCoordsZ, level);

        if (this->outputTimeSeries)
            timeseriesFileNames.push_back(this->writeTimeseriesHeader(level));
    }
}

void Probe::addProbeStruct(std::vector<int>& probeIndices, std::vector<real>& distX, std::vector<real>& distY,
                           std::vector<real>& distZ, std::vector<real>& pointCoordsX, std::vector<real>& pointCoordsY,
                           std::vector<real>& pointCoordsZ, int level)
{
    probeParams[level] = std::make_shared<ProbeStruct>();
    probeParams[level]->nTimesteps = this->getNumberOfTimestepsInTimeseries(level);

    // Note: need to have both nPoints and nIndices because they differ in PlanarAverage
    probeParams[level]->nPoints = uint(pointCoordsX.size());
    probeParams[level]->nIndices = uint(probeIndices.size());

    probeParams[level]->pointCoordsX = (real*)malloc(probeParams[level]->nPoints * sizeof(real));
    probeParams[level]->pointCoordsY = (real*)malloc(probeParams[level]->nPoints * sizeof(real));
    probeParams[level]->pointCoordsZ = (real*)malloc(probeParams[level]->nPoints * sizeof(real));

    std::copy(pointCoordsX.begin(), pointCoordsX.end(), probeParams[level]->pointCoordsX);
    std::copy(pointCoordsY.begin(), pointCoordsY.end(), probeParams[level]->pointCoordsY);
    std::copy(pointCoordsZ.begin(), pointCoordsZ.end(), probeParams[level]->pointCoordsZ);

    // Note, dist only needed for kernels that do interpolate
    if (!distX.empty() && !distY.empty() && !distZ.empty()) {
        probeParams[level]->hasDistances = true;
        cudaMemoryManager->cudaAllocProbeDistances(this, level);
        std::copy(distX.begin(), distX.end(), probeParams[level]->distXH);
        std::copy(distY.begin(), distY.end(), probeParams[level]->distYH);
        std::copy(distZ.begin(), distZ.end(), probeParams[level]->distZH);
        cudaMemoryManager->cudaCopyProbeDistancesHtoD(this, level);
    }

    cudaMemoryManager->cudaAllocProbeIndices(this, level);
    std::copy(probeIndices.begin(), probeIndices.end(), probeParams[level]->pointIndicesH);
    cudaMemoryManager->cudaCopyProbeIndicesHtoD(this, level);

    uint arrOffset = 0;

    cudaMemoryManager->cudaAllocProbeQuantitiesAndOffsets(this, level);

    for (int var = 0; var < int(Statistic::LAST); var++) {
        if (this->quantities[var]) {
            probeParams[level]->quantitiesH[var] = true;
            probeParams[level]->arrayOffsetsH[var] = arrOffset;
            arrOffset += uint(this->getPostProcessingVariables(static_cast<Statistic>(var)).size());
        }
    }

    cudaMemoryManager->cudaCopyProbeQuantitiesAndOffsetsHtoD(this, level);

    probeParams[level]->nArrays = arrOffset;

    cudaMemoryManager->cudaAllocProbeQuantityArray(this, level);

    std::fill_n(probeParams[level]->quantitiesArrayH,
                probeParams[level]->nArrays * probeParams[level]->nPoints * probeParams[level]->nTimesteps, c0o1);

    if (this->hasDeviceQuantityArray)
        cudaMemoryManager->cudaCopyProbeQuantityArrayHtoD(this, level);
}

void Probe::sample(int level, uint t)
{
    const uint t_level = para->getTimeStep(level, t, false);

    SPtr<ProbeStruct> probeStruct = this->getProbeStruct(level);

    //! Skip empty probes
    if (probeStruct->nPoints == 0)
        return;

    //! if tAvg==1 the probe will be evaluated in every sub-timestep of each respective level
    //! else, the probe will only be evaluated in each synchronous time step tAvg

    const uint levelFactor = exp2(level);

    const uint tAvg_level = this->tAvg == 1 ? this->tAvg : this->tAvg * levelFactor;
    const uint tOut_level = this->tOut * levelFactor;
    const uint tStartOut_level = this->tStartOut * levelFactor;
    const uint tStartAvg_level = this->tStartAvg * levelFactor;

    const uint tAfterStartAvg = t_level - tStartAvg_level;
    const uint tAfterStartOut = t_level - tStartOut_level;

    if ((t > this->tStartAvg) && (tAfterStartAvg % tAvg_level == 0)) {
        this->calculateQuantities(probeStruct, t_level, level);

        if (t > this->tStartTmpAveraging)
            probeStruct->numberOfAveragedValues++;
        if (this->outputTimeSeries && (t_level >= tStartOut_level))
            probeStruct->timestepInTimeseries++;
    }

    //! output only in synchronous timesteps
    if ((t > this->tStartOut) && (tAfterStartOut % tOut_level == 0)) {
        if (this->hasDeviceQuantityArray)
            cudaMemoryManager->cudaCopyProbeQuantityArrayDtoH(this, level);
        this->write(level, t);

        if (level == 0 && !this->outputTimeSeries)
            this->writeParallelFile(t);

        if (this->outputTimeSeries) {
            probeStruct->lastTimestepInOldTimeseries =
                probeStruct->timestepInTimeseries > 0 ? probeStruct->timestepInTimeseries - 1 : 0;
            probeStruct->timestepInTimeseries = 0;
        }
    }
}

Probe::~Probe()
{
    for (int level = 0; level <= para->getMaxLevel(); level++) {
        if (this->probeParams[level]->hasDistances)
            cudaMemoryManager->cudaFreeProbeDistances(this, level);
        cudaMemoryManager->cudaFreeProbeIndices(this, level);
        cudaMemoryManager->cudaFreeProbeQuantityArray(this, level);
        cudaMemoryManager->cudaFreeProbeQuantitiesAndOffsets(this, level);
    }
}

void Probe::addStatistic(Statistic variable)
{
    if (!this->isAvailableStatistic(variable))
        throw std::runtime_error("Probe::addStatistic(): Statistic not available for this probe type!");

    this->quantities[int(variable)] = true;
    switch (variable) {
        case Statistic::Variances:
            this->addStatistic(Statistic::Means);
            break;

        default:
            break;
    }
}

void Probe::addAllAvailableStatistics()
{
    for (int var = 0; var < int(Statistic::LAST); var++) {
        if (this->isAvailableStatistic(static_cast<Statistic>(var)))
            this->addStatistic(static_cast<Statistic>(var));
    }
}

void Probe::write(int level, int t)
{
    if (this->outputTimeSeries) {
        this->appendTimeseriesFile(level, t);
    } else {
        const int t_write = this->fileNameLU ? t : t / this->tOut;

        const uint numberOfParts = this->getProbeStruct(level)->nPoints / FilePartCalculator::limitOfNodesForVTK + 1;

        std::vector<std::string> fnames;
        for (uint i = 1; i <= numberOfParts; i++) {
            this->writeGridFile(level, t_write, i);
        }
    }
}

void Probe::writeParallelFile(int t)
{
    const int t_write = this->fileNameLU ? t : t / this->tOut;
    const std::string filename = this->outputPath + makeParallelFileName(probeName, para->getMyProcessID(), t_write);

    std::vector<std::string> nodedatanames = this->getVarNames();
    std::vector<std::string> cellNames;

    getWriter()->writeParallelFile(filename, fileNamesForCollectionFile, nodedatanames, cellNames);

    this->fileNamesForCollectionFile.clear();
}

void Probe::writeGridFile(int level, int t, uint part)
{
    const std::string fname = this->outputPath + makeGridFileName(probeName, level, para->getMyProcessID(), t, part);

    std::vector<UbTupleFloat3> nodes;
    std::vector<std::string> nodedatanames = this->getVarNames();

    std::vector<std::vector<double>> nodedata(nodedatanames.size());

    auto probeStruct = this->getProbeStruct(level);

    const uint startpos = (part - 1) * FilePartCalculator::limitOfNodesForVTK;
    const uint sizeOfNodes = std::min(FilePartCalculator::limitOfNodesForVTK, probeStruct->nPoints - startpos);
    const uint endpos = startpos + sizeOfNodes;

    //////////////////////////////////////////////////////////////////////////
    nodes.resize(sizeOfNodes);

    for (uint pos = startpos; pos < endpos; pos++) {
        nodes[pos - startpos] = makeUbTuple(float(probeStruct->pointCoordsX[pos]), float(probeStruct->pointCoordsY[pos]),
                                            float(probeStruct->pointCoordsZ[pos]));
    }

    for (auto it = nodedata.begin(); it != nodedata.end(); it++)
        it->resize(sizeOfNodes);

    const uint arrayLength = probeStruct->nPoints;
    const int nTimesteps = probeStruct->nTimesteps;
    const int timestep = probeStruct->timestepInTimeseries;

    for (int statistic = 0; statistic < int(Statistic::LAST); statistic++) {
        if (!this->quantities[statistic])
            continue;

        std::vector<PostProcessingVariable> postProcessingVariables = this->getPostProcessingVariables(statistic);

        const uint statisticOffset = probeStruct->arrayOffsetsH[statistic];

        for (uint arr = 0; arr < uint(postProcessingVariables.size()); arr++) {
            const real coeff = postProcessingVariables[arr].conversionFactor(level);
            const int arrayIndex = statisticOffset + arr;
            const int startIndex = calcArrayIndex(startpos, arrayLength, timestep, nTimesteps, arrayIndex);
            for (uint idx = 0; idx < endpos - startpos; idx++) {
                nodedata[arrayIndex][idx] = double(probeStruct->quantitiesArrayH[startIndex + idx] * coeff);
            }
        }
    }
    std::string fullName = getWriter()->writeNodesWithNodeData(fname, nodes, nodedatanames, nodedata);
    this->fileNamesForCollectionFile.push_back(fullName.substr(fullName.find_last_of('/') + 1));
}

std::string Probe::writeTimeseriesHeader(int level)
{
    /*
    File Layout:
    TimeseriesOutput
    Quantities: Quant1 Quant2 Quant3
    Positions:
    point1.x, point1.y, point1.z
    point2.x, point2.y, point2.z
    ...
    t0 point1.quant1 point2.quant1 ... point1.quant2 point2.quant2 ...
    t1 point1.quant1 point2.quant1 ... point1.quant2 point2.quant2 ...
    */
    auto probeStruct = this->getProbeStruct(level);
    std::filesystem::create_directories(this->outputPath);
    const std::string fname = this->outputPath + makeTimeseriesFileName(probeName, level, para->getMyProcessID());
    std::ofstream out(fname.c_str(), std::ios::out | std::ios::binary);

    if (!out.is_open())
        throw std::runtime_error("Could not open timeseries file " + fname + "!");

    out << "TimeseriesOutput \n";
    out << "Quantities: ";
    for (const std::string& name : getVarNames())
        out << name << ", ";
    out << "\n";
    out << "Number of points in this file: \n";
    out << probeStruct->nPoints << "\n";
    out << "Positions: x, y, z\n";
    for (uint i = 0; i < probeStruct->nPoints; i++)
        out << probeStruct->pointCoordsX[i] << ", " << probeStruct->pointCoordsY[i] << ", " << probeStruct->pointCoordsZ[i]
            << "\n";

    out.close();

    return fname;
}

std::vector<real> Probe::getTimestepData(real time, uint numberOfValues, int timestep, ProbeStruct* probeStruct, int level)
{
    std::vector<real> timestepData;
    timestepData.resize(numberOfValues+1);
    timestepData[0] = time;

    int valueIndex = 1;

    for (int statistic = 0; statistic < int(Statistic::LAST); statistic++) {
        if (!this->quantities[statistic])
            continue;

        std::vector<PostProcessingVariable> variables = this->getPostProcessingVariables(statistic);
        const uint offsetStatistic = probeStruct->arrayOffsetsH[statistic];

        for (uint variable = 0; variable < uint(variables.size()); variable++) {
            const real conversionFactor = variables[variable].conversionFactor(level);
            const real variableIndex = offsetStatistic + variable;
            const uint startIndex =
                calcArrayIndex(0, probeStruct->nPoints, timestep, probeStruct->nTimesteps, variableIndex);

            for (uint point = 0; point < probeStruct->nPoints; point++) {
                timestepData[valueIndex + point] = probeStruct->quantitiesArrayH[startIndex + point] * conversionFactor;
            }
            valueIndex += probeStruct->nPoints;
        }
    }
    return timestepData;
}

void Probe::appendTimeseriesFile(int level, int t)
{
    std::ofstream out(this->timeseriesFileNames[level], std::ios::app | std::ios::binary);

    const uint tAvg_level = this->tAvg == 1 ? this->tAvg : this->tAvg * exp2(-level);
    const real deltaT = para->getTimeRatio() * tAvg_level;
    auto probeStruct = this->getProbeStruct(level).get();

    const real tStart = (t - this->tOut) * para->getTimeRatio();

    const int valuesPerTimestep = probeStruct->nPoints * probeStruct->nArrays;

    for (uint timestep = 0; timestep < probeStruct->timestepInTimeseries; timestep++) {
        const real time = tStart + timestep * deltaT;
        std::vector<real> timestepData = this->getTimestepData(time, valuesPerTimestep, timestep, probeStruct, level);
        out.write((char*)timestepData.data(), sizeof(real) * valuesPerTimestep);
    }
    out.close();
}

std::vector<std::string> Probe::getVarNames()
{
    std::vector<std::string> varNames;
    for (int statistic = 0; statistic < int(Statistic::LAST); statistic++) {
        if (!this->quantities[statistic])
            continue;
        std::vector<PostProcessingVariable> postProcessingVariables = this->getPostProcessingVariables(statistic);
        for (size_t i = 0; i < postProcessingVariables.size(); i++)
            varNames.push_back(postProcessingVariables[i].name);
    }
    return varNames;
}

GridParams getGridParams(ProbeStruct* probeStruct, LBMSimulationParameter* para)
{
    return GridParams {
        probeStruct->pointIndicesD, para->velocityX, para->velocityY, para->velocityZ, para->rho,
    };
}

InterpolationParams getInterpolationParams(ProbeStruct* probeStruct, LBMSimulationParameter* para)
{
    return InterpolationParams { probeStruct->distXD, probeStruct->distYD, probeStruct->distZD,
                                 para->neighborX,     para->neighborY,     para->neighborZ };
}

TimeseriesParams getTimeseriesParams(ProbeStruct* probeStruct)
{
    return TimeseriesParams {
        calcOldTimestep(probeStruct->timestepInTimeseries, probeStruct->lastTimestepInOldTimeseries),
        probeStruct->nTimesteps,
    };
}

ProbeArray getProbeArray(ProbeStruct* probeStruct)
{
    return ProbeArray { probeStruct->quantitiesArrayD, probeStruct->arrayOffsetsD, probeStruct->quantitiesD,
                        probeStruct->nPoints };
}

template <typename T>
std::string nameComponent(std::string name, T value)
{
    return "_" + name + "_" + StringUtil::toString<T>(value);
}

std::string makeParallelFileName(const std::string probeName, int id, int t)
{
    return probeName + "_bin" + nameComponent("ID", id) + nameComponent("t", t) + ".vtk";
}

std::string makeGridFileName(const std::string probeName, int level, int id, int t, uint part)
{
    return probeName + "_bin" + nameComponent("lev", level) + nameComponent("ID", id) + nameComponent<int>("Part", part) +
           nameComponent("t", t) + ".vtk";
}

std::string makeTimeseriesFileName(const std::string probeName, int level, int id)
{
    return probeName + "_timeseries" + nameComponent("lev", level) + nameComponent("ID", id) + ".txt";
}
//! \}
