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

#include "TimeseriesFileWriter.h"
#include "Utilities.h"
#include "cuda_helper/CudaGrid.h"
#include "gpu/core/Cuda/CudaMemoryManager.h"
#include "gpu/core/DataStructureInitializer/GridProvider.h"
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

real* getStatisticArray(Probe::ProbeData probeData, Probe::Statistic statistic)
{
    switch (statistic) {
        case Probe::Statistic::Instantaneous:
            return probeData.instantaneous;
        case Probe::Statistic::Means:
            return probeData.means;
        case Probe::Statistic::Variances:
            return probeData.variances;
        default:
            throw std::runtime_error("getStatisticArray: Statistic unavailable!");
    }
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

__device__ void calculatePointwiseQuantities(uint numberOfAveragedValues, Probe::ProbeData probeData, uint node, real velocityX,
                                             real velocityY, real velocityZ, real density)
{
    //"https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    // also has extensions for higher order and covariances
    const real invCount = c1o1 / real(numberOfAveragedValues + 1);
    const uint nPoints = probeData.numberOfPoints;
    const uint indexVx = calcArrayIndex(node, nPoints, 0);
    const uint indexVy = calcArrayIndex(node, nPoints, 1);
    const uint indexVz = calcArrayIndex(node, nPoints, 2);
    const uint indexRho = calcArrayIndex(node, nPoints, 3);

    if (probeData.computeInstantaneoues) {
        probeData.instantaneous[calcArrayIndex(node, nPoints, 0)] = velocityX;
        probeData.instantaneous[calcArrayIndex(node, nPoints, 1)] = velocityY;
        probeData.instantaneous[calcArrayIndex(node, nPoints, 2)] = velocityZ;
        probeData.instantaneous[calcArrayIndex(node, nPoints, 3)] = density;
    }

    if (probeData.computeMean) {

        real vxMeanOld, vxMeanNew, vyMeanOld, vyMeanNew, vzMeanOld, vzMeanNew, rhoMeanOld, rhoMeanNew;
        {
            vxMeanOld = probeData.means[indexVx];
            vyMeanOld = probeData.means[indexVy];
            vzMeanOld = probeData.means[indexVz];
            rhoMeanOld = probeData.means[indexRho];

            vxMeanNew = computeAndSaveMean(probeData.means, vxMeanOld, indexVx, velocityX, invCount);
            vyMeanNew = computeAndSaveMean(probeData.means, vyMeanOld, indexVy, velocityY, invCount);
            vzMeanNew = computeAndSaveMean(probeData.means, vzMeanOld, indexVz, velocityZ, invCount);
            rhoMeanNew = computeAndSaveMean(probeData.means, rhoMeanOld, indexRho, density, invCount);
        }

        if (probeData.computeVariance) {

            const real vxVarianceOld = probeData.variances[indexVx];
            const real vyVarianceOld = probeData.variances[indexVy];
            const real vzVarianceOld = probeData.variances[indexVz];
            const real rhoVarianceOld = probeData.variances[indexRho];

            computeAndSaveVariance(probeData.variances, vxVarianceOld, indexVx, velocityX, vxMeanOld, vxMeanNew,
                                   numberOfAveragedValues, invCount);
            computeAndSaveVariance(probeData.variances, vyVarianceOld, indexVy, velocityY, vyMeanOld, vyMeanNew,
                                   numberOfAveragedValues, invCount);
            computeAndSaveVariance(probeData.variances, vzVarianceOld, indexVz, velocityZ, vzMeanOld, vzMeanNew,
                                   numberOfAveragedValues, invCount);
            computeAndSaveVariance(probeData.variances, rhoVarianceOld, indexRho, density, rhoMeanOld, rhoMeanNew,
                                   numberOfAveragedValues, invCount);
        }
    }
}

__device__ void calculatePointwiseQuantitiesInTimeseries(uint numberOfAveragedValues, Probe::TimeseriesParams timeseriesParams,
                                                         Probe::ProbeData probeData, uint node, real vx, real vy, real vz, real rho)
{
    const uint currentTimestep = timeseriesParams.lastTimestep + 1;
    const uint nTimesteps = probeData.numberOfTimesteps;
    const uint lastTimestep = timeseriesParams.lastTimestep;
    const real invCount = c1o1 / real(numberOfAveragedValues + 1);
    const real nPoints = probeData.numberOfPoints;
    const uint indexVxCurrent = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, 0);
    const uint indexVyCurrent = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, 1);
    const uint indexVzCurrent = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, 2);
    const uint indexRhoCurrent = calcArrayIndex(node, nPoints, currentTimestep, nTimesteps, 3);

    const uint indexVxLast = calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, 0);
    const uint indexVyLast = calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, 1);
    const uint indexVzLast = calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, 2);
    const uint indexRhoLast = calcArrayIndex(node, nPoints, lastTimestep, nTimesteps, 3);

    if (probeData.computeInstantaneoues) {
        probeData.instantaneous[indexVxCurrent] = vx;
        probeData.instantaneous[indexVyCurrent] = vy;
        probeData.instantaneous[indexVzCurrent] = vz;
        probeData.instantaneous[indexRhoCurrent] = rho;
    }

    if (probeData.computeMean) {

        real vxMeanOld, vxMeanNew, vyMeanOld, vyMeanNew, vzMeanOld, vzMeanNew, rhoMeanOld, rhoMeanNew;
        {

            vxMeanOld = probeData.means[indexVxLast];
            vyMeanOld = probeData.means[indexVyLast];
            vzMeanOld = probeData.means[indexVzLast];
            rhoMeanOld = probeData.means[indexRhoLast];

            vxMeanNew = computeAndSaveMean(probeData.means, vxMeanOld, indexVxCurrent, vx, invCount);
            vyMeanNew = computeAndSaveMean(probeData.means, vyMeanOld, indexVyCurrent, vy, invCount);
            vzMeanNew = computeAndSaveMean(probeData.means, vzMeanOld, indexVzCurrent, vz, invCount);
            rhoMeanNew = computeAndSaveMean(probeData.means, rhoMeanOld, indexRhoCurrent, rho, invCount);
        }

        if (probeData.computeVariance) {

            const real vxVarianceOld = probeData.variances[indexVxLast];
            const real vyVarianceOld = probeData.variances[indexVyLast];
            const real vzVarianceOld = probeData.variances[indexVzLast];
            const real rhoVarianceOld = probeData.variances[indexRhoLast];

            computeAndSaveVariance(probeData.variances, vxVarianceOld, indexVxCurrent, vx, vxMeanOld, vxMeanNew,
                                   numberOfAveragedValues, invCount);
            computeAndSaveVariance(probeData.variances, vyVarianceOld, indexVxCurrent, vy, vyMeanOld, vyMeanNew,
                                   numberOfAveragedValues, invCount);
            computeAndSaveVariance(probeData.variances, vzVarianceOld, indexVxCurrent, vz, vzMeanOld, vzMeanNew,
                                   numberOfAveragedValues, invCount);
            computeAndSaveVariance(probeData.variances, rhoVarianceOld, indexRhoCurrent, rho, rhoMeanOld, rhoMeanNew,
                                   numberOfAveragedValues, invCount);
        }
    }
}

__global__ void calculateQuantitiesKernel(uint numberOfAveragedValues, Probe::GridParams gridParams, Probe::ProbeData probeData)
{
    const uint nodeIndex = vf::gpu::getNodeIndex();

    if (nodeIndex >= probeData.numberOfPoints)
        return;

    const uint gridNodeIndex = probeData.indices[nodeIndex];

    calculatePointwiseQuantities(numberOfAveragedValues, probeData, nodeIndex, gridParams.velocityX[gridNodeIndex],
                                 gridParams.velocityY[gridNodeIndex], gridParams.velocityZ[gridNodeIndex],
                                 gridParams.density[gridNodeIndex]);
}

__global__ void calculateQuantitiesKernelInTimeseries(uint numberOfAveragedValues, Probe::GridParams gridParams,
                                                      Probe::ProbeData probeData, Probe::TimeseriesParams timeseriesParams)
{
    const uint nodeIndex = vf::gpu::getNodeIndex();

    if (nodeIndex >= probeData.numberOfPoints)
        return;

    const uint gridNodeIndex = probeData.indices[nodeIndex];

    calculatePointwiseQuantitiesInTimeseries(numberOfAveragedValues, timeseriesParams, probeData, nodeIndex,
                                             gridParams.velocityX[gridNodeIndex], gridParams.velocityY[gridNodeIndex],
                                             gridParams.velocityZ[gridNodeIndex], gridParams.density[gridNodeIndex]);
}


std::vector<Probe::PostProcessingVariable> Probe::getPostProcessingVariables(Statistic statistic, int level)
{
    const real velocityRatio = para->getScaledVelocityRatio(level);
    const real stressRatio = para->getScaledStressRatio(level);
    const real densityRatio = para->getScaledDensityRatio(level);
    std::vector<PostProcessingVariable> postProcessingVariables;
    switch (statistic) {
        case Statistic::Instantaneous:
            postProcessingVariables.emplace_back("vx", velocityRatio);
            postProcessingVariables.emplace_back("vy", velocityRatio);
            postProcessingVariables.emplace_back("vz", velocityRatio);
            postProcessingVariables.emplace_back("rho", densityRatio);
            break;
        case Statistic::Means:
            postProcessingVariables.emplace_back("vx_mean", velocityRatio);
            postProcessingVariables.emplace_back("vy_mean", velocityRatio);
            postProcessingVariables.emplace_back("vz_mean", velocityRatio);
            postProcessingVariables.emplace_back("rho_mean", densityRatio);
            break;
        case Statistic::Variances:
            postProcessingVariables.emplace_back("vx_var", stressRatio);
            postProcessingVariables.emplace_back("vy_var", stressRatio);
            postProcessingVariables.emplace_back("vz_var", stressRatio);
            postProcessingVariables.emplace_back("rho_var", densityRatio);
            break;

        default:
            throw std::runtime_error("Probe::getPostProcessingVariables: Statistic unavailable!");
            break;
    }
    return postProcessingVariables;
}

std::vector<Probe::PostProcessingVariable> Probe::getAllPostProcessingVariables(int level)
{
    std::vector<PostProcessingVariable> postProcessingVariables;
    if (enableComputationInstantaneous) {
        auto instantPostProcessingVariables = getPostProcessingVariables(Statistic::Instantaneous, level);
        postProcessingVariables.insert(postProcessingVariables.end(), instantPostProcessingVariables.begin(),
                                       instantPostProcessingVariables.end());
    }
    if (enableComputationMeans) {
        auto meanPostProcessingVariables = getPostProcessingVariables(Statistic::Means, level);
        postProcessingVariables.insert(postProcessingVariables.end(), meanPostProcessingVariables.begin(),
                                       meanPostProcessingVariables.end());
    }
    if (enableComputationVariances) {
        auto varPostProcessingVariables = getPostProcessingVariables(Statistic::Variances, level);
        postProcessingVariables.insert(postProcessingVariables.end(), varPostProcessingVariables.begin(),
                                       varPostProcessingVariables.end());
    }
    return postProcessingVariables;
}

void Probe::init()
{
    for (int level = 0; level <= para->getMaxLevel(); level++) {
        this->addLevelData(level);

        if (this->outputTimeSeries) {
            auto levelData = levelDatas[level];
            const std::string fileName = makeTimeseriesFileName(probeName, level, para->getMyProcessID());
            auto variableNames = getVarNames();
            TimeseriesFileWriter::writeHeader(fileName, levelData.probeDataH.numberOfPoints, variableNames,
                                              levelData.coordinatesX.data(), levelData.coordinatesY.data(),
                                              levelData.coordinatesZ.data());
            timeseriesFileNames.push_back(fileName);
        }
    }
}


void Probe::addLevelData(int level)
{
    std::vector<uint> indices;
    auto levelData = levelDatas.emplace_back();
    const real* coordinateX = para->getParH(level)->coordinateX;
    const real* coordinateY = para->getParH(level)->coordinateY;
    const real* coordinateZ = para->getParH(level)->coordinateZ;
    const real deltaX = coordinateX[para->getParH(level)->neighborX[1]] - coordinateX[1];
    for (unsigned long long pos = 1; pos < para->getParH(level)->numberOfNodes; pos++) {
        const real pointCoordX = coordinateX[pos];
        const real pointCoordY = coordinateY[pos];
        const real pointCoordZ = coordinateZ[pos];
        for (auto point : points) {
            const real distX = point.x - pointCoordX;
            const real distY = point.y - pointCoordY;
            const real distZ = point.z - pointCoordZ;
            if (distX <= deltaX && distY <= deltaX && distZ <= deltaX && distX > c0o1 && distY > c0o1 && distZ > c0o1 &&
                isValidProbePoint(pos, para.get(), level)) {
                indices.push_back(static_cast<uint>(pos));
                levelData.coordinatesX.push_back(pointCoordX);
                levelData.coordinatesY.push_back(pointCoordX);
                levelData.coordinatesZ.push_back(pointCoordX);
                continue;
            }
        }
        for (auto plane : planes) {
            const real distanceX = pointCoordX - plane.startX;
            const real distanceY = pointCoordY - plane.startY;
            const real distanceZ = pointCoordZ - plane.startZ;

            if (distanceX <= plane.length && distanceY <= plane.width && distanceZ <= plane.height && distanceX >= c0o1 &&
                distanceY >= c0o1 && distanceZ >= c0o1 && isValidProbePoint(pos, para.get(), level)) {
                indices.push_back(static_cast<uint>(pos));
                levelData.coordinatesX.push_back(pointCoordX);
                levelData.coordinatesY.push_back(pointCoordY);
                levelData.coordinatesZ.push_back(pointCoordZ);
                continue;
            }
        }
    }
    makeProbeData(indices, levelData.probeDataH, levelData.probeDataD, level);
}

void Probe::makeProbeData(std::vector<uint>& indices, ProbeData& probeDataH, ProbeData& probeDataD, int level)
{
    probeDataH.numberOfPoints = static_cast<uint>(indices.size());
    probeDataD.numberOfPoints = static_cast<uint>(indices.size());
    probeDataH.numberOfTimesteps = getNumberOfTimestepsInTimeseries(level);
    probeDataD.numberOfTimesteps = getNumberOfTimestepsInTimeseries(level);
    probeDataH.computeInstantaneoues = enableComputationInstantaneous;
    probeDataH.computeMean = enableComputationMeans;
    probeDataH.computeVariance = enableComputationVariances;
    probeDataD.computeInstantaneoues = enableComputationInstantaneous;
    probeDataD.computeMean = enableComputationMeans;
    probeDataD.computeVariance = enableComputationVariances;

    cudaMemoryManager->cudaAllocProbeData(this, level);

    std::copy(indices.begin(), indices.end(), probeDataH.indices);

    if (probeDataH.computeInstantaneoues)
        std::fill_n(probeDataH.instantaneous, probeDataH.numberOfPoints * probeDataH.numberOfPoints, c0o1);
    if (probeDataH.computeMean)
        std::fill_n(probeDataH.means, probeDataH.numberOfPoints * probeDataH.numberOfPoints, c0o1);
    if (probeDataH.computeVariance)
        std::fill_n(probeDataH.variances, probeDataH.numberOfPoints * probeDataH.numberOfPoints, c0o1);

    cudaMemoryManager->cudaCopyProbeDataHtoD(this, level);
}

void Probe::sample(int level, uint t)
{
    const uint tLevel = para->getTimeStep(level, t, false);

    //! if tBetweenAverages==1 the probe will be evaluated in every sub-timestep of each respective level
    //! else, the probe will only be evaluated in each synchronous time step tBetweenAverages

    const uint levelFactor = exp2(level);

    const uint tAvgLevel = this->tBetweenAverages == 1 ? this->tBetweenAverages : this->tBetweenAverages * levelFactor;
    const uint tOutLevel = this->tBetweenWriting * levelFactor;
    const uint tStartOutLevel = this->tStartWritingOutput * levelFactor;
    const uint tStartAvgLevel = this->tStartAveraging * levelFactor;

    const uint tAfterStartAvg = tLevel - tStartAvgLevel;
    const uint tAfterStartOut = tLevel - tStartOutLevel;

    auto levelData = &levelDatas[level];

    auto gridParams = getGridParams(para->getParD(level).get());

    const vf::cuda::CudaGrid grid(para->getParD(level)->numberofthreads, levelData->probeDataD.numberOfPoints);

    if ((t > this->tStartAveraging) && (tAfterStartAvg % tAvgLevel == 0)) {
        if (outputTimeSeries) {
            calculateQuantitiesKernelInTimeseries<<<grid.grid, grid.threads>>>(
                levelData->numberOfAveragedValues, gridParams, levelData->probeDataD, levelData->timeseriesParams);
            if (tLevel >= tStartOutLevel)
                levelData->timeseriesParams.lastTimestep = calcOldTimestep(
                    levelData->timeseriesParams.lastTimestep, levelData->timeseriesParams.lastTimestepInOldTimeseries);
        } else {
            calculateQuantitiesKernel<<<grid.grid, grid.threads>>>(levelData->numberOfAveragedValues, gridParams,
                                                                   levelData->probeDataD);
        }

        //! output only in synchronous timesteps
        if ((t > this->tStartWritingOutput) && (tAfterStartOut % tOutLevel == 0)) {

            cudaMemoryManager->cudaCopyProbeDataDtoH(this, level);
            if (outputTimeSeries) {
                this->appendTimeseriesFile(level, t);
                levelData->timeseriesParams.lastTimestepInOldTimeseries =
                    levelData->timeseriesParams.lastTimestep > 0 ? levelData->timeseriesParams.lastTimestep - 1 : 0;
                levelData->timeseriesParams.lastTimestep = 0;
            } else {
                this->writeGridFiles(level, t);
                if (level == 0)
                    this->writeParallelFile(t);
            }
        }
    }
}

Probe::~Probe()
{
    for (int level = 0; level <= para->getMaxLevel(); level++) {
        cudaMemoryManager->cudaFreeProbeData(this, level);
    }
}

void Probe::addStatistic(Statistic variable)
{
    switch (variable) {
        case Statistic::Instantaneous:
            enableComputationInstantaneous = true;
            break;
        case Statistic::Means:
            enableComputationMeans = true;
            break;
        case Statistic::Variances:
            enableComputationVariances = true;
            enableComputationMeans = true;
            break;
        default:
            throw std::runtime_error("Probe::addStatistic: Statistic unavailable!");
            break;
    }
}

void Probe::writeGridFiles(int level, int t)
{
    const int tWrite = this->fileNameLU ? t : (t - tStartWritingOutput) / this->tBetweenWriting;

    const uint numberOfParts = levelDatas[level].probeDataH.numberOfPoints / FilePartCalculator::limitOfNodesForVTK + 1;

    std::vector<std::string> fnames;
    for (uint i = 1; i <= numberOfParts; i++) {
        this->writeGridFile(level, tWrite, i);
    }
}

void Probe::writeParallelFile(int t)
{
    const int t_write = this->fileNameLU ? t : t / this->tBetweenWriting;
    const std::string filename = this->outputPath + makeParallelFileName(probeName, para->getMyProcessID(), t_write);

    std::vector<std::string> nodedatanames = this->getVarNames();
    std::vector<std::string> cellNames;

    getWriter()->writeParallelFile(filename, fileNamesForCollectionFile, nodedatanames, cellNames);

    this->fileNamesForCollectionFile.clear();
}

void Probe::appendStatisticToNodeData(Statistic statistic, uint startPos, uint endPos, uint timestep, int level,
                                      std::vector<std::vector<double>>& nodedata)
{
    auto levelData = &levelDatas[level];
    const uint numberOfNodes = levelData->probeDataH.numberOfPoints;
    const real* data = getStatisticArray(levelData->probeDataH, statistic);
    std::vector<PostProcessingVariable> postProcessingVariables = this->getPostProcessingVariables(statistic, level);
    for (uint arr = 0; arr < uint(postProcessingVariables.size()); arr++) {
        std::vector<double> quantityData(numberOfNodes);
        const real coeff = postProcessingVariables[arr].conversionFactor;
        const int startIndex =
            calcArrayIndex(startPos, numberOfNodes, timestep, levelData->probeDataH.numberOfTimesteps, arr);
        for (uint idx = 0; idx < endPos - startPos; idx++) {
            quantityData[idx] = double(data[startIndex + idx] * coeff);
        }
        nodedata.push_back(quantityData);
    }
}

void Probe::writeGridFile(int level, int t, uint part)
{
    const std::string fname = this->outputPath + makeGridFileName(probeName, level, para->getMyProcessID(), t, part);

    std::vector<UbTupleFloat3> nodes;
    std::vector<std::string> nodedatanames = this->getVarNames();

    std::vector<std::vector<double>> nodedata;

    auto levelData = &levelDatas[level];

    const uint startpos = (part - 1) * FilePartCalculator::limitOfNodesForVTK;
    const uint sizeOfNodes =
        std::min(FilePartCalculator::limitOfNodesForVTK, levelData->probeDataH.numberOfPoints - startpos);
    const uint endpos = startpos + sizeOfNodes;

    //////////////////////////////////////////////////////////////////////////
    nodes.resize(sizeOfNodes);

    for (uint pos = startpos; pos < endpos; pos++) {
        nodes[pos - startpos] = makeUbTuple(float(levelData->coordinatesX[pos]), float(levelData->coordinatesY[pos]),
                                            float(levelData->coordinatesZ[pos]));
    }

    if (enableComputationInstantaneous)
        appendStatisticToNodeData(Statistic::Instantaneous, startpos, endpos, 0, level, nodedata);
    if (enableComputationMeans)
        appendStatisticToNodeData(Statistic::Means, startpos, endpos, level, 0, nodedata);
    if (enableComputationVariances)
        appendStatisticToNodeData(Statistic::Variances, startpos, endpos, level, 0, nodedata);
    std::string fullName = getWriter()->writeNodesWithNodeData(fname, nodes, nodedatanames, nodedata);
    this->fileNamesForCollectionFile.push_back(fullName.substr(fullName.find_last_of('/') + 1));
}

void Probe::appendStatisticToTimestepData(int timestep, std::vector<real>& timestepData, Statistic statistic, int level)
{
    std::vector<PostProcessingVariable> variables = this->getPostProcessingVariables(statistic, level);
    auto probeData = levelDatas[level].probeDataH;
    const real* data = getStatisticArray(probeData, statistic);

    for (uint variable = 0; variable < uint(variables.size()); variable++) {
        const real conversionFactor = variables[variable].conversionFactor;
        const uint startIndex = calcArrayIndex(0, probeData.numberOfPoints, timestep, probeData.numberOfTimesteps, variable);

        for (uint point = 0; point < probeData.numberOfPoints; point++) {
            timestepData.push_back(data[startIndex + point] * conversionFactor);
        }
    }
}

std::vector<real> Probe::getTimestepData(real time, int timestep, int level)
{
    std::vector<real> timestepData;
    timestepData.push_back(time);

    if (enableComputationInstantaneous)
        appendStatisticToTimestepData(timestep, timestepData, Statistic::Instantaneous, level);
    if (enableComputationMeans)
        appendStatisticToTimestepData(timestep, timestepData, Statistic::Means, level);
    if (enableComputationVariances)
        appendStatisticToTimestepData(timestep, timestepData, Statistic::Variances, level);
    return timestepData;
}

void Probe::appendTimeseriesFile(int level, int t)
{
    std::ofstream out(this->timeseriesFileNames[level], std::ios::app | std::ios::binary);

    const uint tAvg_level = this->tBetweenAverages == 1 ? this->tBetweenAverages : this->tBetweenAverages * exp2(-level);
    const real deltaT = para->getTimeRatio() * tAvg_level;
    auto levelData = levelDatas[level];

    const real tStart = (t - this->tBetweenWriting) * para->getTimeRatio();

    for (uint timestep = 0; timestep < levelData.timeseriesParams.lastTimestep; timestep++) {
        const real time = tStart + timestep * deltaT;
        std::vector<real> timestepData = this->getTimestepData(time, timestep, level);
        out.write((char*)timestepData.data(), sizeof(real) * timestepData.size());
    }
    out.close();
}

std::vector<std::string> Probe::getVarNames()
{
    std::vector<std::string> varNames;
    for (auto variable : getAllPostProcessingVariables(0))
        varNames.push_back(variable.name);
    return varNames;
}

void Probe::getTaggedFluidNodes(GridProvider* gridProvider)
{
    for (int level = 0; level <= para->getMaxLevel(); level++) {
        auto probeData = levelDatas[level].probeDataH;
        std::vector<uint> probeIndices(probeData.indices, probeData.indices + probeData.numberOfPoints);
        gridProvider->tagFluidNodeIndices(probeIndices, CollisionTemplate::WriteMacroVars, level);
    }
}

Probe::GridParams Probe::getGridParams(LBMSimulationParameter* para)
{
    return {
        para->velocityX,
        para->velocityY,
        para->velocityZ,
        para->rho,
    };
}

bool isCoarseInterpolationCell(unsigned long long pointIndex, Parameter* para, int level)
{
    if (level == para->getMaxLevel())
        return false;
    auto interpolationCells = para->getParH(level)->fineToCoarse;
    for (uint i = 0; i < interpolationCells.numberOfCells; i++) {
        if (interpolationCells.coarseCellIndices[i] == pointIndex) {
            return true;
        }
    }
    return false;
}

bool isFineInterpolationCell(unsigned long long pointIndex, Parameter* para, int level)
{
    if (level == 0)
        return false;
    auto interpolationCells = para->getParH(level - 1)->coarseToFine;
    const uint* neighborX = para->getParH(level)->neighborX;
    const uint* neighborY = para->getParH(level)->neighborY;
    const uint* neighborZ = para->getParH(level)->neighborZ;
    for (uint i = 0; i < interpolationCells.numberOfCells; i++) {
        const uint kMMM = interpolationCells.fineCellIndices[i];
        uint kPMM, kMPM, kMMP, kPPM, kPMP, kMPP, kPPP;
        getNeighborIndicesOfBSW(kMMM, kPMM, kMPM, kMMP, kPPM, kPMP, kMPP, kPPP, neighborX, neighborY, neighborZ);
        if (kMMM == pointIndex || kPMM == pointIndex || kMPM == pointIndex || kMMP == pointIndex || kPPM == pointIndex ||
            kPMP == pointIndex || kMPP == pointIndex || kPPP == pointIndex) {
            return true;
        }
    }
    return false;
}

bool isValidProbePoint(unsigned long long pointIndex, Parameter* para, int level)
{
    return GEO_FLUID == para->getParH(level)->typeOfGridNode[pointIndex] &&
           !isCoarseInterpolationCell(pointIndex, para, level) && !isFineInterpolationCell(pointIndex, para, level);
}
//! \}
