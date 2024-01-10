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

uint calcOldTimestep(uint currentTimestep, uint lastTimestepInOldSeries)
{
    return currentTimestep > 0 ? currentTimestep - 1 : lastTimestepInOldSeries;
}

__host__ __device__ real computeMean(real oldMean, real newValue, real inverseNumberOfValues)
{
    return oldMean + (newValue - oldMean) * inverseNumberOfValues;
}

__host__ __device__ real computeVariance(real oldVariance, real oldMean, real newMean, real newValue, uint numberOfValuesM1,
                                         real inverseNumberOfValues)
{
    return numberOfValuesM1 * oldVariance + (newValue - oldMean) * (newValue - newMean) * inverseNumberOfValues;
}

__device__ void calculatePointwiseQuantities(uint oldTimestepInTimeseries, uint timestepInTimeseries, uint timestepInAverage,
                                             uint nTimesteps, real* quantityArray, bool* quantities,
                                             uint* quantityArrayOffsets, uint nPoints, uint node, real vx, real vy, real vz,
                                             real rho)
{
    //"https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    // also has extensions for higher order and covariances
    const real invNumberOfTimestepsInAvg = 1 / real(timestepInAverage + 1);

    if (quantities[int(Statistic::Instantaneous)]) {
        const uint arrOff = quantityArrayOffsets[int(Statistic::Instantaneous)];
        quantityArray[calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrOff + 0)] = vx;
        quantityArray[calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrOff + 1)] = vy;
        quantityArray[calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrOff + 2)] = vz;
        quantityArray[calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrOff + 3)] = rho;
    }

    if (quantities[int(Statistic::Means)]) {
        const uint arrayOffsetMean = quantityArrayOffsets[int(Statistic::Means)];
        const uint indexMeanVX = calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrayOffsetMean + 0);
        const uint indexMeanVY = calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrayOffsetMean + 1);
        const uint indexMeanVZ = calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrayOffsetMean + 2);
        const uint indexMeanRho = calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrayOffsetMean + 3);

        const real vxMeanOld = quantityArray[indexMeanVX];
        const real vyMeanOld = quantityArray[indexMeanVY];
        const real vzMeanOld = quantityArray[indexMeanVZ];
        const real rhoMeanOld = quantityArray[indexMeanRho];

        const real vxMeanNew = computeMean(vxMeanOld, vx, invNumberOfTimestepsInAvg);
        const real vyMeanNew = computeMean(vyMeanOld, vy, invNumberOfTimestepsInAvg);
        const real vzMeanNew = computeMean(vzMeanOld, vz, invNumberOfTimestepsInAvg);
        const real rhoMeanNew = computeMean(rhoMeanOld, rho, invNumberOfTimestepsInAvg);

        quantityArray[indexMeanVX] = vxMeanNew;
        quantityArray[indexMeanVY] = vyMeanNew;
        quantityArray[indexMeanVZ] = vzMeanNew;
        quantityArray[indexMeanRho] = rhoMeanNew;

        if (quantities[int(Statistic::Variances)]) {
            const uint arrayOffsetVariance = quantityArrayOffsets[int(Statistic::Variances)];
            const uint indexVarianceVX =
                calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrayOffsetVariance + 0);
            const uint indexVarianceVY =
                calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrayOffsetVariance + 1);
            const uint indexVarianceVZ =
                calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrayOffsetVariance + 2);
            const uint indexVarianceRho =
                calcArrayIndex(node, nPoints, timestepInTimeseries, nTimesteps, arrayOffsetVariance + 3);

            quantityArray[indexVarianceVX] = computeVariance(quantityArray[indexVarianceVX], vxMeanOld, vxMeanNew, vx,
                                                             timestepInAverage, invNumberOfTimestepsInAvg);
            quantityArray[indexVarianceVY] = computeVariance(quantityArray[indexVarianceVY], vyMeanOld, vyMeanNew, vy,
                                                             timestepInAverage, invNumberOfTimestepsInAvg);
            quantityArray[indexVarianceVZ] = computeVariance(quantityArray[indexVarianceVZ], vzMeanOld, vzMeanNew, vz,
                                                             timestepInAverage, invNumberOfTimestepsInAvg);
            quantityArray[indexVarianceRho] = computeVariance(quantityArray[indexVarianceRho], rhoMeanOld, rhoMeanNew, rho,
                                                              timestepInAverage, invNumberOfTimestepsInAvg);
        }
    }
}

__global__ void calcQuantitiesKernel(uint* pointIndices, uint nPoints, uint oldTimestepInTimeseries,
                                     uint timestepInTimeseries, uint timestepInAverage, uint nTimesteps, real* vx, real* vy,
                                     real* vz, real* rho, bool* quantities, uint* quantityArrayOffsets, real* quantityArray)
{
    const uint nodeIndex = vf::gpu::getNodeIndex();

    if (nodeIndex >= nPoints)
        return;

    const uint gridNodeIndex = pointIndices[nodeIndex];

    const real u_interpX = vx[gridNodeIndex];
    const real u_interpY = vy[gridNodeIndex];
    const real u_interpZ = vz[gridNodeIndex];
    const real rho_interp = rho[gridNodeIndex];

    calculatePointwiseQuantities(oldTimestepInTimeseries, timestepInTimeseries, timestepInAverage, nTimesteps, quantityArray,
                                 quantities, quantityArrayOffsets, nPoints, nodeIndex, u_interpX, u_interpY, u_interpZ,
                                 rho_interp);
}

__global__ void interpAndCalcQuantitiesKernel(uint* pointIndices, uint nPoints, uint oldTimestepInTimeseries,
                                              uint timestepInTimeseries, uint timestepInAverage, uint nTimesteps,
                                              real* distX, real* distY, real* distZ, real* vx, real* vy, real* vz, real* rho,
                                              uint* neighborX, uint* neighborY, uint* neighborZ, bool* quantities,
                                              uint* quantityArrayOffsets, real* quantityArray)
{
    const uint node = vf::gpu::getNodeIndex();

    if (node >= nPoints)
        return;

    const uint index_MMM = pointIndices[node];

    uint index_PMM, index_MPM, index_MMP, index_PPM, index_PMP, index_MPP, index_PPP;
    getNeighborIndicesOfBSW(index_MMM, index_PMM, index_MPM, index_MMP, index_PPM, index_PMP, index_MPP, index_PPP,
                            neighborX, neighborY, neighborZ);

    const real dXM = distX[node];
    const real dYM = distY[node];
    const real dZM = distZ[node];

    const real u_interpX = trilinearInterpolation(dXM, dYM, dZM, index_MMM, index_PMM, index_MPM, index_MMP, index_PPM,
                                                  index_PMP, index_MPP, index_PPP, vx);
    const real u_interpY = trilinearInterpolation(dXM, dYM, dZM, index_MMM, index_PMM, index_MPM, index_MMP, index_PPM,
                                                  index_PMP, index_MPP, index_PPP, vy);
    const real u_interpZ = trilinearInterpolation(dXM, dYM, dZM, index_MMM, index_PMM, index_MPM, index_MMP, index_PPM,
                                                  index_PMP, index_MPP, index_PPP, vz);
    const real rho_interp = trilinearInterpolation(dXM, dYM, dZM, index_MMM, index_PMM, index_MPM, index_MMP, index_PPM,
                                                   index_PMP, index_MPP, index_PPP, rho);

    calculatePointwiseQuantities(oldTimestepInTimeseries, timestepInTimeseries, timestepInAverage, nTimesteps, quantityArray,
                                 quantities, quantityArrayOffsets, nPoints, node, u_interpX, u_interpY, u_interpZ,
                                 rho_interp);
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
    using std::placeholders::_1;
    this->velocityRatio = std::bind(&Parameter::getScaledVelocityRatio, para, _1);
    this->densityRatio = std::bind(&Parameter::getScaledDensityRatio, para, _1);
    this->forceRatio = std::bind(&Parameter::getScaledForceRatio, para, _1);
    this->stressRatio = std::bind(&Parameter::getScaledStressRatio, para, _1);
    this->viscosityRatio = std::bind(&Parameter::getScaledViscosityRatio, para, _1);
    this->nondimensional = std::bind(&Probe::getNondimensionalConversionFactor, this, _1);

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
            probeStruct->timestepInTimeAverage++;
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

template <typename T>
std::string nameComponent(std::string name, T value)
{
    return "_" + name + "_" + StringUtil::toString<T>(value);
}

std::string Probe::makeParallelFileName(int id, int t)
{
    return this->probeName + "_bin" + nameComponent("ID", id) + nameComponent("t", t) + ".vtk";
}

std::string Probe::makeGridFileName(int level, int id, int t, uint part)
{
    return this->probeName + "_bin" + nameComponent("lev", level) + nameComponent("ID", id) +
           nameComponent<int>("Part", part) + nameComponent("t", t) + ".vtk";
}

std::string Probe::makeTimeseriesFileName(int level, int id)
{
    return this->probeName + "_timeseries" + nameComponent("lev", level) + nameComponent("ID", id) + ".txt";
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
    const std::string filename = this->outputPath + this->makeParallelFileName(para->getMyProcessID(), t_write);

    std::vector<std::string> nodedatanames = this->getVarNames();
    std::vector<std::string> cellNames;

    getWriter()->writeParallelFile(filename, fileNamesForCollectionFile, nodedatanames, cellNames);

    this->fileNamesForCollectionFile.clear();
}

void Probe::writeGridFile(int level, int t, uint part)
{
    const std::string fname = this->outputPath + this->makeGridFileName(level, para->getMyProcessID(), t, part);

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
        nodes[pos - startpos] = makeUbTuple(float(probeStruct->pointCoordsX[pos]),
                                            float(probeStruct->pointCoordsY[pos]),
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
            for (uint idx = 0; idx < endpos-startpos; idx++) {
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
    const std::string fname = this->outputPath + this->makeTimeseriesFileName(level, para->getMyProcessID());
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

void Probe::appendTimeseriesFile(int level, int t)
{
    std::ofstream out(this->timeseriesFileNames[level], std::ios::app | std::ios::binary);

    const uint tAvg_level = this->tAvg == 1 ? this->tAvg : this->tAvg * exp2(-level);
    const real deltaT = para->getTimeRatio() * tAvg_level;
    auto probeStruct = this->getProbeStruct(level);

    const real tStart = (t - this->tOut) * para->getTimeRatio();

    const int valuesPerTimestep = probeStruct->nPoints * probeStruct->nArrays + 1;

    std::vector<real> values(valuesPerTimestep);

    for (uint timestep = 0; timestep < probeStruct->timestepInTimeseries; timestep++) {
        values[0] = tStart + timestep * deltaT;

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
                    values[valueIndex + point] = probeStruct->quantitiesArrayH[startIndex + point] * conversionFactor;
                }

                valueIndex += probeStruct->nPoints;
            }
        }
        out.write((char*)values.data(), sizeof(real) * valuesPerTimestep);
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

//! \}
