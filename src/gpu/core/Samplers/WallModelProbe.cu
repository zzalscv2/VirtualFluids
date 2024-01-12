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
#include "WallModelProbe.h"

#include <functional>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include <basics/DataTypes.h>

#include "Cuda/CudaMemoryManager.h"
#include "DataStructureInitializer/GridProvider.h"
#include "Parameter/Parameter.h"
#include "Utilities.h"
#include "basics/constants/NumericConstants.h"

using namespace vf::basics::constant;
using valueIterator = thrust::device_vector<real>::iterator;
using indexIterator = thrust::device_vector<uint>::iterator;

///////////////////////////////////////////////////////////////////////////////////

std::vector<PostProcessingVariable> WallModelProbe::getPostProcessingVariables()
{
    std::function<real(int)> velocityRatio = [this](int level) { return para->getScaledVelocityRatio(level); };
    std::function<real(int)> stressRatio = [this](int level) { return para->getScaledStressRatio(level); };
    std::function<real(int)> forceRatio = [this](int level) { return para->getScaledForceRatio(level); };

    std::vector<PostProcessingVariable> postProcessingVariables;
    postProcessingVariables.emplace_back("vx_el_spatMean", velocityRatio);
    postProcessingVariables.emplace_back("vy_el_spatMean", velocityRatio);
    postProcessingVariables.emplace_back("vz_el_spatMean", velocityRatio);
    postProcessingVariables.emplace_back("vx1_spatMean", velocityRatio);
    postProcessingVariables.emplace_back("vy1_spatMean", velocityRatio);
    postProcessingVariables.emplace_back("vz1_spatMean", velocityRatio);
    postProcessingVariables.emplace_back("u_star_spatMean", velocityRatio);
    postProcessingVariables.emplace_back("Fx_spatMean", outputStress ? stressRatio : forceRatio);
    postProcessingVariables.emplace_back("Fy_spatMean", outputStress ? stressRatio : forceRatio);
    postProcessingVariables.emplace_back("Fz_spatMean", outputStress ? stressRatio : forceRatio);
    if (computeTemporalAverages) {
        postProcessingVariables.emplace_back("vx_el_spatTmpMean", velocityRatio);
        postProcessingVariables.emplace_back("vy_el_spatTmpMean", velocityRatio);
        postProcessingVariables.emplace_back("vz_el_spatTmpMean", velocityRatio);
        postProcessingVariables.emplace_back("vx1_spatTmpMean", velocityRatio);
        postProcessingVariables.emplace_back("vy1_spatTmpMean", velocityRatio);
        postProcessingVariables.emplace_back("vz1_spatTmpMean", velocityRatio);
        postProcessingVariables.emplace_back("u_star_spatTmpMean", velocityRatio);
        postProcessingVariables.emplace_back("Fx_spatTmpMean", outputStress ? stressRatio : forceRatio);
        postProcessingVariables.emplace_back("Fy_spatTmpMean", outputStress ? stressRatio : forceRatio);
        postProcessingVariables.emplace_back("Fz_spatTmpMean", outputStress ? stressRatio : forceRatio);
    }
    if (evaluatePressureGradient) {
        postProcessingVariables.emplace_back("dpdx_spatMean", forceRatio);
        postProcessingVariables.emplace_back("dpdy_spatMean", forceRatio);
        postProcessingVariables.emplace_back("dpdz_spatMean", forceRatio);
        if (computeTemporalAverages) {
            postProcessingVariables.emplace_back("dpdx_spatTmpMean", forceRatio);
            postProcessingVariables.emplace_back("dpdy_spatTmpMean", forceRatio);
            postProcessingVariables.emplace_back("dpdz_spatTmpMean", forceRatio);
        }
    }
    return postProcessingVariables;
}

///////////////////////////////////////////////////////////////////////////////////
void WallModelProbe::init()
{

    std::vector<std::string> variableNames;
    for (auto variable : getPostProcessingVariables()) {
        variableNames.push_back(variable.name);
    }

    const real x[1] { 0 };
    const real y[1] { 0 };
    const real z[1] { 0 };

    for (int level = 0; level <= para->getMaxLevel(); level++) {
        const std::string fileName = outputPath + makeTimeseriesFileName(probeName, level, para->getMyProcessID());
        TimeseriesFileWriter::writeHeader(fileName, 1, variableNames, x, y, z);
        const uint numberOfFluidNodes = evaluatePressureGradient ? countFluidNodes(level) : 0;
        levelData.emplace_back(fileName, numberOfFluidNodes);
        levelData.back().data.push_back(std::vector<real>(variableNames.size(), 0));
    }
}

void WallModelProbe::sample(int level, uint t)
{
    const uint tLevel = para->getTimeStep(level, t, false);
    const bool isCoarseTimestep = tLevel % t == 0;

    auto data = &levelData[level];

    const uint tAfterStartAvg = t - tStartAveraging;
    const uint tAfterStartOut = t - tStartWritingOutput;

    if (t > tStartAveraging && ((tAfterStartAvg % tBetweenAverages == 0 && isCoarseTimestep) || averageEveryTimestep)) {
        calculateQuantities(data, tLevel, level);
    }

    if (t > tStartWritingOutput && isCoarseTimestep && tAfterStartOut % tBetweenWriting == 0) {
        write(level);
    }
}

///////////////////////////////////////////////////////////////////////////////////

template <typename T>
T computeMean(T* device_pointer, uint numberOfPoints)
{
    thrust::device_ptr<T> thrust_pointer = thrust::device_pointer_cast(device_pointer);
    return thrust::reduce(thrust_pointer, thrust_pointer + numberOfPoints) / real(numberOfPoints);
}

struct isValidNode {
    __host__ __device__ real operator()(thrust::tuple<real, uint> x)
    {
        return thrust::get<1>(x) == GEO_FLUID ? thrust::get<0>(x) : c0o1;
    }
};

template <typename T>
T computeIndexBasedMean(T* device_pointer, uint* typeOfGridNode, uint numberOfNodes, uint numberOfFluidNodes)
{
    thrust::device_ptr<T> thrust_pointer = thrust::device_pointer_cast(device_pointer);
    thrust::device_ptr<uint> typePointer = thrust::device_pointer_cast(typeOfGridNode);
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(thrust_pointer, typePointer));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(thrust_pointer + numberOfNodes, typePointer + numberOfNodes));
    auto iter_begin = thrust::make_transform_iterator(begin, isValidNode());
    auto iter_end = thrust::make_transform_iterator(end, isValidNode());

    return thrust::reduce(iter_begin, iter_end) / real(numberOfFluidNodes);
}

template <typename T>
void computeAndSaveMean(T* device_pointer, uint numberOfPoints, std::vector<T>& quantityArray)
{
    quantityArray.push_back(computeMean(device_pointer, numberOfPoints));
}

template <typename T>
void computeAndSaveIndexBasedMean(T* device_pointer, uint* typeOfGridNode, uint numberOfNodes, uint numberOfFluidNodes,
                                  std::vector<real>& quantitiesArray)
{
    quantitiesArray.push_back(computeIndexBasedMean(device_pointer, typeOfGridNode, numberOfNodes, numberOfFluidNodes));
}

template <typename T>
void computeTemporalAverage(std::vector<T>& quantityArray, T oldMean, T currentValue, real invNumberOfAverages)
{
    quantityArray.push_back(oldMean + (currentValue - oldMean) * invNumberOfAverages);
}

uint WallModelProbe::countFluidNodes(int level)
{
    uint* typePointer = para->getParH(level)->typeOfGridNode;
    return std::count(typePointer, typePointer + para->getParH(level)->numberOfNodes, GEO_FLUID);
}

void WallModelProbe::calculateQuantities(LevelData* data, uint t, int level)
{
    const bool doTemporalAveraging = (t > tStartTemporalAveraging) && computeTemporalAverages;
    const uint nPoints = para->getParD(level)->stressBC.numberOfBCnodes;
    if (nPoints < 1)
        return;
    const real inverseNumberOfAveragedValues = c1o1 / real(data->numberOfAveragedValues + 1);
    auto paraDevice = para->getParD(level);

    std::vector<real> newTimestep;
    newTimestep.reserve(getPostProcessingVariables().size() + 1);
    newTimestep.push_back(t * para->getScaledTimeRatio(level));

    computeAndSaveMean(paraDevice->stressBC.Vx, nPoints, newTimestep);
    computeAndSaveMean(paraDevice->stressBC.Vy, nPoints, newTimestep);
    computeAndSaveMean(paraDevice->stressBC.Vz, nPoints, newTimestep);

    computeAndSaveMean(paraDevice->stressBC.Vx1, nPoints, newTimestep);
    computeAndSaveMean(paraDevice->stressBC.Vy1, nPoints, newTimestep);
    computeAndSaveMean(paraDevice->stressBC.Vz1, nPoints, newTimestep);

    computeAndSaveMean(paraDevice->wallModel.u_star, nPoints, newTimestep);

    computeAndSaveMean(paraDevice->wallModel.Fx, nPoints, newTimestep);
    computeAndSaveMean(paraDevice->wallModel.Fy, nPoints, newTimestep);
    computeAndSaveMean(paraDevice->wallModel.Fz, nPoints, newTimestep);

    if (doTemporalAveraging) {
        std::vector<real>& oldMeans = data->data.back();
        const size_t start = newTimestep.size();
        computeTemporalAverage(newTimestep, oldMeans[start + 0], newTimestep[0], inverseNumberOfAveragedValues);
        computeTemporalAverage(newTimestep, oldMeans[start + 1], newTimestep[1], inverseNumberOfAveragedValues);
        computeTemporalAverage(newTimestep, oldMeans[start + 2], newTimestep[2], inverseNumberOfAveragedValues);

        computeTemporalAverage(newTimestep, oldMeans[start + 3], newTimestep[3], inverseNumberOfAveragedValues);
        computeTemporalAverage(newTimestep, oldMeans[start + 4], newTimestep[4], inverseNumberOfAveragedValues);
        computeTemporalAverage(newTimestep, oldMeans[start + 5], newTimestep[5], inverseNumberOfAveragedValues);

        computeTemporalAverage(newTimestep, oldMeans[start + 6], newTimestep[6], inverseNumberOfAveragedValues);

        computeTemporalAverage(newTimestep, oldMeans[start + 7], newTimestep[7], inverseNumberOfAveragedValues);
        computeTemporalAverage(newTimestep, oldMeans[start + 8], newTimestep[8], inverseNumberOfAveragedValues);
        computeTemporalAverage(newTimestep, oldMeans[start + 9], newTimestep[9], inverseNumberOfAveragedValues);
        data->numberOfAveragedValues++;
    }

    if (this->evaluatePressureGradient) {
        const size_t startPressureGradient = newTimestep.size();
        computeAndSaveIndexBasedMean(paraDevice->forceX_SP, paraDevice->typeOfGridNode, paraDevice->numberOfNodes,
                                     data->numberOfFluidNodes, newTimestep);
        computeAndSaveIndexBasedMean(paraDevice->forceY_SP, paraDevice->typeOfGridNode, paraDevice->numberOfNodes,
                                     data->numberOfFluidNodes, newTimestep);
        computeAndSaveIndexBasedMean(paraDevice->forceZ_SP, paraDevice->typeOfGridNode, paraDevice->numberOfNodes,
                                     data->numberOfFluidNodes, newTimestep);

        if (doTemporalAveraging) {
            const size_t startTempAvg = newTimestep.size();
            std::vector<real>& oldMeans = data->data.back();
            computeTemporalAverage(newTimestep, oldMeans[startTempAvg + 0], newTimestep[startPressureGradient + 0],
                                   inverseNumberOfAveragedValues);
            computeTemporalAverage(newTimestep, oldMeans[startTempAvg + 1], newTimestep[startPressureGradient + 1],
                                   inverseNumberOfAveragedValues);
            computeTemporalAverage(newTimestep, oldMeans[startTempAvg + 2], newTimestep[startPressureGradient + 2],
                                   inverseNumberOfAveragedValues);
            computeTemporalAverage(newTimestep, oldMeans[startTempAvg + 3], newTimestep[startPressureGradient + 3],
                                   inverseNumberOfAveragedValues);
        }
    }
    data->data.push_back(newTimestep);
}

void WallModelProbe::write(int level)
{
    auto data = &levelData[level];
    std::vector dataToWrite(data->data.begin() + 1, data->data.end());
    TimeseriesFileWriter::appendData(data->timeseriesFileName, dataToWrite);
    auto lastTimestep = data->data.back();
    data->data.clear();
    data->data.push_back(lastTimestep);
}

//! \}
