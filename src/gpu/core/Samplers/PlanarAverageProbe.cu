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
#include "PlanarAverageProbe.h"
#include "Utilities/KernelUtilities.h"

#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdexcept>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <tuple>

#include <basics/DataTypes.h>
#include <basics/constants/NumericConstants.h>
#include <basics/utilities/UbTuple.h>
#include <basics/writer/WbWriterVtkXmlBinary.h>

#include "gpu/core/Output/FilePartCalculator.h"

#include "PostProcessor/MacroscopicQuantities.cuh"
#include "gpu/core/Cuda/CudaMemoryManager.h"
#include "gpu/core/DataStructureInitializer/GridProvider.h"
#include "gpu/core/Parameter/Parameter.h"
#include "gpu/core/Samplers/Utilities.h"
#include "gpu/cuda_helper/CudaGrid.h"

using namespace vf::basics::constant;

using valIterator = thrust::device_vector<real>::iterator;
using indIterator = thrust::device_vector<unsigned long long>::iterator;
using permIterator = thrust::permutation_iterator<valIterator, indIterator>;
using iterPair = std::pair<permIterator, permIterator>;

iterPair getPermutationIterators(real* values, unsigned long long* indices, uint numberOfIndices)
{
    auto val_pointer = thrust::device_pointer_cast(values);
    auto indices_pointer = thrust::device_pointer_cast(indices);
    permIterator iter_begin(val_pointer, indices_pointer);
    permIterator iter_end(val_pointer, indices_pointer + numberOfIndices);
    return std::make_pair(iter_begin, iter_end);
}

struct covariance
{
    const real mean_x, mean_y;
    covariance(real mean_x, real mean_y) : mean_x(mean_x), mean_y(mean_y)
    {
    }

    template <typename Tuple>
    __host__ __device__ real operator()(const Tuple& t) const
    {
        return (thrust::get<0>(t) - mean_x) * (thrust::get<1>(t) - mean_y);
    }
};

struct skewness
{
    const real mean;
    skewness(real mean) : mean(mean)
    {
    }
    __host__ __device__ real operator()(const real& x) const
    {
        return (x - mean) * (x - mean) * (x - mean);
    }
};

struct flatness
{
    const real mean;
    flatness(real mean) : mean(mean)
    {
    }
    __host__ __device__ real operator()(const real& x) const
    {
        return (x - mean) * (x - mean) * (x - mean) * (x - mean);
    }
};

struct Means
{
    real vx, vy, vz;
};

struct Covariances
{
    real vxvx, vyvy, vzvz, vxvy, vxvz, vyvz;
};

struct Skewnesses
{
    real Sx, Sy, Sz;
};

struct Flatnesses
{
    real Fx, Fy, Fz;
};

///////////////////////////////////////////////////////////////////////////////////

__global__ void moveIndicesInPosNormalDir(unsigned long long* pointIndices, uint nPoints, const uint* neighborNormal)
{
    const uint nodeIndex = vf::gpu::getNodeIndex();

    if (nodeIndex >= nPoints)
        return;

    pointIndices[nodeIndex] = (unsigned long long)neighborNormal[pointIndices[nodeIndex]];
}

///////////////////////////////////////////////////////////////////////////////////
std::vector<PostProcessingVariable> PlanarAverageProbe::getPostProcessingVariables(PlanarAverageProbe::Statistic statistic,
                                                                                   bool includeTimeAverages)
{
    auto velocityRatio = [this](int level) { return para->getScaledVelocityRatio(level); };
    auto viscosityRatio = [this](int level) { return para->getScaledViscosityRatio(level); };
    auto stressRatio = [this](int level) { return para->getScaledStressRatio(level); };
    auto nondimensional = [this](int level) { return c1o1; };

    std::vector<PostProcessingVariable> postProcessingVariables;
    switch (statistic) {
        case Statistic::Means:
            postProcessingVariables.emplace_back("vx_spatialMean", velocityRatio);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("vx_spatioTemporalMean", velocityRatio);
            postProcessingVariables.emplace_back("vy_spatialMean", velocityRatio);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("vy_spatioTemporalMean", velocityRatio);
            postProcessingVariables.emplace_back("vz_spatialMean", velocityRatio);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("vz_spatioTemporalMean", velocityRatio);
            if(para->getUseTurbulentViscosity()){
                postProcessingVariables.emplace_back("EddyViscosity_spatialMean", viscosityRatio);
                if (includeTimeAverages)
                    postProcessingVariables.emplace_back("EddyViscosity_spatioTemporalMean", viscosityRatio);
            }
            break;
        case Statistic::Covariances:
            postProcessingVariables.emplace_back("vxvx_spatialMean", stressRatio);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("vxvx_spatioTemporalMean", stressRatio);
            postProcessingVariables.emplace_back("vyvy_spatialMean", stressRatio);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("vyvy_spatioTemporalMean", stressRatio);
            postProcessingVariables.emplace_back("vzvz_spatialMean", stressRatio);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("vzvz_spatioTemporalMean", stressRatio);
            postProcessingVariables.emplace_back("vxvy_spatialMean", stressRatio);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("vxvy_spatioTemporalMean", stressRatio);
            postProcessingVariables.emplace_back("vxvz_spatialMean", stressRatio);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("vyvz_spatialMean", stressRatio);
            postProcessingVariables.emplace_back("vxvz_spatioTemporalMean", stressRatio);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("vyvz_spatioTemporalMean", stressRatio);
            break;
        case Statistic::Skewness:
            postProcessingVariables.emplace_back("SkewnessX_spatialMean", nondimensional);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("SkewnessX_spatioTemporalMean", nondimensional);
            postProcessingVariables.emplace_back("SkewnessY_spatialMean", nondimensional);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("SkewnessY_spatioTemporalMean", nondimensional);
            postProcessingVariables.emplace_back("SkewnessZ_spatialMean", nondimensional);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("SkewnessZ_spatioTemporalMean", nondimensional);
            break;
        case Statistic::Flatness:
            postProcessingVariables.emplace_back("FlatnessX_spatialMean", nondimensional);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("FlatnessX_spatioTemporalMean", nondimensional);
            postProcessingVariables.emplace_back("FlatnessY_spatialMean", nondimensional);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("FlatnessY_spatioTemporalMean", nondimensional);
            postProcessingVariables.emplace_back("FlatnessZ_spatialMean", nondimensional);
            if (includeTimeAverages)
                postProcessingVariables.emplace_back("FlatnessZ_spatioTemporalMean", nondimensional);
            break;

        default:
            throw std::runtime_error("PlanarAverageProbe::getPostProcessingVariables: Statistic unavailable!");
            break;
    }
    return postProcessingVariables;
}

void PlanarAverageProbe::addStatistic(PlanarAverageProbe::Statistic statistic)
{
    if (!isStatisticIn(statistic, statistics))
        statistics.push_back(statistic);
}

///////////////////////////////////////////////////////////////////////////////////
void PlanarAverageProbe::init()
{
    const size_t numberOfVariables = getVarNames(false).size();
    for (int level = 0; level <= para->getMaxLevel(); level++) {
        levelData.emplace_back();
        auto data = &levelData.back();
        std::vector<unsigned long long> indices = findIndicesInPlane(level);
        findCoordinatesForPlanes(level, data->coordinateX, data->coordinateY, data->coordinateZ);
        data->numberOfPlanes = static_cast<uint>(data->coordinateX.size());
        data->numberOfPointsPerPlane = indices.size();
        cudaMemoryManager->cudaAllocPlanarAverageProbeIndices(this, level);
        std::copy(indices.begin(), indices.end(), data->indicesOfFirstPlaneH);
        cudaMemoryManager->cudaCopyPlanarAverageProbeIndicesHtoD(this, level);
        data->instantaneous.resize(data->numberOfPlanes, std::vector<real>(numberOfVariables, 0.0f));
        if (computeTimeAverages) {
            data->timeAverages.resize(data->numberOfPlanes, std::vector<real>(numberOfVariables, 0.0f));
        }
    }
}

void PlanarAverageProbe::sample(int level, uint t)
{
    if (t < tStartAveraging)
        return;

    const uint t_level = para->getTimeStep(level, t, true);
    const uint levelFactor = exp2(level);
    const bool doTimeAverages = t_level >= (tStartTemporalAveraging * levelFactor);

    if ((t_level - tStartAveraging * levelFactor) % tBetweenAverages * levelFactor == 0) {
        calculateQuantities(&levelData[level], t_level, level, doTimeAverages);
    }

    if (t_level % tBetweenWriting * exp(level) == 0) {
        writeGridFile(level, t);
        if (level == 0)
            writeParallelFile(t);
    }
}

PlanarAverageProbe::~PlanarAverageProbe()
{
    for (int level = 0; level <= para->getMaxLevel(); level++) {
        cudaMemoryManager->cudaFreePlanarAverageProbeIndices(this, level);
    }
}

std::vector<unsigned long long> PlanarAverageProbe::findIndicesInPlane(int level)
{
    std::vector<unsigned long long> indices;
    auto param = para->getParH(level);

    const real* coordinatesPlaneNormal = [&] {
        switch (planeNormal) {
            case PlaneNormal::x:
                return param->coordinateX;
            case PlaneNormal::y:
                return param->coordinateY;
            case PlaneNormal::z:
                return param->coordinateZ;
            default:
                throw std::runtime_error("PlaneNormal not defined!");
        }
    }();

    const auto firstIndex = param->neighborZ[param->neighborY[param->neighborX[1]]];

    const real coordFirstPlane = coordinatesPlaneNormal[firstIndex];

    for (unsigned long long node = 1; node < param->numberOfNodes; node++) {
        if (coordinatesPlaneNormal[node] == coordFirstPlane && param->typeOfGridNode[node] == GEO_FLUID)
            indices.push_back(node);
    }

    return indices;
}

void PlanarAverageProbe::findCoordinatesForPlanes(int level, std::vector<real>& coordinateX, std::vector<real>& coordinateY,
                                                  std::vector<real>& coordinateZ)
{
    uint* neighborInNormal = [&](PlaneNormal planeNormal) {
        switch (planeNormal) {
            case PlaneNormal::x:
                return para->getParH(level)->neighborX;
            case PlaneNormal::y:
                return para->getParH(level)->neighborY;
            case PlaneNormal::z:
                return para->getParH(level)->neighborZ;
        };
        throw std::runtime_error("PlaneNormal not defined!");
    }(planeNormal);
    unsigned long long next_index =
        para->getParH(level)->neighborZ[para->getParH(level)->neighborY[para->getParH(level)->neighborX[1]]];
    while (GEO_FLUID == para->getParH(level)->typeOfGridNode[next_index]) {
        coordinateX.push_back(para->getParH(level)->coordinateX[next_index]);
        coordinateY.push_back(para->getParH(level)->coordinateY[next_index]);
        coordinateZ.push_back(para->getParH(level)->coordinateZ[next_index]);
        next_index = neighborInNormal[next_index];
    }
}

real updateTimeAverage(real newValue, real oldAverage, real invNumberOfTimesteps)
{
    return oldAverage + (newValue - oldAverage) * invNumberOfTimesteps;
}

real computeMean(iterPair x, real invNPointsPerPlane)
{
    const real sum = thrust::reduce(std::get<0>(x), std::get<1>(x));
    return sum * invNPointsPerPlane;
}

Means computeMeans(iterPair vx, iterPair vy, iterPair vz, real invNPointsPerPlane)
{
    Means means;
    means.vx = computeMean(vx, invNPointsPerPlane);
    means.vy = computeMean(vy, invNPointsPerPlane);
    means.vz = computeMean(vz, invNPointsPerPlane);
    return means;
}

real computeCovariance(iterPair x, iterPair y, real mean_x, real mean_y, real invNPointsPerPlane)
{
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(x.first, y.first));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(x.second, y.second));
    return thrust::transform_reduce(begin, end, covariance(mean_x, mean_y), c0o1, thrust::plus<real>()) * invNPointsPerPlane;
}

Covariances computeCovariances(iterPair vx, iterPair vy, iterPair vz, Means means, real invNPointsPerPlane)
{
    Covariances covariances;

    covariances.vxvx = computeCovariance(vx, vx, means.vx, means.vx, invNPointsPerPlane);
    covariances.vyvy = computeCovariance(vy, vy, means.vy, means.vy, invNPointsPerPlane);
    covariances.vzvz = computeCovariance(vz, vz, means.vz, means.vz, invNPointsPerPlane);
    covariances.vxvy = computeCovariance(vx, vy, means.vx, means.vy, invNPointsPerPlane);
    covariances.vxvz = computeCovariance(vx, vz, means.vx, means.vz, invNPointsPerPlane);
    covariances.vyvz = computeCovariance(vy, vz, means.vy, means.vz, invNPointsPerPlane);

    return covariances;
}

real computeSkewness(iterPair x, real mean, real covariance, real invNPointsPerPlane)
{
    return thrust::transform_reduce(x.first, x.second, skewness(mean), c0o1, thrust::plus<real>()) * invNPointsPerPlane *
           pow(covariance, -1.5f);
}

Skewnesses computeSkewnesses(Means means, Covariances covariances, iterPair vx, iterPair vy, iterPair vz,
                             real invNPointsPerPlane)
{
    Skewnesses skewnesses;

    skewnesses.Sx = computeSkewness(vx, means.vx, covariances.vxvx, invNPointsPerPlane);
    skewnesses.Sy = computeSkewness(vy, means.vy, covariances.vyvy, invNPointsPerPlane);
    skewnesses.Sz = computeSkewness(vz, means.vz, covariances.vzvz, invNPointsPerPlane);

    return skewnesses;
}

real computeFlatness(iterPair x, real mean, real covariance, real invNPointsPerPlane)
{
    return thrust::transform_reduce(x.first, x.second, flatness(mean), c0o1, thrust::plus<real>()) * invNPointsPerPlane *
           pow(covariance, -2.0f);
}

Flatnesses computeFlatnesses(iterPair vx, iterPair vy, iterPair vz, Means means, Covariances covariances,
                             real invNPointsPerPlane)
{
    Flatnesses flatnesses;

    flatnesses.Fx = computeFlatness(vx, means.vx, covariances.vxvx, invNPointsPerPlane);
    flatnesses.Fy = computeFlatness(vy, means.vy, covariances.vyvy, invNPointsPerPlane);
    flatnesses.Fz = computeFlatness(vz, means.vz, covariances.vzvz, invNPointsPerPlane);

    return flatnesses;
}

std::vector<real> computePlaneStatistics(std::vector<PlanarAverageProbe::Statistic>& statistics, iterPair vx, iterPair vy,
                                         iterPair vz, iterPair turbulentViscosity, real invNPointsPerPlane, bool useTurbulentViscosity)
{
    std::vector<real> averages;

    if (!isStatisticIn(PlanarAverageProbe::Statistic::Means, statistics))
        return averages;

    const auto means = computeMeans(vx, vy, vz, invNPointsPerPlane);
    averages.push_back(means.vx);
    averages.push_back(means.vy);
    averages.push_back(means.vz);
    if(useTurbulentViscosity)
        averages.push_back(computeMean(turbulentViscosity, invNPointsPerPlane));

    if (!isStatisticIn(PlanarAverageProbe::Statistic::Covariances, statistics))
        return averages;

    const auto covariances = computeCovariances(vx, vy, vz, means, invNPointsPerPlane);
    averages.push_back(covariances.vxvx);
    averages.push_back(covariances.vyvy);
    averages.push_back(covariances.vzvz);
    averages.push_back(covariances.vxvy);
    averages.push_back(covariances.vxvz);
    averages.push_back(covariances.vyvz);

    if (!isStatisticIn(PlanarAverageProbe::Statistic::Skewness, statistics))
        return averages;

    const auto skewnesses = computeSkewnesses(means, covariances, vx, vy, vz, invNPointsPerPlane);
    averages.push_back(skewnesses.Sx);
    averages.push_back(skewnesses.Sy);
    averages.push_back(skewnesses.Sz);

    if (!isStatisticIn(PlanarAverageProbe::Statistic::Flatness, statistics))
        return averages;

    const auto flatnesses = computeFlatnesses(vx, vy, vz, means, covariances, invNPointsPerPlane);
    averages.push_back(flatnesses.Fx);
    averages.push_back(flatnesses.Fy);
    averages.push_back(flatnesses.Fz);

    return averages;
}

std::vector<real> computeNewTimeAverages(std::vector<real>& oldAverages, std::vector<real>& instantaneous,
                                         real invNumberOfTimesteps)
{
    std::vector<real> newAverages;
    newAverages.reserve(oldAverages.size());
    for (uint i = 0; i < oldAverages.size(); i++) {
        newAverages.push_back(updateTimeAverage(instantaneous[i], oldAverages[i], invNumberOfTimesteps));
    }
    return newAverages;
}

void PlanarAverageProbe::calculateQuantities(PlanarAverageProbeLevelData* data, uint t_level, int level, bool doTimeAverages)
{

    auto parameter = para->getParD(level);
    calculateMacroscopicQuantitiesCompressible(
        parameter->velocityX, parameter->velocityY, parameter->velocityZ, parameter->rho, parameter->pressure,
        parameter->typeOfGridNode, parameter->neighborX, parameter->neighborY, parameter->neighborZ,
        parameter->numberOfNodes, parameter->numberofthreads, parameter->distributions.f[0], parameter->isEvenTimestep);
    cudaDeviceSynchronize();

    const uint* neighborInNormalDirection = getNeighborIndicesInPlaneNormal(level);

    const bool doTmpAveraging = t_level >= (tStartTemporalAveraging * exp2(level));
    const bool useTurbulentViscosity = para->getUseTurbulentViscosity();

    const real invNPointsPerPlane = c1o1 / static_cast<real>(data->numberOfPointsPerPlane);
    const real invNumberOfTimesteps = c1o1 / static_cast<real>(data->numberOfTimestepsInTimeAverage + 1);

    const auto velocityX =
        getPermutationIterators(parameter->velocityX, data->indicesOfFirstPlaneD, data->numberOfPointsPerPlane);
    const auto velocityY =
        getPermutationIterators(parameter->velocityY, data->indicesOfFirstPlaneD, data->numberOfPointsPerPlane);
    const auto velocityZ =
        getPermutationIterators(parameter->velocityZ, data->indicesOfFirstPlaneD, data->numberOfPointsPerPlane);
    const auto turbulentViscosity = getPermutationIterators(para->getParD(level)->turbViscosity, data->indicesOfFirstPlaneD,
                                             data->numberOfPointsPerPlane);

    for (uint plane = 0; plane < data->numberOfPlanes; plane++) {
        data->instantaneous[plane] = computePlaneStatistics(statistics, velocityX, velocityY, velocityZ, turbulentViscosity, invNPointsPerPlane, para->getUseTurbulentViscosity());

        if (doTimeAverages)
            data->timeAverages[plane] =
                computeNewTimeAverages(data->timeAverages[plane], data->instantaneous[plane], invNumberOfTimesteps);

        vf::cuda::CudaGrid grid = vf::cuda::CudaGrid(para->getParH(level)->numberofthreads, data->numberOfPointsPerPlane);
        moveIndicesInPosNormalDir<<<grid.grid, grid.threads>>>(data->indicesOfFirstPlaneD, data->numberOfPointsPerPlane,
                                                               neighborInNormalDirection);
    }
    cudaMemoryManager->cudaCopyPlanarAverageProbeIndicesHtoD(this, level);

    getLastCudaError("PlanarAverageProbe::calculateQuantities execution failed");
}

const uint* PlanarAverageProbe::getNeighborIndicesInPlaneNormal(int level)
{
    switch (planeNormal) {
        case PlaneNormal::x:
            return para->getParD(level)->neighborX;
        case PlaneNormal::y:
            return para->getParD(level)->neighborY;
        case PlaneNormal::z:
            return para->getParD(level)->neighborZ;
    };

    throw std::runtime_error("PlaneNormal not defined!");
}

void PlanarAverageProbe::writeParallelFile(uint t)
{
    const int t_write = this->nameFilesWithFileCount ? (t - tStartWritingOutput) / this->tBetweenWriting : t;
    const std::string filename = this->outputPath + makeParallelFileName(probeName, para->getMyProcessID(), t_write);

    std::vector<std::string> nodedatanames = this->getVarNames(this->computeTimeAverages);
    std::vector<std::string> cellNames;

    WbWriterVtkXmlBinary::getInstance()->writeParallelFile(filename, fileNamesForCollectionFile, nodedatanames, cellNames);

    this->fileNamesForCollectionFile.clear();
}

void PlanarAverageProbe::writeGridFile(int level, uint t)
{
    const std::string fname = outputPath + makeGridFileName(probeName, level, para->getMyProcessID(), t, 1);
    std::vector<UbTupleFloat3> nodes;
    std::vector<std::string> nodedatanames = this->getVarNames(this->computeTimeAverages);

    std::vector<std::vector<double>> nodedata(nodedatanames.size());

    auto data = &levelData[level];

    nodes.resize(data->numberOfPlanes);

    for (uint pos = 0; pos < data->numberOfPlanes; pos++) {
        nodes[pos] =
            makeUbTuple(float(data->coordinateX[pos]), float(data->coordinateY[pos]), float(data->coordinateZ[pos]));
    }

    for (auto it = nodedata.begin(); it != nodedata.end(); it++)
        it->resize(data->numberOfPlanes);

    int arrayIndex = 0;

    for (auto statistic : statistics) {

        std::vector<PostProcessingVariable> postProcessingVariables = this->getPostProcessingVariables(statistic, false);

        for (uint arr = 0; arr < uint(postProcessingVariables.size()); arr++) {
            const real coeff = postProcessingVariables[arr].conversionFactor(level);
            for (uint plane = 0; plane < data->numberOfPlanes; plane++) {
                if (computeTimeAverages) {
                    nodedata[2 * arrayIndex][plane] = double(data->instantaneous[plane][arrayIndex + arr] * coeff);
                    nodedata[2 * arrayIndex + 1][plane] = double(data->instantaneous[plane][arrayIndex + arr] * coeff);
                } else {
                    nodedata[arrayIndex + arr][plane] = double(data->instantaneous[plane][arrayIndex + arr] * coeff);
                }
            }
            arrayIndex++;
        }
    }
    std::string fullName =
        WbWriterVtkXmlBinary::getInstance()->writeNodesWithNodeData(fname, nodes, nodedatanames, nodedata);
    this->fileNamesForCollectionFile.push_back(fullName.substr(fullName.find_last_of('/') + 1));
}

std::vector<std::string> PlanarAverageProbe::getVarNames(bool includeTimeAverages)
{
    std::vector<std::string> varNames;
    for (auto statistic : statistics) {
        std::vector<PostProcessingVariable> postProcessingVariables =
            this->getPostProcessingVariables(statistic, includeTimeAverages);
        for (auto postProcessingVariable : postProcessingVariables) {
            varNames.push_back(postProcessingVariable.name);
        }
    }
    return varNames;
}

bool isStatisticIn(PlanarAverageProbe::Statistic statistic, std::vector<PlanarAverageProbe::Statistic> statistics)
{
    return std::find(statistics.begin(), statistics.end(), statistic) != statistics.end();
}

//! \}