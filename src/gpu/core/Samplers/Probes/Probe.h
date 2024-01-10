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
//! \date 13/05/2022
//! \brief Base class for probes called in UpdateGrid27

//=======================================================================================

#ifndef Probe_H
#define Probe_H

#include "Samplers/Sampler.h"

#include <cuda.h>
#include <functional>
#include <stdexcept>
#include <string>

#include "basics/writer/WbWriterVtkXmlBinary.h"
#include <basics/DataTypes.h>
#include <basics/PointerDefinitions.h>

struct LBMSimulationParameter;

//=======================================================================================
//! \note How to add new Statistics
//! Generally, the Statistic enum refers to the type of statistic to be calculated.
//! It then depends on the derived probe class, which of these statistics are available.
//! Some type of statistics are only suitable for a certain probe class, others might
//! simply not have been implemented, yet.
//! For the same reasons it is also probe-specific, for which quantities (e.g. velocities, rho, etc.) these statistics are
//! computed. The specific quantity (e.g., mean of vx, or variance of rho) is defined as PostProcessingVariable in
//! getPostProcessingVariables of each respective probe. PostProcessingVariable also holds the name and conversionFactor of
//! the quantity that is required when writing the data to file
//!
//! To add new Statistics:
//!     1. Add enum here, LAST has to stay last
//!     2. For PointProbe and PlaneProbe: add the computation of the statistic in switch statement in
//!     calculatePointwiseQuantities.
//!     3. For PlanarAverageProbe and WallModelProbe: add the computation directly in calculateQuantities.
//!     4. In getPostProcessingVariables add the static in the switch statement and add the corresponding
//!     PostProcessingVariables
//!     5. Add Statistic to isAvailableStatistic of the respective probe
//!
//!  When adding new quantities to existing statistics (e.g., add rho to PlanarAverageProbe which currently only computes
//!  stats of velocity) only do steps 2 to 4
//!

enum class Statistic {
    // Variables currently available in Point and Plane probe (all temporal pointwise statistics)
    Instantaneous,
    Means,
    Variances,

    // Variables available in PlanarAverage probe and (partially) in WallModelProbe
    // Spatial statistics are typically computed across fixed spatial subdomains, e.g. a plane of constant height
    // Spatio-temporal statistics additionally average the spatial stats in time
    SpatialMeans,
    SpatioTemporalMeans,
    SpatialCovariances,
    SpatioTemporalCovariances,
    SpatialSkewness,
    SpatioTemporalSkewness,
    SpatialFlatness,
    SpatioTemporalFlatness,
    LAST,
};

struct PostProcessingVariable
{
    std::string name;
    std::function<real(int)> conversionFactor;
    PostProcessingVariable(std::string name, std::function<real(int)> conversionFactor)
        : name(name), conversionFactor(conversionFactor) {};
};

struct ProbeStruct
{
    uint nPoints, nIndices, nArrays;
    uint nTimesteps = 1;
    uint timestepInTimeseries = 0;
    uint numberOfAveragedValues = 0;
    uint lastTimestepInOldTimeseries = 0;
    uint *pointIndicesH, *pointIndicesD;
    real *pointCoordsX, *pointCoordsY, *pointCoordsZ;
    bool hasDistances = false;
    real *distXH, *distYH, *distZH, *distXD, *distYD, *distZD;
    real *quantitiesArrayH, *quantitiesArrayD;
    bool *quantitiesH, *quantitiesD;
    uint *arrayOffsetsH, *arrayOffsetsD;
    bool isEvenTAvg = true;
};

struct GridParams
{
    uint* gridNodeIndices;
    real *velocityX, *velocityY, *velocityZ, *density;
};

GridParams getGridParams(ProbeStruct* probeStruct, LBMSimulationParameter* para);

struct ProbeArray
{
    real* data;
    uint* offsets;
    bool* statistics;
    uint numberOfPoints;
};

ProbeArray getProbeArray(ProbeStruct* probeStruct);

struct InterpolationParams
{
    real *distanceX, *distanceY, *distanceZ;
    uint *neighborX, *neighborY, *neighborZ;
};

InterpolationParams getInterpolationParams(ProbeStruct* probeStruct, LBMSimulationParameter* para);

struct TimeseriesParams
{
    uint lastTimestep, numberOfTimesteps;
};

TimeseriesParams getTimeseriesParams(ProbeStruct* probeStruct);

__host__ __device__ int calcArrayIndex(int node, int nNodes, int timestep, int nTimesteps, int array);

__global__ void calculateQuantitiesKernel(uint numberOfAveragedValues, GridParams gridParams, ProbeArray array);

__global__ void interpolateAndCalculateQuantitiesKernel(uint numberOfAveragedValues, GridParams gridParams, ProbeArray array,
                                                   InterpolationParams interpolationParams);

__global__ void calculateQuantitiesInTimeseriesKernel(uint numberOfAveragedValues, GridParams gridParams, ProbeArray array,
                                                      TimeseriesParams timeseriesParams);

__global__ void interpolateAndCalculateQuantitiesInTimeseriesKernel(uint numberOfAveragedValues, GridParams gridParams, ProbeArray array,
                                                                    InterpolationParams interpolationParams,
                                                                    TimeseriesParams timeseriesParams);

uint calcOldTimestep(uint currentTimestep, uint lastTimestepInOldSeries);

class Probe : public Sampler
{
public:
    Probe(SPtr<Parameter> para, SPtr<CudaMemoryManager> cudaMemoryManager, const std::string probeName,
          const std::string outputPath, const uint tStartAvg, const uint tStartTmpAvg, const uint tAvg, const uint tStartOut,
          const uint tOut, const bool hasDeviceQuantityArray, const bool outputTimeSeries)
        : probeName(probeName), outputPath(outputPath + (outputPath.back() == '/' ? "" : "/")), tStartAvg(tStartAvg),
          tStartTmpAveraging(tStartTmpAvg), tAvg(tAvg), tStartOut(tStartOut), tOut(tOut),
          hasDeviceQuantityArray(hasDeviceQuantityArray), outputTimeSeries(outputTimeSeries),
          Sampler(para, cudaMemoryManager)
    {
        if (tStartOut < tStartAvg)
            throw std::runtime_error("Probe: tStartOut must be larger than tStartAvg!");
    }

    virtual ~Probe();

    void init() override;
    void sample(int level, uint t) override;

    SPtr<ProbeStruct> getProbeStruct(int level)
    {
        return this->probeParams[level];
    }

    void addStatistic(Statistic variable);
    void addAllAvailableStatistics();

    bool getHasDeviceQuantityArray();
    uint getTStartTmpAveraging()
    {
        return this->tStartTmpAveraging;
    }

    void setFileNameToNOut()
    {
        this->fileNameLU = false;
    }

protected:
    virtual WbWriterVtkXmlBinary* getWriter()
    {
        return WbWriterVtkXmlBinary::getInstance();
    };
    real getNondimensionalConversionFactor(int level);


private:
    virtual bool isAvailableStatistic(Statistic variable) = 0;

    virtual std::vector<PostProcessingVariable> getPostProcessingVariables(Statistic variable) = 0;
    std::vector<PostProcessingVariable> getPostProcessingVariables(int statistic)
    {
        return getPostProcessingVariables(static_cast<Statistic>(statistic));
    };

    virtual void findPoints(std::vector<int>& probeIndices, std::vector<real>& distancesX, std::vector<real>& distancesY,
                            std::vector<real>& distancesZ, std::vector<real>& pointCoordsX, std::vector<real>& pointCoordsY,
                            std::vector<real>& pointCoordsZ, int level) = 0;
    void addProbeStruct(std::vector<int>& probeIndices, std::vector<real>& distX, std::vector<real>& distY,
                        std::vector<real>& distZ, std::vector<real>& pointCoordsX, std::vector<real>& pointCoordsY,
                        std::vector<real>& pointCoordsZ, int level);
    virtual void calculateQuantities(SPtr<ProbeStruct> probeStruct, uint t, int level) = 0;

    virtual void write(int level, int t);
    virtual void writeParallelFile(int t);
    virtual void writeGridFile(int level, int t, uint part);
    std::string writeTimeseriesHeader(int level);
    void appendTimeseriesFile(int level, int t);
    std::vector<real> getTimestepData(real time, uint length, int timestep, ProbeStruct* probeStruct, int level);

    std::vector<std::string> getVarNames();


    virtual uint getNumberOfTimestepsInTimeseries(int level)
    {
        return 1;
    }



protected:
    const std::string probeName;
    const std::string outputPath;

    std::vector<SPtr<ProbeStruct>> probeParams;
    bool quantities[int(Statistic::LAST)] = {};
    //! flag initiating memCopy in Point and PlaneProbe. Other probes are only based on
    //! thrust reduce functions and therefore dont need explict memCopy in interact()
    const bool hasDeviceQuantityArray;

    //! flag initiating time series output in Point and WallModelProbe.
    const bool outputTimeSeries;
    std::vector<std::string> fileNamesForCollectionFile;
    std::vector<std::string> timeseriesFileNames;

    //! if true, written file name contains time step in LU, else is the number of the written probe files
    bool fileNameLU = true;
    const uint tStartAvg;

    //! only non-zero in PlanarAverageProbe and WallModelProbe to switch on Spatio-temporal
    //! averaging (while only doing spatial averaging for t<tStartTmpAveraging)
    const uint tStartTmpAveraging;
    //! for tAvg==1 the probe will be evaluated in every sub-timestep of each respective level, else, the
    //! probe will only be evaluated in each synchronous time step
    const uint tAvg;
    const uint tStartOut;
    const uint tOut;

    std::function<real(int)> velocityRatio;
    std::function<real(int)> densityRatio;
    std::function<real(int)> forceRatio;
    std::function<real(int)> stressRatio;
    std::function<real(int)> viscosityRatio;
    std::function<real(int)> nondimensional;
};

std::string makeGridFileName(std::string probeName, int level, int id, int t, uint part);
std::string makeParallelFileName(std::string probeName, int id, int t);
std::string makeTimeseriesFileName(std::string probeName, int level, int id);

#endif
//! \}
