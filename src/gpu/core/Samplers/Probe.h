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
class Parameter;
class CudaMemoryManager;

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

class Probe : public Sampler
{
public:

    enum class Statistic { Instantaneous, Means, Variances };

    Probe(SPtr<Parameter> para, SPtr<CudaMemoryManager> cudaMemoryManager, std::string outputPath, std::string probeName,
          uint tStartAveraging, uint tBetweenAverages, uint tStartWritingOutput, uint tBetweenWriting, bool outputTimeSeries)
        : para(para), cudaMemoryManager(cudaMemoryManager), tStartAveraging(tStartAveraging),
          tBetweenAverages(tBetweenAverages), tStartWritingOutput(tStartWritingOutput), tBetweenWriting(tBetweenWriting),
          outputTimeSeries(outputTimeSeries), Sampler(outputPath, probeName)
    {
        if (tStartWritingOutput < tStartAveraging)
            throw std::runtime_error("Probe: tStartWritingOutput must be larger than tStartAveraging!");
    }

    ~Probe();

    void init() override;
    void sample(int level, uint t) override;
    void addProbePlane(real startX, real startY, real startZ, real length, real width, real height)
    {
        planes.emplace_back(Plane { startX, startY, startZ, length, width, height });
    }

    void addProbePoint(real x, real y, real z)
    {
        points.emplace_back(Point { x, y, z });
    }
    struct PostProcessingVariable
    {
        std::string name;
        real conversionFactor;
        PostProcessingVariable(std::string name, real conversionFactor) : name(name), conversionFactor(conversionFactor)
        {
        }
    };

    struct ProbeData
    {
        real *instantaneous, *means, *variances;
        bool computeInstantaneous, computeMeans, computeVariances;
        uint numberOfPoints, numberOfQuantities, numberOfTimesteps;
        uint* indices;
        __device__ __host__ ProbeData(bool computeInstantaneous, bool computeMeans, bool computeVariance, uint numberOfPoints, uint numberOfQuantities, uint numberOfTimesteps)
            : computeInstantaneous(computeInstantaneous), computeMeans(computeMeans), computeVariances(computeVariance),
              numberOfPoints(numberOfPoints), numberOfQuantities(numberOfQuantities), numberOfTimesteps(numberOfTimesteps)
        {
        }
    };

    struct TimeseriesParams
    {
        uint lastTimestep {}, numberOfTimesteps {}, lastTimestepInOldTimeseries {};
    };

    struct LevelData
    {
        ProbeData probeDataH, probeDataD;
        TimeseriesParams timeseriesParams;
        std::vector<real> coordinatesX , coordinatesY , coordinatesZ ;
        uint numberOfAveragedValues {};
        LevelData(ProbeData probeDataH, ProbeData probeDataD, std::vector<real> coordinatesX, std::vector<real> coordinatesY, std::vector<real> coordinatesZ)
            : probeDataH(probeDataH), probeDataD(probeDataD), coordinatesX(coordinatesX), coordinatesY(coordinatesY), coordinatesZ(coordinatesZ)
        {
        }
    };

    struct GridParams
    {
        real *velocityX, *velocityY, *velocityZ, *density;
    };

    GridParams getGridParams(LBMSimulationParameter* para);

    struct Plane
    {
        real startX, startY, startZ;
        real length, width, height;
    };

    struct Point
    {
        real x, y, z;
    };


    void addProbePointsFromList(std::vector<real> coordsX, std::vector<real> coordY, std::vector<real> coordZ)
    {
        if (coordsX.size() != coordY.size() || coordsX.size() != coordZ.size())
            throw std::runtime_error("Probe: Point coordinates must have the same size!");
        for (uint i = 0; i < coordsX.size(); i++)
            points.emplace_back(Point { coordsX[i], coordY[i], coordZ[i] });
    }

    void getTaggedFluidNodes(GridProvider* gridProvider) override;

    void addStatistic(Probe::Statistic variable);
    void addAllAvailableStatistics()
    {
        addStatistic(Probe::Statistic::Instantaneous);
        addStatistic(Probe::Statistic::Means);
        addStatistic(Probe::Statistic::Variances);
    }

    void setFileNameToNOut()
    {
        this->fileNameLU = false;
    }

    LevelData* getLevelData(int level)
    {
        return &levelDatas[level];
    }

private:
    void addLevelData(int level);
    void addPointsToLevelData(LevelData& levelData, std::vector<uint>& indices, int level);

    WbWriterVtkXmlBinary* getWriter()
    {
        return WbWriterVtkXmlBinary::getInstance();
    };
    std::vector<PostProcessingVariable> getPostProcessingVariables(Statistic variable, int level);
    std::vector<PostProcessingVariable> getAllPostProcessingVariables(int level);

    void writeParallelFile(int t);
    void writeGridFiles(int level, int t);
    void writeGridFile(int level, int t, uint part);
    void appendStatisticToNodeData(Statistic statistic, uint startPos, uint endPos, uint timestep, int level,
                                   std::vector<std::vector<double>>& nodeData);
    void appendTimeseriesFile(int level, int t);
    void appendStatisticToTimestepData(int timestep, std::vector<real>& timestepData, Statistic statistic, int level);

    std::vector<real> getTimestepData(real time, int timestep, int level);

    std::vector<std::string> getVarNames();

    uint getNumberOfTimestepsInTimeseries(int level)
    {
        if(outputTimeSeries)
            return tBetweenWriting * exp2(level);
        return 1;
    }

protected:
    SPtr<Parameter> para;
    SPtr<CudaMemoryManager> cudaMemoryManager;
    std::vector<LevelData> levelDatas;
    bool enableComputationInstantaneous {}, enableComputationMeans {}, enableComputationVariances {};
    //! flag initiating time series output.
    const bool outputTimeSeries;
    std::vector<std::string> fileNamesForCollectionFile;
    std::vector<std::string> timeseriesFileNames;

    //! if true, written file name contains time step in LU, else is the number of the written probe files
    bool fileNameLU = true;
    const uint tStartAveraging;

    const uint tBetweenAverages;
    const uint tStartWritingOutput;
    const uint tBetweenWriting;
    std::vector<Plane> planes;
    std::vector<Point> points;
};

bool isValidProbePoint(unsigned long long pointIndex, Parameter* para, int level);

#endif
//! \}
