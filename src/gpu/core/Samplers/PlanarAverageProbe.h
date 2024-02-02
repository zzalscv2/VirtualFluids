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
//! \author Henrik Asmuth
//! \date 13/05/2022
//! \brief Probe computing statistics across planes spanning the entire domain
//!
//! Computes spatial statistics across x, y or z-normal planes defined by planeNormal.
//! The planes include all points of the domain at each respective position along that normal direction.
//! The spatial statistics can additionally be averaged in time.
//!
//=======================================================================================

#ifndef PlanarAverageProbe_H
#define PlanarAverageProbe_H

#include "Sampler.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <basics/DataTypes.h>

struct PlanarAverageProbeLevelData
{
    unsigned long long *indicesOfFirstPlaneH, *indicesOfFirstPlaneD;
    uint numberOfPlanes, numberOfPointsPerPlane, numberOfTimestepsInTimeAverage;
    std::vector<real> coordinateX, coordinateY, coordinateZ;
    std::vector<std::vector<real>> instantaneous;
    std::vector<std::vector<real>> timeAverages;
};

class PlanarAverageProbe : public Sampler
{
public:
    enum class PlaneNormal { x, y, z };
    enum class Statistic {
        Means,
        Covariances,
        Skewness,
        Flatness,
    };

public:
    PlanarAverageProbe(SPtr<Parameter> para, SPtr<CudaMemoryManager> cudaMemoryManager, const std::string outputPath,
                       const std::string probeName, uint tStartAveraging, uint tStartTemporalAveraging,
                       uint tBetweenAverages, uint tStartWritingOutput, uint tBetweenWriting,
                       PlanarAverageProbe::PlaneNormal planeNormal, bool computeTimeAverages)
        : tStartAveraging(tStartAveraging), tStartTemporalAveraging(tStartTemporalAveraging),
          tBetweenAverages(tBetweenAverages), tStartWritingOutput(tStartWritingOutput), tBetweenWriting(tBetweenWriting),
          computeTimeAverages(computeTimeAverages), planeNormal(planeNormal),
          Sampler(para, cudaMemoryManager, outputPath, probeName)
    {
        if (tStartTemporalAveraging < tStartAveraging && computeTimeAverages)
            throw std::runtime_error("PlaneAverageProbe: tStartTemporalAveraging must be larger than tStartAveraging!");
        if(tBetweenWriting == 0)
            throw std::runtime_error("PlaneAverageProbe: tBetweenWriting must be larger than 0!");
    }
    ~PlanarAverageProbe();

    void init() override;
    void sample(int level, uint t) override;
    void getTaggedFluidNodes(GridProvider* gridProvider) override {};
    PlanarAverageProbeLevelData* getLevelData(int level)
    {
        return &levelData[level];
    }
    void addAllAvailableStatistics()
    {
        addStatistic(PlanarAverageProbe::Statistic::Means);
        addStatistic(PlanarAverageProbe::Statistic::Covariances);
        addStatistic(PlanarAverageProbe::Statistic::Skewness);
        addStatistic(PlanarAverageProbe::Statistic::Flatness);
    }
    void addStatistic(PlanarAverageProbe::Statistic statistic);
    void setFileNameToNOut()
    {
        nameFilesWithFileCount = true;
    }

private:
    std::vector<PostProcessingVariable> getPostProcessingVariables(PlanarAverageProbe::Statistic variable,
                                                                   bool includeTimeAverage);
    void calculateQuantities(PlanarAverageProbeLevelData* data, uint t, int level, bool doTimeAverages);
    std::vector<unsigned long long> findIndicesInPlane(int level);
    void findCoordinatesForPlanes(int level, std::vector<real>& coordinateX, std::vector<real>& coordinateY,
                                  std::vector<real>& coordinateZ);

    std::vector<std::string> getVarNames(bool includeTimeAverage);
    const uint* getNeighborIndicesInPlaneNormal(int level);
    void writeGridFile(int level, uint t);
    void writeParallelFile(uint t);

private:
    uint tStartAveraging, tStartTemporalAveraging, tBetweenAverages, tStartWritingOutput, tBetweenWriting;
    bool computeTimeAverages, nameFilesWithFileCount = false;
    PlaneNormal planeNormal;
    std::vector<PlanarAverageProbe::Statistic> statistics;
    std::vector<PlanarAverageProbeLevelData> levelData;
    std::vector<std::string> fileNamesForCollectionFile;
};

bool isStatisticIn(PlanarAverageProbe::Statistic statistic, std::vector<PlanarAverageProbe::Statistic> statistics);

#endif
//! \}
