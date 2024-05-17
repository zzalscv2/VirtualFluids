// //=======================================================================================
// // ____          ____    __    ______     __________   __      __       __        __
// // \    \       |    |  |  |  |   _   \  |___    ___| |  |    |  |     /  \      |  |
// //  \    \      |    |  |  |  |  |_)   |     |  |     |  |    |  |    /    \     |  |
// //   \    \     |    |  |  |  |   _   /      |  |     |  |    |  |   /  /\  \    |  |
// //    \    \    |    |  |  |  |  | \  \      |  |     |   \__/   |  /  ____  \   |  |____
// //     \    \   |    |  |__|  |__|  \__\     |__|      \________/  /__/    \__\  |_______|
// //      \    \  |    |   ________________________________________________________________
// //       \    \ |    |  |  ______________________________________________________________|
// //        \    \|    |  |  |         __          __     __     __     ______      _______
// //         \         |  |  |_____   |  |        |  |   |  |   |  |   |   _  \    /  _____)
// //          \        |  |   _____|  |  |        |  |   |  |   |  |   |  | \  \   \_______
// //           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  |
// //            \ _____|  |__|        |________|   \_______/    |__|   |______/    (_______/
// //
// //  This file is part of VirtualFluids. VirtualFluids is free software: you can
// //  redistribute it and/or modify it under the terms of the GNU General Public
// //  License as published by the Free Software Foundation, either version 3 of
// //  the License, or (at your option) any later version.
// //
// //  VirtualFluids is distributed in the hope that it will be useful, but WITHOUT
// //  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// //  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// //  for more details.
// //
// //  SPDX-License-Identifier: GPL-3.0-or-later
// //  SPDX-FileCopyrightText: Copyright © VirtualFluids Project contributors, see AUTHORS.md in root folder
// //
// //! \addtogroup gpu_PreCollisionInteractor PreCollisionInteractor
// //! \ingroup gpu_core core
// //! \{
// //! \author Henry Korb, Henrik Asmuth
// //! \date 13/05/2022
// //! \brief Probe computing statistics for a set of points in space
// //!
// //! The set of points can be defined by providing a list or on an x-normal plane (the latter being somewhat redundant with
// //! PlaneProbe) All statistics are temporal.
// //!
// //=======================================================================================

// #ifndef PointProbe_H
// #define PointProbe_H

// #include "Probe.h"

// class PointProbe : public Probe
// {
// public:
//     PointProbe(SPtr<Parameter> para, SPtr<CudaMemoryManager> cudaMemoryManager, const std::string outputPath,
//                const std::string probeName, uint tStartAvg, uint tBetweenAverages, uint tStartWritingOutput,
//                uint tBetweenWriting, bool outputTimeseries = false)
//         : Probe(para, cudaMemoryManager, outputPath, probeName, tStartAvg, 0, tBetweenAverages, tStartWritingOutput,
//                 tBetweenWriting, true, outputTimeseries)
//     {
//     }

//     ~PointProbe() = default;

//     void addProbePoint(real pointCoordX, real pointCoordY, real pointCoordZ);
//     void addProbePointsFromList(std::vector<real>& _pointCoordsX, std::vector<real>& _pointCoordsY,
//                                 std::vector<real>& _pointCoordsZ);
//     void getTaggedFluidNodes(GridProvider* gridProvider) override;

// private:
//     bool isAvailableStatistic(Statistic variable) override;

//     std::vector<PostProcessingVariable> getPostProcessingVariables(Statistic variable) override;

//     void findPoints(std::vector<int>& probeIndices, std::vector<real>& distancesX, std::vector<real>& distancesY,
//                     std::vector<real>& distancesZ, std::vector<real>& pointCoordsX, std::vector<real>& pointCoordsY,
//                     std::vector<real>& pointCoordsZ, int level) override;

//     void calculateQuantities(SPtr<ProbeStruct> probeStruct, uint t, int level) override;
//     uint getNumberOfTimestepsInTimeseries(int level) override
//     {
        
//     }

// private:
//     std::vector<real> pointCoordsX, pointCoordsY, pointCoordsZ;
// };

// #endif
// //! \}
