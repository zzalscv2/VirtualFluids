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
//! \addtogroup gpu_Probes Utilities
//! \ingroup gpu_core core
//! \{
#include <basics/StringUtilities/StringUtil.h>
#include <string>

template <typename T>
inline std::string nameComponent(const std::string& name, T value)
{
    return "_" + name + "_" + StringUtil::toString<T>(value);
}

inline std::string makeParallelFileName(const std::string& probeName, int id, int t)
{
    return probeName + "_bin" + nameComponent("ID", id) + nameComponent("t", t) + ".vtk";
}

inline std::string makeGridFileName(const std::string& probeName, int level, int id, int t, uint part)
{
    return probeName + "_bin" + nameComponent("lev", level) + nameComponent("ID", id) + nameComponent<int>("Part", part) +
           nameComponent("t", t) + ".vtk";
}

inline std::string makeTimeseriesFileName(const std::string& probeName, int level, int id)
{
    return probeName + "_timeseries" + nameComponent("lev", level) + nameComponent("ID", id) + ".txt";
}

template <typename T>
__host__ __device__ inline T computeNewTimeAverage(T oldAverage, T newValue, real inverseNumberOfTimesteps){
    return oldAverage + (newValue - oldAverage) * inverseNumberOfTimesteps;
}
//! \}