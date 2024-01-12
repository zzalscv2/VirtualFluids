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
#include "TimeseriesFileWriter.h"

#include <string>
#include <vector>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <filesystem>


#include <basics/DataTypes.h>

void TimeseriesFileWriter::writeHeader(const std::string& fileName, int numberOfPoints, std::vector<std::string>& variableNames, const real* coordsX, const real* coordsY, const real* coordsZ)
{
    std::filesystem::create_directories(std::filesystem::path(fileName).parent_path());
    std::ofstream out(fileName.c_str(), std::ios::out | std::ios::binary);

    if (!out.is_open())
        throw std::runtime_error("Could not open timeseries file " + fileName + "!");

    out << "TimeseriesOutput \n";
    out << "Quantities: ";
    for (const std::string& name : variableNames)
        out << name << ", ";
    out << "\n";
    out << "Number of points in this file: \n";
    out << numberOfPoints << "\n";
    out << "Positions: x, y, z\n";
    for (int i = 0; i < numberOfPoints; i++)
        out << coordsX[i] << ", " << coordsY[i] << ", " << coordsZ[i]
    << "\n";

    out.close();
}

void TimeseriesFileWriter::appendData(const std::string& fileName, std::vector<std::vector<real>>& data) {
    std::ofstream out(fileName.c_str(), std::ios::app | std::ios::binary);
    for (auto& timestepData : data) {
        out.write((char*)timestepData.data(), sizeof(real) * timestepData.size());
    }
    out.close();
}

//! \}