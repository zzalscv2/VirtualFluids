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
//! \addtogroup gpu_Probes Probes
//! \ingroup gpu_core core
//! \{
#ifndef TIMESERIESFILEWRITER_H
#define TIMESERIESFILEWRITER_H
#include <string>
#include <vector>

#include <basics/DataTypes.h>

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

class TimeseriesFileWriter{
public: 
    TimeseriesFileWriter() = default;

    static void writeHeader(const std::string& fileName, int numberOfPoints, std::vector<std::string>& variableNames, const real* coordsX, const real* coordsY, const real* coordsZ);
    static void appendData(const std::string& fileName, std::vector<std::vector<real>>& data);
};

#endif

//! \}