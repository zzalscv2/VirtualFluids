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
//! \addtogroup NumericalTests
//! \ingroup numerical_tests
//! \{
//=======================================================================================
#ifndef SHEAR_WAVE_INFORMATION_H
#define SHEAR_WAVE_INFORMATION_H

#include "Utilities/LogFileInformation/LogFileInformationImp.h"
#include "Utilities/LogFileInformation/SimulationLogFileInformation/SimulationLogFileInformation.h"

#include "Calculation/Calculation.h"

#include <memory>
#include <vector>

struct ShearWaveParameterStruct;
struct GridInformationStruct;

class ShearWaveInformation : public SimulationLogFileInformation
{
public:
    static std::shared_ptr<ShearWaveInformation> getNewInstance(std::shared_ptr<ShearWaveParameterStruct> simParaStruct, std::vector<std::shared_ptr<GridInformationStruct> > gridInfoStruct);

    std::string getOutput();
    std::vector<std::string> getFilePathExtension();

private:
    ShearWaveInformation() {};
    ShearWaveInformation(std::shared_ptr<ShearWaveParameterStruct> simParaStruct, std::vector<std::shared_ptr<GridInformationStruct> > gridInfoStruct);

    double ux;
    double uz;
    std::vector<real> lx;
    int l0;
};
#endif
//! \}
