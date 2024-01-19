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
#ifndef SIMULATION_PARAMETER_TAYLORGREENVORTEX_UZ_H
#define SIMULATION_PARAMETER_TAYLORGREENVORTEX_UZ_H

#include "Utilities/SimulationParameter/SimulationParameterImp.h"

#include <string>
#include <memory>

struct TaylorGreenVortexUzParameterStruct;
struct GridInformationStruct;

class SimulationParameterTaylorGreenUz : public SimulationParameterImp
{
public:
    static std::shared_ptr<SimulationParameterTaylorGreenUz> getNewInstance(std::string kernel, double viscosity, std::shared_ptr<TaylorGreenVortexUzParameterStruct> tgvParameterStruct, std::shared_ptr<GridInformationStruct> gridInfo);
    
protected:
    SimulationParameterTaylorGreenUz(std::string kernel, double viscosity, std::shared_ptr<TaylorGreenVortexUzParameterStruct> tgvParameterStruct, std::shared_ptr<GridInformationStruct> gridInfo);

};
#endif 

//! \}
