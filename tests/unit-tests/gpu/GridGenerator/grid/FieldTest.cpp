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
//! \addtogroup gpu_grid_tests grid
//! \ingroup gpu_GridGenerator_tests GridGenerator
//! \{
//=======================================================================================
#include <gmock/gmock.h>

#include <gpu/GridGenerator/grid/Field.h>
#include <gpu/GridGenerator/grid/NodeValues.h>

using namespace vf::gpu;

TEST(FieldTest, isFluid)
{
    auto field = Field(1);
    field.allocateMemory();
    uint index = 0;

    field.setFieldEntry(index, FLUID);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: FLUID";

    field.setFieldEntry(index, FLUID_CFC);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: FLUID_CFC";
    field.setFieldEntry(index, FLUID_FCF);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: FLUID_FCF";
    field.setFieldEntry(index, FLUID_CFF);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: FLUID_CFF";
    field.setFieldEntry(index, FLUID_FCC);

    field.setFieldEntry(index, BC_PRESSURE);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: BC_PRESSURE";
    field.setFieldEntry(index, BC_VELOCITY);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: BC_VELOCITY";
    field.setFieldEntry(index, BC_SOLID);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: BC_SOLID";
    field.setFieldEntry(index, BC_SLIP);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: BC_SLIP";
    field.setFieldEntry(index, BC_NOSLIP);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: BC_NOSLIP";
    field.setFieldEntry(index, BC_OUTFLOW);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: BC_OUTFLOW";
    field.setFieldEntry(index, BC_STRESS);
    EXPECT_TRUE(field.isFluid(index)) << "tested type: BC_STRESS";

    field.setFieldEntry(index, STOPPER_OUT_OF_GRID);
    EXPECT_FALSE(field.isFluid(index)) << "tested type: STOPPER_OUT_OF_GRID";
    field.setFieldEntry(index, STOPPER_COARSE_UNDER_FINE);
    EXPECT_FALSE(field.isFluid(index)) << "tested type: STOPPER_COARSE_UNDER_FINE";
    field.setFieldEntry(index, STOPPER_SOLID);
    EXPECT_FALSE(field.isFluid(index)) << "tested type: STOPPER_SOLID";
    field.setFieldEntry(index, STOPPER_OUT_OF_GRID_BOUNDARY);
    EXPECT_FALSE(field.isFluid(index)) << "tested type: STOPPER_OUT_OF_GRID_BOUNDARY";

    field.setFieldEntry(index, INVALID_OUT_OF_GRID);
    EXPECT_FALSE(field.isFluid(index)) << "tested type: INVALID_OUT_OF_GRID";
    field.setFieldEntry(index, INVALID_COARSE_UNDER_FINE);
    EXPECT_FALSE(field.isFluid(index)) << "tested type: INVALID_COARSE_UNDER_FINE";
    field.setFieldEntry(index, INVALID_SOLID);
    EXPECT_FALSE(field.isFluid(index)) << "tested type: INVALID_SOLID";

    field.freeMemory();
}

TEST(FieldTest, isBoundaryConditionNode)
{
    auto field = Field(1);
    field.allocateMemory();
    uint index = 0;

    EXPECT_FALSE(field.isBoundaryConditionNode(index));

    field.setFieldEntry(index, FLUID);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: FLUID";
    field.setFieldEntry(index, FLUID_CFC);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: FLUID_CFC";
    field.setFieldEntry(index, FLUID_FCF);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: FLUID_FCF";
    field.setFieldEntry(index, FLUID_CFF);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: FLUID_CFF";
    field.setFieldEntry(index, FLUID_FCC);

    field.setFieldEntry(index, BC_PRESSURE);
    EXPECT_TRUE(field.isBoundaryConditionNode(index)) << "tested type: BC_PRESSURE";
    field.setFieldEntry(index, BC_VELOCITY);
    EXPECT_TRUE(field.isBoundaryConditionNode(index)) << "tested type: BC_VELOCITY";
    field.setFieldEntry(index, BC_SOLID);
    EXPECT_TRUE(field.isBoundaryConditionNode(index)) << "tested type: BC_SOLID";
    field.setFieldEntry(index, BC_SLIP);
    EXPECT_TRUE(field.isBoundaryConditionNode(index)) << "tested type: BC_SLIP";
    field.setFieldEntry(index, BC_NOSLIP);
    EXPECT_TRUE(field.isBoundaryConditionNode(index)) << "tested type: BC_NOSLIP";
    field.setFieldEntry(index, BC_OUTFLOW);
    EXPECT_TRUE(field.isBoundaryConditionNode(index)) << "tested type: BC_OUTFLOW";
    field.setFieldEntry(index, BC_STRESS);
    EXPECT_TRUE(field.isBoundaryConditionNode(index)) << "tested type: BC_STRESS";

    field.setFieldEntry(index, STOPPER_OUT_OF_GRID);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: STOPPER_OUT_OF_GRID";
    field.setFieldEntry(index, STOPPER_COARSE_UNDER_FINE);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: STOPPER_COARSE_UNDER_FINE";
    field.setFieldEntry(index, STOPPER_SOLID);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: STOPPER_SOLID";
    field.setFieldEntry(index, STOPPER_OUT_OF_GRID_BOUNDARY);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: STOPPER_OUT_OF_GRID_BOUNDARY";

    field.setFieldEntry(index, INVALID_OUT_OF_GRID);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: INVALID_OUT_OF_GRID";
    field.setFieldEntry(index, INVALID_COARSE_UNDER_FINE);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: INVALID_COARSE_UNDER_FINE";
    field.setFieldEntry(index, INVALID_SOLID);
    EXPECT_FALSE(field.isBoundaryConditionNode(index)) << "tested type: INVALID_SOLID";

    field.freeMemory();
}

//! \}
