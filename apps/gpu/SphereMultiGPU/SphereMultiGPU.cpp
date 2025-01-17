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
//! \addtogroup SphereMultiGPU
//! \ingroup gpu_apps
//! \{
//! \author Anna Wellmann
//=======================================================================================
#include <string>

#include <basics/DataTypes.h>
#include <basics/StringUtilities/StringUtil.h>
#include <basics/config/ConfigurationFile.h>

#include <logger/Logger.h>

#include <parallel/MPICommunicator.h>

#include "GridGenerator/geometries/Cuboid/Cuboid.h"
#include "GridGenerator/geometries/Sphere/Sphere.h"
#include "GridGenerator/geometries/TriangularMesh/TriangularMesh.h"
#include "GridGenerator/grid/BoundaryConditions/Side.h"
#include "GridGenerator/grid/GridBuilder/LevelGridBuilder.h"
#include "GridGenerator/grid/GridBuilder/MultipleGridBuilder.h"
#include "GridGenerator/grid/MultipleGridBuilderFacade.h"
#include "GridGenerator/io/GridVTKWriter/GridVTKWriter.h"
#include "GridGenerator/io/STLReaderWriter/STLReader.h"
#include "GridGenerator/io/STLReaderWriter/STLWriter.h"
#include "GridGenerator/io/SimulationFileWriter/SimulationFileWriter.h"
#include "GridGenerator/utilities/communication.h"

#include "gpu/core/BoundaryConditions/BoundaryConditionFactory.h"
#include "gpu/core/Calculation/Simulation.h"
#include "gpu/core/GridScaling/GridScalingFactory.h"
#include "gpu/core/Kernel/KernelFactory/KernelFactoryImp.h"
#include "gpu/core/Kernel/KernelTypes.h"
#include "gpu/core/Parameter/Parameter.h"
#include "gpu/core/PreProcessor/PreProcessorFactory/PreProcessorFactoryImp.h"

void run(const vf::basics::ConfigurationFile& config)
{
    vf::parallel::Communicator& communicator = *vf::parallel::MPICommunicator::getInstance();
    const auto numberOfProcesses = communicator.getNumberOfProcesses();
    const auto processID = communicator.getProcessID();
    SPtr<Parameter> para = std::make_shared<Parameter>(numberOfProcesses, processID, &config);
    BoundaryConditionFactory bcFactory = BoundaryConditionFactory();
    GridScalingFactory scalingFactory = GridScalingFactory();

    // configure simulation parameters

    const bool useLevels = true;

    const std::string outPath("output/" + std::to_string(para->getNumprocs()) + "GPU/");
    const std::string gridPath = "output/";
    std::string simulationName("SphereMultiGPU");

    para->useReducedCommunicationAfterFtoC = para->getNumprocs() != 1;

    const real length = 1.0;
    const real reynoldsNumber = 1000.0;
    const real velocity = 1.0;
    const real velocityLB = 0.05;
    const uint numberOfNodesX = 80;

    // compute  parameters in lattcie units

    const real dxGrid = length / real(numberOfNodesX);
    const real deltaT = velocityLB / velocity * dxGrid;
    const real viscosityLB = numberOfNodesX * velocityLB / reynoldsNumber;

    // set parameters

    para->setVelocityLB(velocityLB);
    para->setViscosityLB(viscosityLB);
    para->setVelocityRatio(velocity / velocityLB);
    para->setDensityRatio((real)1.0);

    para->setOutputPath(outPath);
    para->setOutputPrefix(simulationName);
    para->setPrintFiles(true);

    para->configureMainKernel(vf::collisionKernel::compressible::K17CompressibleNavierStokes);
    scalingFactory.setScalingFactory(GridScalingFactory::GridScaling::ScaleCompressible);

    vf::logging::Logger::changeLogPath(outPath + "vflog_process" + std::to_string(processID) );
    vf::logging::Logger::initializeLogger();

    // log simulation parameters

    VF_LOG_INFO("LB parameters:");
    VF_LOG_INFO("velocity LB [dx/dt]              = {}", velocityLB);
    VF_LOG_INFO("viscosity LB [dx/dt]             = {}", viscosityLB);
    VF_LOG_INFO("dxGrid [-]                       = {}\n", dxGrid);
    VF_LOG_INFO("deltaT [s]                       = {}", deltaT);
    VF_LOG_INFO("simulation parameters:");
    VF_LOG_INFO("mainKernel                       = {}\n", para->getMainKernel());

    // configure simulation grid

    UPtr<MultipleGridBuilderFacade> gridBuilderFacade;

    auto domainDimensions = std::make_shared<GridDimensions>();
    domainDimensions->minX = -0.5 * length;
    domainDimensions->maxX = 0.5 * length;
    domainDimensions->minY = -0.5 * length;
    domainDimensions->maxY = 0.5 * length;
    domainDimensions->minZ = -0.5 * length;
    domainDimensions->maxZ = 0.5 * length;
    domainDimensions->delta = dxGrid;

    gridBuilderFacade = std::make_unique<MultipleGridBuilderFacade>(std::move(domainDimensions), 8. * dxGrid);

    gridBuilderFacade->addGeometry(std::make_shared<Sphere>(0.0, 0.0, 0.0, 0.1 * length));

    std::shared_ptr<Object> level1 = nullptr;
    if (useLevels) {
        gridBuilderFacade->setNumberOfLayersForRefinement(10, 8);
        level1 = std::make_shared<Sphere>(0.0, 0.0, 0.0, 0.25 * length);
        gridBuilderFacade->addFineGrid(level1, 1);
    }

    // configure subdomains for simulation on multiple gpus

    const real xSplit = 0.0;
    const real ySplit = 0.0;
    const real zSplit = 0.0;

    if (numberOfProcesses == 2) {
        gridBuilderFacade->addDomainSplit(zSplit, Axis::z);
    } else if (numberOfProcesses == 4) {
        gridBuilderFacade->addDomainSplit(xSplit, Axis::y);
        gridBuilderFacade->addDomainSplit(zSplit, Axis::z);
    } else if (numberOfProcesses == 8) {
        gridBuilderFacade->addDomainSplit(xSplit, Axis::x);
        gridBuilderFacade->addDomainSplit(ySplit, Axis::y);
        gridBuilderFacade->addDomainSplit(zSplit, Axis::z);
    }

    // create grids
    gridBuilderFacade->createGrids(processID);

    // configure boundary conditions

    // call after createGrids()
    gridBuilderFacade->setPeriodicBoundaryCondition(false, false, false);
    gridBuilderFacade->setVelocityBoundaryCondition(SideType::MX, velocityLB, 0.0, 0.0);
    gridBuilderFacade->setVelocityBoundaryCondition(SideType::MY, velocityLB, 0.0, 0.0);
    gridBuilderFacade->setVelocityBoundaryCondition(SideType::PY, velocityLB, 0.0, 0.0);
    gridBuilderFacade->setVelocityBoundaryCondition(SideType::MZ, velocityLB, 0.0, 0.0);
    gridBuilderFacade->setVelocityBoundaryCondition(SideType::PZ, velocityLB, 0.0, 0.0);
    gridBuilderFacade->setPressureBoundaryCondition(SideType::PX, 0.0);

    bcFactory.setVelocityBoundaryCondition(BoundaryConditionFactory::VelocityBC::VelocityInterpolatedCompressible);
    bcFactory.setPressureBoundaryCondition(BoundaryConditionFactory::PressureBC::PressureNonEquilibriumCompressible);

    Simulation simulation(para, gridBuilderFacade->getGridBuilder(), &bcFactory, &scalingFactory);
    simulation.run();
}

int main(int argc, char* argv[])
{
    try {
        vf::logging::Logger::initializeLogger();
        vf::basics::ConfigurationFile config =
            vf::basics::loadConfig(argc, argv, "./apps/gpu/SphereMultiGPU/sphere_1gpu.cfg");
        run(config);
    } catch (const std::exception& e) {
        VF_LOG_WARNING("{}", e.what());
        return 1;
    }
    return 0;
}

//! \}
