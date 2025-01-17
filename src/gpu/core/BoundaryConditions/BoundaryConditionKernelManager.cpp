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
//! \addtogroup gpu_BoundaryConditions BoundaryConditions
//! \ingroup gpu_core core
//! \{
//! \author Martin Schoenherr
//=======================================================================================
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <stdexcept>
#include <string>

#include "BoundaryConditionKernelManager.h"
#include "BoundaryConditions/BoundaryConditionFactory.h"
#include "GridGenerator/TransientBCSetter/TransientBCSetter.h"
#include "Parameter/Parameter.h"
#include "PostProcessor/CP27.cuh"
#include "PostProcessor/Cp.h"
#include "PostProcessor/DragLift.cuh"
#include "PostProcessor/DragLift.h"

#include "BoundaryConditions/Outflow/Outflow.h"
#include "BoundaryConditions/Pressure/Pressure.h"


BoundaryConditionKernelManager::BoundaryConditionKernelManager(SPtr<Parameter> parameter, const BoundaryConditionFactory *bcFactory) : para(parameter)
{
    this->velocityBoundaryConditionPost = bcFactory->getVelocityBoundaryConditionPost();
    this->noSlipBoundaryConditionPost = bcFactory->getNoSlipBoundaryConditionPost();
    this->slipBoundaryConditionPost = bcFactory->getSlipBoundaryConditionPost();
    this->geometryBoundaryConditionPost = bcFactory->getGeometryBoundaryConditionPost();
    this->stressBoundaryConditionPost = bcFactory->getStressBoundaryConditionPost();
    this->precursorBoundaryConditionPost = bcFactory->getPrecursorBoundaryConditionPost();

    if (bcFactory->hasDirectionalPressureBoundaryCondition())
        this->directionalPressureBoundaryConditionPre = std::get<directionalBoundaryCondition>(bcFactory->getPressureBoundaryConditionPre());
    else
        this->pressureBoundaryConditionPre = std::get<boundaryCondition>(bcFactory->getPressureBoundaryConditionPre());

    checkBoundaryCondition(this->velocityBoundaryConditionPost, this->para->getParD(0)->velocityBC,
                           "velocityBoundaryConditionPost");
    checkBoundaryCondition(this->noSlipBoundaryConditionPost, this->para->getParD(0)->noSlipBC,
                           "noSlipBoundaryConditionPost");
    checkBoundaryCondition(this->slipBoundaryConditionPost, this->para->getParD(0)->slipBC,
                           "slipBoundaryConditionPost");
    checkBoundaryCondition(this->geometryBoundaryConditionPost, this->para->getParD(0)->geometryBC,
                           "geometryBoundaryConditionPost");
    checkBoundaryCondition(this->stressBoundaryConditionPost, this->para->getParD(0)->stressBC,
                           "stressBoundaryConditionPost");
    checkBoundaryCondition(this->precursorBoundaryConditionPost, this->para->getParD(0)->precursorBC,
                           "precursorBoundaryConditionPost");
    checkBoundaryCondition(this->pressureBoundaryConditionPre, this->para->getParD(0)->pressureBC,
                           "pressureBoundaryConditionPre");
    checkBoundaryCondition(this->directionalPressureBoundaryConditionPre, this->para->getParD(0)->pressureBCDirectional,
                           "directionalPressureBoundaryConditionPre");
}

void BoundaryConditionKernelManager::runVelocityBCKernelPre(int level) const
{
    if (para->getParD(level)->velocityBC.numberOfBCnodes > 0)
    {
        ////////////////////////////////////////////////////////////////////////////
        // high viscosity incompressible
        // QVelDevIncompHighNu27(
        //     para->getParD(level)->numberofthreads,
        //     para->getParD(level)->velocityBC.Vx,
        //     para->getParD(level)->velocityBC.Vy,
        //     para->getParD(level)->velocityBC.Vz,
        //     para->getParD(level)->distributions.f[0],
        //     para->getParD(level)->velocityBC.k,
        //     para->getParD(level)->velocityBC.q27[0],
        //     para->getParD(level)->velocityBC.numberOfBCnodes,
        //     para->getParD(level)->omega,
        //     para->getParD(level)->neighborX,
        //     para->getParD(level)->neighborY,
        //     para->getParD(level)->neighborZ,
        //     para->getParD(level)->numberOfNodes,
        //     para->getParD(level)->isEvenTimestep);

        ////////////////////////////////////////////////////////////////////////////
        // high viscosity compressible
        // QVelDevCompHighNu27(
        //     para->getParD(level)->numberofthreads,
        //     para->getParD(level)->velocityBC.Vx,
        //     para->getParD(level)->velocityBC.Vy,
        //     para->getParD(level)->velocityBC.Vz,
        //     para->getParD(level)->distributions.f[0],
        //     para->getParD(level)->velocityBC.k,
        //     para->getParD(level)->velocityBC.q27[0],
        //     para->getParD(level)->velocityBC.numberOfBCnodes,
        //     para->getParD(level)->omega,
        //     para->getParD(level)->neighborX,
        //     para->getParD(level)->neighborY,
        //     para->getParD(level)->neighborZ,
        //     para->getParD(level)->numberOfNodes,
        //     para->getParD(level)->isEvenTimestep);
    }
}

void BoundaryConditionKernelManager::runVelocityBCKernelPost(int level) const
{
     if (para->getParD(level)->velocityBC.numberOfBCnodes > 0)
     {
        velocityBoundaryConditionPost(para->getParD(level).get(), &(para->getParD(level)->velocityBC));

        //////////////////////////////////////////////////////////////////////////
        // D E P R E C A T E D
        //////////////////////////////////////////////////////////////////////////

        // QVelDevice1h27( para->getParD(level)->numberofthreads, para->getParD(level)->nx,           para->getParD(level)->ny,
        //                para->getParD(level)->velocityBC.Vx,      para->getParD(level)->velocityBC.Vy,   para->getParD(level)->velocityBC.Vz,
        //                para->getParD(level)->distributions.f[0],       para->getParD(level)->velocityBC.k,    para->getParD(level)->velocityBC.q27[0],
        //                para->getParD(level)->velocityBC.numberOfBCnodes,      para->getParD(level)->omega,
        //                para->getPhi(),                        para->getAngularVelocity(),
        //                para->getParD(level)->neighborX,    para->getParD(level)->neighborY, para->getParD(level)->neighborZ,
        //                para->getParD(level)->coordinateX,       para->getParD(level)->coordinateY,    para->getParD(level)->coordinateZ,
        //                para->getParD(level)->numberOfNodes,     para->getParD(level)->isEvenTimestep);
        // getLastCudaError("QVelDev27 execution failed");
     }
}

void BoundaryConditionKernelManager::runGeoBCKernelPre(int level, unsigned int t, CudaMemoryManager* cudaMemoryManager) const{
    if (para->getParD(level)->geometryBC.numberOfBCnodes > 0){
        if (para->getCalcDragLift())
        {
            //Drag and Lift Part II
            DragLiftPreD27(
                para->getParD(level)->distributions.f[0],
                para->getParD(level)->geometryBC.k,
                para->getParD(level)->geometryBC.q27[0],
                para->getParD(level)->geometryBC.numberOfBCnodes,
                para->getParD(level)->DragLiftPreProcessingInXdirection,
                para->getParD(level)->DragLiftPreProcessingInYdirection,
                para->getParD(level)->DragLiftPreProcessingInZdirection,
                para->getParD(level)->neighborX,
                para->getParD(level)->neighborY,
                para->getParD(level)->neighborZ,
                para->getParD(level)->numberOfNodes,
                para->getParD(level)->isEvenTimestep,
                para->getParD(level)->numberofthreads);
            ////////////////////////////////////////////////////////////////////////////////
            //Calculation of Drag and Lift
            ////////////////////////////////////////////////////////////////////////////////
            calcDragLift(para.get(), cudaMemoryManager, level);
            ////////////////////////////////////////////////////////////////////////////////
        }

        if (para->getCalcCp())
        {
            ////////////////////////////////////////////////////////////////////////////////
            //Calculation of cp
            ////////////////////////////////////////////////////////////////////////////////

            if(t > para->getTimestepStartOut())
            {
                ////////////////////////////////////////////////////////////////////////////////
                CalcCPtop27(
                    para->getParD(level)->distributions.f[0],
                    para->getParD(level)->cpTopIndex,
                    para->getParD(level)->numberOfPointsCpTop,
                    para->getParD(level)->cpPressTop,
                    para->getParD(level)->neighborX,
                    para->getParD(level)->neighborY,
                    para->getParD(level)->neighborZ,
                    para->getParD(level)->numberOfNodes,
                    para->getParD(level)->isEvenTimestep,
                    para->getParD(level)->numberofthreads);
                //////////////////////////////////////////////////////////////////////////////////
                CalcCPbottom27(
                    para->getParD(level)->distributions.f[0],
                    para->getParD(level)->cpBottomIndex,
                    para->getParD(level)->numberOfPointsCpBottom,
                    para->getParD(level)->cpPressBottom,
                    para->getParD(level)->neighborX,
                    para->getParD(level)->neighborY,
                    para->getParD(level)->neighborZ,
                    para->getParD(level)->numberOfNodes,
                    para->getParD(level)->isEvenTimestep,
                    para->getParD(level)->numberofthreads);
                //////////////////////////////////////////////////////////////////////////////////
                CalcCPbottom27(
                    para->getParD(level)->distributions.f[0],
                    para->getParD(level)->cpBottom2Index,
                    para->getParD(level)->numberOfPointsCpBottom2,
                    para->getParD(level)->cpPressBottom2,
                    para->getParD(level)->neighborX,
                    para->getParD(level)->neighborY,
                    para->getParD(level)->neighborZ,
                    para->getParD(level)->numberOfNodes,
                    para->getParD(level)->isEvenTimestep,
                    para->getParD(level)->numberofthreads);
                //////////////////////////////////////////////////////////////////////////////////
                calcCp(para.get(), cudaMemoryManager, level);
            }            
        }

        ////////////////////////////////////////////////////////////////////////////////
        // high viscosity incompressible
        // QDevIncompHighNu27(
        //     para->getParD(level)->numberofthreads,
        //     para->getParD(level)->distributions.f[0],
        //     para->getParD(level)->geometryBC.k,
        //     para->getParD(level)->geometryBC.q27[0],
        //     para->getParD(level)->geometryBC.numberOfBCnodes,
        //     para->getParD(level)->omega,
        //     para->getParD(level)->neighborX,
        //     para->getParD(level)->neighborY,
        //     para->getParD(level)->neighborZ,
        //     para->getParD(level)->numberOfNodes,
        //     para->getParD(level)->isEvenTimestep);

        //////////////////////////////////////////////////////////////////////////////////
        // high viscosity compressible
        // QDevCompHighNu27(
        //     para->getParD(level)->numberofthreads,
        //     para->getParD(level)->distributions.f[0],
        //     para->getParD(level)->geometryBC.k,
        //     para->getParD(level)->geometryBC.q27[0],
        //     para->getParD(level)->geometryBC.numberOfBCnodes,
        //     para->getParD(level)->omega,
        //     para->getParD(level)->neighborX,
        //     para->getParD(level)->neighborY,
        //     para->getParD(level)->neighborZ,
        //     para->getParD(level)->numberOfNodes,
        //     para->getParD(level)->isEvenTimestep);

    }
}

void BoundaryConditionKernelManager::runGeoBCKernelPost(int level) const
{
    if (para->getParD(level)->geometryBC.numberOfBCnodes > 0)
    {
        if (para->getCalcDragLift())
        {
            //Drag and Lift Part I
            DragLiftPostD27(para->getParD(level)->distributions.f[0],
                            para->getParD(level)->geometryBC.k,
                            para->getParD(level)->geometryBC.q27[0],
                            para->getParD(level)->geometryBC.numberOfBCnodes,
                            para->getParD(level)->DragLiftPostProcessingInXdirection,
                            para->getParD(level)->DragLiftPostProcessingInYdirection,
                            para->getParD(level)->DragLiftPostProcessingInZdirection,
                            para->getParD(level)->neighborX,
                            para->getParD(level)->neighborY,
                            para->getParD(level)->neighborZ,
                            para->getParD(level)->numberOfNodes,
                            para->getParD(level)->isEvenTimestep,
                            para->getParD(level)->numberofthreads);
            getLastCudaError("DragLift27 execution failed");
        }

        geometryBoundaryConditionPost(para->getParD(level).get(), &(para->getParD(level)->geometryBC));

        //////////////////////////////////////////////////////////////////////////
        // D E P R E C A T E D
        //////////////////////////////////////////////////////////////////////////
        // the GridGenerator does currently not provide normals!

        //     QSlipGeomDevComp27(
        //         para->getParD(level)->numberofthreads,
        //         para->getParD(level)->distributions.f[0],
        //         para->getParD(level)->geometryBC.k,
        //         para->getParD(level)->geometryBC.q27[0],
        //         para->getParD(level)->geometryBC.numberOfBCnodes,
        //         para->getParD(level)->omega,
        //         para->getParD(level)->geometryBCnormalX.q27[0],
        //         para->getParD(level)->geometryBCnormalY.q27[0],
        //         para->getParD(level)->geometryBCnormalZ.q27[0],
        //         para->getParD(level)->neighborX,
        //         para->getParD(level)->neighborY,
        //         para->getParD(level)->neighborZ,
        //         para->getParD(level)->numberOfNodes,
        //         para->getParD(level)->isEvenTimestep);

        //     QSlipNormDevComp27(
        //         para->getParD(level)->numberofthreads,
        //         para->getParD(level)->distributions.f[0],
        //         para->getParD(level)->geometryBC.k,
        //         para->getParD(level)->geometryBC.q27[0],
        //         para->getParD(level)->geometryBC.numberOfBCnodes,
        //         para->getParD(level)->omega,
        //         para->getParD(level)->geometryBCnormalX.q27[0],
        //         para->getParD(level)->geometryBCnormalY.q27[0],
        //         para->getParD(level)->geometryBCnormalZ.q27[0],
        //         para->getParD(level)->neighborX,
        //         para->getParD(level)->neighborY,
        //         para->getParD(level)->neighborZ,
        //         para->getParD(level)->numberOfNodes,
        //         para->getParD(level)->isEvenTimestep);
    }
}

void BoundaryConditionKernelManager::runPressureBCKernelPre(int level) const
{
    for (auto boundaryConditionStruct : para->getParD(level)->pressureBCDirectional) {
        this->directionalPressureBoundaryConditionPre(para->getParD(level).get(), &boundaryConditionStruct);
    }
    if (para->getParD(level)->pressureBC.numberOfBCnodes > 0)
        this->pressureBoundaryConditionPre(para->getParD(level).get(), &(para->getParD(level)->pressureBC));
}

void BoundaryConditionKernelManager::runStressWallModelKernelPost(int level) const
{
    if (para->getParD(level)->stressBC.numberOfBCnodes > 0)
        stressBoundaryConditionPost(para.get(), &(para->getParD(level)->stressBC), level);
}

void BoundaryConditionKernelManager::runSlipBCKernelPost(int level) const{
    if (para->getParD(level)->slipBC.numberOfBCnodes > 0)
        slipBoundaryConditionPost(para->getParD(level).get(), &(para->getParD(level)->slipBC));
}

void BoundaryConditionKernelManager::runNoSlipBCKernelPost(int level) const{
    if (para->getParD(level)->noSlipBC.numberOfBCnodes > 0)
        noSlipBoundaryConditionPost(para->getParD(level).get(), &(para->getParD(level)->noSlipBC));
}

void BoundaryConditionKernelManager::runPrecursorBCKernelPost(int level, uint t, CudaMemoryManager* cudaMemoryManager)
{
    if(para->getParH(level)->precursorBC.numberOfBCnodes == 0) return;

    uint t_level = para->getTimeStep(level, t, true);

    uint lastTime =    (para->getParD(level)->precursorBC.nPrecursorReads-2)*para->getParD(level)->precursorBC.timeStepsBetweenReads; // timestep currently loaded into last arrays
    uint currentTime = (para->getParD(level)->precursorBC.nPrecursorReads-1)*para->getParD(level)->precursorBC.timeStepsBetweenReads; // timestep currently loaded into current arrays
    uint nextTime =     para->getParD(level)->precursorBC.nPrecursorReads   *para->getParD(level)->precursorBC.timeStepsBetweenReads; // timestep currently loaded into next arrays
    
    if(t_level>=currentTime)
    {
        //cycle time
        lastTime = currentTime;
        currentTime = nextTime;
        nextTime += para->getParD(level)->precursorBC.timeStepsBetweenReads;

        //cycle pointers
        real* tmp = para->getParD(level)->precursorBC.last;
        para->getParD(level)->precursorBC.last = para->getParD(level)->precursorBC.current;
        para->getParD(level)->precursorBC.current = para->getParD(level)->precursorBC.next;
        para->getParD(level)->precursorBC.next = tmp;

        real loadTime = nextTime*exp2(-level)*para->getTimeRatio();

        for(auto reader : para->getParH(level)->transientBCInputFileReader)
        {   
            reader->getNextData(para->getParH(level)->precursorBC.next, para->getParH(level)->precursorBC.numberOfPrecursorNodes, loadTime);
        }
        cudaMemoryManager->cudaCopyPrecursorData(level);
        para->getParD(level)->precursorBC.nPrecursorReads++;
        para->getParH(level)->precursorBC.nPrecursorReads++;  
    }
    
    real tRatio = real(t_level-lastTime)/para->getParD(level)->precursorBC.timeStepsBetweenReads;
    precursorBoundaryConditionPost(para->getParD(level).get(), &para->getParD(level)->precursorBC, tRatio, para->getVelocityRatio());
}

//! \}
