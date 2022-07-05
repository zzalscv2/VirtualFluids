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
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with VirtualFluids (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file LBKernelManager.cpp
//! \ingroup KernelManager
//! \author Martin Schoenherr
//=======================================================================================
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "LBKernelManager.h"
#include "Parameter/Parameter.h"	
#include "GPU/GPU_Interface.h"
#include "Calculation/DragLift.h"
#include "Calculation/Cp.h"
#include "BoundaryConditions/BoundaryConditionFactory.h"

LBKernelManager::LBKernelManager(SPtr<Parameter> parameter, BoundaryConditionFactory* bcFactory): para(parameter)
{
    this->velocityBoundaryConditionPost = bcFactory->getVelocityBoundaryConditionPost();
    this->noSlipBoundaryConditionPost   = bcFactory->getNoSlipBoundaryConditionPost();
    this->slipBoundaryConditionPost     = bcFactory->getSlipBoundaryConditionPost();
    this->pressureBoundaryConditionPre  = bcFactory->getPressureBoundaryConditionPre();
    this->geometryBoundaryConditionPost = bcFactory->getGeometryBoundaryConditionPost();
}

void LBKernelManager::runVelocityBCKernelPre(const int level) const
{
    if (para->getParD(level)->velocityBC.numberOfBCnodes > 0)
    {
        // TODO: https://git.rz.tu-bs.de/irmb/VirtualFluids_dev/-/issues/29
        // if ( myid == 0)
        // {
        //    VelSchlaffer27(para->getParD(level)->numberofthreads, t,
        //                   para->getParD(level)->distributions.f[0],       para->getParD(level)->velocityBC.Vz,
        //                   para->getParD(level)->velocityBC.deltaVz, para->getParD(level)->velocityBC.k,
        //                   para->getParD(level)->velocityBC.kN,      para->getParD(level)->velocityBC.numberOfBCnodes,
        //                   para->getParD(level)->omega,           para->getParD(level)->neighborX,
        //                   para->getParD(level)->neighborY,    para->getParD(level)->neighborZ,
        //                   para->getParD(level)->numberOfNodes,     para->getParD(level)->isEvenTimestep);
        //    getLastCudaError("VelSchlaffer27 execution failed");
        // }
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

void LBKernelManager::runVelocityBCKernelPost(const int level) const
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

void LBKernelManager::runGeoBCKernelPre(const int level, unsigned int t, CudaMemoryManager* cudaMemoryManager) const{
    if (para->getParD(level)->geometryBC.numberOfBCnodes > 0){
        if (para->getCalcDragLift())
        {
            //Drag and Lift Part II
            DragLiftPreD27(
                para->getParD(level)->distributions.f[0],
                para->getParD(level)->geometryBC.k,
                para->getParD(level)->geometryBC.q27[0],
                para->getParD(level)->geometryBC.numberOfBCnodes,
                para->getParD(level)->DragPreX,
                para->getParD(level)->DragPreY,
                para->getParD(level)->DragPreZ,
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

            if(t > para->getTStartOut())
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

void LBKernelManager::runGeoBCKernelPost(const int level) const
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
                            para->getParD(level)->DragPostX,
                            para->getParD(level)->DragPostY,
                            para->getParD(level)->DragPostZ,
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

void LBKernelManager::runOutflowBCKernelPre(const int level) const{
    if (para->getParD(level)->outflowBC.numberOfBCnodes > 0)
    {
        QPressNoRhoDev27(para->getParD(level).get(), &(para->getParD(level)->outflowBC));

        // TODO: https://git.rz.tu-bs.de/irmb/VirtualFluids_dev/-/issues/29
        // if (  myid == numprocs - 1)
        // PressSchlaffer27(
        //     para->getParD(level)->numberofthreads,
        //     para->getParD(level)->outflowBC.RhoBC,
        //     para->getParD(level)->distributions.f[0],
        //     para->getParD(level)->outflowBC.Vx,
        //     para->getParD(level)->outflowBC.Vy,
        //     para->getParD(level)->outflowBC.Vz,
        //     para->getParD(level)->outflowBC.deltaVz,
        //     para->getParD(level)->outflowBC.k,
        //     para->getParD(level)->outflowBC.kN,
        //     para->getParD(level)->outflowBC.numberOfBCnodes,
        //     para->getParD(level)->omega,
        //     para->getParD(level)->neighborX,
        //     para->getParD(level)->neighborY,
        //     para->getParD(level)->neighborZ,
        //     para->getParD(level)->numberOfNodes,
        //     para->getParD(level)->isEvenTimestep);
    }
}

void LBKernelManager::runPressureBCKernelPre(const int level) const{
    if (para->getParD(level)->pressureBC.numberOfBCnodes > 0)
    {
        this->pressureBoundaryConditionPre(para->getParD(level).get(), &(para->getParD(level)->pressureBC));
    }
}

void LBKernelManager::runPressureBCKernelPost(const int level) const{
    if (para->getParD(level)->pressureBC.numberOfBCnodes > 0)
    {
        // QPressDev27_IntBB(
        //     para->getParD(level)->numberofthreads, 
        //     para->getParD(level)->pressureBC.RhoBC,
        //     para->getParD(level)->distributions.f[0],
        //     para->getParD(level)->pressureBC.k,
        //     para->getParD(level)->pressureBC.q27[0],
        //     para->getParD(level)->pressureBC.numberOfBCnodes,
        //     para->getParD(level)->omega,
        //     para->getParD(level)->neighborX,
        //     para->getParD(level)->neighborY,
        //     para->getParD(level)->neighborZ,
        //     para->getParD(level)->numberOfNodes,
        //     para->getParD(level)->isEvenTimestep);
    }
}

void LBKernelManager::runStressWallModelKernelPost(const int level) const{
    if (para->getParD(level)->stressBC.numberOfBCnodes > 0)
    {
        // QStressDevComp27(para->getParD(level)->numberofthreads, para->getParD(level)->distributions.f[0],
        //                 para->getParD(level)->stressBC.k,       para->getParD(level)->stressBC.kN,
        //                 para->getParD(level)->stressBC.q27[0],  para->getParD(level)->stressBC.numberOfBCnodes,
        //                 para->getParD(level)->omega,            para->getParD(level)->turbViscosity,
        //                 para->getParD(level)->velocityX,        para->getParD(level)->velocityY,          para->getParD(level)->velocityY,
        //                 para->getParD(level)->stressBC.normalX, para->getParD(level)->stressBC.normalY,   para->getParD(level)->stressBC.normalZ,
        //                 para->getParD(level)->stressBC.Vx,      para->getParD(level)->stressBC.Vy,        para->getParD(level)->stressBC.Vz,
        //                 para->getParD(level)->stressBC.Vx1,     para->getParD(level)->stressBC.Vy1,       para->getParD(level)->stressBC.Vz1,
        //                 para->getParD(level)->wallModel.samplingOffset, para->getParD(level)->wallModel.z0,
        //                 para->getHasWallModelMonitor(),        para->getParD(level)->wallModel.u_star,
        //                 para->getParD(level)->wallModel.Fx,    para->getParD(level)->wallModel.Fy,        para->getParD(level)->wallModel.Fz,
        //                 para->getParD(level)->neighborX,       para->getParD(level)->neighborY,           para->getParD(level)->neighborZ,
        //                 para->getParD(level)->numberOfNodes,        para->getParD(level)->isEvenTimestep);

        BBStressDev27( para->getParD(level)->numberofthreads,   para->getParD(level)->distributions.f[0],
                        para->getParD(level)->stressBC.k,       para->getParD(level)->stressBC.kN,
                        para->getParD(level)->stressBC.q27[0],  para->getParD(level)->stressBC.numberOfBCnodes,
                        para->getParD(level)->velocityX,        para->getParD(level)->velocityY,          para->getParD(level)->velocityY,
                        para->getParD(level)->stressBC.normalX, para->getParD(level)->stressBC.normalY,   para->getParD(level)->stressBC.normalZ,
                        para->getParD(level)->stressBC.Vx,      para->getParD(level)->stressBC.Vy,        para->getParD(level)->stressBC.Vz,
                        para->getParD(level)->stressBC.Vx1,     para->getParD(level)->stressBC.Vy1,       para->getParD(level)->stressBC.Vz1,
                        para->getParD(level)->wallModel.samplingOffset, para->getParD(level)->wallModel.z0,
                        para->getHasWallModelMonitor(),         para->getParD(level)->wallModel.u_star,
                        para->getParD(level)->wallModel.Fx,     para->getParD(level)->wallModel.Fy,      para->getParD(level)->wallModel.Fz,
                        para->getParD(level)->neighborX,        para->getParD(level)->neighborY,         para->getParD(level)->neighborZ,
                        para->getParD(level)->numberOfNodes,    para->getParD(level)->isEvenTimestep);
    }
}


void LBKernelManager::runSlipBCKernelPost(const int level) const{
    if (para->getParD(level)->slipBC.numberOfBCnodes > 0)
    {
        slipBoundaryConditionPost(para->getParD(level).get(), &(para->getParD(level)->slipBC));
    }
}

void LBKernelManager::runNoSlipBCKernelPost(const int level) const{
    if (para->getParD(level)->noSlipBC.numberOfBCnodes > 0)
    {
        noSlipBoundaryConditionPost(para->getParD(level).get(), &(para->getParD(level)->noSlipBC));
    }
}

// void LBKernelManager::calculateMacroscopicValues(const int level) const
// {
//     if (para->getIsADcalculationOn()) {
//           CalcMacADCompSP27(
//                para->getParD()->velocityX,
//                para->getParD()->velocityY,
//                para->getParD()->velocityZ,
//                para->getParD()->rho,
//                para->getParD()->pressure,
//                para->getParD()->typeOfGridNode,
//                para->getParD()->neighborX,
//                para->getParD()->neighborY,
//                para->getParD()->neighborZ,
//                para->getParD()->numberOfNodes,
//                para->getParD()->numberofthreads,
//                para->getParD()->distributions.f[0],
//                para->getParD()->distributionsAD.f[0],
//             para->getParD()->forcing,
//                para->getParD()->isEvenTimestep);
//     } else {
//           CalcMacCompSP27(
//                para->getParD()->velocityX,
//                para->getParD()->velocityY,
//                para->getParD()->velocityZ,
//                para->getParD()->rho,
//                para->getParD()->pressure,
//                para->getParD()->typeOfGridNode,
//                para->getParD()->neighborX,
//                para->getParD()->neighborY,
//                para->getParD()->neighborZ,
//                para->getParD()->numberOfNodes,
//                para->getParD()->numberofthreads,
//                para->getParD()->distributions.f[0],
//                para->getParD()->isEvenTimestep);
//      }
// }
