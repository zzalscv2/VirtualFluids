//  _    ___      __              __________      _     __        ______________   __
// | |  / (_)____/ /___  ______ _/ / ____/ /_  __(_)___/ /____   /  ___/ __  / /  / /
// | | / / / ___/ __/ / / / __ `/ / /_  / / / / / / __  / ___/  / /___/ /_/ / /  / /
// | |/ / / /  / /_/ /_/ / /_/ / / __/ / / /_/ / / /_/ (__  )  / /_) / ____/ /__/ / 
// |___/_/_/   \__/\__,_/\__,_/_/_/   /_/\__,_/_/\__,_/____/   \____/_/    \_____/
//
//////////////////////////////////////////////////////////////////////////
#include "Temperature/FindTemperature.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "GPU/CudaMemoryManager.h"
#include "GPU/GPU_Interface.h"
#include "Parameter/Parameter.h"
#include "Temperature/FindQTemp.h"

void initTemperatur(Parameter* para, CudaMemoryManager* cudaMemoryManager, int lev)
{
    cudaMemoryManager->cudaAllocTempFs(lev);

    cudaMemoryManager->cudaCopyConcentrationHostToDevice(lev);

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      para->getParD(lev)->isEvenTimestep = false;
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //InitADDev27( ); 
      getLastCudaError("Kernel execution failed"); 
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      para->getParD(lev)->isEvenTimestep = true;
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //InitADDev27(  ); 
      getLastCudaError("Kernel execution failed"); 
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      CalcConcentration27(
                     para->getParD(lev)->numberofthreads,
                     para->getParD(lev)->concentration,
                     para->getParD(lev)->typeOfGridNode,
                     para->getParD(lev)->neighborX,
                     para->getParD(lev)->neighborY,
                     para->getParD(lev)->neighborZ,
                     para->getParD(lev)->numberOfNodes,
                     para->getParD(lev)->distributionsAD.f[0],
                     para->getParD(lev)->isEvenTimestep);
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   cudaMemoryManager->cudaCopyConcentrationDeviceToHost(lev);
}


void findTempSim(Parameter* para, CudaMemoryManager* cudaMemoryManager)
{
   findKforTemp(para);

   cudaMemoryManager->cudaAllocTempNoSlipBC(para->getCoarse());

   findTemp(para);

   cudaMemoryManager->cudaCopyTempNoSlipBCHD(para->getCoarse());
}


void findTempVelSim(Parameter* para, CudaMemoryManager* cudaMemoryManager)
{
   findKforTempVel(para);

   cudaMemoryManager->cudaAllocTempVeloBC(para->getCoarse());

   findTempVel(para);

   cudaMemoryManager->cudaCopyTempVeloBCHD(para->getCoarse());
}


void findTempPressSim(Parameter* para, CudaMemoryManager* cudaMemoryManager)
{
   findKforTempPress(para);

   cudaMemoryManager->cudaAllocTempPressBC(para->getCoarse());

   findTempPress(para);

   cudaMemoryManager->cudaCopyTempPressBCHD(para->getCoarse());
}