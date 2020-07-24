#ifndef  CudaRunKernel_HPP
#define  CudaRunKernel_HPP

#include <string>
#include <device_launch_parameters.h>
#include <omp.h>
#include <iostream>

#include "CudaUtility/CudaUtility.h"

namespace GksGpu {

template<typename KernelFunctor, typename FunctionFunctor, typename... TArgs>
void runKernel(KernelFunctor kernel, FunctionFunctor function, std::string deviceType, const CudaUtility::CudaGrid& grid, TArgs... args)
{
    if( grid.numberOfEntities == 0 ) return;

    if( deviceType == "GPU" )
    {
        kernel<<< grid.blocks, grid.threads, 0, grid.stream >>>( args..., grid.numberOfEntities );
    }
    else
    {
//#pragma omp parallel for
        for( int index = 0; index < grid.numberOfEntities; index++ )
        {
            function( args..., index );
        }
    }
}

} // namespace GksGpu

#endif