#include "InitCompSP27.h"

#include "InitCompSP27_Device.cuh"
#include "Parameter/Parameter.h"

std::shared_ptr<PreProcessorStrategy> InitCompSP27::getNewInstance(std::shared_ptr<Parameter> para)
{
	return std::shared_ptr<PreProcessorStrategy>(new InitCompSP27(para));
}

void InitCompSP27::init(int level)
{
	int numberOfThreads = para->getParD(level)->numberofthreads;
	int size_Mat = (int)para->getParD(level)->numberOfNodes;

	int Grid = (size_Mat / numberOfThreads) + 1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid / Grid1) + 1;
	}
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1);

    if( ! para->getUseInitNeq() )
    {
        LB_Init_Comp_SP_27 <<< grid, threads >>> (
            para->getParD(level)->neighborX,
            para->getParD(level)->neighborY,
            para->getParD(level)->neighborZ,
            para->getParD(level)->typeOfGridNode,
            para->getParD(level)->rho,
            para->getParD(level)->velocityX,
            para->getParD(level)->velocityY,
            para->getParD(level)->velocityZ,
            para->getParD(level)->numberOfNodes,
            para->getParD(level)->distributions.f[0],
            para->getParD(level)->isEvenTimestep);
        getLastCudaError("LB_Init_Comp_SP_27 execution failed");
    }
    else
    {
        LB_Init_Comp_Neq_SP_27 <<< grid, threads >>> (
            para->getParD(level)->neighborX,
            para->getParD(level)->neighborY,
            para->getParD(level)->neighborZ,
            para->getParD(level)->neighborInverse,
            para->getParD(level)->typeOfGridNode,
            para->getParD(level)->rho,
            para->getParD(level)->velocityX,
            para->getParD(level)->velocityY,
            para->getParD(level)->velocityZ,
            para->getParD(level)->numberOfNodes,
            para->getParD(level)->distributions.f[0],
            para->getParD(level)->omega,
            para->getParD(level)->isEvenTimestep);
        cudaDeviceSynchronize();
        getLastCudaError("LB_Init_Comp_Neq_SP_27 execution failed");
    }



}

bool InitCompSP27::checkParameter()
{
	return false;
}

InitCompSP27::InitCompSP27(std::shared_ptr<Parameter> para)
{
	this->para = para;
}

InitCompSP27::InitCompSP27()
{
}
