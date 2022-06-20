#include "InitSP27.h"

#include "InitSP27_Device.cuh"
#include "Parameter/Parameter.h"

std::shared_ptr<PreProcessorStrategy> InitSP27::getNewInstance(std::shared_ptr<Parameter> para)
{
	return std::shared_ptr<PreProcessorStrategy>(new InitSP27(para));
}

void InitSP27::init(int level)
{
	int numberOfThreads = para->getParD(level)->numberofthreads;
	int size_Mat = para->getParD(level)->numberOfNodes;

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

	LB_Init_SP_27 << < grid, threads >> >(	para->getParD(level)->neighborX,
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
	getLastCudaError("LBInitSP27 execution failed");
}

bool InitSP27::checkParameter()
{
	return false;
}

InitSP27::InitSP27(std::shared_ptr<Parameter> para)
{
	this->para = para;
}

InitSP27::InitSP27()
{
}
