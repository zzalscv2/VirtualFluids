#include "CascadeCompSP27.h"

#include "CascadeCompSP27_Device.cuh"
#include "Parameter\Parameter.h"

std::shared_ptr<CascadeCompSP27> CascadeCompSP27::getNewInstance(std::shared_ptr<Parameter> para, int level)
{
	return std::shared_ptr<CascadeCompSP27>(new CascadeCompSP27(para, level));
}

void CascadeCompSP27::run()
{
	int size_Mat = para->getParD(level)->size_Mat_SP;
	int numberOfThreads = para->getParD(level)->numberofthreads;

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

	LB_Kernel_Cascade_Comp_SP_27 << < grid, threads >> >(	para->getParD(level)->omega,
															para->getParD(level)->geoSP,
															para->getParD(level)->neighborX_SP,
															para->getParD(level)->neighborY_SP,
															para->getParD(level)->neighborZ_SP,
															para->getParD(level)->d0SP.f[0],
															para->getParD(level)->size_Mat_SP,
															para->getParD(level)->evenOrOdd);
	getLastCudaError("LB_Kernel_Cascade_Comp_SP_27 execution failed");
}

CascadeCompSP27::CascadeCompSP27(std::shared_ptr<Parameter> para, int level)
{
	this->para = para;
	this->level = level;

	myPreProcessorTypes.push_back(InitCompSP27);

	myKernelGroup = BasicKernel;
}

CascadeCompSP27::CascadeCompSP27()
{
}
