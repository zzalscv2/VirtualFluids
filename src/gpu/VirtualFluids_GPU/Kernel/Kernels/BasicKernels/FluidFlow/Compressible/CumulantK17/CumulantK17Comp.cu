#include "CumulantK17Comp.h"

#include "Parameter/Parameter.h"
#include "CumulantK17Comp_Device.cuh"

std::shared_ptr<CumulantK17Comp> CumulantK17Comp::getNewInstance(std::shared_ptr<Parameter> para, int level)
{
	return std::shared_ptr<CumulantK17Comp>(new CumulantK17Comp(para,level));
}

void CumulantK17Comp::run()
{
	int numberOfThreads = para->getParD(level)->numberofthreads;
	int size_Mat = para->getParD(level)->size_Mat_SP;

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

	LB_Kernel_CumulantK17Comp <<< grid, threads >>>(para->getParD(level)->omega,
													para->getParD(level)->geoSP,
													para->getParD(level)->neighborX_SP,
													para->getParD(level)->neighborY_SP,
													para->getParD(level)->neighborZ_SP,
													para->getParD(level)->d0SP.f[0],
													para->getParD(level)->size_Mat_SP,
													level,
													para->getForcesDev(),
                                                    para->getQuadricLimitersDev(),
													para->getParD(level)->evenOrOdd);
	getLastCudaError("LB_Kernel_CumulantK17Comp execution failed");
}

CumulantK17Comp::CumulantK17Comp(std::shared_ptr<Parameter> para, int level)
{
	this->para = para;
	this->level = level;

	myPreProcessorTypes.push_back(InitCompSP27);

	myKernelGroup = BasicKernel;
}