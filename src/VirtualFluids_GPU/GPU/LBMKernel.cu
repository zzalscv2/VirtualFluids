// includes, cuda
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "LBM/LB.h"

// includes, kernels
#include "GPU/GPU_Kernels.cuh"
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelCas27( unsigned int grid_nx, 
                             unsigned int grid_ny, 
                             unsigned int grid_nz, 
                             real s9,
                             unsigned int* bcMatD,
                             unsigned int* neighborX,
                             unsigned int* neighborY,
                             unsigned int* neighborZ,
                             real* DD,
                             int size_Mat,
                             bool EvenOrOdd)
{
   dim3 threads       ( grid_nx, 1, 1 );
   dim3 grid          ( grid_ny, grid_nz );   // Gitter fuer Kollision und Propagation

      LB_Kernel_Casc27<< < grid, threads >>>( s9,
                                             bcMatD,
                                             neighborX,
                                             neighborY,
                                             neighborZ,
                                             DD,
                                             size_Mat,
                                             EvenOrOdd); 
     getLastCudaError("LB_Kernel_Casc27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelCasSP27( unsigned int numberOfThreads, 
                               real s9,
                               unsigned int* bcMatD,
                               unsigned int* neighborX,
                               unsigned int* neighborY,
                               unsigned int* neighborZ,
                               real* DD,
                               int size_Mat,
                               bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LB_Kernel_Casc_SP_27<<< grid, threads >>>(s9,
                                                bcMatD,
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                DD,
                                                size_Mat,
                                                EvenOrOdd); 
      getLastCudaError("LB_Kernel_Casc_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelCasSPMS27( unsigned int numberOfThreads, 
                                 real s9,
                                 unsigned int* bcMatD,
                                 unsigned int* neighborX,
                                 unsigned int* neighborY,
                                 unsigned int* neighborZ,
                                 real* DD,
                                 int size_Mat,
                                 bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LB_Kernel_Casc_SP_MS_27<<< grid, threads >>>(s9,
                                                   bcMatD,
                                                   neighborX,
                                                   neighborY,
                                                   neighborZ,
                                                   DD,
                                                   size_Mat,
                                                   EvenOrOdd); 
      getLastCudaError("LB_Kernel_Casc_SP_MS_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelCasSPMSOHM27( unsigned int numberOfThreads, 
                                    real s9,
                                    unsigned int* bcMatD,
                                    unsigned int* neighborX,
                                    unsigned int* neighborY,
                                    unsigned int* neighborZ,
                                    real* DD,
                                    int size_Mat,
                                    bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LB_Kernel_Casc_SP_MS_OHM_27<<< grid, threads >>>(  s9,
                                                         bcMatD,
                                                         neighborX,
                                                         neighborY,
                                                         neighborZ,
                                                         DD,
                                                         size_Mat,
                                                         EvenOrOdd); 
      getLastCudaError("LB_Kernel_Casc_SP_MS_OHM_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelCasKumSP27(unsigned int numberOfThreads, 
                                 real s9,
                                 unsigned int* bcMatD,
                                 unsigned int* neighborX,
                                 unsigned int* neighborY,
                                 unsigned int* neighborZ,
                                 real* DD,
                                 int size_Mat,
                                 bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LB_Kernel_Casc_Kum_SP_27<<< grid, threads >>>(s9,
                                                    bcMatD,
                                                    neighborX,
                                                    neighborY,
                                                    neighborZ,
                                                    DD,
                                                    size_Mat,
                                                    EvenOrOdd); 
      getLastCudaError("LB_Kernel_Casc_Kum_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelBGKPlusSP27(unsigned int numberOfThreads, 
								  real s9,
								  unsigned int* bcMatD,
								  unsigned int* neighborX,
								  unsigned int* neighborY,
								  unsigned int* neighborZ,
								  real* DD,
								  int size_Mat,
								  bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	LB_Kernel_BGK_Plus_SP_27<<< grid, threads >>>(  s9,
													bcMatD,
													neighborX,
													neighborY,
													neighborZ,
													DD,
													size_Mat,
													EvenOrOdd); 
	getLastCudaError("LB_Kernel_BGK_Plus_SP_27 execution failed"); 
}

//////////////////////////////////////////////////////////////////////////
//extern "C" void KernelBGKPlusCompSP27(unsigned int numberOfThreads, 
//									  real s9,
//									  unsigned int* bcMatD,
//									  unsigned int* neighborX,
//									  unsigned int* neighborY,
//									  unsigned int* neighborZ,
//									  real* DD,
//									  int size_Mat,
//									  bool EvenOrOdd)
//{
//	int Grid = (size_Mat / numberOfThreads)+1;
//	int Grid1, Grid2;
//	if (Grid>512)
//	{
//		Grid1 = 512;
//		Grid2 = (Grid/Grid1)+1;
//	} 
//	else
//	{
//		Grid1 = 1;
//		Grid2 = Grid;
//	}
//	dim3 grid(Grid1, Grid2);
//	dim3 threads(numberOfThreads, 1, 1 );
//
//	LB_Kernel_BGK_Plus_Comp_SP_27<<< grid, threads >>>( s9,
//														bcMatD,
//														neighborX,
//														neighborY,
//														neighborZ,
//														DD,
//														size_Mat,
//														EvenOrOdd); 
//	getLastCudaError("LB_Kernel_BGK_Plus_Comp_SP_27 execution failed"); 
//}
////////////////////////////////////////////////////////////////////////////

extern "C" void KernelBGKCompSP27(unsigned int numberOfThreads, 
								  real s9,
								  unsigned int* bcMatD,
								  unsigned int* neighborX,
								  unsigned int* neighborY,
								  unsigned int* neighborZ,
								  real* DD,
								  int size_Mat,
								  bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	LB_Kernel_BGK_Comp_SP_27<<< grid, threads >>>(  s9,
													bcMatD,
													neighborX,
													neighborY,
													neighborZ,
													DD,
													size_Mat,
													EvenOrOdd); 
	getLastCudaError("LB_Kernel_BGK_Comp_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelBGKSP27(unsigned int numberOfThreads, 
							  real s9,
							  unsigned int* bcMatD,
							  unsigned int* neighborX,
							  unsigned int* neighborY,
							  unsigned int* neighborZ,
							  real* DD,
							  int size_Mat,
							  bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	LB_Kernel_BGK_SP_27<<< grid, threads >>>(s9,
											 bcMatD,
											 neighborX,
											 neighborY,
											 neighborZ,
											 DD,
											 size_Mat,
											 EvenOrOdd); 
	getLastCudaError("LB_Kernel_BGK_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelMRTSP27(unsigned int numberOfThreads, 
							  real s9,
							  unsigned int* bcMatD,
							  unsigned int* neighborX,
							  unsigned int* neighborY,
							  unsigned int* neighborZ,
							  real* DD,
							  int size_Mat,
							  bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

		LB_Kernel_MRT_SP_27<<< grid, threads >>>(   s9,
													bcMatD,
													neighborX,
													neighborY,
													neighborZ,
													DD,
													size_Mat,
													EvenOrOdd); 
		getLastCudaError("LB_Kernel_MRT_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelMRTCompSP27(unsigned int numberOfThreads, 
								  real s9,
								  unsigned int* bcMatD,
								  unsigned int* neighborX,
								  unsigned int* neighborY,
								  unsigned int* neighborZ,
								  real* DD,
								  int size_Mat,
								  bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

		LB_Kernel_MRT_Comp_SP_27<<< grid, threads >>>(  s9,
														bcMatD,
														neighborX,
														neighborY,
														neighborZ,
														DD,
														size_Mat,
														EvenOrOdd); 
		getLastCudaError("LB_Kernel_MRT_Comp_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////





//////////////////////////////////////////////////////////////////////////
extern "C" void KernelKumAA2016CompBulkSP27(unsigned int numberOfThreads, 
											real s9,
											unsigned int* bcMatD,
											unsigned int* neighborX,
											unsigned int* neighborY,
											unsigned int* neighborZ,
											real* DD,
											int size_Mat,
											int size_Array,
											int level,
											real* forces,
											bool EvenOrOdd)
{
	int Grid = size_Array / numberOfThreads;
	dim3 grid(Grid, 1, 1);
	dim3 threads(numberOfThreads, 1, 1);

		LB_Kernel_Kum_AA2016_Comp_Bulk_SP_27<<< grid, threads >>>(s9,
																  bcMatD,
																  neighborX,
																  neighborY,
																  neighborZ,
																  DD,
																  size_Mat,
																  level,
																  forces,
																  EvenOrOdd); 
		getLastCudaError("LB_Kernel_Kum_AA2016_Comp_Bulk_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////


//extern "C" void KernelKumAA2016CompSP27(unsigned int numberOfThreads, 
//										real s9,
//										unsigned int* bcMatD,
//										unsigned int* neighborX,
//										unsigned int* neighborY,
//										unsigned int* neighborZ,
//										real* DD,
//										int size_Mat,
//										int level,
//										real* forces,
//										bool EvenOrOdd)
//{
//	int Grid = (size_Mat / numberOfThreads)+1;
//	int Grid1, Grid2;
//	if (Grid>512)
//	{
//		Grid1 = 512;
//		Grid2 = (Grid/Grid1)+1;
//	} 
//	else
//	{
//		Grid1 = 1;
//		Grid2 = Grid;
//	}
//	dim3 grid(Grid1, Grid2);
//	dim3 threads(numberOfThreads, 1, 1 );
//
//		LB_Kernel_Kum_AA2016_Comp_SP_27<<< grid, threads >>>(s9,
//															 bcMatD,
//															 neighborX,
//															 neighborY,
//															 neighborZ,
//															 DD,
//															 size_Mat,
//															 level,
//															 forces,
//															 EvenOrOdd); 
//		getLastCudaError("LB_Kernel_Kum_AA2016_Comp_SP_27 execution failed"); 
//}


//////////////////////////////////////////////////////////////////////////
extern "C" void KernelKumNewCompSpongeSP27(unsigned int numberOfThreads, 
									       real s9,
									       unsigned int* bcMatD,
									       unsigned int* neighborX,
									       unsigned int* neighborY,
									       unsigned int* neighborZ,
									       real* coordX,
									       real* coordY,
									       real* coordZ,
									       real* DD,
									       int size_Mat,
									       bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

		LB_Kernel_Kum_New_Comp_Sponge_SP_27<<< grid, threads >>>(s9,
															     bcMatD,
															     neighborX,
															     neighborY,
															     neighborZ,
													             coordX,
													             coordY,
													             coordZ,
															     DD,
															     size_Mat,
															     EvenOrOdd); 
		getLastCudaError("LB_Kernel_Kum_New_Comp_Sponge_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelKum1hSP27(    unsigned int numberOfThreads, 
									real omega,
									real deltaPhi,
									real angularVelocity,
									unsigned int* bcMatD,
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									real* coordX,
									real* coordY,
									real* coordZ,
									real* DDStart,
									int size_Mat,
									bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

		LB_Kernel_Kum_1h_SP_27<<< grid, threads >>>(omega,
													deltaPhi,
													angularVelocity,
													bcMatD,
													neighborX,
													neighborY,
													neighborZ,
													coordX,
													coordY,
													coordZ,
													DDStart,
													size_Mat,
													EvenOrOdd); 
		getLastCudaError("LB_Kernel_Kum_New_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelCascadeSP27(  unsigned int numberOfThreads, 
									real s9,
									unsigned int* bcMatD,
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									real* DD,
									int size_Mat,
									bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

		LB_Kernel_Cascade_SP_27<<< grid, threads >>>(s9,
													bcMatD,
													neighborX,
													neighborY,
													neighborZ,
													DD,
													size_Mat,
													EvenOrOdd); 
		getLastCudaError("LB_Kernel_Cascade_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelCascadeCompSP27(  unsigned int numberOfThreads, 
										real s9,
										unsigned int* bcMatD,
										unsigned int* neighborX,
										unsigned int* neighborY,
										unsigned int* neighborZ,
										real* DD,
										int size_Mat,
										bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	LB_Kernel_Cascade_Comp_SP_27<<< grid, threads >>>(  s9,
														bcMatD,
														neighborX,
														neighborY,
														neighborZ,
														DD,
														size_Mat,
														EvenOrOdd); 
	getLastCudaError("LB_Kernel_Cascade_Comp_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelKumNewSP27(   unsigned int numberOfThreads, 
									real s9,
									unsigned int* bcMatD,
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									real* DD,
									int size_Mat,
									bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

		LB_Kernel_Kum_New_SP_27<<< grid, threads >>>(s9,
													bcMatD,
													neighborX,
													neighborY,
													neighborZ,
													DD,
													size_Mat,
													EvenOrOdd); 
		getLastCudaError("LB_Kernel_Kum_New_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelKumNewCompBulkSP27(unsigned int numberOfThreads, 
										 real s9,
										 unsigned int* bcMatD,
										 unsigned int* neighborX,
										 unsigned int* neighborY,
										 unsigned int* neighborZ,
										 real* DD,
										 int size_Mat,
										 int level,
										 real* forces,
										 bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

		LB_Kernel_Kum_New_Comp_Bulk_SP_27<<< grid, threads >>>(	s9,
																bcMatD,
																neighborX,
																neighborY,
																neighborZ,
																DD,
																size_Mat,
																level,
																forces,
																EvenOrOdd); 
		getLastCudaError("LB_Kernel_Kum_New_Comp_Bulk_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelKumNewCompSP27(unsigned int numberOfThreads, 
									real s9,
									unsigned int* bcMatD,
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									real* DD,
									int size_Mat,
									int size_Array,
									int level,
									real* forces,
									bool EvenOrOdd)
{
	//int Grid = size_Array / numberOfThreads;
	//dim3 grid(Grid, 1, 1);
	//dim3 threads(numberOfThreads, 1, 1 );

	int Grid = (size_Mat / numberOfThreads) + 1;
	int Grid1, Grid2;
	if (Grid > 512)
	{
		Grid1 = 512;
		Grid2 = (Grid / Grid1) + 1;
	}
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2, 1);
	dim3 threads(numberOfThreads, 1, 1);

		//LB_Kernel_Kum_New_Comp_SP_27<<< grid, threads >>>(	s9,
		//													bcMatD,
		//													neighborX,
		//													neighborY,
		//													neighborZ,
		//													DD,
		//													size_Mat,
		//													level,
		//													forces,
		//													EvenOrOdd); 
		//getLastCudaError("LB_Kernel_Kum_New_Comp_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelKumIsoTestSP27(unsigned int numberOfThreads, 
									 real s9,
									 unsigned int* bcMatD,
									 unsigned int* neighborX,
									 unsigned int* neighborY,
									 unsigned int* neighborZ,
									 real* DD,
									 real* dxxUx,
									 real* dyyUy,
									 real* dzzUz,
									 int size_Mat,
									 bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	LB_Kernel_Kum_IsoTest_SP_27<<< grid, threads >>>(s9,
													bcMatD,
													neighborX,
													neighborY,
													neighborZ,
													DD,
													dxxUx,
													dyyUy,
													dzzUz,
													size_Mat,
													EvenOrOdd); 
	getLastCudaError("LB_Kernel_Kum_IsoTest_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelKumCompSP27(  unsigned int numberOfThreads, 
									real s9,
									unsigned int* bcMatD,
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									real* DD,
									int size_Mat,
									bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

		LB_Kernel_Kum_Comp_SP_27<<< grid, threads >>>(s9,
													bcMatD,
													neighborX,
													neighborY,
													neighborZ,
													DD,
													size_Mat,
													EvenOrOdd); 
		getLastCudaError("LB_Kernel_Kum_Comp_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelWaleCumOneCompSP27(unsigned int numberOfThreads,
										 real s9,
										 unsigned int* bcMatD,
										 unsigned int* neighborX,
										 unsigned int* neighborY,
										 unsigned int* neighborZ,
										 unsigned int* neighborWSB,
										 real* veloX,
										 real* veloY,
										 real* veloZ,
										 real* DD,
										 real* turbulentViscosity,
										 int size_Mat,
										 int size_Array,
										 int level,
										 real* forces,
										 bool EvenOrOdd)
{
	//int Grid = size_Array / numberOfThreads;
	//dim3 grid(Grid, 1, 1);
	//dim3 threads(numberOfThreads, 1, 1 );

	int Grid = (size_Mat / numberOfThreads) + 1;
	int Grid1, Grid2;
	if (Grid > 512)
	{
		Grid1 = 512;
		Grid2 = (Grid / Grid1) + 1;
	}
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2, 1);
	dim3 threads(numberOfThreads, 1, 1);

	LB_Kernel_Wale_Cum_One_Comp_SP_27 <<< grid, threads >>>(s9,
															bcMatD,
															neighborX,
															neighborY,
															neighborZ,
															neighborWSB,
															veloX,
															veloY,
															veloZ,
															DD,
															turbulentViscosity,
															size_Mat,
															level,
															forces,
															EvenOrOdd); 
		getLastCudaError("LB_Kernel_Wale_Cum_One_Comp_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelPMCumOneCompSP27(unsigned int numberOfThreads, 
									   real omega,
									   unsigned int* neighborX,
									   unsigned int* neighborY,
									   unsigned int* neighborZ,
									   real* DD,
									   int size_Mat,
									   int level,
									   real* forces,
									   real porosity,
									   real darcy,
									   real forchheimer,
									   unsigned int sizeOfPorousMedia,
									   unsigned int* nodeIdsPorousMedia, 
									   bool EvenOrOdd)
{
	int Grid = (size_Mat / numberOfThreads) + 1;
	int Grid1, Grid2;
	if (Grid > 512)
	{
		Grid1 = 512;
		Grid2 = (Grid / Grid1) + 1;
	}
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2, 1);
	dim3 threads(numberOfThreads, 1, 1);

	LB_Kernel_PM_Cum_One_Comp_SP_27 <<< grid, threads >>>(omega,
														  neighborX,
														  neighborY,
														  neighborZ,
														  DD,
														  size_Mat,
														  level,
														  forces,
														  porosity,
														  darcy,
														  forchheimer,
														  sizeOfPorousMedia,
														  nodeIdsPorousMedia,
														  EvenOrOdd); 
	getLastCudaError("LB_Kernel_PM_Cum_One_Comp_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelWaleBySoniMalavCumOneCompSP27( unsigned int numberOfThreads,
													 real s9,
													 unsigned int* bcMatD,
													 unsigned int* neighborX,
													 unsigned int* neighborY,
													 unsigned int* neighborZ,
													 unsigned int* neighborWSB,
													 real* veloX,
													 real* veloY,
													 real* veloZ,
													 real* DD,
													 real* turbulentViscosity,
													 int size_Mat,
													 int size_Array,
													 int level,
													 real* forces,
													 bool EvenOrOdd)
{
	//int Grid = size_Array / numberOfThreads;
	//dim3 grid(Grid, 1, 1);
	//dim3 threads(numberOfThreads, 1, 1 );

	int Grid = (size_Mat / numberOfThreads) + 1;
	int Grid1, Grid2;
	if (Grid > 512)
	{
		Grid1 = 512;
		Grid2 = (Grid / Grid1) + 1;
	}
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2, 1);
	dim3 threads(numberOfThreads, 1, 1);

	LB_Kernel_WaleBySoniMalav_Cum_One_Comp_SP_27 <<< grid, threads >>>( s9,
																		bcMatD,
																		neighborX,
																		neighborY,
																		neighborZ,
																		neighborWSB,
																		veloX,
																		veloY,
																		veloZ,
																		DD,
																		turbulentViscosity,
																		size_Mat,
																		level,
																		forces,
																		EvenOrOdd); 
		getLastCudaError("LB_Kernel_WaleBySoniMalav_Cum_One_Comp_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelWaleCumAA2016CompSP27( unsigned int numberOfThreads,
											 real s9,
											 unsigned int* bcMatD,
											 unsigned int* neighborX,
											 unsigned int* neighborY,
											 unsigned int* neighborZ,
											 unsigned int* neighborWSB,
											 real* veloX,
											 real* veloY,
											 real* veloZ,
											 real* DD,
											 real* turbulentViscosity,
											 int size_Mat,
											 int size_Array,
											 int level,
											 real* forces,
											 bool EvenOrOdd)
{
	//int Grid = size_Array / numberOfThreads;
	//dim3 grid(Grid, 1, 1);
	//dim3 threads(numberOfThreads, 1, 1 );

	int Grid = (size_Mat / numberOfThreads) + 1;
	int Grid1, Grid2;
	if (Grid > 512)
	{
		Grid1 = 512;
		Grid2 = (Grid / Grid1) + 1;
	}
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2, 1);
	dim3 threads(numberOfThreads, 1, 1);

	LB_Kernel_Wale_Cum_AA2016_Comp_SP_27 <<< grid, threads >>>( s9,
																bcMatD,
																neighborX,
																neighborY,
																neighborZ,
																neighborWSB,
																veloX,
																veloY,
																veloZ,
																DD,
																turbulentViscosity,
																size_Mat,
																level,
																forces,
																EvenOrOdd); 
		getLastCudaError("LB_Kernel_Wale_Cum_AA2016_Comp_SP_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelWaleCumAA2016DebugCompSP27(
	unsigned int numberOfThreads,
	real s9,
	unsigned int* bcMatD,
	unsigned int* neighborX,
	unsigned int* neighborY,
	unsigned int* neighborZ,
	unsigned int* neighborWSB,
	real* veloX,
	real* veloY,
	real* veloZ,
	real* DD,
	real* turbulentViscosity,
	real* gSij,
	real* gSDij,
	real* gDxvx,
	real* gDyvx,
	real* gDzvx,
	real* gDxvy,
	real* gDyvy,
	real* gDzvy,
	real* gDxvz,
	real* gDyvz,
	real* gDzvz,
	int size_Mat,
	int size_Array,
	int level,
	real* forces,
	bool EvenOrOdd)
{
	//int Grid = size_Array / numberOfThreads;
	//dim3 grid(Grid, 1, 1);
	//dim3 threads(numberOfThreads, 1, 1 );

	int Grid = (size_Mat / numberOfThreads) + 1;
	int Grid1, Grid2;
	if (Grid > 512)
	{
		Grid1 = 512;
		Grid2 = (Grid / Grid1) + 1;
	}
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2, 1);
	dim3 threads(numberOfThreads, 1, 1);

	LB_Kernel_Wale_Cum_AA2016_Debug_Comp_SP_27 << < grid, threads >> >(
		s9,
		bcMatD,
		neighborX,
		neighborY,
		neighborZ,
		neighborWSB,
		veloX,
		veloY,
		veloZ,
		DD,
		turbulentViscosity,
		gSij,
		gSDij,
		gDxvx,
		gDyvx,
		gDzvx,
		gDxvy,
		gDyvy,
		gDzvy,
		gDxvz,
		gDyvz,
		gDzvz,
		size_Mat,
		level,
		forces,
		EvenOrOdd);
	getLastCudaError("LB_Kernel_Wale_Cum_AA2016_Debug_Comp_SP_27 execution failed");
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelThS7(unsigned int numberOfThreads, 
                           real diffusivity,
                           unsigned int* bcMatD,
                           unsigned int* neighborX,
                           unsigned int* neighborY,
                           unsigned int* neighborZ,
                           real* DD,
                           real* DD7,
                           int size_Mat,
                           bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LB_Kernel_ThS7<<< grid, threads >>>(diffusivity,
                                          bcMatD,
                                          neighborX,
                                          neighborY,
                                          neighborZ,
                                          DD,
                                          DD7,
                                          size_Mat,
                                          EvenOrOdd); 
      getLastCudaError("LB_Kernel_ThS7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelThS27(  unsigned int numberOfThreads, 
                              real diffusivity,
                              unsigned int* bcMatD,
                              unsigned int* neighborX,
                              unsigned int* neighborY,
                              unsigned int* neighborZ,
                              real* DD,
                              real* DD27,
                              int size_Mat,
                              bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LB_Kernel_ThS27<<< grid, threads >>>( diffusivity,
                                            bcMatD,
                                            neighborX,
                                            neighborY,
                                            neighborZ,
                                            DD,
                                            DD27,
                                            size_Mat,
                                            EvenOrOdd); 
      getLastCudaError("LB_Kernel_ThS27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelADincomp7(   unsigned int numberOfThreads, 
								   real diffusivity,
								   unsigned int* bcMatD,
								   unsigned int* neighborX,
								   unsigned int* neighborY,
								   unsigned int* neighborZ,
								   real* DD,
								   real* DD7,
								   int size_Mat,
								   bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LB_Kernel_AD_Incomp_7<<< grid, threads >>>( diffusivity,
												  bcMatD,
												  neighborX,
												  neighborY,
												  neighborZ,
												  DD,
												  DD7,
												  size_Mat,
												  EvenOrOdd); 
      getLastCudaError("LB_Kernel_AD_Incomp_7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void KernelADincomp27( unsigned int numberOfThreads, 
								  real diffusivity,
								  unsigned int* bcMatD,
								  unsigned int* neighborX,
								  unsigned int* neighborY,
								  unsigned int* neighborZ,
								  real* DD,
								  real* DD27,
								  int size_Mat,
								  bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LB_Kernel_AD_Incomp_27<<< grid, threads >>>( diffusivity,
													bcMatD,
													neighborX,
													neighborY,
													neighborZ,
													DD,
													DD27,
													size_Mat,
													EvenOrOdd); 
      getLastCudaError("LB_Kernel_AD_Incomp_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void Init27( int myid,
                        int numprocs,
                        real u0,
                        unsigned int* geoD,
                        unsigned int* neighborX,
                        unsigned int* neighborY,
                        unsigned int* neighborZ,
                        real* vParab,
                        unsigned int size_Mat,
                        unsigned int grid_nx, 
                        unsigned int grid_ny, 
                        unsigned int grid_nz, 
                        real* DD,
                        int level,
                        int maxlevel)
{ 
   dim3 threads       ( grid_nx, 1, 1 );
   dim3 grid          ( grid_ny, grid_nz );   // Gitter fuer Kollision und Propagation

      LBInit27<<< grid, threads >>> (  myid, 
                                       numprocs, 
                                       u0, 
                                       geoD, 
                                       neighborX,
                                       neighborY,
                                       neighborZ,
                                       vParab, 
                                       size_Mat, 
                                       grid_nx, 
                                       grid_ny, 
                                       grid_nz, 
                                       DD,
                                       level,
                                       maxlevel); 
      getLastCudaError("LBInit27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void InitF3(     unsigned int numberOfThreads,
                            unsigned int* neighborX,
                            unsigned int* neighborY,
                            unsigned int* neighborZ,
                            unsigned int* geoD,
                            real* rho,
                            real* ux,
                            real* uy,
                            real* uz,
                            unsigned int size_Mat,
                            real* G6,
                            bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBInitF3<<< grid, threads >>>( neighborX,
                                       neighborY,
                                       neighborZ,
                                       geoD,
                                       rho,
                                       ux,
                                       uy,
                                       uz,
                                       size_Mat,
                                       G6,
                                       EvenOrOdd);
      getLastCudaError("LBInitF3 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void InitSP27(   unsigned int numberOfThreads,
                            unsigned int* neighborX,
                            unsigned int* neighborY,
                            unsigned int* neighborZ,
                            unsigned int* geoD,
                            real* rho,
                            real* ux,
                            real* uy,
                            real* uz,
                            unsigned int size_Mat,
                            real* DD,
                            bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBInitSP27<<< grid, threads >>>( neighborX,
                                       neighborY,
                                       neighborZ,
                                       geoD,
                                       rho,
                                       ux,
                                       uy,
                                       uz,
                                       size_Mat,
                                       DD,
                                       EvenOrOdd);
      getLastCudaError("LBInitSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void InitCompSP27(   unsigned int numberOfThreads,
								unsigned int* neighborX,
								unsigned int* neighborY,
								unsigned int* neighborZ,
								unsigned int* geoD,
								real* rho,
								real* ux,
								real* uy,
								real* uz,
								unsigned int size_Mat,
								real* DD,
								bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBInitCompSP27<<< grid, threads >>>( neighborX,
										   neighborY,
										   neighborZ,
										   geoD,
										   rho,
										   ux,
										   uy,
										   uz,
										   size_Mat,
										   DD,
										   EvenOrOdd);
      getLastCudaError("LBInitSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void InitThS7(     unsigned int numberOfThreads,
                              unsigned int* neighborX,
                              unsigned int* neighborY,
                              unsigned int* neighborZ,
                              unsigned int* geoD,
                              real* Conc,
                              real* ux,
                              real* uy,
                              real* uz,
                              unsigned int size_Mat,
                              real* DD7,
                              bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBInitThS7<<< grid, threads >>>( neighborX,
                                       neighborY,
                                       neighborZ,
                                       geoD,
                                       Conc,
                                       ux,
                                       uy,
                                       uz,
                                       size_Mat,
                                       DD7,
                                       EvenOrOdd);
      getLastCudaError("LBInitThS7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void InitThS27( unsigned int numberOfThreads,
                           unsigned int* neighborX,
                           unsigned int* neighborY,
                           unsigned int* neighborZ,
                           unsigned int* geoD,
                           real* Conc,
                           real* ux,
                           real* uy,
                           real* uz,
                           unsigned int size_Mat,
                           real* DD27,
                           bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBInitThS27<<< grid, threads >>>(neighborX,
                                       neighborY,
                                       neighborZ,
                                       geoD,
                                       Conc,
                                       ux,
                                       uy,
                                       uz,
                                       size_Mat,
                                       DD27,
                                       EvenOrOdd);
      getLastCudaError("LBInitThS27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void InitIncompAD7(unsigned int numberOfThreads,
                              unsigned int* neighborX,
                              unsigned int* neighborY,
                              unsigned int* neighborZ,
                              unsigned int* geoD,
                              real* Conc,
                              real* ux,
                              real* uy,
                              real* uz,
                              unsigned int size_Mat,
                              real* DD7,
                              bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBInitIncompAD7<<< grid, threads >>>(neighborX,
										   neighborY,
										   neighborZ,
										   geoD,
										   Conc,
										   ux,
										   uy,
										   uz,
										   size_Mat,
										   DD7,
										   EvenOrOdd);
      getLastCudaError("LBInitIncompAD7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void InitIncompAD27(unsigned int numberOfThreads,
							   unsigned int* neighborX,
							   unsigned int* neighborY,
							   unsigned int* neighborZ,
							   unsigned int* geoD,
							   real* Conc,
							   real* ux,
							   real* uy,
							   real* uz,
							   unsigned int size_Mat,
							   real* DD27,
							   bool EvenOrOdd)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBInitIncompAD27<<< grid, threads >>>(   neighborX,
											   neighborY,
											   neighborZ,
											   geoD,
											   Conc,
											   ux,
											   uy,
											   uz,
											   size_Mat,
											   DD27,
											   EvenOrOdd);
      getLastCudaError("LBInitIncompAD27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcMac27( real* vxD,
                           real* vyD,
                           real* vzD,
                           real* rhoD,
                           unsigned int* geoD,
                           unsigned int* neighborX,
                           unsigned int* neighborY,
                           unsigned int* neighborZ,
                           unsigned int size_Mat,
                           unsigned int grid_nx, 
                           unsigned int grid_ny, 
                           unsigned int grid_nz, 
                           real* DD,
                           bool evenOrOdd)
{ 
   dim3 threads       ( grid_nx, 1, 1 );
   dim3 grid          ( grid_ny, grid_nz );

      LBCalcMac27<<< grid, threads >>> (  vxD, 
                                          vyD, 
                                          vzD, 
                                          rhoD, 
                                          geoD, 
                                          neighborX,
                                          neighborY,
                                          neighborZ,
                                          size_Mat, 
                                          DD, 
                                          evenOrOdd); 
      getLastCudaError("LBCalcMac27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcMacSP27( real* vxD,
                             real* vyD,
                             real* vzD,
                             real* rhoD,
                             real* pressD,
                             unsigned int* geoD,
                             unsigned int* neighborX,
                             unsigned int* neighborY,
                             unsigned int* neighborZ,
                             unsigned int size_Mat,
                             unsigned int numberOfThreads, 
                             real* DD,
                             bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcMacSP27<<< grid, threads >>> (   vxD, 
                                             vyD, 
                                             vzD, 
                                             rhoD, 
                                             pressD, 
                                             geoD, 
                                             neighborX,
                                             neighborY,
                                             neighborZ,
                                             size_Mat, 
                                             DD, 
                                             evenOrOdd); 
      getLastCudaError("LBCalcMacSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcMacCompSP27( real* vxD,
								 real* vyD,
								 real* vzD,
								 real* rhoD,
								 real* pressD,
								 unsigned int* geoD,
								 unsigned int* neighborX,
								 unsigned int* neighborY,
								 unsigned int* neighborZ,
								 unsigned int size_Mat,
								 unsigned int numberOfThreads, 
								 real* DD,
								 bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcMacCompSP27<<< grid, threads >>> (   vxD, 
												 vyD, 
												 vzD, 
												 rhoD, 
												 pressD, 
												 geoD, 
												 neighborX,
												 neighborY,
												 neighborZ,
												 size_Mat, 
												 DD, 
												 evenOrOdd); 
      getLastCudaError("LBCalcMacSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcMacThS7(  real* Conc,
                              unsigned int* geoD,
                              unsigned int* neighborX,
                              unsigned int* neighborY,
                              unsigned int* neighborZ,
                              unsigned int size_Mat,
                              unsigned int numberOfThreads, 
                              real* DD7,
                              bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcMacThS7<<< grid, threads >>> (Conc, 
                                          geoD, 
                                          neighborX,
                                          neighborY,
                                          neighborZ,
                                          size_Mat, 
                                          DD7, 
                                          evenOrOdd); 
      getLastCudaError("LBCalcMacThS7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void PlaneConcThS7(real* Conc,
							  int* kPC,
							  unsigned int numberOfPointskPC,
							  unsigned int* geoD,
							  unsigned int* neighborX,
							  unsigned int* neighborY,
							  unsigned int* neighborZ,
							  unsigned int size_Mat,
                              unsigned int numberOfThreads, 
							  real* DD7,
							  bool evenOrOdd)
{ 
   int Grid = (numberOfPointskPC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      GetPlaneConcThS7<<< grid, threads >>> (	Conc,
												kPC,
												numberOfPointskPC,
												geoD, 
												neighborX,
												neighborY,
												neighborZ,
												size_Mat, 
												DD7, 
												evenOrOdd); 
      getLastCudaError("GetPlaneConcThS7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void PlaneConcThS27(real* Conc,
							   int* kPC,
							   unsigned int numberOfPointskPC,
							   unsigned int* geoD,
							   unsigned int* neighborX,
							   unsigned int* neighborY,
							   unsigned int* neighborZ,
							   unsigned int size_Mat,
                               unsigned int numberOfThreads, 
							   real* DD27,
							   bool evenOrOdd)
{ 
   int Grid = (numberOfPointskPC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      GetPlaneConcThS27<<< grid, threads >>> (	Conc,
												kPC,
												numberOfPointskPC,
												geoD, 
												neighborX,
												neighborY,
												neighborZ,
												size_Mat, 
												DD27, 
												evenOrOdd); 
      getLastCudaError("GetPlaneConcThS27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcMacThS27( real* Conc,
                              unsigned int* geoD,
                              unsigned int* neighborX,
                              unsigned int* neighborY,
                              unsigned int* neighborZ,
                              unsigned int size_Mat,
                              unsigned int numberOfThreads, 
                              real* DD27,
                              bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcMacThS27<<< grid, threads >>> (  Conc, 
                                             geoD, 
                                             neighborX,
                                             neighborY,
                                             neighborZ,
                                             size_Mat, 
                                             DD27, 
                                             evenOrOdd); 
      getLastCudaError("LBCalcMacThS27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcMedSP27(  real* vxD,
                              real* vyD,
                              real* vzD,
                              real* rhoD,
                              real* pressD,
                              unsigned int* geoD,
                              unsigned int* neighborX,
                              unsigned int* neighborY,
                              unsigned int* neighborZ,
                              unsigned int size_Mat,
                              unsigned int numberOfThreads, 
                              real* DD,
                              bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcMedSP27<<< grid, threads >>> (   vxD, 
                                             vyD, 
                                             vzD, 
                                             rhoD, 
                                             pressD, 
                                             geoD, 
                                             neighborX,
                                             neighborY,
                                             neighborZ,
                                             size_Mat, 
                                             DD, 
                                             evenOrOdd); 
      getLastCudaError("LBCalcMedSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcMedCompSP27(  real* vxD,
								  real* vyD,
								  real* vzD,
								  real* rhoD,
								  real* pressD,
								  unsigned int* geoD,
								  unsigned int* neighborX,
								  unsigned int* neighborY,
								  unsigned int* neighborZ,
								  unsigned int size_Mat,
								  unsigned int numberOfThreads, 
								  real* DD,
								  bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcMedCompSP27<<< grid, threads >>> (   vxD, 
												 vyD, 
												 vzD, 
												 rhoD, 
												 pressD, 
												 geoD, 
												 neighborX,
												 neighborY,
												 neighborZ,
												 size_Mat, 
												 DD, 
												 evenOrOdd); 
      getLastCudaError("LBCalcMedSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcMacMedSP27(  real* vxD,
                                 real* vyD,
                                 real* vzD,
                                 real* rhoD,
                                 real* pressD,
                                 unsigned int* geoD,
                                 unsigned int* neighborX,
                                 unsigned int* neighborY,
                                 unsigned int* neighborZ,
                                 unsigned int tdiff,
                                 unsigned int size_Mat,
                                 unsigned int numberOfThreads, 
                                 bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcMacMedSP27<<< grid, threads >>> (   vxD, 
                                                vyD, 
                                                vzD, 
                                                rhoD, 
                                                pressD, 
                                                geoD, 
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                tdiff,
                                                size_Mat,
                                                evenOrOdd); 
      getLastCudaError("LBCalcMacMedSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ResetMedianValuesSP27(
	real* vxD,
	real* vyD,
	real* vzD,
	real* rhoD,
	real* pressD,
	unsigned int size_Mat,
	unsigned int numberOfThreads,
	bool evenOrOdd)
{
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

	LBResetMedianValuesSP27 << < grid, threads >> > (
		vxD,
		vyD,
		vzD,
		rhoD,
		pressD,
		size_Mat,
		evenOrOdd);
	getLastCudaError("LBResetMedianValuesSP27 execution failed");
}
//////////////////////////////////////////////////////////////////////////
extern "C" void Calc2ndMomentsIncompSP27(real* kxyFromfcNEQ,
										 real* kyzFromfcNEQ,
										 real* kxzFromfcNEQ,
										 real* kxxMyyFromfcNEQ,
										 real* kxxMzzFromfcNEQ,
										 unsigned int* geoD,
										 unsigned int* neighborX,
										 unsigned int* neighborY,
										 unsigned int* neighborZ,
										 unsigned int size_Mat,
										 unsigned int numberOfThreads, 
										 real* DD,
										 bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalc2ndMomentsIncompSP27<<< grid, threads >>> (  kxyFromfcNEQ,
														 kyzFromfcNEQ,
														 kxzFromfcNEQ,
														 kxxMyyFromfcNEQ,
														 kxxMzzFromfcNEQ, 
														 geoD, 
														 neighborX,
														 neighborY,
														 neighborZ,
														 size_Mat, 
														 DD, 
														 evenOrOdd); 
      getLastCudaError("LBCalc2ndMomentsIncompSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void Calc2ndMomentsCompSP27( real* kxyFromfcNEQ,
										real* kyzFromfcNEQ,
										real* kxzFromfcNEQ,
										real* kxxMyyFromfcNEQ,
										real* kxxMzzFromfcNEQ,
										unsigned int* geoD,
										unsigned int* neighborX,
										unsigned int* neighborY,
										unsigned int* neighborZ,
										unsigned int size_Mat,
										unsigned int numberOfThreads, 
										real* DD,
										bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalc2ndMomentsCompSP27<<< grid, threads >>> (kxyFromfcNEQ,
													 kyzFromfcNEQ,
													 kxzFromfcNEQ,
													 kxxMyyFromfcNEQ,
													 kxxMzzFromfcNEQ, 
													 geoD, 
													 neighborX,
													 neighborY,
													 neighborZ,
													 size_Mat, 
													 DD, 
													 evenOrOdd); 
      getLastCudaError("LBCalc2ndMomentsCompSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void Calc3rdMomentsIncompSP27(real* CUMbbb,
										 real* CUMabc,
										 real* CUMbac,
										 real* CUMbca,
										 real* CUMcba,
										 real* CUMacb,
										 real* CUMcab,
										 unsigned int* geoD,
										 unsigned int* neighborX,
										 unsigned int* neighborY,
										 unsigned int* neighborZ,
										 unsigned int size_Mat,
										 unsigned int numberOfThreads, 
										 real* DD,
										 bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalc3rdMomentsIncompSP27<<< grid, threads >>> (  CUMbbb,
														 CUMabc,
														 CUMbac,
														 CUMbca,
														 CUMcba, 
														 CUMacb, 
														 CUMcab, 
														 geoD, 
														 neighborX,
														 neighborY,
														 neighborZ,
														 DD, 
														 size_Mat, 
														 evenOrOdd); 
      getLastCudaError("LBCalc3rdMomentsIncompSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void Calc3rdMomentsCompSP27( real* CUMbbb,
										real* CUMabc,
										real* CUMbac,
										real* CUMbca,
										real* CUMcba,
										real* CUMacb,
										real* CUMcab,
										unsigned int* geoD,
										unsigned int* neighborX,
										unsigned int* neighborY,
										unsigned int* neighborZ,
										unsigned int size_Mat,
										unsigned int numberOfThreads, 
										real* DD,
										bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalc3rdMomentsCompSP27<<< grid, threads >>> (CUMbbb,
													 CUMabc,
													 CUMbac,
													 CUMbca,
													 CUMcba, 
													 CUMacb, 
													 CUMcab, 
													 geoD, 
													 neighborX,
													 neighborY,
													 neighborZ,
													 DD, 
													 size_Mat, 
													 evenOrOdd); 
      getLastCudaError("LBCalc3rdMomentsCompSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcHigherMomentsIncompSP27(real* CUMcbb,
											real* CUMbcb,
											real* CUMbbc,
											real* CUMcca,
											real* CUMcac,
											real* CUMacc,
											real* CUMbcc,
											real* CUMcbc,
											real* CUMccb,
											real* CUMccc,
											unsigned int* geoD,
											unsigned int* neighborX,
											unsigned int* neighborY,
											unsigned int* neighborZ,
											unsigned int size_Mat,
											unsigned int numberOfThreads, 
											real* DD,
											bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcHigherMomentsIncompSP27<<< grid, threads >>> (CUMcbb,
														  CUMbcb,
														  CUMbbc,
														  CUMcca,
														  CUMcac, 
														  CUMacc, 
														  CUMbcc, 
														  CUMcbc, 
														  CUMccb, 
														  CUMccc, 
														  geoD, 
														  neighborX,
														  neighborY,
														  neighborZ,
														  DD, 
														  size_Mat, 
														  evenOrOdd); 
      getLastCudaError("LBCalcHigherMomentsIncompSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcHigherMomentsCompSP27(  real* CUMcbb,
											real* CUMbcb,
											real* CUMbbc,
											real* CUMcca,
											real* CUMcac,
											real* CUMacc,
											real* CUMbcc,
											real* CUMcbc,
											real* CUMccb,
											real* CUMccc,
											unsigned int* geoD,
											unsigned int* neighborX,
											unsigned int* neighborY,
											unsigned int* neighborZ,
											unsigned int size_Mat,
											unsigned int numberOfThreads, 
											real* DD,
											bool evenOrOdd)
{ 
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcHigherMomentsCompSP27<<< grid, threads >>> (  CUMcbb,
														  CUMbcb,
														  CUMbbc,
														  CUMcca,
														  CUMcac, 
														  CUMacc, 
														  CUMbcc, 
														  CUMcbc, 
														  CUMccb, 
														  CUMccc, 
														  geoD, 
														  neighborX,
														  neighborY,
														  neighborZ,
														  DD, 
														  size_Mat, 
														  evenOrOdd); 
      getLastCudaError("LBCalcHigherMomentsCompSP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void LBCalcMeasurePoints27(real* vxMP,
                                      real* vyMP,
                                      real* vzMP,
                                      real* rhoMP,
                                      unsigned int* kMP,
                                      unsigned int numberOfPointskMP,
                                      unsigned int MPClockCycle,
                                      unsigned int t,
                                      unsigned int* geoD,
                                      unsigned int* neighborX,
                                      unsigned int* neighborY,
                                      unsigned int* neighborZ,
                                      unsigned int size_Mat,
                                      real* DD,
                                      unsigned int numberOfThreads, 
                                      bool evenOrOdd)
{ 
   int Grid = (numberOfPointskMP / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBCalcMeasurePoints<<< grid, threads >>> (vxMP,
                                                vyMP,
                                                vzMP,
                                                rhoMP,
                                                kMP,
                                                numberOfPointskMP,
                                                MPClockCycle,
                                                t,
                                                geoD,
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                size_Mat,
                                                DD,
                                                evenOrOdd); 
      getLastCudaError("LBCalcMeasurePoints execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void BcPress27( int nx, 
                           int ny, 
                           int tz, 
                           unsigned int grid_nx, 
                           unsigned int grid_ny, 
                           unsigned int* bcMatD, 
                           unsigned int* neighborX,
                           unsigned int* neighborY,
                           unsigned int* neighborZ,
                           real* DD, 
                           unsigned int size_Mat, 
                           bool evenOrOdd) 
{
   dim3 threads       ( grid_nx, 1, 1 );
   dim3 grid          ( grid_ny, 1 );

      LB_BC_Press_East27<<< grid, threads >>> ( nx, 
                                                ny, 
                                                tz, 
                                                bcMatD, 
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                DD, 
                                                size_Mat, 
                                                evenOrOdd); 
      getLastCudaError("LB_BC_Press_East27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void BcVel27(int nx, 
                        int ny, 
                        int nz, 
                        int itz, 
                        unsigned int grid_nx, 
                        unsigned int grid_ny, 
                        unsigned int* bcMatD, 
                        unsigned int* neighborX,
                        unsigned int* neighborY,
                        unsigned int* neighborZ,
                        real* DD, 
                        unsigned int size_Mat, 
                        bool evenOrOdd, 
                        real u0x, 
                        real om)
{
   dim3 threads       ( grid_nx, 1, 1 );
   dim3 grid          ( grid_ny, 1 );

      LB_BC_Vel_West_27<<< grid, threads >>> (  nx, 
                                                ny, 
                                                nz, 
                                                itz, 
                                                bcMatD, 
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                DD, 
                                                size_Mat, 
                                                evenOrOdd, 
                                                u0x,
                                                grid_nx, 
                                                grid_ny, 
                                                om); 
      getLastCudaError("LB_BC_Vel_West_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADPressDev7( unsigned int numberOfThreads,
                              int nx,
                              int ny,
                              real* DD, 
                              real* DD7,
                              real* temp,
                              real* velo,
                              real diffusivity,
                              int* k_Q, 
                              real* QQ,
                              unsigned int sizeQ,
                              unsigned int kQ, 
                              real om1, 
                              unsigned int* neighborX,
                              unsigned int* neighborY,
                              unsigned int* neighborZ,
                              unsigned int size_Mat, 
                              bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADPress7<<< gridQ, threads >>>( nx,
                                       ny,
                                       DD, 
                                       DD7,
                                       temp,
                                       velo,
                                       diffusivity,
                                       k_Q, 
                                       QQ,
                                       sizeQ,
                                       kQ, 
                                       om1, 
                                       neighborX,
                                       neighborY,
                                       neighborZ,
                                       size_Mat, 
                                       evenOrOdd);
      getLastCudaError("QADPress7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADPressDev27(unsigned int numberOfThreads,
                              int nx,
                              int ny,
                              real* DD, 
                              real* DD27,
                              real* temp,
                              real* velo,
                              real diffusivity,
                              int* k_Q, 
                              real* QQ,
                              unsigned int sizeQ,
                              unsigned int kQ, 
                              real om1, 
                              unsigned int* neighborX,
                              unsigned int* neighborY,
                              unsigned int* neighborZ,
                              unsigned int size_Mat, 
                              bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADPress27<<< gridQ, threads >>>(   nx,
                                          ny,
                                          DD, 
                                          DD27,
                                          temp,
                                          velo,
                                          diffusivity,
                                          k_Q, 
                                          QQ,
                                          sizeQ,
                                          kQ, 
                                          om1, 
                                          neighborX,
                                          neighborY,
                                          neighborZ,
                                          size_Mat, 
                                          evenOrOdd);
      getLastCudaError("QADPress27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADVelDev7(unsigned int numberOfThreads,
                           int nx,
                           int ny,
                           real* DD, 
                           real* DD7,
                           real* temp,
                           real* velo,
                           real diffusivity,
                           int* k_Q, 
                           real* QQ,
                           unsigned int sizeQ,
                           unsigned int kQ, 
                           real om1, 
                           unsigned int* neighborX,
                           unsigned int* neighborY,
                           unsigned int* neighborZ,
                           unsigned int size_Mat, 
                           bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADVel7<<< gridQ, threads >>> (  nx,
                                       ny,
                                       DD, 
                                       DD7,
                                       temp,
                                       velo,
                                       diffusivity,
                                       k_Q, 
                                       QQ,
                                       sizeQ,
                                       kQ, 
                                       om1, 
                                       neighborX,
                                       neighborY,
                                       neighborZ,
                                       size_Mat, 
                                       evenOrOdd);
      getLastCudaError("QADVel7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADVelDev27(  unsigned int numberOfThreads,
                              int nx,
                              int ny,
                              real* DD, 
                              real* DD27,
                              real* temp,
                              real* velo,
                              real diffusivity,
                              int* k_Q, 
                              real* QQ,
                              unsigned int sizeQ,
                              unsigned int kQ, 
                              real om1, 
                              unsigned int* neighborX,
                              unsigned int* neighborY,
                              unsigned int* neighborZ,
                              unsigned int size_Mat, 
                              bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADVel27<<< gridQ, threads >>> (nx,
                                      ny,
                                      DD, 
                                      DD27,
                                      temp,
                                      velo,
                                      diffusivity,
                                      k_Q, 
                                      QQ,
                                      sizeQ,
                                      kQ, 
                                      om1, 
                                      neighborX,
                                      neighborY,
                                      neighborZ,
                                      size_Mat, 
                                      evenOrOdd);
      getLastCudaError("QADVel27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADDev7(unsigned int numberOfThreads,
                        int nx,
                        int ny,
                        real* DD, 
                        real* DD7,
                        real* temp,
                        real diffusivity,
                        int* k_Q, 
                        real* QQ,
                        unsigned int sizeQ,
                        unsigned int kQ, 
                        real om1, 
                        unsigned int* neighborX,
                        unsigned int* neighborY,
                        unsigned int* neighborZ,
                        unsigned int size_Mat, 
                        bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QAD7<<< gridQ, threads >>> (     nx,
                                       ny,
                                       DD, 
                                       DD7,
                                       temp,
                                       diffusivity,
                                       k_Q, 
                                       QQ,
                                       sizeQ,
                                       kQ, 
                                       om1, 
                                       neighborX,
                                       neighborY,
                                       neighborZ,
                                       size_Mat, 
                                       evenOrOdd);
      getLastCudaError("QAD7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADDirichletDev27( unsigned int numberOfThreads,
								   int nx,
								   int ny,
								   real* DD, 
								   real* DD27,
								   real* temp,
								   real diffusivity,
								   int* k_Q, 
								   real* QQ,
								   unsigned int sizeQ,
								   unsigned int kQ, 
								   real om1, 
								   unsigned int* neighborX,
								   unsigned int* neighborY,
								   unsigned int* neighborZ,
								   unsigned int size_Mat, 
								   bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADDirichlet27<<< gridQ, threads >>> (   nx,
											   ny,
											   DD, 
											   DD27,
											   temp,
											   diffusivity,
											   k_Q, 
											   QQ,
											   sizeQ,
											   kQ, 
											   om1, 
											   neighborX,
											   neighborY,
											   neighborZ,
											   size_Mat, 
											   evenOrOdd);
      getLastCudaError("QAD27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADBBDev27(unsigned int numberOfThreads,
                           int nx,
                           int ny,
                           real* DD, 
                           real* DD27,
                           real* temp,
                           real diffusivity,
                           int* k_Q, 
                           real* QQ,
                           unsigned int sizeQ,
                           unsigned int kQ, 
                           real om1, 
                           unsigned int* neighborX,
                           unsigned int* neighborY,
                           unsigned int* neighborZ,
                           unsigned int size_Mat, 
                           bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADBB27<<< gridQ, threads >>> (  nx,
                                       ny,
                                       DD, 
                                       DD27,
                                       temp,
                                       diffusivity,
                                       k_Q, 
                                       QQ,
                                       sizeQ,
                                       kQ, 
                                       om1, 
                                       neighborX,
                                       neighborY,
                                       neighborZ,
                                       size_Mat, 
                                       evenOrOdd);
      getLastCudaError("QADBB27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QNoSlipADincompDev7(unsigned int numberOfThreads,
									int nx,
									int ny,
									real* DD, 
									real* DD7,
									real* temp,
									real diffusivity,
									int* k_Q, 
									real* QQ,
									unsigned int sizeQ,
									unsigned int kQ, 
									real om1, 
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									unsigned int size_Mat, 
									bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QNoSlipADincomp7<<< gridQ, threads >>> ( nx,
											   ny,
											   DD, 
											   DD7,
											   temp,
											   diffusivity,
											   k_Q, 
											   QQ,
											   sizeQ,
											   kQ, 
											   om1, 
											   neighborX,
											   neighborY,
											   neighborZ,
											   size_Mat, 
											   evenOrOdd);
      getLastCudaError("QNoSlipADincomp7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QNoSlipADincompDev27(  unsigned int numberOfThreads,
									   int nx,
									   int ny,
									   real* DD, 
									   real* DD27,
									   real* temp,
									   real diffusivity,
									   int* k_Q, 
									   real* QQ,
									   unsigned int sizeQ,
									   unsigned int kQ, 
									   real om1, 
									   unsigned int* neighborX,
									   unsigned int* neighborY,
									   unsigned int* neighborZ,
									   unsigned int size_Mat, 
									   bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QNoSlipADincomp27<<< gridQ, threads >>> (nx,
											   ny,
											   DD, 
											   DD27,
											   temp,
											   diffusivity,
											   k_Q, 
											   QQ,
											   sizeQ,
											   kQ, 
											   om1, 
											   neighborX,
											   neighborY,
											   neighborZ,
											   size_Mat, 
											   evenOrOdd);
      getLastCudaError("QNoSlipADincomp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADVeloIncompDev7( unsigned int numberOfThreads,
								   int nx,
								   int ny,
								   real* DD, 
								   real* DD7,
								   real* temp,
								   real* velo,
								   real diffusivity,
								   int* k_Q, 
								   real* QQ,
								   unsigned int sizeQ,
								   unsigned int kQ, 
								   real om1, 
								   unsigned int* neighborX,
								   unsigned int* neighborY,
								   unsigned int* neighborZ,
								   unsigned int size_Mat, 
								   bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADVeloIncomp7<<< gridQ, threads >>> (   nx,
											   ny,
											   DD, 
											   DD7,
											   temp,
											   velo,
											   diffusivity,
											   k_Q, 
											   QQ,
											   sizeQ,
											   kQ, 
											   om1, 
											   neighborX,
											   neighborY,
											   neighborZ,
											   size_Mat, 
											   evenOrOdd);
      getLastCudaError("QADVeloIncomp7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADVeloIncompDev27(   unsigned int numberOfThreads,
									  int nx,
									  int ny,
									  real* DD, 
									  real* DD27,
									  real* temp,
									  real* velo,
									  real diffusivity,
									  int* k_Q, 
									  real* QQ,
									  unsigned int sizeQ,
									  unsigned int kQ, 
									  real om1, 
									  unsigned int* neighborX,
									  unsigned int* neighborY,
									  unsigned int* neighborZ,
									  unsigned int size_Mat, 
									  bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADVeloIncomp27<<< gridQ, threads >>> ( nx,
											  ny,
											  DD, 
											  DD27,
											  temp,
											  velo,
											  diffusivity,
											  k_Q, 
											  QQ,
											  sizeQ,
											  kQ, 
											  om1, 
											  neighborX,
											  neighborY,
											  neighborZ,
											  size_Mat, 
											  evenOrOdd);
      getLastCudaError("QADVeloIncomp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADPressIncompDev7(   unsigned int numberOfThreads,
									  int nx,
									  int ny,
									  real* DD, 
									  real* DD7,
									  real* temp,
									  real* velo,
									  real diffusivity,
									  int* k_Q, 
									  real* QQ,
									  unsigned int sizeQ,
									  unsigned int kQ, 
									  real om1, 
									  unsigned int* neighborX,
									  unsigned int* neighborY,
									  unsigned int* neighborZ,
									  unsigned int size_Mat, 
									  bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADPressIncomp7<<< gridQ, threads >>>(   nx,
											   ny,
											   DD, 
											   DD7,
											   temp,
											   velo,
											   diffusivity,
											   k_Q, 
											   QQ,
											   sizeQ,
											   kQ, 
											   om1, 
											   neighborX,
											   neighborY,
											   neighborZ,
											   size_Mat, 
											   evenOrOdd);
      getLastCudaError("QADPressIncomp7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QADPressIncompDev27(  unsigned int numberOfThreads,
									  int nx,
									  int ny,
									  real* DD, 
									  real* DD27,
									  real* temp,
									  real* velo,
									  real diffusivity,
									  int* k_Q, 
									  real* QQ,
									  unsigned int sizeQ,
									  unsigned int kQ, 
									  real om1, 
									  unsigned int* neighborX,
									  unsigned int* neighborY,
									  unsigned int* neighborZ,
									  unsigned int size_Mat, 
									  bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QADPressIncomp27<<< gridQ, threads >>>( nx,
											  ny,
											  DD, 
											  DD27,
											  temp,
											  velo,
											  diffusivity,
											  k_Q, 
											  QQ,
											  sizeQ,
											  kQ, 
											  om1, 
											  neighborX,
											  neighborY,
											  neighborZ,
											  size_Mat, 
											  evenOrOdd);
      getLastCudaError("QADPressIncomp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QDev27( unsigned int numberOfThreads,
                        int nx,
                        int ny,
                        real* DD, 
                        int* k_Q, 
                        real* QQ,
                        unsigned int sizeQ,
                        unsigned int kQ, 
                        real om1, 
                        unsigned int* neighborX,
                        unsigned int* neighborY,
                        unsigned int* neighborZ,
                        unsigned int size_Mat, 
                        bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QDevice27<<< gridQ, threads >>> (nx,
                                       ny,
                                       DD, 
                                       k_Q, 
                                       QQ,
                                       sizeQ,
                                       kQ, 
                                       om1, 
                                       neighborX,
                                       neighborY,
                                       neighborZ,
                                       size_Mat, 
                                       evenOrOdd);
      getLastCudaError("QDevice27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QDevComp27( unsigned int numberOfThreads,
							int nx,
							int ny,
							real* DD, 
							int* k_Q, 
							real* QQ,
							unsigned int sizeQ,
							unsigned int kQ, 
							real om1, 
							unsigned int* neighborX,
							unsigned int* neighborY,
							unsigned int* neighborZ,
							unsigned int size_Mat, 
							bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QDeviceComp27<<< gridQ, threads >>> (nx,
										   ny,
										   DD, 
										   k_Q, 
										   QQ,
										   sizeQ,
										   kQ, 
										   om1, 
										   neighborX,
										   neighborY,
										   neighborZ,
										   size_Mat, 
										   evenOrOdd);
      getLastCudaError("QDeviceComp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QDevCompThinWallsPartOne27( unsigned int numberOfThreads,
											int nx,
											int ny,
											real* DD, 
											int* k_Q, 
											real* QQ,
											unsigned int sizeQ,
											unsigned int kQ, 
											real om1, 
											unsigned int* neighborX,
											unsigned int* neighborY,
											unsigned int* neighborZ,
											unsigned int size_Mat, 
											bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

   QDeviceCompThinWallsPartOne27 <<< gridQ, threads >>> (nx,
														 ny,
														 DD, 
														 k_Q, 
														 QQ,
														 sizeQ,
														 kQ, 
														 om1, 
														 neighborX,
														 neighborY,
														 neighborZ,
														 size_Mat, 
														 evenOrOdd);
   getLastCudaError("QDeviceCompThinWallsPartOne27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QDevCompThinWallsPartTwo27( unsigned int numberOfThreads,
											int nx,
											int ny,
											real* DD, 
											int* k_Q, 
											real* QQ,
											unsigned int sizeQ,
											unsigned int kQ, 
											real om1, 
											unsigned int* geom,
											unsigned int* neighborX,
											unsigned int* neighborY,
											unsigned int* neighborZ,
											unsigned int* neighborWSB,
											unsigned int size_Mat, 
											bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

   QDeviceCompThinWallsPartTwo27 <<< gridQ, threads >>> (nx,
														 ny,
														 DD, 
														 k_Q, 
														 QQ,
														 sizeQ,
														 kQ, 
														 om1, 
														 geom,
														 neighborX,
														 neighborY,
														 neighborZ,
														 neighborWSB,
														 size_Mat, 
														 evenOrOdd);
   getLastCudaError("QDeviceCompThinWallsPartTwo27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QDev3rdMomentsComp27(   unsigned int numberOfThreads,
										int nx,
										int ny,
										real* DD, 
										int* k_Q, 
										real* QQ,
										unsigned int sizeQ,
										unsigned int kQ, 
										real om1, 
										unsigned int* neighborX,
										unsigned int* neighborY,
										unsigned int* neighborZ,
										unsigned int size_Mat, 
										bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QDevice3rdMomentsComp27<<< gridQ, threads >>> (  nx,
													   ny,
													   DD, 
													   k_Q, 
													   QQ,
													   sizeQ,
													   kQ, 
													   om1, 
													   neighborX,
													   neighborY,
													   neighborZ,
													   size_Mat, 
													   evenOrOdd);
      getLastCudaError("QDevice3rdMomentsComp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QDevIncompHighNu27( unsigned int numberOfThreads,
									int nx,
									int ny,
									real* DD, 
									int* k_Q, 
									real* QQ,
									unsigned int sizeQ,
									unsigned int kQ, 
									real om1, 
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									unsigned int size_Mat, 
									bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QDeviceIncompHighNu27<<< gridQ, threads >>> (nx,
												   ny,
												   DD, 
												   k_Q, 
												   QQ,
												   sizeQ,
												   kQ, 
												   om1, 
												   neighborX,
												   neighborY,
												   neighborZ,
												   size_Mat, 
												   evenOrOdd);
      getLastCudaError("QDeviceIncompHighNu27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QDevCompHighNu27(   unsigned int numberOfThreads,
									int nx,
									int ny,
									real* DD, 
									int* k_Q, 
									real* QQ,
									unsigned int sizeQ,
									unsigned int kQ, 
									real om1, 
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									unsigned int size_Mat, 
									bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QDeviceCompHighNu27<<< gridQ, threads >>> (  nx,
												   ny,
												   DD, 
												   k_Q, 
												   QQ,
												   sizeQ,
												   kQ, 
												   om1, 
												   neighborX,
												   neighborY,
												   neighborZ,
												   size_Mat, 
												   evenOrOdd);
      getLastCudaError("QDevice27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVelDevicePlainBB27(unsigned int numberOfThreads,
									real* vx,
									real* vy,
									real* vz,
									real* DD,
									int* k_Q, 
									real* QQ,
									unsigned int sizeQ,
									int kQ, 
									real om1, 
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									unsigned int size_Mat, 
									bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QVelDevPlainBB27<<< gridQ, threads >>> (  vx,
												vy,
												vz,
												DD,
												k_Q, 
												QQ,
												sizeQ,
												kQ, 
												om1, 
												neighborX,
												neighborY,
												neighborZ,
												size_Mat,
												evenOrOdd);
      getLastCudaError("QVelDevicePlainBB27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVelDeviceCouhette27(unsigned int numberOfThreads,
									real* vx,
									real* vy,
									real* vz,
									real* DD,
									int* k_Q, 
									real* QQ,
									unsigned int sizeQ,
									int kQ, 
									real om1, 
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									unsigned int size_Mat, 
									bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QVelDevCouhette27<<< gridQ, threads >>> ( vx,
												vy,
												vz,
												DD,
												k_Q, 
												QQ,
												sizeQ,
												kQ, 
												om1, 
												neighborX,
												neighborY,
												neighborZ,
												size_Mat,
												evenOrOdd);
      getLastCudaError("QVelDevicePlainBB27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVelDevice1h27(   unsigned int numberOfThreads,
								  int nx,
								  int ny,
								  real* vx,
								  real* vy,
								  real* vz,
								  real* DD, 
								  int* k_Q, 
								  real* QQ,
								  unsigned int sizeQ,
								  unsigned int kQ, 
								  real om1, 
								  real Phi, 
								  real angularVelocity,
								  unsigned int* neighborX,
								  unsigned int* neighborY,
								  unsigned int* neighborZ,
								  real* coordX,
								  real* coordY,
								  real* coordZ,
								  unsigned int size_Mat, 
								  bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QVelDev1h27<<< gridQ, threads >>> (nx,
                                          ny,
                                          vx,
                                          vy,
                                          vz,
                                          DD, 
                                          k_Q, 
                                          QQ,
                                          sizeQ,
                                          kQ, 
                                          om1,
										  Phi,
										  angularVelocity,
                                          neighborX,
                                          neighborY,
                                          neighborZ,
										  coordX,
										  coordY,
										  coordZ,
                                          size_Mat, 
                                          evenOrOdd);
      getLastCudaError("QVelDevice27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVelDev27(unsigned int numberOfThreads,
                          int nx,
                          int ny,
                          real* vx,
                          real* vy,
                          real* vz,
                          real* DD, 
                          int* k_Q, 
                          real* QQ,
                          unsigned int sizeQ,
                          unsigned int kQ, 
                          real om1, 
                          unsigned int* neighborX,
                          unsigned int* neighborY,
                          unsigned int* neighborZ,
                          unsigned int size_Mat, 
                          bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QVelDevice27<<< gridQ, threads >>> (nx,
                                          ny,
                                          vx,
                                          vy,
                                          vz,
                                          DD, 
                                          k_Q, 
                                          QQ,
                                          sizeQ,
                                          kQ, 
                                          om1, 
                                          neighborX,
                                          neighborY,
                                          neighborZ,
                                          size_Mat, 
                                          evenOrOdd);
      getLastCudaError("QVelDevice27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVelDevCompPlusSlip27(unsigned int numberOfThreads,
									  int nx,
									  int ny,
									  real* vx,
									  real* vy,
									  real* vz,
									  real* DD, 
									  int* k_Q, 
									  real* QQ,
									  unsigned int sizeQ,
									  unsigned int kQ, 
									  real om1, 
									  unsigned int* neighborX,
									  unsigned int* neighborY,
									  unsigned int* neighborZ,
									  unsigned int size_Mat, 
									  bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QVelDeviceCompPlusSlip27<<< gridQ, threads >>> (nx,
													  ny,
													  vx,
													  vy,
													  vz,
													  DD, 
													  k_Q, 
													  QQ,
													  sizeQ,
													  kQ, 
													  om1, 
													  neighborX,
													  neighborY,
													  neighborZ,
													  size_Mat, 
													  evenOrOdd);
      getLastCudaError("QVelDeviceCompPlusSlip27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVelDevComp27(unsigned int numberOfThreads,
							  int nx,
							  int ny,
							  real* vx,
							  real* vy,
							  real* vz,
							  real* DD, 
							  int* k_Q, 
							  real* QQ,
							  unsigned int sizeQ,
							  unsigned int kQ, 
							  real om1, 
							  unsigned int* neighborX,
							  unsigned int* neighborY,
							  unsigned int* neighborZ,
							  unsigned int size_Mat, 
							  bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QVelDeviceComp27<<< gridQ, threads >>> (nx,
											  ny,
											  vx,
											  vy,
											  vz,
											  DD, 
											  k_Q, 
											  QQ,
											  sizeQ,
											  kQ, 
											  om1, 
											  neighborX,
											  neighborY,
											  neighborZ,
											  size_Mat, 
											  evenOrOdd);
      getLastCudaError("QVelDeviceComp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVelDevCompZeroPress27(   unsigned int numberOfThreads,
										  int nx,
										  int ny,
										  real* vx,
										  real* vy,
										  real* vz,
										  real* DD, 
										  int* k_Q, 
										  real* QQ,
										  unsigned int sizeQ,
										  int kArray, 
										  real om1, 
										  unsigned int* neighborX,
										  unsigned int* neighborY,
										  unsigned int* neighborZ,
										  unsigned int size_Mat, 
										  bool evenOrOdd)
{
   int Grid = kArray / numberOfThreads;
   //int Grid = (kQ / numberOfThreads)+1;
   //int Grid1, Grid2;
   //if (Grid>512)
   //{
   //   Grid1 = 512;
   //   Grid2 = (Grid/Grid1)+1;
   //} 
   //else
   //{
   //   Grid1 = 1;
   //   Grid2 = Grid;
   //}
   //dim3 gridQ(Grid1, Grid2);
   dim3 gridQ(Grid, 1, 1);
   dim3 threads(numberOfThreads, 1, 1 );

      QVelDeviceCompZeroPress27<<< gridQ, threads >>> (   nx,
														  ny,
														  vx,
														  vy,
														  vz,
														  DD, 
														  k_Q, 
														  QQ,
														  sizeQ,
														  //kQ, 
														  om1, 
														  neighborX,
														  neighborY,
														  neighborZ,
														  size_Mat, 
														  evenOrOdd);
      getLastCudaError("QVelDeviceComp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVelDevIncompHighNu27(unsigned int numberOfThreads,
									  int nx,
									  int ny,
									  real* vx,
									  real* vy,
									  real* vz,
									  real* DD, 
									  int* k_Q, 
									  real* QQ,
									  unsigned int sizeQ,
									  unsigned int kQ, 
									  real om1, 
									  unsigned int* neighborX,
									  unsigned int* neighborY,
									  unsigned int* neighborZ,
									  unsigned int size_Mat, 
									  bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QVelDeviceIncompHighNu27<<< gridQ, threads >>> (nx,
													  ny,
													  vx,
													  vy,
													  vz,
													  DD, 
													  k_Q, 
													  QQ,
													  sizeQ,
													  kQ, 
													  om1, 
													  neighborX,
													  neighborY,
													  neighborZ,
													  size_Mat, 
													  evenOrOdd);
      getLastCudaError("QVelDeviceIncompHighNu27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVelDevCompHighNu27(  unsigned int numberOfThreads,
									  int nx,
									  int ny,
									  real* vx,
									  real* vy,
									  real* vz,
									  real* DD, 
									  int* k_Q, 
									  real* QQ,
									  unsigned int sizeQ,
									  unsigned int kQ, 
									  real om1, 
									  unsigned int* neighborX,
									  unsigned int* neighborY,
									  unsigned int* neighborZ,
									  unsigned int size_Mat, 
									  bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QVelDeviceCompHighNu27<<< gridQ, threads >>> (  nx,
													  ny,
													  vx,
													  vy,
													  vz,
													  DD, 
													  k_Q, 
													  QQ,
													  sizeQ,
													  kQ, 
													  om1, 
													  neighborX,
													  neighborY,
													  neighborZ,
													  size_Mat, 
													  evenOrOdd);
      getLastCudaError("QVelDeviceComp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QVeloDevEQ27(unsigned int numberOfThreads,
							 real* VeloX,
							 real* VeloY,
							 real* VeloZ,
							 real* DD, 
							 int* k_Q, 
							 int kQ, 
							 real om1, 
							 unsigned int* neighborX,
							 unsigned int* neighborY,
							 unsigned int* neighborZ,
							 unsigned int size_Mat, 
							 bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QVeloDeviceEQ27<<< gridQ, threads >>> (VeloX,
											 VeloY,	
											 VeloZ,	
											 DD, 	
											 k_Q, 		
											 kQ, 		
											 om1, 	
											 neighborX,
											 neighborY,
											 neighborZ,
											 size_Mat, 	
											 evenOrOdd);		
      getLastCudaError("QVeloDeviceEQ27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QSlipDev27(unsigned int numberOfThreads,
                           real* DD, 
                           int* k_Q, 
                           real* QQ,
                           unsigned int sizeQ,
                           real om1, 
                           unsigned int* neighborX,
                           unsigned int* neighborY,
                           unsigned int* neighborZ,
                           unsigned int size_Mat, 
                           bool evenOrOdd)
{
   int Grid = (sizeQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QSlipDevice27<<< gridQ, threads >>> (DD, 
                                           k_Q, 
                                           QQ,
                                           sizeQ,
                                           om1, 
                                           neighborX,
                                           neighborY,
                                           neighborZ,
                                           size_Mat, 
                                           evenOrOdd);
      getLastCudaError("QSlipDevice27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QSlipDevComp27(unsigned int numberOfThreads,
							   real* DD, 
							   int* k_Q, 
							   real* QQ,
							   unsigned int sizeQ,
							   real om1, 
							   unsigned int* neighborX,
							   unsigned int* neighborY,
							   unsigned int* neighborZ,
							   unsigned int size_Mat, 
							   bool evenOrOdd)
{
   int Grid = (sizeQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QSlipDeviceComp27<<< gridQ, threads >>> (DD, 
											   k_Q, 
											   QQ,
											   sizeQ,
											   om1, 
											   neighborX,
											   neighborY,
											   neighborZ,
											   size_Mat, 
											   evenOrOdd);
      getLastCudaError("QSlipDeviceComp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QSlipGeomDevComp27(unsigned int numberOfThreads,
								   real* DD, 
								   int* k_Q, 
								   real* QQ,
								   unsigned int sizeQ,
								   real om1, 
								   real* NormalX,
								   real* NormalY,
								   real* NormalZ,
								   unsigned int* neighborX,
								   unsigned int* neighborY,
								   unsigned int* neighborZ,
								   unsigned int size_Mat, 
								   bool evenOrOdd)
{
   int Grid = (sizeQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QSlipGeomDeviceComp27<<< gridQ, threads >>> (DD, 
												   k_Q, 
												   QQ,
												   sizeQ,
												   om1,
												   NormalX,
												   NormalY,
												   NormalZ,
												   neighborX,
												   neighborY,
												   neighborZ,
												   size_Mat, 
												   evenOrOdd);
      getLastCudaError("QSlipGeomDeviceComp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QSlipNormDevComp27(unsigned int numberOfThreads,
								   real* DD, 
								   int* k_Q, 
								   real* QQ,
								   unsigned int sizeQ,
								   real om1, 
								   real* NormalX,
								   real* NormalY,
								   real* NormalZ,
								   unsigned int* neighborX,
								   unsigned int* neighborY,
								   unsigned int* neighborZ,
								   unsigned int size_Mat, 
								   bool evenOrOdd)
{
   int Grid = (sizeQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QSlipNormDeviceComp27<<< gridQ, threads >>> (DD, 
												   k_Q, 
												   QQ,
												   sizeQ,
												   om1,
												   NormalX,
												   NormalY,
												   NormalZ,
												   neighborX,
												   neighborY,
												   neighborZ,
												   size_Mat, 
												   evenOrOdd);
      getLastCudaError("QSlipGeomDeviceComp27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDev27(unsigned int numberOfThreads,
                             int nx,
                             int ny,
                             real* rhoBC,
                             real* DD, 
                             int* k_Q, 
                             real* QQ,
                             unsigned int sizeQ,
                             unsigned int kQ, 
                             real om1, 
                             unsigned int* neighborX,
                             unsigned int* neighborY,
                             unsigned int* neighborZ,
                             unsigned int size_Mat, 
                             bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressDevice27<<< gridQ, threads >>> (nx,
                                             ny,
                                             rhoBC,
                                             DD, 
                                             k_Q, 
                                             QQ,
                                             sizeQ,
                                             kQ, 
                                             om1, 
                                             neighborX,
                                             neighborY,
                                             neighborZ,
                                             size_Mat, 
                                             evenOrOdd);
      getLastCudaError("QPressDevice27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDevAntiBB27(  unsigned int numberOfThreads,
                                    real* rhoBC,
									real* vx,
									real* vy,
									real* vz,
									real* DD, 
									int* k_Q, 
									real* QQ,
									int kQ, 
									real om1, 
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									unsigned int size_Mat, 
									bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

    QPressDeviceAntiBB27<<< gridQ, threads >>>( rhoBC,
												vx,
												vy,
												vz,
												DD, 
												k_Q, 
												QQ,
												kQ, 
												om1, 
												neighborX,
												neighborY,
												neighborZ,
												size_Mat, 
												evenOrOdd);
    getLastCudaError("QPressDeviceAntiBB27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDevFixBackflow27( unsigned int numberOfThreads,
                                        real* rhoBC,
                                        real* DD, 
                                        int* k_Q, 
                                        unsigned int kQ, 
                                        real om1, 
                                        unsigned int* neighborX,
                                        unsigned int* neighborY,
                                        unsigned int* neighborZ,
                                        unsigned int size_Mat, 
                                        bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressDeviceFixBackflow27<<< gridQ, threads >>> (  rhoBC,
                                                         DD, 
                                                         k_Q, 
                                                         kQ, 
                                                         om1, 
                                                         neighborX,
                                                         neighborY,
                                                         neighborZ,
                                                         size_Mat, 
                                                         evenOrOdd);
      getLastCudaError("QPressDeviceFixBackflow27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDevDirDepBot27(  unsigned int numberOfThreads,
                                       real* rhoBC,
                                       real* DD, 
                                       int* k_Q, 
                                       unsigned int kQ, 
                                       real om1, 
                                       unsigned int* neighborX,
                                       unsigned int* neighborY,
                                       unsigned int* neighborZ,
                                       unsigned int size_Mat, 
                                       bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressDeviceDirDepBot27<<< gridQ, threads >>> ( rhoBC,
                                                      DD, 
                                                      k_Q, 
                                                      kQ, 
                                                      om1, 
                                                      neighborX,
                                                      neighborY,
                                                      neighborZ,
                                                      size_Mat, 
                                                      evenOrOdd);
      getLastCudaError("QPressDeviceDirDepBot27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressNoRhoDev27(unsigned int numberOfThreads,
                                 real* rhoBC,
                                 real* DD, 
                                 int* k_Q, 
                                 int* k_N, 
                                 unsigned int kQ, 
                                 real om1, 
                                 unsigned int* neighborX,
                                 unsigned int* neighborY,
                                 unsigned int* neighborZ,
                                 unsigned int size_Mat, 
                                 bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressNoRhoDevice27<<< gridQ, threads >>> (   rhoBC,
													DD, 
													k_Q, 
													k_N, 
													kQ, 
													om1, 
													neighborX,
													neighborY,
													neighborZ,
													size_Mat, 
													evenOrOdd);
      getLastCudaError("QPressNoRhoDevice27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QInflowScaleByPressDev27(unsigned int numberOfThreads,
										 real* rhoBC,
										 real* DD, 
										 int* k_Q, 
										 int* k_N, 
										 unsigned int kQ, 
										 real om1, 
										 unsigned int* neighborX,
										 unsigned int* neighborY,
										 unsigned int* neighborZ,
										 unsigned int size_Mat, 
										 bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

   QInflowScaleByPressDevice27<<< gridQ, threads >>> (  rhoBC,
														DD, 
														k_Q, 
														k_N, 
														kQ, 
														om1, 
														neighborX,
														neighborY,
														neighborZ,
														size_Mat, 
														evenOrOdd);
   getLastCudaError("QInflowScaleByPressDevice27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDevOld27(  unsigned int numberOfThreads,
                                     real* rhoBC,
                                     real* DD, 
                                     int* k_Q, 
                                     int* k_N, 
                                     unsigned int kQ, 
                                     real om1, 
                                     unsigned int* neighborX,
                                     unsigned int* neighborY,
                                     unsigned int* neighborZ,
                                     unsigned int size_Mat, 
                                     bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressDeviceOld27<<< gridQ, threads >>> ( rhoBC,
                                                DD, 
                                                k_Q, 
                                                k_N, 
                                                kQ, 
                                                om1, 
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                size_Mat, 
                                                evenOrOdd);
      getLastCudaError("QPressDeviceOld27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDevIncompNEQ27(unsigned int numberOfThreads,
                                     real* rhoBC,
                                     real* DD, 
                                     int* k_Q, 
                                     int* k_N, 
                                     unsigned int kQ, 
                                     real om1, 
                                     unsigned int* neighborX,
                                     unsigned int* neighborY,
                                     unsigned int* neighborZ,
                                     unsigned int size_Mat, 
                                     bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressDeviceIncompNEQ27<<< gridQ, threads >>> (   rhoBC,
														DD, 
														k_Q, 
														k_N, 
														kQ, 
														om1, 
														neighborX,
														neighborY,
														neighborZ,
														size_Mat, 
														evenOrOdd);
      getLastCudaError("QPressDeviceIncompNEQ27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDevNEQ27(  unsigned int numberOfThreads,
                                     real* rhoBC,
                                     real* DD, 
                                     int* k_Q, 
                                     int* k_N, 
                                     unsigned int kQ, 
                                     real om1, 
                                     unsigned int* neighborX,
                                     unsigned int* neighborY,
                                     unsigned int* neighborZ,
                                     unsigned int size_Mat, 
                                     bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressDeviceNEQ27<<< gridQ, threads >>> ( rhoBC,
                                                DD, 
                                                k_Q, 
                                                k_N, 
                                                kQ, 
                                                om1, 
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                size_Mat, 
                                                evenOrOdd);
      getLastCudaError("QPressDeviceOld27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDevEQZ27(  unsigned int numberOfThreads,
                                     real* rhoBC,
                                     real* DD, 
                                     int* k_Q, 
                                     int* k_N, 
                                     real* kTestRE, 
                                     unsigned int kQ, 
                                     real om1, 
                                     unsigned int* neighborX,
                                     unsigned int* neighborY,
                                     unsigned int* neighborZ,
                                     unsigned int size_Mat, 
                                     bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressDeviceEQZ27<<< gridQ, threads >>> ( rhoBC,
                                                DD, 
                                                k_Q, 
                                                k_N, 
                                                kTestRE, 
                                                kQ, 
                                                om1, 
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                size_Mat, 
                                                evenOrOdd);
      getLastCudaError("QPressDeviceEQZ27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDevZero27(unsigned int numberOfThreads,
                                real* DD, 
                                int* k_Q, 
                                unsigned int kQ, 
                                unsigned int* neighborX,
                                unsigned int* neighborY,
                                unsigned int* neighborZ,
                                unsigned int size_Mat, 
                                bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressDeviceZero27<<< gridQ, threads >>> (DD, 
                                                k_Q, 
                                                kQ, 
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                size_Mat, 
                                                evenOrOdd);
      getLastCudaError("QPressDeviceOld27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDevFake27(     unsigned int numberOfThreads,
                                     real* rhoBC,
                                     real* DD, 
                                     int* k_Q, 
                                     int* k_N, 
                                     unsigned int kQ, 
                                     real om1, 
                                     unsigned int* neighborX,
                                     unsigned int* neighborY,
                                     unsigned int* neighborZ,
                                     unsigned int size_Mat, 
                                     bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      QPressDeviceFake27<<< gridQ, threads >>> (rhoBC,
                                                DD, 
                                                k_Q, 
                                                k_N, 
                                                kQ, 
                                                om1, 
                                                neighborX,
                                                neighborY,
                                                neighborZ,
                                                size_Mat, 
                                                evenOrOdd);
      getLastCudaError("QPressDeviceFake27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void BBDev27( unsigned int numberOfThreads,
                       int nx,
                       int ny,
                       real* DD, 
                       int* k_Q, 
                       real* QQ,
                       unsigned int sizeQ,
                       unsigned int kQ, 
                       real om1, 
                       unsigned int* neighborX,
                       unsigned int* neighborY,
                       unsigned int* neighborZ,
                       unsigned int size_Mat, 
                       bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      BBDevice27<<< gridQ, threads >>> (  nx,
                                          ny,
                                          DD, 
                                          k_Q, 
                                          QQ,
                                          sizeQ,
                                          kQ, 
                                          om1, 
                                          neighborX,
                                          neighborY,
                                          neighborZ,
                                          size_Mat, 
                                          evenOrOdd);
      getLastCudaError("BBDevice27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void QPressDev27_IntBB(  unsigned int numberOfThreads,
									real* rho,
									real* DD, 
									int* k_Q, 
									real* QQ,
									unsigned int sizeQ,
									unsigned int kQ, 
									real om1, 
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									unsigned int size_Mat, 
									bool evenOrOdd)
{
	int Grid = (kQ / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 gridQ(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

		QPressDevice27_IntBB<<< gridQ, threads >>> (rho,
													DD, 
													k_Q, 
													QQ,
													sizeQ,
													kQ, 
													om1, 
													neighborX,
													neighborY,
													neighborZ,
													size_Mat, 
													evenOrOdd);
		getLastCudaError("QPressDevice27_IntBB execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void PressSchlaffer27(unsigned int numberOfThreads,
                                 real* rhoBC,
                                 real* DD,
                                 real* vx0,
                                 real* vy0,
                                 real* vz0,
                                 real* deltaVz0,
                                 int* k_Q, 
                                 int* k_N, 
                                 int kQ, 
                                 real om1, 
                                 unsigned int* neighborX,
                                 unsigned int* neighborY,
                                 unsigned int* neighborZ,
                                 unsigned int size_Mat, 
                                 bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      PressSchlaff27<<< gridQ, threads >>>(  rhoBC,
                                             DD,
                                             vx0,
                                             vy0,
                                             vz0,
                                             deltaVz0,
                                             k_Q, 
                                             k_N, 
                                             kQ, 
                                             om1, 
                                             neighborX,
                                             neighborY,
                                             neighborZ,
                                             size_Mat, 
                                             evenOrOdd);                                 
      getLastCudaError("PressSchlaff27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void VelSchlaffer27(  unsigned int numberOfThreads,  
                                 int t,
                                 real* DD,
                                 real* vz0,
                                 real* deltaVz0,
                                 int* k_Q, 
                                 int* k_N, 
                                 int kQ, 
                                 real om1, 
                                 unsigned int* neighborX,
                                 unsigned int* neighborY,
                                 unsigned int* neighborZ,
                                 unsigned int size_Mat, 
                                 bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      VelSchlaff27<<< gridQ, threads >>>( t,
                                          DD,
                                          vz0,
                                          deltaVz0,
                                          k_Q, 
                                          k_N, 
                                          kQ, 
                                          om1, 
                                          neighborX,
                                          neighborY,
                                          neighborZ,
                                          size_Mat, 
                                          evenOrOdd);
      getLastCudaError("VelSchlaff27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void PropVelo(   unsigned int numberOfThreads,
                            unsigned int* neighborX,
                            unsigned int* neighborY,
                            unsigned int* neighborZ,
                            real* rho,
                            real* ux,
                            real* uy,
                            real* uz,
                            int* k_Q, 
							unsigned int size_Prop,
                            unsigned int size_Mat,
                            unsigned int* bcMatD,
                            real* DD,
                            bool EvenOrOdd)
{
   int Grid = (size_Prop / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 grid(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      PropellerBC<<< grid, threads >>>(neighborX,
                                       neighborY,
                                       neighborZ,
                                       rho,
                                       ux,
                                       uy,
                                       uz,
									   k_Q,
									   size_Prop,
                                       size_Mat,
									   bcMatD,
                                       DD,
                                       EvenOrOdd);
      getLastCudaError("PropellerBC execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF27( real* DC, 
                        real* DF, 
                        unsigned int* neighborCX,
                        unsigned int* neighborCY,
                        unsigned int* neighborCZ,
                        unsigned int* neighborFX,
                        unsigned int* neighborFY,
                        unsigned int* neighborFZ,
                        unsigned int size_MatC, 
                        unsigned int size_MatF, 
                        bool evenOrOdd,
                        unsigned int* posCSWB, 
                        unsigned int* posFSWB, 
                        unsigned int kCF, 
                        real omCoarse, 
                        real omFine, 
                        real nu, 
                        unsigned int nxC, 
                        unsigned int nyC, 
                        unsigned int nxF, 
                        unsigned int nyF,
                        unsigned int numberOfThreads)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF27<<< gridINT_CF, threads >>> ( DC,  
                                             DF, 
                                             neighborCX,
                                             neighborCY,
                                             neighborCZ,
                                             neighborFX,
                                             neighborFY,
                                             neighborFZ,
                                             size_MatC, 
                                             size_MatF, 
                                             evenOrOdd,
                                             posCSWB, 
                                             posFSWB, 
                                             kCF, 
                                             omCoarse, 
                                             omFine, 
                                             nu, 
                                             nxC, 
                                             nyC, 
                                             nxF, 
                                             nyF);
      getLastCudaError("scaleCF27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCFEff27(real* DC, 
                             real* DF, 
                             unsigned int* neighborCX,
                             unsigned int* neighborCY,
                             unsigned int* neighborCZ,
                             unsigned int* neighborFX,
                             unsigned int* neighborFY,
                             unsigned int* neighborFZ,
                             unsigned int size_MatC, 
                             unsigned int size_MatF, 
                             bool evenOrOdd,
                             unsigned int* posCSWB, 
                             unsigned int* posFSWB, 
                             unsigned int kCF, 
                             real omCoarse, 
                             real omFine, 
                             real nu, 
                             unsigned int nxC, 
                             unsigned int nyC, 
                             unsigned int nxF, 
                             unsigned int nyF,
                             unsigned int numberOfThreads,
                             OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCFEff27<<< gridINT_CF, threads >>> ( DC,  
                                                DF, 
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posCSWB, 
                                                posFSWB, 
                                                kCF, 
                                                omCoarse, 
                                                omFine, 
                                                nu, 
                                                nxC, 
                                                nyC, 
                                                nxF, 
                                                nyF,
                                                offCF);
      getLastCudaError("scaleCFEff27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCFLast27(real* DC, 
                              real* DF, 
                              unsigned int* neighborCX,
                              unsigned int* neighborCY,
                              unsigned int* neighborCZ,
                              unsigned int* neighborFX,
                              unsigned int* neighborFY,
                              unsigned int* neighborFZ,
                              unsigned int size_MatC, 
                              unsigned int size_MatF, 
                              bool evenOrOdd,
                              unsigned int* posCSWB, 
                              unsigned int* posFSWB, 
                              unsigned int kCF, 
                              real omCoarse, 
                              real omFine, 
                              real nu, 
                              unsigned int nxC, 
                              unsigned int nyC, 
                              unsigned int nxF, 
                              unsigned int nyF,
                              unsigned int numberOfThreads,
                              OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCFLast27<<< gridINT_CF, threads >>> (DC,  
                                                DF, 
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posCSWB, 
                                                posFSWB, 
                                                kCF, 
                                                omCoarse, 
                                                omFine, 
                                                nu, 
                                                nxC, 
                                                nyC, 
                                                nxF, 
                                                nyF,
                                                offCF);
      getLastCudaError("scaleCFLast27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCFpress27(  real* DC, 
                                 real* DF, 
                                 unsigned int* neighborCX,
                                 unsigned int* neighborCY,
                                 unsigned int* neighborCZ,
                                 unsigned int* neighborFX,
                                 unsigned int* neighborFY,
                                 unsigned int* neighborFZ,
                                 unsigned int size_MatC, 
                                 unsigned int size_MatF, 
                                 bool evenOrOdd,
                                 unsigned int* posCSWB, 
                                 unsigned int* posFSWB, 
                                 unsigned int kCF, 
                                 real omCoarse, 
                                 real omFine, 
                                 real nu, 
                                 unsigned int nxC, 
                                 unsigned int nyC, 
                                 unsigned int nxF, 
                                 unsigned int nyF,
                                 unsigned int numberOfThreads,
                                 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCFpress27<<< gridINT_CF, threads >>>(DC,  
                                                DF, 
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posCSWB, 
                                                posFSWB, 
                                                kCF, 
                                                omCoarse, 
                                                omFine, 
                                                nu, 
                                                nxC, 
                                                nyC, 
                                                nxF, 
                                                nyF,
                                                offCF);
      getLastCudaError("scaleCFpress27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF_Fix_27(  real* DC, 
                                 real* DF, 
                                 unsigned int* neighborCX,
                                 unsigned int* neighborCY,
                                 unsigned int* neighborCZ,
                                 unsigned int* neighborFX,
                                 unsigned int* neighborFY,
                                 unsigned int* neighborFZ,
                                 unsigned int size_MatC, 
                                 unsigned int size_MatF, 
                                 bool evenOrOdd,
                                 unsigned int* posCSWB, 
                                 unsigned int* posFSWB, 
                                 unsigned int kCF, 
                                 real omCoarse, 
                                 real omFine, 
                                 real nu, 
                                 unsigned int nxC, 
                                 unsigned int nyC, 
                                 unsigned int nxF, 
                                 unsigned int nyF,
                                 unsigned int numberOfThreads,
                                 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF_Fix_27<<< gridINT_CF, threads >>>(DC,  
                                                DF, 
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posCSWB, 
                                                posFSWB, 
                                                kCF, 
                                                omCoarse, 
                                                omFine, 
                                                nu, 
                                                nxC, 
                                                nyC, 
                                                nxF, 
                                                nyF,
                                                offCF);
      getLastCudaError("scaleCF_Fix_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF_Fix_comp_27( real* DC, 
									 real* DF, 
									 unsigned int* neighborCX,
									 unsigned int* neighborCY,
									 unsigned int* neighborCZ,
									 unsigned int* neighborFX,
									 unsigned int* neighborFY,
									 unsigned int* neighborFZ,
									 unsigned int size_MatC, 
									 unsigned int size_MatF, 
									 bool evenOrOdd,
									 unsigned int* posCSWB, 
									 unsigned int* posFSWB, 
									 unsigned int kCF, 
									 real omCoarse, 
									 real omFine, 
									 real nu, 
									 unsigned int nxC, 
									 unsigned int nyC, 
									 unsigned int nxF, 
									 unsigned int nyF,
									 unsigned int numberOfThreads,
									 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF_Fix_comp_27<<< gridINT_CF, threads >>>(   DC,  
														DF, 
														neighborCX,
														neighborCY,
														neighborCZ,
														neighborFX,
														neighborFY,
														neighborFZ,
														size_MatC, 
														size_MatF, 
														evenOrOdd,
														posCSWB, 
														posFSWB, 
														kCF, 
														omCoarse, 
														omFine, 
														nu, 
														nxC, 
														nyC, 
														nxF, 
														nyF,
														offCF);
      getLastCudaError("scaleCF_Fix_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF_0817_comp_27(real* DC, 
									 real* DF, 
									 unsigned int* neighborCX,
									 unsigned int* neighborCY,
									 unsigned int* neighborCZ,
									 unsigned int* neighborFX,
									 unsigned int* neighborFY,
									 unsigned int* neighborFZ,
									 unsigned int size_MatC, 
									 unsigned int size_MatF, 
									 bool evenOrOdd,
									 unsigned int* posCSWB, 
									 unsigned int* posFSWB, 
									 unsigned int kCF, 
									 real omCoarse, 
									 real omFine, 
									 real nu, 
									 unsigned int nxC, 
									 unsigned int nyC, 
									 unsigned int nxF, 
									 unsigned int nyF,
									 unsigned int numberOfThreads,
									 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF_0817_comp_27<<< gridINT_CF, threads >>>(  DC,  
														DF, 
														neighborCX,
														neighborCY,
														neighborCZ,
														neighborFX,
														neighborFY,
														neighborFZ,
														size_MatC, 
														size_MatF, 
														evenOrOdd,
														posCSWB, 
														posFSWB, 
														kCF, 
														omCoarse, 
														omFine, 
														nu, 
														nxC, 
														nyC, 
														nxF, 
														nyF,
														offCF);
      getLastCudaError("scaleCF_0817_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF_comp_D3Q27F3(real* DC,
									 real* DF,
									 real* G6, 
									 unsigned int* neighborCX,
									 unsigned int* neighborCY,
									 unsigned int* neighborCZ,
									 unsigned int* neighborFX,
									 unsigned int* neighborFY,
									 unsigned int* neighborFZ,
									 unsigned int size_MatC, 
									 unsigned int size_MatF, 
									 bool evenOrOdd,
									 unsigned int* posCSWB, 
									 unsigned int* posFSWB, 
									 unsigned int kCF, 
									 real omCoarse, 
									 real omFine, 
									 real nu, 
									 unsigned int nxC, 
									 unsigned int nyC, 
									 unsigned int nxF, 
									 unsigned int nyF,
									 unsigned int numberOfThreads,
									 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF_comp_D3Q27F3 <<< gridINT_CF, threads >>>( DC,
														DF,
														G6,
														neighborCX,
														neighborCY,
														neighborCZ,
														neighborFX,
														neighborFY,
														neighborFZ,
														size_MatC, 
														size_MatF, 
														evenOrOdd,
														posCSWB, 
														posFSWB, 
														kCF, 
														omCoarse, 
														omFine, 
														nu, 
														nxC, 
														nyC, 
														nxF, 
														nyF,
														offCF);
      getLastCudaError("scaleCF_0817_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF_staggered_time_comp_27(  real* DC, 
												 real* DF, 
												 unsigned int* neighborCX,
												 unsigned int* neighborCY,
												 unsigned int* neighborCZ,
												 unsigned int* neighborFX,
												 unsigned int* neighborFY,
												 unsigned int* neighborFZ,
												 unsigned int size_MatC, 
												 unsigned int size_MatF, 
												 bool evenOrOdd,
												 unsigned int* posCSWB, 
												 unsigned int* posFSWB, 
												 unsigned int kCF, 
												 real omCoarse, 
												 real omFine, 
												 real nu, 
												 unsigned int nxC, 
												 unsigned int nyC, 
												 unsigned int nxF, 
												 unsigned int nyF,
												 unsigned int numberOfThreads,
												 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF_staggered_time_comp_27<<< gridINT_CF, threads >>>(    DC,  
																	DF, 
																	neighborCX,
																	neighborCY,
																	neighborCZ,
																	neighborFX,
																	neighborFY,
																	neighborFZ,
																	size_MatC, 
																	size_MatF, 
																	evenOrOdd,
																	posCSWB, 
																	posFSWB, 
																	kCF, 
																	omCoarse, 
																	omFine, 
																	nu, 
																	nxC, 
																	nyC, 
																	nxF, 
																	nyF,
																	offCF);
      getLastCudaError("scaleCF_Fix_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF_RhoSq_comp_27(   real* DC, 
										 real* DF, 
										 unsigned int* neighborCX,
										 unsigned int* neighborCY,
										 unsigned int* neighborCZ,
										 unsigned int* neighborFX,
										 unsigned int* neighborFY,
										 unsigned int* neighborFZ,
										 unsigned int size_MatC, 
										 unsigned int size_MatF, 
										 bool evenOrOdd,
										 unsigned int* posCSWB, 
										 unsigned int* posFSWB, 
										 unsigned int kCF, 
										 real omCoarse, 
										 real omFine, 
										 real nu, 
										 unsigned int nxC, 
										 unsigned int nyC, 
										 unsigned int nxF, 
										 unsigned int nyF,
										 unsigned int numberOfThreads,
										 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF_RhoSq_comp_27<<< gridINT_CF, threads >>>( DC,  
														DF, 
														neighborCX,
														neighborCY,
														neighborCZ,
														neighborFX,
														neighborFY,
														neighborFZ,
														size_MatC, 
														size_MatF, 
														evenOrOdd,
														posCSWB, 
														posFSWB, 
														kCF, 
														omCoarse, 
														omFine, 
														nu, 
														nxC, 
														nyC, 
														nxF, 
														nyF,
														offCF);
      getLastCudaError("scaleCF_RhoSq_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF_RhoSq_3rdMom_comp_27(real* DC, 
											 real* DF, 
											 unsigned int* neighborCX,
											 unsigned int* neighborCY,
											 unsigned int* neighborCZ,
											 unsigned int* neighborFX,
											 unsigned int* neighborFY,
											 unsigned int* neighborFZ,
											 unsigned int size_MatC, 
											 unsigned int size_MatF, 
											 bool evenOrOdd,
											 unsigned int* posCSWB, 
											 unsigned int* posFSWB, 
											 unsigned int kCF, 
											 real omCoarse, 
											 real omFine, 
											 real nu, 
											 unsigned int nxC, 
											 unsigned int nyC, 
											 unsigned int nxF, 
											 unsigned int nyF,
											 unsigned int numberOfThreads,
											 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF_RhoSq_3rdMom_comp_27<<< gridINT_CF, threads >>>(  DC,  
																DF, 
																neighborCX,
																neighborCY,
																neighborCZ,
																neighborFX,
																neighborFY,
																neighborFZ,
																size_MatC, 
																size_MatF, 
																evenOrOdd,
																posCSWB, 
																posFSWB, 
																kCF, 
																omCoarse, 
																omFine, 
																nu, 
																nxC, 
																nyC, 
																nxF, 
																nyF,
																offCF);
      getLastCudaError("scaleCF_RhoSq_3rdMom_comp_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF_AA2016_comp_27(real* DC, 
									   real* DF, 
									   unsigned int* neighborCX,
									   unsigned int* neighborCY,
									   unsigned int* neighborCZ,
									   unsigned int* neighborFX,
									   unsigned int* neighborFY,
									   unsigned int* neighborFZ,
									   unsigned int size_MatC, 
									   unsigned int size_MatF, 
									   bool evenOrOdd,
									   unsigned int* posCSWB, 
									   unsigned int* posFSWB, 
									   unsigned int kCF, 
									   real omCoarse, 
									   real omFine, 
									   real nu, 
									   unsigned int nxC, 
									   unsigned int nyC, 
									   unsigned int nxF, 
									   unsigned int nyF,
									   unsigned int numberOfThreads,
									   OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF_AA2016_comp_27<<< gridINT_CF, threads >>>(DC,  
														DF, 
														neighborCX,
														neighborCY,
														neighborCZ,
														neighborFX,
														neighborFY,
														neighborFZ,
														size_MatC, 
														size_MatF, 
														evenOrOdd,
														posCSWB, 
														posFSWB, 
														kCF, 
														omCoarse, 
														omFine, 
														nu, 
														nxC, 
														nyC, 
														nxF, 
														nyF,
														offCF);
      getLastCudaError("scaleCF_AA2016_comp_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCF_NSPress_27(  real* DC, 
									 real* DF, 
									 unsigned int* neighborCX,
									 unsigned int* neighborCY,
									 unsigned int* neighborCZ,
									 unsigned int* neighborFX,
									 unsigned int* neighborFY,
									 unsigned int* neighborFZ,
									 unsigned int size_MatC, 
									 unsigned int size_MatF, 
									 bool evenOrOdd,
									 unsigned int* posCSWB, 
									 unsigned int* posFSWB, 
									 unsigned int kCF, 
									 real omCoarse, 
									 real omFine, 
									 real nu, 
									 unsigned int nxC, 
									 unsigned int nyC, 
									 unsigned int nxF, 
									 unsigned int nyF,
									 unsigned int numberOfThreads,
									 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCF_NSPress_27<<< gridINT_CF, threads >>>(DC,  
													DF, 
													neighborCX,
													neighborCY,
													neighborCZ,
													neighborFX,
													neighborFY,
													neighborFZ,
													size_MatC, 
													size_MatF, 
													evenOrOdd,
													posCSWB, 
													posFSWB, 
													kCF, 
													omCoarse, 
													omFine, 
													nu, 
													nxC, 
													nyC, 
													nxF, 
													nyF,
													offCF);
      getLastCudaError("scaleCF_Fix_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCFThSMG7(   real* DC, 
                                 real* DF,
                                 real* DD7C, 
                                 real* DD7F,
                                 unsigned int* neighborCX,
                                 unsigned int* neighborCY,
                                 unsigned int* neighborCZ,
                                 unsigned int* neighborFX,
                                 unsigned int* neighborFY,
                                 unsigned int* neighborFZ,
                                 unsigned int size_MatC, 
                                 unsigned int size_MatF, 
                                 bool evenOrOdd,
                                 unsigned int* posCSWB, 
                                 unsigned int* posFSWB, 
                                 unsigned int kCF, 
                                 real nu,
                                 real diffusivity_fine,
                                 unsigned int numberOfThreads,
                                 OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCFThSMG7<<< gridINT_CF, threads >>> (DC,  
                                                DF,
                                                DD7C,
                                                DD7F,
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posCSWB, 
                                                posFSWB, 
                                                kCF, 
                                                nu,
                                                diffusivity_fine,
                                                offCF);
      getLastCudaError("scaleCFThSMG7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCFThS7(  real* DC, 
                              real* DF,
                              real* DD7C, 
                              real* DD7F,
                              unsigned int* neighborCX,
                              unsigned int* neighborCY,
                              unsigned int* neighborCZ,
                              unsigned int* neighborFX,
                              unsigned int* neighborFY,
                              unsigned int* neighborFZ,
                              unsigned int size_MatC, 
                              unsigned int size_MatF, 
                              bool evenOrOdd,
                              unsigned int* posCSWB, 
                              unsigned int* posFSWB, 
                              unsigned int kCF, 
                              real nu,
                              real diffusivity_fine,
                              unsigned int numberOfThreads)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCFThS7<<< gridINT_CF, threads >>> (  DC,  
                                                DF,
                                                DD7C,
                                                DD7F,
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posCSWB, 
                                                posFSWB, 
                                                kCF, 
                                                nu,
                                                diffusivity_fine);
      getLastCudaError("scaleCFThS7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleCFThS27( real* DC, 
                              real* DF,
                              real* DD27C, 
                              real* DD27F,
                              unsigned int* neighborCX,
                              unsigned int* neighborCY,
                              unsigned int* neighborCZ,
                              unsigned int* neighborFX,
                              unsigned int* neighborFY,
                              unsigned int* neighborFZ,
                              unsigned int size_MatC, 
                              unsigned int size_MatF, 
                              bool evenOrOdd,
                              unsigned int* posCSWB, 
                              unsigned int* posFSWB, 
                              unsigned int kCF, 
                              real nu,
                              real diffusivity_fine,
                              unsigned int numberOfThreads,
							  OffCF offCF)
{
   int Grid = (kCF / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_CF(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleCFThS27<<< gridINT_CF, threads >>> ( DC,  
                                                DF,
                                                DD27C,
                                                DD27F,
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posCSWB, 
                                                posFSWB, 
                                                kCF, 
                                                nu,
                                                diffusivity_fine,
										        offCF);
      getLastCudaError("scaleCFThS27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC27( real* DC, 
                           real* DF, 
                           unsigned int* neighborCX,
                           unsigned int* neighborCY,
                           unsigned int* neighborCZ,
                           unsigned int* neighborFX,
                           unsigned int* neighborFY,
                           unsigned int* neighborFZ,
                           unsigned int size_MatC, 
                           unsigned int size_MatF, 
                           bool evenOrOdd,
                           unsigned int* posC, 
                           unsigned int* posFSWB, 
                           unsigned int kFC, 
                           real omCoarse, 
                           real omFine, 
                           real nu, 
                           unsigned int nxC, 
                           unsigned int nyC, 
                           unsigned int nxF, 
                           unsigned int nyF,
                           unsigned int numberOfThreads)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFC27<<< gridINT_FC, threads >>> ( DC, 
                                             DF, 
                                             neighborCX,
                                             neighborCY,
                                             neighborCZ,
                                             neighborFX,
                                             neighborFY,
                                             neighborFZ,
                                             size_MatC, 
                                             size_MatF, 
                                             evenOrOdd,
                                             posC, 
                                             posFSWB, 
                                             kFC, 
                                             omCoarse, 
                                             omFine, 
                                             nu, 
                                             nxC, 
                                             nyC, 
                                             nxF, 
                                             nyF);
      getLastCudaError("scaleFC27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFCEff27(real* DC, 
                             real* DF, 
                             unsigned int* neighborCX,
                             unsigned int* neighborCY,
                             unsigned int* neighborCZ,
                             unsigned int* neighborFX,
                             unsigned int* neighborFY,
                             unsigned int* neighborFZ,
                             unsigned int size_MatC, 
                             unsigned int size_MatF, 
                             bool evenOrOdd,
                             unsigned int* posC, 
                             unsigned int* posFSWB, 
                             unsigned int kFC, 
                             real omCoarse, 
                             real omFine, 
                             real nu, 
                             unsigned int nxC, 
                             unsigned int nyC, 
                             unsigned int nxF, 
                             unsigned int nyF,
                             unsigned int numberOfThreads,
                             OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFCEff27<<< gridINT_FC, threads >>> ( DC, 
                                                DF, 
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posC, 
                                                posFSWB, 
                                                kFC, 
                                                omCoarse, 
                                                omFine, 
                                                nu, 
                                                nxC, 
                                                nyC, 
                                                nxF, 
                                                nyF,
                                                offFC);
      getLastCudaError("scaleFCEff27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFCLast27(real* DC, 
                              real* DF, 
                              unsigned int* neighborCX,
                              unsigned int* neighborCY,
                              unsigned int* neighborCZ,
                              unsigned int* neighborFX,
                              unsigned int* neighborFY,
                              unsigned int* neighborFZ,
                              unsigned int size_MatC, 
                              unsigned int size_MatF, 
                              bool evenOrOdd,
                              unsigned int* posC, 
                              unsigned int* posFSWB, 
                              unsigned int kFC, 
                              real omCoarse, 
                              real omFine, 
                              real nu, 
                              unsigned int nxC, 
                              unsigned int nyC, 
                              unsigned int nxF, 
                              unsigned int nyF,
                              unsigned int numberOfThreads,
                              OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFCLast27<<< gridINT_FC, threads >>> (DC, 
                                                DF, 
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posC, 
                                                posFSWB, 
                                                kFC, 
                                                omCoarse, 
                                                omFine, 
                                                nu, 
                                                nxC, 
                                                nyC, 
                                                nxF, 
                                                nyF,
                                                offFC);
      getLastCudaError("Kernel execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFCpress27(real* DC, 
                              real* DF, 
                              unsigned int* neighborCX,
                              unsigned int* neighborCY,
                              unsigned int* neighborCZ,
                              unsigned int* neighborFX,
                              unsigned int* neighborFY,
                              unsigned int* neighborFZ,
                              unsigned int size_MatC, 
                              unsigned int size_MatF, 
                              bool evenOrOdd,
                              unsigned int* posC, 
                              unsigned int* posFSWB, 
                              unsigned int kFC, 
                              real omCoarse, 
                              real omFine, 
                              real nu, 
                              unsigned int nxC, 
                              unsigned int nyC, 
                              unsigned int nxF, 
                              unsigned int nyF,
                              unsigned int numberOfThreads,
                              OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFCpress27<<< gridINT_FC, threads >>> (  DC, 
                                                   DF, 
                                                   neighborCX,
                                                   neighborCY,
                                                   neighborCZ,
                                                   neighborFX,
                                                   neighborFY,
                                                   neighborFZ,
                                                   size_MatC, 
                                                   size_MatF, 
                                                   evenOrOdd,
                                                   posC, 
                                                   posFSWB, 
                                                   kFC, 
                                                   omCoarse, 
                                                   omFine, 
                                                   nu, 
                                                   nxC, 
                                                   nyC, 
                                                   nxF, 
                                                   nyF,
                                                   offFC);
      getLastCudaError("scaleFCpress27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC_Fix_27(real* DC, 
                              real* DF, 
                              unsigned int* neighborCX,
                              unsigned int* neighborCY,
                              unsigned int* neighborCZ,
                              unsigned int* neighborFX,
                              unsigned int* neighborFY,
                              unsigned int* neighborFZ,
                              unsigned int size_MatC, 
                              unsigned int size_MatF, 
                              bool evenOrOdd,
                              unsigned int* posC, 
                              unsigned int* posFSWB, 
                              unsigned int kFC, 
                              real omCoarse, 
                              real omFine, 
                              real nu, 
                              unsigned int nxC, 
                              unsigned int nyC, 
                              unsigned int nxF, 
                              unsigned int nyF,
                              unsigned int numberOfThreads,
                              OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFC_Fix_27<<< gridINT_FC, threads >>> (  DC, 
                                                   DF, 
                                                   neighborCX,
                                                   neighborCY,
                                                   neighborCZ,
                                                   neighborFX,
                                                   neighborFY,
                                                   neighborFZ,
                                                   size_MatC, 
                                                   size_MatF, 
                                                   evenOrOdd,
                                                   posC, 
                                                   posFSWB, 
                                                   kFC, 
                                                   omCoarse, 
                                                   omFine, 
                                                   nu, 
                                                   nxC, 
                                                   nyC, 
                                                   nxF, 
                                                   nyF,
                                                   offFC);
      getLastCudaError("scaleFC_Fix_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC_Fix_comp_27(  real* DC, 
									  real* DF, 
									  unsigned int* neighborCX,
									  unsigned int* neighborCY,
									  unsigned int* neighborCZ,
									  unsigned int* neighborFX,
									  unsigned int* neighborFY,
									  unsigned int* neighborFZ,
									  unsigned int size_MatC, 
									  unsigned int size_MatF, 
									  bool evenOrOdd,
									  unsigned int* posC, 
									  unsigned int* posFSWB, 
									  unsigned int kFC, 
									  real omCoarse, 
									  real omFine, 
									  real nu, 
									  unsigned int nxC, 
									  unsigned int nyC, 
									  unsigned int nxF, 
									  unsigned int nyF,
									  unsigned int numberOfThreads,
									  OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFC_Fix_comp_27<<< gridINT_FC, threads >>> ( DC, 
													   DF, 
													   neighborCX,
													   neighborCY,
													   neighborCZ,
													   neighborFX,
													   neighborFY,
													   neighborFZ,
													   size_MatC, 
													   size_MatF, 
													   evenOrOdd,
													   posC, 
													   posFSWB, 
													   kFC, 
													   omCoarse, 
													   omFine, 
													   nu, 
													   nxC, 
													   nyC, 
													   nxF, 
													   nyF,
													   offFC);
      getLastCudaError("scaleFC_Fix_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC_0817_comp_27( real* DC,
									  real* DF, 
									  unsigned int* neighborCX,
									  unsigned int* neighborCY,
									  unsigned int* neighborCZ,
									  unsigned int* neighborFX,
									  unsigned int* neighborFY,
									  unsigned int* neighborFZ,
									  unsigned int size_MatC, 
									  unsigned int size_MatF, 
									  bool evenOrOdd,
									  unsigned int* posC, 
									  unsigned int* posFSWB, 
									  unsigned int kFC, 
									  real omCoarse, 
									  real omFine, 
									  real nu, 
									  unsigned int nxC, 
									  unsigned int nyC, 
									  unsigned int nxF, 
									  unsigned int nyF,
									  unsigned int numberOfThreads,
									  OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFC_0817_comp_27<<< gridINT_FC, threads >>> (DC, 
													   DF, 
													   neighborCX,
													   neighborCY,
													   neighborCZ,
													   neighborFX,
													   neighborFY,
													   neighborFZ,
													   size_MatC, 
													   size_MatF, 
													   evenOrOdd,
													   posC, 
													   posFSWB, 
													   kFC, 
													   omCoarse, 
													   omFine, 
													   nu, 
													   nxC, 
													   nyC, 
													   nxF, 
													   nyF,
													   offFC);
      getLastCudaError("scaleFC_0817_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC_comp_D3Q27F3( real* DC,
									  real* DF,
									  real* G6,
									  unsigned int* neighborCX,
									  unsigned int* neighborCY,
									  unsigned int* neighborCZ,
									  unsigned int* neighborFX,
									  unsigned int* neighborFY,
									  unsigned int* neighborFZ,
									  unsigned int size_MatC, 
									  unsigned int size_MatF, 
									  bool evenOrOdd,
									  unsigned int* posC, 
									  unsigned int* posFSWB, 
									  unsigned int kFC, 
									  real omCoarse, 
									  real omFine, 
									  real nu, 
									  unsigned int nxC, 
									  unsigned int nyC, 
									  unsigned int nxF, 
									  unsigned int nyF,
									  unsigned int numberOfThreads,
									  OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

     scaleFC_comp_D3Q27F3 <<< gridINT_FC, threads >>> (DC,
													   DF,
													   G6,
													   neighborCX,
													   neighborCY,
													   neighborCZ,
													   neighborFX,
													   neighborFY,
													   neighborFZ,
													   size_MatC, 
													   size_MatF, 
													   evenOrOdd,
													   posC, 
													   posFSWB, 
													   kFC, 
													   omCoarse, 
													   omFine, 
													   nu, 
													   nxC, 
													   nyC, 
													   nxF, 
													   nyF,
													   offFC);
      getLastCudaError("scaleFC_0817_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC_staggered_time_comp_27(   real* DC, 
												  real* DF, 
												  unsigned int* neighborCX,
												  unsigned int* neighborCY,
												  unsigned int* neighborCZ,
												  unsigned int* neighborFX,
												  unsigned int* neighborFY,
												  unsigned int* neighborFZ,
												  unsigned int size_MatC, 
												  unsigned int size_MatF, 
												  bool evenOrOdd,
												  unsigned int* posC, 
												  unsigned int* posFSWB, 
												  unsigned int kFC, 
												  real omCoarse, 
												  real omFine, 
												  real nu, 
												  unsigned int nxC, 
												  unsigned int nyC, 
												  unsigned int nxF, 
												  unsigned int nyF,
												  unsigned int numberOfThreads,
												  OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFC_staggered_time_comp_27<<< gridINT_FC, threads >>> (  DC, 
																   DF, 
																   neighborCX,
																   neighborCY,
																   neighborCZ,
																   neighborFX,
																   neighborFY,
																   neighborFZ,
																   size_MatC, 
																   size_MatF, 
																   evenOrOdd,
																   posC, 
																   posFSWB, 
																   kFC, 
																   omCoarse, 
																   omFine, 
																   nu, 
																   nxC, 
																   nyC, 
																   nxF, 
																   nyF,
																   offFC);
      getLastCudaError("scaleFC_Fix_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC_RhoSq_comp_27(real* DC, 
									  real* DF, 
									  unsigned int* neighborCX,
									  unsigned int* neighborCY,
									  unsigned int* neighborCZ,
									  unsigned int* neighborFX,
									  unsigned int* neighborFY,
									  unsigned int* neighborFZ,
									  unsigned int size_MatC, 
									  unsigned int size_MatF, 
									  bool evenOrOdd,
									  unsigned int* posC, 
									  unsigned int* posFSWB, 
									  unsigned int kFC, 
									  real omCoarse, 
									  real omFine, 
									  real nu, 
									  unsigned int nxC, 
									  unsigned int nyC, 
									  unsigned int nxF, 
									  unsigned int nyF,
									  unsigned int numberOfThreads,
									  OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFC_RhoSq_comp_27<<< gridINT_FC, threads >>>(DC, 
													   DF, 
													   neighborCX,
													   neighborCY,
													   neighborCZ,
													   neighborFX,
													   neighborFY,
													   neighborFZ,
													   size_MatC, 
													   size_MatF, 
													   evenOrOdd,
													   posC, 
													   posFSWB, 
													   kFC, 
													   omCoarse, 
													   omFine, 
													   nu, 
													   nxC, 
													   nyC, 
													   nxF, 
													   nyF,
													   offFC);
      getLastCudaError("scaleFC_RhoSq_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC_RhoSq_3rdMom_comp_27( real* DC, 
											  real* DF, 
											  unsigned int* neighborCX,
											  unsigned int* neighborCY,
											  unsigned int* neighborCZ,
											  unsigned int* neighborFX,
											  unsigned int* neighborFY,
											  unsigned int* neighborFZ,
											  unsigned int size_MatC, 
											  unsigned int size_MatF, 
											  bool evenOrOdd,
											  unsigned int* posC, 
											  unsigned int* posFSWB, 
											  unsigned int kFC, 
											  real omCoarse, 
											  real omFine, 
											  real nu, 
											  unsigned int nxC, 
											  unsigned int nyC, 
											  unsigned int nxF, 
											  unsigned int nyF,
											  unsigned int numberOfThreads,
											  OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFC_RhoSq_3rdMom_comp_27<<< gridINT_FC, threads >>>(DC, 
															  DF, 
															  neighborCX,
															  neighborCY,
															  neighborCZ,
															  neighborFX,
															  neighborFY,
															  neighborFZ,
															  size_MatC, 
															  size_MatF, 
															  evenOrOdd,
															  posC, 
															  posFSWB, 
															  kFC, 
															  omCoarse, 
															  omFine, 
															  nu, 
															  nxC, 
															  nyC, 
															  nxF, 
															  nyF,
															  offFC);
      getLastCudaError("scaleFC_RhoSq_3rdMom_comp_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC_AA2016_comp_27( real* DC, 
										real* DF, 
										unsigned int* neighborCX,
										unsigned int* neighborCY,
										unsigned int* neighborCZ,
										unsigned int* neighborFX,
										unsigned int* neighborFY,
										unsigned int* neighborFZ,
										unsigned int size_MatC, 
										unsigned int size_MatF, 
										bool evenOrOdd,
										unsigned int* posC, 
										unsigned int* posFSWB, 
										unsigned int kFC, 
										real omCoarse, 
										real omFine, 
										real nu, 
										unsigned int nxC, 
										unsigned int nyC, 
										unsigned int nxF, 
										unsigned int nyF,
										unsigned int numberOfThreads,
										OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFC_AA2016_comp_27<<< gridINT_FC, threads >>>(DC, 
														DF, 
														neighborCX,
														neighborCY,
														neighborCZ,
														neighborFX,
														neighborFY,
														neighborFZ,
														size_MatC, 
														size_MatF, 
														evenOrOdd,
														posC, 
														posFSWB, 
														kFC, 
														omCoarse, 
														omFine, 
														nu, 
														nxC, 
														nyC, 
														nxF, 
														nyF,
														offFC);
      getLastCudaError("scaleFC_AA2016_comp_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFC_NSPress_27(real* DC, 
								  real* DF, 
								  unsigned int* neighborCX,
								  unsigned int* neighborCY,
								  unsigned int* neighborCZ,
								  unsigned int* neighborFX,
								  unsigned int* neighborFY,
								  unsigned int* neighborFZ,
								  unsigned int size_MatC, 
								  unsigned int size_MatF, 
								  bool evenOrOdd,
								  unsigned int* posC, 
								  unsigned int* posFSWB, 
								  unsigned int kFC, 
								  real omCoarse, 
								  real omFine, 
								  real nu, 
								  unsigned int nxC, 
								  unsigned int nyC, 
								  unsigned int nxF, 
								  unsigned int nyF,
								  unsigned int numberOfThreads,
								  OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFC_NSPress_27<<< gridINT_FC, threads >>> (  DC, 
													   DF, 
													   neighborCX,
													   neighborCY,
													   neighborCZ,
													   neighborFX,
													   neighborFY,
													   neighborFZ,
													   size_MatC, 
													   size_MatF, 
													   evenOrOdd,
													   posC, 
													   posFSWB, 
													   kFC, 
													   omCoarse, 
													   omFine, 
													   nu, 
													   nxC, 
													   nyC, 
													   nxF, 
													   nyF,
													   offFC);
      getLastCudaError("scaleFC_Fix_27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFCThSMG7(real* DC, 
                              real* DF,
                              real* DD7C, 
                              real* DD7F,
                              unsigned int* neighborCX,
                              unsigned int* neighborCY,
                              unsigned int* neighborCZ,
                              unsigned int* neighborFX,
                              unsigned int* neighborFY,
                              unsigned int* neighborFZ,
                              unsigned int size_MatC, 
                              unsigned int size_MatF, 
                              bool evenOrOdd,
                              unsigned int* posC, 
                              unsigned int* posFSWB, 
                              unsigned int kFC, 
                              real nu,
                              real diffusivity_coarse,
                              unsigned int numberOfThreads,
                              OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFCThSMG7<<< gridINT_FC, threads >>>( DC, 
                                                DF,
                                                DD7C, 
                                                DD7F,
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posC, 
                                                posFSWB, 
                                                kFC, 
                                                nu,
                                                diffusivity_coarse,
                                                offFC);
      getLastCudaError("scaleFCThSMG7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFCThS7(  real* DC, 
                              real* DF,
                              real* DD7C, 
                              real* DD7F,
                              unsigned int* neighborCX,
                              unsigned int* neighborCY,
                              unsigned int* neighborCZ,
                              unsigned int* neighborFX,
                              unsigned int* neighborFY,
                              unsigned int* neighborFZ,
                              unsigned int size_MatC, 
                              unsigned int size_MatF, 
                              bool evenOrOdd,
                              unsigned int* posC, 
                              unsigned int* posFSWB, 
                              unsigned int kFC, 
                              real nu,
                              real diffusivity_coarse,
                              unsigned int numberOfThreads)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFCThS7<<< gridINT_FC, threads >>>(DC, 
                                             DF,
                                             DD7C, 
                                             DD7F,
                                             neighborCX,
                                             neighborCY,
                                             neighborCZ,
                                             neighborFX,
                                             neighborFY,
                                             neighborFZ,
                                             size_MatC, 
                                             size_MatF, 
                                             evenOrOdd,
                                             posC, 
                                             posFSWB, 
                                             kFC, 
                                             nu,
                                             diffusivity_coarse);
      getLastCudaError("scaleFCThS7 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void ScaleFCThS27( real* DC, 
                              real* DF,
                              real* DD27C, 
                              real* DD27F,
                              unsigned int* neighborCX,
                              unsigned int* neighborCY,
                              unsigned int* neighborCZ,
                              unsigned int* neighborFX,
                              unsigned int* neighborFY,
                              unsigned int* neighborFZ,
                              unsigned int size_MatC, 
                              unsigned int size_MatF, 
                              bool evenOrOdd,
                              unsigned int* posC, 
                              unsigned int* posFSWB, 
                              unsigned int kFC, 
                              real nu,
                              real diffusivity_coarse,
                              unsigned int numberOfThreads,
							  OffFC offFC)
{
   int Grid = (kFC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridINT_FC(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      scaleFCThS27<<< gridINT_FC, threads >>>(  DC, 
                                                DF,
                                                DD27C, 
                                                DD27F,
                                                neighborCX,
                                                neighborCY,
                                                neighborCZ,
                                                neighborFX,
                                                neighborFY,
                                                neighborFZ,
                                                size_MatC, 
                                                size_MatF, 
                                                evenOrOdd,
                                                posC, 
                                                posFSWB, 
                                                kFC, 
                                                nu,
                                                diffusivity_coarse,
												offFC);
      getLastCudaError("scaleFCThS27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void DragLiftPostD27(real* DD, 
								int* k_Q, 
								real* QQ,
								int kQ, 
								double *DragX,
								double *DragY,
								double *DragZ,
								unsigned int* neighborX,
								unsigned int* neighborY,
								unsigned int* neighborZ,
								unsigned int size_Mat, 
								bool evenOrOdd,
								unsigned int numberOfThreads)
{
	int Grid = (kQ / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	DragLiftPost27<<< grid, threads >>>(DD, 
										k_Q, 
										QQ,
										kQ, 
										DragX,
										DragY,
										DragZ,
										neighborX,
										neighborY,
										neighborZ,
										size_Mat, 
										evenOrOdd);
	getLastCudaError("DragLift27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void DragLiftPreD27( real* DD, 
								int* k_Q, 
								real* QQ,
								int kQ, 
								double *DragX,
								double *DragY,
								double *DragZ,
								unsigned int* neighborX,
								unsigned int* neighborY,
								unsigned int* neighborZ,
								unsigned int size_Mat, 
								bool evenOrOdd,
								unsigned int numberOfThreads)
{
	int Grid = (kQ / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	DragLiftPre27<<< grid, threads >>>( DD, 
										k_Q, 
										QQ,
										kQ, 
										DragX,
										DragY,
										DragZ,
										neighborX,
										neighborY,
										neighborZ,
										size_Mat, 
										evenOrOdd);
	getLastCudaError("DragLift27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcCPtop27(real* DD, 
							int* cpIndex, 
							int nonCp, 
							double *cpPress,
							unsigned int* neighborX,
							unsigned int* neighborY,
							unsigned int* neighborZ,
							unsigned int size_Mat, 
							bool evenOrOdd,
							unsigned int numberOfThreads)
{
	int Grid = (nonCp / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	CalcCP27<<< grid, threads >>>(DD, 
								  cpIndex, 
								  nonCp, 
								  cpPress,
								  neighborX,
								  neighborY,
								  neighborZ,
								  size_Mat, 
								  evenOrOdd);
	getLastCudaError("CalcCP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void CalcCPbottom27( real* DD, 
								int* cpIndex, 
								int nonCp, 
								double *cpPress,
								unsigned int* neighborX,
								unsigned int* neighborY,
								unsigned int* neighborZ,
								unsigned int size_Mat, 
								bool evenOrOdd,
								unsigned int numberOfThreads)
{
	int Grid = (nonCp / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	CalcCP27<<< grid, threads >>>(DD, 
								  cpIndex, 
								  nonCp, 
								  cpPress,
								  neighborX,
								  neighborY,
								  neighborZ,
								  size_Mat, 
								  evenOrOdd);
	getLastCudaError("CalcCP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void GetSendFsPreDev27(real* DD,
								  real* bufferFs,
								  int* sendIndex,
								  int buffmax,
								  unsigned int* neighborX,
								  unsigned int* neighborY,
								  unsigned int* neighborZ,
								  unsigned int size_Mat, 
								  bool evenOrOdd,
								  unsigned int numberOfThreads)
{
	int Grid = (buffmax / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	getSendFsPre27<<< grid, threads >>>(DD, 
										bufferFs, 
										sendIndex, 
										buffmax,
										neighborX,
										neighborY,
										neighborZ,
										size_Mat, 
										evenOrOdd);
	getLastCudaError("getSendFsPre27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void GetSendFsPostDev27(real* DD,
								   real* bufferFs,
								   int* sendIndex,
								   int buffmax,
								   unsigned int* neighborX,
								   unsigned int* neighborY,
								   unsigned int* neighborZ,
								   unsigned int size_Mat, 
								   bool evenOrOdd,
								   unsigned int numberOfThreads)
{
	int Grid = (buffmax / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	getSendFsPost27<<< grid, threads >>>(DD, 
										 bufferFs, 
										 sendIndex, 
										 buffmax,
										 neighborX,
										 neighborY,
										 neighborZ,
										 size_Mat, 
										 evenOrOdd);
	getLastCudaError("getSendFsPost27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void SetRecvFsPreDev27(real* DD,
								  real* bufferFs,
								  int* recvIndex,
								  int buffmax,
								  unsigned int* neighborX,
								  unsigned int* neighborY,
								  unsigned int* neighborZ,
								  unsigned int size_Mat, 
								  bool evenOrOdd,
								  unsigned int numberOfThreads)
{
	int Grid = (buffmax / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	setRecvFsPre27<<< grid, threads >>>(DD, 
										bufferFs, 
										recvIndex, 
										buffmax,
										neighborX,
										neighborY,
										neighborZ,
										size_Mat, 
										evenOrOdd);
	getLastCudaError("setRecvFsPre27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void SetRecvFsPostDev27(real* DD,
								   real* bufferFs,
								   int* recvIndex,
								   int buffmax,
								   unsigned int* neighborX,
								   unsigned int* neighborY,
								   unsigned int* neighborZ,
								   unsigned int size_Mat, 
								   bool evenOrOdd,
								   unsigned int numberOfThreads)
{
	int Grid = (buffmax / numberOfThreads)+1;
	int Grid1, Grid2;
	if (Grid>512)
	{
		Grid1 = 512;
		Grid2 = (Grid/Grid1)+1;
	} 
	else
	{
		Grid1 = 1;
		Grid2 = Grid;
	}
	dim3 grid(Grid1, Grid2);
	dim3 threads(numberOfThreads, 1, 1 );

	setRecvFsPost27<<< grid, threads >>>(DD, 
										 bufferFs, 
										 recvIndex, 
										 buffmax,
										 neighborX,
										 neighborY,
										 neighborZ,
										 size_Mat, 
										 evenOrOdd);
	getLastCudaError("setRecvFsPost27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void WallFuncDev27(unsigned int numberOfThreads,
							  int nx,
							  int ny,
							  real* vx,
							  real* vy,
							  real* vz,
							  real* DD, 
							  int* k_Q, 
							  real* QQ,
							  unsigned int sizeQ,
							  unsigned int kQ, 
							  real om1, 
							  unsigned int* neighborX,
							  unsigned int* neighborY,
							  unsigned int* neighborZ,
							  unsigned int size_Mat, 
							  bool evenOrOdd)
{
   int Grid = (kQ / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      WallFunction27<<< gridQ, threads >>> (  nx,
											  ny,
											  vx,
											  vy,
											  vz,
											  DD, 
											  k_Q, 
											  QQ,
											  sizeQ,
											  kQ, 
											  om1, 
											  neighborX,
											  neighborY,
											  neighborZ,
											  size_Mat, 
											  evenOrOdd);
      getLastCudaError("WallFunction27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void SetOutputWallVelocitySP27(unsigned int numberOfThreads,
										  real* vxD,
										  real* vyD,
										  real* vzD,
										  real* vxWall,
										  real* vyWall,
										  real* vzWall,
										  int numberOfWallNodes, 
										  int* kWallNodes, 
										  real* rhoD,
										  real* pressD,
										  unsigned int* geoD,
										  unsigned int* neighborX,
										  unsigned int* neighborY,
										  unsigned int* neighborZ,
										  unsigned int size_Mat,
										  real* DD,
										  bool evenOrOdd)
{
   int Grid = (numberOfWallNodes / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      LBSetOutputWallVelocitySP27<<< gridQ, threads >>> (	vxD,
															vyD,
															vzD,
															vxWall,
															vyWall,
															vzWall,
															numberOfWallNodes, 
															kWallNodes, 
															rhoD,
															pressD,
															geoD,
															neighborX,
															neighborY,
															neighborZ,
															size_Mat,
															DD,
															evenOrOdd);
      getLastCudaError("LBSetOutputWallVelocitySP27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void GetVelotoForce27(unsigned int numberOfThreads,
								 real* DD, 
								 int* bcIndex, 
								 int nonAtBC, 
								 real* Vx,
								 real* Vy,
								 real* Vz,
								 unsigned int* neighborX,
								 unsigned int* neighborY,
								 unsigned int* neighborZ,
								 unsigned int size_Mat, 
								 bool evenOrOdd)
{
   int Grid = (nonAtBC / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

      GetVeloforForcing27<<< gridQ, threads >>> (DD,
												bcIndex,
												nonAtBC,
												Vx,
												Vy,
												Vz,
												neighborX,
												neighborY,
												neighborZ,
												size_Mat,
												evenOrOdd);
      getLastCudaError("GetVeloforForcing27 execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void InitParticlesDevice(real* coordX,
									real* coordY,
									real* coordZ, 
									real* coordParticleXlocal,
									real* coordParticleYlocal,
									real* coordParticleZlocal, 
									real* coordParticleXglobal,
									real* coordParticleYglobal,
									real* coordParticleZglobal,
									real* veloParticleX,
									real* veloParticleY,
									real* veloParticleZ,
									real* randArray,
									unsigned int* particleID,
									unsigned int* cellBaseID,
									unsigned int* bcMatD,
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									unsigned int* neighborWSB,
							        int level,
									unsigned int numberOfParticles, 
									unsigned int size_Mat,
									unsigned int numberOfThreads)
{
   int Grid = (numberOfParticles / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

   InitParticles<<< gridQ, threads >>> (coordX,
										coordY,
										coordZ, 
										coordParticleXlocal,
										coordParticleYlocal,
										coordParticleZlocal, 
										coordParticleXglobal,
										coordParticleYglobal,
										coordParticleZglobal,
										veloParticleX,
										veloParticleY,
										veloParticleZ,
										randArray,
										particleID,
										cellBaseID,
										bcMatD,
										neighborX,
										neighborY,
										neighborZ,
										neighborWSB,
										level,
										numberOfParticles,
										size_Mat);
      getLastCudaError("InitParticles execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void MoveParticlesDevice(real* coordX,
									real* coordY,
									real* coordZ, 
									real* coordParticleXlocal,
									real* coordParticleYlocal,
									real* coordParticleZlocal, 
									real* coordParticleXglobal,
									real* coordParticleYglobal,
									real* coordParticleZglobal,
									real* veloParticleX,
									real* veloParticleY,
									real* veloParticleZ,
									real* DD,
									real  omega,
									unsigned int* particleID,
									unsigned int* cellBaseID,
									unsigned int* bcMatD,
									unsigned int* neighborX,
									unsigned int* neighborY,
									unsigned int* neighborZ,
									unsigned int* neighborWSB,
							        int level,
									unsigned int timestep, 
									unsigned int numberOfTimesteps, 
									unsigned int numberOfParticles, 
									unsigned int size_Mat,
									unsigned int numberOfThreads,
									bool evenOrOdd)
{
   int Grid = (numberOfParticles / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

   MoveParticles<<< gridQ, threads >>> (coordX,
										coordY,
										coordZ, 
										coordParticleXlocal,
										coordParticleYlocal,
										coordParticleZlocal, 
										coordParticleXglobal,
										coordParticleYglobal,
										coordParticleZglobal,
										veloParticleX,
										veloParticleY,
										veloParticleZ,
										DD,
										omega,
										particleID,
										cellBaseID,
										bcMatD,
										neighborX,
										neighborY,
										neighborZ,
										neighborWSB,
										level,
										timestep,
										numberOfTimesteps,
										numberOfParticles,
										size_Mat,
										evenOrOdd);
      getLastCudaError("MoveParticles execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void initRandomDevice(curandState* state,
								 unsigned int size_Mat,
								 unsigned int numberOfThreads)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

   initRandom<<< gridQ, threads >>> (state);
   getLastCudaError("initRandom execution failed"); 
}
//////////////////////////////////////////////////////////////////////////
extern "C" void generateRandomValuesDevice( curandState* state,
											unsigned int size_Mat,
											real* randArray,
											unsigned int numberOfThreads)
{
   int Grid = (size_Mat / numberOfThreads)+1;
   int Grid1, Grid2;
   if (Grid>512)
   {
      Grid1 = 512;
      Grid2 = (Grid/Grid1)+1;
   } 
   else
   {
      Grid1 = 1;
      Grid2 = Grid;
   }
   dim3 gridQ(Grid1, Grid2);
   dim3 threads(numberOfThreads, 1, 1 );

   generateRandomValues<<< gridQ, threads >>> (state,randArray);
   getLastCudaError("generateRandomValues execution failed"); 
}













