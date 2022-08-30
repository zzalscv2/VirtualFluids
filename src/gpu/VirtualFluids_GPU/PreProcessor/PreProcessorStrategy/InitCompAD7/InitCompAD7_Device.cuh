#ifndef LB_INIT_COMP_AD_7_H
#define LB_INIT_COMP_AD_7_H

#include <DataTypes.h>
#include <curand.h>

__global__ void LB_Init_Comp_AD_7(unsigned int* neighborX,
	unsigned int* neighborY,
	unsigned int* neighborZ,
	unsigned int* geoD,
	real* Conc,
	real* ux,
	real* uy,
	real* uz,
	unsigned int size_Mat,
	real* DD7,
	bool EvenOrOdd);

#endif