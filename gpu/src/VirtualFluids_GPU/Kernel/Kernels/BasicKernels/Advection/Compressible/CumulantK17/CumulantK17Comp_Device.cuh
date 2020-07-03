#ifndef LB_Kernel_CUMULANT_K17_COMP_H
#define LB_Kernel_CUMULANT_K17_COMP_H

#include <DataTypes.h>
#include <curand.h>

extern "C" __global__ void LB_Kernel_CumulantK17Comp(	real omega,
														unsigned int* bcMatD,
														unsigned int* neighborX,
														unsigned int* neighborY,
														unsigned int* neighborZ,
														real* DDStart,
														int size_Mat,
														int level,
														real* forces,
                                                        real* quadricLimiters,
														bool EvenOrOdd);
#endif