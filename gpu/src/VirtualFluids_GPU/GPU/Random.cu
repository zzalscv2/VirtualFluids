/* Device code */
#include "LBM/LB.h" 
#include "LBM/D3Q27.h"
#include "Core/RealConstants.h"
//random numbers
#include <curand.h>
#include <curand_kernel.h>


//////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void initRandom(curandState* state)
{
   ////////////////////////////////////////////////////////////////////////////////
   const unsigned  x = threadIdx.x;  // Globaler x-Index 
   const unsigned  y = blockIdx.x;   // Globaler y-Index 
   const unsigned  z = blockIdx.y;   // Globaler z-Index 

   const unsigned nx = blockDim.x;
   const unsigned ny = gridDim.x;

   const unsigned k = nx*(ny*z + y) + x;
   //////////////////////////////////////////////////////////////////////////

   curand_init(k, k, 0, &state[k]);

   //////////////////////////////////////////////////////////////////////////	
}
//////////////////////////////////////////////////////////////////////////////





//////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void generateRandomValues(curandState* state, real* randArray)
{
   ////////////////////////////////////////////////////////////////////////////////
   const unsigned  x = threadIdx.x;  // Globaler x-Index 
   const unsigned  y = blockIdx.x;   // Globaler y-Index 
   const unsigned  z = blockIdx.y;   // Globaler z-Index 

   const unsigned nx = blockDim.x;
   const unsigned ny = gridDim.x;

   const unsigned k = nx*(ny*z + y) + x;
   //////////////////////////////////////////////////////////////////////////

   randArray[k] = (real)curand_uniform(&state[k]);

   //////////////////////////////////////////////////////////////////////////
}
//////////////////////////////////////////////////////////////////////////////


