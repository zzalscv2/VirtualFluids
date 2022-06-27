#include "TurbulentViscosity.h"
#include "Core/DataTypes.h"
#include "lbm/constants/NumericConstants.h"
#include "Parameter/Parameter.h"
#include "cuda/CudaGrid.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "LBM/LB.h"

using namespace vf::lbm::constant;

__host__ __device__ __forceinline__ real calcDamping(real kappa, real xPos, real x0, real x1)
{
    real x = max((xPos-x0)/(x1-x0), 0.f);
    return kappa*x*x*(3-2*x); // polynomial with f(0)=0, f'(0) = 0, f(1) = 1, f'(1)=0
}

__host__ __device__ __forceinline__ void calcDerivatives(const uint& k, uint& kM, uint& kP, uint* typeOfGridNode, real* vx, real* vy, real* vz, real& dvx, real& dvy, real& dvz)
{
    bool fluidP = (typeOfGridNode[kP] == GEO_FLUID);
    bool fluidM = (typeOfGridNode[kM] == GEO_FLUID);
    real div = (fluidM & fluidP) ? c1o2 : c1o1;

    dvx = ((fluidP ? vx[kP] : vx[k])-(fluidM ? vx[kM] : vx[k]))*div;
    dvy = ((fluidP ? vy[kP] : vy[k])-(fluidM ? vy[kM] : vy[k]))*div;
    dvz = ((fluidP ? vz[kP] : vz[k])-(fluidM ? vz[kM] : vz[k]))*div;
}

__global__ void calcAMD(real* vx,
                        real* vy,
                        real* vz,
                        real* turbulentViscosity,
                        real viscosity,
                        uint* neighborX,
                        uint* neighborY,
                        uint* neighborZ,
                        uint* neighborWSB,
                        real* coordX,
                        uint* typeOfGridNode,
                        uint size_Mat,
                        real SGSConstant)
{

    const uint x = threadIdx.x; 
    const uint y = blockIdx.x; 
    const uint z = blockIdx.y; 

    const uint nx = blockDim.x;
    const uint ny = gridDim.x;

    const uint k = nx*(ny*z + y) + x;
    if(k >= size_Mat) return;
    if(typeOfGridNode[k] != GEO_FLUID) return;

    uint kPx = neighborX[k];
    uint kPy = neighborY[k];
    uint kPz = neighborZ[k];
    uint kMxyz = neighborWSB[k];
    uint kMx = neighborZ[neighborY[kMxyz]];
    uint kMy = neighborZ[neighborX[kMxyz]];
    uint kMz = neighborY[neighborX[kMxyz]];

    real dvxdx, dvxdy, dvxdz,
         dvydx, dvydy, dvydz,
         dvzdx, dvzdy, dvzdz;

    calcDerivatives(k, kMx, kPx, typeOfGridNode, vx, vy, vz, dvxdx, dvydx, dvzdx);
    calcDerivatives(k, kMy, kPy, typeOfGridNode, vx, vy, vz, dvxdy, dvydy, dvzdy);
    calcDerivatives(k, kMz, kPz, typeOfGridNode, vx, vy, vz, dvxdz, dvydz, dvzdz);

    real denominator =  dvxdx*dvxdx + dvydx*dvydx + dvzdx*dvzdx + 
                        dvxdy*dvxdy + dvydy*dvydy + dvzdy*dvzdy +
                        dvxdz*dvxdz + dvydz*dvydz + dvzdz*dvzdz;
    real enumerator =   (dvxdx*dvxdx + dvxdy*dvxdy + dvxdz*dvxdz) * dvxdx + 
                        (dvydx*dvydx + dvydy*dvydy + dvydz*dvydz) * dvydy + 
                        (dvzdx*dvzdx + dvzdy*dvzdy + dvzdz*dvzdz) * dvzdz +
                        (dvxdx*dvydx + dvxdy*dvydy + dvxdz*dvydz) * (dvxdy+dvydx) +
                        (dvxdx*dvzdx + dvxdy*dvzdy + dvxdz*dvzdz) * (dvxdz+dvzdx) + 
                        (dvydx*dvzdx + dvydy*dvzdy + dvydz*dvzdz) * (dvydz+dvzdy);

    const real kappa = 10000.f; // multiplier of the viscosity 
    const real x0 = 5500.f; // start of damping layer
    const real x1 = 6000.f; // total length of domain
    real nuSGS = max(c0o1,-SGSConstant*enumerator)/denominator;
    real xPos = coordX[k];
    real nuDamping = calcDamping(kappa, xPos, x0, x1)*viscosity;

    real nu = nuSGS + nuDamping;
    // if(k >= 800600 && k <= 800637) printf("k %d x %f nu %f nu SGS %f nu damping %f \n ", k, xPos, nu, nuSGS, nuDamping);
    turbulentViscosity[k] = nu;
}


void calcTurbulentViscosityAMD(Parameter* para, int level)
{
    vf::cuda::CudaGrid grid = vf::cuda::CudaGrid(para->getParH(level)->numberofthreads, para->getParH(level)->numberOfNodes);
    calcAMD<<<grid.grid, grid.threads>>>(
        para->getParD(level)->velocityX,
        para->getParD(level)->velocityY,
        para->getParD(level)->velocityZ,
        para->getParD(level)->turbViscosity,
        para->getViscosity(),
        para->getParD(level)->neighborX,
        para->getParD(level)->neighborY,
        para->getParD(level)->neighborZ,
        para->getParD(level)->neighborInverse,
        para->getParD(level)->coordinateX,
        para->getParD(level)->typeOfGridNode,
        para->getParD(level)->numberOfNodes,
        para->getSGSConstant()
    );
    getLastCudaError("calcAMD execution failed");
}
    