#include "LBM/LB.h" 
#include <lbm/constants/NumericConstants.h>
#include <lbm/constants/D3Q27.h>
#include <lbm/MacroscopicQuantities.h>

#include "VirtualFluids_GPU/Kernel/Utilities/DistributionHelper.cuh"
#include "VirtualFluids_GPU/GPU/KernelUtilities.h"

using namespace vf::lbm::constant;
using namespace vf::lbm::dir;

__global__ void QPrecursorDeviceCompZeroPress( 	int* subgridDistanceIndices,
                                                int numberOfBCnodes,
                                                int numberOfPrecursorNodes,
                                                int sizeQ,
                                                real omega,
                                                real* distributions,
                                                real* subgridDistances,
                                                uint* neighborX, 
                                                uint* neighborY, 
                                                uint* neighborZ,
                                                uint* neighborsNT, 
                                                uint* neighborsNB,
                                                uint* neighborsST,
                                                uint* neighborsSB,
                                                real* weightsNT, 
                                                real* weightsNB,
                                                real* weightsST,
                                                real* weightsSB,
                                                real* vLast, 
                                                real* vCurrent,
                                                real velocityX,
                                                real velocityY,
                                                real velocityZ,
                                                real tRatio,
                                                real velocityRatio,
                                                unsigned long long numberOfLBnodes,
                                                bool isEvenTimestep)
{
    const unsigned k = vf::gpu::getNodeIndex();

    if(k>=numberOfBCnodes) return;

    ////////////////////////////////////////////////////////////////////////////////
    // interpolation of velocity
    real vxLastInterpd, vyLastInterpd, vzLastInterpd; 
    real vxNextInterpd, vyNextInterpd, vzNextInterpd; 

    uint kNT = neighborsNT[k];
    real dNT = weightsNT[k];

    real* vxLast = vLast;
    real* vyLast = &vLast[numberOfPrecursorNodes];
    real* vzLast = &vLast[2*numberOfPrecursorNodes];

    real* vxCurrent = vCurrent;
    real* vyCurrent = &vCurrent[numberOfPrecursorNodes];
    real* vzCurrent = &vCurrent[2*numberOfPrecursorNodes];

    if(dNT < 1e6)
    {
        uint kNB = neighborsNB[k];
        uint kST = neighborsST[k];
        uint kSB = neighborsSB[k];

        real dNB = weightsNB[k];
        real dST = weightsST[k];
        real dSB = weightsSB[k];

        real invWeightSum = 1.f/(dNT+dNB+dST+dSB);

        vxLastInterpd = (vxLast[kNT]*dNT + vxLast[kNB]*dNB + vxLast[kST]*dST + vxLast[kSB]*dSB)*invWeightSum;
        vyLastInterpd = (vyLast[kNT]*dNT + vyLast[kNB]*dNB + vyLast[kST]*dST + vyLast[kSB]*dSB)*invWeightSum;
        vzLastInterpd = (vzLast[kNT]*dNT + vzLast[kNB]*dNB + vzLast[kST]*dST + vzLast[kSB]*dSB)*invWeightSum;

        vxNextInterpd = (vxCurrent[kNT]*dNT + vxCurrent[kNB]*dNB + vxCurrent[kST]*dST + vxCurrent[kSB]*dSB)*invWeightSum;
        vyNextInterpd = (vyCurrent[kNT]*dNT + vyCurrent[kNB]*dNB + vyCurrent[kST]*dST + vyCurrent[kSB]*dSB)*invWeightSum;
        vzNextInterpd = (vzCurrent[kNT]*dNT + vzCurrent[kNB]*dNB + vzCurrent[kST]*dST + vzCurrent[kSB]*dSB)*invWeightSum;
    }
    else
    {
        vxLastInterpd = vxLast[kNT];
        vyLastInterpd = vyLast[kNT];
        vzLastInterpd = vzLast[kNT];

        vxNextInterpd = vxCurrent[kNT];
        vyNextInterpd = vyCurrent[kNT];
        vzNextInterpd = vzCurrent[kNT];
    }

    // if(k==16300)s printf("%f %f %f\n", vxLastInterpd, vyLastInterpd, vzLastInterpd);
    real VeloX = (velocityX + (1.f-tRatio)*vxLastInterpd + tRatio*vxNextInterpd)/velocityRatio;
    real VeloY = (velocityY + (1.f-tRatio)*vyLastInterpd + tRatio*vyNextInterpd)/velocityRatio; 
    real VeloZ = (velocityZ + (1.f-tRatio)*vzLastInterpd + tRatio*vzNextInterpd)/velocityRatio;
    // From here on just a copy of QVelDeviceCompZeroPress
    ////////////////////////////////////////////////////////////////////////////////

    Distributions27 dist;
    getPointersToDistributions(dist, distributions, numberOfLBnodes, isEvenTimestep);

    unsigned int KQK  = subgridDistanceIndices[k];
    unsigned int kzero= KQK;
    unsigned int ke   = KQK;
    unsigned int kw   = neighborX[KQK];
    unsigned int kn   = KQK;
    unsigned int ks   = neighborY[KQK];
    unsigned int kt   = KQK;
    unsigned int kb   = neighborZ[KQK];
    unsigned int ksw  = neighborY[kw];
    unsigned int kne  = KQK;
    unsigned int kse  = ks;
    unsigned int knw  = kw;
    unsigned int kbw  = neighborZ[kw];
    unsigned int kte  = KQK;
    unsigned int kbe  = kb;
    unsigned int ktw  = kw;
    unsigned int kbs  = neighborZ[ks];
    unsigned int ktn  = KQK;
    unsigned int kbn  = kb;
    unsigned int kts  = ks;
    unsigned int ktse = ks;
    unsigned int kbnw = kbw;
    unsigned int ktnw = kw;
    unsigned int kbse = kbs;
    unsigned int ktsw = ksw;
    unsigned int kbne = kb;
    unsigned int ktne = KQK;
    unsigned int kbsw = neighborZ[ksw];

    ////////////////////////////////////////////////////////////////////////////////
    //! - Set local distributions
    //!
    real f_W    = (dist.f[DIR_P00   ])[ke   ];
    real f_E    = (dist.f[DIR_M00   ])[kw   ];
    real f_S    = (dist.f[DIR_0P0   ])[kn   ];
    real f_N    = (dist.f[DIR_0M0   ])[ks   ];
    real f_B    = (dist.f[DIR_00P   ])[kt   ];
    real f_T    = (dist.f[DIR_00M   ])[kb   ];
    real f_SW   = (dist.f[DIR_PP0  ])[kne  ];
    real f_NE   = (dist.f[DIR_MM0  ])[ksw  ];
    real f_NW   = (dist.f[DIR_PM0  ])[kse  ];
    real f_SE   = (dist.f[DIR_MP0  ])[knw  ];
    real f_BW   = (dist.f[DIR_P0P  ])[kte  ];
    real f_TE   = (dist.f[DIR_M0M  ])[kbw  ];
    real f_TW   = (dist.f[DIR_P0M  ])[kbe  ];
    real f_BE   = (dist.f[DIR_M0P  ])[ktw  ];
    real f_BS   = (dist.f[DIR_0PP  ])[ktn  ];
    real f_TN   = (dist.f[DIR_0MM  ])[kbs  ];
    real f_TS   = (dist.f[DIR_0PM  ])[kbn  ];
    real f_BN   = (dist.f[DIR_0MP  ])[kts  ];
    real f_BSW  = (dist.f[DIR_PPP ])[ktne ];
    real f_BNE  = (dist.f[DIR_MMP ])[ktsw ];
    real f_BNW  = (dist.f[DIR_PMP ])[ktse ];
    real f_BSE  = (dist.f[DIR_MPP ])[ktnw ];
    real f_TSW  = (dist.f[DIR_PPM ])[kbne ];
    real f_TNE  = (dist.f[DIR_MMM ])[kbsw ];
    real f_TNW  = (dist.f[DIR_PMM ])[kbse ];
    real f_TSE  = (dist.f[DIR_MPM ])[kbnw ];
    
    SubgridDistances27 subgridD;
    getPointersToSubgridDistances(subgridD, subgridDistances, numberOfBCnodes);
    
    ////////////////////////////////////////////////////////////////////////////////
      real drho   =  f_TSE + f_TNW + f_TNE + f_TSW + f_BSE + f_BNW + f_BNE + f_BSW +
                     f_BN + f_TS + f_TN + f_BS + f_BE + f_TW + f_TE + f_BW + f_SE + f_NW + f_NE + f_SW + 
                     f_T + f_B + f_N + f_S + f_E + f_W + ((dist.f[DIR_000])[kzero]); 

      real vx1    =  (((f_TSE - f_BNW) - (f_TNW - f_BSE)) + ((f_TNE - f_BSW) - (f_TSW - f_BNE)) +
                      ((f_BE - f_TW)   + (f_TE - f_BW))   + ((f_SE - f_NW)   + (f_NE - f_SW)) +
                      (f_E - f_W)) / (c1o1 + drho); 
         

      real vx2    =   ((-(f_TSE - f_BNW) + (f_TNW - f_BSE)) + ((f_TNE - f_BSW) - (f_TSW - f_BNE)) +
                       ((f_BN - f_TS)   + (f_TN - f_BS))    + (-(f_SE - f_NW)  + (f_NE - f_SW)) +
                       (f_N - f_S)) / (c1o1 + drho); 

      real vx3    =   (((f_TSE - f_BNW) + (f_TNW - f_BSE)) + ((f_TNE - f_BSW) + (f_TSW - f_BNE)) +
                       (-(f_BN - f_TS)  + (f_TN - f_BS))   + ((f_TE - f_BW)   - (f_BE - f_TW)) +
                       (f_T - f_B)) / (c1o1 + drho); 

    
    // if(k==16383 || k==0) printf("k %d kQ %d drho = %f u %f v %f w %f\n",k, KQK, drho, vx1, vx2, vx3);
      real cu_sq=c3o2*(vx1*vx1+vx2*vx2+vx3*vx3) * (c1o1 + drho);
    //////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////////////
    //! - Update distributions with subgrid distance (q) between zero and one
    real feq, q, velocityLB, velocityBC;
    q = (subgridD.q[DIR_P00])[k];
    if (q>=c0o1 && q<=c1o1) // only update distribution for q between zero and one
    {
        velocityLB = vx1;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c2o27);
        velocityBC = VeloX;
        (dist.f[DIR_M00])[kw] = getInterpolatedDistributionForVeloWithPressureBC(q, f_E, f_W, feq, omega, drho, velocityBC, c2o27);
    }

    q = (subgridD.q[DIR_M00])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx1;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c2o27);
        velocityBC = -VeloX;
        (dist.f[DIR_P00])[ke] = getInterpolatedDistributionForVeloWithPressureBC(q, f_W, f_E, feq, omega, drho, velocityBC, c2o27);
    }

    q = (subgridD.q[DIR_0P0])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx2;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c2o27);
        velocityBC = VeloY;
        (dist.f[DIR_0M0])[DIR_0M0] = getInterpolatedDistributionForVeloWithPressureBC(q, f_N, f_S, feq, omega, drho, velocityBC, c2o27);
    }

    q = (subgridD.q[DIR_0M0])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx2;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c2o27);
        velocityBC = -VeloY;
        (dist.f[DIR_0P0])[kn] = getInterpolatedDistributionForVeloWithPressureBC(q, f_S, f_N, feq, omega, drho, velocityBC, c2o27);
    }

    q = (subgridD.q[DIR_00P])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c2o27);
        velocityBC = VeloZ;
        (dist.f[DIR_00M])[kb] = getInterpolatedDistributionForVeloWithPressureBC(q, f_T, f_B, feq, omega, drho, velocityBC, c2o27);
    }

    q = (subgridD.q[DIR_00M])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c2o27);
        velocityBC = -VeloZ;
        (dist.f[DIR_00P])[kt] = getInterpolatedDistributionForVeloWithPressureBC(q, f_B, f_T, feq, omega, drho, velocityBC, c2o27);
    }

    q = (subgridD.q[DIR_PP0])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx1 + vx2;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = VeloX + VeloY;
        (dist.f[DIR_MM0])[ksw] = getInterpolatedDistributionForVeloWithPressureBC(q, f_NE, f_SW, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_MM0])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx1 - vx2;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = -VeloX - VeloY;
        (dist.f[DIR_PP0])[kne] = getInterpolatedDistributionForVeloWithPressureBC(q, f_SW, f_NE, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_PM0])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx1 - vx2;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = VeloX - VeloY;
        (dist.f[DIR_MP0])[knw] = getInterpolatedDistributionForVeloWithPressureBC(q, f_SE, f_NW, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_MP0])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx1 + vx2;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = -VeloX + VeloY;
        (dist.f[DIR_PM0])[kse] = getInterpolatedDistributionForVeloWithPressureBC(q, f_NW, f_SE, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_P0P])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx1 + vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = VeloX + VeloZ;
        (dist.f[DIR_M0M])[kbw] = getInterpolatedDistributionForVeloWithPressureBC(q, f_TE, f_BW, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_M0M])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx1 - vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = -VeloX - VeloZ;
        (dist.f[DIR_P0P])[kte] = getInterpolatedDistributionForVeloWithPressureBC(q, f_BW, f_TE, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_P0M])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx1 - vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = VeloX - VeloZ;
        (dist.f[DIR_M0P])[ktw] = getInterpolatedDistributionForVeloWithPressureBC(q, f_BE, f_TW, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_M0P])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx1 + vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = -VeloX + VeloZ;
        (dist.f[DIR_P0M])[kbe] = getInterpolatedDistributionForVeloWithPressureBC(q, f_TW, f_BE, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_0PP])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx2 + vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = VeloY + VeloZ;
        (dist.f[DIR_0MM])[kbs] = getInterpolatedDistributionForVeloWithPressureBC(q, f_TN, f_BS, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_0MM])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx2 - vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = -VeloY - VeloZ;
        (dist.f[DIR_0PP])[ktn] = getInterpolatedDistributionForVeloWithPressureBC(q, f_BS, f_TN, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_0PM])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx2 - vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = VeloY - VeloZ;
        (dist.f[DIR_0MP])[kts] = getInterpolatedDistributionForVeloWithPressureBC(q, f_BN, f_TS, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_0MP])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx2 + vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o54);
        velocityBC = -VeloY + VeloZ;
        (dist.f[DIR_0PM])[kbn] = getInterpolatedDistributionForVeloWithPressureBC(q, f_TS, f_BN, feq, omega, drho, velocityBC, c1o54);
    }

    q = (subgridD.q[DIR_PPP])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx1 + vx2 + vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o216);
        velocityBC = VeloX + VeloY + VeloZ;
        (dist.f[DIR_MMM])[kbsw] = getInterpolatedDistributionForVeloWithPressureBC(q, f_TNE, f_BSW, feq, omega, drho, velocityBC, c1o216);
    }

    q = (subgridD.q[DIR_MMM])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx1 - vx2 - vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o216);
        velocityBC = -VeloX - VeloY - VeloZ;
        (dist.f[DIR_PPP])[ktne] = getInterpolatedDistributionForVeloWithPressureBC(q, f_BSW, f_TNE, feq, omega, drho, velocityBC, c1o216);
    }

    q = (subgridD.q[DIR_PPM])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx1 + vx2 - vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o216);
        velocityBC = VeloX + VeloY - VeloZ;
        (dist.f[DIR_MMP])[ktsw] = getInterpolatedDistributionForVeloWithPressureBC(q, f_BNE, f_TSW, feq, omega, drho, velocityBC, c1o216);
    }

    q = (subgridD.q[DIR_MMP])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx1 - vx2 + vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o216);
        velocityBC = -VeloX - VeloY + VeloZ;
        (dist.f[DIR_PPM])[kbne] = getInterpolatedDistributionForVeloWithPressureBC(q, f_TSW, f_BNE, feq, omega, drho, velocityBC, c1o216);
    }

    q = (subgridD.q[DIR_PMP])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx1 - vx2 + vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o216);
        velocityBC = VeloX - VeloY + VeloZ;
        (dist.f[DIR_MPM])[kbnw] = getInterpolatedDistributionForVeloWithPressureBC(q, f_TSE, f_BNW, feq, omega, drho, velocityBC, c1o216);
    }

    q = (subgridD.q[DIR_MPM])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx1 + vx2 - vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o216);
        velocityBC = -VeloX + VeloY - VeloZ;
        (dist.f[DIR_PMP])[ktse] = getInterpolatedDistributionForVeloWithPressureBC(q, f_BNW, f_TSE, feq, omega, drho, velocityBC, c1o216);
    }

    q = (subgridD.q[DIR_PMM])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = vx1 - vx2 - vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o216);
        velocityBC = VeloX - VeloY - VeloZ;
        (dist.f[DIR_MPP])[ktnw] = getInterpolatedDistributionForVeloWithPressureBC(q, f_BSE, f_TNW, feq, omega, drho, velocityBC, c1o216);
    }

    q = (subgridD.q[DIR_MPP])[k];
    if (q>=c0o1 && q<=c1o1)
    {
        velocityLB = -vx1 + vx2 + vx3;
        feq = getEquilibriumForBC(drho, velocityLB, cu_sq, c1o216);
        velocityBC = -VeloX + VeloY + VeloZ;
        (dist.f[DIR_PMM])[kbse] = getInterpolatedDistributionForVeloWithPressureBC(q, f_TNW, f_BSE, feq, omega, drho, velocityBC, c1o216);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void PrecursorDeviceEQ27( 	int* subgridDistanceIndices,
                                        int numberOfBCnodes,
                                        int numberOfPrecursorNodes,
                                        real omega,
                                        real* distributions,
                                        uint* neighborX, 
                                        uint* neighborY, 
                                        uint* neighborZ,
                                        uint* neighborsNT, 
                                        uint* neighborsNB,
                                        uint* neighborsST,
                                        uint* neighborsSB,
                                        real* weightsNT, 
                                        real* weightsNB,
                                        real* weightsST,
                                        real* weightsSB,
                                        real* vLast, 
                                        real* vCurrent,
                                        real velocityX,
                                        real velocityY,
                                        real velocityZ,
                                        real tRatio,
                                        real velocityRatio,
                                        unsigned long long numberOfLBnodes,
                                        bool isEvenTimestep)
{
    const unsigned k = vf::gpu::getNodeIndex();

    if(k>=numberOfBCnodes) return;

    ////////////////////////////////////////////////////////////////////////////////
    // interpolation of velocity
    real vxLastInterpd, vyLastInterpd, vzLastInterpd; 
    real vxNextInterpd, vyNextInterpd, vzNextInterpd; 

    uint kNT = neighborsNT[k];
    real dNT = weightsNT[k];

    real* vxLast = vLast;
    real* vyLast = &vLast[numberOfPrecursorNodes];
    real* vzLast = &vLast[2*numberOfPrecursorNodes];

    real* vxCurrent = vCurrent;
    real* vyCurrent = &vCurrent[numberOfPrecursorNodes];
    real* vzCurrent = &vCurrent[2*numberOfPrecursorNodes];

    if(dNT < 1e6)
    {
        uint kNB = neighborsNB[k];
        uint kST = neighborsST[k];
        uint kSB = neighborsSB[k];

        real dNB = weightsNB[k];
        real dST = weightsST[k];
        real dSB = weightsSB[k];

        real invWeightSum = 1.f/(dNT+dNB+dST+dSB);

        vxLastInterpd = (vxLast[kNT]*dNT + vxLast[kNB]*dNB + vxLast[kST]*dST + vxLast[kSB]*dSB)*invWeightSum;
        vyLastInterpd = (vyLast[kNT]*dNT + vyLast[kNB]*dNB + vyLast[kST]*dST + vyLast[kSB]*dSB)*invWeightSum;
        vzLastInterpd = (vzLast[kNT]*dNT + vzLast[kNB]*dNB + vzLast[kST]*dST + vzLast[kSB]*dSB)*invWeightSum;

        vxNextInterpd = (vxCurrent[kNT]*dNT + vxCurrent[kNB]*dNB + vxCurrent[kST]*dST + vxCurrent[kSB]*dSB)*invWeightSum;
        vyNextInterpd = (vyCurrent[kNT]*dNT + vyCurrent[kNB]*dNB + vyCurrent[kST]*dST + vyCurrent[kSB]*dSB)*invWeightSum;
        vzNextInterpd = (vzCurrent[kNT]*dNT + vzCurrent[kNB]*dNB + vzCurrent[kST]*dST + vzCurrent[kSB]*dSB)*invWeightSum;
    }
    else
    {
        vxLastInterpd = vxLast[kNT];
        vyLastInterpd = vyLast[kNT];
        vzLastInterpd = vzLast[kNT];

        vxNextInterpd = vxCurrent[kNT];
        vyNextInterpd = vyCurrent[kNT];
        vzNextInterpd = vzCurrent[kNT];
    }

    // if(k==16300) printf("%f %f %f\n", vxLastInterpd, vyLastInterpd, vzLastInterpd);
    real VeloX = (velocityX + (1.f-tRatio)*vxLastInterpd + tRatio*vxNextInterpd)/velocityRatio;
    real VeloY = (velocityY + (1.f-tRatio)*vyLastInterpd + tRatio*vyNextInterpd)/velocityRatio; 
    real VeloZ = (velocityZ + (1.f-tRatio)*vzLastInterpd + tRatio*vzNextInterpd)/velocityRatio;
    // From here on just a copy of QVelDeviceCompZeroPress
    ////////////////////////////////////////////////////////////////////////////////

    Distributions27 dist;
    getPointersToDistributions(dist, distributions, numberOfLBnodes, isEvenTimestep);

    unsigned int KQK  = subgridDistanceIndices[k];
    unsigned int kzero= KQK;
    unsigned int ke   = KQK;
    unsigned int kw   = neighborX[KQK];
    unsigned int kn   = KQK;
    unsigned int ks   = neighborY[KQK];
    unsigned int kt   = KQK;
    unsigned int kb   = neighborZ[KQK];
    unsigned int ksw  = neighborY[kw];
    unsigned int kne  = KQK;
    unsigned int kse  = ks;
    unsigned int knw  = kw;
    unsigned int kbw  = neighborZ[kw];
    unsigned int kte  = KQK;
    unsigned int kbe  = kb;
    unsigned int ktw  = kw;
    unsigned int kbs  = neighborZ[ks];
    unsigned int ktn  = KQK;
    unsigned int kbn  = kb;
    unsigned int kts  = ks;
    unsigned int ktse = ks;
    unsigned int kbnw = kbw;
    unsigned int ktnw = kw;
    unsigned int kbse = kbs;
    unsigned int ktsw = ksw;
    unsigned int kbne = kb;
    unsigned int ktne = KQK;
    unsigned int kbsw = neighborZ[ksw];

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // based on BGK Plus Comp
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    real f_W    = (dist.f[DIR_P00])[ke   ];
    real f_E    = (dist.f[DIR_M00])[kw   ];
    real f_S    = (dist.f[DIR_0P0])[kn   ];
    real f_N    = (dist.f[DIR_0M0])[ks   ];
    real f_B    = (dist.f[DIR_00P])[kt   ];
    real f_T    = (dist.f[DIR_00M])[kb   ];
    real f_SW   = (dist.f[DIR_PP0])[kne  ];
    real f_NE   = (dist.f[DIR_MM0])[ksw  ];
    real f_NW   = (dist.f[DIR_PM0])[kse  ];
    real f_SE   = (dist.f[DIR_MP0])[knw  ];
    real f_BW   = (dist.f[DIR_P0P])[kte  ];
    real f_TE   = (dist.f[DIR_M0M])[kbw  ];
    real f_TW   = (dist.f[DIR_P0M])[kbe  ];
    real f_BE   = (dist.f[DIR_M0P])[ktw  ];
    real f_BS   = (dist.f[DIR_0PP])[ktn  ];
    real f_TN   = (dist.f[DIR_0MM])[kbs  ];
    real f_TS   = (dist.f[DIR_0PM])[kbn  ];
    real f_BN   = (dist.f[DIR_0MP])[kts  ];
    real f_ZERO = (dist.f[DIR_000])[kzero];
    real f_BSW  = (dist.f[DIR_PPP])[ktne ];
    real f_BNE  = (dist.f[DIR_MMP])[ktsw ];
    real f_BNW  = (dist.f[DIR_PMP])[ktse ];
    real f_BSE  = (dist.f[DIR_MPP])[ktnw ];
    real f_TSW  = (dist.f[DIR_PPM])[kbne ];
    real f_TNE  = (dist.f[DIR_MMM])[kbsw ];
    real f_TNW  = (dist.f[DIR_PMM])[kbse ];
    real f_TSE  = (dist.f[DIR_MPM])[kbnw ];

      ////////////////////////////////////////////////////////////////////////////////
      //! - Set macroscopic quantities
      //!
      real drho = c0o1;

      real vx1  = VeloX;          

      real vx2  = VeloY; 

      real vx3  = VeloZ; 

      real cusq = c3o2 * (vx1 * vx1 + vx2 * vx2 + vx3 * vx3);

      ////////////////////////////////////////////////////////////////////////////////
      f_ZERO  = c8o27*  (drho-(drho+c1o1)*cusq);
      f_E     = c2o27*  (drho+(drho+c1o1)*(c3o1*( vx1        )+c9o2*( vx1        )*( vx1        )-cusq));
      f_W     = c2o27*  (drho+(drho+c1o1)*(c3o1*(-vx1        )+c9o2*(-vx1        )*(-vx1        )-cusq));
      f_N     = c2o27*  (drho+(drho+c1o1)*(c3o1*(    vx2     )+c9o2*(     vx2    )*(     vx2    )-cusq));
      f_S     = c2o27*  (drho+(drho+c1o1)*(c3o1*(   -vx2     )+c9o2*(    -vx2    )*(    -vx2    )-cusq));
      f_T     = c2o27*  (drho+(drho+c1o1)*(c3o1*(         vx3)+c9o2*(         vx3)*(         vx3)-cusq));
      f_B     = c2o27*  (drho+(drho+c1o1)*(c3o1*(        -vx3)+c9o2*(        -vx3)*(        -vx3)-cusq));
      f_NE    = c1o54*  (drho+(drho+c1o1)*(c3o1*( vx1+vx2    )+c9o2*( vx1+vx2    )*( vx1+vx2    )-cusq));
      f_SW    = c1o54*  (drho+(drho+c1o1)*(c3o1*(-vx1-vx2    )+c9o2*(-vx1-vx2    )*(-vx1-vx2    )-cusq));
      f_SE    =  c1o54* (drho+(drho+c1o1)*(c3o1*( vx1-vx2    )+c9o2*( vx1-vx2    )*( vx1-vx2    )-cusq));
      f_NW    =  c1o54* (drho+(drho+c1o1)*(c3o1*(-vx1+vx2    )+c9o2*(-vx1+vx2    )*(-vx1+vx2    )-cusq));
      f_TE    =  c1o54* (drho+(drho+c1o1)*(c3o1*( vx1    +vx3)+c9o2*( vx1    +vx3)*( vx1    +vx3)-cusq));
      f_BW    =  c1o54* (drho+(drho+c1o1)*(c3o1*(-vx1    -vx3)+c9o2*(-vx1    -vx3)*(-vx1    -vx3)-cusq));
      f_BE    =  c1o54* (drho+(drho+c1o1)*(c3o1*( vx1    -vx3)+c9o2*( vx1    -vx3)*( vx1    -vx3)-cusq));
      f_TW    =  c1o54* (drho+(drho+c1o1)*(c3o1*(-vx1    +vx3)+c9o2*(-vx1    +vx3)*(-vx1    +vx3)-cusq));
      f_TN    =  c1o54* (drho+(drho+c1o1)*(c3o1*(     vx2+vx3)+c9o2*(     vx2+vx3)*(     vx2+vx3)-cusq));
      f_BS    =  c1o54* (drho+(drho+c1o1)*(c3o1*(    -vx2-vx3)+c9o2*(    -vx2-vx3)*(    -vx2-vx3)-cusq));
      f_BN    =  c1o54* (drho+(drho+c1o1)*(c3o1*(     vx2-vx3)+c9o2*(     vx2-vx3)*(     vx2-vx3)-cusq));
      f_TS    =  c1o54* (drho+(drho+c1o1)*(c3o1*(    -vx2+vx3)+c9o2*(    -vx2+vx3)*(    -vx2+vx3)-cusq));
      f_TNE   =  c1o216*(drho+(drho+c1o1)*(c3o1*( vx1+vx2+vx3)+c9o2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cusq));
      f_BSW   =  c1o216*(drho+(drho+c1o1)*(c3o1*(-vx1-vx2-vx3)+c9o2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cusq));
      f_BNE   =  c1o216*(drho+(drho+c1o1)*(c3o1*( vx1+vx2-vx3)+c9o2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cusq));
      f_TSW   =  c1o216*(drho+(drho+c1o1)*(c3o1*(-vx1-vx2+vx3)+c9o2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cusq));
      f_TSE   =  c1o216*(drho+(drho+c1o1)*(c3o1*( vx1-vx2+vx3)+c9o2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cusq));
      f_BNW   =  c1o216*(drho+(drho+c1o1)*(c3o1*(-vx1+vx2-vx3)+c9o2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cusq));
      f_BSE   =  c1o216*(drho+(drho+c1o1)*(c3o1*( vx1-vx2-vx3)+c9o2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cusq));
      f_TNW   =  c1o216*(drho+(drho+c1o1)*(c3o1*(-vx1+vx2+vx3)+c9o2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cusq));

      ////////////////////////////////////////////////////////////////////////////////
      //! write the new distributions to the bc nodes
      //!
      (dist.f[DIR_P00   ])[ke  ] = f_W   ;
      (dist.f[DIR_M00   ])[kw  ] = f_E   ;
      (dist.f[DIR_0P0   ])[kn  ] = f_S   ;
      (dist.f[DIR_0M0   ])[ks  ] = f_N   ;
      (dist.f[DIR_00P   ])[kt  ] = f_B   ;
      (dist.f[DIR_00M   ])[kb  ] = f_T   ;
      (dist.f[DIR_PP0  ])[kne  ] = f_SW  ;
      (dist.f[DIR_MM0  ])[ksw  ] = f_NE  ;
      (dist.f[DIR_PM0  ])[kse  ] = f_NW  ;
      (dist.f[DIR_MP0  ])[knw  ] = f_SE  ;
      (dist.f[DIR_P0P  ])[kte  ] = f_BW  ;
      (dist.f[DIR_M0M  ])[kbw  ] = f_TE  ;
      (dist.f[DIR_P0M  ])[kbe  ] = f_TW  ;
      (dist.f[DIR_M0P  ])[ktw  ] = f_BE  ;
      (dist.f[DIR_0PP  ])[ktn  ] = f_BS  ;
      (dist.f[DIR_0MM  ])[kbs  ] = f_TN  ;
      (dist.f[DIR_0PM  ])[kbn  ] = f_TS  ;
      (dist.f[DIR_0MP  ])[kts  ] = f_BN  ;
      (dist.f[DIR_000])[kzero] = f_ZERO;
      (dist.f[DIR_PPP ])[ktne ] = f_BSW ;
      (dist.f[DIR_MMP ])[ktsw ] = f_BNE ;
      (dist.f[DIR_PMP ])[ktse ] = f_BNW ;
      (dist.f[DIR_MPP ])[ktnw ] = f_BSE ;
      (dist.f[DIR_PPM ])[kbne ] = f_TSW ;
      (dist.f[DIR_MMM ])[kbsw ] = f_TNE ;
      (dist.f[DIR_PMM ])[kbse ] = f_TNW ;
      (dist.f[DIR_MPM ])[kbnw ] = f_TSE ;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void PrecursorDeviceDistributions( 	int* subgridDistanceIndices,
												int numberOfBCnodes,
                                                int numberOfPrecursorNodes,
												real* distributions,
												uint* neighborX, 
												uint* neighborY, 
												uint* neighborZ,
                                                uint* typeOfGridNode,       //DEBUG: remove later
												uint* neighborsNT, 
												uint* neighborsNB,
												uint* neighborsST,
												uint* neighborsSB,
												real* weightsNT, 
												real* weightsNB,
												real* weightsST,
												real* weightsSB,
												real* fsLast, 
												real* fsNext,
												real tRatio,
												unsigned long long numberOfLBnodes,
												bool isEvenTimestep)
{
    const unsigned k = vf::gpu::getNodeIndex();

    if(k>=numberOfBCnodes) return;

    uint kNT = neighborsNT[k];
    real dNT = weightsNT[k];

    real f0LastInterp, f1LastInterp, f2LastInterp, f3LastInterp, f4LastInterp, f5LastInterp, f6LastInterp, f7LastInterp, f8LastInterp, f9LastInterp, f10LastInterp, f11LastInterp, f12LastInterp, f13LastInterp, f14LastInterp;
    real f0NextInterp, f1NextInterp, f2NextInterp, f3NextInterp, f4NextInterp, f5NextInterp, f6NextInterp, f7NextInterp, f8NextInterp, f9NextInterp, f10NextInterp, f11NextInterp, f12NextInterp, f13NextInterp, f14NextInterp;

    real* f0Last = fsLast;
    real* f1Last = &fsLast[  numberOfPrecursorNodes];
    real* f2Last = &fsLast[2*numberOfPrecursorNodes];
    real* f3Last = &fsLast[3*numberOfPrecursorNodes];
    real* f4Last = &fsLast[4*numberOfPrecursorNodes];
    real* f5Last = &fsLast[5*numberOfPrecursorNodes];
    real* f6Last = &fsLast[6*numberOfPrecursorNodes];
    real* f7Last = &fsLast[7*numberOfPrecursorNodes];
    real* f8Last = &fsLast[8*numberOfPrecursorNodes];

    real* f9Last = &fsLast[9*numberOfPrecursorNodes];
    real* f10Last = &fsLast[10*numberOfPrecursorNodes];
    real* f11Last = &fsLast[11*numberOfPrecursorNodes];
    real* f12Last = &fsLast[12*numberOfPrecursorNodes];
    real* f13Last = &fsLast[13*numberOfPrecursorNodes];
    real* f14Last = &fsLast[14*numberOfPrecursorNodes];

    real* f0Next = fsNext;
    real* f1Next = &fsNext[  numberOfPrecursorNodes];
    real* f2Next = &fsNext[2*numberOfPrecursorNodes];
    real* f3Next = &fsNext[3*numberOfPrecursorNodes];
    real* f4Next = &fsNext[4*numberOfPrecursorNodes];
    real* f5Next = &fsNext[5*numberOfPrecursorNodes];
    real* f6Next = &fsNext[6*numberOfPrecursorNodes];
    real* f7Next = &fsNext[7*numberOfPrecursorNodes];
    real* f8Next = &fsNext[8*numberOfPrecursorNodes];

    real* f9Next  = &fsNext[9*numberOfPrecursorNodes];
    real* f10Next = &fsNext[10*numberOfPrecursorNodes];
    real* f11Next = &fsNext[11*numberOfPrecursorNodes];
    real* f12Next = &fsNext[12*numberOfPrecursorNodes];
    real* f13Next = &fsNext[13*numberOfPrecursorNodes];
    real* f14Next = &fsNext[14*numberOfPrecursorNodes];


    if(dNT<1e6)
    {
        uint kNB = neighborsNB[k];
        uint kST = neighborsST[k];
        uint kSB = neighborsSB[k];

        real dNB = weightsNB[k];
        real dST = weightsST[k];
        real dSB = weightsSB[k];

        real invWeightSum = 1.f/(dNT+dNB+dST+dSB);

        f0LastInterp = (f0Last[kNT]*dNT + f0Last[kNB]*dNB + f0Last[kST]*dST + f0Last[kSB]*dSB)*invWeightSum;
        f0NextInterp = (f0Next[kNT]*dNT + f0Next[kNB]*dNB + f0Next[kST]*dST + f0Next[kSB]*dSB)*invWeightSum;
        
        f1LastInterp = (f1Last[kNT]*dNT + f1Last[kNB]*dNB + f1Last[kST]*dST + f1Last[kSB]*dSB)*invWeightSum;
        f1NextInterp = (f1Next[kNT]*dNT + f1Next[kNB]*dNB + f1Next[kST]*dST + f1Next[kSB]*dSB)*invWeightSum;
        
        f2LastInterp = (f2Last[kNT]*dNT + f2Last[kNB]*dNB + f2Last[kST]*dST + f2Last[kSB]*dSB)*invWeightSum;
        f2NextInterp = (f2Next[kNT]*dNT + f2Next[kNB]*dNB + f2Next[kST]*dST + f2Next[kSB]*dSB)*invWeightSum;
        
        f3LastInterp = (f3Last[kNT]*dNT + f3Last[kNB]*dNB + f3Last[kST]*dST + f3Last[kSB]*dSB)*invWeightSum;
        f3NextInterp = (f3Next[kNT]*dNT + f3Next[kNB]*dNB + f3Next[kST]*dST + f3Next[kSB]*dSB)*invWeightSum;
        
        f4LastInterp = (f4Last[kNT]*dNT + f4Last[kNB]*dNB + f4Last[kST]*dST + f4Last[kSB]*dSB)*invWeightSum;
        f4NextInterp = (f4Next[kNT]*dNT + f4Next[kNB]*dNB + f4Next[kST]*dST + f4Next[kSB]*dSB)*invWeightSum;
        
        f5LastInterp = (f5Last[kNT]*dNT + f5Last[kNB]*dNB + f5Last[kST]*dST + f5Last[kSB]*dSB)*invWeightSum;
        f5NextInterp = (f5Next[kNT]*dNT + f5Next[kNB]*dNB + f5Next[kST]*dST + f5Next[kSB]*dSB)*invWeightSum;
        
        f6LastInterp = (f6Last[kNT]*dNT + f6Last[kNB]*dNB + f6Last[kST]*dST + f6Last[kSB]*dSB)*invWeightSum;
        f6NextInterp = (f6Next[kNT]*dNT + f6Next[kNB]*dNB + f6Next[kST]*dST + f6Next[kSB]*dSB)*invWeightSum;
        
        f7LastInterp = (f7Last[kNT]*dNT + f7Last[kNB]*dNB + f7Last[kST]*dST + f7Last[kSB]*dSB)*invWeightSum;
        f7NextInterp = (f7Next[kNT]*dNT + f7Next[kNB]*dNB + f7Next[kST]*dST + f7Next[kSB]*dSB)*invWeightSum;
        
        f8LastInterp = (f8Last[kNT]*dNT + f8Last[kNB]*dNB + f8Last[kST]*dST + f8Last[kSB]*dSB)*invWeightSum;
        f8NextInterp = (f8Next[kNT]*dNT + f8Next[kNB]*dNB + f8Next[kST]*dST + f8Next[kSB]*dSB)*invWeightSum;

        f9LastInterp = (f9Last[kNT]*dNT + f9Last[kNB]*dNB + f9Last[kST]*dST + f9Last[kSB]*dSB)*invWeightSum;
        f9NextInterp = (f9Next[kNT]*dNT + f9Next[kNB]*dNB + f9Next[kST]*dST + f9Next[kSB]*dSB)*invWeightSum;

        f10LastInterp = (f10Last[kNT]*dNT + f10Last[kNB]*dNB + f10Last[kST]*dST + f10Last[kSB]*dSB)*invWeightSum;
        f10NextInterp = (f10Next[kNT]*dNT + f10Next[kNB]*dNB + f10Next[kST]*dST + f10Next[kSB]*dSB)*invWeightSum;

        f11LastInterp = (f11Last[kNT]*dNT + f11Last[kNB]*dNB + f11Last[kST]*dST + f11Last[kSB]*dSB)*invWeightSum;
        f11NextInterp = (f11Next[kNT]*dNT + f11Next[kNB]*dNB + f11Next[kST]*dST + f11Next[kSB]*dSB)*invWeightSum;

        f12LastInterp = (f12Last[kNT]*dNT + f12Last[kNB]*dNB + f12Last[kST]*dST + f12Last[kSB]*dSB)*invWeightSum;
        f12NextInterp = (f12Next[kNT]*dNT + f12Next[kNB]*dNB + f12Next[kST]*dST + f12Next[kSB]*dSB)*invWeightSum;

        f13LastInterp = (f13Last[kNT]*dNT + f13Last[kNB]*dNB + f13Last[kST]*dST + f13Last[kSB]*dSB)*invWeightSum;
        f13NextInterp = (f13Next[kNT]*dNT + f13Next[kNB]*dNB + f13Next[kST]*dST + f13Next[kSB]*dSB)*invWeightSum;

        f14LastInterp = (f14Last[kNT]*dNT + f14Last[kNB]*dNB + f14Last[kST]*dST + f14Last[kSB]*dSB)*invWeightSum;
        f14NextInterp = (f14Next[kNT]*dNT + f14Next[kNB]*dNB + f14Next[kST]*dST + f14Next[kSB]*dSB)*invWeightSum;
    
    } else {
        f0LastInterp = f0Last[kNT];
        f1LastInterp = f1Last[kNT];
        f2LastInterp = f2Last[kNT];
        f3LastInterp = f3Last[kNT];
        f4LastInterp = f4Last[kNT];
        f5LastInterp = f5Last[kNT];
        f6LastInterp = f6Last[kNT];
        f7LastInterp = f7Last[kNT];
        f8LastInterp = f8Last[kNT];

        f9LastInterp = f9Last[kNT];
        f10LastInterp = f10Last[kNT];
        f11LastInterp = f11Last[kNT];
        f12LastInterp = f12Last[kNT];
        f13LastInterp = f13Last[kNT];
        f14LastInterp = f14Last[kNT];

        f0NextInterp = f0Next[kNT];
        f1NextInterp = f1Next[kNT];
        f2NextInterp = f2Next[kNT];
        f3NextInterp = f3Next[kNT];
        f4NextInterp = f4Next[kNT];
        f5NextInterp = f5Next[kNT];
        f6NextInterp = f6Next[kNT];
        f7NextInterp = f7Next[kNT];
        f8NextInterp = f8Next[kNT];

        f9NextInterp = f9Next[kNT];
        f10NextInterp = f10Next[kNT];
        f11NextInterp = f11Next[kNT];
        f12NextInterp = f12Next[kNT];
        f13NextInterp = f13Next[kNT];
        f14NextInterp = f14Next[kNT];
    }
    Distributions27 dist;
    getPointersToDistributions(dist, distributions, numberOfLBnodes, isEvenTimestep);

    unsigned int KQK  = subgridDistanceIndices[k];
    // unsigned int kzero= KQK;
    unsigned int ke   = KQK;
    // unsigned int kw   = neighborX[KQK];
    unsigned int kn   = KQK;
    unsigned int ks   = neighborY[KQK];
    unsigned int kt   = KQK;
    unsigned int kb   = neighborZ[KQK];
    // unsigned int ksw  = neighborY[kw];
    unsigned int kne  = KQK;
    unsigned int kse  = ks;
    // unsigned int knw  = kw;
    // unsigned int kbw  = neighborZ[kw];
    unsigned int kte  = KQK;
    unsigned int kbe  = kb;
    // unsigned int ktw  = kw;
    unsigned int kbs  = neighborZ[ks];
    unsigned int ktn  = KQK;
    // unsigned int kbn  = kb;
    unsigned int kts  = ks;
    unsigned int ktse = ks;
    // unsigned int kbnw = kbw;
    // unsigned int ktnw = kw;
    unsigned int kbse = kbs;
    // unsigned int ktsw = ksw;
    unsigned int kbne = kb;
    unsigned int ktne = KQK;
    // unsigned int kbsw = neighborZ[ksw];

    dist.f[DIR_P00][ke]   = f0LastInterp*(1.f-tRatio) + f0NextInterp*tRatio;
    dist.f[DIR_PP0][kne]  = f1LastInterp*(1.f-tRatio) + f1NextInterp*tRatio;
    dist.f[DIR_PM0][kse]  = f2LastInterp*(1.f-tRatio) + f2NextInterp*tRatio;
    dist.f[DIR_P0P][kte]  = f3LastInterp*(1.f-tRatio) + f3NextInterp*tRatio;
    dist.f[DIR_P0M][kbe]  = f4LastInterp*(1.f-tRatio) + f4NextInterp*tRatio;
    dist.f[DIR_PPP][ktne] = f5LastInterp*(1.f-tRatio) + f5NextInterp*tRatio;
    dist.f[DIR_PMP][ktse] = f6LastInterp*(1.f-tRatio) + f6NextInterp*tRatio;
    dist.f[DIR_PPM][kbne] = f7LastInterp*(1.f-tRatio) + f7NextInterp*tRatio;
    dist.f[DIR_PMM][kbse] = f8LastInterp*(1.f-tRatio) + f8NextInterp*tRatio;

    dist.f[DIR_0P0][kn]  = f9LastInterp*(1.f-tRatio) + f9NextInterp*tRatio;
    dist.f[DIR_0M0][ks] = f10LastInterp*(1.f-tRatio) + f10NextInterp*tRatio;
    dist.f[DIR_00P][kt] = f11LastInterp*(1.f-tRatio) + f11NextInterp*tRatio;
    dist.f[DIR_00M][kb] = f12LastInterp*(1.f-tRatio) + f12NextInterp*tRatio;
    dist.f[DIR_0PP][ktn] = f13LastInterp*(1.f-tRatio) + f13NextInterp*tRatio;
    dist.f[DIR_0MP][kts] = f14LastInterp*(1.f-tRatio) + f14NextInterp*tRatio;


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void QPrecursorDeviceDistributions( 	int* subgridDistanceIndices,
                                                real* subgridDistances,
                                                int sizeQ,
												int numberOfBCnodes,
                                                int numberOfPrecursorNodes,
												real* distributions,
												uint* neighborX, 
												uint* neighborY, 
												uint* neighborZ,
												uint* neighborsNT, 
												uint* neighborsNB,
												uint* neighborsST,
												uint* neighborsSB,
												real* weightsNT, 
												real* weightsNB,
												real* weightsST,
												real* weightsSB,
												real* fsLast, 
												real* fsNext,
												real tRatio,
												unsigned long long numberOfLBnodes,
												bool isEvenTimestep)
{
    const unsigned k = vf::gpu::getNodeIndex();

    if(k>=numberOfBCnodes) return;

    uint kNT = neighborsNT[k];
    real dNT = weightsNT[k];

    real f0LastInterp, f1LastInterp, f2LastInterp, f3LastInterp, f4LastInterp, f5LastInterp, f6LastInterp, f7LastInterp, f8LastInterp;
    real f0NextInterp, f1NextInterp, f2NextInterp, f3NextInterp, f4NextInterp, f5NextInterp, f6NextInterp, f7NextInterp, f8NextInterp;

    real* f0Last = fsLast;
    real* f1Last = &fsLast[  numberOfPrecursorNodes];
    real* f2Last = &fsLast[2*numberOfPrecursorNodes];
    real* f3Last = &fsLast[3*numberOfPrecursorNodes];
    real* f4Last = &fsLast[4*numberOfPrecursorNodes];
    real* f5Last = &fsLast[5*numberOfPrecursorNodes];
    real* f6Last = &fsLast[6*numberOfPrecursorNodes];
    real* f7Last = &fsLast[7*numberOfPrecursorNodes];
    real* f8Last = &fsLast[8*numberOfPrecursorNodes];

    real* f0Next = fsNext;
    real* f1Next = &fsNext[  numberOfPrecursorNodes];
    real* f2Next = &fsNext[2*numberOfPrecursorNodes];
    real* f3Next = &fsNext[3*numberOfPrecursorNodes];
    real* f4Next = &fsNext[4*numberOfPrecursorNodes];
    real* f5Next = &fsNext[5*numberOfPrecursorNodes];
    real* f6Next = &fsNext[6*numberOfPrecursorNodes];
    real* f7Next = &fsNext[7*numberOfPrecursorNodes];
    real* f8Next = &fsNext[8*numberOfPrecursorNodes];


    if(dNT<1e6)
    {
        uint kNB = neighborsNB[k];
        uint kST = neighborsST[k];
        uint kSB = neighborsSB[k];

        real dNB = weightsNB[k];
        real dST = weightsST[k];
        real dSB = weightsSB[k];

        real invWeightSum = 1.f/(dNT+dNB+dST+dSB);

        f0LastInterp = (f0Last[kNT]*dNT + f0Last[kNB]*dNB + f0Last[kST]*dST + f0Last[kSB]*dSB)*invWeightSum;
        f0NextInterp = (f0Next[kNT]*dNT + f0Next[kNB]*dNB + f0Next[kST]*dST + f0Next[kSB]*dSB)*invWeightSum;
        
        f1LastInterp = (f1Last[kNT]*dNT + f1Last[kNB]*dNB + f1Last[kST]*dST + f1Last[kSB]*dSB)*invWeightSum;
        f1NextInterp = (f1Next[kNT]*dNT + f1Next[kNB]*dNB + f1Next[kST]*dST + f1Next[kSB]*dSB)*invWeightSum;
        
        f2LastInterp = (f2Last[kNT]*dNT + f2Last[kNB]*dNB + f2Last[kST]*dST + f2Last[kSB]*dSB)*invWeightSum;
        f2NextInterp = (f2Next[kNT]*dNT + f2Next[kNB]*dNB + f2Next[kST]*dST + f2Next[kSB]*dSB)*invWeightSum;
        
        f3LastInterp = (f3Last[kNT]*dNT + f3Last[kNB]*dNB + f3Last[kST]*dST + f3Last[kSB]*dSB)*invWeightSum;
        f3NextInterp = (f3Next[kNT]*dNT + f3Next[kNB]*dNB + f3Next[kST]*dST + f3Next[kSB]*dSB)*invWeightSum;
        
        f4LastInterp = (f4Last[kNT]*dNT + f4Last[kNB]*dNB + f4Last[kST]*dST + f4Last[kSB]*dSB)*invWeightSum;
        f4NextInterp = (f4Next[kNT]*dNT + f4Next[kNB]*dNB + f4Next[kST]*dST + f4Next[kSB]*dSB)*invWeightSum;
        
        f5LastInterp = (f5Last[kNT]*dNT + f5Last[kNB]*dNB + f5Last[kST]*dST + f5Last[kSB]*dSB)*invWeightSum;
        f5NextInterp = (f5Next[kNT]*dNT + f5Next[kNB]*dNB + f5Next[kST]*dST + f5Next[kSB]*dSB)*invWeightSum;
        
        f6LastInterp = (f6Last[kNT]*dNT + f6Last[kNB]*dNB + f6Last[kST]*dST + f6Last[kSB]*dSB)*invWeightSum;
        f6NextInterp = (f6Next[kNT]*dNT + f6Next[kNB]*dNB + f6Next[kST]*dST + f6Next[kSB]*dSB)*invWeightSum;
        
        f7LastInterp = (f7Last[kNT]*dNT + f7Last[kNB]*dNB + f7Last[kST]*dST + f7Last[kSB]*dSB)*invWeightSum;
        f7NextInterp = (f7Next[kNT]*dNT + f7Next[kNB]*dNB + f7Next[kST]*dST + f7Next[kSB]*dSB)*invWeightSum;
        
        f8LastInterp = (f8Last[kNT]*dNT + f8Last[kNB]*dNB + f8Last[kST]*dST + f8Last[kSB]*dSB)*invWeightSum;
        f8NextInterp = (f8Next[kNT]*dNT + f8Next[kNB]*dNB + f8Next[kST]*dST + f8Next[kSB]*dSB)*invWeightSum;
    
    } else {
        f0LastInterp = f0Last[kNT];
        f1LastInterp = f1Last[kNT];
        f2LastInterp = f2Last[kNT];
        f3LastInterp = f3Last[kNT];
        f4LastInterp = f4Last[kNT];
        f5LastInterp = f5Last[kNT];
        f6LastInterp = f6Last[kNT];
        f7LastInterp = f7Last[kNT];
        f8LastInterp = f8Last[kNT];

        f0NextInterp = f0Next[kNT];
        f1NextInterp = f1Next[kNT];
        f2NextInterp = f2Next[kNT];
        f3NextInterp = f3Next[kNT];
        f4NextInterp = f4Next[kNT];
        f5NextInterp = f5Next[kNT];
        f6NextInterp = f6Next[kNT];
        f7NextInterp = f7Next[kNT];
        f8NextInterp = f8Next[kNT];
    }
    Distributions27 dist;
    getPointersToDistributions(dist, distributions, numberOfLBnodes, isEvenTimestep);

    unsigned int KQK  = subgridDistanceIndices[k];
    // unsigned int kzero= KQK;
    unsigned int ke   = KQK;
    // unsigned int kw   = neighborX[KQK];
    // unsigned int kn   = KQK;
    unsigned int ks   = neighborY[KQK];
    // unsigned int kt   = KQK;
    unsigned int kb   = neighborZ[KQK];
    // unsigned int ksw  = neighborY[kw];
    unsigned int kne  = KQK;
    unsigned int kse  = ks;
    // unsigned int knw  = kw;
    // unsigned int kbw  = neighborZ[kw];
    unsigned int kte  = KQK;
    unsigned int kbe  = kb;
    // unsigned int ktw  = kw;
    unsigned int kbs  = neighborZ[ks];
    // unsigned int ktn  = KQK;
    // unsigned int kbn  = kb;
    // unsigned int kts  = ks;
    unsigned int ktse = ks;
    // unsigned int kbnw = kbw;
    // unsigned int ktnw = kw;
    unsigned int kbse = kbs;
    // unsigned int ktsw = ksw;
    unsigned int kbne = kb;
    unsigned int ktne = KQK;
    // unsigned int kbsw = neighborZ[ksw];
    SubgridDistances27 qs;
    getPointersToSubgridDistances(qs, subgridDistances, sizeQ);

    real q;
    q = qs.q[DIR_P00][k]; if(q>= c0o1 && q <= c1o1) dist.f[DIR_P00][ke]   = f0LastInterp*(1.f-tRatio) + f0NextInterp*tRatio;
    q = qs.q[DIR_PP0][k]; if(q>= c0o1 && q <= c1o1) dist.f[DIR_PP0][kne]  = f1LastInterp*(1.f-tRatio) + f1NextInterp*tRatio;
    q = qs.q[DIR_PM0][k]; if(q>= c0o1 && q <= c1o1) dist.f[DIR_PM0][kse]  = f2LastInterp*(1.f-tRatio) + f2NextInterp*tRatio;
    q = qs.q[DIR_P0P][k]; if(q>= c0o1 && q <= c1o1) dist.f[DIR_P0P][kte]  = f3LastInterp*(1.f-tRatio) + f3NextInterp*tRatio;
    q = qs.q[DIR_P0M][k]; if(q>= c0o1 && q <= c1o1) dist.f[DIR_P0M][kbe]  = f4LastInterp*(1.f-tRatio) + f4NextInterp*tRatio;
    q = qs.q[DIR_PPP][k]; if(q>= c0o1 && q <= c1o1) dist.f[DIR_PPP][ktne] = f5LastInterp*(1.f-tRatio) + f5NextInterp*tRatio;
    q = qs.q[DIR_PMP][k]; if(q>= c0o1 && q <= c1o1) dist.f[DIR_PMP][ktse] = f6LastInterp*(1.f-tRatio) + f6NextInterp*tRatio;
    q = qs.q[DIR_PPM][k]; if(q>= c0o1 && q <= c1o1) dist.f[DIR_PPM][kbne] = f7LastInterp*(1.f-tRatio) + f7NextInterp*tRatio;
    q = qs.q[DIR_PMM][k]; if(q>= c0o1 && q <= c1o1) dist.f[DIR_PMM][kbse] = f8LastInterp*(1.f-tRatio) + f8NextInterp*tRatio;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
