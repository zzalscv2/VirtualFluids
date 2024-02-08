//=======================================================================================
// ____          ____    __    ______     __________   __      __       __        __
// \    \       |    |  |  |  |   _   \  |___    ___| |  |    |  |     /  \      |  |
//  \    \      |    |  |  |  |  |_)   |     |  |     |  |    |  |    /    \     |  |
//   \    \     |    |  |  |  |   _   /      |  |     |  |    |  |   /  /\  \    |  |
//    \    \    |    |  |  |  |  | \  \      |  |     |   \__/   |  /  ____  \   |  |____
//     \    \   |    |  |__|  |__|  \__\     |__|      \________/  /__/    \__\  |_______|
//      \    \  |    |   ________________________________________________________________
//       \    \ |    |  |  ______________________________________________________________|
//        \    \|    |  |  |         __          __     __     __     ______      _______
//         \         |  |  |_____   |  |        |  |   |  |   |  |   |   _  \    /  _____)
//          \        |  |   _____|  |  |        |  |   |  |   |  |   |  | \  \   \_______
//           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  |
//            \ _____|  |__|        |________|   \_______/    |__|   |______/    (_______/
//
//  This file is part of VirtualFluids. VirtualFluids is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  VirtualFluids is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
//  for more details.
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//  SPDX-FileCopyrightText: Copyright © VirtualFluids Project contributors, see AUTHORS.md in root folder
//
//! \addtogroup gpu_BoundaryConditions BoundaryConditions
//! \ingroup gpu_core core
//! \{
//! \author Martin Schoenherr, Anna Wellmann
//======================================================================================
#include "Calculation/Calculation.h"
#include "lbm/constants/D3Q27.h"
#include "basics/constants/NumericConstants.h"
#include "lbm/MacroscopicQuantities.h"
#include "Utilities/KernelUtilities.h"

using namespace vf::basics::constant;
using namespace vf::lbm::dir;
using namespace vf::gpu;

__global__ void PressureNonEquilibriumCompressible_Device(
    real* rhoBC,
    real* distributions,
    int* bcNodeIndices,
    int* bcNeighborIndices,
    int numberOfBCnodes,
    real omega1,
    unsigned int* neighborX,
    unsigned int* neighborY,
    unsigned int* neighborZ,
    unsigned long long numberOfLBnodes,
    bool isEvenTimestep,
    size_t direction)
{
   ////////////////////////////////////////////////////////////////////////////////
   //! The pressure boundary condition is executed in the following steps
   //!

   ////////////////////////////////////////////////////////////////////////////////
   //! - Get node index coordinates from threadIdx, blockIdx, blockDim and gridDim.
   //!
   const unsigned nodeIndex = getNodeIndex();

   ////////////////////////////////////////////////////////////////////////////////
   //! - Run for all indices in size of boundary condition (numberOfBCnodes)
   //!
   if(nodeIndex < numberOfBCnodes)
   {
      //////////////////////////////////////////////////////////////////////////
      //! - Read distributions: style of reading and writing the distributions from/to stored arrays dependent on timestep is based on the esoteric twist algorithm \ref
      //! <a href="https://doi.org/10.3390/computation5020019"><b>[ M. Geier et al. (2017), DOI:10.3390/computation5020019 ]</b></a>
      //!
      Distributions27 dist;
      getPointersToDistributions(dist, distributions, numberOfLBnodes, isEvenTimestep);

      ////////////////////////////////////////////////////////////////////////////////
      //! - Set local pressure
      //!
      real rhoBClocal = rhoBC[nodeIndex];

      ////////////////////////////////////////////////////////////////////////////////
      //! - Set neighbor indices (necessary for indirect addressing) for current node
      //!
      vf::gpu::ListIndices neighborIndices(bcNodeIndices[nodeIndex], neighborX, neighborY, neighborZ);

      ////////////////////////////////////////////////////////////////////////////////
      //! - Set neighbor indices (necessary for indirect addressing) for neighboring node
      //!
      vf::gpu::ListIndices neighborIndicesOfNeighbor(bcNeighborIndices[nodeIndex], neighborX, neighborY, neighborZ);
      ////////////////////////////////////////////////////////////////////////////////
      //! - Set local distributions for neighboring node
      //!
      real f_Neighbor[27];
      vf::gpu::getPreCollisionDistribution(f_Neighbor, dist, neighborIndicesOfNeighbor);

      ////////////////////////////////////////////////////////////////////////////////
      //! - Calculate macroscopic quantities (for neighboring node)
      //!
      real drho1;
      real vx1;
      real vx2;
      real vx3;
      vf::lbm::getCompressibleMacroscopicValues(f_Neighbor, drho1, vx1, vx2, vx3);
      real cusq = c3o2 * (vx1 * vx1 + vx2 * vx2 + vx3 * vx3);

      ////////////////////////////////////////////////////////////////////////////////
      //! subtract the equilibrium (eq) to obtain the non-equilibrium (neq) (for neighboring node)
      //!

      real f_Neighbor_d000 = f_Neighbor[d000];
      real f_Neighbor_dP00 = f_Neighbor[dP00];
      real f_Neighbor_dM00 = f_Neighbor[dM00];
      real f_Neighbor_d0P0 = f_Neighbor[d0P0];
      real f_Neighbor_d0M0 = f_Neighbor[d0M0];
      real f_Neighbor_d00P = f_Neighbor[d00P];
      real f_Neighbor_d00M = f_Neighbor[d00M];
      real f_Neighbor_dPP0 = f_Neighbor[dPP0];
      real f_Neighbor_dMM0 = f_Neighbor[dMM0];
      real f_Neighbor_dPM0 = f_Neighbor[dPM0];
      real f_Neighbor_dMP0 = f_Neighbor[dMP0];
      real f_Neighbor_dP0P = f_Neighbor[dP0P];
      real f_Neighbor_dM0M = f_Neighbor[dM0M];
      real f_Neighbor_dP0M = f_Neighbor[dP0M];
      real f_Neighbor_dM0P = f_Neighbor[dM0P];
      real f_Neighbor_d0PP = f_Neighbor[d0PP];
      real f_Neighbor_d0MM = f_Neighbor[d0MM];
      real f_Neighbor_d0PM = f_Neighbor[d0PM];
      real f_Neighbor_d0MP = f_Neighbor[d0MP];
      real f_Neighbor_dPPP = f_Neighbor[dPPP];
      real f_Neighbor_dMMM = f_Neighbor[dMMM];
      real f_Neighbor_dPPM = f_Neighbor[dPPM];
      real f_Neighbor_dMMP = f_Neighbor[dMMP];
      real f_Neighbor_dPMP = f_Neighbor[dPMP];
      real f_Neighbor_dMPM = f_Neighbor[dMPM];
      real f_Neighbor_dPMM = f_Neighbor[dPMM];
      real f_Neighbor_dMPP = f_Neighbor[dMPP];

      f_Neighbor_d000 -= c8o27*  (drho1-(drho1+c1o1)*cusq);
      f_Neighbor_dP00 -= c2o27*  (drho1+(drho1+c1o1)*(c3o1*( vx1        )+c9o2*( vx1        )*( vx1        )-cusq));
      f_Neighbor_dM00 -= c2o27*  (drho1+(drho1+c1o1)*(c3o1*(-vx1        )+c9o2*(-vx1        )*(-vx1        )-cusq));
      f_Neighbor_d0P0 -= c2o27*  (drho1+(drho1+c1o1)*(c3o1*(    vx2     )+c9o2*(     vx2    )*(     vx2    )-cusq));
      f_Neighbor_d0M0 -= c2o27*  (drho1+(drho1+c1o1)*(c3o1*(   -vx2     )+c9o2*(    -vx2    )*(    -vx2    )-cusq));
      f_Neighbor_d00P -= c2o27*  (drho1+(drho1+c1o1)*(c3o1*(         vx3)+c9o2*(         vx3)*(         vx3)-cusq));
      f_Neighbor_d00M -= c2o27*  (drho1+(drho1+c1o1)*(c3o1*(        -vx3)+c9o2*(        -vx3)*(        -vx3)-cusq));
      f_Neighbor_dPP0 -= c1o54*  (drho1+(drho1+c1o1)*(c3o1*( vx1+vx2    )+c9o2*( vx1+vx2    )*( vx1+vx2    )-cusq));
      f_Neighbor_dMM0 -= c1o54*  (drho1+(drho1+c1o1)*(c3o1*(-vx1-vx2    )+c9o2*(-vx1-vx2    )*(-vx1-vx2    )-cusq));
      f_Neighbor_dPM0 -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*( vx1-vx2    )+c9o2*( vx1-vx2    )*( vx1-vx2    )-cusq));
      f_Neighbor_dMP0 -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(-vx1+vx2    )+c9o2*(-vx1+vx2    )*(-vx1+vx2    )-cusq));
      f_Neighbor_dP0P -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*( vx1    +vx3)+c9o2*( vx1    +vx3)*( vx1    +vx3)-cusq));
      f_Neighbor_dM0M -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(-vx1    -vx3)+c9o2*(-vx1    -vx3)*(-vx1    -vx3)-cusq));
      f_Neighbor_dP0M -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*( vx1    -vx3)+c9o2*( vx1    -vx3)*( vx1    -vx3)-cusq));
      f_Neighbor_dM0P -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(-vx1    +vx3)+c9o2*(-vx1    +vx3)*(-vx1    +vx3)-cusq));
      f_Neighbor_d0PP -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(     vx2+vx3)+c9o2*(     vx2+vx3)*(     vx2+vx3)-cusq));
      f_Neighbor_d0MM -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(    -vx2-vx3)+c9o2*(    -vx2-vx3)*(    -vx2-vx3)-cusq));
      f_Neighbor_d0PM -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(     vx2-vx3)+c9o2*(     vx2-vx3)*(     vx2-vx3)-cusq));
      f_Neighbor_d0MP -=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(    -vx2+vx3)+c9o2*(    -vx2+vx3)*(    -vx2+vx3)-cusq));
      f_Neighbor_dPPP -=  c1o216*(drho1+(drho1+c1o1)*(c3o1*( vx1+vx2+vx3)+c9o2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cusq));
      f_Neighbor_dMMM -=  c1o216*(drho1+(drho1+c1o1)*(c3o1*(-vx1-vx2-vx3)+c9o2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cusq));
      f_Neighbor_dPPM -=  c1o216*(drho1+(drho1+c1o1)*(c3o1*( vx1+vx2-vx3)+c9o2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cusq));
      f_Neighbor_dMMP -=  c1o216*(drho1+(drho1+c1o1)*(c3o1*(-vx1-vx2+vx3)+c9o2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cusq));
      f_Neighbor_dPMP -=  c1o216*(drho1+(drho1+c1o1)*(c3o1*( vx1-vx2+vx3)+c9o2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cusq));
      f_Neighbor_dMPM -=  c1o216*(drho1+(drho1+c1o1)*(c3o1*(-vx1+vx2-vx3)+c9o2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cusq));
      f_Neighbor_dPMM -=  c1o216*(drho1+(drho1+c1o1)*(c3o1*( vx1-vx2-vx3)+c9o2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cusq));
      f_Neighbor_dMPP -=  c1o216*(drho1+(drho1+c1o1)*(c3o1*(-vx1+vx2+vx3)+c9o2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cusq));

      ////////////////////////////////////////////////////////////////////////////////
      //! redefine drho1 with rhoBClocal
      //!
      drho1 = rhoBClocal;

      ////////////////////////////////////////////////////////////////////////////////
      //! add the equilibrium (eq), which is calculated with rhoBClocal (for neighboring node)
      //!
      f_Neighbor_d000 += c8o27*  (drho1-(drho1+c1o1)*cusq);
      f_Neighbor_dP00 += c2o27*  (drho1+(drho1+c1o1)*(c3o1*( vx1        )+c9o2*( vx1        )*( vx1        )-cusq));
      f_Neighbor_dM00 += c2o27*  (drho1+(drho1+c1o1)*(c3o1*(-vx1        )+c9o2*(-vx1        )*(-vx1        )-cusq));
      f_Neighbor_d0P0 += c2o27*  (drho1+(drho1+c1o1)*(c3o1*(    vx2     )+c9o2*(     vx2    )*(     vx2    )-cusq));
      f_Neighbor_d0M0 += c2o27*  (drho1+(drho1+c1o1)*(c3o1*(   -vx2     )+c9o2*(    -vx2    )*(    -vx2    )-cusq));
      f_Neighbor_d00P += c2o27*  (drho1+(drho1+c1o1)*(c3o1*(         vx3)+c9o2*(         vx3)*(         vx3)-cusq));
      f_Neighbor_d00M += c2o27*  (drho1+(drho1+c1o1)*(c3o1*(        -vx3)+c9o2*(        -vx3)*(        -vx3)-cusq));
      f_Neighbor_dPP0 += c1o54*  (drho1+(drho1+c1o1)*(c3o1*( vx1+vx2    )+c9o2*( vx1+vx2    )*( vx1+vx2    )-cusq));
      f_Neighbor_dMM0 += c1o54*  (drho1+(drho1+c1o1)*(c3o1*(-vx1-vx2    )+c9o2*(-vx1-vx2    )*(-vx1-vx2    )-cusq));
      f_Neighbor_dPM0 +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*( vx1-vx2    )+c9o2*( vx1-vx2    )*( vx1-vx2    )-cusq));
      f_Neighbor_dMP0 +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(-vx1+vx2    )+c9o2*(-vx1+vx2    )*(-vx1+vx2    )-cusq));
      f_Neighbor_dP0P +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*( vx1    +vx3)+c9o2*( vx1    +vx3)*( vx1    +vx3)-cusq));
      f_Neighbor_dM0M +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(-vx1    -vx3)+c9o2*(-vx1    -vx3)*(-vx1    -vx3)-cusq));
      f_Neighbor_dP0M +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*( vx1    -vx3)+c9o2*( vx1    -vx3)*( vx1    -vx3)-cusq));
      f_Neighbor_dM0P +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(-vx1    +vx3)+c9o2*(-vx1    +vx3)*(-vx1    +vx3)-cusq));
      f_Neighbor_d0PP +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(     vx2+vx3)+c9o2*(     vx2+vx3)*(     vx2+vx3)-cusq));
      f_Neighbor_d0MM +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(    -vx2-vx3)+c9o2*(    -vx2-vx3)*(    -vx2-vx3)-cusq));
      f_Neighbor_d0PM +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(     vx2-vx3)+c9o2*(     vx2-vx3)*(     vx2-vx3)-cusq));
      f_Neighbor_d0MP +=  c1o54* (drho1+(drho1+c1o1)*(c3o1*(    -vx2+vx3)+c9o2*(    -vx2+vx3)*(    -vx2+vx3)-cusq));
      f_Neighbor_dPPP +=  c1o216*(drho1+(drho1+c1o1)*(c3o1*( vx1+vx2+vx3)+c9o2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cusq));
      f_Neighbor_dMMM +=  c1o216*(drho1+(drho1+c1o1)*(c3o1*(-vx1-vx2-vx3)+c9o2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cusq));
      f_Neighbor_dPPM +=  c1o216*(drho1+(drho1+c1o1)*(c3o1*( vx1+vx2-vx3)+c9o2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cusq));
      f_Neighbor_dMMP +=  c1o216*(drho1+(drho1+c1o1)*(c3o1*(-vx1-vx2+vx3)+c9o2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cusq));
      f_Neighbor_dPMP +=  c1o216*(drho1+(drho1+c1o1)*(c3o1*( vx1-vx2+vx3)+c9o2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cusq));
      f_Neighbor_dMPM +=  c1o216*(drho1+(drho1+c1o1)*(c3o1*(-vx1+vx2-vx3)+c9o2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cusq));
      f_Neighbor_dPMM +=  c1o216*(drho1+(drho1+c1o1)*(c3o1*( vx1-vx2-vx3)+c9o2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cusq));
      f_Neighbor_dMPP +=  c1o216*(drho1+(drho1+c1o1)*(c3o1*(-vx1+vx2+vx3)+c9o2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cusq));

      //////////////////////////////////////////////////////////////////////////

      __syncthreads();

      ////////////////////////////////////////////////////////////////////////////////
      //! write the new distributions to the bc nodes (only for the relevant directions)
      //!

      f_Neighbor[d000] = f_Neighbor_d000;
      f_Neighbor[dP00] = f_Neighbor_dP00;
      f_Neighbor[dM00] = f_Neighbor_dM00;
      f_Neighbor[d0P0] = f_Neighbor_d0P0;
      f_Neighbor[d0M0] = f_Neighbor_d0M0;
      f_Neighbor[d00P] = f_Neighbor_d00P;
      f_Neighbor[d00M] = f_Neighbor_d00M;
      f_Neighbor[dPP0] = f_Neighbor_dPP0;
      f_Neighbor[dMM0] = f_Neighbor_dMM0;
      f_Neighbor[dPM0] = f_Neighbor_dPM0;
      f_Neighbor[dMP0] = f_Neighbor_dMP0;
      f_Neighbor[dP0P] = f_Neighbor_dP0P;
      f_Neighbor[dM0M] = f_Neighbor_dM0M;
      f_Neighbor[dP0M] = f_Neighbor_dP0M;
      f_Neighbor[dM0P] = f_Neighbor_dM0P;
      f_Neighbor[d0PP] = f_Neighbor_d0PP;
      f_Neighbor[d0MM] = f_Neighbor_d0MM;
      f_Neighbor[d0PM] = f_Neighbor_d0PM;
      f_Neighbor[d0MP] = f_Neighbor_d0MP;
      f_Neighbor[dPPP] = f_Neighbor_dPPP;
      f_Neighbor[dMMM] = f_Neighbor_dMMM;
      f_Neighbor[dPPM] = f_Neighbor_dPPM;
      f_Neighbor[dMMP] = f_Neighbor_dMMP;
      f_Neighbor[dPMP] = f_Neighbor_dPMP;
      f_Neighbor[dMPM] = f_Neighbor_dMPM;
      f_Neighbor[dPMM] = f_Neighbor_dPMM;
      f_Neighbor[dMPP] = f_Neighbor_dMPP;

      // write specific directions
      switch (direction)
      {
         case dM00:
            (dist.f[d000])[neighborIndices.k_000] = f_Neighbor[d000];
            (dist.f[d0P0])[neighborIndices.k_000] = f_Neighbor[d0P0];
            (dist.f[d0M0])[neighborIndices.k_0M0] = f_Neighbor[d0M0];
            (dist.f[d00P])[neighborIndices.k_000] = f_Neighbor[d00P];
            (dist.f[d00M])[neighborIndices.k_00M] = f_Neighbor[d00M];
            (dist.f[d0PP])[neighborIndices.k_000] = f_Neighbor[d0PP];
            (dist.f[d0MM])[neighborIndices.k_0MM] = f_Neighbor[d0MM];
            (dist.f[d0PM])[neighborIndices.k_00M] = f_Neighbor[d0PM];
            (dist.f[d0MP])[neighborIndices.k_0M0] = f_Neighbor[d0MP];

            (dist.f[dP00])[neighborIndices.k_000] = f_Neighbor[dP00];
            (dist.f[dPP0])[neighborIndices.k_000] = f_Neighbor[dPP0];
            (dist.f[dPM0])[neighborIndices.k_0M0] = f_Neighbor[dPM0];
            (dist.f[dP0P])[neighborIndices.k_000] = f_Neighbor[dP0P];
            (dist.f[dP0M])[neighborIndices.k_00M] = f_Neighbor[dP0M];
            (dist.f[dPPP])[neighborIndices.k_000] = f_Neighbor[dPPP];
            (dist.f[dPMP])[neighborIndices.k_0M0] = f_Neighbor[dPMP];
            (dist.f[dPPM])[neighborIndices.k_00M] = f_Neighbor[dPPM];
            (dist.f[dPMM])[neighborIndices.k_0MM] = f_Neighbor[dPMM];
            break;
         case dP00:
            (dist.f[d000])[neighborIndices.k_000] = f_Neighbor[d000];
            (dist.f[d0P0])[neighborIndices.k_000] = f_Neighbor[d0P0];
            (dist.f[d0M0])[neighborIndices.k_0M0] = f_Neighbor[d0M0];
            (dist.f[d00P])[neighborIndices.k_000] = f_Neighbor[d00P];
            (dist.f[d00M])[neighborIndices.k_00M] = f_Neighbor[d00M];
            (dist.f[d0PP])[neighborIndices.k_000] = f_Neighbor[d0PP];
            (dist.f[d0MM])[neighborIndices.k_0MM] = f_Neighbor[d0MM];
            (dist.f[d0PM])[neighborIndices.k_00M] = f_Neighbor[d0PM];
            (dist.f[d0MP])[neighborIndices.k_0M0] = f_Neighbor[d0MP];

            (dist.f[dM00])[neighborIndices.k_M00] = f_Neighbor[dM00];
            (dist.f[dMM0])[neighborIndices.k_MM0] = f_Neighbor[dMM0];
            (dist.f[dMP0])[neighborIndices.k_M00] = f_Neighbor[dMP0];
            (dist.f[dM0M])[neighborIndices.k_M0M] = f_Neighbor[dM0M];
            (dist.f[dM0P])[neighborIndices.k_M00] = f_Neighbor[dM0P];
            (dist.f[dMPP])[neighborIndices.k_M00] = f_Neighbor[dMPP];
            (dist.f[dMMP])[neighborIndices.k_MM0] = f_Neighbor[dMMP];
            (dist.f[dMPM])[neighborIndices.k_M0M] = f_Neighbor[dMPM];
            (dist.f[dMMM])[neighborIndices.k_MMM] = f_Neighbor[dMMM];
            break;
         case d0M0:
            (dist.f[d000])[neighborIndices.k_000] = f_Neighbor[d000];
            (dist.f[dP00])[neighborIndices.k_000] = f_Neighbor[dP00];
            (dist.f[dM00])[neighborIndices.k_M00] = f_Neighbor[dM00];
            (dist.f[d00P])[neighborIndices.k_000] = f_Neighbor[d00P];
            (dist.f[d00M])[neighborIndices.k_00M] = f_Neighbor[d00M];
            (dist.f[dP0P])[neighborIndices.k_000] = f_Neighbor[dP0P];
            (dist.f[dM0M])[neighborIndices.k_M0M] = f_Neighbor[dM0M];
            (dist.f[dP0M])[neighborIndices.k_00M] = f_Neighbor[dP0M];
            (dist.f[dM0P])[neighborIndices.k_M00] = f_Neighbor[dM0P];

            (dist.f[d0P0])[neighborIndices.k_000] = f_Neighbor[d0P0];
            (dist.f[dPP0])[neighborIndices.k_000] = f_Neighbor[dPP0];
            (dist.f[dMP0])[neighborIndices.k_M00] = f_Neighbor[dMP0];
            (dist.f[d0PP])[neighborIndices.k_000] = f_Neighbor[d0PP];
            (dist.f[d0PM])[neighborIndices.k_00M] = f_Neighbor[d0PM];
            (dist.f[dPPP])[neighborIndices.k_000] = f_Neighbor[dPPP];
            (dist.f[dMPP])[neighborIndices.k_M00] = f_Neighbor[dMPP];
            (dist.f[dPPM])[neighborIndices.k_00M] = f_Neighbor[dPPM];
            (dist.f[dMPM])[neighborIndices.k_M0M] = f_Neighbor[dMPM];
            break;
         case d0P0:
            (dist.f[d000])[neighborIndices.k_000] = f_Neighbor[d000];
            (dist.f[dP00])[neighborIndices.k_000] = f_Neighbor[dP00];
            (dist.f[dM00])[neighborIndices.k_M00] = f_Neighbor[dM00];
            (dist.f[d00P])[neighborIndices.k_000] = f_Neighbor[d00P];
            (dist.f[d00M])[neighborIndices.k_00M] = f_Neighbor[d00M];
            (dist.f[dP0P])[neighborIndices.k_000] = f_Neighbor[dP0P];
            (dist.f[dM0M])[neighborIndices.k_M0M] = f_Neighbor[dM0M];
            (dist.f[dP0M])[neighborIndices.k_00M] = f_Neighbor[dP0M];
            (dist.f[dM0P])[neighborIndices.k_M00] = f_Neighbor[dM0P];

            (dist.f[d0M0])[neighborIndices.k_0M0] = f_Neighbor[d0M0];
            (dist.f[dMM0])[neighborIndices.k_MM0] = f_Neighbor[dMM0];
            (dist.f[dPM0])[neighborIndices.k_0M0] = f_Neighbor[dPM0];
            (dist.f[d0MM])[neighborIndices.k_0MM] = f_Neighbor[d0MM];
            (dist.f[d0MP])[neighborIndices.k_0M0] = f_Neighbor[d0MP];
            (dist.f[dPMP])[neighborIndices.k_0M0] = f_Neighbor[dPMP];
            (dist.f[dMMP])[neighborIndices.k_MM0] = f_Neighbor[dMMP];
            (dist.f[dPMM])[neighborIndices.k_0MM] = f_Neighbor[dPMM];
            (dist.f[dMMM])[neighborIndices.k_MMM] = f_Neighbor[dMMM];
            break;
         case d00M:
            (dist.f[d000])[neighborIndices.k_000] = f_Neighbor[d000];
            (dist.f[dP00])[neighborIndices.k_000] = f_Neighbor[dP00];
            (dist.f[dM00])[neighborIndices.k_M00] = f_Neighbor[dM00];
            (dist.f[d0P0])[neighborIndices.k_000] = f_Neighbor[d0P0];
            (dist.f[d0M0])[neighborIndices.k_0M0] = f_Neighbor[d0M0];
            (dist.f[dPP0])[neighborIndices.k_000] = f_Neighbor[dPP0];
            (dist.f[dMM0])[neighborIndices.k_MM0] = f_Neighbor[dMM0];
            (dist.f[dPM0])[neighborIndices.k_0M0] = f_Neighbor[dPM0];
            (dist.f[dMP0])[neighborIndices.k_M00] = f_Neighbor[dMP0];

            (dist.f[d00P])[neighborIndices.k_000] = f_Neighbor[d00P];
            (dist.f[dP0P])[neighborIndices.k_000] = f_Neighbor[dP0P];
            (dist.f[dM0P])[neighborIndices.k_M00] = f_Neighbor[dM0P];
            (dist.f[d0PP])[neighborIndices.k_000] = f_Neighbor[d0PP];
            (dist.f[d0MP])[neighborIndices.k_0M0] = f_Neighbor[d0MP];
            (dist.f[dPPP])[neighborIndices.k_000] = f_Neighbor[dPPP];
            (dist.f[dMPP])[neighborIndices.k_M00] = f_Neighbor[dMPP];
            (dist.f[dPMP])[neighborIndices.k_0M0] = f_Neighbor[dPMP];
            (dist.f[dMMP])[neighborIndices.k_MM0] = f_Neighbor[dMMP];
            break;
         case d00P:
            (dist.f[d000])[neighborIndices.k_000] = f_Neighbor[d000];
            (dist.f[dP00])[neighborIndices.k_000] = f_Neighbor[dP00];
            (dist.f[dM00])[neighborIndices.k_M00] = f_Neighbor[dM00];
            (dist.f[d0P0])[neighborIndices.k_000] = f_Neighbor[d0P0];
            (dist.f[d0M0])[neighborIndices.k_0M0] = f_Neighbor[d0M0];
            (dist.f[dPP0])[neighborIndices.k_000] = f_Neighbor[dPP0];
            (dist.f[dMM0])[neighborIndices.k_MM0] = f_Neighbor[dMM0];
            (dist.f[dPM0])[neighborIndices.k_0M0] = f_Neighbor[dPM0];
            (dist.f[dMP0])[neighborIndices.k_M00] = f_Neighbor[dMP0];

            (dist.f[d00M])[neighborIndices.k_00M] = f_Neighbor[d00M];
            (dist.f[dM0M])[neighborIndices.k_M0M] = f_Neighbor[dM0M];
            (dist.f[dP0M])[neighborIndices.k_00M] = f_Neighbor[dP0M];
            (dist.f[d0MM])[neighborIndices.k_0MM] = f_Neighbor[d0MM];
            (dist.f[d0PM])[neighborIndices.k_00M] = f_Neighbor[d0PM];
            (dist.f[dPPM])[neighborIndices.k_00M] = f_Neighbor[dPPM];
            (dist.f[dMPM])[neighborIndices.k_M0M] = f_Neighbor[dMPM];
            (dist.f[dPMM])[neighborIndices.k_0MM] = f_Neighbor[dPMM];
            (dist.f[dMMM])[neighborIndices.k_MMM] = f_Neighbor[dMMM];
            break;
         default:
            break; 
      }
   }
}


//! \}
