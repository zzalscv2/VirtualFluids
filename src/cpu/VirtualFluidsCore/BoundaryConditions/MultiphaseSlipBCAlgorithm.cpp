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
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with VirtualFluids (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file MultiphaseSlipBCAlgorithm.cpp
//! \ingroup BoundarConditions
//! \author Hesameddin Safari
//=======================================================================================

#include "MultiphaseSlipBCAlgorithm.h"
#include "DistributionArray3D.h"
#include "BoundaryConditions.h"

MultiphaseSlipBCAlgorithm::MultiphaseSlipBCAlgorithm()
{
   BCAlgorithm::type = BCAlgorithm::SlipBCAlgorithm;
   BCAlgorithm::preCollision = false;
}
//////////////////////////////////////////////////////////////////////////
MultiphaseSlipBCAlgorithm::~MultiphaseSlipBCAlgorithm()
{

}
//////////////////////////////////////////////////////////////////////////
SPtr<BCAlgorithm> MultiphaseSlipBCAlgorithm::clone()
{
   SPtr<BCAlgorithm> bc(new MultiphaseSlipBCAlgorithm());
   return bc;
}
//////////////////////////////////////////////////////////////////////////
void MultiphaseSlipBCAlgorithm::addDistributions(SPtr<DistributionArray3D> distributions)
{
   this->distributions = distributions;
}
//////////////////////////////////////////////////////////////////////////
void MultiphaseSlipBCAlgorithm::addDistributionsH(SPtr<DistributionArray3D> distributionsH)
{
	this->distributionsH = distributionsH;
}
//////////////////////////////////////////////////////////////////////////
void MultiphaseSlipBCAlgorithm::applyBC()
{
   LBMReal f[D3Q27System::ENDF+1];
   LBMReal h[D3Q27System::ENDF+1];
   LBMReal feq[D3Q27System::ENDF+1];
   LBMReal heq[D3Q27System::ENDF+1];
   distributions->getDistributionInv(f, x1, x2, x3);
   distributionsH->getDistributionInv(h, x1, x2, x3);

   LBMReal p1, vx1, vx2, vx3, phi, rho;

   D3Q27System::calcDensity(h, phi);
   //LBMReal collFactorM = collFactorL + (collFactorL - collFactorG)*(phi - phiH)/(phiH - phiL);


   calcMacrosFct(f, p1, vx1, vx2, vx3);
   D3Q27System::calcMultiphaseFeqVB(feq, p1, vx1, vx2, vx3);
   D3Q27System::calcMultiphaseHeq(heq, phi, vx1, vx2, vx3); 

   UbTupleFloat3 normale = bcPtr->getNormalVector();
   LBMReal amp = vx1*val<1>(normale)+vx2*val<2>(normale)+vx3*val<3>(normale);

   vx1 = vx1 - amp * val<1>(normale); //normale zeigt von struktur weg!
   vx2 = vx2 - amp * val<2>(normale); //normale zeigt von struktur weg!
   vx3 = vx3 - amp * val<3>(normale); //normale zeigt von struktur weg!

   //rho = 1.0+drho*compressibleFactor;
   rho = 1.0; // In multiphase model set to 1.0!

   for (int fdir = D3Q27System::FSTARTDIR; fdir<=D3Q27System::FENDDIR; fdir++)
   {
      if (bcPtr->hasSlipBoundaryFlag(fdir))
      {
         //quadratic bounce back
         const int invDir = D3Q27System::INVDIR[fdir];
         LBMReal q = bcPtr->getQ(invDir);// m+m q=0 stabiler
         //vx3=0;
         LBMReal velocity = 0.0;
         switch (invDir)
         {
         case D3Q27System::DIR_P00: velocity = (UbMath::c4o9*(+vx1)); break;      //(2/cs^2)(=6)*rho_0(=1 bei imkompr)*wi*u*ei mit cs=1/sqrt(3)
         case D3Q27System::DIR_M00: velocity = (UbMath::c4o9*(-vx1)); break;      //z.B. aus paper manfred MRT LB models in three dimensions (2002)   
         case D3Q27System::DIR_0P0: velocity = (UbMath::c4o9*(+vx2)); break;
         case D3Q27System::DIR_0M0: velocity = (UbMath::c4o9*(-vx2)); break;
         case D3Q27System::DIR_00P: velocity = (UbMath::c4o9*(+vx3)); break;
         case D3Q27System::DIR_00M: velocity = (UbMath::c4o9*(-vx3)); break;
         case D3Q27System::DIR_PP0: velocity = (UbMath::c1o9*(+vx1+vx2)); break;
         case D3Q27System::DIR_MM0: velocity = (UbMath::c1o9*(-vx1-vx2)); break;
         case D3Q27System::DIR_PM0: velocity = (UbMath::c1o9*(+vx1-vx2)); break;
         case D3Q27System::DIR_MP0: velocity = (UbMath::c1o9*(-vx1+vx2)); break;
         case D3Q27System::DIR_P0P: velocity = (UbMath::c1o9*(+vx1             +vx3)); break;
         case D3Q27System::DIR_M0M: velocity = (UbMath::c1o9*(-vx1             -vx3)); break;
         case D3Q27System::DIR_P0M: velocity = (UbMath::c1o9*(+vx1             -vx3)); break;
         case D3Q27System::DIR_M0P: velocity = (UbMath::c1o9*(-vx1             +vx3)); break;
         case D3Q27System::DIR_0PP: velocity = (UbMath::c1o9*(+vx2+vx3)); break;
         case D3Q27System::DIR_0MM: velocity = (UbMath::c1o9*(-vx2-vx3)); break;
         case D3Q27System::DIR_0PM: velocity = (UbMath::c1o9*(+vx2-vx3)); break;
         case D3Q27System::DIR_0MP: velocity = (UbMath::c1o9*(-vx2+vx3)); break;
         case D3Q27System::DIR_PPP: velocity = (UbMath::c1o36*(+vx1+vx2+vx3)); break;
         case D3Q27System::DIR_MMM: velocity = (UbMath::c1o36*(-vx1-vx2-vx3)); break;
         case D3Q27System::DIR_PPM: velocity = (UbMath::c1o36*(+vx1+vx2-vx3)); break;
         case D3Q27System::DIR_MMP: velocity = (UbMath::c1o36*(-vx1-vx2+vx3)); break;
         case D3Q27System::DIR_PMP: velocity = (UbMath::c1o36*(+vx1-vx2+vx3)); break;
         case D3Q27System::DIR_MPM: velocity = (UbMath::c1o36*(-vx1+vx2-vx3)); break;
         case D3Q27System::DIR_PMM: velocity = (UbMath::c1o36*(+vx1-vx2-vx3)); break;
         case D3Q27System::DIR_MPP: velocity = (UbMath::c1o36*(-vx1+vx2+vx3)); break;
         default: throw UbException(UB_EXARGS, "unknown error");
         }
         LBMReal fReturn = ((1.0-q)/(1.0+q))*((f[invDir]-feq[invDir])/(1.0-collFactor)+feq[invDir])+((q*(f[invDir]+f[fdir])-velocity*rho)/(1.0+q));
         distributions->setDistributionForDirection(fReturn, x1+D3Q27System::DX1[invDir], x2+D3Q27System::DX2[invDir], x3+D3Q27System::DX3[invDir], fdir);

		 //LBMReal hReturn = ((1.0-q)/(1.0+q))*((h[invDir]-heq[invDir])/(1.0-collFactorPh)+heq[invDir])+((q/(1.0+q))*(h[invDir]+h[fdir]));
		 LBMReal hReturn = h[invDir];
		 distributionsH->setDistributionForDirection(hReturn, x1+D3Q27System::DX1[invDir], x2+D3Q27System::DX2[invDir], x3+D3Q27System::DX3[invDir], fdir);
      }
   }
}