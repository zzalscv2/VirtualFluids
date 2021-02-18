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
//! \file VelocityBCAlgorithmMultiphase.cpp
//! \ingroup BoundarConditions
//! \author Hesameddin Safari
//=======================================================================================

#include "VelocityBCAlgorithmMultiphase.h"
#include "DistributionArray3D.h"
#include "BoundaryConditions.h"

VelocityBCAlgorithmMultiphase::VelocityBCAlgorithmMultiphase()
{
   BCAlgorithm::type = BCAlgorithm::VelocityBCAlgorithm;
   BCAlgorithm::preCollision = false;
}
//////////////////////////////////////////////////////////////////////////
VelocityBCAlgorithmMultiphase::~VelocityBCAlgorithmMultiphase()
{
}
//////////////////////////////////////////////////////////////////////////
SPtr<BCAlgorithm> VelocityBCAlgorithmMultiphase::clone()
{
   SPtr<BCAlgorithm> bc(new VelocityBCAlgorithmMultiphase());
   return bc;
}
//////////////////////////////////////////////////////////////////////////
void VelocityBCAlgorithmMultiphase::addDistributions(SPtr<DistributionArray3D> distributions)
{
   this->distributions = distributions;
}
//////////////////////////////////////////////////////////////////////////
void VelocityBCAlgorithmMultiphase::addDistributionsH(SPtr<DistributionArray3D> distributionsH)
{
	this->distributionsH = distributionsH;
}
//////////////////////////////////////////////////////////////////////////
void VelocityBCAlgorithmMultiphase::applyBC()
{
   LBMReal f[D3Q27System::ENDF+1];
   LBMReal h[D3Q27System::ENDF+1];
   LBMReal feq[D3Q27System::ENDF+1];
   LBMReal heq[D3Q27System::ENDF+1];
   LBMReal htemp[D3Q27System::ENDF+1];
   
   distributions->getDistributionInv(f, x1, x2, x3);
   distributionsH->getDistributionInv(h, x1, x2, x3);
   LBMReal phi, rho, vx1, vx2, vx3, p1, phiBC;
   
   D3Q27System::calcDensity(h, phi);
   
   //LBMReal collFactorM = phi*collFactorL + (1-phi)*collFactorG;
   //LBMReal collFactorM = collFactorL + (collFactorL - collFactorG)*(phi - phiH)/(phiH - phiL);

   

   //rho = phi + (1.0 - phi)*1.0/densityRatio;
   LBMReal rhoH = 1.0;
   LBMReal rhoL = 1.0/densityRatio;
   rho = rhoH + (rhoH - rhoL)*(phi - phiH)/(phiH - phiL);
   

   calcMacrosFct(f, p1, vx1, vx2, vx3);
   /*vx1/=(rho*c1o3);
   vx2/=(rho*c1o3);
   vx3/=(rho*c1o3);*/

   //D3Q27System::calcMultiphaseFeq(feq, rho, p1, vx1, vx2, vx3);
   D3Q27System::calcMultiphaseFeqVB(feq, p1, vx1, vx2, vx3);
   D3Q27System::calcMultiphaseHeq(heq, phi, vx1, vx2, vx3);

   ///// added for phase field //////

   int nx1 = x1;
   int nx2 = x2;
   int nx3 = x3;
   int direction = -1;
   //flag points in direction of fluid
   if      (bcPtr->hasVelocityBoundaryFlag(D3Q27System::E)) { nx1 -= 1; direction = D3Q27System::E; }
   else if (bcPtr->hasVelocityBoundaryFlag(D3Q27System::W)) { nx1 += 1; direction = D3Q27System::W; }
   else if (bcPtr->hasVelocityBoundaryFlag(D3Q27System::N)) { nx2 -= 1; direction = D3Q27System::N; }
   else if (bcPtr->hasVelocityBoundaryFlag(D3Q27System::S)) { nx2 += 1; direction = D3Q27System::S; }
   else if (bcPtr->hasVelocityBoundaryFlag(D3Q27System::T)) { nx3 -= 1; direction = D3Q27System::T; }
   else if (bcPtr->hasVelocityBoundaryFlag(D3Q27System::B)) { nx3 += 1; direction = D3Q27System::B; }
   else UB_THROW(UbException(UB_EXARGS, "Danger...no orthogonal BC-Flag on velocity boundary..."));
   
   phiBC = bcPtr->getBoundaryPhaseField();
   
   D3Q27System::calcMultiphaseHeq(htemp, phiBC, vx1, vx2, vx3);

   for (int fdir = D3Q27System::STARTF; fdir<=D3Q27System::ENDF; fdir++)
   {
	   if (bcPtr->hasVelocityBoundaryFlag(fdir))
	   {
		   LBMReal hReturn = htemp[fdir]+h[fdir]-heq[fdir];
		   distributionsH->setDistributionForDirection(hReturn, nx1, nx2, nx3, fdir);
	   }
   }

   //////////////////////////////////



   
   for (int fdir = D3Q27System::FSTARTDIR; fdir<=D3Q27System::FENDDIR; fdir++)
   {
      if (bcPtr->hasVelocityBoundaryFlag(fdir))
      {
         const int invDir = D3Q27System::INVDIR[fdir];
         LBMReal q = bcPtr->getQ(invDir);// m+m q=0 stabiler
         LBMReal velocity = bcPtr->getBoundaryVelocity(invDir);
         //LBMReal fReturn = ((1.0-q)/(1.0+q))*((f[invDir]-feq[invDir])/(1.0-collFactor)+feq[invDir])+((q*(f[invDir]+f[fdir])-velocity*rho*c1o3)/(1.0+q));
		 LBMReal fReturn = ((1.0-q)/(1.0+q))*((f[invDir]-feq[invDir])/(1.0-collFactor)+feq[invDir])+((q*(f[invDir]+f[fdir])-velocity)/(1.0+q));
         distributions->setDistributionForDirection(fReturn, x1+D3Q27System::DX1[invDir], x2+D3Q27System::DX2[invDir], x3+D3Q27System::DX3[invDir], fdir);

		 //LBMReal hReturn = ((1.0-q)/(1.0+q))*((h[invDir]-heq[invDir])/(1.0-collFactorM)+heq[invDir])+((q/(1.0+q))*(h[invDir]+h[fdir]));
		 //distributionsH->setDistributionForDirection(hReturn, x1+D3Q27System::DX1[invDir], x2+D3Q27System::DX2[invDir], x3+D3Q27System::DX3[invDir], fdir);
      }
   }

}

