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
//! \file RheologyInterpolationProcessor.cpp
//! \ingroup LBM
//! \author Konstantin Kutscher
//=======================================================================================

#include "RheologyInterpolationProcessor.h"
#include "D3Q27System.h"
#include "Rheology.h"


RheologyInterpolationProcessor::RheologyInterpolationProcessor()
   : omegaC(0.0), omegaF(0.0), omegaMin(0.0)
{

}
//////////////////////////////////////////////////////////////////////////
RheologyInterpolationProcessor::RheologyInterpolationProcessor(LBMReal omegaC, LBMReal omegaF, LBMReal omegaMin)
   : omegaC(omegaC), omegaF(omegaF), omegaMin(omegaMin)
{

}
//////////////////////////////////////////////////////////////////////////
RheologyInterpolationProcessor::~RheologyInterpolationProcessor()
{

}
//////////////////////////////////////////////////////////////////////////
InterpolationProcessorPtr RheologyInterpolationProcessor::clone()
{
   InterpolationProcessorPtr iproc = InterpolationProcessorPtr (new RheologyInterpolationProcessor(this->omegaC, this->omegaF, this->omegaMin));
   return iproc;
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::setOmegas( LBMReal omegaC, LBMReal omegaF )
{
   this->omegaC = omegaC;
   this->omegaF = omegaF;
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::setOmegaMin( LBMReal omegaMin )
{
   this->omegaMin = omegaMin;
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::setOffsets(LBMReal xoff, LBMReal yoff, LBMReal zoff)
{
   this->xoff = xoff;
   this->yoff = yoff;
   this->zoff = zoff;     
   this->xoff_sq = xoff * xoff;
   this->yoff_sq = yoff * yoff;
   this->zoff_sq = zoff * zoff;
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::interpolateCoarseToFine(D3Q27ICell& icellC, D3Q27ICell& icellF, LBMReal xoff, LBMReal yoff, LBMReal zoff)
{
    setOffsets(xoff, yoff, zoff);
    calcInterpolatedCoefficiets_intern(icellC, omegaC, 0.5, 0.25, -0.25, -0.25, -1, -1, -1);
    calcInterpolatedNode(icellF.BSW, /*omegaF,*/ -0.25, -0.25, -0.25, calcPressBSW(), -1, -1, -1);
    calcInterpolatedCoefficiets_intern(icellC, omegaC, 0.5, 0.25, 0.25, -0.25, 1, 1, -1);
    calcInterpolatedNode(icellF.BNE, /*omegaF,*/  0.25,  0.25, -0.25, calcPressBNE(),  1,  1, -1);
    calcInterpolatedCoefficiets_intern(icellC, omegaC, 0.5, -0.25, 0.25, 0.25, -1, 1, 1);
    calcInterpolatedNode(icellF.TNW, /*omegaF,*/ -0.25,  0.25,  0.25, calcPressTNW(), -1,  1,  1);
    calcInterpolatedCoefficiets_intern(icellC, omegaC, 0.5, 0.25, -0.25, 0.25, 1, -1, 1);
    calcInterpolatedNode(icellF.TSE, /*omegaF,*/  0.25, -0.25,  0.25, calcPressTSE(),  1, -1,  1);
    calcInterpolatedCoefficiets_intern(icellC, omegaC, 0.5, -0.25, 0.25, -0.25, -1, 1, -1);
    calcInterpolatedNode(icellF.BNW, /*omegaF,*/ -0.25,  0.25, -0.25, calcPressBNW(), -1,  1, -1);
    calcInterpolatedCoefficiets_intern(icellC, omegaC, 0.5, 0.25, -0.25, -0.25, 1, -1, -1);
    calcInterpolatedNode(icellF.BSE, /*omegaF,*/  0.25, -0.25, -0.25, calcPressBSE(),  1, -1, -1);
    calcInterpolatedCoefficiets_intern(icellC, omegaC, 0.5, -0.25, -0.25, 0.25, -1, -1, 1);
    calcInterpolatedNode(icellF.TSW, /*omegaF,*/ -0.25, -0.25,  0.25, calcPressTSW(), -1, -1,  1);
    calcInterpolatedCoefficiets_intern(icellC, omegaC, 0.5, 0.25, 0.25, 0.25, 1, 1, 1);
    calcInterpolatedNode(icellF.TNE, /*omegaF,*/  0.25,  0.25,  0.25, calcPressTNE(),  1,  1,  1);
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::interpolateFineToCoarse(D3Q27ICell& icellF, LBMReal* icellC, LBMReal xoff, LBMReal yoff, LBMReal zoff)
{
   setOffsets(xoff, yoff, zoff);
    calcInterpolatedCoefficiets_intern(icellF, omegaF, 2.0, 0, 0, 0, 0, 0, 0);
   calcInterpolatedNodeFC(icellC, omegaC);
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::calcMoments(const LBMReal* const f, LBMReal omegaInf, LBMReal& press, LBMReal& vx1, LBMReal& vx2, LBMReal& vx3, LBMReal& kxy, LBMReal& kyz, LBMReal& kxz, LBMReal& kxxMyy, LBMReal& kxxMzz)
{
   using namespace D3Q27System;
   using namespace vf::lbm::dir;

   rho = 0.0;
   D3Q27System::calcIncompMacroscopicValues(f,rho,vx1,vx2,vx3);

   shearRate = D3Q27System::getShearRate(f, omegaInf);

   LBMReal omega = Rheology::getHerschelBulkleyCollFactor(omegaInf, shearRate, rho);

   press = rho; //interpolate rho!

   kxy   = -3.*omega*((((f[DIR_MMP]+f[DIR_PPM])-(f[DIR_MPP]+f[DIR_PMM]))+((f[DIR_MMM]+f[DIR_PPP])-(f[DIR_MPM]+f[DIR_PMP])))+((f[DIR_MM0]+f[DIR_PP0])-(f[DIR_MP0]+f[DIR_PM0]))-(vx1*vx2));// might not be optimal MG 25.2.13
   kyz   = -3.*omega*((((f[DIR_MMM]+f[DIR_PPP])-(f[DIR_PMP]+f[DIR_MPM]))+((f[DIR_PMM]+f[DIR_MPP])-(f[DIR_MMP]+f[DIR_PPM])))+((f[DIR_0MM]+f[DIR_0PP])-(f[DIR_0MP]+f[DIR_0PM]))-(vx2*vx3));
   kxz   = -3.*omega*((((f[DIR_MPM]+f[DIR_PMP])-(f[DIR_MMP]+f[DIR_PPM]))+((f[DIR_MMM]+f[DIR_PPP])-(f[DIR_PMM]+f[DIR_MPP])))+((f[DIR_M0M]+f[DIR_P0P])-(f[DIR_M0P]+f[DIR_P0M]))-(vx1*vx3));
   kxxMyy = -3./2.*omega*((((f[DIR_M0M]+f[DIR_P0P])-(f[DIR_0MM]+f[DIR_0PP]))+((f[DIR_M0P]+f[DIR_P0M])-(f[DIR_0MP]+f[DIR_0PM])))+((f[DIR_M00]+f[DIR_P00])-(f[DIR_0M0]+f[DIR_0P0]))-(vx1*vx1-vx2*vx2));
   kxxMzz = -3./2.*omega*((((f[DIR_MP0]+f[DIR_PM0])-(f[DIR_0MM]+f[DIR_0PP]))+((f[DIR_MM0]+f[DIR_PP0])-(f[DIR_0MP]+f[DIR_0PM])))+((f[DIR_M00]+f[DIR_P00])-(f[DIR_00M]+f[DIR_00P]))-(vx1*vx1-vx3*vx3));
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::calcInterpolatedCoefficiets_intern(const D3Q27ICell& icell,
                                                                          LBMReal omega,
                                                                          LBMReal eps_new,
                                                                          LBMReal x,
                                                                          LBMReal y,
                                                                          LBMReal z,
                                                                          LBMReal xs,
                                                                          LBMReal ys,
                                                                          LBMReal zs)
{
   LBMReal        vx1_SWT,vx2_SWT,vx3_SWT;
   LBMReal        vx1_NWT,vx2_NWT,vx3_NWT;
   LBMReal        vx1_NET,vx2_NET,vx3_NET;
   LBMReal        vx1_SET,vx2_SET,vx3_SET;
   LBMReal        vx1_SWB,vx2_SWB,vx3_SWB;
   LBMReal        vx1_NWB,vx2_NWB,vx3_NWB;
   LBMReal        vx1_NEB,vx2_NEB,vx3_NEB;
   LBMReal        vx1_SEB,vx2_SEB,vx3_SEB;

   LBMReal        kxyFromfcNEQ_SWT, kyzFromfcNEQ_SWT, kxzFromfcNEQ_SWT, kxxMyyFromfcNEQ_SWT, kxxMzzFromfcNEQ_SWT;
   LBMReal        kxyFromfcNEQ_NWT, kyzFromfcNEQ_NWT, kxzFromfcNEQ_NWT, kxxMyyFromfcNEQ_NWT, kxxMzzFromfcNEQ_NWT;
   LBMReal        kxyFromfcNEQ_NET, kyzFromfcNEQ_NET, kxzFromfcNEQ_NET, kxxMyyFromfcNEQ_NET, kxxMzzFromfcNEQ_NET;
   LBMReal        kxyFromfcNEQ_SET, kyzFromfcNEQ_SET, kxzFromfcNEQ_SET, kxxMyyFromfcNEQ_SET, kxxMzzFromfcNEQ_SET;
   LBMReal        kxyFromfcNEQ_SWB, kyzFromfcNEQ_SWB, kxzFromfcNEQ_SWB, kxxMyyFromfcNEQ_SWB, kxxMzzFromfcNEQ_SWB;
   LBMReal        kxyFromfcNEQ_NWB, kyzFromfcNEQ_NWB, kxzFromfcNEQ_NWB, kxxMyyFromfcNEQ_NWB, kxxMzzFromfcNEQ_NWB;
   LBMReal        kxyFromfcNEQ_NEB, kyzFromfcNEQ_NEB, kxzFromfcNEQ_NEB, kxxMyyFromfcNEQ_NEB, kxxMzzFromfcNEQ_NEB;
   LBMReal        kxyFromfcNEQ_SEB, kyzFromfcNEQ_SEB, kxzFromfcNEQ_SEB, kxxMyyFromfcNEQ_SEB, kxxMzzFromfcNEQ_SEB;

   calcMoments(icell.TSW,omega,press_SWT,vx1_SWT,vx2_SWT,vx3_SWT, kxyFromfcNEQ_SWT, kyzFromfcNEQ_SWT, kxzFromfcNEQ_SWT, kxxMyyFromfcNEQ_SWT, kxxMzzFromfcNEQ_SWT);
   calcMoments(icell.TNW,omega,press_NWT,vx1_NWT,vx2_NWT,vx3_NWT, kxyFromfcNEQ_NWT, kyzFromfcNEQ_NWT, kxzFromfcNEQ_NWT, kxxMyyFromfcNEQ_NWT, kxxMzzFromfcNEQ_NWT);
   calcMoments(icell.TNE,omega,press_NET,vx1_NET,vx2_NET,vx3_NET, kxyFromfcNEQ_NET, kyzFromfcNEQ_NET, kxzFromfcNEQ_NET, kxxMyyFromfcNEQ_NET, kxxMzzFromfcNEQ_NET);
   calcMoments(icell.TSE,omega,press_SET,vx1_SET,vx2_SET,vx3_SET, kxyFromfcNEQ_SET, kyzFromfcNEQ_SET, kxzFromfcNEQ_SET, kxxMyyFromfcNEQ_SET, kxxMzzFromfcNEQ_SET);
   calcMoments(icell.BSW,omega,press_SWB,vx1_SWB,vx2_SWB,vx3_SWB, kxyFromfcNEQ_SWB, kyzFromfcNEQ_SWB, kxzFromfcNEQ_SWB, kxxMyyFromfcNEQ_SWB, kxxMzzFromfcNEQ_SWB);
   calcMoments(icell.BNW,omega,press_NWB,vx1_NWB,vx2_NWB,vx3_NWB, kxyFromfcNEQ_NWB, kyzFromfcNEQ_NWB, kxzFromfcNEQ_NWB, kxxMyyFromfcNEQ_NWB, kxxMzzFromfcNEQ_NWB);
   calcMoments(icell.BNE,omega,press_NEB,vx1_NEB,vx2_NEB,vx3_NEB, kxyFromfcNEQ_NEB, kyzFromfcNEQ_NEB, kxzFromfcNEQ_NEB, kxxMyyFromfcNEQ_NEB, kxxMzzFromfcNEQ_NEB);
   calcMoments(icell.BSE,omega,press_SEB,vx1_SEB,vx2_SEB,vx3_SEB, kxyFromfcNEQ_SEB, kyzFromfcNEQ_SEB, kxzFromfcNEQ_SEB, kxxMyyFromfcNEQ_SEB, kxxMzzFromfcNEQ_SEB);

   a0 = (-kxxMyyFromfcNEQ_NEB - kxxMyyFromfcNEQ_NET + kxxMyyFromfcNEQ_NWB + kxxMyyFromfcNEQ_NWT -
      kxxMyyFromfcNEQ_SEB - kxxMyyFromfcNEQ_SET + kxxMyyFromfcNEQ_SWB + kxxMyyFromfcNEQ_SWT -
      kxxMzzFromfcNEQ_NEB - kxxMzzFromfcNEQ_NET + kxxMzzFromfcNEQ_NWB + kxxMzzFromfcNEQ_NWT -
      kxxMzzFromfcNEQ_SEB - kxxMzzFromfcNEQ_SET + kxxMzzFromfcNEQ_SWB + kxxMzzFromfcNEQ_SWT -
      2.*kxyFromfcNEQ_NEB - 2.*kxyFromfcNEQ_NET - 2.*kxyFromfcNEQ_NWB - 2.*kxyFromfcNEQ_NWT +
      2.*kxyFromfcNEQ_SEB + 2.*kxyFromfcNEQ_SET + 2.*kxyFromfcNEQ_SWB + 2.*kxyFromfcNEQ_SWT +
      2.*kxzFromfcNEQ_NEB - 2.*kxzFromfcNEQ_NET + 2.*kxzFromfcNEQ_NWB - 2.*kxzFromfcNEQ_NWT +
      2.*kxzFromfcNEQ_SEB - 2.*kxzFromfcNEQ_SET + 2.*kxzFromfcNEQ_SWB - 2.*kxzFromfcNEQ_SWT +
      8.*vx1_NEB + 8.*vx1_NET + 8.*vx1_NWB + 8.*vx1_NWT + 8.*vx1_SEB +
      8.*vx1_SET + 8.*vx1_SWB + 8.*vx1_SWT + 2.*vx2_NEB + 2.*vx2_NET -
      2.*vx2_NWB - 2.*vx2_NWT - 2.*vx2_SEB - 2.*vx2_SET + 2.*vx2_SWB +
      2.*vx2_SWT - 2.*vx3_NEB + 2.*vx3_NET + 2.*vx3_NWB - 2.*vx3_NWT -
      2.*vx3_SEB + 2.*vx3_SET + 2.*vx3_SWB - 2.*vx3_SWT)/64.;
   b0 = (2.*kxxMyyFromfcNEQ_NEB + 2.*kxxMyyFromfcNEQ_NET + 2.*kxxMyyFromfcNEQ_NWB + 2.*kxxMyyFromfcNEQ_NWT -
      2.*kxxMyyFromfcNEQ_SEB - 2.*kxxMyyFromfcNEQ_SET - 2.*kxxMyyFromfcNEQ_SWB - 2.*kxxMyyFromfcNEQ_SWT -
      kxxMzzFromfcNEQ_NEB - kxxMzzFromfcNEQ_NET - kxxMzzFromfcNEQ_NWB - kxxMzzFromfcNEQ_NWT +
      kxxMzzFromfcNEQ_SEB + kxxMzzFromfcNEQ_SET + kxxMzzFromfcNEQ_SWB + kxxMzzFromfcNEQ_SWT -
      2.*kxyFromfcNEQ_NEB - 2.*kxyFromfcNEQ_NET + 2.*kxyFromfcNEQ_NWB + 2.*kxyFromfcNEQ_NWT -
      2.*kxyFromfcNEQ_SEB - 2.*kxyFromfcNEQ_SET + 2.*kxyFromfcNEQ_SWB + 2.*kxyFromfcNEQ_SWT +
      2.*kyzFromfcNEQ_NEB - 2.*kyzFromfcNEQ_NET + 2.*kyzFromfcNEQ_NWB - 2.*kyzFromfcNEQ_NWT +
      2.*kyzFromfcNEQ_SEB - 2.*kyzFromfcNEQ_SET + 2.*kyzFromfcNEQ_SWB - 2.*kyzFromfcNEQ_SWT +
      2.*vx1_NEB + 2.*vx1_NET - 2.*vx1_NWB - 2.*vx1_NWT -
      2.*vx1_SEB - 2.*vx1_SET + 2.*vx1_SWB + 2.*vx1_SWT +
      8.*vx2_NEB + 8.*vx2_NET + 8.*vx2_NWB + 8.*vx2_NWT +
      8.*vx2_SEB + 8.*vx2_SET + 8.*vx2_SWB + 8.*vx2_SWT -
      2.*vx3_NEB + 2.*vx3_NET - 2.*vx3_NWB + 2.*vx3_NWT +
      2.*vx3_SEB - 2.*vx3_SET + 2.*vx3_SWB - 2.*vx3_SWT)/64.;
   c0 = (kxxMyyFromfcNEQ_NEB - kxxMyyFromfcNEQ_NET + kxxMyyFromfcNEQ_NWB - kxxMyyFromfcNEQ_NWT +
      kxxMyyFromfcNEQ_SEB - kxxMyyFromfcNEQ_SET + kxxMyyFromfcNEQ_SWB - kxxMyyFromfcNEQ_SWT -
      2.*kxxMzzFromfcNEQ_NEB + 2.*kxxMzzFromfcNEQ_NET - 2.*kxxMzzFromfcNEQ_NWB + 2.*kxxMzzFromfcNEQ_NWT -
      2.*kxxMzzFromfcNEQ_SEB + 2.*kxxMzzFromfcNEQ_SET - 2.*kxxMzzFromfcNEQ_SWB + 2.*kxxMzzFromfcNEQ_SWT -
      2.*kxzFromfcNEQ_NEB - 2.*kxzFromfcNEQ_NET + 2.*kxzFromfcNEQ_NWB + 2.*kxzFromfcNEQ_NWT -
      2.*kxzFromfcNEQ_SEB - 2.*kxzFromfcNEQ_SET + 2.*kxzFromfcNEQ_SWB + 2.*kxzFromfcNEQ_SWT -
      2.*kyzFromfcNEQ_NEB - 2.*kyzFromfcNEQ_NET - 2.*kyzFromfcNEQ_NWB - 2.*kyzFromfcNEQ_NWT +
      2.*kyzFromfcNEQ_SEB + 2.*kyzFromfcNEQ_SET + 2.*kyzFromfcNEQ_SWB + 2.*kyzFromfcNEQ_SWT -
      2.*vx1_NEB + 2.*vx1_NET + 2.*vx1_NWB - 2.*vx1_NWT -
      2.*vx1_SEB + 2.*vx1_SET + 2.*vx1_SWB - 2.*vx1_SWT -
      2.*vx2_NEB + 2.*vx2_NET - 2.*vx2_NWB + 2.*vx2_NWT +
      2.*vx2_SEB - 2.*vx2_SET + 2.*vx2_SWB - 2.*vx2_SWT +
      8.*vx3_NEB + 8.*vx3_NET + 8.*vx3_NWB + 8.*vx3_NWT +
      8.*vx3_SEB + 8.*vx3_SET + 8.*vx3_SWB + 8.*vx3_SWT)/64.;
   ax = (vx1_NEB + vx1_NET - vx1_NWB - vx1_NWT + vx1_SEB + vx1_SET - vx1_SWB - vx1_SWT)/4.;
   bx = (vx2_NEB + vx2_NET - vx2_NWB - vx2_NWT + vx2_SEB + vx2_SET - vx2_SWB - vx2_SWT)/4.;
   cx = (vx3_NEB + vx3_NET - vx3_NWB - vx3_NWT + vx3_SEB + vx3_SET - vx3_SWB - vx3_SWT)/4.;
   axx= (kxxMyyFromfcNEQ_NEB + kxxMyyFromfcNEQ_NET - kxxMyyFromfcNEQ_NWB - kxxMyyFromfcNEQ_NWT +
      kxxMyyFromfcNEQ_SEB + kxxMyyFromfcNEQ_SET - kxxMyyFromfcNEQ_SWB - kxxMyyFromfcNEQ_SWT +
      kxxMzzFromfcNEQ_NEB + kxxMzzFromfcNEQ_NET - kxxMzzFromfcNEQ_NWB - kxxMzzFromfcNEQ_NWT +
      kxxMzzFromfcNEQ_SEB + kxxMzzFromfcNEQ_SET - kxxMzzFromfcNEQ_SWB - kxxMzzFromfcNEQ_SWT +
      2.*vx2_NEB + 2.*vx2_NET - 2.*vx2_NWB - 2.*vx2_NWT -
      2.*vx2_SEB - 2.*vx2_SET + 2.*vx2_SWB + 2.*vx2_SWT -
      2.*vx3_NEB + 2.*vx3_NET + 2.*vx3_NWB - 2.*vx3_NWT -
      2.*vx3_SEB + 2.*vx3_SET + 2.*vx3_SWB - 2.*vx3_SWT)/16.;
   bxx= (kxyFromfcNEQ_NEB + kxyFromfcNEQ_NET - kxyFromfcNEQ_NWB - kxyFromfcNEQ_NWT +
      kxyFromfcNEQ_SEB + kxyFromfcNEQ_SET - kxyFromfcNEQ_SWB - kxyFromfcNEQ_SWT -
      2.*vx1_NEB - 2.*vx1_NET + 2.*vx1_NWB + 2.*vx1_NWT +
      2.*vx1_SEB + 2.*vx1_SET - 2.*vx1_SWB - 2.*vx1_SWT)/8.;
   cxx= (kxzFromfcNEQ_NEB + kxzFromfcNEQ_NET - kxzFromfcNEQ_NWB - kxzFromfcNEQ_NWT +
      kxzFromfcNEQ_SEB + kxzFromfcNEQ_SET - kxzFromfcNEQ_SWB - kxzFromfcNEQ_SWT +
      2.*vx1_NEB - 2.*vx1_NET - 2.*vx1_NWB + 2.*vx1_NWT +
      2.*vx1_SEB - 2.*vx1_SET - 2.*vx1_SWB + 2.*vx1_SWT)/8.;
   ay = (vx1_NEB + vx1_NET + vx1_NWB + vx1_NWT - vx1_SEB - vx1_SET - vx1_SWB - vx1_SWT)/4.;
   by = (vx2_NEB + vx2_NET + vx2_NWB + vx2_NWT - vx2_SEB - vx2_SET - vx2_SWB - vx2_SWT)/4.;
   cy = (vx3_NEB + vx3_NET + vx3_NWB + vx3_NWT - vx3_SEB - vx3_SET - vx3_SWB - vx3_SWT)/4.;
   ayy= (kxyFromfcNEQ_NEB + kxyFromfcNEQ_NET + kxyFromfcNEQ_NWB + kxyFromfcNEQ_NWT -
      kxyFromfcNEQ_SEB - kxyFromfcNEQ_SET - kxyFromfcNEQ_SWB - kxyFromfcNEQ_SWT -
      2.*vx2_NEB - 2.*vx2_NET + 2.*vx2_NWB + 2.*vx2_NWT +
      2.*vx2_SEB + 2.*vx2_SET - 2.*vx2_SWB - 2.*vx2_SWT)/8.;
   byy= (-2.*kxxMyyFromfcNEQ_NEB - 2.*kxxMyyFromfcNEQ_NET - 2.*kxxMyyFromfcNEQ_NWB - 2.*kxxMyyFromfcNEQ_NWT +
      2.*kxxMyyFromfcNEQ_SEB + 2.*kxxMyyFromfcNEQ_SET + 2.*kxxMyyFromfcNEQ_SWB + 2.*kxxMyyFromfcNEQ_SWT +
      kxxMzzFromfcNEQ_NEB + kxxMzzFromfcNEQ_NET + kxxMzzFromfcNEQ_NWB + kxxMzzFromfcNEQ_NWT -
      kxxMzzFromfcNEQ_SEB - kxxMzzFromfcNEQ_SET - kxxMzzFromfcNEQ_SWB - kxxMzzFromfcNEQ_SWT +
      2.*vx1_NEB + 2.*vx1_NET - 2.*vx1_NWB - 2.*vx1_NWT -
      2.*vx1_SEB - 2.*vx1_SET + 2.*vx1_SWB + 2.*vx1_SWT -
      2.*vx3_NEB + 2.*vx3_NET - 2.*vx3_NWB + 2.*vx3_NWT +
      2.*vx3_SEB - 2.*vx3_SET + 2.*vx3_SWB - 2.*vx3_SWT)/16.;
   cyy= (kyzFromfcNEQ_NEB + kyzFromfcNEQ_NET + kyzFromfcNEQ_NWB + kyzFromfcNEQ_NWT -
      kyzFromfcNEQ_SEB - kyzFromfcNEQ_SET - kyzFromfcNEQ_SWB - kyzFromfcNEQ_SWT +
      2.*vx2_NEB - 2.*vx2_NET + 2.*vx2_NWB - 2.*vx2_NWT -
      2.*vx2_SEB + 2.*vx2_SET - 2.*vx2_SWB + 2.*vx2_SWT)/8.;
   az = (-vx1_NEB + vx1_NET - vx1_NWB + vx1_NWT - vx1_SEB + vx1_SET - vx1_SWB + vx1_SWT)/4.;
   bz = (-vx2_NEB + vx2_NET - vx2_NWB + vx2_NWT - vx2_SEB + vx2_SET - vx2_SWB + vx2_SWT)/4.;
   cz = (-vx3_NEB + vx3_NET - vx3_NWB + vx3_NWT - vx3_SEB + vx3_SET - vx3_SWB + vx3_SWT)/4.;
   azz= (-kxzFromfcNEQ_NEB + kxzFromfcNEQ_NET - kxzFromfcNEQ_NWB + kxzFromfcNEQ_NWT -
      kxzFromfcNEQ_SEB + kxzFromfcNEQ_SET - kxzFromfcNEQ_SWB + kxzFromfcNEQ_SWT +
      2.*vx3_NEB - 2.*vx3_NET - 2.*vx3_NWB + 2.*vx3_NWT +
      2.*vx3_SEB - 2.*vx3_SET - 2.*vx3_SWB + 2.*vx3_SWT)/8.;
   bzz= (-kyzFromfcNEQ_NEB + kyzFromfcNEQ_NET - kyzFromfcNEQ_NWB + kyzFromfcNEQ_NWT -
      kyzFromfcNEQ_SEB + kyzFromfcNEQ_SET - kyzFromfcNEQ_SWB + kyzFromfcNEQ_SWT +
      2.*vx3_NEB - 2.*vx3_NET + 2.*vx3_NWB - 2.*vx3_NWT -
      2.*vx3_SEB + 2.*vx3_SET - 2.*vx3_SWB + 2.*vx3_SWT)/8.;
   czz= (-kxxMyyFromfcNEQ_NEB + kxxMyyFromfcNEQ_NET - kxxMyyFromfcNEQ_NWB + kxxMyyFromfcNEQ_NWT -
      kxxMyyFromfcNEQ_SEB + kxxMyyFromfcNEQ_SET - kxxMyyFromfcNEQ_SWB + kxxMyyFromfcNEQ_SWT +
      2.*kxxMzzFromfcNEQ_NEB - 2.*kxxMzzFromfcNEQ_NET + 2.*kxxMzzFromfcNEQ_NWB - 2.*kxxMzzFromfcNEQ_NWT +
      2.*kxxMzzFromfcNEQ_SEB - 2.*kxxMzzFromfcNEQ_SET + 2.*kxxMzzFromfcNEQ_SWB - 2.*kxxMzzFromfcNEQ_SWT -
      2.*vx1_NEB + 2.*vx1_NET + 2.*vx1_NWB - 2.*vx1_NWT -
      2.*vx1_SEB + 2.*vx1_SET + 2.*vx1_SWB - 2.*vx1_SWT -
      2.*vx2_NEB + 2.*vx2_NET - 2.*vx2_NWB + 2.*vx2_NWT +
      2.*vx2_SEB - 2.*vx2_SET + 2.*vx2_SWB - 2.*vx2_SWT)/16.;
   axy= (vx1_NEB + vx1_NET - vx1_NWB - vx1_NWT - vx1_SEB - vx1_SET + vx1_SWB + vx1_SWT)/2.;
   bxy= (vx2_NEB + vx2_NET - vx2_NWB - vx2_NWT - vx2_SEB - vx2_SET + vx2_SWB + vx2_SWT)/2.;
   cxy= (vx3_NEB + vx3_NET - vx3_NWB - vx3_NWT - vx3_SEB - vx3_SET + vx3_SWB + vx3_SWT)/2.;
   axz= (-vx1_NEB + vx1_NET + vx1_NWB - vx1_NWT - vx1_SEB + vx1_SET + vx1_SWB - vx1_SWT)/2.;
   bxz= (-vx2_NEB + vx2_NET + vx2_NWB - vx2_NWT - vx2_SEB + vx2_SET + vx2_SWB - vx2_SWT)/2.;
   cxz= (-vx3_NEB + vx3_NET + vx3_NWB - vx3_NWT - vx3_SEB + vx3_SET + vx3_SWB - vx3_SWT)/2.;
   ayz= (-vx1_NEB + vx1_NET - vx1_NWB + vx1_NWT + vx1_SEB - vx1_SET + vx1_SWB - vx1_SWT)/2.;
   byz= (-vx2_NEB + vx2_NET - vx2_NWB + vx2_NWT + vx2_SEB - vx2_SET + vx2_SWB - vx2_SWT)/2.;
   cyz= (-vx3_NEB + vx3_NET - vx3_NWB + vx3_NWT + vx3_SEB - vx3_SET + vx3_SWB - vx3_SWT)/2.;
   axyz=-vx1_NEB + vx1_NET + vx1_NWB - vx1_NWT + vx1_SEB - vx1_SET - vx1_SWB + vx1_SWT;
   bxyz=-vx2_NEB + vx2_NET + vx2_NWB - vx2_NWT + vx2_SEB - vx2_SET - vx2_SWB + vx2_SWT;
   cxyz=-vx3_NEB + vx3_NET + vx3_NWB - vx3_NWT + vx3_SEB - vx3_SET - vx3_SWB + vx3_SWT;

   kxyAverage       =0;
   kyzAverage       =0;
   kxzAverage       =0;
   kxxMyyAverage    =0;
   kxxMzzAverage    =0;
   ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //
   // Bernd das Brot
   //
   ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   a0 = a0 + xoff * ax + yoff * ay + zoff * az + xoff_sq * axx + yoff_sq * ayy + zoff_sq * azz + xoff*yoff*axy + xoff*zoff*axz + yoff*zoff*ayz + xoff*yoff*zoff*axyz ;
   ax = ax + 2. * xoff * axx + yoff * axy + zoff * axz + yoff*zoff*axyz;
   ay = ay + 2. * yoff * ayy + xoff * axy + zoff * ayz + xoff*zoff*axyz;
   az = az + 2. * zoff * azz + xoff * axz + yoff * ayz + xoff*yoff*axyz;
   b0 = b0 + xoff * bx + yoff * by + zoff * bz + xoff_sq * bxx + yoff_sq * byy + zoff_sq * bzz + xoff*yoff*bxy + xoff*zoff*bxz + yoff*zoff*byz + xoff*yoff*zoff*bxyz;
   bx = bx + 2. * xoff * bxx + yoff * bxy + zoff * bxz + yoff*zoff*bxyz;
   by = by + 2. * yoff * byy + xoff * bxy + zoff * byz + xoff*zoff*bxyz;
   bz = bz + 2. * zoff * bzz + xoff * bxz + yoff * byz + xoff*yoff*bxyz;
   c0 = c0 + xoff * cx + yoff * cy + zoff * cz + xoff_sq * cxx + yoff_sq * cyy + zoff_sq * czz + xoff*yoff*cxy + xoff*zoff*cxz + yoff*zoff*cyz + xoff*yoff*zoff*cxyz;
   cx = cx + 2. * xoff * cxx + yoff * cxy + zoff * cxz + yoff*zoff*cxyz;
   cy = cy + 2. * yoff * cyy + xoff * cxy + zoff * cyz + xoff*zoff*cxyz;
   cz = cz + 2. * zoff * czz + xoff * cxz + yoff * cyz + xoff*yoff*cxyz;
   axy= axy + zoff*axyz;
   axz= axz + yoff*axyz;
   ayz= ayz + xoff*axyz;
   bxy= bxy + zoff*bxyz;
   bxz= bxz + yoff*bxyz;
   byz= byz + xoff*bxyz;
   cxy= cxy + zoff*cxyz;
   cxz= cxz + yoff*cxyz;
   cyz= cyz + xoff*cxyz;
   ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   LBMReal dxux = ax + 0.5*axx*xs+ 0.25*(axy*ys+axz*zs)+0.0625*axyz*ys*zs;
   LBMReal dyuy = by + 0.5 * byy * ys + 0.25 * (bxy * xs + byz * zs) + 0.0625 * bxyz * xs * zs;
   LBMReal dzuz = cz + 0.5 * czz * zs + 0.25 * (cxz * xs + cyz * ys) + 0.0625 * cxyz * xs * ys;

   LBMReal Dxy = bx + 0.5 * bxx * xs + 0.25 * (bxy * ys + bxz * zs) + 0.0625 * bxyz * ys * zs + ay + 0.5 * ayy * ys + 0.25 * (axy * xs + ayz * zs) + 0.0625 * axyz * xs * zs;
   LBMReal Dxz = cx + 0.5 * cxx * xs + 0.25 * (cxy * ys + cxz * zs) + 0.0625 * cxyz * ys * zs + az + 0.5 * azz * zs + 0.25 * (axz * xs + ayz * ys) + 0.0625 * axyz * xs * ys;
   LBMReal Dyz = cy + 0.5 * cyy * ys + 0.25 * (cxy * xs + cyz * zs) + 0.0625 * cxyz * xs * zs + bz + 0.5 * bzz * zs + 0.25 * (bxz * xs + byz * ys) + 0.0625 * bxyz * xs * ys;

   shearRate = sqrt(dxux * dxux + dyuy * dyuy + dzuz * dzuz + Dxy * Dxy + Dxz * Dxz + Dyz * Dyz);


   LBMReal o = Rheology::getHerschelBulkleyCollFactorBackward(shearRate, rho); //omega;

   if (o < omegaMin)
      o = omegaMin;

   f_E = eps_new*((2*(-2*ax + by + cz-kxxMzzAverage-kxxMyyAverage))/(27.*o));
   f_N = eps_new*((2*(ax - 2*by + cz+2*kxxMyyAverage-kxxMzzAverage))/(27.*o));
   f_T = eps_new*((2*(ax + by - 2*cz-kxxMyyAverage+2*kxxMzzAverage))/(27.*o));
   f_NE = eps_new*(-(ax + 3*ay + 3*bx + by - 2*cz+2*kxxMyyAverage-kxxMyyAverage+3*kxyAverage)/(54.*o));
   f_SE = eps_new*(-(ax - 3*ay - 3*bx + by - 2*cz+2*kxxMyyAverage-kxxMyyAverage-3*kxyAverage)/(54.*o));
   f_TE = eps_new*(-(ax + 3*az - 2*by + 3*cx + cz+2*kxxMyyAverage-kxxMzzAverage+3*kxzAverage)/(54.*o));
   f_BE = eps_new*(-(ax - 3*az - 2*by - 3*cx + cz+2*kxxMyyAverage-kxxMzzAverage-3*kxzAverage)/(54.*o));
   f_TN = eps_new*(-(-2*ax + by + 3*bz + 3*cy + cz-kxxMyyAverage-kxxMzzAverage+3*kyzAverage)/(54.*o));
   f_BN = eps_new*(-(-2*ax + by - 3*bz - 3*cy + cz-kxxMyyAverage-kxxMzzAverage-3*kyzAverage)/(54.*o));
   f_ZERO = 0.;
   f_TNE = eps_new*(-(ay + az + bx + bz + cx + cy+kxyAverage+kxzAverage+kyzAverage)/(72.*o));
   f_TSW = eps_new*((-ay + az - bx + bz + cx + cy-kxyAverage+kxzAverage+kyzAverage)/(72.*o));
   f_TSE = eps_new*((ay - az + bx + bz - cx + cy+kxyAverage-kxzAverage+kyzAverage)/(72.*o));
   f_TNW = eps_new*((ay + az + bx - bz + cx - cy+kxyAverage+kxzAverage-kyzAverage)/(72.*o));

   x_E = 0.25*eps_new*((2*(-4*axx + bxy + cxz))/(27.*o));
   x_N = 0.25*eps_new*((2*(2*axx - 2*bxy + cxz))/(27.*o));
   x_T = 0.25*eps_new*((2*(2*axx + bxy - 2*cxz))/(27.*o));
   x_NE = 0.25*eps_new*(-((2*axx + 3*axy + 6*bxx + bxy - 2*cxz))/(54.*o));
   x_SE = 0.25*eps_new*(-((2*axx - 3*axy - 6*bxx + bxy - 2*cxz))/(54.*o));
   x_TE = 0.25*eps_new*(-((2*axx + 3*axz - 2*bxy + 6*cxx + cxz))/(54.*o));
   x_BE = 0.25*eps_new*(-((2*axx - 3*axz - 2*bxy - 6*cxx + cxz))/(54.*o));
   x_TN = 0.25*eps_new*(-((-4*axx + bxy + 3*bxz + 3*cxy + cxz))/(54.*o));
   x_BN = 0.25*eps_new*(-((-4*axx + bxy - 3*bxz - 3*cxy + cxz))/(54.*o));
   x_ZERO = 0.;
   x_TNE = 0.25*eps_new*(-((axy + axz + 2*bxx + bxz + 2*cxx + cxy))/(72.*o));
   x_TSW = 0.25*eps_new*(((-axy + axz - 2*bxx + bxz + 2*cxx + cxy))/(72.*o));
   x_TSE = 0.25*eps_new*(((axy - axz + 2*bxx + bxz - 2*cxx + cxy))/(72.*o));
   x_TNW = 0.25*eps_new*(((axy + axz + 2*bxx - bxz + 2*cxx - cxy))/(72.*o));

   y_E = 0.25*eps_new*(2*(-2*axy + 2*byy + cyz))/(27.*o);
   y_N = 0.25*eps_new*(2*(axy - 4*byy + cyz))/(27.*o);
   y_T = 0.25*eps_new*(2*(axy + 2*byy - 2*cyz))/(27.*o);
   y_NE = 0.25*eps_new*(-((axy + 6*ayy + 3*bxy + 2*byy - 2*cyz))/(54.*o));
   y_SE = 0.25*eps_new*(-((axy - 6*ayy - 3*bxy + 2*byy - 2*cyz))/(54.*o));
   y_TE = 0.25*eps_new*(-((axy + 3*ayz - 4*byy + 3*cxy + cyz))/(54.*o));
   y_BE = 0.25*eps_new*(-((axy - 3*ayz - 4*byy - 3*cxy + cyz))/(54.*o));
   y_TN = 0.25*eps_new*(-((-2*axy + 2*byy + 3*byz + 6*cyy + cyz))/(54.*o));
   y_BN = 0.25*eps_new*(-((-2*axy + 2*byy - 3*byz - 6*cyy + cyz))/(54.*o));
   y_ZERO = 0.;
   y_TNE = 0.25*eps_new*(-((2*ayy + ayz + bxy + byz + cxy + 2*cyy))/(72.*o));
   y_TSW = 0.25*eps_new*(((-2*ayy + ayz - bxy + byz + cxy + 2*cyy))/(72.*o));
   y_TSE = 0.25*eps_new*(((2*ayy - ayz + bxy + byz - cxy + 2*cyy))/(72.*o));
   y_TNW = 0.25*eps_new*(((2*ayy + ayz + bxy - byz + cxy - 2*cyy))/(72.*o));

   z_E = 0.25*eps_new*((2*(-2*axz + byz + 2*czz))/(27.*o));
   z_N = 0.25*eps_new*((2*(axz - 2*byz + 2*czz))/(27.*o));
   z_T = 0.25*eps_new*((2*(axz + byz - 4*czz))/(27.*o));
   z_NE = 0.25*eps_new*(-((axz + 3*ayz + 3*bxz + byz - 4*czz))/(54.*o));
   z_SE = 0.25*eps_new*(-((axz - 3*ayz - 3*bxz + byz - 4*czz))/(54.*o));
   z_TE = 0.25*eps_new*(-((axz + 6*azz - 2*byz + 3*cxz + 2*czz))/(54.*o));
   z_BE = 0.25*eps_new*(-((axz - 6*azz - 2*byz - 3*cxz + 2*czz))/(54.*o));
   z_TN = 0.25*eps_new*(-((-2*axz + byz + 6*bzz + 3*cyz + 2*czz))/(54.*o));
   z_BN = 0.25*eps_new*(-((-2*axz + byz - 6*bzz - 3*cyz + 2*czz))/(54.*o));
   z_ZERO = 0.;
   z_TNE = 0.25*eps_new*(-((ayz + 2*azz + bxz + 2*bzz + cxz + cyz))/(72.*o));
   z_TSW = 0.25*eps_new*(((-ayz + 2*azz - bxz + 2*bzz + cxz + cyz))/(72.*o));
   z_TSE = 0.25*eps_new*(((ayz - 2*azz + bxz + 2*bzz - cxz + cyz))/(72.*o));
   z_TNW = 0.25*eps_new*(((ayz + 2*azz + bxz - 2*bzz + cxz - cyz))/(72.*o));

   xy_E   =   0.0625*eps_new *((                       2.*cxyz)/(27.*o));
   xy_N   =   0.0625*eps_new *((                       2.*cxyz)/(27.*o));
   xy_T   = -(0.0625*eps_new *((                       4.*cxyz)/(27.*o)));
   xy_NE  =   0.0625*eps_new *(                            cxyz /(27.*o));
   xy_SE  =   0.0625*eps_new *(                            cxyz /(27.*o));
   xy_TE  = -(0.0625*eps_new *(( 3.*axyz            +     cxyz)/(54.*o)));
   xy_BE  = -(0.0625*eps_new *((-3.*axyz            +     cxyz)/(54.*o)));
   xy_TN  = -(0.0625*eps_new *((            3.*bxyz +     cxyz)/(54.*o)));
   xy_BN  = -(0.0625*eps_new *((          - 3.*bxyz +     cxyz)/(54.*o)));

   xy_TNE = -(0.0625*eps_new *((     axyz +     bxyz           )/(72.*o)));
   xy_TSW =   0.0625*eps_new *((     axyz +     bxyz           )/(72.*o));
   xy_TSE =   0.0625*eps_new *((-    axyz +     bxyz           )/(72.*o));
   xy_TNW =   0.0625*eps_new *((     axyz -     bxyz           )/(72.*o));

   xz_E   =   0.0625*eps_new *((            2.*bxyz           )/(27.*o));
   xz_N   = -(0.0625*eps_new *((            4.*bxyz           )/(27.*o)));
   xz_T   =   0.0625*eps_new *((            2.*bxyz           )/(27.*o));
   xz_NE  = -(0.0625*eps_new *(( 3.*axyz +     bxyz           )/(54.*o)));
   xz_SE  = -(0.0625*eps_new *((-3.*axyz +     bxyz           )/(54.*o)));
   xz_TE  =   0.0625*eps_new *((                bxyz           )/(27.*o));
   xz_BE  =   0.0625*eps_new *((                bxyz           )/(27.*o));
   xz_TN  = -(0.0625*eps_new *((                bxyz + 3.*cxyz)/(54.*o)));
   xz_BN  = -(0.0625*eps_new *((                bxyz - 3.*cxyz)/(54.*o)));

   xz_TNE = -(0.0625*eps_new *((     axyz            +     cxyz)/(72.*o)));
   xz_TSW =   0.0625*eps_new *((-    axyz            +     cxyz)/(72.*o));
   xz_TSE =   0.0625*eps_new *((     axyz            +     cxyz)/(72.*o));
   xz_TNW =   0.0625*eps_new *((     axyz            -     cxyz)/(72.*o));

   yz_E   = -(0.0625*eps_new *(( 4.*axyz                      )/(27.*o)));
   yz_N   =   0.0625*eps_new *(( 2.*axyz                      )/(27.*o));
   yz_T   =   0.0625*eps_new *(( 2.*axyz                      )/(27.*o));
   yz_NE  = -(0.0625*eps_new *((     axyz + 3.*bxyz           )/(54.*o)));
   yz_SE  = -(0.0625*eps_new *((     axyz - 3.*bxyz           )/(54.*o)));
   yz_TE  = -(0.0625*eps_new *((     axyz            + 3.*cxyz)/(54.*o)));
   yz_BE  = -(0.0625*eps_new *((     axyz            - 3.*cxyz)/(54.*o)));
   yz_TN  =   0.0625*eps_new *((     axyz                      )/(27.*o));
   yz_BN  =   0.0625*eps_new *((     axyz                      )/(27.*o));

   yz_TNE = -(0.0625*eps_new *((                bxyz +     cxyz)/(72.*o)));
   yz_TSW =   0.0625*eps_new *((          -     bxyz +     cxyz)/(72.*o));
   yz_TSE =   0.0625*eps_new *((                bxyz -     cxyz)/(72.*o));
   yz_TNW =   0.0625*eps_new *((                bxyz +     cxyz)/(72.*o));
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::calcInterpolatedNode(LBMReal* f, /*LBMReal omega,*/ LBMReal x, LBMReal y, LBMReal z, LBMReal press, LBMReal xs, LBMReal ys, LBMReal zs)
{
   using namespace D3Q27System;
   using namespace vf::lbm::dir;

   LBMReal rho  = press ;
   LBMReal vx1  = a0 + 0.25*( xs*ax + ys*ay + zs*az) + 0.0625*(axx + xs*ys*axy + xs*zs*axz + ayy + ys*zs*ayz + azz) + 0.015625*(xs*ys*zs*axyz);
   LBMReal vx2  = b0 + 0.25*( xs*bx + ys*by + zs*bz) + 0.0625*(bxx + xs*ys*bxy + xs*zs*bxz + byy + ys*zs*byz + bzz) + 0.015625*(xs*ys*zs*bxyz);
   LBMReal vx3  = c0 + 0.25*( xs*cx + ys*cy + zs*cz) + 0.0625*(cxx + xs*ys*cxy + xs*zs*cxz + cyy + ys*zs*cyz + czz) + 0.015625*(xs*ys*zs*cxyz);

   LBMReal feq[ENDF+1];
   D3Q27System::calcIncompFeq(feq,rho,vx1,vx2,vx3);

   f[DIR_P00]    = f_E    + xs*x_E    + ys*y_E    + zs*z_E    + xs*ys*xy_E    + xs*zs*xz_E    + ys*zs*yz_E    + feq[DIR_P00];
   f[DIR_M00]    = f_E    + xs*x_E    + ys*y_E    + zs*z_E    + xs*ys*xy_E    + xs*zs*xz_E    + ys*zs*yz_E    + feq[DIR_M00];
   f[DIR_0P0]    = f_N    + xs*x_N    + ys*y_N    + zs*z_N    + xs*ys*xy_N    + xs*zs*xz_N    + ys*zs*yz_N    + feq[DIR_0P0];
   f[DIR_0M0]    = f_N    + xs*x_N    + ys*y_N    + zs*z_N    + xs*ys*xy_N    + xs*zs*xz_N    + ys*zs*yz_N    + feq[DIR_0M0];
   f[DIR_00P]    = f_T    + xs*x_T    + ys*y_T    + zs*z_T    + xs*ys*xy_T    + xs*zs*xz_T    + ys*zs*yz_T    + feq[DIR_00P];
   f[DIR_00M]    = f_T    + xs*x_T    + ys*y_T    + zs*z_T    + xs*ys*xy_T    + xs*zs*xz_T    + ys*zs*yz_T    + feq[DIR_00M];
   f[DIR_PP0]   = f_NE   + xs*x_NE   + ys*y_NE   + zs*z_NE   + xs*ys*xy_NE   + xs*zs*xz_NE   + ys*zs*yz_NE   + feq[DIR_PP0];
   f[DIR_MM0]   = f_NE   + xs*x_NE   + ys*y_NE   + zs*z_NE   + xs*ys*xy_NE   + xs*zs*xz_NE   + ys*zs*yz_NE   + feq[DIR_MM0];
   f[DIR_PM0]   = f_SE   + xs*x_SE   + ys*y_SE   + zs*z_SE   + xs*ys*xy_SE   + xs*zs*xz_SE   + ys*zs*yz_SE   + feq[DIR_PM0];
   f[DIR_MP0]   = f_SE   + xs*x_SE   + ys*y_SE   + zs*z_SE   + xs*ys*xy_SE   + xs*zs*xz_SE   + ys*zs*yz_SE   + feq[DIR_MP0];
   f[DIR_P0P]   = f_TE   + xs*x_TE   + ys*y_TE   + zs*z_TE   + xs*ys*xy_TE   + xs*zs*xz_TE   + ys*zs*yz_TE   + feq[DIR_P0P];
   f[DIR_M0M]   = f_TE   + xs*x_TE   + ys*y_TE   + zs*z_TE   + xs*ys*xy_TE   + xs*zs*xz_TE   + ys*zs*yz_TE   + feq[DIR_M0M];
   f[DIR_P0M]   = f_BE   + xs*x_BE   + ys*y_BE   + zs*z_BE   + xs*ys*xy_BE   + xs*zs*xz_BE   + ys*zs*yz_BE   + feq[DIR_P0M];
   f[DIR_M0P]   = f_BE   + xs*x_BE   + ys*y_BE   + zs*z_BE   + xs*ys*xy_BE   + xs*zs*xz_BE   + ys*zs*yz_BE   + feq[DIR_M0P];
   f[DIR_0PP]   = f_TN   + xs*x_TN   + ys*y_TN   + zs*z_TN   + xs*ys*xy_TN   + xs*zs*xz_TN   + ys*zs*yz_TN   + feq[DIR_0PP];
   f[DIR_0MM]   = f_TN   + xs*x_TN   + ys*y_TN   + zs*z_TN   + xs*ys*xy_TN   + xs*zs*xz_TN   + ys*zs*yz_TN   + feq[DIR_0MM];
   f[DIR_0PM]   = f_BN   + xs*x_BN   + ys*y_BN   + zs*z_BN   + xs*ys*xy_BN   + xs*zs*xz_BN   + ys*zs*yz_BN   + feq[DIR_0PM];
   f[DIR_0MP]   = f_BN   + xs*x_BN   + ys*y_BN   + zs*z_BN   + xs*ys*xy_BN   + xs*zs*xz_BN   + ys*zs*yz_BN   + feq[DIR_0MP];
   f[DIR_PPP]  = f_TNE  + xs*x_TNE  + ys*y_TNE  + zs*z_TNE  + xs*ys*xy_TNE  + xs*zs*xz_TNE  + ys*zs*yz_TNE  + feq[DIR_PPP];
   f[DIR_MMP]  = f_TSW  + xs*x_TSW  + ys*y_TSW  + zs*z_TSW  + xs*ys*xy_TSW  + xs*zs*xz_TSW  + ys*zs*yz_TSW  + feq[DIR_MMP];
   f[DIR_PMP]  = f_TSE  + xs*x_TSE  + ys*y_TSE  + zs*z_TSE  + xs*ys*xy_TSE  + xs*zs*xz_TSE  + ys*zs*yz_TSE  + feq[DIR_PMP];
   f[DIR_MPP]  = f_TNW  + xs*x_TNW  + ys*y_TNW  + zs*z_TNW  + xs*ys*xy_TNW  + xs*zs*xz_TNW  + ys*zs*yz_TNW  + feq[DIR_MPP];
   f[DIR_PPM]  = f_TSW  + xs*x_TSW  + ys*y_TSW  + zs*z_TSW  + xs*ys*xy_TSW  + xs*zs*xz_TSW  + ys*zs*yz_TSW  + feq[DIR_PPM];
   f[DIR_MMM]  = f_TNE  + xs*x_TNE  + ys*y_TNE  + zs*z_TNE  + xs*ys*xy_TNE  + xs*zs*xz_TNE  + ys*zs*yz_TNE  + feq[DIR_MMM];
   f[DIR_PMM]  = f_TNW  + xs*x_TNW  + ys*y_TNW  + zs*z_TNW  + xs*ys*xy_TNW  + xs*zs*xz_TNW  + ys*zs*yz_TNW  + feq[DIR_PMM];
   f[DIR_MPM]  = f_TSE  + xs*x_TSE  + ys*y_TSE  + zs*z_TSE  + xs*ys*xy_TSE  + xs*zs*xz_TSE  + ys*zs*yz_TSE  + feq[DIR_MPM];
   f[DIR_000] = f_ZERO + xs*x_ZERO + ys*y_ZERO + zs*z_ZERO                                                 + feq[DIR_000];
}
//////////////////////////////////////////////////////////////////////////
//Position SWB -0.25, -0.25, -0.25
LBMReal RheologyInterpolationProcessor::calcPressBSW()
{
   return   press_SWT * (0.140625 + 0.1875 * xoff + 0.1875 * yoff - 0.5625 * zoff) +
      press_NWT * (0.046875 + 0.0625 * xoff - 0.1875 * yoff - 0.1875 * zoff) +
      press_SET * (0.046875 - 0.1875 * xoff + 0.0625 * yoff - 0.1875 * zoff) +
      press_NET * (0.015625 - 0.0625 * xoff - 0.0625 * yoff - 0.0625 * zoff) +
      press_NEB * (0.046875 - 0.1875 * xoff - 0.1875 * yoff + 0.0625 * zoff) +
      press_NWB * (0.140625 + 0.1875 * xoff - 0.5625 * yoff + 0.1875 * zoff) +
      press_SEB * (0.140625 - 0.5625 * xoff + 0.1875 * yoff + 0.1875 * zoff) +
      press_SWB * (0.421875 + 0.5625 * xoff + 0.5625 * yoff + 0.5625 * zoff);
}
//////////////////////////////////////////////////////////////////////////
//Position SWT -0.25, -0.25, 0.25
LBMReal RheologyInterpolationProcessor::calcPressTSW()
{
   return   press_SWT * (0.421875 + 0.5625 * xoff + 0.5625 * yoff - 0.5625 * zoff) +
      press_NWT * (0.140625 + 0.1875 * xoff - 0.5625 * yoff - 0.1875 * zoff) +
      press_SET * (0.140625 - 0.5625 * xoff + 0.1875 * yoff - 0.1875 * zoff) +
      press_NET * (0.046875 - 0.1875 * xoff - 0.1875 * yoff - 0.0625 * zoff) +
      press_NEB * (0.015625 - 0.0625 * xoff - 0.0625 * yoff + 0.0625 * zoff) +
      press_NWB * (0.046875 + 0.0625 * xoff - 0.1875 * yoff + 0.1875 * zoff) +
      press_SEB * (0.046875 - 0.1875 * xoff + 0.0625 * yoff + 0.1875 * zoff) +
      press_SWB * (0.140625 + 0.1875 * xoff + 0.1875 * yoff + 0.5625 * zoff);
}
//////////////////////////////////////////////////////////////////////////
//Position SET 0.25, -0.25, 0.25
LBMReal RheologyInterpolationProcessor::calcPressTSE()
{
   return   press_SET * (0.421875 - 0.5625 * xoff + 0.5625 * yoff - 0.5625 * zoff) +
      press_NET * (0.140625 - 0.1875 * xoff - 0.5625 * yoff - 0.1875 * zoff) +
      press_SWT * (0.140625 + 0.5625 * xoff + 0.1875 * yoff - 0.1875 * zoff) +
      press_NWT * (0.046875 + 0.1875 * xoff - 0.1875 * yoff - 0.0625 * zoff) +
      press_NWB * (0.015625 + 0.0625 * xoff - 0.0625 * yoff + 0.0625 * zoff) +
      press_NEB * (0.046875 - 0.0625 * xoff - 0.1875 * yoff + 0.1875 * zoff) +
      press_SWB * (0.046875 + 0.1875 * xoff + 0.0625 * yoff + 0.1875 * zoff) +
      press_SEB * (0.140625 - 0.1875 * xoff + 0.1875 * yoff + 0.5625 * zoff);
}
//////////////////////////////////////////////////////////////////////////
//Position SEB 0.25, -0.25, -0.25
LBMReal RheologyInterpolationProcessor::calcPressBSE()
{
   return   press_SET * (0.140625 - 0.1875 * xoff + 0.1875 * yoff - 0.5625 * zoff) +
      press_NET * (0.046875 - 0.0625 * xoff - 0.1875 * yoff - 0.1875 * zoff) +
      press_SWT * (0.046875 + 0.1875 * xoff + 0.0625 * yoff - 0.1875 * zoff) +
      press_NWT * (0.015625 + 0.0625 * xoff - 0.0625 * yoff - 0.0625 * zoff) +
      press_NWB * (0.046875 + 0.1875 * xoff - 0.1875 * yoff + 0.0625 * zoff) +
      press_NEB * (0.140625 - 0.1875 * xoff - 0.5625 * yoff + 0.1875 * zoff) +
      press_SWB * (0.140625 + 0.5625 * xoff + 0.1875 * yoff + 0.1875 * zoff) +
      press_SEB * (0.421875 - 0.5625 * xoff + 0.5625 * yoff + 0.5625 * zoff);
}
//////////////////////////////////////////////////////////////////////////
//Position NWB -0.25, 0.25, -0.25
LBMReal RheologyInterpolationProcessor::calcPressBNW()
{
   return   press_NWT * (0.140625 + 0.1875 * xoff - 0.1875 * yoff - 0.5625 * zoff) +
      press_NET * (0.046875 - 0.1875 * xoff - 0.0625 * yoff - 0.1875 * zoff) +
      press_SWT * (0.046875 + 0.0625 * xoff + 0.1875 * yoff - 0.1875 * zoff) +
      press_SET * (0.015625 - 0.0625 * xoff + 0.0625 * yoff - 0.0625 * zoff) +
      press_SEB * (0.046875 - 0.1875 * xoff + 0.1875 * yoff + 0.0625 * zoff) +
      press_NEB * (0.140625 - 0.5625 * xoff - 0.1875 * yoff + 0.1875 * zoff) +
      press_SWB * (0.140625 + 0.1875 * xoff + 0.5625 * yoff + 0.1875 * zoff) +
      press_NWB * (0.421875 + 0.5625 * xoff - 0.5625 * yoff + 0.5625 * zoff);
}
//////////////////////////////////////////////////////////////////////////
//Position NWT -0.25, 0.25, 0.25
LBMReal RheologyInterpolationProcessor::calcPressTNW()
{
   return   press_NWT * (0.421875 + 0.5625 * xoff - 0.5625 * yoff - 0.5625 * zoff) +
      press_NET * (0.140625 - 0.5625 * xoff - 0.1875 * yoff - 0.1875 * zoff) +
      press_SWT * (0.140625 + 0.1875 * xoff + 0.5625 * yoff - 0.1875 * zoff) +
      press_SET * (0.046875 - 0.1875 * xoff + 0.1875 * yoff - 0.0625 * zoff) +
      press_SEB * (0.015625 - 0.0625 * xoff + 0.0625 * yoff + 0.0625 * zoff) +
      press_NEB * (0.046875 - 0.1875 * xoff - 0.0625 * yoff + 0.1875 * zoff) +
      press_SWB * (0.046875 + 0.0625 * xoff + 0.1875 * yoff + 0.1875 * zoff) +
      press_NWB * (0.140625 + 0.1875 * xoff - 0.1875 * yoff + 0.5625 * zoff);
}
//////////////////////////////////////////////////////////////////////////
//Position NET 0.25, 0.25, 0.25
LBMReal RheologyInterpolationProcessor::calcPressTNE()
{
   return   press_NET * (0.421875 - 0.5625 * xoff - 0.5625 * yoff - 0.5625 * zoff) +
      press_NWT * (0.140625 + 0.5625 * xoff - 0.1875 * yoff - 0.1875 * zoff) +
      press_SET * (0.140625 - 0.1875 * xoff + 0.5625 * yoff - 0.1875 * zoff) +
      press_SWT * (0.046875 + 0.1875 * xoff + 0.1875 * yoff - 0.0625 * zoff) +
      press_SWB * (0.015625 + 0.0625 * xoff + 0.0625 * yoff + 0.0625 * zoff) +
      press_NWB * (0.046875 + 0.1875 * xoff - 0.0625 * yoff + 0.1875 * zoff) +
      press_SEB * (0.046875 - 0.0625 * xoff + 0.1875 * yoff + 0.1875 * zoff) +
      press_NEB * (0.140625 - 0.1875 * xoff - 0.1875 * yoff + 0.5625 * zoff);
}
//////////////////////////////////////////////////////////////////////////
//Position NEB 0.25, 0.25, -0.25
LBMReal RheologyInterpolationProcessor::calcPressBNE()
{
   return   press_NET * (0.140625 - 0.1875 * xoff - 0.1875 * yoff - 0.5625 * zoff) +
      press_NWT * (0.046875 + 0.1875 * xoff - 0.0625 * yoff - 0.1875 * zoff) +
      press_SET * (0.046875 - 0.0625 * xoff + 0.1875 * yoff - 0.1875 * zoff) +
      press_SWT * (0.015625 + 0.0625 * xoff + 0.0625 * yoff - 0.0625 * zoff) +
      press_SWB * (0.046875 + 0.1875 * xoff + 0.1875 * yoff + 0.0625 * zoff) +
      press_NWB * (0.140625 + 0.5625 * xoff - 0.1875 * yoff + 0.1875 * zoff) +
      press_SEB * (0.140625 - 0.1875 * xoff + 0.5625 * yoff + 0.1875 * zoff) +
      press_NEB * (0.421875 - 0.5625 * xoff - 0.5625 * yoff + 0.5625 * zoff);
}
//////////////////////////////////////////////////////////////////////////
//Position C 0.0, 0.0, 0.0
void RheologyInterpolationProcessor::calcInterpolatedNodeFC(LBMReal* f, LBMReal omega)
{
   using namespace D3Q27System;
   using namespace vf::lbm::dir;

   LBMReal press  =  press_NET * (0.125 - 0.25 * xoff - 0.25 * yoff - 0.25 * zoff) +
      press_NWT * (0.125 + 0.25 * xoff - 0.25 * yoff - 0.25 * zoff) +
      press_SET * (0.125 - 0.25 * xoff + 0.25 * yoff - 0.25 * zoff) +
      press_SWT * (0.125 + 0.25 * xoff + 0.25 * yoff - 0.25 * zoff) +
      press_NEB * (0.125 - 0.25 * xoff - 0.25 * yoff + 0.25 * zoff) +
      press_NWB * (0.125 + 0.25 * xoff - 0.25 * yoff + 0.25 * zoff) +
      press_SEB * (0.125 - 0.25 * xoff + 0.25 * yoff + 0.25 * zoff) +
      press_SWB * (0.125 + 0.25 * xoff + 0.25 * yoff + 0.25 * zoff);
   LBMReal vx1  = a0;
   LBMReal vx2  = b0;
   LBMReal vx3  = c0;

   LBMReal rho = press ;

   LBMReal feq[ENDF+1];
   D3Q27System::calcIncompFeq(feq,rho,vx1,vx2,vx3);

   LBMReal eps_new = 2.;
   

   LBMReal dxux = ax;
   LBMReal dyuy = by;
   LBMReal dzuz = cz;

   LBMReal Dxy = bx + ay;
   LBMReal Dxz = cx + az;
   LBMReal Dyz = cy + bz;

   shearRate = sqrt(dxux * dxux + dyuy * dyuy + dzuz * dzuz + Dxy * Dxy + Dxz * Dxz + Dyz * Dyz);


   LBMReal o = Rheology::getHerschelBulkleyCollFactorBackward(shearRate, rho); //omega;

   if (o < omegaMin)
      o = omegaMin;

   f_E = eps_new*((2*(-2*ax + by + cz-kxxMzzAverage-kxxMyyAverage))/(27.*o));
   f_N = eps_new*((2*(ax - 2*by + cz+2*kxxMyyAverage-kxxMzzAverage))/(27.*o));
   f_T = eps_new*((2*(ax + by - 2*cz-kxxMyyAverage+2*kxxMzzAverage))/(27.*o));
   f_NE = eps_new*(-(ax + 3*ay + 3*bx + by - 2*cz+2*kxxMyyAverage-kxxMyyAverage+3*kxyAverage)/(54.*o));
   f_SE = eps_new*(-(ax - 3*ay - 3*bx + by - 2*cz+2*kxxMyyAverage-kxxMyyAverage-3*kxyAverage)/(54.*o));
   f_TE = eps_new*(-(ax + 3*az - 2*by + 3*cx + cz+2*kxxMyyAverage-kxxMzzAverage+3*kxzAverage)/(54.*o));
   f_BE = eps_new*(-(ax - 3*az - 2*by - 3*cx + cz+2*kxxMyyAverage-kxxMzzAverage-3*kxzAverage)/(54.*o));
   f_TN = eps_new*(-(-2*ax + by + 3*bz + 3*cy + cz-kxxMyyAverage-kxxMzzAverage+3*kyzAverage)/(54.*o));
   f_BN = eps_new*(-(-2*ax + by - 3*bz - 3*cy + cz-kxxMyyAverage-kxxMzzAverage-3*kyzAverage)/(54.*o));
   f_ZERO = 0.;
   f_TNE = eps_new*(-(ay + az + bx + bz + cx + cy+kxyAverage+kxzAverage+kyzAverage)/(72.*o));
   f_TSW = eps_new*((-ay + az - bx + bz + cx + cy-kxyAverage+kxzAverage+kyzAverage)/(72.*o));
   f_TSE = eps_new*((ay - az + bx + bz - cx + cy+kxyAverage-kxzAverage+kyzAverage)/(72.*o));
   f_TNW = eps_new*((ay + az + bx - bz + cx - cy+kxyAverage+kxzAverage-kyzAverage)/(72.*o));

   f[DIR_P00]    = f_E    + feq[DIR_P00];
   f[DIR_M00]    = f_E    + feq[DIR_M00];
   f[DIR_0P0]    = f_N    + feq[DIR_0P0];
   f[DIR_0M0]    = f_N    + feq[DIR_0M0];
   f[DIR_00P]    = f_T    + feq[DIR_00P];
   f[DIR_00M]    = f_T    + feq[DIR_00M];
   f[DIR_PP0]   = f_NE   + feq[DIR_PP0];
   f[DIR_MM0]   = f_NE   + feq[DIR_MM0];
   f[DIR_PM0]   = f_SE   + feq[DIR_PM0];
   f[DIR_MP0]   = f_SE   + feq[DIR_MP0];
   f[DIR_P0P]   = f_TE   + feq[DIR_P0P];
   f[DIR_M0M]   = f_TE   + feq[DIR_M0M];
   f[DIR_P0M]   = f_BE   + feq[DIR_P0M];
   f[DIR_M0P]   = f_BE   + feq[DIR_M0P];
   f[DIR_0PP]   = f_TN   + feq[DIR_0PP];
   f[DIR_0MM]   = f_TN   + feq[DIR_0MM];
   f[DIR_0PM]   = f_BN   + feq[DIR_0PM];
   f[DIR_0MP]   = f_BN   + feq[DIR_0MP];
   f[DIR_PPP]  = f_TNE  + feq[DIR_PPP];
   f[DIR_MPP]  = f_TNW  + feq[DIR_MPP];
   f[DIR_PMP]  = f_TSE  + feq[DIR_PMP];
   f[DIR_MMP]  = f_TSW  + feq[DIR_MMP];
   f[DIR_PPM]  = f_TSW  + feq[DIR_PPM];
   f[DIR_MPM]  = f_TSE  + feq[DIR_MPM];
   f[DIR_PMM]  = f_TNW  + feq[DIR_PMM];
   f[DIR_MMM]  = f_TNE  + feq[DIR_MMM];
   f[DIR_000] = f_ZERO + feq[DIR_000];
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::calcInterpolatedVelocity(LBMReal x, LBMReal y, LBMReal z, LBMReal& vx1, LBMReal& vx2, LBMReal& vx3)
{
	vx1  = a0 + ax*x + ay*y + az*z + axx*x*x + ayy*y*y + azz*z*z + axy*x*y + axz*x*z + ayz*y*z+axyz*x*y*z;
	vx2  = b0 + bx*x + by*y + bz*z + bxx*x*x + byy*y*y + bzz*z*z + bxy*x*y + bxz*x*z + byz*y*z+bxyz*x*y*z;
	vx3  = c0 + cx*x + cy*y + cz*z + cxx*x*x + cyy*y*y + czz*z*z + cxy*x*y + cxz*x*z + cyz*y*z+cxyz*x*y*z;
}
//////////////////////////////////////////////////////////////////////////
void RheologyInterpolationProcessor::calcInterpolatedShearStress(LBMReal x, LBMReal y, LBMReal z,LBMReal& tauxx, LBMReal& tauyy, LBMReal& tauzz,LBMReal& tauxy, LBMReal& tauxz, LBMReal& tauyz)
{
	tauxx=ax+2*axx*x+axy*y+axz*z+axyz*y*z;
	tauyy=by+2*byy*y+bxy*x+byz*z+bxyz*x*z;
	tauzz=cz+2*czz*z+cxz*x+cyz*y+cxyz*x*y;
	tauxy=0.5*((ay+2.0*ayy*y+axy*x+ayz*z+axyz*x*z)+(bx+2.0*bxx*x+bxy*y+bxz*z+bxyz*y*z));
	tauxz=0.5*((az+2.0*azz*z+axz*x+ayz*y+axyz*x*y)+(cx+2.0*cxx*x+cxy*y+cxz*z+cxyz*y*z));
	tauyz=0.5*((bz+2.0*bzz*z+bxz*x+byz*y+bxyz*x*y)+(cy+2.0*cyy*y+cxy*x+cyz*z+cxyz*x*z));
}
