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
//! \file MultiphaseNonReflectingOutflowBCStrategy.cpp
//! \ingroup BoundarConditions
//! \author Hesameddin Safari
//=======================================================================================

#include "MultiphaseNonReflectingOutflowBCStrategy.h"
#include "BoundaryConditions.h"
#include "D3Q27System.h"
#include "DistributionArray3D.h"

MultiphaseNonReflectingOutflowBCStrategy::MultiphaseNonReflectingOutflowBCStrategy()
{
    BCStrategy::type = BCStrategy::NonReflectingOutflowBCStrategy;
    BCStrategy::preCollision = true;
}
//////////////////////////////////////////////////////////////////////////
MultiphaseNonReflectingOutflowBCStrategy::~MultiphaseNonReflectingOutflowBCStrategy()
{
}
//////////////////////////////////////////////////////////////////////////
SPtr<BCStrategy> MultiphaseNonReflectingOutflowBCStrategy::clone()
{
    SPtr<BCStrategy> bc(new MultiphaseNonReflectingOutflowBCStrategy());
    return bc;
}
//////////////////////////////////////////////////////////////////////////
void MultiphaseNonReflectingOutflowBCStrategy::addDistributions(SPtr<DistributionArray3D> distributions)
{
    this->distributions = distributions;
}
//////////////////////////////////////////////////////////////////////////
void MultiphaseNonReflectingOutflowBCStrategy::addDistributionsH(SPtr<DistributionArray3D> distributionsH)
{
    this->distributionsH = distributionsH;
}
//////////////////////////////////////////////////////////////////////////
void MultiphaseNonReflectingOutflowBCStrategy::addDistributionsH2(SPtr<DistributionArray3D> distributionsH2)
{
    this->distributionsH2 = distributionsH2;
}
//////////////////////////////////////////////////////////////////////////
void MultiphaseNonReflectingOutflowBCStrategy::applyBC()
{
    using namespace D3Q27System;
//    using namespace UbMath;
    using namespace vf::lbm::dir;
    using namespace vf::basics::constant;

    real f[ENDF + 1];
    real ftemp[ENDF + 1];
    real h[D3Q27System::ENDF + 1];
    real htemp[ENDF + 1];
    real h2[D3Q27System::ENDF + 1];
    real h2temp[ENDF + 1];

    int nx1 = x1;
    int nx2 = x2;
    int nx3 = x3;
    int direction = -1;

    // flag points in direction of fluid
    if (bcPtr->hasDensityBoundaryFlag(dP00)) {
        nx1 += 1;
        direction = dP00;
    } else if (bcPtr->hasDensityBoundaryFlag(dM00)) {
        nx1 -= 1;
        direction = dM00;
    } else if (bcPtr->hasDensityBoundaryFlag(DIR_0P0)) {
        nx2 += 1;
        direction = DIR_0P0;
    } else if (bcPtr->hasDensityBoundaryFlag(DIR_0M0)) {
        nx2 -= 1;
        direction = DIR_0M0;
    } else if (bcPtr->hasDensityBoundaryFlag(DIR_00P)) {
        nx3 += 1;
        direction = DIR_00P;
    } else if (bcPtr->hasDensityBoundaryFlag(DIR_00M)) {
        nx3 -= 1;
        direction = DIR_00M;
    } else
        UB_THROW(UbException(UB_EXARGS, "Danger...no orthogonal BC-Flag on density boundary..."));

    distributions->getPreCollisionDistribution(f, x1, x2, x3);
    distributions->getPreCollisionDistribution(ftemp, nx1, nx2, nx3);
    distributionsH->getPreCollisionDistribution(h, x1, x2, x3);
    distributionsH->getPreCollisionDistribution(htemp, nx1, nx2, nx3);
    distributionsH2->getPreCollisionDistribution(h2, x1, x2, x3);
    distributionsH2->getPreCollisionDistribution(h2temp, nx1, nx2, nx3);

    real /* phi,*/ p1, vx1, vx2, vx3;

    // D3Q27System::calcDensity(h, phi);

    calcMacrosFct(f, p1, vx1, vx2, vx3);

    switch (direction) {
        case dP00:
            f[dP00] = ftemp[dP00] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * f[dP00];
            f[DIR_PP0] = ftemp[DIR_PP0] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * f[DIR_PP0];
            f[DIR_PM0] = ftemp[DIR_PM0] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * f[DIR_PM0];
            f[DIR_P0P] = ftemp[DIR_P0P] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * f[DIR_P0P];
            f[DIR_P0M] = ftemp[DIR_P0M] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * f[DIR_P0M];
            f[DIR_PPP] = ftemp[DIR_PPP] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * f[DIR_PPP];
            f[DIR_PMP] = ftemp[DIR_PMP] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * f[DIR_PMP];
            f[DIR_PPM] = ftemp[DIR_PPM] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * f[DIR_PPM];
            f[DIR_PMM] = ftemp[DIR_PMM] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * f[DIR_PMM];

            distributions->setPreCollisionDistributionForDirection(f[dP00], x1 + DX1[dM00], x2 + DX2[dM00], x3 + DX3[dM00], dM00);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PP0], x1 + DX1[DIR_MM0], x2 + DX2[DIR_MM0], x3 + DX3[DIR_MM0], DIR_MM0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PM0], x1 + DX1[DIR_MP0], x2 + DX2[DIR_MP0], x3 + DX3[DIR_MP0], DIR_MP0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_P0P], x1 + DX1[DIR_M0M], x2 + DX2[DIR_M0M], x3 + DX3[DIR_M0M], DIR_M0M);
            distributions->setPreCollisionDistributionForDirection(f[DIR_P0M], x1 + DX1[DIR_M0P], x2 + DX2[DIR_M0P], x3 + DX3[DIR_M0P], DIR_M0P);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PPP], x1 + DX1[DIR_MMM], x2 + DX2[DIR_MMM], x3 + DX3[DIR_MMM], DIR_MMM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PMP], x1 + DX1[DIR_MPM], x2 + DX2[DIR_MPM], x3 + DX3[DIR_MPM], DIR_MPM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PPM], x1 + DX1[DIR_MMP], x2 + DX2[DIR_MMP], x3 + DX3[DIR_MMP], DIR_MMP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PMM], x1 + DX1[DIR_MPP], x2 + DX2[DIR_MPP], x3 + DX3[DIR_MPP], DIR_MPP);

            h[dP00] = htemp[dP00] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h[dP00];
            h[DIR_PP0] = htemp[DIR_PP0] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h[DIR_PP0];
            h[DIR_PM0] = htemp[DIR_PM0] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h[DIR_PM0];
            h[DIR_P0P] = htemp[DIR_P0P] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h[DIR_P0P];
            h[DIR_P0M] = htemp[DIR_P0M] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h[DIR_P0M];
            h[DIR_PPP] = htemp[DIR_PPP] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h[DIR_PPP];
            h[DIR_PMP] = htemp[DIR_PMP] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h[DIR_PMP];
            h[DIR_PPM] = htemp[DIR_PPM] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h[DIR_PPM];
            h[DIR_PMM] = htemp[DIR_PMM] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h[DIR_PMM];

            distributionsH->setPreCollisionDistributionForDirection(h[dP00], x1 + DX1[dM00], x2 + DX2[dM00], x3 + DX3[dM00], dM00);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PP0], x1 + DX1[DIR_MM0], x2 + DX2[DIR_MM0], x3 + DX3[DIR_MM0], DIR_MM0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PM0], x1 + DX1[DIR_MP0], x2 + DX2[DIR_MP0], x3 + DX3[DIR_MP0], DIR_MP0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_P0P], x1 + DX1[DIR_M0M], x2 + DX2[DIR_M0M], x3 + DX3[DIR_M0M], DIR_M0M);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_P0M], x1 + DX1[DIR_M0P], x2 + DX2[DIR_M0P], x3 + DX3[DIR_M0P], DIR_M0P);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PPP], x1 + DX1[DIR_MMM], x2 + DX2[DIR_MMM], x3 + DX3[DIR_MMM], DIR_MMM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PMP], x1 + DX1[DIR_MPM], x2 + DX2[DIR_MPM], x3 + DX3[DIR_MPM], DIR_MPM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PPM], x1 + DX1[DIR_MMP], x2 + DX2[DIR_MMP], x3 + DX3[DIR_MMP], DIR_MMP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PMM], x1 + DX1[DIR_MPP], x2 + DX2[DIR_MPP], x3 + DX3[DIR_MPP], DIR_MPP);

            h2[dP00] = c1o2 * (h2temp[dP00] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h2[dP00]);
            h2[DIR_PP0] = c1o2 * (h2temp[DIR_PP0] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h2[DIR_PP0]);
            h2[DIR_PM0] = c1o2 * (h2temp[DIR_PM0] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h2[DIR_PM0]);
            h2[DIR_P0P] = c1o2 * (h2temp[DIR_P0P] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h2[DIR_P0P]);
            h2[DIR_P0M] = c1o2 * (h2temp[DIR_P0M] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h2[DIR_P0M]);
            h2[DIR_PPP] = c1o2 * (h2temp[DIR_PPP] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h2[DIR_PPP]);
            h2[DIR_PMP] = c1o2 * (h2temp[DIR_PMP] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h2[DIR_PMP]);
            h2[DIR_PPM] = c1o2 * (h2temp[DIR_PPM] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h2[DIR_PPM]);
            h2[DIR_PMM] = c1o2 * (h2temp[DIR_PMM] * (one_over_sqrt3 + vx1) + (c1o1 - one_over_sqrt3 - vx1) * h2[DIR_PMM]);

            distributionsH2->setPreCollisionDistributionForDirection(h2[dP00], x1 + DX1[dM00], x2 + DX2[dM00], x3 + DX3[dM00], dM00);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PP0], x1 + DX1[DIR_MM0], x2 + DX2[DIR_MM0], x3 + DX3[DIR_MM0], DIR_MM0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PM0], x1 + DX1[DIR_MP0], x2 + DX2[DIR_MP0], x3 + DX3[DIR_MP0], DIR_MP0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_P0P], x1 + DX1[DIR_M0M], x2 + DX2[DIR_M0M], x3 + DX3[DIR_M0M], DIR_M0M);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_P0M], x1 + DX1[DIR_M0P], x2 + DX2[DIR_M0P], x3 + DX3[DIR_M0P], DIR_M0P);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PPP], x1 + DX1[DIR_MMM], x2 + DX2[DIR_MMM], x3 + DX3[DIR_MMM], DIR_MMM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PMP], x1 + DX1[DIR_MPM], x2 + DX2[DIR_MPM], x3 + DX3[DIR_MPM], DIR_MPM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PPM], x1 + DX1[DIR_MMP], x2 + DX2[DIR_MMP], x3 + DX3[DIR_MMP], DIR_MMP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PMM], x1 + DX1[DIR_MPP], x2 + DX2[DIR_MPP], x3 + DX3[DIR_MPP], DIR_MPP);

            break;
        case dM00:
            f[dM00] = ftemp[dM00] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * f[dM00];
            f[DIR_MP0] = ftemp[DIR_MP0] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * f[DIR_MP0];
            f[DIR_MM0] = ftemp[DIR_MM0] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * f[DIR_MM0];
            f[DIR_M0P] = ftemp[DIR_M0P] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * f[DIR_M0P];
            f[DIR_M0M] = ftemp[DIR_M0M] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * f[DIR_M0M];
            f[DIR_MPP] = ftemp[DIR_MPP] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * f[DIR_MPP];
            f[DIR_MMP] = ftemp[DIR_MMP] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * f[DIR_MMP];
            f[DIR_MPM] = ftemp[DIR_MPM] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * f[DIR_MPM];
            f[DIR_MMM] = ftemp[DIR_MMM] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * f[DIR_MMM];

            distributions->setPreCollisionDistributionForDirection(f[dM00], x1 + DX1[dP00], x2 + DX2[dP00], x3 + DX3[dP00], dP00);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MP0], x1 + DX1[DIR_PM0], x2 + DX2[DIR_PM0], x3 + DX3[DIR_PM0], DIR_PM0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MM0], x1 + DX1[DIR_PP0], x2 + DX2[DIR_PP0], x3 + DX3[DIR_PP0], DIR_PP0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_M0P], x1 + DX1[DIR_P0M], x2 + DX2[DIR_P0M], x3 + DX3[DIR_P0M], DIR_P0M);
            distributions->setPreCollisionDistributionForDirection(f[DIR_M0M], x1 + DX1[DIR_P0P], x2 + DX2[DIR_P0P], x3 + DX3[DIR_P0P], DIR_P0P);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MPP], x1 + DX1[DIR_PMM], x2 + DX2[DIR_PMM], x3 + DX3[DIR_PMM], DIR_PMM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MMP], x1 + DX1[DIR_PPM], x2 + DX2[DIR_PPM], x3 + DX3[DIR_PPM], DIR_PPM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MPM], x1 + DX1[DIR_PMP], x2 + DX2[DIR_PMP], x3 + DX3[DIR_PMP], DIR_PMP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MMM], x1 + DX1[DIR_PPP], x2 + DX2[DIR_PPP], x3 + DX3[DIR_PPP], DIR_PPP);

            h[dM00] = htemp[dM00] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h[dM00];
            h[DIR_MP0] = htemp[DIR_MP0] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h[DIR_MP0];
            h[DIR_MM0] = htemp[DIR_MM0] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h[DIR_MM0];
            h[DIR_M0P] = htemp[DIR_M0P] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h[DIR_M0P];
            h[DIR_M0M] = htemp[DIR_M0M] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h[DIR_M0M];
            h[DIR_MPP] = htemp[DIR_MPP] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h[DIR_MPP];
            h[DIR_MMP] = htemp[DIR_MMP] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h[DIR_MMP];
            h[DIR_MPM] = htemp[DIR_MPM] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h[DIR_MPM];
            h[DIR_MMM] = htemp[DIR_MMM] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h[DIR_MMM];

            distributionsH->setPreCollisionDistributionForDirection(h[dM00], x1 + DX1[dP00], x2 + DX2[dP00], x3 + DX3[dP00], dP00);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MP0], x1 + DX1[DIR_PM0], x2 + DX2[DIR_PM0], x3 + DX3[DIR_PM0], DIR_PM0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MM0], x1 + DX1[DIR_PP0], x2 + DX2[DIR_PP0], x3 + DX3[DIR_PP0], DIR_PP0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_M0P], x1 + DX1[DIR_P0M], x2 + DX2[DIR_P0M], x3 + DX3[DIR_P0M], DIR_P0M);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_M0M], x1 + DX1[DIR_P0P], x2 + DX2[DIR_P0P], x3 + DX3[DIR_P0P], DIR_P0P);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MPP], x1 + DX1[DIR_PMM], x2 + DX2[DIR_PMM], x3 + DX3[DIR_PMM], DIR_PMM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MMP], x1 + DX1[DIR_PPM], x2 + DX2[DIR_PPM], x3 + DX3[DIR_PPM], DIR_PPM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MPM], x1 + DX1[DIR_PMP], x2 + DX2[DIR_PMP], x3 + DX3[DIR_PMP], DIR_PMP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MMM], x1 + DX1[DIR_PPP], x2 + DX2[DIR_PPP], x3 + DX3[DIR_PPP], DIR_PPP);

            h2[dM00] = c1o2 * (htemp[dM00] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h2[dM00]);
            h2[DIR_MP0] = c1o2 * (htemp[DIR_MP0] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h2[DIR_MP0]);
            h2[DIR_MM0] = c1o2 * (htemp[DIR_MM0] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h2[DIR_MM0]);
            h2[DIR_M0P] = c1o2 * (htemp[DIR_M0P] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h2[DIR_M0P]);
            h2[DIR_M0M] = c1o2 * (htemp[DIR_M0M] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h2[DIR_M0M]);
            h2[DIR_MPP] = c1o2 * (htemp[DIR_MPP] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h2[DIR_MPP]);
            h2[DIR_MMP] = c1o2 * (htemp[DIR_MMP] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h2[DIR_MMP]);
            h2[DIR_MPM] = c1o2 * (htemp[DIR_MPM] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h2[DIR_MPM]);
            h2[DIR_MMM] = c1o2 * (htemp[DIR_MMM] * (one_over_sqrt3 - vx1) + (c1o1 - one_over_sqrt3 + vx1) * h2[DIR_MMM]);

            distributionsH2->setPreCollisionDistributionForDirection(h2[dM00], x1 + DX1[dP00], x2 + DX2[dP00], x3 + DX3[dP00], dP00);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MP0], x1 + DX1[DIR_PM0], x2 + DX2[DIR_PM0], x3 + DX3[DIR_PM0], DIR_PM0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MM0], x1 + DX1[DIR_PP0], x2 + DX2[DIR_PP0], x3 + DX3[DIR_PP0], DIR_PP0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_M0P], x1 + DX1[DIR_P0M], x2 + DX2[DIR_P0M], x3 + DX3[DIR_P0M], DIR_P0M);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_M0M], x1 + DX1[DIR_P0P], x2 + DX2[DIR_P0P], x3 + DX3[DIR_P0P], DIR_P0P);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MPP], x1 + DX1[DIR_PMM], x2 + DX2[DIR_PMM], x3 + DX3[DIR_PMM], DIR_PMM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MMP], x1 + DX1[DIR_PPM], x2 + DX2[DIR_PPM], x3 + DX3[DIR_PPM], DIR_PPM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MPM], x1 + DX1[DIR_PMP], x2 + DX2[DIR_PMP], x3 + DX3[DIR_PMP], DIR_PMP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MMM], x1 + DX1[DIR_PPP], x2 + DX2[DIR_PPP], x3 + DX3[DIR_PPP], DIR_PPP);
            break;
        case DIR_0P0:
            f[DIR_0P0] = ftemp[DIR_0P0] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * f[DIR_0P0];
            f[DIR_PP0] = ftemp[DIR_PP0] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * f[DIR_PP0];
            f[DIR_MP0] = ftemp[DIR_MP0] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * f[DIR_MP0];
            f[DIR_0PP] = ftemp[DIR_0PP] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * f[DIR_0PP];
            f[DIR_0PM] = ftemp[DIR_0PM] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * f[DIR_0PM];
            f[DIR_PPP] = ftemp[DIR_PPP] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * f[DIR_PPP];
            f[DIR_MPP] = ftemp[DIR_MPP] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * f[DIR_MPP];
            f[DIR_PPM] = ftemp[DIR_PPM] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * f[DIR_PPM];
            f[DIR_MPM] = ftemp[DIR_MPM] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * f[DIR_MPM];

            distributions->setPreCollisionDistributionForDirection(f[DIR_0P0], x1 + DX1[DIR_0M0], x2 + DX2[DIR_0M0], x3 + DX3[DIR_0M0], DIR_0M0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PP0], x1 + DX1[DIR_MM0], x2 + DX2[DIR_MM0], x3 + DX3[DIR_MM0], DIR_MM0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MP0], x1 + DX1[DIR_PM0], x2 + DX2[DIR_PM0], x3 + DX3[DIR_PM0], DIR_PM0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_0PP], x1 + DX1[DIR_0MM], x2 + DX2[DIR_0MM], x3 + DX3[DIR_0MM], DIR_0MM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_0PM], x1 + DX1[DIR_0MP], x2 + DX2[DIR_0MP], x3 + DX3[DIR_0MP], DIR_0MP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PPP], x1 + DX1[DIR_MMM], x2 + DX2[DIR_MMM], x3 + DX3[DIR_MMM], DIR_MMM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MPP], x1 + DX1[DIR_PMM], x2 + DX2[DIR_PMM], x3 + DX3[DIR_PMM], DIR_PMM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PPM], x1 + DX1[DIR_MMP], x2 + DX2[DIR_MMP], x3 + DX3[DIR_MMP], DIR_MMP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MPM], x1 + DX1[DIR_PMP], x2 + DX2[DIR_PMP], x3 + DX3[DIR_PMP], DIR_PMP);

            h[DIR_0P0] = htemp[DIR_0P0] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h[DIR_0P0];
            h[DIR_PP0] = htemp[DIR_PP0] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h[DIR_PP0];
            h[DIR_MP0] = htemp[DIR_MP0] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h[DIR_MP0];
            h[DIR_0PP] = htemp[DIR_0PP] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h[DIR_0PP];
            h[DIR_0PM] = htemp[DIR_0PM] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h[DIR_0PM];
            h[DIR_PPP] = htemp[DIR_PPP] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h[DIR_PPP];
            h[DIR_MPP] = htemp[DIR_MPP] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h[DIR_MPP];
            h[DIR_PPM] = htemp[DIR_PPM] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h[DIR_PPM];
            h[DIR_MPM] = htemp[DIR_MPM] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h[DIR_MPM];

            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0P0], x1 + DX1[DIR_0M0], x2 + DX2[DIR_0M0], x3 + DX3[DIR_0M0], DIR_0M0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PP0], x1 + DX1[DIR_MM0], x2 + DX2[DIR_MM0], x3 + DX3[DIR_MM0], DIR_MM0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MP0], x1 + DX1[DIR_PM0], x2 + DX2[DIR_PM0], x3 + DX3[DIR_PM0], DIR_PM0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0PP], x1 + DX1[DIR_0MM], x2 + DX2[DIR_0MM], x3 + DX3[DIR_0MM], DIR_0MM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0PM], x1 + DX1[DIR_0MP], x2 + DX2[DIR_0MP], x3 + DX3[DIR_0MP], DIR_0MP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PPP], x1 + DX1[DIR_MMM], x2 + DX2[DIR_MMM], x3 + DX3[DIR_MMM], DIR_MMM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MPP], x1 + DX1[DIR_PMM], x2 + DX2[DIR_PMM], x3 + DX3[DIR_PMM], DIR_PMM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PPM], x1 + DX1[DIR_MMP], x2 + DX2[DIR_MMP], x3 + DX3[DIR_MMP], DIR_MMP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MPM], x1 + DX1[DIR_PMP], x2 + DX2[DIR_PMP], x3 + DX3[DIR_PMP], DIR_PMP);

            h2[DIR_0P0] = c1o2 * (htemp[DIR_0P0] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h2[DIR_0P0]);
            h2[DIR_PP0] = c1o2 * (htemp[DIR_PP0] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h2[DIR_PP0]);
            h2[DIR_MP0] = c1o2 * (htemp[DIR_MP0] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h2[DIR_MP0]);
            h2[DIR_0PP] = c1o2 * (htemp[DIR_0PP] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h2[DIR_0PP]);
            h2[DIR_0PM] = c1o2 * (htemp[DIR_0PM] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h2[DIR_0PM]);
            h2[DIR_PPP] = c1o2 * (htemp[DIR_PPP] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h2[DIR_PPP]);
            h2[DIR_MPP] = c1o2 * (htemp[DIR_MPP] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h2[DIR_MPP]);
            h2[DIR_PPM] = c1o2 * (htemp[DIR_PPM] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h2[DIR_PPM]);
            h2[DIR_MPM] = c1o2 * (htemp[DIR_MPM] * (one_over_sqrt3 + vx2) + (c1o1 - one_over_sqrt3 - vx2) * h2[DIR_MPM]);

            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0P0], x1 + DX1[DIR_0M0], x2 + DX2[DIR_0M0], x3 + DX3[DIR_0M0], DIR_0M0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PP0], x1 + DX1[DIR_MM0], x2 + DX2[DIR_MM0], x3 + DX3[DIR_MM0], DIR_MM0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MP0], x1 + DX1[DIR_PM0], x2 + DX2[DIR_PM0], x3 + DX3[DIR_PM0], DIR_PM0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0PP], x1 + DX1[DIR_0MM], x2 + DX2[DIR_0MM], x3 + DX3[DIR_0MM], DIR_0MM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0PM], x1 + DX1[DIR_0MP], x2 + DX2[DIR_0MP], x3 + DX3[DIR_0MP], DIR_0MP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PPP], x1 + DX1[DIR_MMM], x2 + DX2[DIR_MMM], x3 + DX3[DIR_MMM], DIR_MMM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MPP], x1 + DX1[DIR_PMM], x2 + DX2[DIR_PMM], x3 + DX3[DIR_PMM], DIR_PMM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PPM], x1 + DX1[DIR_MMP], x2 + DX2[DIR_MMP], x3 + DX3[DIR_MMP], DIR_MMP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MPM], x1 + DX1[DIR_PMP], x2 + DX2[DIR_PMP], x3 + DX3[DIR_PMP], DIR_PMP);

            break;
        case DIR_0M0:
            f[DIR_0M0] = ftemp[DIR_0M0] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * f[DIR_0M0];
            f[DIR_PM0] = ftemp[DIR_PM0] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * f[DIR_PM0];
            f[DIR_MM0] = ftemp[DIR_MM0] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * f[DIR_MM0];
            f[DIR_0MP] = ftemp[DIR_0MP] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * f[DIR_0MP];
            f[DIR_0MM] = ftemp[DIR_0MM] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * f[DIR_0MM];
            f[DIR_PMP] = ftemp[DIR_PMP] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * f[DIR_PMP];
            f[DIR_MMP] = ftemp[DIR_MMP] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * f[DIR_MMP];
            f[DIR_PMM] = ftemp[DIR_PMM] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * f[DIR_PMM];
            f[DIR_MMM] = ftemp[DIR_MMM] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * f[DIR_MMM];

            distributions->setPreCollisionDistributionForDirection(f[DIR_0M0], x1 + DX1[DIR_0P0], x2 + DX2[DIR_0P0], x3 + DX3[DIR_0P0], DIR_0P0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PM0], x1 + DX1[DIR_MP0], x2 + DX2[DIR_MP0], x3 + DX3[DIR_MP0], DIR_MP0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MM0], x1 + DX1[DIR_PP0], x2 + DX2[DIR_PP0], x3 + DX3[DIR_PP0], DIR_PP0);
            distributions->setPreCollisionDistributionForDirection(f[DIR_0MP], x1 + DX1[DIR_0PM], x2 + DX2[DIR_0PM], x3 + DX3[DIR_0PM], DIR_0PM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_0MM], x1 + DX1[DIR_0PP], x2 + DX2[DIR_0PP], x3 + DX3[DIR_0PP], DIR_0PP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PMP], x1 + DX1[DIR_MPM], x2 + DX2[DIR_MPM], x3 + DX3[DIR_MPM], DIR_MPM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MMP], x1 + DX1[DIR_PPM], x2 + DX2[DIR_PPM], x3 + DX3[DIR_PPM], DIR_PPM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PMM], x1 + DX1[DIR_MPP], x2 + DX2[DIR_MPP], x3 + DX3[DIR_MPP], DIR_MPP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MMM], x1 + DX1[DIR_PPP], x2 + DX2[DIR_PPP], x3 + DX3[DIR_PPP], DIR_PPP);

            h[DIR_0M0] = htemp[DIR_0M0] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h[DIR_0M0];
            h[DIR_PM0] = htemp[DIR_PM0] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h[DIR_PM0];
            h[DIR_MM0] = htemp[DIR_MM0] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h[DIR_MM0];
            h[DIR_0MP] = htemp[DIR_0MP] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h[DIR_0MP];
            h[DIR_0MM] = htemp[DIR_0MM] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h[DIR_0MM];
            h[DIR_PMP] = htemp[DIR_PMP] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h[DIR_PMP];
            h[DIR_MMP] = htemp[DIR_MMP] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h[DIR_MMP];
            h[DIR_PMM] = htemp[DIR_PMM] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h[DIR_PMM];
            h[DIR_MMM] = htemp[DIR_MMM] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h[DIR_MMM];

            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0M0], x1 + DX1[DIR_0P0], x2 + DX2[DIR_0P0], x3 + DX3[DIR_0P0], DIR_0P0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PM0], x1 + DX1[DIR_MP0], x2 + DX2[DIR_MP0], x3 + DX3[DIR_MP0], DIR_MP0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MM0], x1 + DX1[DIR_PP0], x2 + DX2[DIR_PP0], x3 + DX3[DIR_PP0], DIR_PP0);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0MP], x1 + DX1[DIR_0PM], x2 + DX2[DIR_0PM], x3 + DX3[DIR_0PM], DIR_0PM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0MM], x1 + DX1[DIR_0PP], x2 + DX2[DIR_0PP], x3 + DX3[DIR_0PP], DIR_0PP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PMP], x1 + DX1[DIR_MPM], x2 + DX2[DIR_MPM], x3 + DX3[DIR_MPM], DIR_MPM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MMP], x1 + DX1[DIR_PPM], x2 + DX2[DIR_PPM], x3 + DX3[DIR_PPM], DIR_PPM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PMM], x1 + DX1[DIR_MPP], x2 + DX2[DIR_MPP], x3 + DX3[DIR_MPP], DIR_MPP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MMM], x1 + DX1[DIR_PPP], x2 + DX2[DIR_PPP], x3 + DX3[DIR_PPP], DIR_PPP);

            h2[DIR_0M0] = c1o2 * (htemp[DIR_0M0] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h2[DIR_0M0]);
            h2[DIR_PM0] = c1o2 * (htemp[DIR_PM0] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h2[DIR_PM0]);
            h2[DIR_MM0] = c1o2 * (htemp[DIR_MM0] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h2[DIR_MM0]);
            h2[DIR_0MP] = c1o2 * (htemp[DIR_0MP] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h2[DIR_0MP]);
            h2[DIR_0MM] = c1o2 * (htemp[DIR_0MM] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h2[DIR_0MM]);
            h2[DIR_PMP] = c1o2 * (htemp[DIR_PMP] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h2[DIR_PMP]);
            h2[DIR_MMP] = c1o2 * (htemp[DIR_MMP] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h2[DIR_MMP]);
            h2[DIR_PMM] = c1o2 * (htemp[DIR_PMM] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h2[DIR_PMM]);
            h2[DIR_MMM] = c1o2 * (htemp[DIR_MMM] * (one_over_sqrt3 - vx2) + (c1o1 - one_over_sqrt3 + vx2) * h2[DIR_MMM]);

            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0M0], x1 + DX1[DIR_0P0], x2 + DX2[DIR_0P0], x3 + DX3[DIR_0P0], DIR_0P0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PM0], x1 + DX1[DIR_MP0], x2 + DX2[DIR_MP0], x3 + DX3[DIR_MP0], DIR_MP0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MM0], x1 + DX1[DIR_PP0], x2 + DX2[DIR_PP0], x3 + DX3[DIR_PP0], DIR_PP0);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0MP], x1 + DX1[DIR_0PM], x2 + DX2[DIR_0PM], x3 + DX3[DIR_0PM], DIR_0PM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0MM], x1 + DX1[DIR_0PP], x2 + DX2[DIR_0PP], x3 + DX3[DIR_0PP], DIR_0PP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PMP], x1 + DX1[DIR_MPM], x2 + DX2[DIR_MPM], x3 + DX3[DIR_MPM], DIR_MPM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MMP], x1 + DX1[DIR_PPM], x2 + DX2[DIR_PPM], x3 + DX3[DIR_PPM], DIR_PPM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PMM], x1 + DX1[DIR_MPP], x2 + DX2[DIR_MPP], x3 + DX3[DIR_MPP], DIR_MPP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MMM], x1 + DX1[DIR_PPP], x2 + DX2[DIR_PPP], x3 + DX3[DIR_PPP], DIR_PPP);

            break;
        case DIR_00P:
            f[DIR_00P] = ftemp[DIR_00P] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * f[DIR_00P];
            f[DIR_P0P] = ftemp[DIR_P0P] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * f[DIR_P0P];
            f[DIR_M0P] = ftemp[DIR_M0P] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * f[DIR_M0P];
            f[DIR_0PP] = ftemp[DIR_0PP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * f[DIR_0PP];
            f[DIR_0MP] = ftemp[DIR_0MP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * f[DIR_0MP];
            f[DIR_PPP] = ftemp[DIR_PPP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * f[DIR_PPP];
            f[DIR_MPP] = ftemp[DIR_MPP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * f[DIR_MPP];
            f[DIR_PMP] = ftemp[DIR_PMP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * f[DIR_PMP];
            f[DIR_MMP] = ftemp[DIR_MMP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * f[DIR_MMP];

            distributions->setPreCollisionDistributionForDirection(f[DIR_00P], x1 + DX1[DIR_00M], x2 + DX2[DIR_00M], x3 + DX3[DIR_00M], DIR_00M);
            distributions->setPreCollisionDistributionForDirection(f[DIR_P0P], x1 + DX1[DIR_M0M], x2 + DX2[DIR_M0M], x3 + DX3[DIR_M0M], DIR_M0M);
            distributions->setPreCollisionDistributionForDirection(f[DIR_M0P], x1 + DX1[DIR_P0M], x2 + DX2[DIR_P0M], x3 + DX3[DIR_P0M], DIR_P0M);
            distributions->setPreCollisionDistributionForDirection(f[DIR_0PP], x1 + DX1[DIR_0MM], x2 + DX2[DIR_0MM], x3 + DX3[DIR_0MM], DIR_0MM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_0MP], x1 + DX1[DIR_0PM], x2 + DX2[DIR_0PM], x3 + DX3[DIR_0PM], DIR_0PM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PPP], x1 + DX1[DIR_MMM], x2 + DX2[DIR_MMM], x3 + DX3[DIR_MMM], DIR_MMM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MPP], x1 + DX1[DIR_PMM], x2 + DX2[DIR_PMM], x3 + DX3[DIR_PMM], DIR_PMM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PMP], x1 + DX1[DIR_MPM], x2 + DX2[DIR_MPM], x3 + DX3[DIR_MPM], DIR_MPM);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MMP], x1 + DX1[DIR_PPM], x2 + DX2[DIR_PPM], x3 + DX3[DIR_PPM], DIR_PPM);

            h[DIR_00P] = htemp[DIR_00P] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h[DIR_00P];
            h[DIR_P0P] = htemp[DIR_P0P] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h[DIR_P0P];
            h[DIR_M0P] = htemp[DIR_M0P] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h[DIR_M0P];
            h[DIR_0PP] = htemp[DIR_0PP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h[DIR_0PP];
            h[DIR_0MP] = htemp[DIR_0MP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h[DIR_0MP];
            h[DIR_PPP] = htemp[DIR_PPP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h[DIR_PPP];
            h[DIR_MPP] = htemp[DIR_MPP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h[DIR_MPP];
            h[DIR_PMP] = htemp[DIR_PMP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h[DIR_PMP];
            h[DIR_MMP] = htemp[DIR_MMP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h[DIR_MMP];

            distributionsH->setPreCollisionDistributionForDirection(h[DIR_00P], x1 + DX1[DIR_00M], x2 + DX2[DIR_00M], x3 + DX3[DIR_00M], DIR_00M);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_P0P], x1 + DX1[DIR_M0M], x2 + DX2[DIR_M0M], x3 + DX3[DIR_M0M], DIR_M0M);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_M0P], x1 + DX1[DIR_P0M], x2 + DX2[DIR_P0M], x3 + DX3[DIR_P0M], DIR_P0M);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0PP], x1 + DX1[DIR_0MM], x2 + DX2[DIR_0MM], x3 + DX3[DIR_0MM], DIR_0MM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0MP], x1 + DX1[DIR_0PM], x2 + DX2[DIR_0PM], x3 + DX3[DIR_0PM], DIR_0PM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PPP], x1 + DX1[DIR_MMM], x2 + DX2[DIR_MMM], x3 + DX3[DIR_MMM], DIR_MMM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MPP], x1 + DX1[DIR_PMM], x2 + DX2[DIR_PMM], x3 + DX3[DIR_PMM], DIR_PMM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PMP], x1 + DX1[DIR_MPM], x2 + DX2[DIR_MPM], x3 + DX3[DIR_MPM], DIR_MPM);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MMP], x1 + DX1[DIR_PPM], x2 + DX2[DIR_PPM], x3 + DX3[DIR_PPM], DIR_PPM);

            h2[DIR_00P] = c1o2 * (htemp[DIR_00P] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h2[DIR_00P]);
            h2[DIR_P0P] = c1o2 * (htemp[DIR_P0P] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h2[DIR_P0P]);
            h2[DIR_M0P] = c1o2 * (htemp[DIR_M0P] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h2[DIR_M0P]);
            h2[DIR_0PP] = c1o2 * (htemp[DIR_0PP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h2[DIR_0PP]);
            h2[DIR_0MP] = c1o2 * (htemp[DIR_0MP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h2[DIR_0MP]);
            h2[DIR_PPP] = c1o2 * (htemp[DIR_PPP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h2[DIR_PPP]);
            h2[DIR_MPP] = c1o2 * (htemp[DIR_MPP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h2[DIR_MPP]);
            h2[DIR_PMP] = c1o2 * (htemp[DIR_PMP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h2[DIR_PMP]);
            h2[DIR_MMP] = c1o2 * (htemp[DIR_MMP] * (one_over_sqrt3 + vx3) + (c1o1 - one_over_sqrt3 - vx3) * h2[DIR_MMP]);

            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_00P], x1 + DX1[DIR_00M], x2 + DX2[DIR_00M], x3 + DX3[DIR_00M], DIR_00M);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_P0P], x1 + DX1[DIR_M0M], x2 + DX2[DIR_M0M], x3 + DX3[DIR_M0M], DIR_M0M);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_M0P], x1 + DX1[DIR_P0M], x2 + DX2[DIR_P0M], x3 + DX3[DIR_P0M], DIR_P0M);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0PP], x1 + DX1[DIR_0MM], x2 + DX2[DIR_0MM], x3 + DX3[DIR_0MM], DIR_0MM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0MP], x1 + DX1[DIR_0PM], x2 + DX2[DIR_0PM], x3 + DX3[DIR_0PM], DIR_0PM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PPP], x1 + DX1[DIR_MMM], x2 + DX2[DIR_MMM], x3 + DX3[DIR_MMM], DIR_MMM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MPP], x1 + DX1[DIR_PMM], x2 + DX2[DIR_PMM], x3 + DX3[DIR_PMM], DIR_PMM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PMP], x1 + DX1[DIR_MPM], x2 + DX2[DIR_MPM], x3 + DX3[DIR_MPM], DIR_MPM);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MMP], x1 + DX1[DIR_PPM], x2 + DX2[DIR_PPM], x3 + DX3[DIR_PPM], DIR_PPM);

            break;
        case DIR_00M:
            f[DIR_00M] = ftemp[DIR_00M] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * f[DIR_00M];
            f[DIR_P0M] = ftemp[DIR_P0M] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * f[DIR_P0M];
            f[DIR_M0M] = ftemp[DIR_M0M] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * f[DIR_M0M];
            f[DIR_0PM] = ftemp[DIR_0PM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * f[DIR_0PM];
            f[DIR_0MM] = ftemp[DIR_0MM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * f[DIR_0MM];
            f[DIR_PPM] = ftemp[DIR_PPM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * f[DIR_PPM];
            f[DIR_MPM] = ftemp[DIR_MPM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * f[DIR_MPM];
            f[DIR_PMM] = ftemp[DIR_PMM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * f[DIR_PMM];
            f[DIR_MMM] = ftemp[DIR_MMM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * f[DIR_MMM];

            distributions->setPreCollisionDistributionForDirection(f[DIR_00M], x1 + DX1[DIR_00P], x2 + DX2[DIR_00P], x3 + DX3[DIR_00P], DIR_00P);
            distributions->setPreCollisionDistributionForDirection(f[DIR_P0M], x1 + DX1[DIR_M0P], x2 + DX2[DIR_M0P], x3 + DX3[DIR_M0P], DIR_M0P);
            distributions->setPreCollisionDistributionForDirection(f[DIR_M0M], x1 + DX1[DIR_P0P], x2 + DX2[DIR_P0P], x3 + DX3[DIR_P0P], DIR_P0P);
            distributions->setPreCollisionDistributionForDirection(f[DIR_0PM], x1 + DX1[DIR_0MP], x2 + DX2[DIR_0MP], x3 + DX3[DIR_0MP], DIR_0MP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_0MM], x1 + DX1[DIR_0PP], x2 + DX2[DIR_0PP], x3 + DX3[DIR_0PP], DIR_0PP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PPM], x1 + DX1[DIR_MMP], x2 + DX2[DIR_MMP], x3 + DX3[DIR_MMP], DIR_MMP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MPM], x1 + DX1[DIR_PMP], x2 + DX2[DIR_PMP], x3 + DX3[DIR_PMP], DIR_PMP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_PMM], x1 + DX1[DIR_MPP], x2 + DX2[DIR_MPP], x3 + DX3[DIR_MPP], DIR_MPP);
            distributions->setPreCollisionDistributionForDirection(f[DIR_MMM], x1 + DX1[DIR_PPP], x2 + DX2[DIR_PPP], x3 + DX3[DIR_PPP], DIR_PPP);

            h[DIR_00M] = htemp[DIR_00M] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h[DIR_00M];
            h[DIR_P0M] = htemp[DIR_P0M] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h[DIR_P0M];
            h[DIR_M0M] = htemp[DIR_M0M] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h[DIR_M0M];
            h[DIR_0PM] = htemp[DIR_0PM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h[DIR_0PM];
            h[DIR_0MM] = htemp[DIR_0MM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h[DIR_0MM];
            h[DIR_PPM] = htemp[DIR_PPM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h[DIR_PPM];
            h[DIR_MPM] = htemp[DIR_MPM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h[DIR_MPM];
            h[DIR_PMM] = htemp[DIR_PMM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h[DIR_PMM];
            h[DIR_MMM] = htemp[DIR_MMM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h[DIR_MMM];

            distributionsH->setPreCollisionDistributionForDirection(h[DIR_00M], x1 + DX1[DIR_00P], x2 + DX2[DIR_00P], x3 + DX3[DIR_00P], DIR_00P);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_P0M], x1 + DX1[DIR_M0P], x2 + DX2[DIR_M0P], x3 + DX3[DIR_M0P], DIR_M0P);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_M0M], x1 + DX1[DIR_P0P], x2 + DX2[DIR_P0P], x3 + DX3[DIR_P0P], DIR_P0P);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0PM], x1 + DX1[DIR_0MP], x2 + DX2[DIR_0MP], x3 + DX3[DIR_0MP], DIR_0MP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_0MM], x1 + DX1[DIR_0PP], x2 + DX2[DIR_0PP], x3 + DX3[DIR_0PP], DIR_0PP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PPM], x1 + DX1[DIR_MMP], x2 + DX2[DIR_MMP], x3 + DX3[DIR_MMP], DIR_MMP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MPM], x1 + DX1[DIR_PMP], x2 + DX2[DIR_PMP], x3 + DX3[DIR_PMP], DIR_PMP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_PMM], x1 + DX1[DIR_MPP], x2 + DX2[DIR_MPP], x3 + DX3[DIR_MPP], DIR_MPP);
            distributionsH->setPreCollisionDistributionForDirection(h[DIR_MMM], x1 + DX1[DIR_PPP], x2 + DX2[DIR_PPP], x3 + DX3[DIR_PPP], DIR_PPP);

            h2[DIR_00M] = c1o2 * (htemp[DIR_00M] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h2[DIR_00M]);
            h2[DIR_P0M] = c1o2 * (htemp[DIR_P0M] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h2[DIR_P0M]);
            h2[DIR_M0M] = c1o2 * (htemp[DIR_M0M] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h2[DIR_M0M]);
            h2[DIR_0PM] = c1o2 * (htemp[DIR_0PM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h2[DIR_0PM]);
            h2[DIR_0MM] = c1o2 * (htemp[DIR_0MM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h2[DIR_0MM]);
            h2[DIR_PPM] = c1o2 * (htemp[DIR_PPM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h2[DIR_PPM]);
            h2[DIR_MPM] = c1o2 * (htemp[DIR_MPM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h2[DIR_MPM]);
            h2[DIR_PMM] = c1o2 * (htemp[DIR_PMM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h2[DIR_PMM]);
            h2[DIR_MMM] = c1o2 * (htemp[DIR_MMM] * (one_over_sqrt3 - vx3) + (c1o1 - one_over_sqrt3 + vx3) * h2[DIR_MMM]);

            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_00M], x1 + DX1[DIR_00P], x2 + DX2[DIR_00P], x3 + DX3[DIR_00P], DIR_00P);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_P0M], x1 + DX1[DIR_M0P], x2 + DX2[DIR_M0P], x3 + DX3[DIR_M0P], DIR_M0P);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_M0M], x1 + DX1[DIR_P0P], x2 + DX2[DIR_P0P], x3 + DX3[DIR_P0P], DIR_P0P);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0PM], x1 + DX1[DIR_0MP], x2 + DX2[DIR_0MP], x3 + DX3[DIR_0MP], DIR_0MP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_0MM], x1 + DX1[DIR_0PP], x2 + DX2[DIR_0PP], x3 + DX3[DIR_0PP], DIR_0PP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PPM], x1 + DX1[DIR_MMP], x2 + DX2[DIR_MMP], x3 + DX3[DIR_MMP], DIR_MMP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MPM], x1 + DX1[DIR_PMP], x2 + DX2[DIR_PMP], x3 + DX3[DIR_PMP], DIR_PMP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_PMM], x1 + DX1[DIR_MPP], x2 + DX2[DIR_MPP], x3 + DX3[DIR_MPP], DIR_MPP);
            distributionsH2->setPreCollisionDistributionForDirection(h2[DIR_MMM], x1 + DX1[DIR_PPP], x2 + DX2[DIR_PPP], x3 + DX3[DIR_PPP], DIR_PPP);

            break;
        default:
            UB_THROW(UbException(UB_EXARGS, "It isn't implemented non reflecting density boundary for this direction!"));
    }
}
