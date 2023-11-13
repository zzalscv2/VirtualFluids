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
//! \file IBsharpInterfaceLBMKernel.cpp
//! \ingroup LBMKernel
//! \author M. Geier, K. Kutscher, Hesameddin Safari
//=======================================================================================

#include "BCArray3D.h"
#include "Block3D.h"
#include "D3Q27EsoTwist3DSplittedVector.h"
#include "D3Q27System.h"
#include "DataSet3D.h"
#include "LBMKernel.h"
#include "IBsharpInterfaceLBMKernel.h"
#include "NonNewtonianFluids/LBM/Rheology.h"
#include <cmath>
#include <iostream>
#include <string>

#define PROOF_CORRECTNESS

using namespace vf::lbm::dir;
using namespace vf::basics::constant;

//////////////////////////////////////////////////////////////////////////
IBsharpInterfaceLBMKernel::IBsharpInterfaceLBMKernel()
{
    this->compressible = false;
}
//////////////////////////////////////////////////////////////////////////
void IBsharpInterfaceLBMKernel::initDataSet()
{
    SPtr<DistributionArray3D> f(new D3Q27EsoTwist3DSplittedVector(nx[0] + 4, nx[1] + 4, nx[2] + 4, -999.9));
    SPtr<DistributionArray3D> h(new D3Q27EsoTwist3DSplittedVector(nx[0] + 4, nx[1] + 4, nx[2] + 4, -999.9)); // For phase-field
    // SPtr<DistributionArray3D> h2(new D3Q27EsoTwist3DSplittedVector(nx[0] + 4, nx[1] + 4, nx[2] + 4, -999.9));
    // SPtr<PhaseFieldArray3D> divU1(new PhaseFieldArray3D(            nx[0] + 4, nx[1] + 4, nx[2] + 4, 0.0));
    // CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr pressure(new  CbArray3D<real, IndexerX3X2X1>(    nx[0] + 4, nx[1] + 4, nx[2] + 4, 0.0));
    // pressureOld = CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr(new  CbArray3D<real, IndexerX3X2X1>(nx[0] + 4, nx[1] + 4, nx[2] + 4, 0.0));
    // p1Old = CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr(new  CbArray3D<real, IndexerX3X2X1>(nx[0] + 4, nx[1] + 4, nx[2] + 4, 0.0));

    rhoNode = CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr(new CbArray3D<real, IndexerX3X2X1>(nx[0] + 4, nx[1] + 4, nx[2] + 4, 0.0));
    vxNode = CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr(new CbArray3D<real, IndexerX3X2X1>(nx[0] + 4, nx[1] + 4, nx[2] + 4, 0.0));
    vyNode = CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr(new CbArray3D<real, IndexerX3X2X1>(nx[0] + 4, nx[1] + 4, nx[2] + 4, 0.0));
    vzNode = CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr(new CbArray3D<real, IndexerX3X2X1>(nx[0] + 4, nx[1] + 4, nx[2] + 4, 0.0));
    dataSet->setFdistributions(f);
    dataSet->setHdistributions(h); // For phase-field
    // dataSet->setH2distributions(h2);
    // dataSet->setPhaseField(divU1);
    // dataSet->setPressureField(pressure);

    phaseField = CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr(new CbArray3D<real, IndexerX3X2X1>(nx[0] + 4, nx[1] + 4, nx[2] + 4, -999.0));
    phaseFieldOld = CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr(new CbArray3D<real, IndexerX3X2X1>(nx[0] + 4, nx[1] + 4, nx[2] + 4, 999.0));

    particleData = std::make_shared<CbArray3D<SPtr<IBdynamicsParticleData>, IndexerX3X2X1>>(nx[0] + 4, nx[1] + 4, nx[2] + 4);

    // divU = CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr(new CbArray3D<real, IndexerX3X2X1>(nx[0] + 4, nx[1] + 4, nx[2] + 4, 0.0));
    int minX1 = 0;
    int minX2 = 0;
    int minX3 = 0;
    int maxX1 = nx[0] + 4;
    int maxX2 = nx[1] + 4;
    int maxX3 = nx[2] + 4;

        for (int x3 = minX3; x3 < maxX3; x3++) {
        for (int x2 = minX2; x2 < maxX2; x2++) {
            for (int x1 = minX1; x1 < maxX1; x1++) {
                (*particleData)(x1, x2, x3) = std::make_shared<IBdynamicsParticleData>();
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////
SPtr<LBMKernel> IBsharpInterfaceLBMKernel::clone()
{
    SPtr<LBMKernel> kernel(new IBsharpInterfaceLBMKernel());
    kernel->setNX(nx);
    dynamicPointerCast<IBsharpInterfaceLBMKernel>(kernel)->initDataSet();
    kernel->setCollisionFactorMultiphase(this->collFactorL, this->collFactorG);
    kernel->setDensityRatio(this->densityRatio);
    // kernel->setMultiphaseModelParameters(this->beta, this->kappa);
    kernel->setSigma(this->sigma);
    kernel->setContactAngle(this->contactAngle);
    kernel->setPhiL(this->phiL);
    kernel->setPhiH(this->phiH);
    kernel->setPhaseFieldRelaxation(this->tauH);
    kernel->setMobility(this->mob);
    kernel->setInterfaceWidth(this->interfaceWidth);

    kernel->setBCSet(bcSet->clone(kernel));
    kernel->setWithForcing(withForcing);
    kernel->setForcingX1(muForcingX1);
    kernel->setForcingX2(muForcingX2);
    kernel->setForcingX3(muForcingX3);
    kernel->setIndex(ix1, ix2, ix3);
    kernel->setDeltaT(deltaT);
    kernel->setGhostLayerWidth(2);
    dynamicPointerCast<IBsharpInterfaceLBMKernel>(kernel)->initForcing();

    return kernel;
}
//////////////////////////////////////////////////////////////////////////
void IBsharpInterfaceLBMKernel::forwardInverseChimeraWithKincompressible(real &mfa, real &mfb, real &mfc, real vv, real v2, real Kinverse, real K, real oneMinusRho)
{
    // using namespace UbMath;
    real m2 = mfa + mfc;
    real m1 = mfc - mfa;
    real m0 = m2 + mfb;
    mfa = m0;
    m0 *= Kinverse;
    m0 += oneMinusRho;
    mfb = (m1 * Kinverse - m0 * vv) * K;
    mfc = ((m2 - c2o1 * m1 * vv) * Kinverse + v2 * m0) * K;
}

////////////////////////////////////////////////////////////////////////////////
void IBsharpInterfaceLBMKernel::backwardInverseChimeraWithKincompressible(real &mfa, real &mfb, real &mfc, real vv, real v2, real Kinverse, real K, real oneMinusRho)
{
    // using namespace UbMath;
    real m0 = (((mfc - mfb) * c1o2 + mfb * vv) * Kinverse + (mfa * Kinverse + oneMinusRho) * (v2 - vv) * c1o2) * K;
    real m1 = (((mfa - mfc) - c2o1 * mfb * vv) * Kinverse + (mfa * Kinverse + oneMinusRho) * (-v2)) * K;
    mfc = (((mfc + mfb) * c1o2 + mfb * vv) * Kinverse + (mfa * Kinverse + oneMinusRho) * (v2 + vv) * c1o2) * K;
    mfa = m0;
    mfb = m1;
}

////////////////////////////////////////////////////////////////////////////////
void IBsharpInterfaceLBMKernel::forwardChimera(real &mfa, real &mfb, real &mfc, real vv, real v2)
{
    // using namespace UbMath;
    real m1 = (mfa + mfc) + mfb;
    real m2 = mfc - mfa;
    mfc = (mfc + mfa) + (v2 * m1 - c2o1 * vv * m2);
    mfb = m2 - vv * m1;
    mfa = m1;
}

void IBsharpInterfaceLBMKernel::backwardChimera(real &mfa, real &mfb, real &mfc, real vv, real v2)
{
    // using namespace UbMath;
    real ma = (mfc + mfa * (v2 - vv)) * c1o2 + mfb * (vv - c1o2);
    real mb = ((mfa - mfc) - mfa * v2) - c2o1 * mfb * vv;
    mfc = (mfc + mfa * (v2 + vv)) * c1o2 + mfb * (vv + c1o2);
    mfb = mb;
    mfa = ma;
}

void IBsharpInterfaceLBMKernel::calculate(int step)
{
    using namespace D3Q27System;
    // using namespace UbMath;

    forcingX1 = 0.0;
    forcingX2 = 0.0;
    forcingX3 = 0.0;

    real oneOverInterfaceScale = c4o1 / interfaceWidth; // 1.0;//1.5;
                                                        /////////////////////////////////////

    localDistributionsF = dynamicPointerCast<D3Q27EsoTwist3DSplittedVector>(dataSet->getFdistributions())->getLocalDistributions();
    nonLocalDistributionsF = dynamicPointerCast<D3Q27EsoTwist3DSplittedVector>(dataSet->getFdistributions())->getNonLocalDistributions();
    restDistributionsF = dynamicPointerCast<D3Q27EsoTwist3DSplittedVector>(dataSet->getFdistributions())->getZeroDistributions();

    localDistributionsH1 = dynamicPointerCast<D3Q27EsoTwist3DSplittedVector>(dataSet->getHdistributions())->getLocalDistributions();
    nonLocalDistributionsH1 = dynamicPointerCast<D3Q27EsoTwist3DSplittedVector>(dataSet->getHdistributions())->getNonLocalDistributions();
    restDistributionsH1 = dynamicPointerCast<D3Q27EsoTwist3DSplittedVector>(dataSet->getHdistributions())->getZeroDistributions();

    CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr pressure = dataSet->getPressureField();

    SPtr<BCArray3D> bcArray = this->getBCSet()->getBCArray();

    const int bcArrayMaxX1 = (int)bcArray->getNX1();
    const int bcArrayMaxX2 = (int)bcArray->getNX2();
    const int bcArrayMaxX3 = (int)bcArray->getNX3();

    int minX1 = ghostLayerWidth;
    int minX2 = ghostLayerWidth;
    int minX3 = ghostLayerWidth;
    int maxX1 = bcArrayMaxX1 - ghostLayerWidth;
    int maxX2 = bcArrayMaxX2 - ghostLayerWidth;
    int maxX3 = bcArrayMaxX3 - ghostLayerWidth;
    // real omegaDRho = 1.0;// 1.25;// 1.3;
    for (int x3 = minX3 - ghostLayerWidth; x3 < maxX3 + ghostLayerWidth; x3++) {
        for (int x2 = minX2 - ghostLayerWidth; x2 < maxX2 + ghostLayerWidth; x2++) {
            for (int x1 = minX1 - ghostLayerWidth; x1 < maxX1 + ghostLayerWidth; x1++) {
                if (!bcArray->isSolid(x1, x2, x3) && !bcArray->isUndefined(x1, x2, x3)) {
                    int x1p = x1 + 1;
                    int x2p = x2 + 1;
                    int x3p = x3 + 1;

                    real mfcbb = (*this->localDistributionsH1)(D3Q27System::ET_E, x1, x2, x3);
                    real mfbcb = (*this->localDistributionsH1)(D3Q27System::ET_N, x1, x2, x3);
                    real mfbbc = (*this->localDistributionsH1)(D3Q27System::ET_T, x1, x2, x3);
                    real mfccb = (*this->localDistributionsH1)(D3Q27System::ET_NE, x1, x2, x3);
                    real mfacb = (*this->localDistributionsH1)(D3Q27System::ET_NW, x1p, x2, x3);
                    real mfcbc = (*this->localDistributionsH1)(D3Q27System::ET_TE, x1, x2, x3);
                    real mfabc = (*this->localDistributionsH1)(D3Q27System::ET_TW, x1p, x2, x3);
                    real mfbcc = (*this->localDistributionsH1)(D3Q27System::ET_TN, x1, x2, x3);
                    real mfbac = (*this->localDistributionsH1)(D3Q27System::ET_TS, x1, x2p, x3);
                    real mfccc = (*this->localDistributionsH1)(D3Q27System::ET_TNE, x1, x2, x3);
                    real mfacc = (*this->localDistributionsH1)(D3Q27System::ET_TNW, x1p, x2, x3);
                    real mfcac = (*this->localDistributionsH1)(D3Q27System::ET_TSE, x1, x2p, x3);
                    real mfaac = (*this->localDistributionsH1)(D3Q27System::ET_TSW, x1p, x2p, x3);
                    real mfabb = (*this->nonLocalDistributionsH1)(D3Q27System::ET_W, x1p, x2, x3);
                    real mfbab = (*this->nonLocalDistributionsH1)(D3Q27System::ET_S, x1, x2p, x3);
                    real mfbba = (*this->nonLocalDistributionsH1)(D3Q27System::ET_B, x1, x2, x3p);
                    real mfaab = (*this->nonLocalDistributionsH1)(D3Q27System::ET_SW, x1p, x2p, x3);
                    real mfcab = (*this->nonLocalDistributionsH1)(D3Q27System::ET_SE, x1, x2p, x3);
                    real mfaba = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BW, x1p, x2, x3p);
                    real mfcba = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BE, x1, x2, x3p);
                    real mfbaa = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BS, x1, x2p, x3p);
                    real mfbca = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BN, x1, x2, x3p);
                    real mfaaa = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BSW, x1p, x2p, x3p);
                    real mfcaa = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BSE, x1, x2p, x3p);
                    real mfaca = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BNW, x1p, x2, x3p);
                    real mfcca = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BNE, x1, x2, x3p);
                    real mfbbb = (*this->restDistributionsH1)(x1, x2, x3);

                    // omegaDRho = 2.0;// 1.5;
                    // real phiOld = (*phaseField)(x1, x2, x3);

                    (*phaseField)(x1, x2, x3) =
                        (((mfaaa + mfccc) + (mfaca + mfcac)) + ((mfaac + mfcca) + (mfcaa + mfacc))) + (((mfaab + mfacb) + (mfcab + mfccb)) + ((mfaba + mfabc) + (mfcba + mfcbc)) + ((mfbaa + mfbac) + (mfbca + mfbcc))) + ((mfabb + mfcbb) + (mfbab + mfbcb) + (mfbba + mfbbc)) + mfbbb;

                    if ((*phaseField)(x1, x2, x3) > 1) {
                        (*phaseField)(x1, x2, x3) = c1o1;
                    }

                    if ((*phaseField)(x1, x2, x3) < 0) {
                        (*phaseField)(x1, x2, x3) = 0;
                    }
                }
            }
        }
    }

    this->swapDistributions();
    for (int x3 = minX3 - ghostLayerWidth + 1; x3 < maxX3 + ghostLayerWidth - 1; x3++) {
        for (int x2 = minX2 - ghostLayerWidth + 1; x2 < maxX2 + ghostLayerWidth - 1; x2++) {
            for (int x1 = minX1 - ghostLayerWidth + 1; x1 < maxX1 + ghostLayerWidth - 1; x1++) {
                if (!bcArray->isSolid(x1, x2, x3) && !bcArray->isUndefined(x1, x2, x3)) {
                    // int x1p = x1 + 1;
                    // int x2p = x2 + 1;
                    // int x3p = x3 + 1;

                    SPtr<DistributionArray3D> distributionH = this->getDataSet()->getHdistributions();
                    real hh[27];
                    distributionH->getPostCollisionDistribution(hh, x1, x2, x3);
                    real phiD, vxP, vyP, vzP;

                    D3Q27System::calcIncompMacroscopicValues(hh, phiD, vxP, vyP, vzP);
                    (*phaseFieldOld)(x1, x2, x3) = phiD;

                    SPtr<DistributionArray3D> distribution = this->getDataSet()->getFdistributions();
                    real ff[27];
                    distribution->getPostCollisionDistribution(ff, x1, x2, x3);
                    real rhoG, vx, vy, vz;

                    D3Q27System::calcIncompMacroscopicValues(ff, rhoG, vx, vy, vz);

                    (*rhoNode)(x1, x2, x3) = rhoG; // *((*phaseField)(x1, x2, x3) > c1o2 ? densityRatio : c1o1);
                    (*vxNode)(x1, x2, x3) = vx;
                    (*vyNode)(x1, x2, x3) = vy;
                    (*vzNode)(x1, x2, x3) = vz;
                }
            }
        }
    }

    SPtr<DistributionArray3D> distribution = this->getDataSet()->getFdistributions();
    real ff[27];
    for (int x3 = minX3 - 1; x3 < maxX3 + 1; x3++) {
        for (int x2 = minX2 - 1; x2 < maxX2 + 1; x2++) {
            for (int x1 = minX1 - 1; x1 < maxX1 + 1; x1++) {
                if (!bcArray->isSolid(x1, x2, x3) && !bcArray->isUndefined(x1, x2, x3)) {
                    // int x1p = x1 + 1;
                    // int x2p = x2 + 1;
                    // int x3p = x3 + 1;
                    findNeighbors(phaseFieldOld, x1, x2, x3);
                    ////////////////////////////////Momentum conservation experiment 06.03.2023
                    // surfacetension

                    if ((((*phaseField)(x1, x2, x3) <= c1o2) || phi[d000] <= c1o2) &&
                        ((phi[dP00] > c1o2) || (phi[dM00] > c1o2) || (phi[d00P] > c1o2) || (phi[d00M] > c1o2) || (phi[d0M0] > c1o2) || (phi[d0P0] > c1o2) || (phi[dPP0] > c1o2) || (phi[dPM0] > c1o2) || (phi[dP0P] > c1o2) || (phi[dP0M] > c1o2) || (phi[dMP0] > c1o2) ||
                         (phi[dMM0] > c1o2) || (phi[dM0P] > c1o2) || (phi[dM0M] > c1o2) || (phi[d0PM] > c1o2) || (phi[d0MM] > c1o2) || (phi[d0PP] > c1o2) || (phi[d0MP] > c1o2) || (phi[dPPP] > c1o2) || (phi[dPMP] > c1o2) || (phi[dMPP] > c1o2) || (phi[dMMP] > c1o2) ||
                         (phi[dPPM] > c1o2) || (phi[dPMM] > c1o2) || (phi[dMPM] > c1o2) || (phi[dMMM] > c1o2))) {
                        real vx = (*vxNode)(x1, x2, x3);
                        real vy = (*vyNode)(x1, x2, x3);
                        real vz = (*vzNode)(x1, x2, x3);
                        findNeighbors(phaseField, x1, x2, x3);
                        real laplacePressure = c12o1 * sigma * computeCurvature_phi();
                        //                  if (step > 5000)
                        //                       UBLOG(logINFO, x1 << ","<< x2 << ","<< x3 << " "<< "3*dP=" << laplacePressure << " dP=" << laplacePressure / 3.0<< " phi=" << phi[d000]<< "\n");
                        findNeighbors(phaseFieldOld, x1, x2, x3);

                        // 16.03.23 c: BB gas side with updated boundary velocity

                        distribution->getPostCollisionDistribution(ff, x1, x2, x3);
                        real rhoG;
                        if (phi[d000] > c1o2) { // initialization necessary
                            real sumRho = 0;
                            real sumWeight = 1.e-100;
                            for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                if ((phi[fdir] <= c1o2)) {
                                    sumRho += WEIGTH[fdir] * (*rhoNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]);
                                    sumWeight += WEIGTH[fdir];
                                }
                            }
                            rhoG = sumRho / sumWeight; // uncheck excpetion: what if there is no adequate neighbor?
                            for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                if ((phi[fdir] > c1o2)) {
                                    real vxBC = ((*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                    real vyBC = ((*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                    real vzBC = ((*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                    real vBC = (D3Q27System::DX1[fdir] * vxBC + D3Q27System::DX2[fdir] * vyBC + D3Q27System::DX3[fdir] * vzBC);
                                    real vDir = (D3Q27System::DX1[fdir] * vx + D3Q27System::DX2[fdir] * vy + D3Q27System::DX3[fdir] * vz);
                                    vBC = (vBC + vDir) / (c2o1 + vBC - vDir);
                                    real fL = distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]);

                                    if ((phi[D3Q27System::INVDIR[fdir]] > c1o2)) {
                                        /// here we need reconstruction from scrach
                                        real feqOLD = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], (*rhoNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]),
                                                                                            (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]),
                                                                                            (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                        real feqNew = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], rhoG, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]),
                                                                                            (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                        // real fBC = (fL - feqOLD) * (c1o1 / collFactorG - c1o1) / (c1o1 / collFactorL - c1o1) + feqNew;

                                        // real fG = distribution->getDistributionInvForDirection(x1, x2, x3, fdir);
                                        // real fGEQOld = D3Q27System::getIncompFeqForDirection(fdir, (*rhoNode)(x1, x2, x3), vx, vy, vz);
                                        // real fGEQNew = D3Q27System::getIncompFeqForDirection(fdir, rhoG, vx, vy, vz);
                                        real fBC = (fL - feqOLD) * (c1o1 / collFactorG - c1o1) / (c1o1 / collFactorL - c1o1) + feqNew; // fL -feqOLD + feqNew;
                                                                                                                                       // real fBC = fGG - c6o1 * WEIGTH[fdir] * (vBC);
                                        distribution->setPostCollisionDistributionForDirection(fBC, x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]);
                                        ///// other possibility is tor replace the node itself instead of the neighbor (only c1o1 of them is allowed!)
                                        // real fG = distribution->getDistributionInvForDirection(x1, x2, x3, fdir);
                                        // real feqOLD = D3Q27System::getIncompFeqForDirection(fdir, (*rhoNode)(x1 , x2 , x3 ), (*vxNode)(x1 , x2 , x3 ), (*vyNode)(x1 , x2 , x3 ), (*vzNode)(x1 , x2 , x3 ));
                                        // real feqNew = D3Q27System::getIncompFeqForDirection(fdir, rhoG, (*vxNode)(x1 , x2 , x3 ), (*vyNode)(x1 , x2 , x3 ), (*vzNode)(x1, x2, x3 ));
                                        // real fBC = fG - feqOLD + feqNew;
                                        // distribution->setPostCollisionDistributionForDirection(fBC, x1, x2, x3, fdir);
                                    }
                                }
                            }
                            // distribution->setPostCollisionDistributionForDirection(D3Q27System::getIncompFeqForDirection(d000, rhoG, vx, vy, vz), x1, x2, x3, d000);
                            {
                                real fL = distribution->getDistributionInvForDirection(x1, x2, x3, d000);
                                real feqOLD = D3Q27System::getIncompFeqForDirection(d000, (*rhoNode)(x1, x2, x3), vx, vy, vz);
                                real feqNew = D3Q27System::getIncompFeqForDirection(d000, rhoG, vx, vy, vz);
                                distribution->setPostCollisionDistributionForDirection(fL - feqOLD + feqNew, x1, x2, x3, d000);
                            }

                        } else { // no refill of gas required
                            rhoG = (*rhoNode)(x1, x2, x3);
                            if ((*phaseField)(x1, x2, x3) <= c1o2) { // no refill liquid
                                for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                    if ((phi[fdir] > c1o2)) {
                                        real vxBC = ((*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                        real vyBC = ((*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                        real vzBC = ((*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                        real vBC = (D3Q27System::DX1[fdir] * vxBC + D3Q27System::DX2[fdir] * vyBC + D3Q27System::DX3[fdir] * vzBC);
                                        real vDir = (D3Q27System::DX1[fdir] * vx + D3Q27System::DX2[fdir] * vy + D3Q27System::DX3[fdir] * vz);
                                        // real dvDir = vBC - vDir;
                                        vBC = (vBC + vDir) / (c2o1 + vBC - vDir);
                                        real fL = distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]);
                                        real fG = distribution->getDistributionInvForDirection(x1, x2, x3, fdir);
                                        real fBC = fG - c6o1 * WEIGTH[fdir] * (vBC);
                                        // real fGEQ = D3Q27System::getIncompFeqForDirection(fdir, rhoG, vx, vy, vz);
                                        //  real fBC = (-fGInv + fGInvEQ + fGEQ - c6o1 * WEIGTH[fdir] * dvDir * (c1o1 / collFactorG - c1o1)) - c6o1 * WEIGTH[fdir] * (vBC);
                                        // real fBC = (fGEQ - c3o1 * WEIGTH[fdir] * dvDir * (c1o1 / collFactorG - c1o1)) - c6o1 * WEIGTH[fdir] * (vBC);

                                        // real feqOLD = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], (*rhoNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                        // D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])); real feqNew =
                                        // D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], rhoG, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                        // D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])); real fG = distribution->getDistributionInvForDirection(x1, x2, x3, fdir); real fBC = (fL - feqOLD) * (c1o1 / collFactorG - c1o1) /
                                        // (c1o1 / collFactorL - c1o1) + feqNew;

                                        // if ((*phaseField)(x1, x2, x3) <= c1o2)
                                        distribution->setPostCollisionDistributionForDirection(fBC, x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]);
                                        if (((*phaseField)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])) > c1o2) {
                                            // real vxBC = c1o2 * (vx + (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                            // real vyBC = c1o2 * (vy + (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                            // real vzBC = c1o2 * (vz + (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                            // real feqL = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                            // D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])); real feqG = D3Q27System::getIncompFeqForDirection(fdir, 0, vx, vy, vz); real feqL =
                                            // D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX1[fdir]) * (D3Q27System::DX1[fdir]), (*vyNode)(x1 +
                                            // D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX2[fdir]) * (D3Q27System::DX2[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) *
                                            // (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir])); real feqG = D3Q27System::getIncompFeqForDirection(fdir, 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX1[fdir]) *
                                            // (D3Q27System::DX1[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX2[fdir]) * (D3Q27System::DX2[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                            // D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir])); real feqG = D3Q27System::getIncompFeqForDirection(fdir, 0, vx * (D3Q27System::DX1[fdir]) * (D3Q27System::DX1[fdir]), vy * (D3Q27System::DX2[fdir]) *
                                            // (D3Q27System::DX2[fdir]), vz * (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]));

                                            // distribution->setPostCollisionDistributionForDirection((fBC + fG) / densityRatio*0 - fL  - (feqG - feqL) * (c1o1 / densityRatio*0 - c1o1) * vBC, x1, x2, x3, fdir);// (vxBC * D3Q27System::DX1[fdir] + vyBC * D3Q27System::DX2[fdir] + vzBC * D3Q27System::DX3[fdir]), x1,
                                            // x2, x3, fdir); distribution->setPostCollisionDistributionForDirection((fBC + fG) / densityRatio * 0 - fL - (feqG - feqL-2*fL+2*feqL) * (c1o1 / densityRatio - c1o1) * vBC, x1, x2, x3, fdir);// (vxBC * D3Q27System::DX1[fdir] + vyBC * D3Q27System::DX2[fdir] + vzBC *
                                            // D3Q27System::DX3[fdir]), x1, x2, x3, fdir); real flW = (fBC + fG) / densityRatio * 0 - fL - (feqG - feqL) * (c1o1 / densityRatio*0 - c1o1) * vBC; real flWW = (fBC + fG) / densityRatio * 0 - fL - (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio*0
                                            // - c1o1) * vBC; real fLi = distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], fdir); real number = 666; distribution->setPostCollisionDistributionForDirection((fBC + fG) / densityRatio *
                                            // 0 - fL - (feqG - feqL) * (c1o1 / densityRatio * 0 - c1o1) * vBC, x1, x2, x3, fdir);
                                            ////	real eqBC= D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, vx, vy, vz);
                                            ////	real eqG = D3Q27System::getIncompFeqForDirection(fdir, 0, vx, vy, vz);
                                            //	real eqBC = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                            //D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])); 	real eqG = D3Q27System::getIncompFeqForDirection(fdir, 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                            //D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));

                                            ////real flNew = (fBC + fG-eqBC-eqG) / densityRatio +eqBC+eqG - fL - (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio  - c1o1) * vBC;
                                            //	distribution->setPostCollisionDistributionForDirection(c2o1*laplacePressure * WEIGTH[fdir] +(fBC + fG - eqBC - eqG) / densityRatio + (eqBC + eqG) - fL, x1, x2, x3, fdir);// -0* (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio  - c1o1) * vBC, x1, x2, x3,
                                            //fdir);// (vxBC * D3Q27System::DX1[fdir] + vyBC * D3Q27System::DX2[fdir] + vzBC * D3Q27System::DX3[fdir]), x1, x2, x3, fdir);
                                            //	//if (vxBC != 0) {
                                            //	//	int set = 0;
                                            //	//}

                                            real feqL = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX1[fdir]) * (D3Q27System::DX1[fdir]),
                                                                                              (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX2[fdir]) * (D3Q27System::DX2[fdir]),
                                                                                              (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]));
                                            real feqG = D3Q27System::getIncompFeqForDirection(fdir, 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX1[fdir]) * (D3Q27System::DX1[fdir]),
                                                                                              (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX2[fdir]) * (D3Q27System::DX2[fdir]),
                                                                                              (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]));

                                            real eqBC = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, vx, vy, vz);
                                            real eqG = D3Q27System::getIncompFeqForDirection(fdir, 0, vx, vy, vz);
                                            real eqBCN = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]),
                                                                                               (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                            real eqGN = D3Q27System::getIncompFeqForDirection(fdir, 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]),
                                                                                              (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));

                                            // real flNew = (fBC + fG-eqBC-eqG) / densityRatio +eqBC+eqG - fL - (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio  - c1o1) * vBC;
                                            real laplacePressureBC;
                                            if ((x1 + D3Q27System::DX1[fdir] > 0) && (x1 + D3Q27System::DX1[fdir] < maxX1 + 1) && (x2 + D3Q27System::DX2[fdir] > 0) && (x2 + D3Q27System::DX2[fdir] < maxX2 + 1) && (x3 + D3Q27System::DX3[fdir] > 0) && (x3 + D3Q27System::DX3[fdir] < maxX3 + 1)) {
                                                findNeighbors(phaseField, x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]);
                                                laplacePressureBC = c6o1 * c2o1 * computeCurvature_phi() * sigma;
                                                findNeighbors(phaseFieldOld, x1, x2, x3);
                                            } else
                                                laplacePressureBC = laplacePressure; // curv; // reset to the above
                                            laplacePressureBC = laplacePressure * (c1o1 - c2o1 * (*phaseField)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])) /
                                                                    (c2o1 * (*phaseField)(x1, x2, x3) - c2o1 * (*phaseField)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])) +
                                                                laplacePressureBC * (-c1o1 + c2o1 * (*phaseField)(x1, x2, x3)) / (c2o1 * (*phaseField)(x1, x2, x3) - c2o1 * (*phaseField)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                            // laplacePressureBC *= sigma;
                                            distribution->setPostCollisionDistributionForDirection(laplacePressureBC * WEIGTH[fdir] + (fBC + fG - eqBC - eqG) / densityRatio + (eqBCN + eqGN) * (c1o1 - c1o1 / densityRatio * 0) - fL - 0 * (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio - c1o1) * vBC, x1,
                                                                                      x2, x3, fdir);
                                        }
                                    }
                                }
                            } else { // refill liquid

                                for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                    if ((phi[fdir] > c1o2)) {
                                        real vxBC = ((*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                        real vyBC = ((*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                        real vzBC = ((*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                        real vBC = (D3Q27System::DX1[fdir] * vxBC + D3Q27System::DX2[fdir] * vyBC + D3Q27System::DX3[fdir] * vzBC);
                                        real vDir = (D3Q27System::DX1[fdir] * vx + D3Q27System::DX2[fdir] * vy + D3Q27System::DX3[fdir] * vz);
                                        // real dvDir = vBC - vDir;
                                        vBC = (vBC + vDir) / (c2o1 + vBC - vDir);
                                        real fL = distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]);
                                        real fG = distribution->getDistributionInvForDirection(x1, x2, x3, fdir);
                                        real fBC = fG - c6o1 * WEIGTH[fdir] * (vBC);

                                        // real fGEQ = D3Q27System::getIncompFeqForDirection(fdir, rhoG, vx, vy, vz);
                                        //  real fBC = (-fGInv + fGInvEQ + fGEQ - c6o1 * WEIGTH[fdir] * dvDir * (c1o1 / collFactorG - c1o1)) - c6o1 * WEIGTH[fdir] * (vBC);
                                        // real fBC = (fGEQ - c3o1 * WEIGTH[fdir] * dvDir * (c1o1 / collFactorG - c1o1)) - c6o1 * WEIGTH[fdir] * (vBC);

                                        // real feqOLD = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], (*rhoNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                        // D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])); real feqNew =
                                        // D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], rhoG, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                        // D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])); real fG = distribution->getDistributionInvForDirection(x1, x2, x3, fdir); real fBC = (fL - feqOLD) * (c1o1 / collFactorG - c1o1) /
                                        // (c1o1 / collFactorL - c1o1) + feqNew;

                                        ff[D3Q27System::INVDIR[fdir]] = fBC;
                                        if (((*phaseField)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])) > c1o2) {
                                            // real feqL = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                            // D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])); real feqG = D3Q27System::getIncompFeqForDirection(fdir, 0, vx, vy, vz); real feqL =
                                            // D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX1[fdir]) * (D3Q27System::DX1[fdir]), (*vyNode)(x1 +
                                            // D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX2[fdir]) * (D3Q27System::DX2[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) *
                                            // (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir])); real feqG = D3Q27System::getIncompFeqForDirection(fdir, 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX1[fdir]) *
                                            // (D3Q27System::DX1[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX2[fdir]) * (D3Q27System::DX2[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                            // D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir])); real feqG = D3Q27System::getIncompFeqForDirection(fdir, 0, vx * (D3Q27System::DX1[fdir]) * (D3Q27System::DX1[fdir]), vy * (D3Q27System::DX2[fdir]) *
                                            // (D3Q27System::DX2[fdir]), vz * (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]));

                                            // distribution->setPostCollisionDistributionForDirection((fBC + fG) / densityRatio*0 - fL- (feqG - feqL) * (c1o1 / densityRatio - c1o1) * (vBC), x1, x2, x3, fdir);
                                            // distribution->setPostCollisionDistributionForDirection((fBC + fG) / densityRatio * 0 - fL - (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio*0 - c1o1) * vBC, x1, x2, x3, fdir);// (vxBC * D3Q27System::DX1[fdir] + vyBC * D3Q27System::DX2[fdir] + vzBC *
                                            // D3Q27System::DX3[fdir]), x1, x2, x3, fdir); distribution->setPostCollisionDistributionForDirection(0, x1, x2, x3, fdir); real flW = (fBC + fG) / densityRatio * 0 - fL - (feqG - feqL) * (c1o1 / densityRatio * 0 - c1o1) * vBC; real flWW = (fBC + fG) / densityRatio * 0
                                            // - fL - (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio * 0 - c1o1) * vBC; real fLi = distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], fdir);
                                            //	real eqBC = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                            //D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])); 	real eqG = D3Q27System::getIncompFeqForDirection(fdir, 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 +
                                            //D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                            //	//real flNew = (fBC + fG - eqBC - eqG) / densityRatio + eqBC + eqG - fL - (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio - c1o1) * vBC;
                                            //	distribution->setPostCollisionDistributionForDirection(c2o1*laplacePressure* WEIGTH[fdir] + (fBC + fG - eqBC - eqG) / densityRatio + (eqBC + eqG)  - fL, x1, x2, x3, fdir);// - 0*(feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio - c1o1) * vBC, x1, x2, x3,
                                            //fdir);// (vxBC * D3Q27System::DX1[fdir] + vyBC * D3Q27System::DX2[fdir] + vzBC * D3Q27System::DX3[fdir]), x1, x2, x3, fdir);

                                            ////	real number = 666;

                                            real feqL = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX1[fdir]) * (D3Q27System::DX1[fdir]),
                                                                                              (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX2[fdir]) * (D3Q27System::DX2[fdir]),
                                                                                              (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]));
                                            real feqG = D3Q27System::getIncompFeqForDirection(fdir, 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX1[fdir]) * (D3Q27System::DX1[fdir]),
                                                                                              (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX2[fdir]) * (D3Q27System::DX2[fdir]),
                                                                                              (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]) * (D3Q27System::DX3[fdir]));

                                            real eqBC = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, vx, vy, vz);
                                            real eqG = D3Q27System::getIncompFeqForDirection(fdir, 0, vx, vy, vz);
                                            real eqBCN = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]),
                                                                                               (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                            real eqGN = D3Q27System::getIncompFeqForDirection(fdir, 0, (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]), (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]),
                                                                                              (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));

                                            // real flNew = (fBC + fG-eqBC-eqG) / densityRatio +eqBC+eqG - fL - (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio  - c1o1) * vBC;
                                            real laplacePressureBC;
                                            if ((x1 + D3Q27System::DX1[fdir] > 0) && (x1 + D3Q27System::DX1[fdir] < maxX1 + 1) && (x2 + D3Q27System::DX2[fdir] > 0) && (x2 + D3Q27System::DX2[fdir] < maxX2 + 1) && (x3 + D3Q27System::DX3[fdir] > 0) && (x3 + D3Q27System::DX3[fdir] < maxX3 + 1)) {
                                                findNeighbors(phaseField, x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]);
                                                laplacePressureBC = c12o1 * computeCurvature_phi() * sigma;
                                                findNeighbors(phaseFieldOld, x1, x2, x3);
                                            } else
                                                laplacePressureBC = laplacePressure; // curv; // reset to the above
                                            laplacePressureBC = laplacePressure * (c1o1 - c2o1 * (*phaseField)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])) /
                                                                    (c2o1 * (*phaseField)(x1, x2, x3) - c2o1 * (*phaseField)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])) +
                                                                laplacePressureBC * (-c1o1 + c2o1 * (*phaseField)(x1, x2, x3)) / (c2o1 * (*phaseField)(x1, x2, x3) - c2o1 * (*phaseField)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]));
                                            // laplacePressureBC *= sigma;
                                            distribution->setPostCollisionDistributionForDirection(laplacePressureBC * WEIGTH[fdir] + (fBC + fG - eqBC - eqG) / densityRatio + (eqBCN + eqGN) * (c1o1 - c1o1 / densityRatio * 0) - fL - 0 * (feqG - feqL - 2 * fL + 2 * feqL) * (c1o1 / densityRatio - c1o1) * vBC, x1,
                                                                                      x2, x3, fdir);
                                        }

                                    } else {
                                        ff[D3Q27System::INVDIR[fdir]] = distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]);
                                        ;
                                    }
                                }

                                real sum2 = 1e-100;
                                real sumRho = 0;
                                real sumVx = 0;
                                real sumVy = 0;
                                real sumVz = 0;
                                for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                    if ((phi[fdir] > c1o2)) {

                                        sumRho += WEIGTH[fdir] * (*rhoNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]); // * tempRho;
                                        sumVx += WEIGTH[fdir] * (*vxNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]);
                                        sumVy += WEIGTH[fdir] * (*vyNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]);
                                        sumVz += WEIGTH[fdir] * (*vzNode)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir]);
                                        sum2 += WEIGTH[fdir];
                                    }
                                }
                                real rhoL;
                                D3Q27System::calcIncompMacroscopicValues(ff, rhoG, vx, vy, vz);
                                rhoL = sumRho / sum2;
                                // vx = sumVx / sum2;
                                // vy = sumVy / sum2;
                                // vz = sumVz / sum2;
                                // rhoL = (*rhoNode)(x1, x2, x3)/densityRatio;

                                // for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                //	ff[D3Q27System::INVDIR[fdir]] = distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]);
                                // }

                                for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                    if (((phi[fdir] <= c1o2))) //&& (((*phaseField)(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir])) > c1o2)))
                                    {
                                        real feqOLD = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], rhoG, vx, vy, vz);
                                        real feqNew = D3Q27System::getIncompFeqForDirection(D3Q27System::INVDIR[fdir], rhoL, vx, vy, vz);
                                        ff[D3Q27System::INVDIR[fdir]] = (ff[D3Q27System::INVDIR[fdir]] - feqOLD) * (c1o1 / collFactorL - c1o1) / (c1o1 / collFactorG - c1o1) + feqNew;
                                        distribution->setPostCollisionDistributionForDirection(ff[D3Q27System::INVDIR[fdir]], x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]);
                                    }
                                }

                                // for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                //	if ((phi[D3Q27System::INVDIR[fdir]] <= c1o2) && (phi[fdir] > c1o2)) {
                                //		//real vxBC = ((*vxNode)(x1 - D3Q27System::DX1[fdir], x2 - D3Q27System::DX2[fdir], x3 - D3Q27System::DX3[fdir]));
                                //		//real vyBC = ((*vyNode)(x1 - D3Q27System::DX1[fdir], x2 - D3Q27System::DX2[fdir], x3 - D3Q27System::DX3[fdir]));
                                //		//real vzBC = ((*vzNode)(x1 - D3Q27System::DX1[fdir], x2 - D3Q27System::DX2[fdir], x3 - D3Q27System::DX3[fdir]));
                                //		//real vBC = -(D3Q27System::DX1[fdir] * vxBC + D3Q27System::DX2[fdir] * vyBC + D3Q27System::DX2[fdir] * vzBC);
                                //		real vDir = -(D3Q27System::DX1[fdir] * vx + D3Q27System::DX2[fdir] * vy + D3Q27System::DX2[fdir] * vz);
                                //		//vBC = (vBC + vDir) / (c2o1 -( vBC - vDir));
                                //		//real fL = distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]) - c6o1 * WEIGTH[fdir] * vDir;
                                //		//real fL = distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]) + c6o1 * WEIGTH[fdir] * (vx * D3Q27System::DX1[fdir] + vy * D3Q27System::DX2[fdir] + vz *
                                //D3Q27System::DX3[fdir]); 		real fL= D3Q27System::getIncompFeqForDirection(fdir, rhoL, vx, vy, vz); 		distribution->setPostCollisionDistributionForDirection(fL, x1 - D3Q27System::DX1[fdir], x2 - D3Q27System::DX2[fdir], x3 - D3Q27System::DX3[fdir], fdir); 		ff[fdir] = fL;
                                //	}
                                //	if (!(phi[fdir] > c1o2)) {
                                //		//std::cout << "Eq at dir=" << fdir << "\n";
                                //		real vxBC = ((*vxNode)(x1 - D3Q27System::DX1[fdir], x2 - D3Q27System::DX2[fdir], x3 - D3Q27System::DX3[fdir]));
                                //		real vyBC = ((*vyNode)(x1 - D3Q27System::DX1[fdir], x2 - D3Q27System::DX2[fdir], x3 - D3Q27System::DX3[fdir]));
                                //		real vzBC = ((*vzNode)(x1 - D3Q27System::DX1[fdir], x2 - D3Q27System::DX2[fdir], x3 - D3Q27System::DX3[fdir]));
                                //		real feqL = D3Q27System::getIncompFeqForDirection(fdir, rhoL, vx, vy, vz);
                                //		distribution->setPostCollisionDistributionForDirection(feqL, x1 - D3Q27System::DX1[fdir], x2 - D3Q27System::DX2[fdir], x3 - D3Q27System::DX3[fdir], fdir);
                                //		ff[fdir] = feqL;
                                //	}
                                // }
                                // real sumRho2= 0;
                                // for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                //	sumRho2 += ff[fdir];// -D3Q27System::getIncompFeqForDirection(fdir, 0, sumVx, sumVy, sumVz);
                                // }
                                // ff[d000] = rhoL - sumRho2;
                                // rhoL = 27.0 / 18.0 * sumRho2;
                                // std::cout << "rhoL=" << rhoL <<" sumRho="<< 27.0 / 18.0 * sumRho2 << " vx=" << vx << " vy=" << vy << "\n";
                                D3Q27System::calcIncompMacroscopicValues(ff, rhoL, vx, vy, vz);
                                // std::cout << "RecalCrhoL=" << rhoL << " sumRho=" << 27.0 / 18.0 * sumRho2 << " vx=" << vx << " vy=" << vy << "ffRest="<<ff[d000]<<"\n";
                                // distribution->setPostCollisionDistributionForDirection(ff[d000], x1, x2, x3, d000);
                                {
                                    real fG = distribution->getDistributionInvForDirection(x1, x2, x3, d000);
                                    real feqOLD = D3Q27System::getIncompFeqForDirection(d000, (*rhoNode)(x1, x2, x3), vx, vy, vz);
                                    real feqNew = D3Q27System::getIncompFeqForDirection(d000, rhoL, vx, vy, vz);
                                    distribution->setPostCollisionDistributionForDirection(fG - feqOLD + feqNew, x1, x2, x3, d000);
                                }
                                // for (int fdir = D3Q27System::FSTARTDIR; fdir <= D3Q27System::FENDDIR; fdir++) {
                                //	ff[D3Q27System::INVDIR[fdir]]=distribution->getDistributionInvForDirection(x1 + D3Q27System::DX1[fdir], x2 + D3Q27System::DX2[fdir], x3 + D3Q27System::DX3[fdir], D3Q27System::INVDIR[fdir]);
                                // }
                                // D3Q27System::calcIncompMacroscopicValues(ff, rhoL, vx, vy, vz);
                                // std::cout << "AfterRead rhoL=" << rhoL << " rhoGToL=" << rhoG/densityRatio << " vx=" << vx << " vy=" << vy << "ffRest=" << ff[d000] <<" x="<<x1<<" y="<<x2<<" z="<<x3<< "\n";

                                // real feqL = D3Q27System::getIncompFeqForDirection(d000, rhoL, vx, vy, vz);
                                // distribution->setPostCollisionDistributionForDirection(feqL, x1, x2, x3, d000);
                            }
                        }

                    } // end Loop
                }
            }
        }
    }

    this->swapDistributions();

    real collFactorM;

    for (int x3 = minX3; x3 < maxX3; x3++) {
        for (int x2 = minX2; x2 < maxX2; x2++) {
            for (int x1 = minX1; x1 < maxX1; x1++) {
                if (!bcArray->isSolid(x1, x2, x3) && !bcArray->isUndefined(x1, x2, x3)) {
                    int x1p = x1 + 1;
                    int x2p = x2 + 1;
                    int x3p = x3 + 1;

                    findNeighbors(phaseField, x1, x2, x3);

                    real mfcbb = (*this->localDistributionsF)(D3Q27System::ET_E, x1, x2, x3);
                    real mfbcb = (*this->localDistributionsF)(D3Q27System::ET_N, x1, x2, x3);
                    real mfbbc = (*this->localDistributionsF)(D3Q27System::ET_T, x1, x2, x3);
                    real mfccb = (*this->localDistributionsF)(D3Q27System::ET_NE, x1, x2, x3);
                    real mfacb = (*this->localDistributionsF)(D3Q27System::ET_NW, x1p, x2, x3);
                    real mfcbc = (*this->localDistributionsF)(D3Q27System::ET_TE, x1, x2, x3);
                    real mfabc = (*this->localDistributionsF)(D3Q27System::ET_TW, x1p, x2, x3);
                    real mfbcc = (*this->localDistributionsF)(D3Q27System::ET_TN, x1, x2, x3);
                    real mfbac = (*this->localDistributionsF)(D3Q27System::ET_TS, x1, x2p, x3);
                    real mfccc = (*this->localDistributionsF)(D3Q27System::ET_TNE, x1, x2, x3);
                    real mfacc = (*this->localDistributionsF)(D3Q27System::ET_TNW, x1p, x2, x3);
                    real mfcac = (*this->localDistributionsF)(D3Q27System::ET_TSE, x1, x2p, x3);
                    real mfaac = (*this->localDistributionsF)(D3Q27System::ET_TSW, x1p, x2p, x3);
                    real mfabb = (*this->nonLocalDistributionsF)(D3Q27System::ET_W, x1p, x2, x3);
                    real mfbab = (*this->nonLocalDistributionsF)(D3Q27System::ET_S, x1, x2p, x3);
                    real mfbba = (*this->nonLocalDistributionsF)(D3Q27System::ET_B, x1, x2, x3p);
                    real mfaab = (*this->nonLocalDistributionsF)(D3Q27System::ET_SW, x1p, x2p, x3);
                    real mfcab = (*this->nonLocalDistributionsF)(D3Q27System::ET_SE, x1, x2p, x3);
                    real mfaba = (*this->nonLocalDistributionsF)(D3Q27System::ET_BW, x1p, x2, x3p);
                    real mfcba = (*this->nonLocalDistributionsF)(D3Q27System::ET_BE, x1, x2, x3p);
                    real mfbaa = (*this->nonLocalDistributionsF)(D3Q27System::ET_BS, x1, x2p, x3p);
                    real mfbca = (*this->nonLocalDistributionsF)(D3Q27System::ET_BN, x1, x2, x3p);
                    real mfaaa = (*this->nonLocalDistributionsF)(D3Q27System::ET_BSW, x1p, x2p, x3p);
                    real mfcaa = (*this->nonLocalDistributionsF)(D3Q27System::ET_BSE, x1, x2p, x3p);
                    real mfaca = (*this->nonLocalDistributionsF)(D3Q27System::ET_BNW, x1p, x2, x3p);
                    real mfcca = (*this->nonLocalDistributionsF)(D3Q27System::ET_BNE, x1, x2, x3p);
                    real mfbbb = (*this->restDistributionsF)(x1, x2, x3);

                    real f[D3Q27System::ENDF + 1];
                    real fEq[D3Q27System::ENDF + 1];
                    real fEqSolid[D3Q27System::ENDF + 1];
                    real fPre[D3Q27System::ENDF + 1];

                    f[vf::lbm::dir::d000] = mfbbb;

                    f[vf::lbm::dir::dP00] = mfcbb;
                    f[vf::lbm::dir::d0P0] = mfbcb;
                    f[vf::lbm::dir::d00P] = mfbbc;
                    f[vf::lbm::dir::dPP0] = mfccb;
                    f[vf::lbm::dir::dMP0] = mfacb;
                    f[vf::lbm::dir::dP0P] = mfcbc;
                    f[vf::lbm::dir::dM0P] = mfabc;
                    f[vf::lbm::dir::d0PP] = mfbcc;
                    f[vf::lbm::dir::d0MP] = mfbac;
                    f[vf::lbm::dir::dPPP] = mfccc;
                    f[vf::lbm::dir::dMPP] = mfacc;
                    f[vf::lbm::dir::dPMP] = mfcac;
                    f[vf::lbm::dir::dMMP] = mfaac;

                    f[vf::lbm::dir::dM00] = mfabb;
                    f[vf::lbm::dir::d0M0] = mfbab;
                    f[vf::lbm::dir::d00M] = mfbba;
                    f[vf::lbm::dir::dMM0] = mfaab;
                    f[vf::lbm::dir::dPM0] = mfcab;
                    f[vf::lbm::dir::dM0M] = mfaba;
                    f[vf::lbm::dir::dP0M] = mfcba;
                    f[vf::lbm::dir::d0MM] = mfbaa;
                    f[vf::lbm::dir::d0PM] = mfbca;
                    f[vf::lbm::dir::dMMM] = mfaaa;
                    f[vf::lbm::dir::dPMM] = mfcaa;
                    f[vf::lbm::dir::dMPM] = mfaca;
                    f[vf::lbm::dir::dPPM] = mfcca;

                    if ((*particleData)(x1, x2, x3)->solidFraction > SOLFRAC_MIN) {
                        fPre[vf::lbm::dir::d000] = mfbbb;

                        fPre[vf::lbm::dir::dP00] = mfcbb;
                        fPre[vf::lbm::dir::d0P0] = mfbcb;
                        fPre[vf::lbm::dir::d00P] = mfbbc;
                        fPre[vf::lbm::dir::dPP0] = mfccb;
                        fPre[vf::lbm::dir::dMP0] = mfacb;
                        fPre[vf::lbm::dir::dP0P] = mfcbc;
                        fPre[vf::lbm::dir::dM0P] = mfabc;
                        fPre[vf::lbm::dir::d0PP] = mfbcc;
                        fPre[vf::lbm::dir::d0MP] = mfbac;
                        fPre[vf::lbm::dir::dPPP] = mfccc;
                        fPre[vf::lbm::dir::dMPP] = mfacc;
                        fPre[vf::lbm::dir::dPMP] = mfcac;
                        fPre[vf::lbm::dir::dMMP] = mfaac;

                        fPre[vf::lbm::dir::dM00] = mfabb;
                        fPre[vf::lbm::dir::d0M0] = mfbab;
                        fPre[vf::lbm::dir::d00M] = mfbba;
                        fPre[vf::lbm::dir::dMM0] = mfaab;
                        fPre[vf::lbm::dir::dPM0] = mfcab;
                        fPre[vf::lbm::dir::dM0M] = mfaba;
                        fPre[vf::lbm::dir::dP0M] = mfcba;
                        fPre[vf::lbm::dir::d0MM] = mfbaa;
                        fPre[vf::lbm::dir::d0PM] = mfbca;
                        fPre[vf::lbm::dir::dMMM] = mfaaa;
                        fPre[vf::lbm::dir::dPMM] = mfcaa;
                        fPre[vf::lbm::dir::dMPM] = mfaca;
                        fPre[vf::lbm::dir::dPPM] = mfcca;
                    }

                    (*particleData)(x1, x2, x3)->hydrodynamicForce.fill(0.0);

  

                        real rhoH = 1.0;
                        real rhoL = 1.0 / densityRatio;

                        // real rhoToPhi = (rhoH - rhoL) / (phiH - phiL);

                        real dX1_phi = gradX1_phi();
                        real dX2_phi = gradX2_phi();
                        real dX3_phi = gradX3_phi();

                        real denom = sqrt(dX1_phi * dX1_phi + dX2_phi * dX2_phi + dX3_phi * dX3_phi) + 1.0e-20; //+ 1e-9+1e-3;
                        // 01.09.2022: unclear what value we have to add to the normal: lager values better cut of in gas phase?
                        real normX1 = dX1_phi / denom;
                        real normX2 = dX2_phi / denom;
                        real normX3 = dX3_phi / denom;

                        collFactorM = phi[d000] > c1o2 ? collFactorL : collFactorG;
                        // real collFactorMInv = phi[d000] > c1o2 ? collFactorG : collFactorL;

                        // real mu = 2 * beta * phi[d000] * (phi[d000] - 1) * (2 * phi[d000] - 1) - kappa * nabla2_phi();

                        //----------- Calculating Macroscopic Values -------------
                        real rho = phi[d000] > c1o2 ? rhoH : rhoL;

                        real m0, m1, m2;
                        real rhoRef = c1o1;

                        real vvx = ((((mfccc - mfaaa) + (mfcac - mfaca)) + ((mfcaa - mfacc) + (mfcca - mfaac))) + (((mfcba - mfabc) + (mfcbc - mfaba)) + ((mfcab - mfacb) + (mfccb - mfaab))) + (mfcbb - mfabb)) / rhoRef;
                        real vvy = ((((mfccc - mfaaa) + (mfaca - mfcac)) + ((mfacc - mfcaa) + (mfcca - mfaac))) + (((mfbca - mfbac) + (mfbcc - mfbaa)) + ((mfacb - mfcab) + (mfccb - mfaab))) + (mfbcb - mfbab)) / rhoRef;
                        real vvz = ((((mfccc - mfaaa) + (mfcac - mfaca)) + ((mfacc - mfcaa) + (mfaac - mfcca))) + (((mfbac - mfbca) + (mfbcc - mfbaa)) + ((mfabc - mfcba) + (mfcbc - mfaba))) + (mfbbc - mfbba)) / rhoRef;
                        /////////////////////

                        forcingX1 = 0.0;
                        forcingX2 = 0.0;
                        forcingX3 = 0.0;

                        if (withForcing) {
                            muRho = rho;
                            muPhi = phi[d000];
                            forcingX1 += muForcingX1.Eval();
                            forcingX2 += muForcingX2.Eval();
                            forcingX3 += muForcingX3.Eval();

                            vvx += (forcingX1)*deltaT * c1o2;
                            vvy += (forcingX2)*deltaT * c1o2;
                            vvz += (forcingX3)*deltaT * c1o2;
                        }
                        if ((*particleData)(x1, x2, x3)->solidFraction <= SOLFRAC_MAX) {
                        real vx2;
                        real vy2;
                        real vz2;
                        vx2 = vvx * vvx;
                        vy2 = vvy * vvy;
                        vz2 = vvz * vvz;
                        ///////////////////////////////////////////////////////////////////////////////////////////
                        real oMdrho;
                        ///////////////

                        oMdrho = mfccc + mfaaa;
                        m0 = mfaca + mfcac;
                        m1 = mfacc + mfcaa;
                        m2 = mfaac + mfcca;
                        oMdrho += m0;
                        m1 += m2;
                        oMdrho += m1;
                        m0 = mfbac + mfbca;
                        m1 = mfbaa + mfbcc;
                        m0 += m1;
                        m1 = mfabc + mfcba;
                        m2 = mfaba + mfcbc;
                        m1 += m2;
                        m0 += m1;
                        m1 = mfacb + mfcab;
                        m2 = mfaab + mfccb;
                        m1 += m2;
                        m0 += m1;
                        oMdrho += m0;
                        m0 = mfabb + mfcbb;
                        m1 = mfbab + mfbcb;
                        m2 = mfbba + mfbbc;
                        m0 += m1 + m2;
                        m0 += mfbbb;                                // hat gefehlt
                        oMdrho = (rhoRef - (oMdrho + m0)) / rhoRef; // 12.03.21 check derivation!!!!

                        ////////////////////////////////////////////////////////////////////////////////////
                        real wadjust;
                        //					real qudricLimit = 0.01 / (c1o1 + 1.0e4 * phi[d000] * (c1o1 - phi[d000]));
                        // real qudricLimit = 0.01 / (c1o1 + (((*phaseField)(x1, x2, x3) > c1o2) ? 1.0e6 * phi[d000] * (c1o1 - phi[d000]):c0o1));
                        real qudricLimit = 0.01;

                        ////////////////////////////////////////////////////////////////////////////////////
                        //! - Chimera transform from well conditioned distributions to central moments as defined in Appendix J in \ref
                        //! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
                        //! see also Eq. (6)-(14) in \ref
                        //! <a href="https://doi.org/10.1016/j.jcp.2017.05.040"><b>[ M. Geier et al. (2017), DOI:10.1016/j.jcp.2017.05.040 ]</b></a>
                        //!
                        ////////////////////////////////////////////////////////////////////////////////////
                        // Z - Dir
                        forwardInverseChimeraWithKincompressible(mfaaa, mfaab, mfaac, vvz, vz2, c36o1, c1o36, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfaba, mfabb, mfabc, vvz, vz2, c9o1, c1o9, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfaca, mfacb, mfacc, vvz, vz2, c36o1, c1o36, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfbaa, mfbab, mfbac, vvz, vz2, c9o1, c1o9, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfbba, mfbbb, mfbbc, vvz, vz2, c9o4, c4o9, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfbca, mfbcb, mfbcc, vvz, vz2, c9o1, c1o9, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfcaa, mfcab, mfcac, vvz, vz2, c36o1, c1o36, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfcba, mfcbb, mfcbc, vvz, vz2, c9o1, c1o9, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfcca, mfccb, mfccc, vvz, vz2, c36o1, c1o36, oMdrho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        // Y - Dir
                        forwardInverseChimeraWithKincompressible(mfaaa, mfaba, mfaca, vvy, vy2, c6o1, c1o6, oMdrho);
                        forwardChimera(mfaab, mfabb, mfacb, vvy, vy2);
                        forwardInverseChimeraWithKincompressible(mfaac, mfabc, mfacc, vvy, vy2, c18o1, c1o18, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfbaa, mfbba, mfbca, vvy, vy2, c3o2, c2o3, oMdrho);
                        forwardChimera(mfbab, mfbbb, mfbcb, vvy, vy2);
                        forwardInverseChimeraWithKincompressible(mfbac, mfbbc, mfbcc, vvy, vy2, c9o2, c2o9, oMdrho);
                        forwardInverseChimeraWithKincompressible(mfcaa, mfcba, mfcca, vvy, vy2, c6o1, c1o6, oMdrho);
                        forwardChimera(mfcab, mfcbb, mfccb, vvy, vy2);
                        forwardInverseChimeraWithKincompressible(mfcac, mfcbc, mfccc, vvy, vy2, c18o1, c1o18, oMdrho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        // X - Dir
                        forwardInverseChimeraWithKincompressible(mfaaa, mfbaa, mfcaa, vvx, vx2, c1o1, c1o1, oMdrho);
                        forwardChimera(mfaba, mfbba, mfcba, vvx, vx2);
                        forwardInverseChimeraWithKincompressible(mfaca, mfbca, mfcca, vvx, vx2, c3o1, c1o3, oMdrho);
                        forwardChimera(mfaab, mfbab, mfcab, vvx, vx2);
                        forwardChimera(mfabb, mfbbb, mfcbb, vvx, vx2);
                        forwardChimera(mfacb, mfbcb, mfccb, vvx, vx2);
                        forwardInverseChimeraWithKincompressible(mfaac, mfbac, mfcac, vvx, vx2, c3o1, c1o3, oMdrho);
                        forwardChimera(mfabc, mfbbc, mfcbc, vvx, vx2);
                        forwardInverseChimeraWithKincompressible(mfacc, mfbcc, mfccc, vvx, vx2, c3o1, c1o9, oMdrho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        ////////////////////////////////////////////////////////////////////////////////////
                        // Cumulants
                        ////////////////////////////////////////////////////////////////////////////////////

                        // mfaaa = 0.0;
                        real OxxPyyPzz = 1.0; // omega2 or bulk viscosity
                                              //   real OxyyPxzz = 1.;//-s9;//2+s9;//
                                              //   real OxyyMxzz  = 1.;//2+s9;//
                        real O4 = 1.;
                        real O5 = 1.;
                        real O6 = 1.;
                        /////

                        /////fourth order parameters; here only for test. Move out of loop!

                        real OxyyPxzz = 8.0 * (collFactorM - 2.0) * (OxxPyyPzz * (3.0 * collFactorM - 1.0) - 5.0 * collFactorM) / (8.0 * (5.0 - 2.0 * collFactorM) * collFactorM + OxxPyyPzz * (8.0 + collFactorM * (9.0 * collFactorM - 26.0)));
                        real OxyyMxzz = 8.0 * (collFactorM - 2.0) * (collFactorM + OxxPyyPzz * (3.0 * collFactorM - 7.0)) / (OxxPyyPzz * (56.0 - 42.0 * collFactorM + 9.0 * collFactorM * collFactorM) - 8.0 * collFactorM);
                        real Oxyz = 24.0 * (collFactorM - 2.0) * (4.0 * collFactorM * collFactorM + collFactorM * OxxPyyPzz * (18.0 - 13.0 * collFactorM) + OxxPyyPzz * OxxPyyPzz * (2.0 + collFactorM * (6.0 * collFactorM - 11.0))) /
                                    (16.0 * collFactorM * collFactorM * (collFactorM - 6.0) - 2.0 * collFactorM * OxxPyyPzz * (216.0 + 5.0 * collFactorM * (9.0 * collFactorM - 46.0)) + OxxPyyPzz * OxxPyyPzz * (collFactorM * (3.0 * collFactorM - 10.0) * (15.0 * collFactorM - 28.0) - 48.0));
                        real A = (4.0 * collFactorM * collFactorM + 2.0 * collFactorM * OxxPyyPzz * (collFactorM - 6.0) + OxxPyyPzz * OxxPyyPzz * (collFactorM * (10.0 - 3.0 * collFactorM) - 4.0)) / ((collFactorM - OxxPyyPzz) * (OxxPyyPzz * (2.0 + 3.0 * collFactorM) - 8.0 * collFactorM));
                        // FIXME:  warning C4459: declaration of 'B' hides global declaration (message : see declaration of 'D3Q27System::B' )
                        real BB = (4.0 * collFactorM * OxxPyyPzz * (9.0 * collFactorM - 16.0) - 4.0 * collFactorM * collFactorM - 2.0 * OxxPyyPzz * OxxPyyPzz * (2.0 + 9.0 * collFactorM * (collFactorM - 2.0))) /
                                  (3.0 * (collFactorM - OxxPyyPzz) * (OxxPyyPzz * (2.0 + 3.0 * collFactorM) - 8.0 * collFactorM));

                        // Cum 4.
                        real CUMcbb = mfcbb - ((mfcaa + c1o3) * mfabb + 2. * mfbba * mfbab);
                        real CUMbcb = mfbcb - ((mfaca + c1o3) * mfbab + 2. * mfbba * mfabb);
                        real CUMbbc = mfbbc - ((mfaac + c1o3) * mfbba + 2. * mfbab * mfabb);

                        real CUMcca = mfcca - ((mfcaa * mfaca + 2. * mfbba * mfbba) + c1o3 * (mfcaa + mfaca) * oMdrho + c1o9 * (oMdrho - c1o1) * oMdrho);
                        real CUMcac = mfcac - ((mfcaa * mfaac + 2. * mfbab * mfbab) + c1o3 * (mfcaa + mfaac) * oMdrho + c1o9 * (oMdrho - c1o1) * oMdrho);
                        real CUMacc = mfacc - ((mfaac * mfaca + 2. * mfabb * mfabb) + c1o3 * (mfaac + mfaca) * oMdrho + c1o9 * (oMdrho - c1o1) * oMdrho);

                        // Cum 5.
                        real CUMbcc = mfbcc - (mfaac * mfbca + mfaca * mfbac + 4. * mfabb * mfbbb + 2. * (mfbab * mfacb + mfbba * mfabc)) - c1o3 * (mfbca + mfbac) * oMdrho;
                        real CUMcbc = mfcbc - (mfaac * mfcba + mfcaa * mfabc + 4. * mfbab * mfbbb + 2. * (mfabb * mfcab + mfbba * mfbac)) - c1o3 * (mfcba + mfabc) * oMdrho;
                        real CUMccb = mfccb - (mfcaa * mfacb + mfaca * mfcab + 4. * mfbba * mfbbb + 2. * (mfbab * mfbca + mfabb * mfcba)) - c1o3 * (mfacb + mfcab) * oMdrho;

                        // Cum 6.
                        real CUMccc = mfccc +
                                      ((-4. * mfbbb * mfbbb - (mfcaa * mfacc + mfaca * mfcac + mfaac * mfcca) - 4. * (mfabb * mfcbb + mfbab * mfbcb + mfbba * mfbbc) - 2. * (mfbca * mfbac + mfcba * mfabc + mfcab * mfacb)) +
                                       (4. * (mfbab * mfbab * mfaca + mfabb * mfabb * mfcaa + mfbba * mfbba * mfaac) + 2. * (mfcaa * mfaca * mfaac) + 16. * mfbba * mfbab * mfabb) - c1o3 * (mfacc + mfcac + mfcca) * oMdrho - c1o9 * oMdrho * oMdrho -
                                       c1o9 * (mfcaa + mfaca + mfaac) * oMdrho * (1. - 2. * oMdrho) - c1o27 * oMdrho * oMdrho * (-2. * oMdrho) + (2. * (mfbab * mfbab + mfabb * mfabb + mfbba * mfbba) + (mfaac * mfaca + mfaac * mfcaa + mfaca * mfcaa)) * c2o3 * oMdrho) +
                                      c1o27 * oMdrho;

                        // 2.
                        //  linear combinations
                        real mxxPyyPzz = mfcaa + mfaca + mfaac;
                        mxxPyyPzz -= mfaaa; // 12.03.21 shifted by mfaaa
                                            // mxxPyyPzz-=(mfaaa+mfaaaS)*c1o2;//12.03.21 shifted by mfaaa
                        real mxxMyy = mfcaa - mfaca;
                        real mxxMzz = mfcaa - mfaac;

                        ///
                        real mmfcaa = c1o3 * (mxxMyy + mxxMzz + mxxPyyPzz);
                        real mmfaca = c1o3 * (-2. * mxxMyy + mxxMzz + mxxPyyPzz);
                        real mmfaac = c1o3 * (mxxMyy - 2. * mxxMzz + mxxPyyPzz);
                        real mmfabb = mfabb;
                        real mmfbab = mfbab;
                        real mmfbba = mfbba;
                        ///

                        real dxux = -c1o2 * collFactorM * (mxxMyy + mxxMzz) + c1o2 * OxxPyyPzz * (/*mfaaa*/ -mxxPyyPzz) * 0;
                        // real dxux = -c1o2 * (mxxMyy + mxxMzz) * collFactorM - mfaaa * c1o3* omegaDRho;
                        real dyuy = dxux + collFactorM * c3o2 * mxxMyy;
                        real dzuz = dxux + collFactorM * c3o2 * mxxMzz;
                        real Dxy = -c3o1 * collFactorM * mfbba;
                        real Dxz = -c3o1 * collFactorM * mfbab;
                        real Dyz = -c3o1 * collFactorM * mfabb;

                        if (phi[d000] > c1o2) {
                            /// QR eddyviscosity:
                            real eddyR = -(Dxy * Dxy + Dxz * Dxz + c1o3 * dxux * dxux) * (dxux) - (Dxy * Dxy + Dyz * Dyz + c1o3 * dyuy * dyuy) * dyuy - (Dxz * Dxz + Dyz * Dyz + c1o3 * dzuz * dzuz) * dzuz - c2o1 * Dxy * Dxz * Dyz;
                            real eddyQ = Dxy * Dxz + Dxy * Dyz + Dxz * Dyz + c1o2 * (dxux * dxux + dyuy * dyuy + dzuz * dzuz);
                            real nuEddy = 5.0e1 * (eddyR / (eddyQ + 1e-100)) * (dX1_phi * dX1_phi + dX2_phi * dX2_phi + dX3_phi * dX3_phi);
                            nuEddy = (nuEddy < c1o1 / collFactorM) ? c1o1 / collFactorM : nuEddy;
                            collFactorM = c1o1 / nuEddy;
                            // collFactorM = c1o1 / (c1o1 / collFactorM +1.e2*nuEddy*(dX1_phi*dX1_phi+dX2_phi*dX2_phi+dX3_phi*dX3_phi));
                            collFactorM = (collFactorM < 1.8) ? 1.8 : collFactorM;
                            OxyyPxzz = 8.0 * (collFactorM - 2.0) * (OxxPyyPzz * (3.0 * collFactorM - 1.0) - 5.0 * collFactorM) / (8.0 * (5.0 - 2.0 * collFactorM) * collFactorM + OxxPyyPzz * (8.0 + collFactorM * (9.0 * collFactorM - 26.0)));
                            OxyyMxzz = 8.0 * (collFactorM - 2.0) * (collFactorM + OxxPyyPzz * (3.0 * collFactorM - 7.0)) / (OxxPyyPzz * (56.0 - 42.0 * collFactorM + 9.0 * collFactorM * collFactorM) - 8.0 * collFactorM);
                            Oxyz = 24.0 * (collFactorM - 2.0) * (4.0 * collFactorM * collFactorM + collFactorM * OxxPyyPzz * (18.0 - 13.0 * collFactorM) + OxxPyyPzz * OxxPyyPzz * (2.0 + collFactorM * (6.0 * collFactorM - 11.0))) /
                                   (16.0 * collFactorM * collFactorM * (collFactorM - 6.0) - 2.0 * collFactorM * OxxPyyPzz * (216.0 + 5.0 * collFactorM * (9.0 * collFactorM - 46.0)) + OxxPyyPzz * OxxPyyPzz * (collFactorM * (3.0 * collFactorM - 10.0) * (15.0 * collFactorM - 28.0) - 48.0));
                            A = (4.0 * collFactorM * collFactorM + 2.0 * collFactorM * OxxPyyPzz * (collFactorM - 6.0) + OxxPyyPzz * OxxPyyPzz * (collFactorM * (10.0 - 3.0 * collFactorM) - 4.0)) / ((collFactorM - OxxPyyPzz) * (OxxPyyPzz * (2.0 + 3.0 * collFactorM) - 8.0 * collFactorM));
                            BB = (4.0 * collFactorM * OxxPyyPzz * (9.0 * collFactorM - 16.0) - 4.0 * collFactorM * collFactorM - 2.0 * OxxPyyPzz * OxxPyyPzz * (2.0 + 9.0 * collFactorM * (collFactorM - 2.0))) /
                                 (3.0 * (collFactorM - OxxPyyPzz) * (OxxPyyPzz * (2.0 + 3.0 * collFactorM) - 8.0 * collFactorM));
                        }

                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        // non Newtonian fluid collision factor
                        if (phi[d000] > c1o2) {
                            real shearRate = sqrt(c2o1 * (dxux * dxux + dyuy * dyuy + dzuz * dzuz) + Dxy * Dxy + Dxz * Dxz + Dyz * Dyz);
                            collFactorM = Rheology::getBinghamCollFactor(collFactorM, shearRate, c1o1);
                            collFactorM = (collFactorM < c1o1) ? c1o1 : collFactorM;
                        }
                        // omega = Rheology::getHerschelBulkleyCollFactor(omega, shearRate, drho);
                        // omega = Rheology::getBinghamCollFactor(omega, shearRate, drho);
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                        /////////
                        // real mxxMyyh = -c2o1 * (dxux - dyuy) / collFactorMInv * c1o3;
                        // real mxxMzzh = -c2o1 * (dxux - dzuz) / collFactorMInv * c1o3;

                        // relax
                        mxxPyyPzz += OxxPyyPzz * (/*mfaaa*/ -mxxPyyPzz) - 3. * (1. - c1o2 * OxxPyyPzz) * (vx2 * dxux + vy2 * dyuy + vz2 * dzuz);
                        mxxMyy += collFactorM * (-mxxMyy) - 3. * (1. - c1o2 * collFactorM) * (vx2 * dxux - vy2 * dyuy);
                        mxxMzz += collFactorM * (-mxxMzz) - 3. * (1. - c1o2 * collFactorM) * (vx2 * dxux - vz2 * dzuz);

                        mfabb += collFactorM * (-mfabb);
                        mfbab += collFactorM * (-mfbab);
                        mfbba += collFactorM * (-mfbba);

                        // mxxMyyh += collFactorMInv * (-mxxMyyh) - 3. * (1. - c1o2 * collFactorMInv) * (vx2 * dxux - vy2 * dyuy);
                        // mxxMzzh += collFactorMInv * (-mxxMzzh) - 3. * (1. - c1o2 * collFactorMInv) * (vx2 * dxux - vz2 * dzuz);

                        mxxPyyPzz += mfaaa; // 12.03.21 shifted by mfaaa

                        // mxxPyyPzz += (mfaaa + mfaaaS) * c1o2;
                        // mfaaa = mfaaaS;
                        // linear combinations back
                        mfcaa = c1o3 * (mxxMyy + mxxMzz + mxxPyyPzz);
                        mfaca = c1o3 * (-2. * mxxMyy + mxxMzz + mxxPyyPzz);
                        mfaac = c1o3 * (mxxMyy - 2. * mxxMzz + mxxPyyPzz);

                        // 3.
                        //  linear combinations
                        real mxxyPyzz = mfcba + mfabc;
                        real mxxyMyzz = mfcba - mfabc;

                        real mxxzPyyz = mfcab + mfacb;
                        real mxxzMyyz = mfcab - mfacb;

                        real mxyyPxzz = mfbca + mfbac;
                        real mxyyMxzz = mfbca - mfbac;

                        mmfcaa += c1o3 * (mxxMyy + mxxMzz + mxxPyyPzz - mfaaa);
                        mmfaca += c1o3 * (-2. * mxxMyy + mxxMzz + mxxPyyPzz - mfaaa);
                        mmfaac += c1o3 * (mxxMyy - 2. * mxxMzz + mxxPyyPzz - mfaaa);
                        mmfabb += mfabb;
                        mmfbab += mfbab;
                        mmfbba += mfbba;

                        // relax
                        wadjust = Oxyz + (1. - Oxyz) * fabs(mfbbb) / (fabs(mfbbb) + qudricLimit);
                        mfbbb += wadjust * (-mfbbb);
                        wadjust = OxyyPxzz + (1. - OxyyPxzz) * fabs(mxxyPyzz) / (fabs(mxxyPyzz) + qudricLimit);
                        mxxyPyzz += wadjust * (-mxxyPyzz);
                        wadjust = OxyyMxzz + (1. - OxyyMxzz) * fabs(mxxyMyzz) / (fabs(mxxyMyzz) + qudricLimit);
                        mxxyMyzz += wadjust * (-mxxyMyzz);
                        wadjust = OxyyPxzz + (1. - OxyyPxzz) * fabs(mxxzPyyz) / (fabs(mxxzPyyz) + qudricLimit);
                        mxxzPyyz += wadjust * (-mxxzPyyz);
                        wadjust = OxyyMxzz + (1. - OxyyMxzz) * fabs(mxxzMyyz) / (fabs(mxxzMyyz) + qudricLimit);
                        mxxzMyyz += wadjust * (-mxxzMyyz);
                        wadjust = OxyyPxzz + (1. - OxyyPxzz) * fabs(mxyyPxzz) / (fabs(mxyyPxzz) + qudricLimit);
                        mxyyPxzz += wadjust * (-mxyyPxzz);
                        wadjust = OxyyMxzz + (1. - OxyyMxzz) * fabs(mxyyMxzz) / (fabs(mxyyMxzz) + qudricLimit);
                        mxyyMxzz += wadjust * (-mxyyMxzz);

                        // linear combinations back
                        mfcba = (mxxyMyzz + mxxyPyzz) * c1o2;
                        mfabc = (-mxxyMyzz + mxxyPyzz) * c1o2;
                        mfcab = (mxxzMyyz + mxxzPyyz) * c1o2;
                        mfacb = (-mxxzMyyz + mxxzPyyz) * c1o2;
                        mfbca = (mxyyMxzz + mxyyPxzz) * c1o2;
                        mfbac = (-mxyyMxzz + mxyyPxzz) * c1o2;

                        // 4.
                        CUMacc = -O4 * (c1o1 / collFactorM - c1o2) * (dyuy + dzuz) * c2o3 * A + (c1o1 - O4) * (CUMacc);
                        CUMcac = -O4 * (c1o1 / collFactorM - c1o2) * (dxux + dzuz) * c2o3 * A + (c1o1 - O4) * (CUMcac);
                        CUMcca = -O4 * (c1o1 / collFactorM - c1o2) * (dyuy + dxux) * c2o3 * A + (c1o1 - O4) * (CUMcca);
                        CUMbbc = -O4 * (c1o1 / collFactorM - c1o2) * Dxy * c1o3 * BB + (c1o1 - O4) * (CUMbbc);
                        CUMbcb = -O4 * (c1o1 / collFactorM - c1o2) * Dxz * c1o3 * BB + (c1o1 - O4) * (CUMbcb);
                        CUMcbb = -O4 * (c1o1 / collFactorM - c1o2) * Dyz * c1o3 * BB + (c1o1 - O4) * (CUMcbb);

                        // 5.
                        CUMbcc += O5 * (-CUMbcc);
                        CUMcbc += O5 * (-CUMcbc);
                        CUMccb += O5 * (-CUMccb);

                        // 6.
                        CUMccc += O6 * (-CUMccc);

                        // back cumulants to central moments
                        // 4.
                        mfcbb = CUMcbb + ((mfcaa + c1o3) * mfabb + 2. * mfbba * mfbab);
                        mfbcb = CUMbcb + ((mfaca + c1o3) * mfbab + 2. * mfbba * mfabb);
                        mfbbc = CUMbbc + ((mfaac + c1o3) * mfbba + 2. * mfbab * mfabb);

                        mfcca = CUMcca + (mfcaa * mfaca + 2. * mfbba * mfbba) + c1o3 * (mfcaa + mfaca) * oMdrho + c1o9 * (oMdrho - c1o1) * oMdrho;
                        mfcac = CUMcac + (mfcaa * mfaac + 2. * mfbab * mfbab) + c1o3 * (mfcaa + mfaac) * oMdrho + c1o9 * (oMdrho - c1o1) * oMdrho;
                        mfacc = CUMacc + (mfaac * mfaca + 2. * mfabb * mfabb) + c1o3 * (mfaac + mfaca) * oMdrho + c1o9 * (oMdrho - c1o1) * oMdrho;

                        // 5.
                        mfbcc = CUMbcc + (mfaac * mfbca + mfaca * mfbac + 4. * mfabb * mfbbb + 2. * (mfbab * mfacb + mfbba * mfabc)) + c1o3 * (mfbca + mfbac) * oMdrho;
                        mfcbc = CUMcbc + (mfaac * mfcba + mfcaa * mfabc + 4. * mfbab * mfbbb + 2. * (mfabb * mfcab + mfbba * mfbac)) + c1o3 * (mfcba + mfabc) * oMdrho;
                        mfccb = CUMccb + (mfcaa * mfacb + mfaca * mfcab + 4. * mfbba * mfbbb + 2. * (mfbab * mfbca + mfabb * mfcba)) + c1o3 * (mfacb + mfcab) * oMdrho;

                        // 6.
                        mfccc = CUMccc -
                                ((-4. * mfbbb * mfbbb - (mfcaa * mfacc + mfaca * mfcac + mfaac * mfcca) - 4. * (mfabb * mfcbb + mfbac * mfbca + mfbba * mfbbc) - 2. * (mfbca * mfbac + mfcba * mfabc + mfcab * mfacb)) +
                                 (4. * (mfbab * mfbab * mfaca + mfabb * mfabb * mfcaa + mfbba * mfbba * mfaac) + 2. * (mfcaa * mfaca * mfaac) + 16. * mfbba * mfbab * mfabb) - c1o3 * (mfacc + mfcac + mfcca) * oMdrho - c1o9 * oMdrho * oMdrho -
                                 c1o9 * (mfcaa + mfaca + mfaac) * oMdrho * (1. - 2. * oMdrho) - c1o27 * oMdrho * oMdrho * (-2. * oMdrho) + (2. * (mfbab * mfbab + mfabb * mfabb + mfbba * mfbba) + (mfaac * mfaca + mfaac * mfcaa + mfaca * mfcaa)) * c2o3 * oMdrho) -
                                c1o27 * oMdrho;

                        ////////

                        ////////////////////////////////////////////////////////////////////////////////////
                        // forcing
                        mfbaa = -mfbaa;
                        mfaba = -mfaba;
                        mfaab = -mfaab;

                        backwardInverseChimeraWithKincompressible(mfaaa, mfbaa, mfcaa, vvx, vx2, c1o1, c1o1, oMdrho);
                        backwardChimera(mfaba, mfbba, mfcba, vvx, vx2);
                        backwardInverseChimeraWithKincompressible(mfaca, mfbca, mfcca, vvx, vx2, c3o1, c1o3, oMdrho);
                        backwardChimera(mfaab, mfbab, mfcab, vvx, vx2);
                        backwardChimera(mfabb, mfbbb, mfcbb, vvx, vx2);
                        backwardChimera(mfacb, mfbcb, mfccb, vvx, vx2);
                        backwardInverseChimeraWithKincompressible(mfaac, mfbac, mfcac, vvx, vx2, c3o1, c1o3, oMdrho);
                        backwardChimera(mfabc, mfbbc, mfcbc, vvx, vx2);
                        backwardInverseChimeraWithKincompressible(mfacc, mfbcc, mfccc, vvx, vx2, c9o1, c1o9, oMdrho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        // Y - Dir
                        backwardInverseChimeraWithKincompressible(mfaaa, mfaba, mfaca, vvy, vy2, c6o1, c1o6, oMdrho);
                        backwardChimera(mfaab, mfabb, mfacb, vvy, vy2);
                        backwardInverseChimeraWithKincompressible(mfaac, mfabc, mfacc, vvy, vy2, c18o1, c1o18, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfbaa, mfbba, mfbca, vvy, vy2, c3o2, c2o3, oMdrho);
                        backwardChimera(mfbab, mfbbb, mfbcb, vvy, vy2);
                        backwardInverseChimeraWithKincompressible(mfbac, mfbbc, mfbcc, vvy, vy2, c9o2, c2o9, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfcaa, mfcba, mfcca, vvy, vy2, c6o1, c1o6, oMdrho);
                        backwardChimera(mfcab, mfcbb, mfccb, vvy, vy2);
                        backwardInverseChimeraWithKincompressible(mfcac, mfcbc, mfccc, vvy, vy2, c18o1, c1o18, oMdrho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        // Z - Dir
                        backwardInverseChimeraWithKincompressible(mfaaa, mfaab, mfaac, vvz, vz2, c36o1, c1o36, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfaba, mfabb, mfabc, vvz, vz2, c9o1, c1o9, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfaca, mfacb, mfacc, vvz, vz2, c36o1, c1o36, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfbaa, mfbab, mfbac, vvz, vz2, c9o1, c1o9, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfbba, mfbbb, mfbbc, vvz, vz2, c9o4, c4o9, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfbca, mfbcb, mfbcc, vvz, vz2, c9o1, c1o9, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfcaa, mfcab, mfcac, vvz, vz2, c36o1, c1o36, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfcba, mfcbb, mfcbc, vvz, vz2, c9o1, c1o9, oMdrho);
                        backwardInverseChimeraWithKincompressible(mfcca, mfccb, mfccc, vvz, vz2, c36o1, c1o36, oMdrho);
                        //////////////////////////////////////////////////////////////////////////
                        // proof correctness
                        //////////////////////////////////////////////////////////////////////////
                        // #ifdef  PROOF_CORRECTNESS
                        real rho_post = (mfaaa + mfaac + mfaca + mfcaa + mfacc + mfcac + mfccc + mfcca) + (mfaab + mfacb + mfcab + mfccb) + (mfaba + mfabc + mfcba + mfcbc) + (mfbaa + mfbac + mfbca + mfbcc) + (mfabb + mfcbb) + (mfbab + mfbcb) + (mfbba + mfbbc) + mfbbb;

                        if (UbMath::isNaN(rho_post) || UbMath::isInfinity(rho_post)) UB_THROW(UbException(UB_EXARGS, "rho_post is not a number (nan or -1.#IND) or infinity number -1.#INF, node=" + UbSystem::toString(x1) + "," + UbSystem::toString(x2) + "," + UbSystem::toString(x3)));

                        //////////////////////////////////////////////////////////////////////////
                        // write distribution
                        //////////////////////////////////////////////////////////////////////////
                        //	if (phi[d000] < c1o2) {
                        (*this->localDistributionsF)(D3Q27System::ET_E, x1, x2, x3) = mfabb;         //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_N, x1, x2, x3) = mfbab;         //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_T, x1, x2, x3) = mfbba;         //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_NE, x1, x2, x3) = mfaab;        //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_NW, x1p, x2, x3) = mfcab;       //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_TE, x1, x2, x3) = mfaba;        //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_TW, x1p, x2, x3) = mfcba;       //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_TN, x1, x2, x3) = mfbaa;        //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_TS, x1, x2p, x3) = mfbca;       //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_TNE, x1, x2, x3) = mfaaa;       //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_TNW, x1p, x2, x3) = mfcaa;      //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_TSE, x1, x2p, x3) = mfaca;      //* rho * c1o3;
                        (*this->localDistributionsF)(D3Q27System::ET_TSW, x1p, x2p, x3) = mfcca;     //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_W, x1p, x2, x3) = mfcbb;     //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_S, x1, x2p, x3) = mfbcb;     //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_B, x1, x2, x3p) = mfbbc;     //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_SW, x1p, x2p, x3) = mfccb;   //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_SE, x1, x2p, x3) = mfacb;    //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BW, x1p, x2, x3p) = mfcbc;   //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BE, x1, x2, x3p) = mfabc;    //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BS, x1, x2p, x3p) = mfbcc;   //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BN, x1, x2, x3p) = mfbac;    //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BSW, x1p, x2p, x3p) = mfccc; //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BSE, x1, x2p, x3p) = mfacc;  //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BNW, x1p, x2, x3p) = mfcac;  //* rho * c1o3;
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BNE, x1, x2, x3p) = mfaac;   //* rho * c1o3;

                        (*this->restDistributionsF)(x1, x2, x3) = mfbbb; // *rho* c1o3;

                        f[vf::lbm::dir::d000] = mfbbb;

                        f[vf::lbm::dir::dP00] = mfcbb;
                        f[vf::lbm::dir::d0P0] = mfbcb;
                        f[vf::lbm::dir::d00P] = mfbbc;
                        f[vf::lbm::dir::dPP0] = mfccb;
                        f[vf::lbm::dir::dMP0] = mfacb;
                        f[vf::lbm::dir::dP0P] = mfcbc;
                        f[vf::lbm::dir::dM0P] = mfabc;
                        f[vf::lbm::dir::d0PP] = mfbcc;
                        f[vf::lbm::dir::d0MP] = mfbac;
                        f[vf::lbm::dir::dPPP] = mfccc;
                        f[vf::lbm::dir::dMPP] = mfacc;
                        f[vf::lbm::dir::dPMP] = mfcac;
                        f[vf::lbm::dir::dMMP] = mfaac;

                        f[vf::lbm::dir::dM00] = mfabb;
                        f[vf::lbm::dir::d0M0] = mfbab;
                        f[vf::lbm::dir::d00M] = mfbba;
                        f[vf::lbm::dir::dMM0] = mfaab;
                        f[vf::lbm::dir::dPM0] = mfcab;
                        f[vf::lbm::dir::dM0M] = mfaba;
                        f[vf::lbm::dir::dP0M] = mfcba;
                        f[vf::lbm::dir::d0MM] = mfbaa;
                        f[vf::lbm::dir::d0PM] = mfbca;
                        f[vf::lbm::dir::dMMM] = mfaaa;
                        f[vf::lbm::dir::dPMM] = mfcaa;
                        f[vf::lbm::dir::dMPM] = mfaca;
                        f[vf::lbm::dir::dPPM] = mfcca;
                    }
                    if ((*particleData)(x1, x2, x3)->solidFraction >= SOLFRAC_MIN) {
                        real vx1, vx2, vx3, drho;
                        D3Q27System::calcIncompMacroscopicValues(f, drho, vx1, vx2, vx3);
                        D3Q27System::calcIncompFeq(fEq, drho, vx1, vx2, vx3);

                        std::array<double, 3> uPart;
                        uPart[0] = (*particleData)(x1, x2, x3)->uPart[0];
                        uPart[1] = (*particleData)(x1, x2, x3)->uPart[1];
                        uPart[2] = (*particleData)(x1, x2, x3)->uPart[2];

                        D3Q27System::calcIncompFeq(fEqSolid, drho, uPart[0], uPart[1], uPart[2]);
                        real rhoPhaseField = (phi[d000] > c1o2) ? c1o1 : c1o1 / densityRatio;
                        if ((*particleData)(x1, x2, x3)->solidFraction > SOLFRAC_MAX) {
                            double const bb0 = fEq[vf::lbm::dir::d000] - fEqSolid[vf::lbm::dir::d000];
                            f[vf::lbm::dir::d000] = fPre[vf::lbm::dir::d000] + bb0;
                            for (int iPop = D3Q27System::FSTARTDIR; iPop <= D3Q27System::FENDDIR; iPop++) {
                                const int iOpp = D3Q27System::INVDIR[iPop];
                                double const bb = ((fPre[iOpp] - fEq[iOpp]) - (fPre[iPop] - fEqSolid[iPop]));
                                double const bbOpp = ((fPre[iPop] - fEq[iPop]) - (fPre[iOpp] - fEqSolid[iOpp]));

                                f[iPop] = fPre[iPop] + bb;
                                f[iOpp] = fPre[iOpp] + bbOpp;

                                (*particleData)(x1, x2, x3)->hydrodynamicForce[0] -= D3Q27System::DX1[iPop] * (bb - bbOpp) * rhoPhaseField;
                                (*particleData)(x1, x2, x3)->hydrodynamicForce[1] -= D3Q27System::DX2[iPop] * (bb - bbOpp) * rhoPhaseField;
                                (*particleData)(x1, x2, x3)->hydrodynamicForce[2] -= D3Q27System::DX3[iPop] * (bb - bbOpp) * rhoPhaseField;
                            }
                        } else { /* particleData.solidFraction < SOLFRAC_MAX */
                                 // #ifdef LBDEM_USE_WEIGHING
                            double const ooo = 1. / collFactorM - 0.5;
                            double const B = (*particleData)(x1, x2, x3)->solidFraction * ooo / ((1. - (*particleData)(x1, x2, x3)->solidFraction) + ooo);
                            // #else
                            //                         T const B = particleData.solidFraction;
                            // #endif
                            double const oneMinB = 1. - B;

                            double const bb0 = fEq[vf::lbm::dir::d000] - fEqSolid[vf::lbm::dir::d000];
                            f[vf::lbm::dir::d000] = fPre[vf::lbm::dir::d000] + oneMinB * (f[vf::lbm::dir::d000] - fPre[vf::lbm::dir::d000]) + B * bb0;

                            for (int iPop = D3Q27System::FSTARTDIR; iPop <= D3Q27System::FENDDIR; iPop++) {
                                int const iOpp = D3Q27System::INVDIR[iPop];
                                double const bb = B * ((fPre[iOpp] - fEq[iOpp]) - (fPre[iPop] - fEqSolid[iPop]));
                                double const bbOpp = B * ((fPre[iPop] - fEq[iPop]) - (fPre[iOpp] - fEqSolid[iOpp]));

                                f[iPop] = fPre[iPop] + oneMinB * (f[iPop] - fPre[iPop]) + bb;
                                f[iOpp] = fPre[iOpp] + oneMinB * (f[iOpp] - fPre[iOpp]) + bbOpp;

                                (*particleData)(x1, x2, x3)->hydrodynamicForce[0] -= D3Q27System::DX1[iPop] * (bb - bbOpp) * rhoPhaseField;
                                (*particleData)(x1, x2, x3)->hydrodynamicForce[1] -= D3Q27System::DX2[iPop] * (bb - bbOpp) * rhoPhaseField;
                                (*particleData)(x1, x2, x3)->hydrodynamicForce[2] -= D3Q27System::DX3[iPop] * (bb - bbOpp) * rhoPhaseField;
                            }
                        } /* if solidFraction > SOLFRAC_MAX */

                        (*this->restDistributionsF)(x1, x2, x3) = f[vf::lbm::dir::d000];

                        (*this->localDistributionsF)(D3Q27System::ET_E, x1, x2, x3) = f[vf::lbm::dir::dM00];
                        (*this->localDistributionsF)(D3Q27System::ET_N, x1, x2, x3) = f[vf::lbm::dir::d0M0];
                        (*this->localDistributionsF)(D3Q27System::ET_T, x1, x2, x3) = f[vf::lbm::dir::d00M];
                        (*this->localDistributionsF)(D3Q27System::ET_NE, x1, x2, x3) = f[vf::lbm::dir::dMM0];
                        (*this->localDistributionsF)(D3Q27System::ET_NW, x1p, x2, x3) = f[vf::lbm::dir::dPM0];
                        (*this->localDistributionsF)(D3Q27System::ET_TE, x1, x2, x3) = f[vf::lbm::dir::dM0M];
                        (*this->localDistributionsF)(D3Q27System::ET_TW, x1p, x2, x3) = f[vf::lbm::dir::dP0M];
                        (*this->localDistributionsF)(D3Q27System::ET_TN, x1, x2, x3) = f[vf::lbm::dir::d0MM];
                        (*this->localDistributionsF)(D3Q27System::ET_TS, x1, x2p, x3) = f[vf::lbm::dir::d0PM];
                        (*this->localDistributionsF)(D3Q27System::ET_TNE, x1, x2, x3) = f[vf::lbm::dir::dMMM];
                        (*this->localDistributionsF)(D3Q27System::ET_TNW, x1p, x2, x3) = f[vf::lbm::dir::dPMM];
                        (*this->localDistributionsF)(D3Q27System::ET_TSE, x1, x2p, x3) = f[vf::lbm::dir::dMPM];
                        (*this->localDistributionsF)(D3Q27System::ET_TSW, x1p, x2p, x3) = f[vf::lbm::dir::dPPM];

                        (*this->nonLocalDistributionsF)(D3Q27System::ET_W, x1p, x2, x3) = f[vf::lbm::dir::dP00];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_S, x1, x2p, x3) = f[vf::lbm::dir::d0P0];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_B, x1, x2, x3p) = f[vf::lbm::dir::d00P];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_SW, x1p, x2p, x3) = f[vf::lbm::dir::dPP0];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_SE, x1, x2p, x3) = f[vf::lbm::dir::dMP0];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BW, x1p, x2, x3p) = f[vf::lbm::dir::dP0P];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BE, x1, x2, x3p) = f[vf::lbm::dir::dM0P];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BS, x1, x2p, x3p) = f[vf::lbm::dir::d0PP];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BN, x1, x2, x3p) = f[vf::lbm::dir::d0MP];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BSW, x1p, x2p, x3p) = f[vf::lbm::dir::dPPP];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BSE, x1, x2p, x3p) = f[vf::lbm::dir::dMPP];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BNW, x1p, x2, x3p) = f[vf::lbm::dir::dPMP];
                        (*this->nonLocalDistributionsF)(D3Q27System::ET_BNE, x1, x2, x3p) = f[vf::lbm::dir::dMMP];                   
                    }



                        /////////////////////  P H A S E - F I E L D   S O L V E R
                     ////////////////////////////////////////////
                                                                     /////CUMULANT PHASE-FIELD
                    real omegaD = 1.0 / (3.0 * mob + 0.5);
                    {
                        mfcbb = (*this->localDistributionsH1)(D3Q27System::ET_E, x1, x2, x3);
                        mfbcb = (*this->localDistributionsH1)(D3Q27System::ET_N, x1, x2, x3);
                        mfbbc = (*this->localDistributionsH1)(D3Q27System::ET_T, x1, x2, x3);
                        mfccb = (*this->localDistributionsH1)(D3Q27System::ET_NE, x1, x2, x3);
                        mfacb = (*this->localDistributionsH1)(D3Q27System::ET_NW, x1p, x2, x3);
                        mfcbc = (*this->localDistributionsH1)(D3Q27System::ET_TE, x1, x2, x3);
                        mfabc = (*this->localDistributionsH1)(D3Q27System::ET_TW, x1p, x2, x3);
                        mfbcc = (*this->localDistributionsH1)(D3Q27System::ET_TN, x1, x2, x3);
                        mfbac = (*this->localDistributionsH1)(D3Q27System::ET_TS, x1, x2p, x3);
                        mfccc = (*this->localDistributionsH1)(D3Q27System::ET_TNE, x1, x2, x3);
                        mfacc = (*this->localDistributionsH1)(D3Q27System::ET_TNW, x1p, x2, x3);
                        mfcac = (*this->localDistributionsH1)(D3Q27System::ET_TSE, x1, x2p, x3);
                        mfaac = (*this->localDistributionsH1)(D3Q27System::ET_TSW, x1p, x2p, x3);
                        mfabb = (*this->nonLocalDistributionsH1)(D3Q27System::ET_W, x1p, x2, x3);
                        mfbab = (*this->nonLocalDistributionsH1)(D3Q27System::ET_S, x1, x2p, x3);
                        mfbba = (*this->nonLocalDistributionsH1)(D3Q27System::ET_B, x1, x2, x3p);
                        mfaab = (*this->nonLocalDistributionsH1)(D3Q27System::ET_SW, x1p, x2p, x3);
                        mfcab = (*this->nonLocalDistributionsH1)(D3Q27System::ET_SE, x1, x2p, x3);
                        mfaba = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BW, x1p, x2, x3p);
                        mfcba = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BE, x1, x2, x3p);
                        mfbaa = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BS, x1, x2p, x3p);
                        mfbca = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BN, x1, x2, x3p);
                        mfaaa = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BSW, x1p, x2p, x3p);
                        mfcaa = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BSE, x1, x2p, x3p);
                        mfaca = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BNW, x1p, x2, x3p);
                        mfcca = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BNE, x1, x2, x3p);
                        mfbbb = (*this->restDistributionsH1)(x1, x2, x3);

                        ////////////////////////////////////////////////////////////////////////////////////
                        //! - Calculate density and velocity using pyramid summation for low round-off errors as in Eq. (J1)-(J3) \ref
                        //! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
                        //!
                        ////////////////////////////////////////////////////////////////////////////////////
                        // second component
                        real concentration =
                            ((((mfccc + mfaaa) + (mfaca + mfcac)) + ((mfacc + mfcaa) + (mfaac + mfcca))) + (((mfbac + mfbca) + (mfbaa + mfbcc)) + ((mfabc + mfcba) + (mfaba + mfcbc)) + ((mfacb + mfcab) + (mfaab + mfccb))) + ((mfabb + mfcbb) + (mfbab + mfbcb) + (mfbba + mfbbc))) + mfbbb;
                        ////////////////////////////////////////////////////////////////////////////////////
                        real oneMinusRho = c1o1 - concentration;

                        real cx = ((((mfccc - mfaaa) + (mfcac - mfaca)) + ((mfcaa - mfacc) + (mfcca - mfaac))) + (((mfcba - mfabc) + (mfcbc - mfaba)) + ((mfcab - mfacb) + (mfccb - mfaab))) + (mfcbb - mfabb));
                        real cy = ((((mfccc - mfaaa) + (mfaca - mfcac)) + ((mfacc - mfcaa) + (mfcca - mfaac))) + (((mfbca - mfbac) + (mfbcc - mfbaa)) + ((mfacb - mfcab) + (mfccb - mfaab))) + (mfbcb - mfbab));
                        real cz = ((((mfccc - mfaaa) + (mfcac - mfaca)) + ((mfacc - mfcaa) + (mfaac - mfcca))) + (((mfbac - mfbca) + (mfbcc - mfbaa)) + ((mfabc - mfcba) + (mfcbc - mfaba))) + (mfbbc - mfbba));

                        ////////////////////////////////////////////////////////////////////////////////////
                        // calculate the square of velocities for this lattice node
                        real cx2 = cx * cx;
                        real cy2 = cy * cy;
                        real cz2 = cz * cz;
                        ////////////////////////////////////////////////////////////////////////////////////
                        //! - Chimera transform from well conditioned distributions to central moments as defined in Appendix J in \ref
                        //! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
                        //! see also Eq. (6)-(14) in \ref
                        //! <a href="https://doi.org/10.1016/j.jcp.2017.05.040"><b>[ M. Geier et al. (2017), DOI:10.1016/j.jcp.2017.05.040 ]</b></a>
                        //!
                        ////////////////////////////////////////////////////////////////////////////////////
                        // Z - Dir
                        forwardInverseChimeraWithKincompressible(mfaaa, mfaab, mfaac, cz, cz2, c36o1, c1o36, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfaba, mfabb, mfabc, cz, cz2, c9o1, c1o9, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfaca, mfacb, mfacc, cz, cz2, c36o1, c1o36, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfbaa, mfbab, mfbac, cz, cz2, c9o1, c1o9, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfbba, mfbbb, mfbbc, cz, cz2, c9o4, c4o9, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfbca, mfbcb, mfbcc, cz, cz2, c9o1, c1o9, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfcaa, mfcab, mfcac, cz, cz2, c36o1, c1o36, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfcba, mfcbb, mfcbc, cz, cz2, c9o1, c1o9, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfcca, mfccb, mfccc, cz, cz2, c36o1, c1o36, oneMinusRho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        // Y - Dir
                        forwardInverseChimeraWithKincompressible(mfaaa, mfaba, mfaca, cy, cy2, c6o1, c1o6, oneMinusRho);
                        forwardChimera(mfaab, mfabb, mfacb, cy, cy2);
                        forwardInverseChimeraWithKincompressible(mfaac, mfabc, mfacc, cy, cy2, c18o1, c1o18, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfbaa, mfbba, mfbca, cy, cy2, c3o2, c2o3, oneMinusRho);
                        forwardChimera(mfbab, mfbbb, mfbcb, cy, cy2);
                        forwardInverseChimeraWithKincompressible(mfbac, mfbbc, mfbcc, cy, cy2, c9o2, c2o9, oneMinusRho);
                        forwardInverseChimeraWithKincompressible(mfcaa, mfcba, mfcca, cy, cy2, c6o1, c1o6, oneMinusRho);
                        forwardChimera(mfcab, mfcbb, mfccb, cy, cy2);
                        forwardInverseChimeraWithKincompressible(mfcac, mfcbc, mfccc, cy, cy2, c18o1, c1o18, oneMinusRho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        // X - Dir
                        forwardInverseChimeraWithKincompressible(mfaaa, mfbaa, mfcaa, cx, cx2, c1o1, c1o1, oneMinusRho);
                        forwardChimera(mfaba, mfbba, mfcba, cx, cx2);
                        forwardInverseChimeraWithKincompressible(mfaca, mfbca, mfcca, cx, cx2, c3o1, c1o3, oneMinusRho);
                        forwardChimera(mfaab, mfbab, mfcab, cx, cx2);
                        forwardChimera(mfabb, mfbbb, mfcbb, cx, cx2);
                        forwardChimera(mfacb, mfbcb, mfccb, cx, cx2);
                        forwardInverseChimeraWithKincompressible(mfaac, mfbac, mfcac, cx, cx2, c3o1, c1o3, oneMinusRho);
                        forwardChimera(mfabc, mfbbc, mfcbc, cx, cx2);
                        forwardInverseChimeraWithKincompressible(mfacc, mfbcc, mfccc, cx, cx2, c3o1, c1o9, oneMinusRho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        //! - experimental Cumulant ... to be published ... hopefully
                        //!

                        // linearized orthogonalization of 3rd order central moments
                        real Mabc = mfabc - mfaba * c1o3;
                        real Mbca = mfbca - mfbaa * c1o3;
                        real Macb = mfacb - mfaab * c1o3;
                        real Mcba = mfcba - mfaba * c1o3;
                        real Mcab = mfcab - mfaab * c1o3;
                        real Mbac = mfbac - mfbaa * c1o3;
                        // linearized orthogonalization of 5th order central moments
                        real Mcbc = mfcbc - mfaba * c1o9;
                        real Mbcc = mfbcc - mfbaa * c1o9;
                        real Mccb = mfccb - mfaab * c1o9;

                        // 31.05.2022 addaptive mobility
                        // omegaD = c1o1 + (sqrt((cx - vvx * concentration) * (cx - vvx * concentration) + (cy - vvy * concentration) * (cy - vvy * concentration) + (cz - vvz * concentration) * (cz - vvz * concentration))) / (sqrt((cx - vvx * concentration) * (cx - vvx * concentration) + (cy - vvy *
                        // concentration) * (cy - vvy * concentration) + (cz - vvz * concentration) * (cz - vvz * concentration)) + fabs((1.0 - concentration) * (concentration)) * c1o6 * oneOverInterfaceScale+1.0e-200); omegaD = c2o1 * (concentration * (concentration - c1o1)) / (-c6o1 * (sqrt((cx -
                        // vvx * concentration) * (cx - vvx * concentration) + (cy - vvy * concentration) * (cy - vvy * concentration) + (cz - vvz * concentration) * (cz - vvz * concentration))) + (concentration * (concentration - c1o1))+1.0e-200);
                        //  collision of 1st order moments
                        cx = cx * (c1o1 - omegaD) + omegaD * vvx * concentration + normX1 * (c1o1 - 0.5 * omegaD) * (1.0 - concentration) * (concentration)*c1o3 * oneOverInterfaceScale;
                        cy = cy * (c1o1 - omegaD) + omegaD * vvy * concentration + normX2 * (c1o1 - 0.5 * omegaD) * (1.0 - concentration) * (concentration)*c1o3 * oneOverInterfaceScale;
                        cz = cz * (c1o1 - omegaD) + omegaD * vvz * concentration + normX3 * (c1o1 - 0.5 * omegaD) * (1.0 - concentration) * (concentration)*c1o3 * oneOverInterfaceScale;

                        cx2 = cx * cx;
                        cy2 = cy * cy;
                        cz2 = cz * cz;

                        // equilibration of 2nd order moments
                        mfbba = c0o1;
                        mfbab = c0o1;
                        mfabb = c0o1;

                        mfcaa = c1o3 * concentration;
                        mfaca = c1o3 * concentration;
                        mfaac = c1o3 * concentration;

                        // equilibration of 3rd order moments
                        Mabc = c0o1;
                        Mbca = c0o1;
                        Macb = c0o1;
                        Mcba = c0o1;
                        Mcab = c0o1;
                        Mbac = c0o1;
                        mfbbb = c0o1;

                        // from linearized orthogonalization 3rd order central moments to central moments
                        mfabc = Mabc + mfaba * c1o3;
                        mfbca = Mbca + mfbaa * c1o3;
                        mfacb = Macb + mfaab * c1o3;
                        mfcba = Mcba + mfaba * c1o3;
                        mfcab = Mcab + mfaab * c1o3;
                        mfbac = Mbac + mfbaa * c1o3;

                        // equilibration of 4th order moments
                        mfacc = c1o9 * concentration;
                        mfcac = c1o9 * concentration;
                        mfcca = c1o9 * concentration;

                        mfcbb = c0o1;
                        mfbcb = c0o1;
                        mfbbc = c0o1;

                        // equilibration of 5th order moments
                        Mcbc = c0o1;
                        Mbcc = c0o1;
                        Mccb = c0o1;

                        // from linearized orthogonalization 5th order central moments to central moments
                        mfcbc = Mcbc + mfaba * c1o9;
                        mfbcc = Mbcc + mfbaa * c1o9;
                        mfccb = Mccb + mfaab * c1o9;

                        // equilibration of 6th order moment
                        mfccc = c1o27 * concentration;

                        ////////////////////////////////////////////////////////////////////////////////////
                        //! - Chimera transform from central moments to well conditioned distributions as defined in Appendix J in
                        //! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
                        //! see also Eq. (88)-(96) in
                        //! <a href="https://doi.org/10.1016/j.jcp.2017.05.040"><b>[ M. Geier et al. (2017), DOI:10.1016/j.jcp.2017.05.040 ]</b></a>
                        //!
                        ////////////////////////////////////////////////////////////////////////////////////
                        // X - Dir
                        backwardInverseChimeraWithKincompressible(mfaaa, mfbaa, mfcaa, cx, cx2, c1o1, c1o1, oneMinusRho);
                        backwardChimera(mfaba, mfbba, mfcba, cx, cx2);
                        backwardInverseChimeraWithKincompressible(mfaca, mfbca, mfcca, cx, cx2, c3o1, c1o3, oneMinusRho);
                        backwardChimera(mfaab, mfbab, mfcab, cx, cx2);
                        backwardChimera(mfabb, mfbbb, mfcbb, cx, cx2);
                        backwardChimera(mfacb, mfbcb, mfccb, cx, cx2);
                        backwardInverseChimeraWithKincompressible(mfaac, mfbac, mfcac, cx, cx2, c3o1, c1o3, oneMinusRho);
                        backwardChimera(mfabc, mfbbc, mfcbc, cx, cx2);
                        backwardInverseChimeraWithKincompressible(mfacc, mfbcc, mfccc, cx, cx2, c9o1, c1o9, oneMinusRho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        // Y - Dir
                        backwardInverseChimeraWithKincompressible(mfaaa, mfaba, mfaca, cy, cy2, c6o1, c1o6, oneMinusRho);
                        backwardChimera(mfaab, mfabb, mfacb, cy, cy2);
                        backwardInverseChimeraWithKincompressible(mfaac, mfabc, mfacc, cy, cy2, c18o1, c1o18, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfbaa, mfbba, mfbca, cy, cy2, c3o2, c2o3, oneMinusRho);
                        backwardChimera(mfbab, mfbbb, mfbcb, cy, cy2);
                        backwardInverseChimeraWithKincompressible(mfbac, mfbbc, mfbcc, cy, cy2, c9o2, c2o9, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfcaa, mfcba, mfcca, cy, cy2, c6o1, c1o6, oneMinusRho);
                        backwardChimera(mfcab, mfcbb, mfccb, cy, cy2);
                        backwardInverseChimeraWithKincompressible(mfcac, mfcbc, mfccc, cy, cy2, c18o1, c1o18, oneMinusRho);

                        ////////////////////////////////////////////////////////////////////////////////////
                        // Z - Dir
                        backwardInverseChimeraWithKincompressible(mfaaa, mfaab, mfaac, cz, cz2, c36o1, c1o36, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfaba, mfabb, mfabc, cz, cz2, c9o1, c1o9, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfaca, mfacb, mfacc, cz, cz2, c36o1, c1o36, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfbaa, mfbab, mfbac, cz, cz2, c9o1, c1o9, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfbba, mfbbb, mfbbc, cz, cz2, c9o4, c4o9, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfbca, mfbcb, mfbcc, cz, cz2, c9o1, c1o9, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfcaa, mfcab, mfcac, cz, cz2, c36o1, c1o36, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfcba, mfcbb, mfcbc, cz, cz2, c9o1, c1o9, oneMinusRho);
                        backwardInverseChimeraWithKincompressible(mfcca, mfccb, mfccc, cz, cz2, c36o1, c1o36, oneMinusRho);

                        (*this->localDistributionsH1)(D3Q27System::ET_E, x1, x2, x3) = mfabb;
                        (*this->localDistributionsH1)(D3Q27System::ET_N, x1, x2, x3) = mfbab;
                        (*this->localDistributionsH1)(D3Q27System::ET_T, x1, x2, x3) = mfbba;
                        (*this->localDistributionsH1)(D3Q27System::ET_NE, x1, x2, x3) = mfaab;
                        (*this->localDistributionsH1)(D3Q27System::ET_NW, x1p, x2, x3) = mfcab;
                        (*this->localDistributionsH1)(D3Q27System::ET_TE, x1, x2, x3) = mfaba;
                        (*this->localDistributionsH1)(D3Q27System::ET_TW, x1p, x2, x3) = mfcba;
                        (*this->localDistributionsH1)(D3Q27System::ET_TN, x1, x2, x3) = mfbaa;
                        (*this->localDistributionsH1)(D3Q27System::ET_TS, x1, x2p, x3) = mfbca;
                        (*this->localDistributionsH1)(D3Q27System::ET_TNE, x1, x2, x3) = mfaaa;
                        (*this->localDistributionsH1)(D3Q27System::ET_TNW, x1p, x2, x3) = mfcaa;
                        (*this->localDistributionsH1)(D3Q27System::ET_TSE, x1, x2p, x3) = mfaca;
                        (*this->localDistributionsH1)(D3Q27System::ET_TSW, x1p, x2p, x3) = mfcca;

                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_W, x1p, x2, x3) = mfcbb;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_S, x1, x2p, x3) = mfbcb;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_B, x1, x2, x3p) = mfbbc;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_SW, x1p, x2p, x3) = mfccb;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_SE, x1, x2p, x3) = mfacb;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_BW, x1p, x2, x3p) = mfcbc;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_BE, x1, x2, x3p) = mfabc;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_BS, x1, x2p, x3p) = mfbcc;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_BN, x1, x2, x3p) = mfbac;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_BSW, x1p, x2p, x3p) = mfccc;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_BSE, x1, x2p, x3p) = mfacc;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_BNW, x1p, x2, x3p) = mfcac;
                        (*this->nonLocalDistributionsH1)(D3Q27System::ET_BNE, x1, x2, x3p) = mfaac;

                        (*this->restDistributionsH1)(x1, x2, x3) = mfbbb;

                    }
                }
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////

real IBsharpInterfaceLBMKernel::gradX1_phi()
{
    using namespace D3Q27System;
    return 3.0 * ((WEIGTH[dPPP] * (((phi[dPPP] - phi[dMMM]) + (phi[dPMM] - phi[dMPP])) + ((phi[dPMP] - phi[dMPM]) + (phi[dPPM] - phi[dMMP]))) +
                   WEIGTH[dPP0] * (((phi[dP0P] - phi[dM0M]) + (phi[dP0M] - phi[dM0P])) + ((phi[dPM0] - phi[dMP0]) + (phi[dPP0] - phi[dMM0])))) +
                  +WEIGTH[d0P0] * (phi[dP00] - phi[dM00]));
}

real IBsharpInterfaceLBMKernel::gradX2_phi()
{
    using namespace D3Q27System;
    return 3.0 * ((WEIGTH[dPPP] * (((phi[dPPP] - phi[dMMM]) - (phi[dPMM] - phi[dMPP])) + ((phi[dPPM] - phi[dMMP]) - (phi[dPMP] - phi[dMPM]))) +
                   WEIGTH[dPP0] * (((phi[d0PP] - phi[d0MM]) + (phi[d0PM] - phi[d0MP])) + ((phi[dPP0] - phi[dMM0]) - (phi[dPM0] - phi[dMP0])))) +
                  +WEIGTH[d0P0] * (phi[d0P0] - phi[d0M0]));
}

real IBsharpInterfaceLBMKernel::gradX3_phi()
{
    using namespace D3Q27System;
    return 3.0 * ((WEIGTH[dPPP] * (((phi[dPPP] - phi[dMMM]) - (phi[dPMM] - phi[dMPP])) + ((phi[dPMP] - phi[dMPM]) - (phi[dPPM] - phi[dMMP]))) +
                   WEIGTH[dPP0] * (((phi[dP0P] - phi[dM0M]) - (phi[dP0M] - phi[dM0P])) + ((phi[d0MP] - phi[d0PM]) + (phi[d0PP] - phi[d0MM])))) +
                  +WEIGTH[d0P0] * (phi[d00P] - phi[d00M]));
}

real IBsharpInterfaceLBMKernel::gradX1_rhoInv(real rhoL, real rhoDIV)
{
    using namespace D3Q27System;
    return 3.0 * ((WEIGTH[dPPP] * (((1.0 / (rhoL + rhoDIV * phi[dPPP]) - 1.0 / (rhoL + rhoDIV * phi[dMMM])) + (1.0 / (rhoL + rhoDIV * phi[dPMM]) - 1.0 / (rhoL + rhoDIV * phi[dMPP]))) +
                                      ((1.0 / (rhoL + rhoDIV * phi[dPMP]) - 1.0 / (rhoL + rhoDIV * phi[dMPM])) + (1.0 / (rhoL + rhoDIV * phi[dPPM]) - 1.0 / (rhoL + rhoDIV * phi[dMMP])))) +
                   WEIGTH[dPP0] * (((1.0 / (rhoL + rhoDIV * phi[dP0P]) - 1.0 / (rhoL + rhoDIV * phi[dM0M])) + (1.0 / (rhoL + rhoDIV * phi[dP0M]) - 1.0 / (rhoL + rhoDIV * phi[dM0P]))) +
                                      ((1.0 / (rhoL + rhoDIV * phi[dPM0]) - 1.0 / (rhoL + rhoDIV * phi[dMP0])) + (1.0 / (rhoL + rhoDIV * phi[dPP0]) - 1.0 / (rhoL + rhoDIV * phi[dMM0]))))) +
                  +WEIGTH[d0P0] * (1.0 / (rhoL + rhoDIV * phi[dP00]) - 1.0 / (rhoL + rhoDIV * phi[dM00])));
}

real IBsharpInterfaceLBMKernel::gradX2_rhoInv(real rhoL, real rhoDIV)
{
    using namespace D3Q27System;
    return 3.0 * ((WEIGTH[dPPP] * (((1.0 / (rhoL + rhoDIV * phi[dPPP]) - 1.0 / (rhoL + rhoDIV * phi[dMMM])) - (1.0 / (rhoL + rhoDIV * phi[dPMM]) - 1.0 / (rhoL + rhoDIV * phi[dMPP]))) +
                                      ((1.0 / (rhoL + rhoDIV * phi[dPPM]) - 1.0 / (rhoL + rhoDIV * phi[dMMP])) - (1.0 / (rhoL + rhoDIV * phi[dPMP]) - 1.0 / (rhoL + rhoDIV * phi[dMPM])))) +
                   WEIGTH[dPP0] * (((1.0 / (rhoL + rhoDIV * phi[d0PP]) - 1.0 / (rhoL + rhoDIV * phi[d0MM])) + (1.0 / (rhoL + rhoDIV * phi[d0PM]) - 1.0 / (rhoL + rhoDIV * phi[d0MP]))) +
                                      ((1.0 / (rhoL + rhoDIV * phi[dPP0]) - 1.0 / (rhoL + rhoDIV * phi[dMM0])) - (1.0 / (rhoL + rhoDIV * phi[dPM0]) - 1.0 / (rhoL + rhoDIV * phi[dMP0]))))) +
                  +WEIGTH[d0P0] * (1.0 / (rhoL + rhoDIV * phi[d0P0]) - 1.0 / (rhoL + rhoDIV * phi[d0M0])));
}

real IBsharpInterfaceLBMKernel::gradX3_rhoInv(real rhoL, real rhoDIV)
{
    using namespace D3Q27System;
    return 3.0 * ((WEIGTH[dPPP] * (((1.0 / (rhoL + rhoDIV * phi[dPPP]) - 1.0 / (rhoL + rhoDIV * phi[dMMM])) - (1.0 / (rhoL + rhoDIV * phi[dPMM]) - 1.0 / (rhoL + rhoDIV * phi[dMPP]))) +
                                      ((1.0 / (rhoL + rhoDIV * phi[dPMP]) - 1.0 / (rhoL + rhoDIV * phi[dMPM])) - (1.0 / (rhoL + rhoDIV * phi[dPPM]) - 1.0 / (rhoL + rhoDIV * phi[dMMP])))) +
                   WEIGTH[dPP0] * (((1.0 / (rhoL + rhoDIV * phi[dP0P]) - 1.0 / (rhoL + rhoDIV * phi[dM0M])) - (1.0 / (rhoL + rhoDIV * phi[dP0M]) - 1.0 / (rhoL + rhoDIV * phi[dM0P]))) +
                                      ((1.0 / (rhoL + rhoDIV * phi[d0MP]) - 1.0 / (rhoL + rhoDIV * phi[d0PM])) + (1.0 / (rhoL + rhoDIV * phi[d0PP]) - 1.0 / (rhoL + rhoDIV * phi[d0MM]))))) +
                  +WEIGTH[d0P0] * (1.0 / (rhoL + rhoDIV * phi[d00P]) - 1.0 / (rhoL + rhoDIV * phi[d00M])));
}

real IBsharpInterfaceLBMKernel::gradX1_phi2()
{
    using namespace D3Q27System;
    return 3.0 * ((WEIGTH[dPPP] * (((phi2[dPPP] - phi2[dMMM]) + (phi2[dPMM] - phi2[dMPP])) + ((phi2[dPMP] - phi2[dMPM]) + (phi2[dPPM] - phi2[dMMP]))) +
                   WEIGTH[dPP0] * (((phi2[dP0P] - phi2[dM0M]) + (phi2[dP0M] - phi2[dM0P])) + ((phi2[dPM0] - phi2[dMP0]) + (phi2[dPP0] - phi2[dMM0])))) +
                  +WEIGTH[d0P0] * (phi2[dP00] - phi2[dM00]));
}

real IBsharpInterfaceLBMKernel::gradX2_phi2()
{
    using namespace D3Q27System;
    return 3.0 * ((WEIGTH[dPPP] * (((phi2[dPPP] - phi2[dMMM]) - (phi2[dPMM] - phi2[dMPP])) + ((phi2[dPPM] - phi2[dMMP]) - (phi2[dPMP] - phi2[dMPM]))) +
                   WEIGTH[dPP0] * (((phi2[d0PP] - phi2[d0MM]) + (phi2[d0PM] - phi2[d0MP])) + ((phi2[dPP0] - phi2[dMM0]) - (phi2[dPM0] - phi2[dMP0])))) +
                  +WEIGTH[d0P0] * (phi2[d0P0] - phi2[d0M0]));
}

real IBsharpInterfaceLBMKernel::gradX3_phi2()
{
    using namespace D3Q27System;
    return 3.0 * ((WEIGTH[dPPP] * (((phi2[dPPP] - phi2[dMMM]) - (phi2[dPMM] - phi2[dMPP])) + ((phi2[dPMP] - phi2[dMPM]) - (phi2[dPPM] - phi2[dMMP]))) +
                   WEIGTH[dPP0] * (((phi2[dP0P] - phi2[dM0M]) - (phi2[dP0M] - phi2[dM0P])) + ((phi2[d0MP] - phi2[d0PM]) + (phi2[d0PP] - phi2[d0MM])))) +
                  +WEIGTH[d0P0] * (phi2[d00P] - phi2[d00M]));
}

real IBsharpInterfaceLBMKernel::nabla2_phi()
{
    using namespace D3Q27System;
    real sum = 0.0;
    sum += WEIGTH[dPPP] * ((((phi[dPPP] - phi[d000]) + (phi[dMMM] - phi[d000])) + ((phi[dMMP] - phi[d000]) + (phi[dPPM] - phi[d000]))) + (((phi[dMPP] - phi[d000]) + (phi[dPMM] - phi[d000])) + ((phi[dPMP] - phi[d000]) + (phi[dMPM] - phi[d000]))));
    sum += WEIGTH[d0PP] * ((((phi[d0PP] - phi[d000]) + (phi[d0MM] - phi[d000])) + ((phi[d0MP] - phi[d000]) + (phi[d0PM] - phi[d000]))) + (((phi[dP0P] - phi[d000]) + (phi[dM0M] - phi[d000])) + ((phi[dM0P] - phi[d000]) + (phi[dP0M] - phi[d000]))) +
                              (((phi[dPP0] - phi[d000]) + (phi[dMM0] - phi[d000])) + ((phi[dMP0] - phi[d000]) + (phi[dPM0] - phi[d000]))));
    sum += WEIGTH[d00P] * (((phi[d00P] - phi[d000]) + (phi[d00M] - phi[d000])) + ((phi[d0P0] - phi[d000]) + (phi[d0M0] - phi[d000])) + ((phi[dP00] - phi[d000]) + (phi[dM00] - phi[d000])));

    return 6.0 * sum;
}

real IBsharpInterfaceLBMKernel::computeCurvature_phi()
{
    using namespace D3Q27System;
    using namespace UbMath;

    real phiX = gradX1_phi();
    real phiY = gradX2_phi();
    real phiZ = gradX3_phi();
    real phiXX =
        c4o9 * (phi[dP00] - c2o1 * phi[d000] + phi[dM00]) + (c1o9 * (((phi[dPP0] - c2o1 * phi[d0P0] + phi[dMP0]) + (phi[dPM0] - c2o1 * phi[d0M0] + phi[dMM0])) + ((phi[dP0P] - c2o1 * phi[d00P] + phi[dM0P]) + (phi[dP0M] - c2o1 * phi[d00M] + phi[dM0M]))) +
                                                                      c1o36 * (((phi[dPPP] - c2o1 * phi[d0PP] + phi[dMPP]) + (phi[dPMP] - c2o1 * phi[d0MP] + phi[dMMP])) + ((phi[dPPM] - c2o1 * phi[d0PM] + phi[dMPM]) + (phi[dPMM] - c2o1 * phi[d0MM] + phi[dMMM]))));
    real phiYY =
        c4o9 * (phi[d0P0] - c2o1 * phi[d000] + phi[d0M0]) + (c1o9 * (((phi[dPP0] - c2o1 * phi[dP00] + phi[dPM0]) + (phi[dMP0] - c2o1 * phi[dM00] + phi[dMM0])) + ((phi[d0PP] - c2o1 * phi[d00P] + phi[d0MP]) + (phi[d0PM] - c2o1 * phi[d00M] + phi[d0MM]))) +
                                                                      c1o36 * (((phi[dPPP] - c2o1 * phi[dP0P] + phi[dPMP]) + (phi[dMPM] - c2o1 * phi[dM0M] + phi[dMMM])) + ((phi[dMPP] - c2o1 * phi[dM0P] + phi[dMMP]) + (phi[dPPM] - c2o1 * phi[dP0M] + phi[dPMM]))));
    real phiZZ =
        c4o9 * (phi[d00P] - c2o1 * phi[d000] + phi[d00M]) + (c1o9 * (((phi[dM0P] - c2o1 * phi[dM00] + phi[dM0M]) + (phi[dP0P] - c2o1 * phi[dP00] + phi[dP0M])) + ((phi[d0MP] - c2o1 * phi[d0M0] + phi[d0MM]) + (phi[d0PP] - c2o1 * phi[d0P0] + phi[d0PM]))) +
                                                                      c1o36 * (((phi[dMPP] - c2o1 * phi[dMP0] + phi[dMPM]) + (phi[dPMP] - c2o1 * phi[dPM0] + phi[dPMM])) + ((phi[dMMP] - c2o1 * phi[dMM0] + phi[dMMM]) + (phi[dPPP] - c2o1 * phi[dPP0] + phi[dPPM]))));
    real phiXY = c1o4 * (c2o3 * (phi[dMM0] - phi[dPM0] + phi[dPP0] - phi[dMP0]) + c1o6 * ((phi[dMMP] - phi[dPMP] + phi[dPPP] - phi[dMPP]) + (phi[dMMM] - phi[dPMM] + phi[dPPM] - phi[dMPM])));
    real phiXZ = c1o4 * (c2o3 * (phi[dM0M] - phi[dP0M] + phi[dP0P] - phi[dM0P]) + c1o6 * ((phi[dMPM] - phi[dPPM] + phi[dPPP] - phi[dMPP]) + (phi[dMMM] - phi[dPMM] + phi[dPMP] - phi[dMMP])));
    real phiYZ = c1o4 * (c2o3 * (phi[d0MM] - phi[d0MP] + phi[d0PP] - phi[d0PM]) + c1o6 * ((phi[dMMM] - phi[dMMP] + phi[dMPP] - phi[dMPM]) + (phi[dPMM] - phi[dPMP] + phi[dPPP] - phi[dPPM])));

    // non isotropic FD (to be improved):
    // real phiX = (phi[dP00] - phi[dM00]) * c1o2; //gradX1_phi();
    // real phiY = (phi[d0P0] - phi[d0M0]) * c1o2; //gradX2_phi();
    // real phiZ = (phi[d00P] - phi[d00M]) * c1o2; //gradX3_phi();

    // real phiXX = phi[dP00] - c2o1 * phi[d000] + phi[dM00];
    // real phiYY = phi[d0P0] - c2o1 * phi[d000] + phi[d0M0];
    // real phiZZ =( phi[d00P] - c2o1 * phi[d000] + phi[d00M]);
    // real phiXY = c1o4 * (phi[dMM0] - phi[dPM0] + phi[dPP0] - phi[dMP0]);
    // real phiXZ = c1o4 * (phi[dM0M] - phi[dP0M] + phi[dP0P] - phi[dM0P]);
    // real phiYZ = c1o4 * (phi[d0MM] - phi[d0MP] + phi[d0PP] - phi[d0PM]);
    // real back= (c2o1 * (phiX * phiY * phiXY + phiX * phiZ * phiXZ + phiY * phiZ * phiYZ) - phiXX * (phiY * phiY + phiZ * phiZ) - phiYY * (phiX * phiX + phiZ * phiZ) - phiZZ * (phiX * phiX + phiY * phiY)) / (c2o1 * pow(phiX * phiX + phiY * phiY + phiZ * phiZ, c3o2));
    return (c2o1 * (phiX * phiY * phiXY + phiX * phiZ * phiXZ + phiY * phiZ * phiYZ) - phiXX * (phiY * phiY + phiZ * phiZ) - phiYY * (phiX * phiX + phiZ * phiZ) - phiZZ * (phiX * phiX + phiY * phiY)) / (c2o1 * pow(phiX * phiX + phiY * phiY + phiZ * phiZ, c3o2));
}

void IBsharpInterfaceLBMKernel::computePhasefield()
{
    using namespace D3Q27System;
    SPtr<DistributionArray3D> distributionsH = dataSet->getHdistributions();

    int minX1 = ghostLayerWidth;
    int minX2 = ghostLayerWidth;
    int minX3 = ghostLayerWidth;
    int maxX1 = (int)distributionsH->getNX1() - ghostLayerWidth;
    int maxX2 = (int)distributionsH->getNX2() - ghostLayerWidth;
    int maxX3 = (int)distributionsH->getNX3() - ghostLayerWidth;

    //------------- Computing the phase-field ------------------
    for (int x3 = minX3; x3 < maxX3; x3++) {
        for (int x2 = minX2; x2 < maxX2; x2++) {
            for (int x1 = minX1; x1 < maxX1; x1++) {
                // if(!bcArray->isSolid(x1,x2,x3) && !bcArray->isUndefined(x1,x2,x3))
                {
                    int x1p = x1 + 1;
                    int x2p = x2 + 1;
                    int x3p = x3 + 1;

                    h[dP00] = (*this->localDistributionsH1)(D3Q27System::ET_E, x1, x2, x3);
                    h[d0P0] = (*this->localDistributionsH1)(D3Q27System::ET_N, x1, x2, x3);
                    h[d00P] = (*this->localDistributionsH1)(D3Q27System::ET_T, x1, x2, x3);
                    h[dPP0] = (*this->localDistributionsH1)(D3Q27System::ET_NE, x1, x2, x3);
                    h[dMP0] = (*this->localDistributionsH1)(D3Q27System::ET_NW, x1p, x2, x3);
                    h[dP0P] = (*this->localDistributionsH1)(D3Q27System::ET_TE, x1, x2, x3);
                    h[dM0P] = (*this->localDistributionsH1)(D3Q27System::ET_TW, x1p, x2, x3);
                    h[d0PP] = (*this->localDistributionsH1)(D3Q27System::ET_TN, x1, x2, x3);
                    h[d0MP] = (*this->localDistributionsH1)(D3Q27System::ET_TS, x1, x2p, x3);
                    h[dPPP] = (*this->localDistributionsH1)(D3Q27System::ET_TNE, x1, x2, x3);
                    h[dMPP] = (*this->localDistributionsH1)(D3Q27System::ET_TNW, x1p, x2, x3);
                    h[dPMP] = (*this->localDistributionsH1)(D3Q27System::ET_TSE, x1, x2p, x3);
                    h[dMMP] = (*this->localDistributionsH1)(D3Q27System::ET_TSW, x1p, x2p, x3);

                    h[dM00] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_W, x1p, x2, x3);
                    h[d0M0] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_S, x1, x2p, x3);
                    h[d00M] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_B, x1, x2, x3p);
                    h[dMM0] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_SW, x1p, x2p, x3);
                    h[dPM0] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_SE, x1, x2p, x3);
                    h[dM0M] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BW, x1p, x2, x3p);
                    h[dP0M] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BE, x1, x2, x3p);
                    h[d0MM] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BS, x1, x2p, x3p);
                    h[d0PM] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BN, x1, x2, x3p);
                    h[dMMM] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BSW, x1p, x2p, x3p);
                    h[dPMM] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BSE, x1, x2p, x3p);
                    h[dMPM] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BNW, x1p, x2, x3p);
                    h[dPPM] = (*this->nonLocalDistributionsH1)(D3Q27System::ET_BNE, x1, x2, x3p);

                    h[d000] = (*this->restDistributionsH1)(x1, x2, x3);
                }
            }
        }
    }
}

void IBsharpInterfaceLBMKernel::findNeighbors(CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr ph, int x1, int x2, int x3)
{
    using namespace D3Q27System;

    SPtr<BCArray3D> bcArray = this->getBCSet()->getBCArray();

    phi[d000] = (*ph)(x1, x2, x3);

    for (int k = FSTARTDIR; k <= FENDDIR; k++) {

        if (!bcArray->isSolid(x1 + DX1[k], x2 + DX2[k], x3 + DX3[k])) {
            phi[k] = (*ph)(x1 + DX1[k], x2 + DX2[k], x3 + DX3[k]);
        } else {
            phi[k] = (*ph)(x1, x2, x3); // neutral wetting
                                        // phi[k] = 0.0;//unwetting
        }
    }
}

void IBsharpInterfaceLBMKernel::findNeighbors2(CbArray3D<real, IndexerX3X2X1>::CbArray3DPtr ph, int x1, int x2, int x3)
{
    using namespace D3Q27System;

    SPtr<BCArray3D> bcArray = this->getBCSet()->getBCArray();

    phi2[d000] = (*ph)(x1, x2, x3);

    for (int k = FSTARTDIR; k <= FENDDIR; k++) {

        if (!bcArray->isSolid(x1 + DX1[k], x2 + DX2[k], x3 + DX3[k])) {
            phi2[k] = (*ph)(x1 + DX1[k], x2 + DX2[k], x3 + DX3[k]);
        } else {
            phi2[k] = 0.05;
        }
    }
}

void IBsharpInterfaceLBMKernel::swapDistributions()
{
    LBMKernel::swapDistributions();
    dataSet->getHdistributions()->swap();
}

void IBsharpInterfaceLBMKernel::initForcing()
{
    muForcingX1.DefineVar("x1", &muX1);
    muForcingX1.DefineVar("x2", &muX2);
    muForcingX1.DefineVar("x3", &muX3);
    muForcingX2.DefineVar("x1", &muX1);
    muForcingX2.DefineVar("x2", &muX2);
    muForcingX2.DefineVar("x3", &muX3);
    muForcingX3.DefineVar("x1", &muX1);
    muForcingX3.DefineVar("x2", &muX2);
    muForcingX3.DefineVar("x3", &muX3);

    muDeltaT = deltaT;

    muForcingX1.DefineVar("dt", &muDeltaT);
    muForcingX2.DefineVar("dt", &muDeltaT);
    muForcingX3.DefineVar("dt", &muDeltaT);

    muNu = (1.0 / 3.0) * (1.0 / collFactor - 1.0 / 2.0);

    muForcingX1.DefineVar("nu", &muNu);
    muForcingX2.DefineVar("nu", &muNu);
    muForcingX3.DefineVar("nu", &muNu);

    muForcingX1.DefineVar("rho", &muRho);
    muForcingX2.DefineVar("rho", &muRho);
    muForcingX3.DefineVar("rho", &muRho);
}
