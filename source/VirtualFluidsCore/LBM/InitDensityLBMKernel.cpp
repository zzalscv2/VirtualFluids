#include "InitDensityLBMKernel.h"
#include "D3Q27EsoTwist3DSplittedVector.h"

InitDensityLBMKernel::InitDensityLBMKernel()
{
}

InitDensityLBMKernel::~InitDensityLBMKernel()
{
}

InitDensityLBMKernel::InitDensityLBMKernel(int nx1, int nx2, int nx3) : LBMKernelETD3Q27(nx1, nx2, nx3)
{
   this->compressible = false;
}

void InitDensityLBMKernel::calculate()
{
   collideAll();
}

LBMKernel3DPtr InitDensityLBMKernel::clone()
{
   LBMKernel3DPtr kernel(new InitDensityLBMKernel(nx1, nx2, nx3));
   boost::dynamic_pointer_cast<InitDensityLBMKernel>(kernel)->init();
   kernel->setCollisionFactor(this->collFactor);
   kernel->setBCProcessor(bcProcessor->clone(kernel));
   kernel->setWithForcing(withForcing);
   kernel->setForcingX1(muForcingX1);
   kernel->setForcingX2(muForcingX2);
   kernel->setForcingX3(muForcingX3);
   kernel->setIndex(ix1, ix2, ix3);
   kernel->setDeltaT(deltaT);
   boost::dynamic_pointer_cast<InitDensityLBMKernel>(kernel)->OxyyMxzz = 1.0;
   return kernel;
}

void InitDensityLBMKernel::setVelocity(int x1, int x2, int x3, LBMReal vvx, LBMReal vvy, LBMReal vvz)
{
   v(0, x1, x2, x3) = vvx;
   v(1, x1, x2, x3) = vvy;
   v(2, x1, x2, x3) = vvz;
}

double InitDensityLBMKernel::getCallculationTime()
{
   return 0;
}

//void InitDensityLBMKernel::collideAll()
//{
//   using namespace D3Q27System;
//
//   localDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(dataSet->getFdistributions())->getLocalDistributions();
//   nonLocalDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(dataSet->getFdistributions())->getNonLocalDistributions();
//   zeroDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(dataSet->getFdistributions())->getZeroDistributions();
//
//   BCArray3D<D3Q27BoundaryCondition>& bcArray = boost::dynamic_pointer_cast<D3Q27ETBCProcessor>(this->getBCProcessor())->getBCArray();
//
//   const int bcArrayMaxX1 = (int)bcArray.getNX1();
//   const int bcArrayMaxX2 = (int)bcArray.getNX2();
//   const int bcArrayMaxX3 = (int)bcArray.getNX3();
//
//   int minX1 = ghostLayerWidth;
//   int minX2 = ghostLayerWidth;
//   int minX3 = ghostLayerWidth;
//   int maxX1 = bcArrayMaxX1-ghostLayerWidth;
//   int maxX2 = bcArrayMaxX2-ghostLayerWidth;
//   int maxX3 = bcArrayMaxX3-ghostLayerWidth;
//
//
//   for (int x3 = minX3; x3<maxX3; x3++)
//   {
//      for (int x2 = minX2; x2<maxX2; x2++)
//      {
//         for (int x1 = minX1; x1<maxX1; x1++)
//         {
//            if (!bcArray.isSolid(x1, x2, x3)&&!bcArray.isUndefined(x1, x2, x3))
//            {
//               int x1p = x1+1;
//               int x2p = x2+1;
//               int x3p = x3+1;
//               //////////////////////////////////////////////////////////////////////////
//               //read distribution
//               ////////////////////////////////////////////////////////////////////////////
//               //////////////////////////////////////////////////////////////////////////
//
//               //E   N  T
//               //c   c  c
//               //////////
//               //W   S  B
//               //a   a  a
//
//               //Rest ist b
//
//               //mfxyz
//               //a - negative
//               //b - null
//               //c - positive
//
//               // a b c
//               //-1 0 1
//
//               LBMReal mfcbb = (*this->localDistributions)(D3Q27System::ET_E, x1, x2, x3);
//               LBMReal mfbcb = (*this->localDistributions)(D3Q27System::ET_N, x1, x2, x3);
//               LBMReal mfbbc = (*this->localDistributions)(D3Q27System::ET_T, x1, x2, x3);
//               LBMReal mfccb = (*this->localDistributions)(D3Q27System::ET_NE, x1, x2, x3);
//               LBMReal mfacb = (*this->localDistributions)(D3Q27System::ET_NW, x1p, x2, x3);
//               LBMReal mfcbc = (*this->localDistributions)(D3Q27System::ET_TE, x1, x2, x3);
//               LBMReal mfabc = (*this->localDistributions)(D3Q27System::ET_TW, x1p, x2, x3);
//               LBMReal mfbcc = (*this->localDistributions)(D3Q27System::ET_TN, x1, x2, x3);
//               LBMReal mfbac = (*this->localDistributions)(D3Q27System::ET_TS, x1, x2p, x3);
//               LBMReal mfccc = (*this->localDistributions)(D3Q27System::ET_TNE, x1, x2, x3);
//               LBMReal mfacc = (*this->localDistributions)(D3Q27System::ET_TNW, x1p, x2, x3);
//               LBMReal mfcac = (*this->localDistributions)(D3Q27System::ET_TSE, x1, x2p, x3);
//               LBMReal mfaac = (*this->localDistributions)(D3Q27System::ET_TSW, x1p, x2p, x3);
//
//               LBMReal mfabb = (*this->nonLocalDistributions)(D3Q27System::ET_W, x1p, x2, x3);
//               LBMReal mfbab = (*this->nonLocalDistributions)(D3Q27System::ET_S, x1, x2p, x3);
//               LBMReal mfbba = (*this->nonLocalDistributions)(D3Q27System::ET_B, x1, x2, x3p);
//               LBMReal mfaab = (*this->nonLocalDistributions)(D3Q27System::ET_SW, x1p, x2p, x3);
//               LBMReal mfcab = (*this->nonLocalDistributions)(D3Q27System::ET_SE, x1, x2p, x3);
//               LBMReal mfaba = (*this->nonLocalDistributions)(D3Q27System::ET_BW, x1p, x2, x3p);
//               LBMReal mfcba = (*this->nonLocalDistributions)(D3Q27System::ET_BE, x1, x2, x3p);
//               LBMReal mfbaa = (*this->nonLocalDistributions)(D3Q27System::ET_BS, x1, x2p, x3p);
//               LBMReal mfbca = (*this->nonLocalDistributions)(D3Q27System::ET_BN, x1, x2, x3p);
//               LBMReal mfaaa = (*this->nonLocalDistributions)(D3Q27System::ET_BSW, x1p, x2p, x3p);
//               LBMReal mfcaa = (*this->nonLocalDistributions)(D3Q27System::ET_BSE, x1, x2p, x3p);
//               LBMReal mfaca = (*this->nonLocalDistributions)(D3Q27System::ET_BNW, x1p, x2, x3p);
//               LBMReal mfcca = (*this->nonLocalDistributions)(D3Q27System::ET_BNE, x1, x2, x3p);
//
//               LBMReal mfbbb = (*this->zeroDistributions)(x1, x2, x3);
//
//               LBMReal m0, m1, m2;
//
//               LBMReal rho = (mfaaa+mfaac+mfaca+mfcaa+mfacc+mfcac+mfccc+mfcca)
//                  +(mfaab+mfacb+mfcab+mfccb)+(mfaba+mfabc+mfcba+mfcbc)+(mfbaa+mfbac+mfbca+mfbcc)
//                  +(mfabb+mfcbb)+(mfbab+mfbcb)+(mfbba+mfbbc)+mfbbb;
//
//               //LBMReal vvx = ((((mfccc-mfaaa)+(mfcac-mfaca))+((mfcaa-mfacc)+(mfcca-mfaac)))+
//               //   (((mfcba-mfabc)+(mfcbc-mfaba))+((mfcab-mfacb)+(mfccb-mfaab)))+
//               //   (mfcbb-mfabb));
//               //LBMReal vvy = ((((mfccc-mfaaa)+(mfaca-mfcac))+((mfacc-mfcaa)+(mfcca-mfaac)))+
//               //   (((mfbca-mfbac)+(mfbcc-mfbaa))+((mfacb-mfcab)+(mfccb-mfaab)))+
//               //   (mfbcb-mfbab));
//               //LBMReal vvz = ((((mfccc-mfaaa)+(mfcac-mfaca))+((mfacc-mfcaa)+(mfaac-mfcca)))+
//               //   (((mfbac-mfbca)+(mfbcc-mfbaa))+((mfabc-mfcba)+(mfcbc-mfaba)))+
//               //   (mfbbc-mfbba));
//
//               LBMReal vvx = v(0,x1,x2,x3);
//               LBMReal vvy = v(1,x1,x2,x3);
//               LBMReal vvz = v(2,x1,x2,x3);
//               //LBMReal rho = v(3,x1,x2,x3);
//          
//               LBMReal oMdrho;
//
//               oMdrho = mfccc+mfaaa;
//               m0 = mfaca+mfcac;
//               m1 = mfacc+mfcaa;
//               m2 = mfaac+mfcca;
//               oMdrho += m0;
//               m1 += m2;
//               oMdrho += m1;
//               m0 = mfbac+mfbca;
//               m1 = mfbaa+mfbcc;
//               m0 += m1;
//               m1 = mfabc+mfcba;
//               m2 = mfaba+mfcbc;
//               m1 += m2;
//               m0 += m1;
//               m1 = mfacb+mfcab;
//               m2 = mfaab+mfccb;
//               m1 += m2;
//               m0 += m1;
//               oMdrho += m0;
//               m0 = mfabb+mfcbb;
//               m1 = mfbab+mfbcb;
//               m2 = mfbba+mfbbc;
//               m0 += m1+m2;
//               m0 += mfbbb; //hat gefehlt
//               oMdrho = 1.-(oMdrho+m0);
//
//               LBMReal vx2;
//               LBMReal vy2;
//               LBMReal vz2;
//               vx2 = vvx*vvx;
//               vy2 = vvy*vvy;
//               vz2 = vvz*vvz;
//               ////////////////////////////////////////////////////////////////////////////////////
//               LBMReal wadjust;
//               LBMReal qudricLimit = 0.01;
//               ////////////////////////////////////////////////////////////////////////////////////
//               //Hin
//               ////////////////////////////////////////////////////////////////////////////////////
//               // mit 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36  Konditionieren
//               ////////////////////////////////////////////////////////////////////////////////////
//               // Z - Dir
//               m2 = mfaaa+mfaac;
//               m1 = mfaac-mfaaa;
//               m0 = m2+mfaab;
//               mfaaa = m0;
//               m0 += c1o36 * oMdrho;
//               mfaab = m1-m0 * vvz;
//               mfaac = m2-2. *   m1 * vvz+vz2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfaba+mfabc;
//               m1 = mfabc-mfaba;
//               m0 = m2+mfabb;
//               mfaba = m0;
//               m0 += c1o9 * oMdrho;
//               mfabb = m1-m0 * vvz;
//               mfabc = m2-2. *   m1 * vvz+vz2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfaca+mfacc;
//               m1 = mfacc-mfaca;
//               m0 = m2+mfacb;
//               mfaca = m0;
//               m0 += c1o36 * oMdrho;
//               mfacb = m1-m0 * vvz;
//               mfacc = m2-2. *   m1 * vvz+vz2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfbaa+mfbac;
//               m1 = mfbac-mfbaa;
//               m0 = m2+mfbab;
//               mfbaa = m0;
//               m0 += c1o9 * oMdrho;
//               mfbab = m1-m0 * vvz;
//               mfbac = m2-2. *   m1 * vvz+vz2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfbba+mfbbc;
//               m1 = mfbbc-mfbba;
//               m0 = m2+mfbbb;
//               mfbba = m0;
//               m0 += c4o9 * oMdrho;
//               mfbbb = m1-m0 * vvz;
//               mfbbc = m2-2. *   m1 * vvz+vz2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfbca+mfbcc;
//               m1 = mfbcc-mfbca;
//               m0 = m2+mfbcb;
//               mfbca = m0;
//               m0 += c1o9 * oMdrho;
//               mfbcb = m1-m0 * vvz;
//               mfbcc = m2-2. *   m1 * vvz+vz2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfcaa+mfcac;
//               m1 = mfcac-mfcaa;
//               m0 = m2+mfcab;
//               mfcaa = m0;
//               m0 += c1o36 * oMdrho;
//               mfcab = m1-m0 * vvz;
//               mfcac = m2-2. *   m1 * vvz+vz2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfcba+mfcbc;
//               m1 = mfcbc-mfcba;
//               m0 = m2+mfcbb;
//               mfcba = m0;
//               m0 += c1o9 * oMdrho;
//               mfcbb = m1-m0 * vvz;
//               mfcbc = m2-2. *   m1 * vvz+vz2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfcca+mfccc;
//               m1 = mfccc-mfcca;
//               m0 = m2+mfccb;
//               mfcca = m0;
//               m0 += c1o36 * oMdrho;
//               mfccb = m1-m0 * vvz;
//               mfccc = m2-2. *   m1 * vvz+vz2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               // mit  1/6, 0, 1/18, 2/3, 0, 2/9, 1/6, 0, 1/18 Konditionieren
//               ////////////////////////////////////////////////////////////////////////////////////
//               // Y - Dir
//               m2 = mfaaa+mfaca;
//               m1 = mfaca-mfaaa;
//               m0 = m2+mfaba;
//               mfaaa = m0;
//               m0 += c1o6 * oMdrho;
//               mfaba = m1-m0 * vvy;
//               mfaca = m2-2. *   m1 * vvy+vy2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfaab+mfacb;
//               m1 = mfacb-mfaab;
//               m0 = m2+mfabb;
//               mfaab = m0;
//               mfabb = m1-m0 * vvy;
//               mfacb = m2-2. *   m1 * vvy+vy2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfaac+mfacc;
//               m1 = mfacc-mfaac;
//               m0 = m2+mfabc;
//               mfaac = m0;
//               m0 += c1o18 * oMdrho;
//               mfabc = m1-m0 * vvy;
//               mfacc = m2-2. *   m1 * vvy+vy2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfbaa+mfbca;
//               m1 = mfbca-mfbaa;
//               m0 = m2+mfbba;
//               mfbaa = m0;
//               m0 += c2o3 * oMdrho;
//               mfbba = m1-m0 * vvy;
//               mfbca = m2-2. *   m1 * vvy+vy2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfbab+mfbcb;
//               m1 = mfbcb-mfbab;
//               m0 = m2+mfbbb;
//               mfbab = m0;
//               mfbbb = m1-m0 * vvy;
//               mfbcb = m2-2. *   m1 * vvy+vy2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfbac+mfbcc;
//               m1 = mfbcc-mfbac;
//               m0 = m2+mfbbc;
//               mfbac = m0;
//               m0 += c2o9 * oMdrho;
//               mfbbc = m1-m0 * vvy;
//               mfbcc = m2-2. *   m1 * vvy+vy2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfcaa+mfcca;
//               m1 = mfcca-mfcaa;
//               m0 = m2+mfcba;
//               mfcaa = m0;
//               m0 += c1o6 * oMdrho;
//               mfcba = m1-m0 * vvy;
//               mfcca = m2-2. *   m1 * vvy+vy2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfcab+mfccb;
//               m1 = mfccb-mfcab;
//               m0 = m2+mfcbb;
//               mfcab = m0;
//               mfcbb = m1-m0 * vvy;
//               mfccb = m2-2. *   m1 * vvy+vy2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfcac+mfccc;
//               m1 = mfccc-mfcac;
//               m0 = m2+mfcbc;
//               mfcac = m0;
//               m0 += c1o18 * oMdrho;
//               mfcbc = m1-m0 * vvy;
//               mfccc = m2-2. *   m1 * vvy+vy2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               // mit     1, 0, 1/3, 0, 0, 0, 1/3, 0, 1/9            Konditionieren
//               ////////////////////////////////////////////////////////////////////////////////////
//               // X - Dir
//               m2 = mfaaa+mfcaa;
//               m1 = mfcaa-mfaaa;
//               m0 = m2+mfbaa;
//               mfaaa = m0;
//               m0 += 1. * oMdrho;
//               mfbaa = m1-m0 * vvx;
//               mfcaa = m2-2. *   m1 * vvx+vx2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfaba+mfcba;
//               m1 = mfcba-mfaba;
//               m0 = m2+mfbba;
//               mfaba = m0;
//               mfbba = m1-m0 * vvx;
//               mfcba = m2-2. *   m1 * vvx+vx2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfaca+mfcca;
//               m1 = mfcca-mfaca;
//               m0 = m2+mfbca;
//               mfaca = m0;
//               m0 += c1o3 * oMdrho;
//               mfbca = m1-m0 * vvx;
//               mfcca = m2-2. *   m1 * vvx+vx2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfaab+mfcab;
//               m1 = mfcab-mfaab;
//               m0 = m2+mfbab;
//               mfaab = m0;
//               mfbab = m1-m0 * vvx;
//               mfcab = m2-2. *   m1 * vvx+vx2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfabb+mfcbb;
//               m1 = mfcbb-mfabb;
//               m0 = m2+mfbbb;
//               mfabb = m0;
//               mfbbb = m1-m0 * vvx;
//               mfcbb = m2-2. *   m1 * vvx+vx2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfacb+mfccb;
//               m1 = mfccb-mfacb;
//               m0 = m2+mfbcb;
//               mfacb = m0;
//               mfbcb = m1-m0 * vvx;
//               mfccb = m2-2. *   m1 * vvx+vx2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfaac+mfcac;
//               m1 = mfcac-mfaac;
//               m0 = m2+mfbac;
//               mfaac = m0;
//               m0 += c1o3 * oMdrho;
//               mfbac = m1-m0 * vvx;
//               mfcac = m2-2. *   m1 * vvx+vx2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfabc+mfcbc;
//               m1 = mfcbc-mfabc;
//               m0 = m2+mfbbc;
//               mfabc = m0;
//               mfbbc = m1-m0 * vvx;
//               mfcbc = m2-2. *   m1 * vvx+vx2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m2 = mfacc+mfccc;
//               m1 = mfccc-mfacc;
//               m0 = m2+mfbcc;
//               mfacc = m0;
//               m0 += c1o9 * oMdrho;
//               mfbcc = m1-m0 * vvx;
//               mfccc = m2-2. *   m1 * vvx+vx2 * m0;
//               ////////////////////////////////////////////////////////////////////////////////////
//               // Cumulants
//               ////////////////////////////////////////////////////////////////////////////////////
//               LBMReal OxxPyyPzz = 1.; //omega2 or bulk viscosity
//               LBMReal OxyyPxzz = 1.;//-s9;//2+s9;//
//               //LBMReal OxyyMxzz  = 1.;//2+s9;//
//               LBMReal O4 = 1.;
//               LBMReal O5 = 1.;
//               LBMReal O6 = 1.;
//
//               //Cum 4.
//               //LBMReal CUMcbb = mfcbb - ((mfcaa + c1o3 * oMdrho) * mfabb + 2. * mfbba * mfbab); // till 18.05.2015
//               //LBMReal CUMbcb = mfbcb - ((mfaca + c1o3 * oMdrho) * mfbab + 2. * mfbba * mfabb); // till 18.05.2015
//               //LBMReal CUMbbc = mfbbc - ((mfaac + c1o3 * oMdrho) * mfbba + 2. * mfbab * mfabb); // till 18.05.2015
//
//               LBMReal CUMcbb = mfcbb-((mfcaa+c1o3) * mfabb+2. * mfbba * mfbab);
//               LBMReal CUMbcb = mfbcb-((mfaca+c1o3) * mfbab+2. * mfbba * mfabb);
//               LBMReal CUMbbc = mfbbc-((mfaac+c1o3) * mfbba+2. * mfbab * mfabb);
//
//               LBMReal CUMcca = mfcca-((mfcaa * mfaca+2. * mfbba * mfbba)+c1o3 * (mfcaa+mfaca) * oMdrho+c1o9*(oMdrho-1)*oMdrho);
//               LBMReal CUMcac = mfcac-((mfcaa * mfaac+2. * mfbab * mfbab)+c1o3 * (mfcaa+mfaac) * oMdrho+c1o9*(oMdrho-1)*oMdrho);
//               LBMReal CUMacc = mfacc-((mfaac * mfaca+2. * mfabb * mfabb)+c1o3 * (mfaac+mfaca) * oMdrho+c1o9*(oMdrho-1)*oMdrho);
//
//               //Cum 5.
//               LBMReal CUMbcc = mfbcc-(mfaac * mfbca+mfaca * mfbac+4. * mfabb * mfbbb+2. * (mfbab * mfacb+mfbba * mfabc))-c1o3 * (mfbca+mfbac) * oMdrho;
//               LBMReal CUMcbc = mfcbc-(mfaac * mfcba+mfcaa * mfabc+4. * mfbab * mfbbb+2. * (mfabb * mfcab+mfbba * mfbac))-c1o3 * (mfcba+mfabc) * oMdrho;
//               LBMReal CUMccb = mfccb-(mfcaa * mfacb+mfaca * mfcab+4. * mfbba * mfbbb+2. * (mfbab * mfbca+mfabb * mfcba))-c1o3 * (mfacb+mfcab) * oMdrho;
//
//               //Cum 6.
//               LBMReal CUMccc = mfccc+((-4. *  mfbbb * mfbbb
//                  -(mfcaa * mfacc+mfaca * mfcac+mfaac * mfcca)
//                  -4. * (mfabb * mfcbb+mfbab * mfbcb+mfbba * mfbbc)
//                  -2. * (mfbca * mfbac+mfcba * mfabc+mfcab * mfacb))
//                  +(4. * (mfbab * mfbab * mfaca+mfabb * mfabb * mfcaa+mfbba * mfbba * mfaac)
//                     +2. * (mfcaa * mfaca * mfaac)
//                     +16. *  mfbba * mfbab * mfabb)
//                  -c1o3* (mfacc+mfcac+mfcca) * oMdrho-c1o9*oMdrho*oMdrho
//                  -c1o9* (mfcaa+mfaca+mfaac) * oMdrho*(1.-2.* oMdrho)-c1o27* oMdrho * oMdrho*(-2.* oMdrho)
//                  +(2. * (mfbab * mfbab+mfabb * mfabb+mfbba * mfbba)
//                     +(mfaac * mfaca+mfaac * mfcaa+mfaca * mfcaa)) * c2o3*oMdrho)+c1o27*oMdrho;
//
//               //2.
//               // linear combinations
//               LBMReal mxxPyyPzz = mfcaa+mfaca+mfaac;
//               LBMReal mxxMyy = mfcaa-mfaca;
//               LBMReal mxxMzz = mfcaa-mfaac;
//
//               LBMReal dxux = -c1o2 * collFactor *(mxxMyy+mxxMzz)+c1o2 * OxxPyyPzz*(mfaaa-mxxPyyPzz);
//               LBMReal dyuy = dxux+collFactor * c3o2 * mxxMyy;
//               LBMReal dzuz = dxux+collFactor * c3o2 * mxxMzz;
//
//               //relax
//               mxxPyyPzz += OxxPyyPzz*(mfaaa-mxxPyyPzz)-3. * (1.-c1o2 * OxxPyyPzz) * (vx2 * dxux+vy2 * dyuy+vz2 * dzuz);
//               mxxMyy += collFactor * (-mxxMyy)-3. * (1.-c1o2 * collFactor) * (vx2 * dxux-vy2 * dyuy);
//               mxxMzz += collFactor * (-mxxMzz)-3. * (1.-c1o2 * collFactor) * (vx2 * dxux-vz2 * dzuz);
//
//               mfabb += collFactor * (-mfabb);
//               mfbab += collFactor * (-mfbab);
//               mfbba += collFactor * (-mfbba);
//
//               // linear combinations back
//               mfcaa = c1o3 * (mxxMyy+mxxMzz+mxxPyyPzz);
//               mfaca = c1o3 * (-2. *  mxxMyy+mxxMzz+mxxPyyPzz);
//               mfaac = c1o3 * (mxxMyy-2. * mxxMzz+mxxPyyPzz);
//
//               //3.
//               // linear combinations
//               LBMReal mxxyPyzz = mfcba+mfabc;
//               LBMReal mxxyMyzz = mfcba-mfabc;
//
//               LBMReal mxxzPyyz = mfcab+mfacb;
//               LBMReal mxxzMyyz = mfcab-mfacb;
//
//               LBMReal mxyyPxzz = mfbca+mfbac;
//               LBMReal mxyyMxzz = mfbca-mfbac;
//
//               //relax
//               wadjust = OxyyMxzz+(1.-OxyyMxzz)*fabs(mfbbb)/(fabs(mfbbb)+qudricLimit);
//               mfbbb += wadjust * (-mfbbb);
//               wadjust = OxyyPxzz+(1.-OxyyPxzz)*fabs(mxxyPyzz)/(fabs(mxxyPyzz)+qudricLimit);
//               mxxyPyzz += wadjust * (-mxxyPyzz);
//               wadjust = OxyyMxzz+(1.-OxyyMxzz)*fabs(mxxyMyzz)/(fabs(mxxyMyzz)+qudricLimit);
//               mxxyMyzz += wadjust * (-mxxyMyzz);
//               wadjust = OxyyPxzz+(1.-OxyyPxzz)*fabs(mxxzPyyz)/(fabs(mxxzPyyz)+qudricLimit);
//               mxxzPyyz += wadjust * (-mxxzPyyz);
//               wadjust = OxyyMxzz+(1.-OxyyMxzz)*fabs(mxxzMyyz)/(fabs(mxxzMyyz)+qudricLimit);
//               mxxzMyyz += wadjust * (-mxxzMyyz);
//               wadjust = OxyyPxzz+(1.-OxyyPxzz)*fabs(mxyyPxzz)/(fabs(mxyyPxzz)+qudricLimit);
//               mxyyPxzz += wadjust * (-mxyyPxzz);
//               wadjust = OxyyMxzz+(1.-OxyyMxzz)*fabs(mxyyMxzz)/(fabs(mxyyMxzz)+qudricLimit);
//               mxyyMxzz += wadjust * (-mxyyMxzz);
//
//               // linear combinations back
//               mfcba = (mxxyMyzz+mxxyPyzz) * c1o2;
//               mfabc = (-mxxyMyzz+mxxyPyzz) * c1o2;
//               mfcab = (mxxzMyyz+mxxzPyyz) * c1o2;
//               mfacb = (-mxxzMyyz+mxxzPyyz) * c1o2;
//               mfbca = (mxyyMxzz+mxyyPxzz) * c1o2;
//               mfbac = (-mxyyMxzz+mxyyPxzz) * c1o2;
//
//               //4.
//               CUMacc += O4 * (-CUMacc);
//               CUMcac += O4 * (-CUMcac);
//               CUMcca += O4 * (-CUMcca);
//
//               CUMbbc += O4 * (-CUMbbc);
//               CUMbcb += O4 * (-CUMbcb);
//               CUMcbb += O4 * (-CUMcbb);
//
//               //5.
//               CUMbcc += O5 * (-CUMbcc);
//               CUMcbc += O5 * (-CUMcbc);
//               CUMccb += O5 * (-CUMccb);
//
//               //6.
//               CUMccc += O6 * (-CUMccc);
//
//               //back cumulants to central moments
//               //4.
//               //mfcbb = CUMcbb + ((mfcaa + c1o3 * oMdrho) * mfabb + 2. * mfbba * mfbab); // till 18.05.2015
//               //mfbcb = CUMbcb + ((mfaca + c1o3 * oMdrho) * mfbab + 2. * mfbba * mfabb); // till 18.05.2015
//               //mfbbc = CUMbbc + ((mfaac + c1o3 * oMdrho) * mfbba + 2. * mfbab * mfabb); // till 18.05.2015
//
//               mfcbb = CUMcbb+((mfcaa+c1o3) * mfabb+2. * mfbba * mfbab);
//               mfbcb = CUMbcb+((mfaca+c1o3) * mfbab+2. * mfbba * mfabb);
//               mfbbc = CUMbbc+((mfaac+c1o3) * mfbba+2. * mfbab * mfabb);
//
//               mfcca = CUMcca+(mfcaa * mfaca+2. * mfbba * mfbba)+c1o3 * (mfcaa+mfaca) * oMdrho+c1o9*(oMdrho-1)*oMdrho;
//               mfcac = CUMcac+(mfcaa * mfaac+2. * mfbab * mfbab)+c1o3 * (mfcaa+mfaac) * oMdrho+c1o9*(oMdrho-1)*oMdrho;
//               mfacc = CUMacc+(mfaac * mfaca+2. * mfabb * mfabb)+c1o3 * (mfaac+mfaca) * oMdrho+c1o9*(oMdrho-1)*oMdrho;
//
//               //5.
//               mfbcc = CUMbcc+(mfaac * mfbca+mfaca * mfbac+4. * mfabb * mfbbb+2. * (mfbab * mfacb+mfbba * mfabc))+c1o3 * (mfbca+mfbac) * oMdrho;
//               mfcbc = CUMcbc+(mfaac * mfcba+mfcaa * mfabc+4. * mfbab * mfbbb+2. * (mfabb * mfcab+mfbba * mfbac))+c1o3 * (mfcba+mfabc) * oMdrho;
//               mfccb = CUMccb+(mfcaa * mfacb+mfaca * mfcab+4. * mfbba * mfbbb+2. * (mfbab * mfbca+mfabb * mfcba))+c1o3 * (mfacb+mfcab) * oMdrho;
//
//               //6.
//               mfccc = CUMccc-((-4. *  mfbbb * mfbbb
//                  -(mfcaa * mfacc+mfaca * mfcac+mfaac * mfcca)
//                  -4. * (mfabb * mfcbb+mfbac * mfbca+mfbba * mfbbc)
//                  -2. * (mfbca * mfbac+mfcba * mfabc+mfcab * mfacb))
//                  +(4. * (mfbab * mfbab * mfaca+mfabb * mfabb * mfcaa+mfbba * mfbba * mfaac)
//                     +2. * (mfcaa * mfaca * mfaac)
//                     +16. *  mfbba * mfbab * mfabb)
//                  -c1o3* (mfacc+mfcac+mfcca) * oMdrho-c1o9*oMdrho*oMdrho
//                  -c1o9* (mfcaa+mfaca+mfaac) * oMdrho*(1.-2.* oMdrho)-c1o27* oMdrho * oMdrho*(-2.* oMdrho)
//                  +(2. * (mfbab * mfbab+mfabb * mfabb+mfbba * mfbba)
//                     +(mfaac * mfaca+mfaac * mfcaa+mfaca * mfcaa)) * c2o3*oMdrho)-c1o27*oMdrho;
//
//               mfaab = 0.0;
//               mfaba = 0.0;
//               mfbaa = 0.0;
//
//               //mfaab *= -0.5;
//               //mfaba *= -0.5;
//               //mfbaa *= -0.5;
//
//
//               ////////////////////////////////////////////////////////////////////////////////////
//               //back
//               ////////////////////////////////////////////////////////////////////////////////////
//               //mit 1, 0, 1/3, 0, 0, 0, 1/3, 0, 1/9   Konditionieren
//               ////////////////////////////////////////////////////////////////////////////////////
//               // Z - Dir
//               m0 = mfaac * c1o2+mfaab * (vvz-c1o2)+(mfaaa+1. * oMdrho) * (vz2-vvz) * c1o2;
//               m1 = -mfaac-2. * mfaab *  vvz+mfaaa                * (1.-vz2)-1. * oMdrho * vz2;
//               m2 = mfaac * c1o2+mfaab * (vvz+c1o2)+(mfaaa+1. * oMdrho) * (vz2+vvz) * c1o2;
//               mfaaa = m0;
//               mfaab = m1;
//               mfaac = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfabc * c1o2+mfabb * (vvz-c1o2)+mfaba * (vz2-vvz) * c1o2;
//               m1 = -mfabc-2. * mfabb *  vvz+mfaba * (1.-vz2);
//               m2 = mfabc * c1o2+mfabb * (vvz+c1o2)+mfaba * (vz2+vvz) * c1o2;
//               mfaba = m0;
//               mfabb = m1;
//               mfabc = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfacc * c1o2+mfacb * (vvz-c1o2)+(mfaca+c1o3 * oMdrho) * (vz2-vvz) * c1o2;
//               m1 = -mfacc-2. * mfacb *  vvz+mfaca                  * (1.-vz2)-c1o3 * oMdrho * vz2;
//               m2 = mfacc * c1o2+mfacb * (vvz+c1o2)+(mfaca+c1o3 * oMdrho) * (vz2+vvz) * c1o2;
//               mfaca = m0;
//               mfacb = m1;
//               mfacc = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfbac * c1o2+mfbab * (vvz-c1o2)+mfbaa * (vz2-vvz) * c1o2;
//               m1 = -mfbac-2. * mfbab *  vvz+mfbaa * (1.-vz2);
//               m2 = mfbac * c1o2+mfbab * (vvz+c1o2)+mfbaa * (vz2+vvz) * c1o2;
//               mfbaa = m0;
//               mfbab = m1;
//               mfbac = m2;
//               /////////b//////////////////////////////////////////////////////////////////////////
//               m0 = mfbbc * c1o2+mfbbb * (vvz-c1o2)+mfbba * (vz2-vvz) * c1o2;
//               m1 = -mfbbc-2. * mfbbb *  vvz+mfbba * (1.-vz2);
//               m2 = mfbbc * c1o2+mfbbb * (vvz+c1o2)+mfbba * (vz2+vvz) * c1o2;
//               mfbba = m0;
//               mfbbb = m1;
//               mfbbc = m2;
//               /////////b//////////////////////////////////////////////////////////////////////////
//               m0 = mfbcc * c1o2+mfbcb * (vvz-c1o2)+mfbca * (vz2-vvz) * c1o2;
//               m1 = -mfbcc-2. * mfbcb *  vvz+mfbca * (1.-vz2);
//               m2 = mfbcc * c1o2+mfbcb * (vvz+c1o2)+mfbca * (vz2+vvz) * c1o2;
//               mfbca = m0;
//               mfbcb = m1;
//               mfbcc = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfcac * c1o2+mfcab * (vvz-c1o2)+(mfcaa+c1o3 * oMdrho) * (vz2-vvz) * c1o2;
//               m1 = -mfcac-2. * mfcab *  vvz+mfcaa                  * (1.-vz2)-c1o3 * oMdrho * vz2;
//               m2 = mfcac * c1o2+mfcab * (vvz+c1o2)+(mfcaa+c1o3 * oMdrho) * (vz2+vvz) * c1o2;
//               mfcaa = m0;
//               mfcab = m1;
//               mfcac = m2;
//               /////////c//////////////////////////////////////////////////////////////////////////
//               m0 = mfcbc * c1o2+mfcbb * (vvz-c1o2)+mfcba * (vz2-vvz) * c1o2;
//               m1 = -mfcbc-2. * mfcbb *  vvz+mfcba * (1.-vz2);
//               m2 = mfcbc * c1o2+mfcbb * (vvz+c1o2)+mfcba * (vz2+vvz) * c1o2;
//               mfcba = m0;
//               mfcbb = m1;
//               mfcbc = m2;
//               /////////c//////////////////////////////////////////////////////////////////////////
//               m0 = mfccc * c1o2+mfccb * (vvz-c1o2)+(mfcca+c1o9 * oMdrho) * (vz2-vvz) * c1o2;
//               m1 = -mfccc-2. * mfccb *  vvz+mfcca                  * (1.-vz2)-c1o9 * oMdrho * vz2;
//               m2 = mfccc * c1o2+mfccb * (vvz+c1o2)+(mfcca+c1o9 * oMdrho) * (vz2+vvz) * c1o2;
//               mfcca = m0;
//               mfccb = m1;
//               mfccc = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               //mit 1/6, 2/3, 1/6, 0, 0, 0, 1/18, 2/9, 1/18   Konditionieren
//               ////////////////////////////////////////////////////////////////////////////////////
//               // Y - Dir
//               m0 = mfaca * c1o2+mfaba * (vvy-c1o2)+(mfaaa+c1o6 * oMdrho) * (vy2-vvy) * c1o2;
//               m1 = -mfaca-2. * mfaba *  vvy+mfaaa                  * (1.-vy2)-c1o6 * oMdrho * vy2;
//               m2 = mfaca * c1o2+mfaba * (vvy+c1o2)+(mfaaa+c1o6 * oMdrho) * (vy2+vvy) * c1o2;
//               mfaaa = m0;
//               mfaba = m1;
//               mfaca = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfacb * c1o2+mfabb * (vvy-c1o2)+(mfaab+c2o3 * oMdrho) * (vy2-vvy) * c1o2;
//               m1 = -mfacb-2. * mfabb *  vvy+mfaab                  * (1.-vy2)-c2o3 * oMdrho * vy2;
//               m2 = mfacb * c1o2+mfabb * (vvy+c1o2)+(mfaab+c2o3 * oMdrho) * (vy2+vvy) * c1o2;
//               mfaab = m0;
//               mfabb = m1;
//               mfacb = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfacc * c1o2+mfabc * (vvy-c1o2)+(mfaac+c1o6 * oMdrho) * (vy2-vvy) * c1o2;
//               m1 = -mfacc-2. * mfabc *  vvy+mfaac                  * (1.-vy2)-c1o6 * oMdrho * vy2;
//               m2 = mfacc * c1o2+mfabc * (vvy+c1o2)+(mfaac+c1o6 * oMdrho) * (vy2+vvy) * c1o2;
//               mfaac = m0;
//               mfabc = m1;
//               mfacc = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfbca * c1o2+mfbba * (vvy-c1o2)+mfbaa * (vy2-vvy) * c1o2;
//               m1 = -mfbca-2. * mfbba *  vvy+mfbaa * (1.-vy2);
//               m2 = mfbca * c1o2+mfbba * (vvy+c1o2)+mfbaa * (vy2+vvy) * c1o2;
//               mfbaa = m0;
//               mfbba = m1;
//               mfbca = m2;
//               /////////b//////////////////////////////////////////////////////////////////////////
//               m0 = mfbcb * c1o2+mfbbb * (vvy-c1o2)+mfbab * (vy2-vvy) * c1o2;
//               m1 = -mfbcb-2. * mfbbb *  vvy+mfbab * (1.-vy2);
//               m2 = mfbcb * c1o2+mfbbb * (vvy+c1o2)+mfbab * (vy2+vvy) * c1o2;
//               mfbab = m0;
//               mfbbb = m1;
//               mfbcb = m2;
//               /////////b//////////////////////////////////////////////////////////////////////////
//               m0 = mfbcc * c1o2+mfbbc * (vvy-c1o2)+mfbac * (vy2-vvy) * c1o2;
//               m1 = -mfbcc-2. * mfbbc *  vvy+mfbac * (1.-vy2);
//               m2 = mfbcc * c1o2+mfbbc * (vvy+c1o2)+mfbac * (vy2+vvy) * c1o2;
//               mfbac = m0;
//               mfbbc = m1;
//               mfbcc = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfcca * c1o2+mfcba * (vvy-c1o2)+(mfcaa+c1o18 * oMdrho) * (vy2-vvy) * c1o2;
//               m1 = -mfcca-2. * mfcba *  vvy+mfcaa                   * (1.-vy2)-c1o18 * oMdrho * vy2;
//               m2 = mfcca * c1o2+mfcba * (vvy+c1o2)+(mfcaa+c1o18 * oMdrho) * (vy2+vvy) * c1o2;
//               mfcaa = m0;
//               mfcba = m1;
//               mfcca = m2;
//               /////////c//////////////////////////////////////////////////////////////////////////
//               m0 = mfccb * c1o2+mfcbb * (vvy-c1o2)+(mfcab+c2o9 * oMdrho) * (vy2-vvy) * c1o2;
//               m1 = -mfccb-2. * mfcbb *  vvy+mfcab                  * (1.-vy2)-c2o9 * oMdrho * vy2;
//               m2 = mfccb * c1o2+mfcbb * (vvy+c1o2)+(mfcab+c2o9 * oMdrho) * (vy2+vvy) * c1o2;
//               mfcab = m0;
//               mfcbb = m1;
//               mfccb = m2;
//               /////////c//////////////////////////////////////////////////////////////////////////
//               m0 = mfccc * c1o2+mfcbc * (vvy-c1o2)+(mfcac+c1o18 * oMdrho) * (vy2-vvy) * c1o2;
//               m1 = -mfccc-2. * mfcbc *  vvy+mfcac                   * (1.-vy2)-c1o18 * oMdrho * vy2;
//               m2 = mfccc * c1o2+mfcbc * (vvy+c1o2)+(mfcac+c1o18 * oMdrho) * (vy2+vvy) * c1o2;
//               mfcac = m0;
//               mfcbc = m1;
//               mfccc = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               //mit 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36 Konditionieren
//               ////////////////////////////////////////////////////////////////////////////////////
//               // X - Dir
//               m0 = mfcaa * c1o2+mfbaa * (vvx-c1o2)+(mfaaa+c1o36 * oMdrho) * (vx2-vvx) * c1o2;
//               m1 = -mfcaa-2. * mfbaa *  vvx+mfaaa                   * (1.-vx2)-c1o36 * oMdrho * vx2;
//               m2 = mfcaa * c1o2+mfbaa * (vvx+c1o2)+(mfaaa+c1o36 * oMdrho) * (vx2+vvx) * c1o2;
//               mfaaa = m0;
//               mfbaa = m1;
//               mfcaa = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfcba * c1o2+mfbba * (vvx-c1o2)+(mfaba+c1o9 * oMdrho) * (vx2-vvx) * c1o2;
//               m1 = -mfcba-2. * mfbba *  vvx+mfaba                  * (1.-vx2)-c1o9 * oMdrho * vx2;
//               m2 = mfcba * c1o2+mfbba * (vvx+c1o2)+(mfaba+c1o9 * oMdrho) * (vx2+vvx) * c1o2;
//               mfaba = m0;
//               mfbba = m1;
//               mfcba = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfcca * c1o2+mfbca * (vvx-c1o2)+(mfaca+c1o36 * oMdrho) * (vx2-vvx) * c1o2;
//               m1 = -mfcca-2. * mfbca *  vvx+mfaca                   * (1.-vx2)-c1o36 * oMdrho * vx2;
//               m2 = mfcca * c1o2+mfbca * (vvx+c1o2)+(mfaca+c1o36 * oMdrho) * (vx2+vvx) * c1o2;
//               mfaca = m0;
//               mfbca = m1;
//               mfcca = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfcab * c1o2+mfbab * (vvx-c1o2)+(mfaab+c1o9 * oMdrho) * (vx2-vvx) * c1o2;
//               m1 = -mfcab-2. * mfbab *  vvx+mfaab                  * (1.-vx2)-c1o9 * oMdrho * vx2;
//               m2 = mfcab * c1o2+mfbab * (vvx+c1o2)+(mfaab+c1o9 * oMdrho) * (vx2+vvx) * c1o2;
//               mfaab = m0;
//               mfbab = m1;
//               mfcab = m2;
//               ///////////b////////////////////////////////////////////////////////////////////////
//               m0 = mfcbb * c1o2+mfbbb * (vvx-c1o2)+(mfabb+c4o9 * oMdrho) * (vx2-vvx) * c1o2;
//               m1 = -mfcbb-2. * mfbbb *  vvx+mfabb                  * (1.-vx2)-c4o9 * oMdrho * vx2;
//               m2 = mfcbb * c1o2+mfbbb * (vvx+c1o2)+(mfabb+c4o9 * oMdrho) * (vx2+vvx) * c1o2;
//               mfabb = m0;
//               mfbbb = m1;
//               mfcbb = m2;
//               ///////////b////////////////////////////////////////////////////////////////////////
//               m0 = mfccb * c1o2+mfbcb * (vvx-c1o2)+(mfacb+c1o9 * oMdrho) * (vx2-vvx) * c1o2;
//               m1 = -mfccb-2. * mfbcb *  vvx+mfacb                  * (1.-vx2)-c1o9 * oMdrho * vx2;
//               m2 = mfccb * c1o2+mfbcb * (vvx+c1o2)+(mfacb+c1o9 * oMdrho) * (vx2+vvx) * c1o2;
//               mfacb = m0;
//               mfbcb = m1;
//               mfccb = m2;
//               ////////////////////////////////////////////////////////////////////////////////////
//               ////////////////////////////////////////////////////////////////////////////////////
//               m0 = mfcac * c1o2+mfbac * (vvx-c1o2)+(mfaac+c1o36 * oMdrho) * (vx2-vvx) * c1o2;
//               m1 = -mfcac-2. * mfbac *  vvx+mfaac                   * (1.-vx2)-c1o36 * oMdrho * vx2;
//               m2 = mfcac * c1o2+mfbac * (vvx+c1o2)+(mfaac+c1o36 * oMdrho) * (vx2+vvx) * c1o2;
//               mfaac = m0;
//               mfbac = m1;
//               mfcac = m2;
//               ///////////c////////////////////////////////////////////////////////////////////////
//               m0 = mfcbc * c1o2+mfbbc * (vvx-c1o2)+(mfabc+c1o9 * oMdrho) * (vx2-vvx) * c1o2;
//               m1 = -mfcbc-2. * mfbbc *  vvx+mfabc                  * (1.-vx2)-c1o9 * oMdrho * vx2;
//               m2 = mfcbc * c1o2+mfbbc * (vvx+c1o2)+(mfabc+c1o9 * oMdrho) * (vx2+vvx) * c1o2;
//               mfabc = m0;
//               mfbbc = m1;
//               mfcbc = m2;
//               ///////////c////////////////////////////////////////////////////////////////////////
//               m0 = mfccc * c1o2+mfbcc * (vvx-c1o2)+(mfacc+c1o36 * oMdrho) * (vx2-vvx) * c1o2;
//               m1 = -mfccc-2. * mfbcc *  vvx+mfacc                   * (1.-vx2)-c1o36 * oMdrho * vx2;
//               m2 = mfccc * c1o2+mfbcc * (vvx+c1o2)+(mfacc+c1o36 * oMdrho) * (vx2+vvx) * c1o2;
//               mfacc = m0;
//               mfbcc = m1;
//               mfccc = m2;
//
//               //////////////////////////////////////////////////////////////////////////
//               //proof correctness
//               //////////////////////////////////////////////////////////////////////////
//#ifdef  PROOF_CORRECTNESS
//               LBMReal rho_post = (mfaaa+mfaac+mfaca+mfcaa+mfacc+mfcac+mfccc+mfcca)
//                  +(mfaab+mfacb+mfcab+mfccb)+(mfaba+mfabc+mfcba+mfcbc)+(mfbaa+mfbac+mfbca+mfbcc)
//                  +(mfabb+mfcbb)+(mfbab+mfbcb)+(mfbba+mfbbc)+mfbbb;
//               //LBMReal dif = fabs(rho - rho_post);
//               LBMReal dif = rho-rho_post;
//#ifdef SINGLEPRECISION
//               if (dif>10.0E-7||dif<-10.0E-7)
//#else
//               if (dif>10.0E-15||dif<-10.0E-15)
//#endif
//               {
//                  UB_THROW(UbException(UB_EXARGS, "rho="+UbSystem::toString(rho)+", rho_post="+UbSystem::toString(rho_post)
//                     +" dif="+UbSystem::toString(dif)
//                     +" rho is not correct for node "+UbSystem::toString(x1)+","+UbSystem::toString(x2)+","+UbSystem::toString(x3)));
//                  //UBLOG(logERROR,"LBMKernelETD3Q27CCLB::collideAll(): rho is not correct for node "+UbSystem::toString(x1)+","+UbSystem::toString(x2)+","+UbSystem::toString(x3));
//                  //exit(EXIT_FAILURE);
//               }
//#endif
//               ////////////////////////////////////////////////////////////////////////////
//               ////write distribution
//               ////////////////////////////////////////////////////////////////////////////
//               (*this->localDistributions)(D3Q27System::ET_E, x1, x2, x3) = mfabb;
//               (*this->localDistributions)(D3Q27System::ET_N, x1, x2, x3) = mfbab;
//               (*this->localDistributions)(D3Q27System::ET_T, x1, x2, x3) = mfbba;
//               (*this->localDistributions)(D3Q27System::ET_NE, x1, x2, x3) = mfaab;
//               (*this->localDistributions)(D3Q27System::ET_NW, x1p, x2, x3) = mfcab;
//               (*this->localDistributions)(D3Q27System::ET_TE, x1, x2, x3) = mfaba;
//               (*this->localDistributions)(D3Q27System::ET_TW, x1p, x2, x3) = mfcba;
//               (*this->localDistributions)(D3Q27System::ET_TN, x1, x2, x3) = mfbaa;
//               (*this->localDistributions)(D3Q27System::ET_TS, x1, x2p, x3) = mfbca;
//               (*this->localDistributions)(D3Q27System::ET_TNE, x1, x2, x3) = mfaaa;
//               (*this->localDistributions)(D3Q27System::ET_TNW, x1p, x2, x3) = mfcaa;
//               (*this->localDistributions)(D3Q27System::ET_TSE, x1, x2p, x3) = mfaca;
//               (*this->localDistributions)(D3Q27System::ET_TSW, x1p, x2p, x3) = mfcca;
//
//               (*this->nonLocalDistributions)(D3Q27System::ET_W, x1p, x2, x3) = mfcbb;
//               (*this->nonLocalDistributions)(D3Q27System::ET_S, x1, x2p, x3) = mfbcb;
//               (*this->nonLocalDistributions)(D3Q27System::ET_B, x1, x2, x3p) = mfbbc;
//               (*this->nonLocalDistributions)(D3Q27System::ET_SW, x1p, x2p, x3) = mfccb;
//               (*this->nonLocalDistributions)(D3Q27System::ET_SE, x1, x2p, x3) = mfacb;
//               (*this->nonLocalDistributions)(D3Q27System::ET_BW, x1p, x2, x3p) = mfcbc;
//               (*this->nonLocalDistributions)(D3Q27System::ET_BE, x1, x2, x3p) = mfabc;
//               (*this->nonLocalDistributions)(D3Q27System::ET_BS, x1, x2p, x3p) = mfbcc;
//               (*this->nonLocalDistributions)(D3Q27System::ET_BN, x1, x2, x3p) = mfbac;
//               (*this->nonLocalDistributions)(D3Q27System::ET_BSW, x1p, x2p, x3p) = mfccc;
//               (*this->nonLocalDistributions)(D3Q27System::ET_BSE, x1, x2p, x3p) = mfacc;
//               (*this->nonLocalDistributions)(D3Q27System::ET_BNW, x1p, x2, x3p) = mfcac;
//               (*this->nonLocalDistributions)(D3Q27System::ET_BNE, x1, x2, x3p) = mfaac;
//
//               (*this->zeroDistributions)(x1, x2, x3) = mfbbb;
//               ////////////////////////////////////////////////////////////////////////////
//
//            }
//         }
//      }
//   }
//
//}

void InitDensityLBMKernel::init()
{
   DistributionArray3DPtr d(new D3Q27EsoTwist3DSplittedVector(nx1+2, nx2+2, nx3+2, -999.0));
   dataSet->setFdistributions(d);
   v.resize(3, nx1+2, nx2+2, nx3+2);

}


void InitDensityLBMKernel::collideAll()
{
   using namespace D3Q27System;

   localDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(dataSet->getFdistributions())->getLocalDistributions();
   nonLocalDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(dataSet->getFdistributions())->getNonLocalDistributions();
   zeroDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(dataSet->getFdistributions())->getZeroDistributions();

   BCArray3D<D3Q27BoundaryCondition>& bcArray = boost::dynamic_pointer_cast<D3Q27ETBCProcessor>(this->getBCProcessor())->getBCArray();
   D3Q27BoundaryConditionPtr bcPtr;
   LBMReal f[D3Q27System::ENDF+1];
   LBMReal feq[D3Q27System::ENDF+1];
   LBMReal drho, vx1, vx2, vx3;
   const int bcArrayMaxX1 = (int)bcArray.getNX1();
   const int bcArrayMaxX2 = (int)bcArray.getNX2();
   const int bcArrayMaxX3 = (int)bcArray.getNX3();

   int minX1 = ghostLayerWidth;
   int minX2 = ghostLayerWidth;
   int minX3 = ghostLayerWidth;
   int maxX1 = bcArrayMaxX1-ghostLayerWidth;
   int maxX2 = bcArrayMaxX2-ghostLayerWidth;
   int maxX3 = bcArrayMaxX3-ghostLayerWidth;

   //collFactor = 1.0/(1.0/2.0+1.0/sqrt(6.0));
   collFactor = 1.0;

   for (int x3 = minX3; x3<maxX3; x3++)
   {
      for (int x2 = minX2; x2<maxX2; x2++)
      {
         for (int x1 = minX1; x1<maxX1; x1++)
         {
            if (!bcArray.isSolid(x1, x2, x3)&&!bcArray.isUndefined(x1, x2, x3))
            {
               int x1p = x1+1;
               int x2p = x2+1;
               int x3p = x3+1;
               //////////////////////////////////////////////////////////////////////////
               //read distribution
               ////////////////////////////////////////////////////////////////////////////
               f[ZERO] = (*this->zeroDistributions)(x1, x2, x3);

               f[E] = (*this->localDistributions)(D3Q27System::ET_E, x1, x2, x3);
               f[N] = (*this->localDistributions)(D3Q27System::ET_N, x1, x2, x3);
               f[T] = (*this->localDistributions)(D3Q27System::ET_T, x1, x2, x3);
               f[NE] = (*this->localDistributions)(D3Q27System::ET_NE, x1, x2, x3);
               f[NW] = (*this->localDistributions)(D3Q27System::ET_NW, x1p, x2, x3);
               f[TE] = (*this->localDistributions)(D3Q27System::ET_TE, x1, x2, x3);
               f[TW] = (*this->localDistributions)(D3Q27System::ET_TW, x1p, x2, x3);
               f[TN] = (*this->localDistributions)(D3Q27System::ET_TN, x1, x2, x3);
               f[TS] = (*this->localDistributions)(D3Q27System::ET_TS, x1, x2p, x3);
               f[TNE] = (*this->localDistributions)(D3Q27System::ET_TNE, x1, x2, x3);
               f[TNW] = (*this->localDistributions)(D3Q27System::ET_TNW, x1p, x2, x3);
               f[TSE] = (*this->localDistributions)(D3Q27System::ET_TSE, x1, x2p, x3);
               f[TSW] = (*this->localDistributions)(D3Q27System::ET_TSW, x1p, x2p, x3);

               f[W] = (*this->nonLocalDistributions)(D3Q27System::ET_W, x1p, x2, x3);
               f[S] = (*this->nonLocalDistributions)(D3Q27System::ET_S, x1, x2p, x3);
               f[B] = (*this->nonLocalDistributions)(D3Q27System::ET_B, x1, x2, x3p);
               f[SW] = (*this->nonLocalDistributions)(D3Q27System::ET_SW, x1p, x2p, x3);
               f[SE] = (*this->nonLocalDistributions)(D3Q27System::ET_SE, x1, x2p, x3);
               f[BW] = (*this->nonLocalDistributions)(D3Q27System::ET_BW, x1p, x2, x3p);
               f[BE] = (*this->nonLocalDistributions)(D3Q27System::ET_BE, x1, x2, x3p);
               f[BS] = (*this->nonLocalDistributions)(D3Q27System::ET_BS, x1, x2p, x3p);
               f[BN] = (*this->nonLocalDistributions)(D3Q27System::ET_BN, x1, x2, x3p);
               f[BSW] = (*this->nonLocalDistributions)(D3Q27System::ET_BSW, x1p, x2p, x3p);
               f[BSE] = (*this->nonLocalDistributions)(D3Q27System::ET_BSE, x1, x2p, x3p);
               f[BNW] = (*this->nonLocalDistributions)(D3Q27System::ET_BNW, x1p, x2, x3p);
               f[BNE] = (*this->nonLocalDistributions)(D3Q27System::ET_BNE, x1, x2, x3p);
               //////////////////////////////////////////////////////////////////////////

               drho = ((f[TNE]+f[BSW])+(f[TSE]+f[BNW]))+((f[BSE]+f[TNW])+(f[TSW]+f[BNE]))
                  +(((f[NE]+f[SW])+(f[SE]+f[NW]))+((f[TE]+f[BW])+(f[BE]+f[TW]))
                     +((f[BN]+f[TS])+(f[TN]+f[BS])))+((f[E]+f[W])+(f[N]+f[S])
                        +(f[T]+f[B]))+f[ZERO];

               //vx1 = ((((f[TNE]-f[BSW])+(f[TSE]-f[BNW]))+((f[BSE]-f[TNW])+(f[BNE]-f[TSW])))+
               //   (((f[BE]-f[TW])+(f[TE]-f[BW]))+((f[SE]-f[NW])+(f[NE]-f[SW])))+
               //   (f[E]-f[W]));

               //vx2 = ((((f[TNE]-f[BSW])+(f[BNW]-f[TSE]))+((f[TNW]-f[BSE])+(f[BNE]-f[TSW])))+
               //   (((f[BN]-f[TS])+(f[TN]-f[BS]))+((f[NW]-f[SE])+(f[NE]-f[SW])))+
               //   (f[N]-f[S]));

               //vx3 = ((((f[TNE]-f[BSW])+(f[TSE]-f[BNW]))+((f[TNW]-f[BSE])+(f[TSW]-f[BNE])))+
               //   (((f[TS]-f[BN])+(f[TN]-f[BS]))+((f[TW]-f[BE])+(f[TE]-f[BW])))+
               //   (f[T]-f[B]));

               vx1 = v(0,x1,x2,x3);
               vx2 = v(1,x1,x2,x3);
               vx3 = v(2,x1,x2,x3);

               //LBMReal vvx = v(0,x1,x2,x3);
               //LBMReal vvy = v(1,x1,x2,x3);
               //LBMReal vvz = v(2,x1,x2,x3);

               //vx1 = vx1+(vvx-vx1);
               //vx2 = vx2+(vvy-vx2);
               //vx3 = vx3+(vvz-vx3);

               LBMReal cu_sq = 1.5*(vx1*vx1+vx2*vx2+vx3*vx3);

               feq[ZERO] = c8o27*(drho-cu_sq);
               feq[E] = c2o27*(drho+3.0*(vx1)+c9o2*(vx1)*(vx1)-cu_sq);
               feq[W] = c2o27*(drho+3.0*(-vx1)+c9o2*(-vx1)*(-vx1)-cu_sq);
               feq[N] = c2o27*(drho+3.0*(vx2)+c9o2*(vx2)*(vx2)-cu_sq);
               feq[S] = c2o27*(drho+3.0*(-vx2)+c9o2*(-vx2)*(-vx2)-cu_sq);
               feq[T] = c2o27*(drho+3.0*(vx3)+c9o2*(vx3)*(vx3)-cu_sq);
               feq[B] = c2o27*(drho+3.0*(-vx3)+c9o2*(-vx3)*(-vx3)-cu_sq);
               feq[NE] = c1o54*(drho+3.0*(vx1+vx2)+c9o2*(vx1+vx2)*(vx1+vx2)-cu_sq);
               feq[SW] = c1o54*(drho+3.0*(-vx1-vx2)+c9o2*(-vx1-vx2)*(-vx1-vx2)-cu_sq);
               feq[SE] = c1o54*(drho+3.0*(vx1-vx2)+c9o2*(vx1-vx2)*(vx1-vx2)-cu_sq);
               feq[NW] = c1o54*(drho+3.0*(-vx1+vx2)+c9o2*(-vx1+vx2)*(-vx1+vx2)-cu_sq);
               feq[TE] = c1o54*(drho+3.0*(vx1+vx3)+c9o2*(vx1+vx3)*(vx1+vx3)-cu_sq);
               feq[BW] = c1o54*(drho+3.0*(-vx1-vx3)+c9o2*(-vx1-vx3)*(-vx1-vx3)-cu_sq);
               feq[BE] = c1o54*(drho+3.0*(vx1-vx3)+c9o2*(vx1-vx3)*(vx1-vx3)-cu_sq);
               feq[TW] = c1o54*(drho+3.0*(-vx1+vx3)+c9o2*(-vx1+vx3)*(-vx1+vx3)-cu_sq);
               feq[TN] = c1o54*(drho+3.0*(vx2+vx3)+c9o2*(vx2+vx3)*(vx2+vx3)-cu_sq);
               feq[BS] = c1o54*(drho+3.0*(-vx2-vx3)+c9o2*(-vx2-vx3)*(-vx2-vx3)-cu_sq);
               feq[BN] = c1o54*(drho+3.0*(vx2-vx3)+c9o2*(vx2-vx3)*(vx2-vx3)-cu_sq);
               feq[TS] = c1o54*(drho+3.0*(-vx2+vx3)+c9o2*(-vx2+vx3)*(-vx2+vx3)-cu_sq);
               feq[TNE] = c1o216*(drho+3.0*(vx1+vx2+vx3)+c9o2*(vx1+vx2+vx3)*(vx1+vx2+vx3)-cu_sq);
               feq[BSW] = c1o216*(drho+3.0*(-vx1-vx2-vx3)+c9o2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cu_sq);
               feq[BNE] = c1o216*(drho+3.0*(vx1+vx2-vx3)+c9o2*(vx1+vx2-vx3)*(vx1+vx2-vx3)-cu_sq);
               feq[TSW] = c1o216*(drho+3.0*(-vx1-vx2+vx3)+c9o2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cu_sq);
               feq[TSE] = c1o216*(drho+3.0*(vx1-vx2+vx3)+c9o2*(vx1-vx2+vx3)*(vx1-vx2+vx3)-cu_sq);
               feq[BNW] = c1o216*(drho+3.0*(-vx1+vx2-vx3)+c9o2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cu_sq);
               feq[BSE] = c1o216*(drho+3.0*(vx1-vx2-vx3)+c9o2*(vx1-vx2-vx3)*(vx1-vx2-vx3)-cu_sq);
               feq[TNW] = c1o216*(drho+3.0*(-vx1+vx2+vx3)+c9o2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cu_sq);

               //Relaxation
               f[ZERO] += (feq[ZERO]-f[ZERO])*collFactor;
               f[E] += (feq[E]-f[E])*collFactor;
               f[W] += (feq[W]-f[W])*collFactor;
               f[N] += (feq[N]-f[N])*collFactor;
               f[S] += (feq[S]-f[S])*collFactor;
               f[T] += (feq[T]-f[T])*collFactor;
               f[B] += (feq[B]-f[B])*collFactor;
               f[NE] += (feq[NE]-f[NE])*collFactor;
               f[SW] += (feq[SW]-f[SW])*collFactor;
               f[SE] += (feq[SE]-f[SE])*collFactor;
               f[NW] += (feq[NW]-f[NW])*collFactor;
               f[TE] += (feq[TE]-f[TE])*collFactor;
               f[BW] += (feq[BW]-f[BW])*collFactor;
               f[BE] += (feq[BE]-f[BE])*collFactor;
               f[TW] += (feq[TW]-f[TW])*collFactor;
               f[TN] += (feq[TN]-f[TN])*collFactor;
               f[BS] += (feq[BS]-f[BS])*collFactor;
               f[BN] += (feq[BN]-f[BN])*collFactor;
               f[TS] += (feq[TS]-f[TS])*collFactor;

               f[TNE] += (feq[TNE]-f[TNE])*collFactor;
               f[BSW] += (feq[BSW]-f[BSW])*collFactor;
               f[BNE] += (feq[BNE]-f[BNE])*collFactor;
               f[TSW] += (feq[TSW]-f[TSW])*collFactor;
               f[TSE] += (feq[TSE]-f[TSE])*collFactor;
               f[BNW] += (feq[BNW]-f[BNW])*collFactor;
               f[BSE] += (feq[BSE]-f[BSE])*collFactor;
               f[TNW] += (feq[TNW]-f[TNW])*collFactor;

               //////////////////////////////////////////////////////////////////////////
#ifdef  PROOF_CORRECTNESS
               LBMReal rho_post = f[ZERO]+f[E]+f[W]+f[N]+f[S]+f[T]+f[B]
                  +f[NE]+f[SW]+f[SE]+f[NW]+f[TE]+f[BW]+f[BE]
                  +f[TW]+f[TN]+f[BS]+f[BN]+f[TS]+f[TNE]+f[TSW]
                  +f[TSE]+f[TNW]+f[BNE]+f[BSW]+f[BSE]+f[BNW];
               LBMReal dif = drho-rho_post;
#ifdef SINGLEPRECISION
               if (dif>10.0E-7||dif<-10.0E-7)
#else
               if (dif>10.0E-15||dif<-10.0E-15)
#endif
               {
                  UB_THROW(UbException(UB_EXARGS, "rho is not correct"));
               }
#endif
               //////////////////////////////////////////////////////////////////////////
               //write distribution
               //////////////////////////////////////////////////////////////////////////
               (*this->localDistributions)(D3Q27System::ET_E, x1, x2, x3) = f[D3Q27System::INV_E];
               (*this->localDistributions)(D3Q27System::ET_N, x1, x2, x3) = f[D3Q27System::INV_N];
               (*this->localDistributions)(D3Q27System::ET_T, x1, x2, x3) = f[D3Q27System::INV_T];
               (*this->localDistributions)(D3Q27System::ET_NE, x1, x2, x3) = f[D3Q27System::INV_NE];
               (*this->localDistributions)(D3Q27System::ET_NW, x1p, x2, x3) = f[D3Q27System::INV_NW];
               (*this->localDistributions)(D3Q27System::ET_TE, x1, x2, x3) = f[D3Q27System::INV_TE];
               (*this->localDistributions)(D3Q27System::ET_TW, x1p, x2, x3) = f[D3Q27System::INV_TW];
               (*this->localDistributions)(D3Q27System::ET_TN, x1, x2, x3) = f[D3Q27System::INV_TN];
               (*this->localDistributions)(D3Q27System::ET_TS, x1, x2p, x3) = f[D3Q27System::INV_TS];
               (*this->localDistributions)(D3Q27System::ET_TNE, x1, x2, x3) = f[D3Q27System::INV_TNE];
               (*this->localDistributions)(D3Q27System::ET_TNW, x1p, x2, x3) = f[D3Q27System::INV_TNW];
               (*this->localDistributions)(D3Q27System::ET_TSE, x1, x2p, x3) = f[D3Q27System::INV_TSE];
               (*this->localDistributions)(D3Q27System::ET_TSW, x1p, x2p, x3) = f[D3Q27System::INV_TSW];

               (*this->nonLocalDistributions)(D3Q27System::ET_W, x1p, x2, x3) = f[D3Q27System::INV_W];
               (*this->nonLocalDistributions)(D3Q27System::ET_S, x1, x2p, x3) = f[D3Q27System::INV_S];
               (*this->nonLocalDistributions)(D3Q27System::ET_B, x1, x2, x3p) = f[D3Q27System::INV_B];
               (*this->nonLocalDistributions)(D3Q27System::ET_SW, x1p, x2p, x3) = f[D3Q27System::INV_SW];
               (*this->nonLocalDistributions)(D3Q27System::ET_SE, x1, x2p, x3) = f[D3Q27System::INV_SE];
               (*this->nonLocalDistributions)(D3Q27System::ET_BW, x1p, x2, x3p) = f[D3Q27System::INV_BW];
               (*this->nonLocalDistributions)(D3Q27System::ET_BE, x1, x2, x3p) = f[D3Q27System::INV_BE];
               (*this->nonLocalDistributions)(D3Q27System::ET_BS, x1, x2p, x3p) = f[D3Q27System::INV_BS];
               (*this->nonLocalDistributions)(D3Q27System::ET_BN, x1, x2, x3p) = f[D3Q27System::INV_BN];
               (*this->nonLocalDistributions)(D3Q27System::ET_BSW, x1p, x2p, x3p) = f[D3Q27System::INV_BSW];
               (*this->nonLocalDistributions)(D3Q27System::ET_BSE, x1, x2p, x3p) = f[D3Q27System::INV_BSE];
               (*this->nonLocalDistributions)(D3Q27System::ET_BNW, x1p, x2, x3p) = f[D3Q27System::INV_BNW];
               (*this->nonLocalDistributions)(D3Q27System::ET_BNE, x1, x2, x3p) = f[D3Q27System::INV_BNE];

               (*this->zeroDistributions)(x1, x2, x3) = f[D3Q27System::ZERO];
               //////////////////////////////////////////////////////////////////////////


            }
         }
      }
   }
}