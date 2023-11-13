#include "SlipBCStrategy.h"
#include "BoundaryConditions.h"
#include "DistributionArray3D.h"

SlipBCStrategy::SlipBCStrategy()
{
    BCStrategy::preCollision = false;
}
//////////////////////////////////////////////////////////////////////////
SlipBCStrategy::~SlipBCStrategy() = default;
//////////////////////////////////////////////////////////////////////////
SPtr<BCStrategy> SlipBCStrategy::clone()
{
    SPtr<BCStrategy> bc(new SlipBCStrategy());
    return bc;
}
//////////////////////////////////////////////////////////////////////////
void SlipBCStrategy::addDistributions(SPtr<DistributionArray3D> distributions) { this->distributions = distributions; }
//////////////////////////////////////////////////////////////////////////
void SlipBCStrategy::applyBC()
{
    using namespace vf::lbm::dir;

    real f[D3Q27System::ENDF + 1];
    real feq[D3Q27System::ENDF + 1];
    distributions->getPostCollisionDistribution(f, x1, x2, x3);
    real rho, vx1, vx2, vx3, drho;
    calcMacrosFct(f, drho, vx1, vx2, vx3);
    calcFeqFct(feq, drho, vx1, vx2, vx3);

    UbTupleFloat3 normale = bcPtr->getNormalVector();
    real amp            = vx1 * val<1>(normale) + vx2 * val<2>(normale) + vx3 * val<3>(normale);

    vx1 = vx1 - amp * val<1>(normale); // normale zeigt von struktur weg!
    vx2 = vx2 - amp * val<2>(normale); // normale zeigt von struktur weg!
    vx3 = vx3 - amp * val<3>(normale); // normale zeigt von struktur weg!

    rho = vf::basics::constant::c1o1 + drho * compressibleFactor;

   for (int fdir = D3Q27System::FSTARTDIR; fdir<=D3Q27System::FENDDIR; fdir++)
   {
      if (bcPtr->hasSlipBoundaryFlag(fdir))
      {
         //quadratic bounce back
         const int invDir = D3Q27System::INVDIR[fdir];
         real q = bcPtr->getQ(invDir);// m+m q=0 stabiler
         //vx3=0;
         real velocity = vf::basics::constant::c0o1;
         switch (invDir)
         {
         case dP00: velocity = (vf::basics::constant::c4o9*(+vx1)); break;      //(2/cs^2)(=6)*rho_0(=1 bei imkompr)*wi*u*ei mit cs=1/sqrt(3)
         case dM00: velocity = (vf::basics::constant::c4o9*(-vx1)); break;      //z.B. aus paper manfred MRT LB models in three dimensions (2002)   
         case d0P0: velocity = (vf::basics::constant::c4o9*(+vx2)); break;
         case d0M0: velocity = (vf::basics::constant::c4o9*(-vx2)); break;
         case d00P: velocity = (vf::basics::constant::c4o9*(+vx3)); break;
         case d00M: velocity = (vf::basics::constant::c4o9*(-vx3)); break;
         case dPP0: velocity = (vf::basics::constant::c1o9*(+vx1+vx2)); break;
         case dMM0: velocity = (vf::basics::constant::c1o9*(-vx1-vx2)); break;
         case dPM0: velocity = (vf::basics::constant::c1o9*(+vx1-vx2)); break;
         case dMP0: velocity = (vf::basics::constant::c1o9*(-vx1+vx2)); break;
         case dP0P: velocity = (vf::basics::constant::c1o9*(+vx1+vx3)); break;
         case dM0M: velocity = (vf::basics::constant::c1o9*(-vx1-vx3)); break;
         case dP0M: velocity = (vf::basics::constant::c1o9*(+vx1-vx3)); break;
         case dM0P: velocity = (vf::basics::constant::c1o9*(-vx1+vx3)); break;
         case d0PP: velocity = (vf::basics::constant::c1o9*(+vx2+vx3)); break;
         case d0MM: velocity = (vf::basics::constant::c1o9*(-vx2-vx3)); break;
         case d0PM: velocity = (vf::basics::constant::c1o9*(+vx2-vx3)); break;
         case d0MP: velocity = (vf::basics::constant::c1o9*(-vx2+vx3)); break;
         case dPPP: velocity = (vf::basics::constant::c1o36*(+vx1+vx2+vx3)); break;
         case dMMM: velocity = (vf::basics::constant::c1o36*(-vx1-vx2-vx3)); break;
         case dPPM: velocity = (vf::basics::constant::c1o36*(+vx1+vx2-vx3)); break;
         case dMMP: velocity = (vf::basics::constant::c1o36*(-vx1-vx2+vx3)); break;
         case dPMP: velocity = (vf::basics::constant::c1o36*(+vx1-vx2+vx3)); break;
         case dMPM: velocity = (vf::basics::constant::c1o36*(-vx1+vx2-vx3)); break;
         case dPMM: velocity = (vf::basics::constant::c1o36*(+vx1-vx2-vx3)); break;
         case dMPP: velocity = (vf::basics::constant::c1o36*(-vx1+vx2+vx3)); break;
         default: throw UbException(UB_EXARGS, "unknown error");
         }
         real fReturn = ((vf::basics::constant::c1o1-q)/(vf::basics::constant::c1o1+q))*((f[invDir]-feq[invDir])/(vf::basics::constant::c1o1-collFactor)+feq[invDir])+((q*(f[invDir]+f[fdir])-velocity*rho)/(vf::basics::constant::c1o1+q));
         distributions->setPostCollisionDistributionForDirection(fReturn, x1+D3Q27System::DX1[invDir], x2+D3Q27System::DX2[invDir], x3+D3Q27System::DX3[invDir], fdir);
      }
   }
}