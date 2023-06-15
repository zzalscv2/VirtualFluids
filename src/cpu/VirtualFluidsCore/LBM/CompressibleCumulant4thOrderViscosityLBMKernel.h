#ifndef CompressibleCumulant4thOrderViscosityLBMKernel_h__
#define CompressibleCumulant4thOrderViscosityLBMKernel_h__

#include "LBMKernel.h"
#include "BCSet.h"
#include "D3Q27System.h"
#include "basics/utilities/UbTiming.h"
#include "basics/container/CbArray4D.h"
#include "basics/container/CbArray3D.h"

//! \brief   compressible cumulant LBM kernel. 
//! \details CFD solver that use Cascaded Cumulant Lattice Boltzmann method for D3Q27 model
//! \author  K. Kutscher, M. Geier
class CompressibleCumulant4thOrderViscosityLBMKernel :  public LBMKernel
{
public:
   //! This option set relaxation parameter: NORMAL  
   enum Parameter{NORMAL, MAGIC};
public:
   CompressibleCumulant4thOrderViscosityLBMKernel();
   ~CompressibleCumulant4thOrderViscosityLBMKernel() override;
   void calculate(int step) override;
   SPtr<LBMKernel> clone() override;
   real getCalculationTime() override;
   //! The value should not be equal to a shear viscosity
   void setBulkViscosity(real value);
protected:
   virtual void initDataSet();
   real f[D3Q27System::ENDF+1];

   UbTimer timer;

   CbArray4D<real,IndexerX4X3X2X1>::CbArray4DPtr localDistributions;
   CbArray4D<real,IndexerX4X3X2X1>::CbArray4DPtr nonLocalDistributions;
   CbArray3D<real,IndexerX3X2X1>::CbArray3DPtr   zeroDistributions;

   mu::value_type muX1,muX2,muX3;
   mu::value_type muDeltaT;
   mu::value_type muNu;
   real forcingX1;
   real forcingX2;
   real forcingX3;
   
   // bulk viscosity
   real OxxPyyPzz; //omega2 (bulk viscosity)
   real bulkViscosity;

};
#endif // CompressibleCumulant4thOrderViscosityLBMKernel_h__


