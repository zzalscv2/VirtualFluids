#include "CalculateTorqueCoProcessor.h"
#include "BCProcessor.h"

#include "Communicator.h"
#include "D3Q27Interactor.h"
#include "UbScheduler.h"
#include "Grid3D.h"
#include "BoundaryConditions.h"
#include "DataSet3D.h"
#include "Block3D.h"
#include "LBMKernel.h"
#include "BCArray3D.h"
#include "EsoTwist3D.h"
#include "DistributionArray3D.h"

CalculateTorqueCoProcessor::CalculateTorqueCoProcessor( SPtr<Grid3D> grid, SPtr<UbScheduler> s, const std::string &path, SPtr<Communicator> comm) : CoProcessor(grid, s), path(path), comm(comm), forceX1global(0), forceX2global(0), forceX3global(0)
{
   if (comm->getProcessID() == comm->getRoot())
   {
      std::ofstream ostr;
      std::string fname = path;
      ostr.open(fname.c_str(), std::ios_base::out | std::ios_base::app);
      if(!ostr)
      { 
         ostr.clear();
         std::string path = UbSystem::getPathFromString(fname);
         if(path.size()>0){ UbSystem::makeDirectory(path); ostr.open(fname.c_str(), std::ios_base::out | std::ios_base::app);}
         if(!ostr) throw UbException(UB_EXARGS,"couldn't open file "+fname);
      }
      ostr.width(12);
      ostr << "step" << "\t";
      ostr.width(10); 
      ostr << "Tx" << "\t";
      ostr.width(18); 
      ostr << "Ty" << "\t";
      ostr.width(18);
      ostr << "Tz" << std::endl;
      ostr.close();
   }
}
//////////////////////////////////////////////////////////////////////////
CalculateTorqueCoProcessor::~CalculateTorqueCoProcessor()
{

}
//////////////////////////////////////////////////////////////////////////
void CalculateTorqueCoProcessor::process( double step )
{
   if(scheduler->isDue(step) )
      collectData(step);

   UBLOG(logDEBUG3, "D3Q27ForcesCoProcessor::update:" << step);
}
//////////////////////////////////////////////////////////////////////////
void CalculateTorqueCoProcessor::collectData( double step )
{
   calculateForces();

   if (comm->getProcessID() == comm->getRoot())
   {
      int istep = static_cast<int>(step);
      std::ofstream ostr;
      std::string fname = path;
      ostr.open(fname.c_str(), std::ios_base::out | std::ios_base::app);
      if(!ostr)
      { 
         ostr.clear();
         std::string path = UbSystem::getPathFromString(fname);
         if(path.size()>0){ UbSystem::makeDirectory(path); ostr.open(fname.c_str(), std::ios_base::out | std::ios_base::app);}
         if(!ostr) throw UbException(UB_EXARGS,"couldn't open file "+fname);
      }

      ostr.width(12); 
      ostr.setf(std::ios::fixed); 
      ostr << istep << "\t";
      write(&ostr, forceX1global, (char*)"\t");
      write(&ostr, forceX2global, (char*)"\t");
      write(&ostr, forceX3global, (char*)"\t");
      ostr << std::endl;
      ostr.close();
   }
}
//////////////////////////////////////////////////////////////////////////
void CalculateTorqueCoProcessor::calculateForces()
{
   forceX1global = 0.0;
   forceX2global = 0.0;
   forceX3global = 0.0;
   int counter = 0;

   for(SPtr<D3Q27Interactor> interactor : interactors)
   {
      double x1Centre = interactor->getGbObject3D()->getX1Centroid();
      double x2Centre = interactor->getGbObject3D()->getX2Centroid();
      double x3Centre = interactor->getGbObject3D()->getX3Centroid();

      for(BcNodeIndicesMap::value_type t : interactor->getBcNodeIndicesMap())
      {
         double torqueX1 = 0.0;
         double torqueX2 = 0.0;
         double torqueX3 = 0.0;

         SPtr<Block3D> block = t.first;
         std::set< std::vector<int> >& transNodeIndicesSet = t.second;

         SPtr<ILBMKernel> kernel = block->getKernel();

         if (kernel->getCompressible())
         {
            calcMacrosFct = &D3Q27System::calcCompMacroscopicValues;
            compressibleFactor = 1.0;
         }
         else
         {
            calcMacrosFct = &D3Q27System::calcIncompMacroscopicValues;
            compressibleFactor = 0.0;
         }

         SPtr<BCArray3D> bcArray = kernel->getBCProcessor()->getBCArray();          
         SPtr<DistributionArray3D> distributions = kernel->getDataSet()->getFdistributions(); 
         distributions->swap();

         int ghostLayerWidth = kernel->getGhostLayerWidth();
         int minX1 = ghostLayerWidth;
         int maxX1 = (int)bcArray->getNX1() - 1 - ghostLayerWidth;
         int minX2 = ghostLayerWidth;
         int maxX2 = (int)bcArray->getNX2() - 1 - ghostLayerWidth;
         int minX3 = ghostLayerWidth;
         int maxX3 = (int)bcArray->getNX3() - 1 - ghostLayerWidth;

         for(std::vector<int> node : transNodeIndicesSet)
         {
            int x1 = node[0];
            int x2 = node[1];
            int x3 = node[2];

            Vector3D worldCoordinates = grid->getNodeCoordinates(block, x1, x2, x3);

            //without ghost nodes
            if (x1 < minX1 || x1 > maxX1 || x2 < minX2 || x2 > maxX2 ||x3 < minX3 || x3 > maxX3 ) continue;

            if(bcArray->isFluid(x1,x2,x3)) //es kann sein, dass der node von einem anderen interactor z.B. als solid gemarkt wurde!!!
            {
               SPtr<BoundaryConditions> bc = bcArray->getBC(x1,x2,x3);
               UbTupleDouble3 forceVec = getForces(x1,x2,x3,distributions,bc);
               torqueX1 += (worldCoordinates[1] - x2Centre) * val<3>(forceVec) - (worldCoordinates[2] - x3Centre) * val<2>(forceVec);
               torqueX2 += (worldCoordinates[2] - x3Centre) * val<1>(forceVec) - (worldCoordinates[0] - x1Centre) * val<3>(forceVec);
               torqueX3 += (worldCoordinates[0] - x1Centre) * val<2>(forceVec) - (worldCoordinates[1] - x2Centre) * val<1>(forceVec);
               //counter++;
               //UBLOG(logINFO, "x1="<<(worldCoordinates[1] - x2Centre)<<",x2=" << (worldCoordinates[2] - x3Centre)<< ",x3=" << (worldCoordinates[0] - x1Centre) <<" forceX3 = " << forceX3);
            }
         }
         //if we have got discretization with more level
         // deltaX is LBM deltaX and equal LBM deltaT 
         double deltaX = LBMSystem::getDeltaT(block->getLevel()); //grid->getDeltaT(block);
         double deltaXquadrat = deltaX*deltaX;
         torqueX1 *= deltaXquadrat;
         torqueX2 *= deltaXquadrat;
         torqueX3 *= deltaXquadrat;

         distributions->swap();

         forceX1global += torqueX1;
         forceX2global += torqueX2;
         forceX3global += torqueX3;

         //UBLOG(logINFO, "forceX3global = " << forceX3global);
      }
   }
   std::vector<double> values;
   std::vector<double> rvalues;
   values.push_back(forceX1global);
   values.push_back(forceX2global);
   values.push_back(forceX3global);

   rvalues = comm->gather(values);
   if (comm->getProcessID() == comm->getRoot())
   {
      forceX1global = 0.0;
      forceX2global = 0.0;
      forceX3global = 0.0;
      
      for (int i = 0; i < (int)rvalues.size(); i+=3)
      {
         forceX1global += rvalues[i];
         forceX2global += rvalues[i+1];
         forceX3global += rvalues[i+2];
      }
   }
}
//////////////////////////////////////////////////////////////////////////
UbTupleDouble3 CalculateTorqueCoProcessor::getForces(int x1, int x2, int x3,  SPtr<DistributionArray3D> distributions, SPtr<BoundaryConditions> bc)
{
   UbTupleDouble3 force(0.0,0.0,0.0);

   LBMReal fs[D3Q27System::ENDF + 1];
   distributions->getDistributionInv(fs, x1, x2, x3);
   LBMReal rho = 0.0, vx1 = 0.0, vx2 = 0.0, vx3 = 0.0, drho = 0.0;
   calcMacrosFct(fs, drho, vx1, vx2, vx3);
   rho = 1.0 + drho * compressibleFactor;
   
   if(bc)
   {
      //references to tuple "force"
      double& forceX1 = val<1>(force);
      double& forceX2 = val<2>(force);
      double& forceX3 = val<3>(force);
      double f,  fnbr;

      for(int fdir=D3Q27System::FSTARTDIR; fdir<=D3Q27System::FENDDIR; fdir++)
      {
         if(bc->hasNoSlipBoundaryFlag(fdir) || bc->hasVelocityBoundaryFlag(fdir))
         {
            const int invDir = D3Q27System::INVDIR[fdir];
            f = dynamicPointerCast<EsoTwist3D>(distributions)->getDistributionInvForDirection(x1, x2, x3, invDir);
            fnbr = dynamicPointerCast<EsoTwist3D>(distributions)->getDistributionInvForDirection(x1+D3Q27System::DX1[invDir], x2+D3Q27System::DX2[invDir], x3+D3Q27System::DX3[invDir], fdir);

            Vector3D boundaryVelocity;
            boundaryVelocity[0] = bc->getBoundaryVelocityX1();
            boundaryVelocity[1] = bc->getBoundaryVelocityX2();
            boundaryVelocity[2] = bc->getBoundaryVelocityX3();
            double correction[3] = { 0.0, 0.0, 0.0 };
            if (bc->hasVelocityBoundaryFlag(fdir))
            {
               const double forceTerm = f - fnbr;
               correction[0] = forceTerm * boundaryVelocity[0];
               correction[1] = forceTerm * boundaryVelocity[1];
               correction[2] = forceTerm * boundaryVelocity[2];
            }

            forceX1 += (f + fnbr) * D3Q27System::DX1[invDir] - 2.0 * D3Q27System::WEIGTH[invDir] * rho - correction[0];
            forceX2 += (f + fnbr) * D3Q27System::DX2[invDir] - 2.0 * D3Q27System::WEIGTH[invDir] * rho - correction[1];
            forceX3 += (f + fnbr) * D3Q27System::DX3[invDir] - 2.0 * D3Q27System::WEIGTH[invDir] * rho - correction[2];
         }
      }
   }
   
   return force;
}
//////////////////////////////////////////////////////////////////////////
void CalculateTorqueCoProcessor::addInteractor( SPtr<D3Q27Interactor> interactor )
{
   interactors.push_back(interactor);
}
//////////////////////////////////////////////////////////////////////////
void CalculateTorqueCoProcessor::write(std::ofstream *fileObject, double value, char *separator) 
{ 
   (*fileObject).width(12); 
   (*fileObject).precision(16); 
   (*fileObject).setf(std::ios::fixed); 
   (*fileObject) << value; 
   (*fileObject) << separator; 
} 

