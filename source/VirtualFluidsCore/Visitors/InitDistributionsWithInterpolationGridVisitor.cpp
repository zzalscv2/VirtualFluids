#include "InitDistributionsWithInterpolationGridVisitor.h"
#include <basics/utilities/UbFileInputASCII.h>
#include "LBMKernel.h"
#include "BCProcessor.h"
#include "Grid3DSystem.h"
#include <CbArray2D.h>
#include "D3Q27EsoTwist3DSplittedVector.h"


using namespace std;

InitDistributionsWithInterpolationGridVisitor::InitDistributionsWithInterpolationGridVisitor(Grid3DPtr oldGrid, InterpolationProcessorPtr iProcessor, LBMReal nu)
   : oldGrid(oldGrid), iProcessor(iProcessor), nu(nu)
{

}
//////////////////////////////////////////////////////////////////////////
InitDistributionsWithInterpolationGridVisitor::~InitDistributionsWithInterpolationGridVisitor()
{
}
//////////////////////////////////////////////////////////////////////////
void InitDistributionsWithInterpolationGridVisitor::visit(Grid3DPtr grid)
{
   newGrid = grid;
   int minInitLevel = newGrid->getCoarsestInitializedLevel();
   int maxInitLevel = newGrid->getFinestInitializedLevel();
   int newGridRank = newGrid->getRank();

   for (int l = minInitLevel; l<=maxInitLevel; l++)
   {
      int n = 0;
      vector<Block3DPtr> blockVector;
      newGrid->getBlocks(l, blockVector);
      vector<Block3DPtr> tBlockID;

      BOOST_FOREACH(Block3DPtr newBlock, blockVector)
      {
         if (!newBlock) 
            UB_THROW(UbException(UB_EXARGS, "block is not exist"));

         int newBlockRank = newBlock->getRank();

            Block3DPtr oldBlock = oldGrid->getBlock(newBlock->getX1(), newBlock->getX2(), newBlock->getX3(), newBlock->getLevel());
            if (oldBlock)
            {
               int oldBlockRank = oldBlock->getRank();
               if (oldBlockRank == newBlockRank && oldBlock->isActive() && newBlockRank == newGridRank && newBlock->isActive())
               {
                  copyLocalBlock(oldBlock, newBlock);
               }
               else
               {
                  copyRemoteBlock(oldBlock, newBlock);
               }
            }
            else
            {
               int newlevel = newBlock->getLevel();
               UbTupleDouble3 coord = newGrid->getNodeCoordinates(newBlock, 1, 1, 1);
               UbTupleInt3 oldGridBlockIndexes = oldGrid->getBlockIndexes(val<1>(coord), val<2>(coord), val<3>(coord), newlevel-1);
               Block3DPtr oldBlock = oldGrid->getBlock(val<1>(oldGridBlockIndexes), val<2>(oldGridBlockIndexes), val<3>(oldGridBlockIndexes), newlevel-1);

               if (oldBlock)
               {
                  int oldBlockRank = oldBlock->getRank();

                  if (oldBlockRank == newBlockRank && oldBlock->isActive() && newBlockRank == newGridRank && newBlock->isActive())
                  {
                     interpolateLocalBlock(oldBlock, newBlock);
                  }
                  else
                  {
                     interpolateRemoteBlock(oldBlock, newBlock);
                  }
               }
            }
      }
   }
}
//////////////////////////////////////////////////////////////////////////
void InitDistributionsWithInterpolationGridVisitor::copyLocalBlock(Block3DPtr oldBlock, Block3DPtr newBlock)
{
   LBMKernelPtr oldKernel = oldBlock->getKernel();
   if (!oldKernel)
      throw UbException(UB_EXARGS, "The LBM kernel isn't exist in block: "+oldBlock->toString());
   EsoTwist3DPtr oldDistributions = boost::dynamic_pointer_cast<EsoTwist3D>(oldKernel->getDataSet()->getFdistributions());

   LBMKernelPtr kernel = newBlock->getKernel();
   if (!kernel)
      throw UbException(UB_EXARGS, "The LBM kernel isn't exist in new block: "+newBlock->toString());
   kernel->getDataSet()->setFdistributions(oldDistributions);
}
//////////////////////////////////////////////////////////////////////////
void InitDistributionsWithInterpolationGridVisitor::interpolateLocalBlock(Block3DPtr oldBlock, Block3DPtr newBlock)
{
   D3Q27ICell icellC;
   D3Q27ICell icellF;

   LBMReal omegaC = LBMSystem::calcCollisionFactor(nu, oldBlock->getLevel());
   LBMReal omegaF =LBMSystem::calcCollisionFactor(nu, newBlock->getLevel());

   iProcessor->setOmegas(omegaC, omegaF);

   LBMKernelPtr oldKernel = oldBlock->getKernel();
   if (!oldKernel)
      throw UbException(UB_EXARGS, "The LBM kernel isn't exist in old block: "+oldBlock->toString());

   EsoTwist3DPtr oldDistributions = boost::dynamic_pointer_cast<EsoTwist3D>(oldKernel->getDataSet()->getFdistributions());

   LBMKernelPtr newKernel = newBlock->getKernel();
   if (!newKernel)
      throw UbException(UB_EXARGS, "The LBM kernel isn't exist in new block: "+newBlock->toString());

   EsoTwist3DPtr newDistributions = boost::dynamic_pointer_cast<EsoTwist3D>(newKernel->getDataSet()->getFdistributions());

   int minX1 = 0;
   int minX2 = 0;
   int minX3 = 0;

   int maxX1 = (int)newDistributions->getNX1()-1;
   int maxX2 = (int)newDistributions->getNX2()-1;
   int maxX3 = (int)newDistributions->getNX3()-1;

   for (int ix3 = minX3; ix3 < maxX3; ix3+=2)
      for (int ix2 = minX2; ix2 < maxX2; ix2+=2)
         for (int ix1 = minX1; ix1 < maxX1; ix1+=2)
         {
            UbTupleDouble3 coord = newGrid->getNodeCoordinates(newBlock, ix1, ix2, ix3);
            UbTupleInt3 oldGridIndexMin = oldGrid->getNodeIndexes(oldBlock, val<1>(coord), val<2>(coord), val<3>(coord));
            iProcessor->readICell(oldDistributions, icellC, val<1>(oldGridIndexMin), val<2>(oldGridIndexMin), val<3>(oldGridIndexMin));
            iProcessor->interpolateCoarseToFine(icellC, icellF);

            iProcessor->writeICell(newDistributions, icellF, ix1, ix2, ix3);
            iProcessor->writeICellInv(newDistributions, icellF, ix1, ix2, ix3);
         }
}
//////////////////////////////////////////////////////////////////////////
void InitDistributionsWithInterpolationGridVisitor::copyRemoteBlock(Block3DPtr oldBlock, Block3DPtr newBlock)
{
   int newGridRank = newGrid->getRank();
   int oldBlockRank = oldBlock->getRank();
   int newBlockRank = newBlock->getRank();

   if (oldBlockRank == newGridRank && oldBlock->isActive())
   {
      LBMKernelPtr oldKernel = oldBlock->getKernel();
      if (!oldKernel)
         throw UbException(UB_EXARGS, "The LBM kernel isn't exist in block: "+oldBlock->toString());
      EsoTwist3DPtr oldDistributions = boost::dynamic_pointer_cast<EsoTwist3D>(oldKernel->getDataSet()->getFdistributions());

      CbArray4D<LBMReal, IndexerX4X3X2X1>::CbArray4DPtr localDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(oldDistributions)->getLocalDistributions();
      CbArray4D<LBMReal, IndexerX4X3X2X1>::CbArray4DPtr nonLocalDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(oldDistributions)->getNonLocalDistributions();
      CbArray3D<LBMReal, IndexerX3X2X1>::CbArray3DPtr   zeroDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(oldDistributions)->getZeroDistributions();
      
      MPI_Send(localDistributions->getStartAdressOfSortedArray(0, 0, 0, 0), localDistributions->getDataVector().size(), MPI_DOUBLE, newBlockRank, 0, MPI_COMM_WORLD);
      MPI_Send(nonLocalDistributions->getStartAdressOfSortedArray(0, 0, 0, 0), nonLocalDistributions->getDataVector().size(), MPI_DOUBLE, newBlockRank, 0, MPI_COMM_WORLD);
      MPI_Send(zeroDistributions->getStartAdressOfSortedArray(0, 0, 0), zeroDistributions->getDataVector().size(), MPI_DOUBLE, newBlockRank, 0, MPI_COMM_WORLD);
   }
   else if(newBlockRank == newGridRank && newBlock->isActive())
   {
      LBMKernelPtr newKernel = newBlock->getKernel();
      if (!newKernel)
         throw UbException(UB_EXARGS, "The LBM kernel isn't exist in new block: "+newBlock->toString()+UbSystem::toString(newGridRank));

      EsoTwist3DPtr newDistributions = boost::dynamic_pointer_cast<EsoTwist3D>(newKernel->getDataSet()->getFdistributions());

      CbArray4D<LBMReal, IndexerX4X3X2X1>::CbArray4DPtr localDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(newDistributions)->getLocalDistributions();
      CbArray4D<LBMReal, IndexerX4X3X2X1>::CbArray4DPtr nonLocalDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(newDistributions)->getNonLocalDistributions();
      CbArray3D<LBMReal, IndexerX3X2X1>::CbArray3DPtr   zeroDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(newDistributions)->getZeroDistributions();

      MPI_Recv(localDistributions->getStartAdressOfSortedArray(0, 0, 0, 0), localDistributions->getDataVector().size(), MPI_DOUBLE, oldBlockRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(nonLocalDistributions->getStartAdressOfSortedArray(0, 0, 0, 0), nonLocalDistributions->getDataVector().size(), MPI_DOUBLE, oldBlockRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(zeroDistributions->getStartAdressOfSortedArray(0, 0, 0), zeroDistributions->getDataVector().size(), MPI_DOUBLE, oldBlockRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   }
}
//////////////////////////////////////////////////////////////////////////
void InitDistributionsWithInterpolationGridVisitor::interpolateRemoteBlock(Block3DPtr oldBlock, Block3DPtr newBlock)
{
   int newGridRank = newGrid->getRank();
   int oldBlockRank = oldBlock->getRank();
   int newBlockRank = newBlock->getRank();

   if (oldBlockRank == newGridRank)
   {
      LBMKernelPtr oldKernel = oldBlock->getKernel();
      if (!oldKernel)
         throw UbException(UB_EXARGS, "The LBM kernel isn't exist in block: "+oldBlock->toString());
      EsoTwist3DPtr oldDistributions = boost::dynamic_pointer_cast<EsoTwist3D>(oldKernel->getDataSet()->getFdistributions());

      CbArray4D<LBMReal, IndexerX4X3X2X1>::CbArray4DPtr localDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(oldDistributions)->getLocalDistributions();
      CbArray4D<LBMReal, IndexerX4X3X2X1>::CbArray4DPtr nonLocalDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(oldDistributions)->getNonLocalDistributions();
      CbArray3D<LBMReal, IndexerX3X2X1>::CbArray3DPtr   zeroDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(oldDistributions)->getZeroDistributions();

      MPI_Send(localDistributions->getStartAdressOfSortedArray(0, 0, 0, 0), localDistributions->getDataVector().size(), MPI_DOUBLE, newBlockRank, 0, MPI_COMM_WORLD);
      MPI_Send(nonLocalDistributions->getStartAdressOfSortedArray(0, 0, 0, 0), nonLocalDistributions->getDataVector().size(), MPI_DOUBLE, newBlockRank, 0, MPI_COMM_WORLD);
      MPI_Send(zeroDistributions->getStartAdressOfSortedArray(0, 0, 0), zeroDistributions->getDataVector().size(), MPI_DOUBLE, newBlockRank, 0, MPI_COMM_WORLD);

      BCArray3D& bcArrayOldBlock = oldBlock->getKernel()->getBCProcessor()->getBCArray();
      std::vector< int >& bcDataVector = bcArrayOldBlock.getBcindexmatrixDataVector(); 
      MPI_Send(&bcDataVector[0], bcDataVector.size(), MPI_INT, newBlockRank, 0, MPI_COMM_WORLD);
   }
   else if(newBlockRank == newGridRank && newBlock->isActive())
   {
      D3Q27ICell icellC;
      D3Q27ICell icellF;
      LBMReal xoff, yoff, zoff;

      LBMReal omegaC = LBMSystem::calcCollisionFactor(nu, oldBlock->getLevel());
      LBMReal omegaF =LBMSystem::calcCollisionFactor(nu, newBlock->getLevel());

      iProcessor->setOmegas(omegaC, omegaF);

      LBMKernelPtr newKernel = newBlock->getKernel();
      if (!newKernel)
         throw UbException(UB_EXARGS, "The LBM kernel isn't exist in new block: "+newBlock->toString());

      EsoTwist3DPtr newDistributions = boost::dynamic_pointer_cast<EsoTwist3D>(newKernel->getDataSet()->getFdistributions());

      int minX1 = 0;
      int minX2 = 0;
      int minX3 = 0;

      int maxX1 = (int)newDistributions->getNX1()-1;
      int maxX2 = (int)newDistributions->getNX2()-1;
      int maxX3 = (int)newDistributions->getNX3()-1;

      int bMaxX1 = (int)newDistributions->getNX1();
      int bMaxX2 = (int)newDistributions->getNX2();
      int bMaxX3 = (int)newDistributions->getNX3();

      EsoTwist3DPtr oldDistributions(new D3Q27EsoTwist3DSplittedVector(bMaxX1, bMaxX2, bMaxX3, 0));

      CbArray4D<LBMReal, IndexerX4X3X2X1>::CbArray4DPtr localDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(oldDistributions)->getLocalDistributions();
      CbArray4D<LBMReal, IndexerX4X3X2X1>::CbArray4DPtr nonLocalDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(oldDistributions)->getNonLocalDistributions();
      CbArray3D<LBMReal, IndexerX3X2X1>::CbArray3DPtr   zeroDistributions = boost::dynamic_pointer_cast<D3Q27EsoTwist3DSplittedVector>(oldDistributions)->getZeroDistributions();

      MPI_Recv(localDistributions->getStartAdressOfSortedArray(0, 0, 0, 0), localDistributions->getDataVector().size(), MPI_DOUBLE, oldBlockRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(nonLocalDistributions->getStartAdressOfSortedArray(0, 0, 0, 0), nonLocalDistributions->getDataVector().size(), MPI_DOUBLE, oldBlockRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(zeroDistributions->getStartAdressOfSortedArray(0, 0, 0), zeroDistributions->getDataVector().size(), MPI_DOUBLE, oldBlockRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      BCArray3D bcArrayOldBlock(bMaxX1, bMaxX2, bMaxX3, BCArray3D::FLUID);
      std::vector< int >& bcDataVector = bcArrayOldBlock.getBcindexmatrixDataVector(); 
      MPI_Recv(&bcDataVector[0], bcDataVector.size(), MPI_INT, oldBlockRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int ix3 = minX3; ix3 < maxX3; ix3+=2)
         for (int ix2 = minX2; ix2 < maxX2; ix2+=2)
            for (int ix1 = minX1; ix1 < maxX1; ix1+=2)
            {
               UbTupleDouble3 coord = newGrid->getNodeCoordinates(newBlock, ix1, ix2, ix3);
               UbTupleInt3 oldGridIndexMin = oldGrid->getNodeIndexes(oldBlock, val<1>(coord), val<2>(coord), val<3>(coord));
               //iProcessor->readICell(oldDistributions, icellC, val<1>(oldGridIndexMin), val<2>(oldGridIndexMin), val<3>(oldGridIndexMin));
               //iProcessor->interpolateCoarseToFine(icellC, icellF);

               int howManySolids= iProcessor->iCellHowManySolids(bcArrayOldBlock, val<1>(oldGridIndexMin), val<2>(oldGridIndexMin), val<3>(oldGridIndexMin));

               if (howManySolids == 0 || howManySolids == 8)
               {
                  iProcessor->readICell(oldDistributions, icellC, val<1>(oldGridIndexMin), val<2>(oldGridIndexMin), val<3>(oldGridIndexMin));
                  //xoff=0.0;
                  //yoff=0.0;
                  //zoff=0.0;
                  iProcessor->interpolateCoarseToFine(icellC, icellF);
               }
               else
               {
                  if (iProcessor->findNeighborICell(bcArrayOldBlock, oldDistributions, icellC, bMaxX1, bMaxX2, bMaxX3, 
                                                     val<1>(oldGridIndexMin), val<2>(oldGridIndexMin), val<3>(oldGridIndexMin), xoff, yoff, zoff))
                  {
                     //std::string err = "For "+oldBlock->toString()+
                     //   " x1="+UbSystem::toString(val<1>(oldGridIndexMin))+
                     //   ", x2=" + UbSystem::toString(val<2>(oldGridIndexMin))+
                     //   ", x3=" + UbSystem::toString(val<3>(oldGridIndexMin))+
                     //   " interpolation is not implemented for other direction"+
                     //   " by using in: "+(std::string)typeid(*this).name()+
                     //   " or maybe you have a solid on the block boundary";
                     //UB_THROW(UbException(UB_EXARGS, err));
                     iProcessor->interpolateCoarseToFine(icellC, icellF, xoff, yoff, zoff);
                  }
                  else
                  {
                     for (int i=0; i<27; i++)
                     {
                        icellF.BSW[i]=0.0;
                        icellF.BSE[i]=0.0;
                        icellF.BNW[i]=0.0;
                        icellF.BNE[i]=0.0;
                        icellF.TSW[i]=0.0;
                        icellF.TSE[i]=0.0;
                        icellF.TNW[i]=0.0;
                        icellF.TNE[i]=0.0;
                     }
                     //                     std::string err = "For "+oldBlock->toString()+
                     //   " x1="+UbSystem::toString(val<1>(oldGridIndexMin))+
                     //   ", x2=" + UbSystem::toString(val<2>(oldGridIndexMin))+
                     //   ", x3=" + UbSystem::toString(val<3>(oldGridIndexMin))+
                     //   " interpolation is not implemented for other direction"+
                     //   " by using in: "+(std::string)typeid(*this).name()+
                     //   " or maybe you have a solid on the block boundary";
                     ////UB_THROW(UbException(UB_EXARGS, err));
                     //                     UBLOG(logINFO, err);
                  }
               }

               

               iProcessor->writeICell(newDistributions, icellF, ix1, ix2, ix3);
               iProcessor->writeICellInv(newDistributions, icellF, ix1, ix2, ix3);
            }
   }
}
//////////////////////////////////////////////////////////////////////////
