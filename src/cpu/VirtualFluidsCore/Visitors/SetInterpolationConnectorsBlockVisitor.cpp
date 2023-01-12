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
//! \file SetInterpolationConnectorsBlockVisitor.cpp
//! \ingroup Visitors
//! \author Konstantin Kutscher
//=======================================================================================

#include "SetInterpolationConnectorsBlockVisitor.h"
#include "CoarseToFineVectorConnector.h"
#include "FineToCoarseVectorConnector.h"
#include "TwoDistributionsFullDirectConnector.h"
#include "TwoDistributionsFullVectorConnector.h"
#include "D3Q27System.h"
#include <basics/transmitter/TbTransmitterLocal.h>

#include <mpi/Communicator.h>
#include "InterpolationProcessor.h"

SetInterpolationConnectorsBlockVisitor::SetInterpolationConnectorsBlockVisitor(std::shared_ptr<vf::mpi::Communicator> comm, LBMReal nue, SPtr<InterpolationProcessor> iProcessor) :
Block3DVisitor(0, D3Q27System::MAXLEVEL), 
	comm(comm),
	nue(nue),
	iProcessor(iProcessor)
{
}
//////////////////////////////////////////////////////////////////////////
SetInterpolationConnectorsBlockVisitor::~SetInterpolationConnectorsBlockVisitor(void)
{
}
//////////////////////////////////////////////////////////////////////////
void SetInterpolationConnectorsBlockVisitor::visit(SPtr<Grid3D> grid, SPtr<Block3D> block)
{
	if(!block) return;

	UBLOG(logDEBUG5, "SetInterpolationConnectorsBlockVisitor::visit() - start");
    UBLOG(logDEBUG5, block->toString());

	gridRank = comm->getProcessID();
	grid->setRank(gridRank);

	if(grid->getFinestInitializedLevel() > grid->getCoarsestInitializedLevel())
		setInterpolationConnectors(grid, block);

	UBLOG(logDEBUG5, "SetInterpolationConnectorsBlockVisitor::visit() - end");
}
//////////////////////////////////////////////////////////////////////////
void SetInterpolationConnectorsBlockVisitor::setInterpolationConnectors(SPtr<Grid3D> grid, SPtr<Block3D> block)
{
	using namespace vf::lbm::dir;

   UBLOG(logDEBUG5, "SetInterpolationConnectorsBlockVisitor::setInterpolationConnectors() - start");

	//search for all blocks with different ranks
	if (block->hasInterpolationFlagCF() && block->isActive())
	{
		int fbx1 = block->getX1() << 1;
		int fbx2 = block->getX2() << 1;
		int fbx3 = block->getX3() << 1;
		int level = block->getLevel() + 1;

		if( block->hasInterpolationFlagCF(DIR_P00))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1+1,fbx2,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockNW = grid->getBlock(fbx1+1,fbx2,fbx3+1,level);
			SPtr<Block3D> fblockNE = grid->getBlock(fbx1+1,fbx2+1,fbx3+1,level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_P00);
		}
		if( block->hasInterpolationFlagCF(DIR_M00))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1,fbx2,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockNW = grid->getBlock(fbx1,fbx2,fbx3+1,level);
			SPtr<Block3D> fblockNE = grid->getBlock(fbx1,fbx2+1,fbx3+1,level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_M00);
		}
		if( block->hasInterpolationFlagCF(DIR_0P0))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockNW = grid->getBlock(fbx1,fbx2+1,fbx3+1,level);
			SPtr<Block3D> fblockNE = grid->getBlock(fbx1+1,fbx2+1,fbx3+1,level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_0P0);
		}
		if( block->hasInterpolationFlagCF(DIR_0M0))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1,fbx2,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2,fbx3,level);
			SPtr<Block3D> fblockNW = grid->getBlock(fbx1,fbx2,fbx3+1,level);
			SPtr<Block3D> fblockNE = grid->getBlock(fbx1+1,fbx2,fbx3+1,level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_0M0);
		}
		if( block->hasInterpolationFlagCF(DIR_00P))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1,fbx2,fbx3+1,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2,fbx3+1,level);
			SPtr<Block3D> fblockNW = grid->getBlock(fbx1,fbx2+1,fbx3+1,level);
			SPtr<Block3D> fblockNE = grid->getBlock(fbx1+1,fbx2+1,fbx3+1,level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_00P);
		}
		if( block->hasInterpolationFlagCF(DIR_00M))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1,fbx2,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2,fbx3,level);
			SPtr<Block3D> fblockNW = grid->getBlock(fbx1,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockNE = grid->getBlock(fbx1+1,fbx2+1,fbx3,level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_00M);
		}

		//////NE-NW-SE-SW
		if( block->hasInterpolationFlagCF(DIR_PP0)&&!block->hasInterpolationFlagCF(DIR_0P0) && !block->hasInterpolationFlagCF(DIR_P00))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1+1,fbx2+1,fbx3+0,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2+1,fbx3+1,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+1, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+1, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_PP0);
		}
		if( block->hasInterpolationFlagCF(DIR_MM0)&& !block->hasInterpolationFlagCF(DIR_M00) && !block->hasInterpolationFlagCF(DIR_0M0))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1,fbx2,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1,fbx2,fbx3+1,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1, fbx2, fbx3+1, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1, fbx2, fbx3+1, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_MM0);
		}
		if( block->hasInterpolationFlagCF(DIR_PM0)&& !block->hasInterpolationFlagCF(DIR_P00) && !block->hasInterpolationFlagCF(DIR_0M0))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1+1,fbx2,fbx3+0,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2,fbx3+1,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+1, fbx2, fbx3+1, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2, fbx3+1, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_PM0);
		}
		if( block->hasInterpolationFlagCF(DIR_MP0)&& !block->hasInterpolationFlagCF(DIR_0P0) && !block->hasInterpolationFlagCF(DIR_M00))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1,fbx2+1,fbx3+1,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1, fbx2+1, fbx3+1, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1, fbx2+1, fbx3+1, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_MP0);
		}

		/////////TE-BW-BE-TW 1-0
		if( block->hasInterpolationFlagCF(DIR_P0P)&& !block->hasInterpolationFlagCF(DIR_P00) && !block->hasInterpolationFlagCF(DIR_00P))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1+1,fbx2+0,fbx3+1,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2+1,fbx3+1,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+1, fbx2+0, fbx3+1, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+1, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_P0P);
		}
		if( block->hasInterpolationFlagCF(DIR_M0M)&& !block->hasInterpolationFlagCF(DIR_M00) && !block->hasInterpolationFlagCF(DIR_00M))
		{

			SPtr<Block3D> fblockSW = grid->getBlock(fbx1,fbx2+0,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1, fbx2+0, fbx3, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1, fbx2+1, fbx3, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_M0M);
		}
		if( block->hasInterpolationFlagCF(DIR_P0M)&& !block->hasInterpolationFlagCF(DIR_P00) && !block->hasInterpolationFlagCF(DIR_00M))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1+1,fbx2+0,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+1, fbx2+0, fbx3, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2+1, fbx3, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_P0M);
		}
		if( block->hasInterpolationFlagCF(DIR_M0P)&& !block->hasInterpolationFlagCF(DIR_M00) && !block->hasInterpolationFlagCF(DIR_00P))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1,fbx2+0,fbx3+1,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1,fbx2+1,fbx3+1,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1, fbx2+0, fbx3+1, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1, fbx2+1, fbx3+1, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_M0P);
		}

		//////TN-BS-BN-TS
		if( block->hasInterpolationFlagCF(DIR_0PP)&& !block->hasInterpolationFlagCF(DIR_0P0) && !block->hasInterpolationFlagCF(DIR_00P))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1+0,fbx2+1,fbx3+1,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2+1,fbx3+1,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+0, fbx2+1, fbx3+1, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+1, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_0PP);
		}
		if( block->hasInterpolationFlagCF(DIR_0MM)&& !block->hasInterpolationFlagCF(DIR_0M0) && !block->hasInterpolationFlagCF(DIR_00M))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1+0,fbx2,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2,fbx3,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+0, fbx2, fbx3, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2, fbx3, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_0MM);
		}
		if( block->hasInterpolationFlagCF(DIR_0PM)&& !block->hasInterpolationFlagCF(DIR_0P0) && !block->hasInterpolationFlagCF(DIR_00M))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1+0,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2+1,fbx3,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+0, fbx2+1, fbx3, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2+1, fbx3, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_0PM);
		}
		if( block->hasInterpolationFlagCF(DIR_0MP)&& !block->hasInterpolationFlagCF(DIR_0M0) && !block->hasInterpolationFlagCF(DIR_00P))
		{
			SPtr<Block3D> fblockSW = grid->getBlock(fbx1+0,fbx2,fbx3+1,level);
			SPtr<Block3D> fblockSE = grid->getBlock(fbx1+1,fbx2,fbx3+1,level);
			SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+0, fbx2, fbx3+1, level);
			SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2, fbx3+1, level);

			setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_0MP);
		}




      //////corners
      if (block->hasInterpolationFlagCF(DIR_PPP)&&!block->hasInterpolationFlagCF(DIR_P0P)&&!block->hasInterpolationFlagCF(DIR_0PP)&&!block->hasInterpolationFlagCF(DIR_PP0)&&!block->hasInterpolationFlagCF(DIR_00P)&&!block->hasInterpolationFlagCF(DIR_0P0) && !block->hasInterpolationFlagCF(DIR_P00))
      {
         SPtr<Block3D> fblockSW = grid->getBlock(fbx1+1, fbx2+1, fbx3+1, level);
         SPtr<Block3D> fblockSE;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+0, level);
         SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+1, level);
         SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+1, level);

         setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_PPP);
      }
      if (block->hasInterpolationFlagCF(DIR_MMP)&&!block->hasInterpolationFlagCF(DIR_M0P)&&!block->hasInterpolationFlagCF(DIR_0MP)&& !block->hasInterpolationFlagCF(DIR_MM0)&& !block->hasInterpolationFlagCF(DIR_00P)&& !block->hasInterpolationFlagCF(DIR_M00) && !block->hasInterpolationFlagCF(DIR_0M0))
      {
         SPtr<Block3D> fblockSW = grid->getBlock(fbx1, fbx2, fbx3+1, level);
         SPtr<Block3D> fblockSE;// = grid->getBlock(fbx1, fbx2, fbx3, level);
         SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1, fbx2, fbx3+1, level);
         SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1, fbx2, fbx3+1, level);

         setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_MMP);
      }
      if (block->hasInterpolationFlagCF(DIR_PMP)&&!block->hasInterpolationFlagCF(DIR_P0P)&&!block->hasInterpolationFlagCF(DIR_0MP)&& !block->hasInterpolationFlagCF(DIR_PM0)&& !block->hasInterpolationFlagCF(DIR_00P)&& !block->hasInterpolationFlagCF(DIR_P00) && !block->hasInterpolationFlagCF(DIR_0M0))
      {
         SPtr<Block3D> fblockSW = grid->getBlock(fbx1+1, fbx2, fbx3+1, level);
         SPtr<Block3D> fblockSE;// = grid->getBlock(fbx1+1, fbx2, fbx3+0, level);
         SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+1, fbx2, fbx3+1, level);
         SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2, fbx3+1, level);

         setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_PMP);
      }
      if (block->hasInterpolationFlagCF(DIR_MPP)&&!block->hasInterpolationFlagCF(DIR_M0P)&&!block->hasInterpolationFlagCF(DIR_0PP)&& !block->hasInterpolationFlagCF(DIR_MP0)&& !block->hasInterpolationFlagCF(DIR_00P)&& !block->hasInterpolationFlagCF(DIR_0P0) && !block->hasInterpolationFlagCF(DIR_M00))
      {
         SPtr<Block3D> fblockSW = grid->getBlock(fbx1, fbx2+1, fbx3+1, level);
         SPtr<Block3D> fblockSE;// = grid->getBlock(fbx1, fbx2+1, fbx3, level);
         SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1, fbx2+1, fbx3+1, level);
         SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1, fbx2+1, fbx3+1, level);

         setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_MPP);
      }
      if (block->hasInterpolationFlagCF(DIR_PPM)&&!block->hasInterpolationFlagCF(DIR_P0M)&&!block->hasInterpolationFlagCF(DIR_0PM)&& !block->hasInterpolationFlagCF(DIR_PP0)&&!block->hasInterpolationFlagCF(DIR_00M)&&!block->hasInterpolationFlagCF(DIR_0P0) && !block->hasInterpolationFlagCF(DIR_P00))
      {
         SPtr<Block3D> fblockSW = grid->getBlock(fbx1+1, fbx2+1, fbx3+0, level);
         SPtr<Block3D> fblockSE;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+0, level);
         SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+1, level);
         SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2+1, fbx3+1, level);

         setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_PPM);
      }
      if (block->hasInterpolationFlagCF(DIR_MMM)&& !block->hasInterpolationFlagCF(DIR_0MM)&& !block->hasInterpolationFlagCF(DIR_M0M)&& !block->hasInterpolationFlagCF(DIR_MM0)&& !block->hasInterpolationFlagCF(DIR_00M)&& !block->hasInterpolationFlagCF(DIR_M00) && !block->hasInterpolationFlagCF(DIR_0M0))
      {
         SPtr<Block3D> fblockSW = grid->getBlock(fbx1, fbx2, fbx3+0, level);
         SPtr<Block3D> fblockSE;// = grid->getBlock(fbx1, fbx2, fbx3, level);
         SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1, fbx2, fbx3+1, level);
         SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1, fbx2, fbx3+1, level);

         setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_MMM);
      }
      if (block->hasInterpolationFlagCF(DIR_PMM)&& !block->hasInterpolationFlagCF(DIR_0MM)&& !block->hasInterpolationFlagCF(DIR_P0M)&& !block->hasInterpolationFlagCF(DIR_PM0)&& !block->hasInterpolationFlagCF(DIR_00M)&& !block->hasInterpolationFlagCF(DIR_P00) && !block->hasInterpolationFlagCF(DIR_0M0))
      {
         SPtr<Block3D> fblockSW = grid->getBlock(fbx1+1, fbx2, fbx3, level);
         SPtr<Block3D> fblockSE;// = grid->getBlock(fbx1+1, fbx2, fbx3+0, level);
         SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1+1, fbx2, fbx3+1, level);
         SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1+1, fbx2, fbx3+1, level);

         setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_PMM);
      }
      if (block->hasInterpolationFlagCF(DIR_MPM)&& !block->hasInterpolationFlagCF(DIR_0PM)&& !block->hasInterpolationFlagCF(DIR_M0M)&& !block->hasInterpolationFlagCF(DIR_MP0)&& !block->hasInterpolationFlagCF(DIR_00M)&& !block->hasInterpolationFlagCF(DIR_0P0) && !block->hasInterpolationFlagCF(DIR_M00))
      {
         SPtr<Block3D> fblockSW = grid->getBlock(fbx1, fbx2+1, fbx3+0, level);
         SPtr<Block3D> fblockSE;// = grid->getBlock(fbx1, fbx2+1, fbx3, level);
         SPtr<Block3D> fblockNW;// = grid->getBlock(fbx1, fbx2+1, fbx3+1, level);
         SPtr<Block3D> fblockNE;// = grid->getBlock(fbx1, fbx2+1, fbx3+1, level);

         setInterpolationConnectors(fblockSW, fblockSE, fblockNW, fblockNE, block, DIR_MPM);
      }

	}
   UBLOG(logDEBUG5, "SetInterpolationConnectorsBlockVisitor::setInterpolationConnectors() - end");
}
//////////////////////////////////////////////////////////////////////////
void SetInterpolationConnectorsBlockVisitor::setInterpolationConnectors(SPtr<Block3D> fBlockSW, SPtr<Block3D> fBlockSE, SPtr<Block3D> fBlockNW, SPtr<Block3D> fBlockNE, SPtr<Block3D> cBlock, int dir)
{
   UBLOG(logDEBUG5, "SetInterpolationConnectorsBlockVisitor::setInterpolationConnectors(...) - start");
	int fBlockSWRank = -999, fBlockSERank = -999, fBlockNWRank = -999, fBlockNERank = -999;
	if(fBlockSW) fBlockSWRank = fBlockSW->getRank();
	if(fBlockNW) fBlockNWRank = fBlockNW->getRank();
	if(fBlockSE) fBlockSERank = fBlockSE->getRank();
	if(fBlockNE) fBlockNERank = fBlockNE->getRank();
	int cBlockRank   = cBlock->getRank();

	LBMReal omegaF {0.0};
	if(fBlockSW) omegaF =LBMSystem::calcCollisionFactor(nue, fBlockSW->getLevel());
	if(fBlockNW) omegaF =LBMSystem::calcCollisionFactor(nue, fBlockNW->getLevel());
	if(fBlockSE) omegaF =LBMSystem::calcCollisionFactor(nue, fBlockSE->getLevel());
	if(fBlockNE) omegaF =LBMSystem::calcCollisionFactor(nue, fBlockNE->getLevel());
	LBMReal omegaC = LBMSystem::calcCollisionFactor(nue, cBlock->getLevel());
	iProcessor->setOmegas(omegaC, omegaF);

	InterpolationProcessorPtr cIProcessor(iProcessor->clone());
	InterpolationProcessorPtr fIProcessorSW(iProcessor->clone());
	InterpolationProcessorPtr fIProcessorSE(iProcessor->clone());
	InterpolationProcessorPtr fIProcessorNW(iProcessor->clone());
	InterpolationProcessorPtr fIProcessorNE(iProcessor->clone());

	CreateTransmittersHelper::TransmitterPtr senderCFevenEvenSW, receiverCFevenEvenSW, 
		senderCFevenOddNW,  receiverCFevenOddNW, 
		senderCFoddEvenSE,  receiverCFoddEvenSE, 
		senderCFoddOddNE,   receiverCFoddOddNE,
		senderFCevenEvenSW, receiverFCevenEvenSW, 
		senderFCevenOddNW,  receiverFCevenOddNW, 
		senderFCoddEvenSE,  receiverFCoddEvenSE, 
		senderFCoddOddNE,   receiverFCoddOddNE;

	if(fBlockSW) createTransmitters(cBlock, fBlockSW, dir, CreateTransmittersHelper::SW, senderCFevenEvenSW, receiverCFevenEvenSW, senderFCevenEvenSW, receiverFCevenEvenSW);
	if(fBlockNW) createTransmitters(cBlock, fBlockNW, dir, CreateTransmittersHelper::NW, senderCFevenOddNW, receiverCFevenOddNW, senderFCevenOddNW, receiverFCevenOddNW);
	if(fBlockSE) createTransmitters(cBlock, fBlockSE, dir, CreateTransmittersHelper::SE, senderCFoddEvenSE, receiverCFoddEvenSE, senderFCoddEvenSE, receiverFCoddEvenSE);
	if(fBlockNE) createTransmitters(cBlock, fBlockNE, dir, CreateTransmittersHelper::NE, senderCFoddOddNE, receiverCFoddOddNE, senderFCoddOddNE, receiverFCoddOddNE);

	if(cBlockRank == gridRank)
	{
      SPtr<Block3DConnector> connector(new CoarseToFineVectorConnector< TbTransmitter< CbVector< LBMReal > > >(cBlock,
			senderCFevenEvenSW, receiverCFevenEvenSW, senderCFevenOddNW,  receiverCFevenOddNW, 
			senderCFoddEvenSE,  receiverCFoddEvenSE,  senderCFoddOddNE,   receiverCFoddOddNE, 
			dir, cIProcessor) );
		cBlock->setConnector(connector);
	}
	if(fBlockSW && fBlockSWRank == gridRank)
	{
		SPtr<Block3DConnector> connector( new FineToCoarseVectorConnector< TbTransmitter< CbVector< LBMReal > > >(fBlockSW, 
			senderFCevenEvenSW, receiverFCevenEvenSW, dir, fIProcessorSW, EvenEvenSW) );
		fBlockSW->setConnector(connector);
	}
	if(fBlockNW && fBlockNWRank == gridRank)
	{
		SPtr<Block3DConnector> connector( new FineToCoarseVectorConnector< TbTransmitter< CbVector< LBMReal > > >(fBlockNW, 
			senderFCevenOddNW, receiverFCevenOddNW, dir, fIProcessorNW, EvenOddNW) );
		fBlockNW->setConnector(connector);
	}
	if(fBlockSE && fBlockSERank == gridRank)
	{
		SPtr<Block3DConnector> connector( new FineToCoarseVectorConnector< TbTransmitter< CbVector< LBMReal > > >(fBlockSE, 
			senderFCoddEvenSE, receiverFCoddEvenSE, dir, fIProcessorSE, OddEvenSE) );
		fBlockSE->setConnector(connector);
	}
	if(fBlockNE && fBlockNERank == gridRank)
	{
		SPtr<Block3DConnector> connector( new FineToCoarseVectorConnector< TbTransmitter< CbVector< LBMReal > > >(fBlockNE, 
			senderFCoddOddNE, receiverFCoddOddNE, dir, fIProcessorNE, OddOddNE) );
		fBlockNE->setConnector(connector);
	}
   UBLOG(logDEBUG5, "SetInterpolationConnectorsBlockVisitor::setInterpolationConnectors(...) - end");
}
//////////////////////////////////////////////////////////////////////////
void SetInterpolationConnectorsBlockVisitor::createTransmitters(SPtr<Block3D> cBlock, SPtr<Block3D> fBlock, int dir, 
                                                        CreateTransmittersHelper::IBlock ib, 
														              CreateTransmittersHelper::TransmitterPtr& senderCF, 
														              CreateTransmittersHelper::TransmitterPtr& receiverCF, 
														              CreateTransmittersHelper::TransmitterPtr& senderFC, 
														              CreateTransmittersHelper::TransmitterPtr& receiverFC)
{
   UBLOG(logDEBUG5, "SetInterpolationConnectorsBlockVisitor::createTransmitters(...) - start");
	CreateTransmittersHelper helper;
	int fBlockRank = fBlock->getRank();
	int cBlockRank = cBlock->getRank();
	if(fBlockRank == cBlockRank && fBlockRank == gridRank)
	{
		senderCF = receiverFC = CreateTransmittersHelper::TransmitterPtr( new TbLocalTransmitter< CbVector< LBMReal > >());
		senderFC = receiverCF = CreateTransmittersHelper::TransmitterPtr( new TbLocalTransmitter< CbVector< LBMReal > >());
	}
	else if(cBlockRank == gridRank)
	{
		helper.createTransmitters(cBlock, fBlock, dir, ib, senderCF, receiverCF, comm, CreateTransmittersHelper::MPI);
	}
	else if(fBlockRank == gridRank)
	{
		helper.createTransmitters(fBlock, cBlock, dir, ib, senderFC, receiverFC, comm, CreateTransmittersHelper::MPI);
	}
   UBLOG(logDEBUG5, "SetInterpolationConnectorsBlockVisitor::createTransmitters(...) - end");
}

