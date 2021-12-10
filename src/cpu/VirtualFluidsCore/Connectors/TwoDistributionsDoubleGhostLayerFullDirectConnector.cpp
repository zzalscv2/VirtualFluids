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
//! \file TwoDistributionsDoubleGhostLayerFullDirectConnector.cpp
//! \ingroup Connectors
//! \author Konstantin Kutscher
//=======================================================================================

#include "TwoDistributionsDoubleGhostLayerFullDirectConnector.h"
#include "LBMKernel.h"
#include "DataSet3D.h"

TwoDistributionsDoubleGhostLayerFullDirectConnector::TwoDistributionsDoubleGhostLayerFullDirectConnector(SPtr<Block3D> from, SPtr<Block3D> to, int sendDir)
    : FullDirectConnector(from, to, sendDir)
{

}
//////////////////////////////////////////////////////////////////////////
void TwoDistributionsDoubleGhostLayerFullDirectConnector::init()
{
    FullDirectConnector::init();

	fFrom =dynamicPointerCast<EsoTwist3D>(from.lock()->getKernel()->getDataSet()->getFdistributions());
	fTo = dynamicPointerCast<EsoTwist3D>(to.lock()->getKernel()->getDataSet()->getFdistributions());
	hFrom = dynamicPointerCast<EsoTwist3D>(from.lock()->getKernel()->getDataSet()->getHdistributions());
	hTo = dynamicPointerCast<EsoTwist3D>(to.lock()->getKernel()->getDataSet()->getHdistributions());
    pressureFrom = from.lock()->getKernel()->getDataSet()->getPressureField();
    pressureTo   = to.lock()->getKernel()->getDataSet()->getPressureField();
}
//////////////////////////////////////////////////////////////////////////
void TwoDistributionsDoubleGhostLayerFullDirectConnector::sendVectors()
{
    updatePointers();
    exchangeData();
}
//////////////////////////////////////////////////////////////////////////
void TwoDistributionsDoubleGhostLayerFullDirectConnector::exchangeData()
{
    ////////////////////////////////////////////////////////////
    // relation between ghost layer and regular nodes
    // maxX1m3 maxX1m2 ... minX1p2 minX1p3 - regular nodes
    // minX1   minX1p1 ... maxX1m1 maxX1   - ghost layer
    ////////////////////////////////////////////////////////////

    int minX1   = 0;
    int minX1p1 = minX1 + 1;
    int minX1p2 = minX1 + 2;
    int minX1p3 = minX1 + 3;
    int maxX1m1 = maxX1 - 1;
    int maxX1m2 = maxX1 - 2;
    int maxX1m3 = maxX1 - 3;

    int minX2   = 0;
    int minX2p1 = minX2 + 1;
    int minX2p2 = minX2 + 2;
    int minX2p3 = minX2 + 3;
    int maxX2m1 = maxX2 - 1;
    int maxX2m2 = maxX2 - 2;
    int maxX2m3 = maxX2 - 3;

    int minX3   = 0;
    int minX3p1 = minX3 + 1;
    int minX3p2 = minX3 + 2;
    int minX3p3 = minX3 + 3;
    int maxX3m1 = maxX3 - 1;
    int maxX3m2 = maxX3 - 2;
    int maxX3m3 = maxX3 - 3;

    // EAST
    if (sendDir == D3Q27System::E) {
        for (int x3 = minX3p2; x3 <= maxX3m2; x3++) {
            for (int x2 = minX2p2; x2 <= maxX2m2; x2++) {
                exchangeData(maxX1m3, x2, x3, minX1, x2, x3);
                exchangeData(maxX1m2, x2, x3, minX1p1, x2, x3);
            }
        }
    }
    // WEST
    else if (sendDir == D3Q27System::W) {
        for (int x3 = minX3p2; x3 <= maxX3m2; x3++) {
            for (int x2 = minX2p2; x2 <= maxX2m2; x2++) {
                exchangeData(minX1p3, x2, x3, maxX1, x2, x3);
                exchangeData(minX1p2, x2, x3, maxX1m1, x2, x3);
            }
        }
    }
    // NORTH
    else if (sendDir == D3Q27System::N) {
        for (int x3 = minX3p2; x3 <= maxX3m2; x3++) {
            for (int x1 = minX1p2; x1 <= maxX1m2; x1++) {
                exchangeData(x1, maxX2m3, x3, x1, minX2, x3);
                exchangeData(x1, maxX2m2, x3, x1, minX2p1, x3);
            }
        }
    }
    // SOUTH
    else if (sendDir == D3Q27System::S) {
        for (int x3 = minX3p2; x3 <= maxX3m2; x3++) {
            for (int x1 = minX1p2; x1 <= maxX1m2; x1++) {
                exchangeData(x1, minX2p3, x3, x1, maxX2, x3);
                exchangeData(x1, minX2p2, x3, x1, maxX2m1, x3);
            }
        }
    }

    // TOP
    else if (sendDir == D3Q27System::T) {
        for (int x2 = minX2p2; x2 <= maxX2m2; x2++) {
            for (int x1 = minX1p2; x1 <= maxX1m2; x1++) {
                exchangeData(x1, x2, maxX3m3, x1, x2, minX3);
                exchangeData(x1, x2, maxX3m2, x1, x2, minX3p1);
            }
        }
    }
    // BOTTOM
    else if (sendDir == D3Q27System::B) {
        for (int x2 = minX2p2; x2 <= maxX2m2; x2++) {
            for (int x1 = minX1p2; x1 <= maxX1m2; x1++) {
                exchangeData(x1, x2, minX3p3, x1, x2, maxX3);
                exchangeData(x1, x2, minX3p2, x1, x2, maxX3m1);
            }
        }
    }
    // NORTHEAST
    else if (sendDir == D3Q27System::NE) {
        for (int x3 = minX3p2; x3 <= maxX3m2; x3++) {
            exchangeData(maxX1m3, maxX2m3, x3, minX1, minX2, x3);
            exchangeData(maxX1m2, maxX2m2, x3, minX1p1, minX2p1, x3);
            exchangeData(maxX1m3, maxX2m2, x3, minX1, minX2p1, x3);
            exchangeData(maxX1m2, maxX2m3, x3, minX1p1, minX2, x3);
        }
    }
    // NORTHWEST
    else if (sendDir == D3Q27System::NW) {
        for (int x3 = minX3p2; x3 <= maxX3m2; x3++) {
            exchangeData(minX1p3, maxX2m3, x3, maxX1, minX2, x3);
            exchangeData(minX1p2, maxX2m2, x3, maxX1m1, minX2p1, x3);
            exchangeData(minX1p3, maxX2m2, x3, maxX1, minX2p1, x3);
            exchangeData(minX1p2, maxX2m3, x3, maxX1m1, minX2, x3);
        }
    }
    // SOUTHWEST
    else if (sendDir == D3Q27System::SW) {
        for (int x3 = minX3p2; x3 <= maxX3m2; x3++) {
            exchangeData(minX1p3, minX2p3, x3, maxX1, maxX2, x3);
            exchangeData(minX1p2, minX2p2, x3, maxX1m1, maxX2m1, x3);
            exchangeData(minX1p3, minX2p2, x3, maxX1, maxX2m1, x3);
            exchangeData(minX1p2, minX2p3, x3, maxX1m1, maxX2, x3);
        }
    }
    // SOUTHEAST
    else if (sendDir == D3Q27System::SE) {
        for (int x3 = minX3p2; x3 <= maxX3m2; x3++) {
            exchangeData(maxX1m3, minX2p3, x3, minX1, maxX2, x3);
            exchangeData(maxX1m2, minX2p2, x3, minX1p1, maxX2m1, x3);
            exchangeData(maxX1m3, minX2p2, x3, minX1, maxX2m1, x3);
            exchangeData(maxX1m2, minX2p3, x3, minX1p1, maxX2, x3);
        }
    } else if (sendDir == D3Q27System::TE)
        for (int x2 = minX2p2; x2 <= maxX2m2; x2++) {
            exchangeData(maxX1m3, x2, maxX3m3, minX1, x2, minX3);
            exchangeData(maxX1m2, x2, maxX3m2, minX1p1, x2, minX3p1);
            exchangeData(maxX1m3, x2, maxX3m2, minX1, x2, minX3p1);
            exchangeData(maxX1m2, x2, maxX3m3, minX1p1, x2, minX3);
        }
    else if (sendDir == D3Q27System::BW)
        for (int x2 = minX2p2; x2 <= maxX2m2; x2++) {
            exchangeData(minX1p3, x2, minX3p3, maxX1, x2, maxX3);
            exchangeData(minX1p2, x2, minX3p2, maxX1m1, x2, maxX3m1);
            exchangeData(minX1p3, x2, minX3p2, maxX1, x2, maxX3m1);
            exchangeData(minX1p2, x2, minX3p3, maxX1m1, x2, maxX3);
        }
    else if (sendDir == D3Q27System::BE)
        for (int x2 = minX2p2; x2 <= maxX2m2; x2++) {
            exchangeData(maxX1m3, x2, minX3p3, minX1, x2, maxX3);
            exchangeData(maxX1m2, x2, minX3p2, minX1p1, x2, maxX3m1);
            exchangeData(maxX1m3, x2, minX3p2, minX1, x2, maxX3m1);
            exchangeData(maxX1m2, x2, minX3p3, minX1p1, x2, maxX3);
        }
    else if (sendDir == D3Q27System::TW)
        for (int x2 = minX2p2; x2 <= maxX2m2; x2++) {
            exchangeData(minX1p3, x2, maxX3m3, maxX1, x2, minX3);
            exchangeData(minX1p2, x2, maxX3m2, maxX1m1, x2, minX3p1);
            exchangeData(minX1p3, x2, maxX3m2, maxX1, x2, minX3p1);
            exchangeData(minX1p2, x2, maxX3m3, maxX1m1, x2, minX3);
        }
    else if (sendDir == D3Q27System::TN)
        for (int x1 = minX1p2; x1 <= maxX1m2; x1++) {
            exchangeData(x1, maxX2m3, maxX3m3, x1, minX2, minX3);
            exchangeData(x1, maxX2m2, maxX3m2, x1, minX2p1, minX3p1);
            exchangeData(x1, maxX2m3, maxX3m2, x1, minX2, minX3p1);
            exchangeData(x1, maxX2m2, maxX3m3, x1, minX2p1, minX3);
        }
    else if (sendDir == D3Q27System::BS)
        for (int x1 = minX1p2; x1 <= maxX1m2; x1++) {
            exchangeData(x1, minX2p3, minX3p3, x1, maxX2, maxX3);
            exchangeData(x1, minX2p2, minX3p2, x1, maxX2m1, maxX3m1);
            exchangeData(x1, minX2p3, minX3p2, x1, maxX2, maxX3m1);
            exchangeData(x1, minX2p2, minX3p3, x1, maxX2m1, maxX3);
        }
    else if (sendDir == D3Q27System::BN)
        for (int x1 = minX1p2; x1 <= maxX1m2; x1++) {
            exchangeData(x1, maxX2m3, minX3p3, x1, minX2, maxX3);
            exchangeData(x1, maxX2m2, minX3p2, x1, minX2p1, maxX3m1);
            exchangeData(x1, maxX2m3, minX3p2, x1, minX2, maxX3m1);
            exchangeData(x1, maxX2m2, minX3p3, x1, minX2p1, maxX3);
        }
    else if (sendDir == D3Q27System::TS)
        for (int x1 = minX1p2; x1 <= maxX1m2; x1++) {
            exchangeData(x1, minX2p3, maxX3m3, x1, maxX2, minX3);
            exchangeData(x1, minX2p2, maxX3m2, x1, maxX2m1, minX3p1);
            exchangeData(x1, minX2p3, maxX3m2, x1, maxX2, minX3p1);
            exchangeData(x1, minX2p2, maxX3m3, x1, maxX2m1, minX3);
        }
    else if (sendDir == D3Q27System::TSW) {
        exchangeData(minX1p3, minX2p3, maxX3m3, maxX1, maxX2, minX3);
        exchangeData(minX1p2, minX2p2, maxX3m2, maxX1m1, maxX2m1, minX3p1);
        exchangeData(minX1p3, minX2p2, maxX3m2, maxX1, maxX2m1, minX3p1);
        exchangeData(minX1p2, minX2p3, maxX3m2, maxX1m1, maxX2, minX3p1);
        exchangeData(minX1p2, minX2p2, maxX3m3, maxX1m1, maxX2m1, minX3);
        exchangeData(minX1p3, minX2p3, maxX3m2, maxX1, maxX2, minX3p1);
        exchangeData(minX1p3, minX2p2, maxX3m3, maxX1, maxX2m1, minX3);
        exchangeData(minX1p2, minX2p3, maxX3m3, maxX1m1, maxX2, minX3);
    } else if (sendDir == D3Q27System::TSE) {
        exchangeData(maxX1m3, minX1p3, maxX3m3, minX1, maxX2, minX3);
        exchangeData(maxX1m2, minX1p2, maxX3m2, minX1p1, maxX2m1, minX3p1);
        exchangeData(maxX1m3, minX1p2, maxX3m2, minX1, maxX2m1, minX3p1);
        exchangeData(maxX1m2, minX1p3, maxX3m2, minX1p1, maxX2, minX3p1);
        exchangeData(maxX1m2, minX1p2, maxX3m3, minX1p1, maxX2m1, minX3);
        exchangeData(maxX1m3, minX1p3, maxX3m2, minX1, maxX2, minX3p1);
        exchangeData(maxX1m3, minX1p2, maxX3m3, minX1, maxX2m1, minX3);
        exchangeData(maxX1m2, minX1p3, maxX3m3, minX1p1, maxX2, minX3);
    } else if (sendDir == D3Q27System::TNW) {
        exchangeData(minX1p3, maxX2m3, maxX3m3, maxX1, minX2, minX3);
        exchangeData(minX1p2, maxX2m2, maxX3m2, maxX1m1, minX2p1, minX3p1);
        exchangeData(minX1p3, maxX2m2, maxX3m2, maxX1, minX2p1, minX3p1);
        exchangeData(minX1p2, maxX2m3, maxX3m2, maxX1m1, minX2, minX3p1);
        exchangeData(minX1p2, maxX2m2, maxX3m3, maxX1m1, minX2p1, minX3);
        exchangeData(minX1p3, maxX2m3, maxX3m2, maxX1, minX2, minX3p1);
        exchangeData(minX1p3, maxX2m2, maxX3m3, maxX1, minX2p1, minX3);
        exchangeData(minX1p2, maxX2m3, maxX3m3, maxX1m1, minX2, minX3);
    } else if (sendDir == D3Q27System::TNE) {
        exchangeData(maxX1m3, maxX2m3, maxX3m3, minX1, minX2, minX3);
        exchangeData(maxX1m2, maxX2m2, maxX3m2, minX1p1, minX2p1, minX3p1);
        exchangeData(maxX1m3, maxX2m2, maxX3m2, minX1, minX2p1, minX3p1);
        exchangeData(maxX1m2, maxX2m3, maxX3m2, minX1p1, minX2, minX3p1);
        exchangeData(maxX1m2, maxX2m2, maxX3m3, minX1p1, minX2p1, minX3);
        exchangeData(maxX1m3, maxX2m3, maxX3m2, minX1, minX2, minX3p1);
        exchangeData(maxX1m3, maxX2m2, maxX3m3, minX1, minX2p1, minX3);
        exchangeData(maxX1m2, maxX2m3, maxX3m3, minX1p1, minX2, minX3);
    } else if (sendDir == D3Q27System::BSW) {
        exchangeData(minX1p3, minX2p3, minX3p3, maxX1, maxX2, maxX3);
        exchangeData(minX1p2, minX2p2, minX3p2, maxX1m1, maxX2m1, maxX3m1);
        exchangeData(minX1p3, minX2p2, minX3p2, maxX1, maxX2m1, maxX3m1);
        exchangeData(minX1p2, minX2p3, minX3p2, maxX1m1, maxX2, maxX3m1);
        exchangeData(minX1p2, minX2p2, minX3p3, maxX1m1, maxX2m1, maxX3);
        exchangeData(minX1p3, minX2p3, minX3p2, maxX1, maxX2, maxX3m1);
        exchangeData(minX1p3, minX2p2, minX3p3, maxX1, maxX2m1, maxX3);
        exchangeData(minX1p2, minX2p3, minX3p3, maxX1m1, maxX2, maxX3);
    } else if (sendDir == D3Q27System::BSE) {
        exchangeData(maxX1m3, minX2p3, minX3p3, minX1, maxX2, maxX3);
        exchangeData(maxX1m2, minX2p2, minX3p2, minX1p1, maxX2m1, maxX3m1);
        exchangeData(maxX1m3, minX2p2, minX3p2, minX1, maxX2m1, maxX3m1);
        exchangeData(maxX1m2, minX2p3, minX3p2, minX1p1, maxX2, maxX3m1);
        exchangeData(maxX1m2, minX2p2, minX3p3, minX1p1, maxX2m1, maxX3);
        exchangeData(maxX1m3, minX2p3, minX3p2, minX1, maxX2, maxX3m1);
        exchangeData(maxX1m3, minX2p2, minX3p3, minX1, maxX2m1, maxX3);
        exchangeData(maxX1m2, minX2p3, minX3p3, minX1p1, maxX2, maxX3);
    } else if (sendDir == D3Q27System::BNW) {
        exchangeData(minX1p3, maxX2m3, minX3p3, maxX1, minX2, maxX3);
        exchangeData(minX1p2, maxX2m2, minX3p2, maxX1m1, minX2p1, maxX3m1);
        exchangeData(minX1p3, maxX2m2, minX3p2, maxX1, minX2p1, maxX3m1);
        exchangeData(minX1p2, maxX2m3, minX3p2, maxX1m1, minX2, maxX3m1);
        exchangeData(minX1p2, maxX2m2, minX3p3, maxX1m1, minX2p1, maxX3);
        exchangeData(minX1p3, maxX2m3, minX3p2, maxX1, minX2, maxX3m1);
        exchangeData(minX1p3, maxX2m2, minX3p3, maxX1, minX2p1, maxX3);
        exchangeData(minX1p2, maxX2m3, minX3p3, maxX1m1, minX2, maxX3);
    } else if (sendDir == D3Q27System::BNE) {
        exchangeData(maxX1m3, maxX2m3, minX3p3, minX1, minX2, maxX3);
        exchangeData(maxX1m2, maxX2m2, minX3p2, minX1p1, minX2p1, maxX3m1);
        exchangeData(maxX1m3, maxX2m2, minX3p2, minX1, minX2p1, maxX3m1);
        exchangeData(maxX1m2, maxX2m3, minX3p2, minX1p1, minX2, maxX3m1);
        exchangeData(maxX1m2, maxX2m2, minX3p3, minX1p1, minX2p1, maxX3);
        exchangeData(maxX1m3, maxX2m3, minX3p2, minX1, minX2, maxX3m1);
        exchangeData(maxX1m3, maxX2m2, minX3p3, minX1, minX2p1, maxX3);
        exchangeData(maxX1m2, maxX2m3, minX3p3, minX1p1, minX2, maxX3);
    } else
        UB_THROW(UbException(UB_EXARGS, "unknown dir"));

}
