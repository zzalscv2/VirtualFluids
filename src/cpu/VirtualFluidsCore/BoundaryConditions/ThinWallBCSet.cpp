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
//! \file ThinWallBCSet.cpp
//! \ingroup BoundarConditions
//! \author Konstantin Kutscher
//=======================================================================================
#include "ThinWallBCSet.h"

#include "ThinWallNoSlipBCStrategy.h"

#include "LBMKernel.h"
#include "BCStrategyType.h"

//////////////////////////////////////////////////////////////////////////
ThinWallBCSet::ThinWallBCSet(SPtr<ILBMKernel> kernel) : BCSet(kernel) {}
//////////////////////////////////////////////////////////////////////////
SPtr<BCSet> ThinWallBCSet::clone(SPtr<ILBMKernel> kernel)
{
    SPtr<BCSet> BCSet(new ThinWallBCSet(kernel));
    return BCSet;
}
//////////////////////////////////////////////////////////////////////////
void ThinWallBCSet::applyPostCollisionBC()
{
    BCSet::applyPostCollisionBC();

    for (SPtr<BCStrategy> bc : postBC) {
        if (bc->getType() == BCStrategyType::ThinWallNoSlipBCStrategy) {
            dynamicPointerCast<ThinWallNoSlipBCStrategy>(bc)->setPass(2);
            bc->applyBC();
            dynamicPointerCast<ThinWallNoSlipBCStrategy>(bc)->setPass(1);
        }
    }
}
