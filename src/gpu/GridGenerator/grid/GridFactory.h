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
//           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  \   
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
//! \file GridFactory.h
//! \ingroup grid
//! \author Soeren Peters, Stephan Lenz
//=======================================================================================
#ifndef GRID_FACTORY_H
#define GRID_FACTORY_H

#include "global.h"

#include "geometries/Cuboid/Cuboid.h"

#include "grid/GridStrategy/GridCpuStrategy/GridCpuStrategy.h"
#include "grid/distributions/Distribution.h"
#include "grid/GridImp.h"

enum class Device
{
    CPU, GPU
};

enum class TriangularMeshDiscretizationMethod
{
    RAYCASTING, POINT_IN_OBJECT, POINT_UNDER_TRIANGLE
};

class GRIDGENERATOR_EXPORT GridFactory
{
public:
    static SPtr<GridFactory> make()
    {
        return SPtr<GridFactory>(new GridFactory());
    }

private:
    GridFactory()
    {
        gridStrategy = SPtr<GridStrategy>(new GridCpuStrategy());
    }

public:
    SPtr<Grid> makeGrid(Object* gridShape, real startX, real startY, real startZ, real endX, real endY, real endZ, real delta, uint level, const std::string& d3Qxx = "D3Q27")
    {
        Distribution distribution = DistributionHelper::getDistribution(d3Qxx);

        SPtr<GridImp> grid;

        grid = GridImp::makeShared(gridShape, startX, startY, startZ, endX, endY, endZ, delta, gridStrategy, distribution, level);

        return grid;
    }


    void setGridStrategy(Device device)
    {
        switch (device)
        {
        case Device::CPU:
            gridStrategy = SPtr<GridStrategy>(new GridCpuStrategy()); break;
        }
    }

    void setGridStrategy(SPtr<GridStrategy> gridStrategy)
    {
        this->gridStrategy = gridStrategy;
    }

private:
    SPtr<GridStrategy> gridStrategy;
};


#endif
