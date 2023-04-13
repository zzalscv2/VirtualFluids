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
//! \file lb.h
//! \ingroup LBM
//! \author Martin Schoenherr
//=======================================================================================#ifndef _LB_H_
#ifndef _LB_H_
#define _LB_H_

//////////////////////////////////////////////////////////////////////////
#define GEO_FLUID_OLD    1
#define GEO_VELO         2
#define GEO_PRESS        4

//////////////////////////
//porous media
#define GEO_PM_0         5
#define GEO_PM_1         6
#define GEO_PM_2         7
//////////////////////////

#define GEO_SOLID       15
#define GEO_VOID        16

#define GEO_FLUID       19
#define OFFSET_BCsInGeo 20
//////////////////////////////////////////////////////////////////////////

#define LES false // LES Simulation

#define STARTOFFX 16
#define STARTOFFY 16
#define STARTOFFZ 16

#define X1PERIODIC true
#define X2PERIODIC true
#define X3PERIODIC true

#define INTERFACE_E 0
#define INTERFACE_W 1
#define INTERFACE_N 2
#define INTERFACE_S 3
#define INTERFACE_T 4
#define INTERFACE_B 5


#include "Core/DataTypes.h"

#include <string>
#include <vector>

//! \brief An enumeration for selecting a turbulence model
enum class TurbulenceModel {
   //! - Smagorinsky
   Smagorinsky,
    //! - AMD (Anisotropic Minimum Dissipation) model, see e.g. Rozema et al., Phys. Fluids 27, 085107 (2015), https://doi.org/10.1063/1.4928700
   AMD,
    //! - QR model by Verstappen 
   QR,
    //! - TODO: move the WALE model here from the old kernels
    //WALE
    //! - No turbulence model
   None
};

//! \brief An enumeration for selecting a template of the collision kernel (CumulantK17)
enum class CollisionTemplate {
   //! - Default: plain collision without additional read/write
   Default,
   //!  - WriteMacroVars: collision \w write out macroscopic variables
   WriteMacroVars,
   //! - ApplyBodyForce: collision \w read and apply body force in the collision kernel
   ApplyBodyForce,
   //! - AllFeatures: collision \w write out macroscopic variables AND read and apply body force
   AllFeatures,
   //! - Border: collision on border nodes
   SubDomainBorder
};
constexpr std::initializer_list<CollisionTemplate> all_CollisionTemplate  = { CollisionTemplate::Default, CollisionTemplate::WriteMacroVars, CollisionTemplate::ApplyBodyForce, CollisionTemplate::AllFeatures, CollisionTemplate::SubDomainBorder};
constexpr std::initializer_list<CollisionTemplate> bulk_CollisionTemplate = { CollisionTemplate::Default, CollisionTemplate::WriteMacroVars, CollisionTemplate::ApplyBodyForce, CollisionTemplate::AllFeatures};

//Interface Cells
// example of old names (pre 2023) ICellCFC: interpolation from Coarse (C) to Fine (F), indices of the Coarse cells (C)
typedef struct ICells{
   uint* fineCellIndices;
   uint* coarseCellIndices;
   uint numberOfCells;
} InterpolationCells;

//! \brief stores location of neighboring cell (necessary for refinement into the wall)
typedef struct ICellNeigh{
   real* x;
   real* y;
   real* z;
} InterpolationCellNeighbor;

// Distribution functions g 6
typedef struct  Distri6 { // ADD IN FUTURE RELEASE
   real* g[6];
} Distributions6;

// Distribution functions f 7
typedef struct  Distri7{ // ADD IN FUTURE RELEASE
   real* f[7];
} Distributions7;

// Distribution functions f 19
typedef struct  Distri19{ // DEPRECATED
   real* f[19];
} Distributions19;

// Distribution functions f 27
typedef struct  Distri27{
   real* f[27];
} Distributions27, DistributionReferences27;

// Subgrid distances q 27
typedef struct SubgridDist27{
   real* q[27];
} SubgridDistances27;

//Q for second order BCs
//! \struct to manage sub-grid-distances (q) for second order Boundary Conditions (BCs)
typedef struct QforBC{
   int* k;
   int* kN;
   long long* valueQ;
   real* qread;
   real* q27[27];
   real* q19[19];
   unsigned int numberOfBCnodes=0;
   int kArray;
   real *Vx,      *Vy,      *Vz;
   real *Vx1,     *Vy1,     *Vz1;
   real *deltaVz, *RhoBC;
   real *normalX, *normalY, *normalZ;
}QforBoundaryConditions;

typedef struct QforPrecursorBC{
   int* k;
   int numberOfBCnodes=0;
   int sizeQ;
   int numberOfPrecursorNodes=0;
   uint streamIndex=0;
   uint nPrecursorReads=0;
   uint timeStepsBetweenReads;
   size_t numberOfQuantities;
   real* q27[27];
   uint* planeNeighbor0PP, *planeNeighbor0PM, *planeNeighbor0MP, *planeNeighbor0MM;
   real* weights0PP, *weights0PM, *weights0MP,  *weights0MM;
   real* last, *current, *next;
   real velocityX, velocityY, velocityZ;
}QforPrecursorBoundaryConditions;

//BCTemp
typedef struct TempforBC{  // ADD IN FUTURE RELEASE
   int* k;
   real* temp;
   int kTemp=0;
}TempforBoundaryConditions;

//BCTempVel
typedef struct TempVelforBC{  // ADD IN FUTURE RELEASE
   int* k;
   real* temp;
   real* tempPulse;
   real* velo;
   int kTemp=0;
}TempVelforBoundaryConditions;

//BCTempPress
typedef struct TempPressforBC{  // ADD IN FUTURE RELEASE
   int* k;
   real* temp;
   real* velo;
   int kTemp=0;
}TempPressforBoundaryConditions;

// Settings for wall model used in StressBC
typedef struct WMparas{
   real* z0;
   int* samplingOffset;
   bool hasMonitor;
   real* u_star;
   real* Fx;
   real* Fy;
   real* Fz;
}WallModelParameters;


//measurePoints
typedef struct MeasP{ // ADD IN FUTURE RELEASE
   std::string name;
   uint k;
   std::vector<real> Vx;
   std::vector<real> Vy;
   std::vector<real> Vz;
   std::vector<real> Rho;
   //real* Vx;
   //real* Vy;
   //real* Vz;
   //real* Rho;
}MeasurePoints;

//Process Neighbors
typedef struct PN27{
   real* f[27];
   uint memsizeFs;
   int* index;
   uint memsizeIndex;
   uint rankNeighbor;
   int numberOfNodes;
   int numberOfFs;
}ProcessNeighbor27;

typedef struct PN_F3 { // ADD IN FUTURE RELEASE
   real* g[6];
   uint memsizeGs;
   int* index;
   uint memsizeIndex;
   uint rankNeighbor;
   int numberOfNodes;
   int numberOfGs;
}ProcessNeighborF3;

//path line particles // DEPRECATED
typedef struct PLP{
   bool *stuck, *hot;
   real *coordXabsolut, *coordYabsolut, *coordZabsolut;
   real *coordXlocal,   *coordYlocal,   *coordZlocal;
   real *veloX,         *veloY,         *veloZ;
   real *randomLocationInit;
   uint *timestep;
   uint *ID;
   uint *cellBaseID;
   uint numberOfParticles, numberOfTimestepsParticles;
   uint memSizeID, memSizeTimestep, memSizerealAll, memSizereal, memSizeBool, memSizeBoolBC;
}PathLineParticles;

//////////////////////////////////////////////////////////////////////////
// DEPRECATED
inline int vectorPosition(int i, int j, int k, int Lx, int Ly )
{
   //return((j+15)*(Lx+2*16)+(i+15));
   return((Lx+2*STARTOFFX)*((Ly+2*STARTOFFY)*(k+STARTOFFZ)+(j+STARTOFFY))+(i+STARTOFFX));
}
//////////////////////////////////////////////////////////////////////////

#endif
