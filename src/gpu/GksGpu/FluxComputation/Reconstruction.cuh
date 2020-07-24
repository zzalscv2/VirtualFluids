#ifndef Reconstruction_CUH
#define Reconstruction_CUH

#include "VirtualFluidsDefinitions.h"

#include "Core/DataTypes.h"

#include "DataBase/DataBase.h"
#include "Parameters/Parameters.h"

#include "FlowStateData/FlowStateData.cuh"
#include "FlowStateData/FlowStateDataConversion.cuh"
#include "FlowStateData/AccessDeviceData.cuh"
#include "FlowStateData/ThermalDependencies.cuh"

namespace GksGpu {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void getCellIndicesN ( const uint faceIndex,
                                                  const DataBaseStruct& dataBase,
                                                  uint& posCellIndexN,
                                                  uint& negCellIndexN )
{
    posCellIndexN = dataBase.faceToCell[ POS_CELL( faceIndex, dataBase.numberOfFaces ) ];
    negCellIndexN = dataBase.faceToCell[ NEG_CELL( faceIndex, dataBase.numberOfFaces ) ];
}

__host__ __device__ inline void getCellIndicesTX( const uint faceIndex,
                                                  const DataBaseStruct& dataBase,
                                                  const uint posCellIndexN,
                                                  const uint negCellIndexN,
                                                  uint* posCellIndexTX,
                                                  uint* negCellIndexTX )
{
    posCellIndexTX[0] = dataBase.cellToCell[ CELL_TO_CELL( posCellIndexN, 0, dataBase.numberOfCells ) ];
    posCellIndexTX[1] = dataBase.cellToCell[ CELL_TO_CELL( negCellIndexN, 0, dataBase.numberOfCells ) ];

    negCellIndexTX[0] = dataBase.cellToCell[ CELL_TO_CELL( posCellIndexN, 1, dataBase.numberOfCells ) ];
    negCellIndexTX[1] = dataBase.cellToCell[ CELL_TO_CELL( negCellIndexN, 1, dataBase.numberOfCells ) ];
}

__host__ __device__ inline void getCellIndicesTY( const uint faceIndex,
                                                  const DataBaseStruct& dataBase,
                                                  const uint posCellIndexN,
                                                  const uint negCellIndexN,
                                                  uint* posCellIndexTY,
                                                  uint* negCellIndexTY )
{
    posCellIndexTY[0] = dataBase.cellToCell[ CELL_TO_CELL( posCellIndexN, 2, dataBase.numberOfCells ) ];
    posCellIndexTY[1] = dataBase.cellToCell[ CELL_TO_CELL( negCellIndexN, 2, dataBase.numberOfCells ) ];

    negCellIndexTY[0] = dataBase.cellToCell[ CELL_TO_CELL( posCellIndexN, 3, dataBase.numberOfCells ) ];
    negCellIndexTY[1] = dataBase.cellToCell[ CELL_TO_CELL( negCellIndexN, 3, dataBase.numberOfCells ) ];
}

__host__ __device__ inline void getCellIndicesTZ( const uint faceIndex,
                                                  const DataBaseStruct& dataBase,
                                                  const uint posCellIndexN,
                                                  const uint negCellIndexN,
                                                  uint* posCellIndexTZ,
                                                  uint* negCellIndexTZ )
{
    posCellIndexTZ[0] = dataBase.cellToCell[ CELL_TO_CELL( posCellIndexN, 4, dataBase.numberOfCells ) ];
    posCellIndexTZ[1] = dataBase.cellToCell[ CELL_TO_CELL( negCellIndexN, 4, dataBase.numberOfCells ) ];

    negCellIndexTZ[0] = dataBase.cellToCell[ CELL_TO_CELL( posCellIndexN, 5, dataBase.numberOfCells ) ];
    negCellIndexTZ[1] = dataBase.cellToCell[ CELL_TO_CELL( negCellIndexN, 5, dataBase.numberOfCells ) ];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void computeFaceCons( const ConservedVariables& posCons,
                                                 const ConservedVariables& negCons,
                                                 ConservedVariables& faceCons )
{
    faceCons.rho  = c1o2 * ( negCons.rho  + posCons.rho  );
    faceCons.rhoU = c1o2 * ( negCons.rhoU + posCons.rhoU );
    faceCons.rhoV = c1o2 * ( negCons.rhoV + posCons.rhoV );
    faceCons.rhoW = c1o2 * ( negCons.rhoW + posCons.rhoW );
    faceCons.rhoE = c1o2 * ( negCons.rhoE + posCons.rhoE );
#ifdef USE_PASSIVE_SCALAR
	faceCons.rhoS_1 = c1o2 * ( negCons.rhoS_1 + posCons.rhoS_1 );
	faceCons.rhoS_2 = c1o2 * ( negCons.rhoS_2 + posCons.rhoS_2 );
#endif // USE_PASSIVE_SCALAR
}

__host__ __device__ inline void computeGradN( const Parameters& parameters,
                                              const ConservedVariables& posCons,
                                              const ConservedVariables& negCons,
                                              const PrimitiveVariables& facePrim,
                                              ConservedVariables& gradN )
{
    gradN.rho  = ( posCons.rho  - negCons.rho  ) / ( parameters.dx * facePrim.rho );
    gradN.rhoU = ( posCons.rhoU - negCons.rhoU ) / ( parameters.dx * facePrim.rho );
    gradN.rhoV = ( posCons.rhoV - negCons.rhoV ) / ( parameters.dx * facePrim.rho );
    gradN.rhoW = ( posCons.rhoW - negCons.rhoW ) / ( parameters.dx * facePrim.rho );
    gradN.rhoE = ( posCons.rhoE - negCons.rhoE ) / ( parameters.dx * facePrim.rho );
#ifdef USE_PASSIVE_SCALAR
	gradN.rhoS_1 = ( posCons.rhoS_1 - negCons.rhoS_1 ) / ( parameters.dx * facePrim.rho );
	gradN.rhoS_2 = ( posCons.rhoS_2 - negCons.rhoS_2 ) / ( parameters.dx * facePrim.rho );
#endif // USE_PASSIVE_SCALAR
}

__host__ __device__ inline void computeGradT( const DataBaseStruct& dataBase,
                                              const Parameters& parameters,
                                              const uint posCellIndexT[2],
                                              const uint negCellIndexT[2],
                                              const PrimitiveVariables& facePrim,
                                              ConservedVariables& gradN )
{
    ConservedVariables cons;

    //////////////////////////////////////////////////////////////////////////
    {
        readCellData(posCellIndexT[0], dataBase, cons);

        gradN.rho  += c1o2 * cons.rho;
        gradN.rhoU += c1o2 * cons.rhoU;
        gradN.rhoV += c1o2 * cons.rhoV;
        gradN.rhoW += c1o2 * cons.rhoW;
        gradN.rhoE += c1o2 * cons.rhoE;
    #ifdef USE_PASSIVE_SCALAR
        gradN.rhoS_1 += c1o2 * cons.rhoS_1;
        gradN.rhoS_2 += c1o2 * cons.rhoS_2;
    #endif // USE_PASSIVE_SCALAR
    }
    {
        readCellData(posCellIndexT[1], dataBase, cons);

        gradN.rho  += c1o2 * cons.rho;
        gradN.rhoU += c1o2 * cons.rhoU;
        gradN.rhoV += c1o2 * cons.rhoV;
        gradN.rhoW += c1o2 * cons.rhoW;
        gradN.rhoE += c1o2 * cons.rhoE;
    #ifdef USE_PASSIVE_SCALAR
        gradN.rhoS_1 += c1o2 * cons.rhoS_1;
        gradN.rhoS_2 += c1o2 * cons.rhoS_2;
    #endif // USE_PASSIVE_SCALAR
    }
    //////////////////////////////////////////////////////////////////////////
    {
        readCellData(negCellIndexT[0], dataBase, cons);

        gradN.rho  -= c1o2 * cons.rho;
        gradN.rhoU -= c1o2 * cons.rhoU;
        gradN.rhoV -= c1o2 * cons.rhoV;
        gradN.rhoW -= c1o2 * cons.rhoW;
        gradN.rhoE -= c1o2 * cons.rhoE;
    #ifdef USE_PASSIVE_SCALAR
        gradN.rhoS_1 -= c1o2 * cons.rhoS_1;
        gradN.rhoS_2 -= c1o2 * cons.rhoS_2;
    #endif // USE_PASSIVE_SCALAR
    }
    {
        readCellData(negCellIndexT[1], dataBase, cons);

        gradN.rho  -= c1o2 * cons.rho;
        gradN.rhoU -= c1o2 * cons.rhoU;
        gradN.rhoV -= c1o2 * cons.rhoV;
        gradN.rhoW -= c1o2 * cons.rhoW;
        gradN.rhoE -= c1o2 * cons.rhoE;
    #ifdef USE_PASSIVE_SCALAR
        gradN.rhoS_1 -= c1o2 * cons.rhoS_1;
        gradN.rhoS_2 -= c1o2 * cons.rhoS_2;
    #endif // USE_PASSIVE_SCALAR
    }
    //////////////////////////////////////////////////////////////////////////
    {
        gradN.rho  /= c2o1 * parameters.dx * facePrim.rho;
        gradN.rhoU /= c2o1 * parameters.dx * facePrim.rho;
        gradN.rhoV /= c2o1 * parameters.dx * facePrim.rho;
        gradN.rhoW /= c2o1 * parameters.dx * facePrim.rho;
        gradN.rhoE /= c2o1 * parameters.dx * facePrim.rho;
    #ifdef USE_PASSIVE_SCALAR
        gradN.rhoS_1 /= c2o1 * parameters.dx * facePrim.rho;
        gradN.rhoS_2 /= c2o1 * parameters.dx * facePrim.rho;
    #endif // USE_PASSIVE_SCALAR
    }
    //////////////////////////////////////////////////////////////////////////
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ inline void reconstructFiniteDifferences( const uint faceIndex,
                                                              const DataBaseStruct& dataBase,
                                                              const Parameters& parameters,
                                                              const char direction,
                                                              ConservedVariables& gradN,
                                                              ConservedVariables& gradT1,
                                                              ConservedVariables& gradT2,
                                                              PrimitiveVariables& facePrim,
                                                              real& K )
{
    uint posCellIndexN, negCellIndexN;

    getCellIndicesN( faceIndex, dataBase, posCellIndexN, negCellIndexN );
    
    {
        ConservedVariables posCons, negCons, faceCons;

        readCellData(posCellIndexN, dataBase, posCons);
        readCellData(negCellIndexN, dataBase, negCons);
        
        computeFaceCons(posCons, negCons, faceCons);

    #ifdef USE_PASSIVE_SCALAR
        {
            //K = getK(faceCons);
        }
    #endif

        facePrim = toPrimitiveVariables( faceCons, K, false );

        computeGradN( parameters, posCons, negCons, facePrim, gradN );
    }

    {
        uint posCellIndexT1[2];
        uint negCellIndexT1[2];
    
        if( direction == 'x' ) getCellIndicesTY(faceIndex, dataBase, posCellIndexN, negCellIndexN, posCellIndexT1, negCellIndexT1);
        if( direction == 'y' ) getCellIndicesTZ(faceIndex, dataBase, posCellIndexN, negCellIndexN, posCellIndexT1, negCellIndexT1);
        if( direction == 'z' ) getCellIndicesTX(faceIndex, dataBase, posCellIndexN, negCellIndexN, posCellIndexT1, negCellIndexT1);

        computeGradT( dataBase, parameters, posCellIndexT1, negCellIndexT1, facePrim, gradT1 );
    }

    {
        uint posCellIndexT2[2];
        uint negCellIndexT2[2];
    
        if( direction == 'x' ) getCellIndicesTZ(faceIndex, dataBase, posCellIndexN, negCellIndexN, posCellIndexT2, negCellIndexT2);
        if( direction == 'y' ) getCellIndicesTX(faceIndex, dataBase, posCellIndexN, negCellIndexN, posCellIndexT2, negCellIndexT2);
        if( direction == 'z' ) getCellIndicesTY(faceIndex, dataBase, posCellIndexN, negCellIndexN, posCellIndexT2, negCellIndexT2);

        computeGradT( dataBase, parameters, posCellIndexT2, negCellIndexT2, facePrim, gradT2 );
    }
}

} // namespace GksGpu






#endif