#include "CellUpdate.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>

#include "Core/PointerDefinitions.h"
#include "Core/RealConstants.h"

#include "DataBase/DataBaseStruct.h"

#include "Definitions/MemoryAccessPattern.h"
#include "Definitions/PassiveScalar.h"

#include "FlowStateData/FlowStateData.cuh"
#include "FlowStateData/FlowStateDataConversion.cuh"
#include "FlowStateData/ThermalDependencies.cuh"

#include "CudaUtility/CudaRunKernel.hpp"

__global__                 void cellUpdateKernel  ( DataBaseStruct dataBase, Parameters parameters, uint startIndex, uint numberOfEntities );

__host__ __device__ inline void cellUpdateFunction( DataBaseStruct dataBase, Parameters parameters, uint startIndex, uint index );

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CellUpdate::run( SPtr<DataBase> dataBase, Parameters parameters, uint level )
{
    CudaUtility::CudaGrid grid( dataBase->perLevelCount[ level ].numberOfBulkCells, 32 );

    runKernel( cellUpdateKernel,
               cellUpdateFunction,
               dataBase->getDeviceType(), grid, 
               dataBase->toStruct(),
               parameters,
               dataBase->perLevelCount[ level ].startOfCells );

    cudaDeviceSynchronize();

    getLastCudaError("CellUpdate::run( SPtr<DataBase> dataBase, Parameters parameters, uint level )");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cellUpdateKernel(DataBaseStruct dataBase, Parameters parameters, uint startIndex, uint numberOfEntities)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if( index >= numberOfEntities ) return;

    cellUpdateFunction( dataBase, parameters, startIndex, index );
}

__host__ __device__ inline void cellUpdateFunction(DataBaseStruct dataBase, Parameters parameters, uint startIndex, uint index)
{
    uint cellIndex = startIndex + index;

    //////////////////////////////////////////////////////////////////////////

    real cellVolume = parameters.dx * parameters.dx * parameters.dx;

    ConservedVariables update;

    update.rho  = dataBase.dataUpdate[ RHO__(cellIndex, dataBase.numberOfCells) ] / cellVolume;
    update.rhoU = dataBase.dataUpdate[ RHO_U(cellIndex, dataBase.numberOfCells) ] / cellVolume;
    update.rhoV = dataBase.dataUpdate[ RHO_V(cellIndex, dataBase.numberOfCells) ] / cellVolume;
    update.rhoW = dataBase.dataUpdate[ RHO_W(cellIndex, dataBase.numberOfCells) ] / cellVolume;
    update.rhoE = dataBase.dataUpdate[ RHO_E(cellIndex, dataBase.numberOfCells) ] / cellVolume;

    dataBase.dataUpdate[ RHO__(cellIndex, dataBase.numberOfCells) ] = zero;
    dataBase.dataUpdate[ RHO_U(cellIndex, dataBase.numberOfCells) ] = zero;
    dataBase.dataUpdate[ RHO_V(cellIndex, dataBase.numberOfCells) ] = zero;
    dataBase.dataUpdate[ RHO_W(cellIndex, dataBase.numberOfCells) ] = zero;
    dataBase.dataUpdate[ RHO_E(cellIndex, dataBase.numberOfCells) ] = zero;

    //////////////////////////////////////////////////////////////////////////

    real rho = dataBase.data[ RHO__(cellIndex, dataBase.numberOfCells) ] + update.rho;

    Vec3 force = parameters.force;

    update.rhoU += force.x * parameters.dt * rho ;
    update.rhoV += force.y * parameters.dt * rho ;
    update.rhoW += force.z * parameters.dt * rho ;
    update.rhoE += force.x * dataBase.massFlux[ VEC_X(cellIndex, dataBase.numberOfCells) ] / ( four * parameters.dx * parameters.dx )
                 + force.y * dataBase.massFlux[ VEC_Y(cellIndex, dataBase.numberOfCells) ] / ( four * parameters.dx * parameters.dx ) 
                 + force.z * dataBase.massFlux[ VEC_Z(cellIndex, dataBase.numberOfCells) ] / ( four * parameters.dx * parameters.dx );

    dataBase.massFlux[ VEC_X(cellIndex, dataBase.numberOfCells) ] = zero;
    dataBase.massFlux[ VEC_Y(cellIndex, dataBase.numberOfCells) ] = zero;
    dataBase.massFlux[ VEC_Z(cellIndex, dataBase.numberOfCells) ] = zero;

    //////////////////////////////////////////////////////////////////////////

    dataBase.data[ RHO__(cellIndex, dataBase.numberOfCells) ] += update.rho ;
    dataBase.data[ RHO_U(cellIndex, dataBase.numberOfCells) ] += update.rhoU;
    dataBase.data[ RHO_V(cellIndex, dataBase.numberOfCells) ] += update.rhoV;
    dataBase.data[ RHO_W(cellIndex, dataBase.numberOfCells) ] += update.rhoW;
    dataBase.data[ RHO_E(cellIndex, dataBase.numberOfCells) ] += update.rhoE;

#ifdef USE_PASSIVE_SCALAR
	update.rhoS_1 = dataBase.dataUpdate[ RHO_S_1(cellIndex, dataBase.numberOfCells) ] / cellVolume;
	update.rhoS_2 = dataBase.dataUpdate[ RHO_S_2(cellIndex, dataBase.numberOfCells) ] / cellVolume;

    dataBase.dataUpdate[ RHO_S_1(cellIndex, dataBase.numberOfCells) ] = zero;
    dataBase.dataUpdate[ RHO_S_2(cellIndex, dataBase.numberOfCells) ] = zero;

    dataBase.data[ RHO_S_1(cellIndex, dataBase.numberOfCells) ] += update.rhoS_1;
    dataBase.data[ RHO_S_2(cellIndex, dataBase.numberOfCells) ] += update.rhoS_2;
#endif // USE_PASSIVE_SCALAR

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_PASSIVE_SCALAR
    if( false )
    {
	    //PrimitiveVariables updatedPrimitive;
	    //ConservedVariables updatedConserved;
	    //real initialConcentration[2];
	    //real finalConcentration[2];
	    //real temp;

     //   //////////////////////////////////////////////////////////////////////////

     //   const real molarMassFuel   = 16.04e-3;
     //   const real molarMassOxygen = 32.00e-3;
     //   const real molarMassInert  = 28.00e-3;

     //   const real Ru = 8.31445984848;

     //   //const real reactionRateCoefficient = 1.7e12;    
     //   //const real activationEnergy        = 2.037608e5;
     //   //const real heatOfReaction		   = 8.0e5;

     //   //////////////////////////////////////////////////////////////////////////

     //   // Lindberg, Hermansson, 2004 => Ansys Fluent 13

     //   real alpha = 1000.0;

     //   const real reactionRateCoefficient = 2.119e11;    
     //   const real activationEnergy        = 2.027e5;
     //   const real heatOfReaction		   = 2.0e5;

     //   const real B = 0.2;
     //   const real C = 1.3;

     //   //////////////////////////////////////////////////////////////////////////

	    //updatedConserved.rho    = dataBase.data[ RHO__(cellIndex, dataBase.numberOfCells) ];
	    //updatedConserved.rhoU   = dataBase.data[ RHO_U(cellIndex, dataBase.numberOfCells) ];
	    //updatedConserved.rhoV   = dataBase.data[ RHO_V(cellIndex, dataBase.numberOfCells) ];
	    //updatedConserved.rhoW   = dataBase.data[ RHO_W(cellIndex, dataBase.numberOfCells) ];
	    //updatedConserved.rhoE   = dataBase.data[ RHO_E(cellIndex, dataBase.numberOfCells) ];
	    //updatedConserved.rhoS_1 = dataBase.data[ RHO_S_1(cellIndex, dataBase.numberOfCells) ];
	    //updatedConserved.rhoS_2 = dataBase.data[ RHO_S_2(cellIndex, dataBase.numberOfCells) ];
    
     //   //////////////////////////////////////////////////////////////////////////

	    //initialConcentration[0] = updatedPrimitive.rho * updatedPrimitive.S_1 / molarMassFuel  ;
	    //initialConcentration[1] = updatedPrimitive.rho * updatedPrimitive.S_2 / molarMassOxygen;

     //   if( initialConcentration[0] < 0.0 ) initialConcentration[0] = 0.0;
     //   if( initialConcentration[1] < 0.0 ) initialConcentration[1] = 0.0;

     //   real R_Mixture = updatedPrimitive.S_1                                * Ru / molarMassFuel  
				 //      + updatedPrimitive.S_2                                * Ru / molarMassOxygen
		   //            + (1.0 - updatedPrimitive.S_1 - updatedPrimitive.S_2) * Ru / molarMassInert ;

	    //temp = alpha / (two * R_Mixture * updatedPrimitive.lambda);

     //   //////////////////////////////////////////////////////////////////////////

     //   real arrhenius    = exp( -activationEnergy / ( Ru * temp ) );

     //   real reactionRate = reactionRateCoefficient * arrhenius 
     //                     * pow(initialConcentration[0], B)
     //                     * pow(initialConcentration[1], C);

     //   real dt_lim_0 =       initialConcentration[0] / reactionRate;
     //   real dt_lim_1 = 0.5 * initialConcentration[1] / reactionRate;

     //   real dt_lim = fmin( dt_lim_0,      dt_lim_1 );
     //   real dt     = fmin( parameters.dt, dt_lim   );

	    //finalConcentration[0] = initialConcentration[0] -       reactionRate * dt;
	    //finalConcentration[1] = initialConcentration[1] - two * reactionRate * dt;

     //   if( finalConcentration[0] < 0.0 ) finalConcentration[0] = 0.0;
     //   if( finalConcentration[1] < 0.0 ) finalConcentration[1] = 0.0;

     //   if( finalConcentration[1] < 0.0 ) printf( "%f", finalConcentration[1] );

     //   updatedPrimitive.S_1 = finalConcentration[0] * molarMassFuel   / updatedPrimitive.rho;
	    //updatedPrimitive.S_2 = finalConcentration[1] * molarMassOxygen / updatedPrimitive.rho;

	    //updatedConserved = toConservedVariables(updatedPrimitive, parameters.K);
	
     //   //////////////////////////////////////////////////////////////////////////

	    ////updatedConserved.rhoE += reactionRate * dt
     //   //                       * parameters.dx * parameters.dx * parameters.dx
	    ////					     * heatOfReaction
	    ////					     * updatedPrimitive.rho;
	
	    //updatedConserved.rhoE += reactionRate * dt * heatOfReaction / alpha;

     //   //////////////////////////////////////////////////////////////////////////

	    //dataBase.data[ RHO__(cellIndex, dataBase.numberOfCells) ]   = updatedConserved.rho   ;
	    //dataBase.data[ RHO_U(cellIndex, dataBase.numberOfCells) ]   = updatedConserved.rhoU  ;
	    //dataBase.data[ RHO_V(cellIndex, dataBase.numberOfCells) ]   = updatedConserved.rhoV  ;
	    //dataBase.data[ RHO_W(cellIndex, dataBase.numberOfCells) ]   = updatedConserved.rhoW  ;
	    //dataBase.data[ RHO_E(cellIndex, dataBase.numberOfCells) ]   = updatedConserved.rhoE  ;
	    //dataBase.data[ RHO_S_1(cellIndex, dataBase.numberOfCells) ] = updatedConserved.rhoS_1;
	    //dataBase.data[ RHO_S_2(cellIndex, dataBase.numberOfCells) ] = updatedConserved.rhoS_2;
    }
    else if (false)
    { 
	    PrimitiveVariables updatedPrimitive;
	    ConservedVariables updatedConserved;

	    updatedConserved.rho    = dataBase.data[ RHO__(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoU   = dataBase.data[ RHO_U(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoV   = dataBase.data[ RHO_V(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoW   = dataBase.data[ RHO_W(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoE   = dataBase.data[ RHO_E(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoS_1 = dataBase.data[ RHO_S_1(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoS_2 = dataBase.data[ RHO_S_2(cellIndex, dataBase.numberOfCells) ];
	
	    updatedPrimitive = toPrimitiveVariables(updatedConserved, parameters.K);

        //////////////////////////////////////////////////////////////////////////

        real Z1 = updatedPrimitive.S_1;
        real Z2 = updatedPrimitive.S_2;

        //real Z1 = updatedConserved.rhoS_1;
        //real Z2 = updatedConserved.rhoS_2;

        real Z = Z1 + Z2;

        real Y_CH4_Inflow = real(1.0  );
        real Y_N2_ambient = real(0.767);
        real Y_O2_ambient = real(0.233);

        real M_CH4 = real(16.0e-3);  // kg / mol
        real M_O2  = real(32.0e-3);  // kg / mol
        real M_N2  = real(28.0e-3);  // kg / mol
        real M_H2O = real(18.0e-3);  // kg / mol
        real M_CO2 = real(44.0e-3);  // kg / mol

        ///////////////////////////////////////////////////////////////////////////////


        real Y_N2  = (one - Z) * Y_N2_ambient;

        real Y_CH4 = Y_CH4_Inflow * Z1;

        //           <--  non burned part -->   <------------  reacted part ------------->
        real Y_O2  = (one - Z) * Y_O2_ambient - two * ( M_O2  / M_CH4 ) * Y_CH4_Inflow * Z2;

        real Y_CO2 =                                  ( M_CO2 / M_CH4 ) * Y_CH4_Inflow * Z2;

        real Y_H2O =                            two * ( M_H2O / M_CH4 ) * Y_CH4_Inflow * Z2;

        real Y_CO;  // currently not modeled

        real Y_S;   // currently not modeled

        //////////////////////////////////////////////////////////////////////////

        //if( Y_CH4 < zero ) Y_CH4 = zero;
        //if( Y_O2  < zero ) Y_O2  = zero;

        //if( Z1 < zero && Z2 < zero ) { Z1 = zero; Z2 = zero; }

        //if( Z1 < zero ) { Z2 += Z1; Z1 = zero; }
        //if( Z2 < zero ) { Z1 += Z2; Z2 = zero; }

        {
            const real heatOfReaction = real(802310.0); // kJ / kmol

            //real s = M_CH4 / (two * M_O2);      // refers to page 49 in FDS technical reference guide

            //real releasedHeat = updatedConserved.rho * fminf(Y_CH4, s * Y_O2) * heatOfReaction / M_CH4;

            real releasedHeat = updatedConserved.rho * fminf(Y_CH4 / M_CH4, c1o2 * Y_O2 / M_O2) * heatOfReaction;

            if (Y_CH4 / M_CH4 < c1o2 * Y_O2 / M_O2) Y_CH4 = zero;
            else                                    Y_CH4 = Y_CH4 - M_CH4 / (two * M_O2) * Y_O2;

            ///////////////////////////////////////////////////////////////////////////////

            //real dZ1 = Z1 - Y_CH4 / Y_CH4_Inflow;

            Z1 = Y_CH4 / Y_CH4_Inflow;

            //Z2 = Z2 + dZ1;

            Z2 = Z - Z1;

            //if(Z2 < zero) abort();

            ///////////////////////////////////////////////////////////////////////////////

            dataBase.data[RHO_S_1(cellIndex, dataBase.numberOfCells)] = Z1 * updatedConserved.rho;
            dataBase.data[RHO_S_2(cellIndex, dataBase.numberOfCells)] = Z2 * updatedConserved.rho;

            dataBase.data[RHO_E(cellIndex, dataBase.numberOfCells)]   = updatedConserved.rhoE + releasedHeat;
        }
    }
    else if (true)
    { 
	    PrimitiveVariables updatedPrimitive;
	    ConservedVariables updatedConserved;

	    updatedConserved.rho    = dataBase.data[ RHO__(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoU   = dataBase.data[ RHO_U(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoV   = dataBase.data[ RHO_V(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoW   = dataBase.data[ RHO_W(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoE   = dataBase.data[ RHO_E(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoS_1 = dataBase.data[ RHO_S_1(cellIndex, dataBase.numberOfCells) ];
	    updatedConserved.rhoS_2 = dataBase.data[ RHO_S_2(cellIndex, dataBase.numberOfCells) ];
	
	    updatedPrimitive = toPrimitiveVariables(updatedConserved, parameters.K);

        //////////////////////////////////////////////////////////////////////////
        
        real Y_F = updatedPrimitive.S_1;
        real Y_P = updatedPrimitive.S_2;

        real Y_A = one - Y_F - Y_P;

        real M = one / ( Y_A / M_A
                       + Y_F / M_F
                       + Y_P / M_P );

        real X_A = Y_A * M / M_A;
        real X_F = Y_F * M / M_F;
        real X_P = Y_P * M / M_P;

        ///////////////////////////////////////////////////////////////////////////////

        real X_O2 = real(0.21) * X_A;

        ///////////////////////////////////////////////////////////////////////////////

        {
            real dX_F = fminf( X_F, c1o2 * X_O2 );

            //const real heatOfReaction = real(802310.0); // kJ / kmol
            const real heatOfReaction = real(80000.0); // kJ / kmol

            real dn_F = updatedConserved.rho * dX_F / M;

            real releasedHeat = dn_F * heatOfReaction;

            ///////////////////////////////////////////////////////////////////////////////

            //real X_F_new = X_F - dX_F;
            //real X_P_new = X_P + dX_F;
            
            real X_A_new = X_A - two * dX_F / real(0.21);
            real X_F_new = X_F -       dX_F;

            real X_P_new = one - X_A_new - X_F_new;

            real Z1 = X_F_new * M_F / M;
            real Z2 = X_P_new * M_P / M;

            ///////////////////////////////////////////////////////////////////////////////

            dataBase.data[RHO_S_1(cellIndex, dataBase.numberOfCells)] = Z1 * updatedConserved.rho;
            dataBase.data[RHO_S_2(cellIndex, dataBase.numberOfCells)] = Z2 * updatedConserved.rho;

            dataBase.data[RHO_E(cellIndex, dataBase.numberOfCells)]   = updatedConserved.rhoE + releasedHeat;
        }
    }

#endif // USE_PASSIVE_SCALAR
}
