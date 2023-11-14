/* Device code */
#include "LBM/LB.h" 
#include "lbm/constants/D3Q27.h"
#include <basics/constants/NumericConstants.h>

using namespace vf::basics::constant;
using namespace vf::lbm::dir;

////////////////////////////////////////////////////////////////////////////////
__global__ void LBInit27( int myid,
                                     int numprocs,
                                     real u0,
                                     unsigned int* geoD,
                                     unsigned int* neighborX,
                                     unsigned int* neighborY,
                                     unsigned int* neighborZ,
                                     real* vParabel,
                                     unsigned long long numberOfLBnodes,
                                     unsigned int grid_nx, 
                                     unsigned int grid_ny, 
                                     unsigned int grid_nz, 
                                     real* DD,
                                     int lev,
                                     int maxlev)
{
   Distributions27 D;
   D.f[dP00] = &DD[dP00 * numberOfLBnodes];
   D.f[dM00] = &DD[dM00 * numberOfLBnodes];
   D.f[d0P0] = &DD[d0P0 * numberOfLBnodes];
   D.f[d0M0] = &DD[d0M0 * numberOfLBnodes];
   D.f[d00P] = &DD[d00P * numberOfLBnodes];
   D.f[d00M] = &DD[d00M * numberOfLBnodes];
   D.f[dPP0] = &DD[dPP0 * numberOfLBnodes];
   D.f[dMM0] = &DD[dMM0 * numberOfLBnodes];
   D.f[dPM0] = &DD[dPM0 * numberOfLBnodes];
   D.f[dMP0] = &DD[dMP0 * numberOfLBnodes];
   D.f[dP0P] = &DD[dP0P * numberOfLBnodes];
   D.f[dM0M] = &DD[dM0M * numberOfLBnodes];
   D.f[dP0M] = &DD[dP0M * numberOfLBnodes];
   D.f[dM0P] = &DD[dM0P * numberOfLBnodes];
   D.f[d0PP] = &DD[d0PP * numberOfLBnodes];
   D.f[d0MM] = &DD[d0MM * numberOfLBnodes];
   D.f[d0PM] = &DD[d0PM * numberOfLBnodes];
   D.f[d0MP] = &DD[d0MP * numberOfLBnodes];
   D.f[d000] = &DD[d000 * numberOfLBnodes];
   D.f[dPPP] = &DD[dPPP * numberOfLBnodes];
   D.f[dMMP] = &DD[dMMP * numberOfLBnodes];
   D.f[dPMP] = &DD[dPMP * numberOfLBnodes];
   D.f[dMPP] = &DD[dMPP * numberOfLBnodes];
   D.f[dPPM] = &DD[dPPM * numberOfLBnodes];
   D.f[dMMM] = &DD[dMMM * numberOfLBnodes];
   D.f[dPMM] = &DD[dPMM * numberOfLBnodes];
   D.f[dMPM] = &DD[dMPM * numberOfLBnodes];
   ////////////////////////////////////////////////////////////////////////////////
   unsigned int  k;                   // Zugriff auf arrays im device
   //
   unsigned int tx = threadIdx.x;     // Thread index = lokaler i index
   unsigned int by = blockIdx.x;      // Block index x
   unsigned int bz = blockIdx.y;      // Block index y
   unsigned int  x = tx + STARTOFFX;  // Globaler x-Index 
   unsigned int  y = by + STARTOFFY;  // Globaler y-Index 
   unsigned int  z = bz + STARTOFFZ;  // Globaler z-Index 

   const unsigned sizeX = blockDim.x;
   const unsigned sizeY = gridDim.x;
   const unsigned nx = sizeX + 2 * STARTOFFX;
   const unsigned ny = sizeY + 2 * STARTOFFY;

   k = nx*(ny*z + y) + x;
   //////////////////////////////////////////////////////////////////////////
   geoD[k] = GEO_FLUID;
   if (lev==0)
   {
      if( by == 0 || by == grid_ny-1 || tx == 0 || tx == grid_nx-1 )             
         geoD[k] = GEO_SOLID;
      else if( bz == grid_nz-1 && myid == numprocs - 1 && geoD[k] != GEO_SOLID )
         geoD[k] = GEO_PRESS;                 
      else if( bz == 0 && myid == 0 && geoD[k] != GEO_SOLID)
         geoD[k] = GEO_SOLID;//GEO_VELO;
   }
   else if (lev==maxlev-1)
   {
      unsigned int centerX = grid_nx / 2;
      unsigned int centerY = grid_ny / 2;
      unsigned int centerZ = grid_nz / 2;
      real        radius  = grid_ny / 2.56;

      unsigned int distSq = (centerX-tx)*(centerX-tx)+(centerY-by)*(centerY-by)+(centerZ-bz)*(centerZ-bz);
      real radiSq = radius*radius;

      if( distSq < radiSq)        geoD[k] = GEO_SOLID;
   }
   //////////////////////////////////////////////////////////////////////////
   real drho = c0o1;
   real  vx1 = c0o1;
   real  vx2 = c0o1;
   real  vx3 = u0;
   vParabel[k] = vx3;
   ////////////////////////////////////////////////////////////////////////////////
   //index
   unsigned int nxny = nx*ny;
   ////////////////////////////////////////////////////////////////////////////////
   //neighborX[k]      = k+1;
   //neighborY[k+1]    = k+nx+1;
   //neighborZ[k+1]    = k+nxny+1;
   //neighborY[k]      = k+nx;
   //neighborX[k+nx]   = k+nx+1;
   //neighborZ[k+nx]   = k+nx+nxny;
   //neighborZ[k]      = k+nxny;
   //neighborX[k+nxny] = k+nxny+1;
   //neighborY[k+nxny] = k+nxny+nx;
   ////////////////////////////////////////////////////////////////////////////////
   unsigned int kzero= k;
   unsigned int ke   = k;
   unsigned int kw   = k + 1;
   unsigned int kn   = k;
   unsigned int ks   = k + nx;
   unsigned int kt   = k;
   unsigned int kb   = k + nxny;
   unsigned int ksw  = k + nx + 1;
   unsigned int kne  = k;
   unsigned int kse  = k + nx;
   unsigned int knw  = k + 1;
   unsigned int kbw  = k + nxny + 1;
   unsigned int kte  = k;
   unsigned int kbe  = k + nxny;
   unsigned int ktw  = k + 1;
   unsigned int kbs  = k + nxny + nx;
   unsigned int ktn  = k;
   unsigned int kbn  = k + nxny;
   unsigned int kts  = k + nx;
   unsigned int ktse = k + nx;
   unsigned int kbnw = k + nxny + 1;
   unsigned int ktnw = k + 1;
   unsigned int kbse = k + nxny + nx;
   unsigned int ktsw = k + nx + 1;
   unsigned int kbne = k + nxny;
   unsigned int ktne = k;
   unsigned int kbsw = k + nxny + nx + 1;
   //////////////////////////////////////////////////////////////////////////

   real cu_sq=c3o2*(vx1*vx1+vx2*vx2+vx3*vx3);

   (D.f[d000])[kzero] =   c8o27* (drho-cu_sq);
   (D.f[dP00])[ke   ] =   c2o27* (drho+c3o1*( vx1        )+c9o2*( vx1        )*( vx1        )-cu_sq);
   (D.f[dM00])[kw   ] =   c2o27* (drho+c3o1*(-vx1        )+c9o2*(-vx1        )*(-vx1        )-cu_sq);
   (D.f[d0P0])[kn   ] =   c2o27* (drho+c3o1*(    vx2     )+c9o2*(     vx2    )*(     vx2    )-cu_sq);
   (D.f[d0M0])[ks   ] =   c2o27* (drho+c3o1*(   -vx2     )+c9o2*(    -vx2    )*(    -vx2    )-cu_sq);
   (D.f[d00P])[kt   ] =   c2o27* (drho+c3o1*(         vx3)+c9o2*(         vx3)*(         vx3)-cu_sq);
   (D.f[d00M])[kb   ] =   c2o27* (drho+c3o1*(        -vx3)+c9o2*(        -vx3)*(        -vx3)-cu_sq);
   (D.f[dPP0])[kne  ] =   c1o54* (drho+c3o1*( vx1+vx2    )+c9o2*( vx1+vx2    )*( vx1+vx2    )-cu_sq);
   (D.f[dMM0])[ksw  ] =   c1o54* (drho+c3o1*(-vx1-vx2    )+c9o2*(-vx1-vx2    )*(-vx1-vx2    )-cu_sq);
   (D.f[dPM0])[kse  ] =   c1o54* (drho+c3o1*( vx1-vx2    )+c9o2*( vx1-vx2    )*( vx1-vx2    )-cu_sq);
   (D.f[dMP0])[knw  ] =   c1o54* (drho+c3o1*(-vx1+vx2    )+c9o2*(-vx1+vx2    )*(-vx1+vx2    )-cu_sq);
   (D.f[dP0P])[kte  ] =   c1o54* (drho+c3o1*( vx1    +vx3)+c9o2*( vx1    +vx3)*( vx1    +vx3)-cu_sq);
   (D.f[dM0M])[kbw  ] =   c1o54* (drho+c3o1*(-vx1    -vx3)+c9o2*(-vx1    -vx3)*(-vx1    -vx3)-cu_sq);
   (D.f[dP0M])[kbe  ] =   c1o54* (drho+c3o1*( vx1    -vx3)+c9o2*( vx1    -vx3)*( vx1    -vx3)-cu_sq);
   (D.f[dM0P])[ktw  ] =   c1o54* (drho+c3o1*(-vx1    +vx3)+c9o2*(-vx1    +vx3)*(-vx1    +vx3)-cu_sq);
   (D.f[d0PP])[ktn  ] =   c1o54* (drho+c3o1*(     vx2+vx3)+c9o2*(     vx2+vx3)*(     vx2+vx3)-cu_sq);
   (D.f[d0MM])[kbs  ] =   c1o54* (drho+c3o1*(    -vx2-vx3)+c9o2*(    -vx2-vx3)*(    -vx2-vx3)-cu_sq);
   (D.f[d0PM])[kbn  ] =   c1o54* (drho+c3o1*(     vx2-vx3)+c9o2*(     vx2-vx3)*(     vx2-vx3)-cu_sq);
   (D.f[d0MP])[kts  ] =   c1o54* (drho+c3o1*(    -vx2+vx3)+c9o2*(    -vx2+vx3)*(    -vx2+vx3)-cu_sq);
   (D.f[dPPP])[ktne ] =   c1o216*(drho+c3o1*( vx1+vx2+vx3)+c9o2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cu_sq);
   (D.f[dMMM])[kbsw ] =   c1o216*(drho+c3o1*(-vx1-vx2-vx3)+c9o2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cu_sq);
   (D.f[dPPM])[kbne ] =   c1o216*(drho+c3o1*( vx1+vx2-vx3)+c9o2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cu_sq);
   (D.f[dMMP])[ktsw ] =   c1o216*(drho+c3o1*(-vx1-vx2+vx3)+c9o2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cu_sq);
   (D.f[dPMP])[ktse ] =   c1o216*(drho+c3o1*( vx1-vx2+vx3)+c9o2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cu_sq);
   (D.f[dMPM])[kbnw ] =   c1o216*(drho+c3o1*(-vx1+vx2-vx3)+c9o2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cu_sq);
   (D.f[dPMM])[kbse ] =   c1o216*(drho+c3o1*( vx1-vx2-vx3)+c9o2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cu_sq);
   (D.f[dMPP])[ktnw ] =   c1o216*(drho+c3o1*(-vx1+vx2+vx3)+c9o2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cu_sq);

}
////////////////////////////////////////////////////////////////////////////////










////////////////////////////////////////////////////////////////////////////////
__global__ void LBInitNonEqPartSP27( unsigned int* neighborX,
                                                unsigned int* neighborY,
                                                unsigned int* neighborZ,
                                                unsigned int* neighborWSB,
                                                unsigned int* geoD,
                                                real* rho,
                                                real* ux,
                                                real* uy,
                                                real* uz,
                                                unsigned long long numberOfLBnodes,
                                                real* DD,
                                                real omega,
                                                bool EvenOrOdd)
{
    ////////////////////////////////////////////////////////////////////////////////
    const unsigned  x = threadIdx.x;  // Globaler x-Index 
    const unsigned  y = blockIdx.x;   // Globaler y-Index 
    const unsigned  z = blockIdx.y;   // Globaler z-Index 
    
    const unsigned nx = blockDim.x;
    const unsigned ny = gridDim.x;
    
    const unsigned k = nx*(ny*z + y) + x;
    //////////////////////////////////////////////////////////////////////////
    
    if(k<numberOfLBnodes)
    {
        ////////////////////////////////////////////////////////////////////////////////
        unsigned int BC;
        BC = geoD[k];

        if( BC != GEO_SOLID &&  BC != GEO_VOID)
        {
            Distributions27 D;
            if (EvenOrOdd==true)
            {
                D.f[dP00] = &DD[dP00 * numberOfLBnodes];
                D.f[dM00] = &DD[dM00 * numberOfLBnodes];
                D.f[d0P0] = &DD[d0P0 * numberOfLBnodes];
                D.f[d0M0] = &DD[d0M0 * numberOfLBnodes];
                D.f[d00P] = &DD[d00P * numberOfLBnodes];
                D.f[d00M] = &DD[d00M * numberOfLBnodes];
                D.f[dPP0] = &DD[dPP0 * numberOfLBnodes];
                D.f[dMM0] = &DD[dMM0 * numberOfLBnodes];
                D.f[dPM0] = &DD[dPM0 * numberOfLBnodes];
                D.f[dMP0] = &DD[dMP0 * numberOfLBnodes];
                D.f[dP0P] = &DD[dP0P * numberOfLBnodes];
                D.f[dM0M] = &DD[dM0M * numberOfLBnodes];
                D.f[dP0M] = &DD[dP0M * numberOfLBnodes];
                D.f[dM0P] = &DD[dM0P * numberOfLBnodes];
                D.f[d0PP] = &DD[d0PP * numberOfLBnodes];
                D.f[d0MM] = &DD[d0MM * numberOfLBnodes];
                D.f[d0PM] = &DD[d0PM * numberOfLBnodes];
                D.f[d0MP] = &DD[d0MP * numberOfLBnodes];
                D.f[d000] = &DD[d000 * numberOfLBnodes];
                D.f[dPPP] = &DD[dPPP * numberOfLBnodes];
                D.f[dMMP] = &DD[dMMP * numberOfLBnodes];
                D.f[dPMP] = &DD[dPMP * numberOfLBnodes];
                D.f[dMPP] = &DD[dMPP * numberOfLBnodes];
                D.f[dPPM] = &DD[dPPM * numberOfLBnodes];
                D.f[dMMM] = &DD[dMMM * numberOfLBnodes];
                D.f[dPMM] = &DD[dPMM * numberOfLBnodes];
                D.f[dMPM] = &DD[dMPM * numberOfLBnodes];
            }
            else
            {
                D.f[dM00] = &DD[dP00 * numberOfLBnodes];
                D.f[dP00] = &DD[dM00 * numberOfLBnodes];
                D.f[d0M0] = &DD[d0P0 * numberOfLBnodes];
                D.f[d0P0] = &DD[d0M0 * numberOfLBnodes];
                D.f[d00M] = &DD[d00P * numberOfLBnodes];
                D.f[d00P] = &DD[d00M * numberOfLBnodes];
                D.f[dMM0] = &DD[dPP0 * numberOfLBnodes];
                D.f[dPP0] = &DD[dMM0 * numberOfLBnodes];
                D.f[dMP0] = &DD[dPM0 * numberOfLBnodes];
                D.f[dPM0] = &DD[dMP0 * numberOfLBnodes];
                D.f[dM0M] = &DD[dP0P * numberOfLBnodes];
                D.f[dP0P] = &DD[dM0M * numberOfLBnodes];
                D.f[dM0P] = &DD[dP0M * numberOfLBnodes];
                D.f[dP0M] = &DD[dM0P * numberOfLBnodes];
                D.f[d0MM] = &DD[d0PP * numberOfLBnodes];
                D.f[d0PP] = &DD[d0MM * numberOfLBnodes];
                D.f[d0MP] = &DD[d0PM * numberOfLBnodes];
                D.f[d0PM] = &DD[d0MP * numberOfLBnodes];
                D.f[d000] = &DD[d000 * numberOfLBnodes];
                D.f[dMMM] = &DD[dPPP * numberOfLBnodes];
                D.f[dPPM] = &DD[dMMP * numberOfLBnodes];
                D.f[dMPM] = &DD[dPMP * numberOfLBnodes];
                D.f[dPMM] = &DD[dMPP * numberOfLBnodes];
                D.f[dMMP] = &DD[dPPM * numberOfLBnodes];
                D.f[dPPP] = &DD[dMMM * numberOfLBnodes];
                D.f[dMPP] = &DD[dPMM * numberOfLBnodes];
                D.f[dPMP] = &DD[dMPM * numberOfLBnodes];
            }
            //////////////////////////////////////////////////////////////////////////
            real drho = rho[k];//0.0f;//
            real  vx1 = ux[k]; //0.0f;//
            real  vx2 = uy[k]; //0.0f;//
            real  vx3 = uz[k]; //0.0f;//
            //////////////////////////////////////////////////////////////////////////
            //index
            //////////////////////////////////////////////////////////////////////////
            unsigned int kzero= k;
            unsigned int ke   = k;
            unsigned int kw   = neighborX[k];
            unsigned int kn   = k;
            unsigned int ks   = neighborY[k];
            unsigned int kt   = k;
            unsigned int kb   = neighborZ[k];
            unsigned int ksw  = neighborY[kw];
            unsigned int kne  = k;
            unsigned int kse  = ks;
            unsigned int knw  = kw;
            unsigned int kbw  = neighborZ[kw];
            unsigned int kte  = k;
            unsigned int kbe  = kb;
            unsigned int ktw  = kw;
            unsigned int kbs  = neighborZ[ks];
            unsigned int ktn  = k;
            unsigned int kbn  = kb;
            unsigned int kts  = ks;
            unsigned int ktse = ks;
            unsigned int kbnw = kbw;
            unsigned int ktnw = kw;
            unsigned int kbse = kbs;
            unsigned int ktsw = ksw;
            unsigned int kbne = kb;
            unsigned int ktne = k;
            unsigned int kbsw = neighborZ[ksw];
            //////////////////////////////////////////////////////////////////////////////
            //neighbor index
            uint kPx   = neighborX[k];
            uint kPy   = neighborY[k];
            uint kPz   = neighborZ[k];
            uint kMxyz = neighborWSB[k];
            uint kMx   = neighborZ[neighborY[kMxyz]];
            uint kMy   = neighborZ[neighborX[kMxyz]];
            uint kMz   = neighborY[neighborX[kMxyz]];
            //////////////////////////////////////////////////////////////////////////
            //getVeloX//
            real vx1NeighborPx = ux[kPx];
            real vx1NeighborMx = ux[kMx];
            real vx1NeighborPy = ux[kPy];
            real vx1NeighborMy = ux[kMy];
            real vx1NeighborPz = ux[kPz];
            real vx1NeighborMz = ux[kMz];
            //getVeloY//
            real vx2NeighborPx = uy[kPx];
            real vx2NeighborMx = uy[kMx];
            real vx2NeighborPy = uy[kPy];
            real vx2NeighborMy = uy[kMy];
            real vx2NeighborPz = uy[kPz];
            real vx2NeighborMz = uy[kMz];
            //getVeloZ//
            real vx3NeighborPx = uz[kPx];
            real vx3NeighborMx = uz[kMx];
            real vx3NeighborPy = uz[kPy];
            real vx3NeighborMy = uz[kMy];
            real vx3NeighborPz = uz[kPz];
            real vx3NeighborMz = uz[kMz];
            //////////////////////////////////////////////////////////////////////////

            real dvx1dx = (vx1NeighborPx - vx1NeighborMx) / c2o1;
            real dvx1dy = (vx1NeighborPy - vx1NeighborMy) / c2o1;
            real dvx1dz = (vx1NeighborPz - vx1NeighborMz) / c2o1;

            real dvx2dx = (vx2NeighborPx - vx2NeighborMx) / c2o1;
            real dvx2dy = (vx2NeighborPy - vx2NeighborMy) / c2o1;
            real dvx2dz = (vx2NeighborPz - vx2NeighborMz) / c2o1;

            real dvx3dx = (vx3NeighborPx - vx3NeighborMx) / c2o1;
            real dvx3dy = (vx3NeighborPy - vx3NeighborMy) / c2o1;
            real dvx3dz = (vx3NeighborPz - vx3NeighborMz) / c2o1;

            //////////////////////////////////////////////////////////////////////////

            // the following code is copy and pasted from VirtualFluidsVisitors/InitDistributionsBlockVisitor.cpp
            // i.e. Konstantins code

            real ax = dvx1dx;
            real ay = dvx1dy;
            real az = dvx1dz;

            real bx = dvx2dx;
            real by = dvx2dy;
            real bz = dvx2dz;

            real cx = dvx3dx;
            real cy = dvx3dy;
            real cz = dvx3dz;

            real eps_new = c1o1;
            real op      = c1o1;
            real o       = omega;

            real f_E    = eps_new *((5.*ax*omega + 5.*by*o + 5.*cz*o - 8.*ax*op + 4.*by*op + 4.*cz*op)/(54.*o*op));

            real f_N    =    f_E   + eps_new *((2.*(ax - by))/(9.*o));
            real f_T    =    f_E   + eps_new *((2.*(ax - cz))/(9.*o));
            real f_NE   =            eps_new *(-(5.*cz*o + 3.*(ay + bx)*op - 2.*cz*op + ax*(5.*o + op) + by*(5.*o + op))/(54.*o*op));
            real f_SE   =    f_NE  + eps_new *((  ay + bx )/(9.*o));
            real f_TE   =            eps_new *(-(5.*cz*o + by*(5.*o - 2.*op) + 3.*(az + cx)*op + cz*op + ax*(5.*o + op))/(54.*o*op));
            real f_BE   =    f_TE  + eps_new *((  az + cx )/(9.*o));
            real f_TN   =            eps_new *(-(5.*ax*o + 5.*by*o + 5.*cz*o - 2.*ax*op + by*op + 3.*bz*op + 3.*cy*op + cz*op)/(54.*o*op));
            real f_BN   =    f_TN  + eps_new *((  bz + cy )/(9.*o));
            real f_ZERO =            eps_new *((5.*(ax + by + cz))/(9.*op));
            real f_TNE  =            eps_new *(-(ay + az + bx + bz + cx + cy)/(72.*o));
            real f_TSW  =  - f_TNE - eps_new *((ay + bx)/(36.*o));
            real f_TSE  =  - f_TNE - eps_new *((az + cx)/(36.*o));
            real f_TNW  =  - f_TNE - eps_new *((bz + cy)/(36.*o));

            //////////////////////////////////////////////////////////////////////////
            real cu_sq=c3o2*(vx1*vx1+vx2*vx2+vx3*vx3);
            
            (D.f[d000])[kzero] =   c8o27* (drho-cu_sq);
            (D.f[dP00])[ke   ] =   c2o27* (drho+c3o1*( vx1        )+c9o2*( vx1        )*( vx1        )-cu_sq);
            (D.f[dM00])[kw   ] =   c2o27* (drho+c3o1*(-vx1        )+c9o2*(-vx1        )*(-vx1        )-cu_sq);
            (D.f[d0P0])[kn   ] =   c2o27* (drho+c3o1*(    vx2     )+c9o2*(     vx2    )*(     vx2    )-cu_sq);
            (D.f[d0M0])[ks   ] =   c2o27* (drho+c3o1*(   -vx2     )+c9o2*(    -vx2    )*(    -vx2    )-cu_sq);
            (D.f[d00P])[kt   ] =   c2o27* (drho+c3o1*(         vx3)+c9o2*(         vx3)*(         vx3)-cu_sq);
            (D.f[d00M])[kb   ] =   c2o27* (drho+c3o1*(        -vx3)+c9o2*(        -vx3)*(        -vx3)-cu_sq);
            (D.f[dPP0])[kne  ] =   c1o54* (drho+c3o1*( vx1+vx2    )+c9o2*( vx1+vx2    )*( vx1+vx2    )-cu_sq);
            (D.f[dMM0])[ksw  ] =   c1o54* (drho+c3o1*(-vx1-vx2    )+c9o2*(-vx1-vx2    )*(-vx1-vx2    )-cu_sq);
            (D.f[dPM0])[kse  ] =   c1o54* (drho+c3o1*( vx1-vx2    )+c9o2*( vx1-vx2    )*( vx1-vx2    )-cu_sq);
            (D.f[dMP0])[knw  ] =   c1o54* (drho+c3o1*(-vx1+vx2    )+c9o2*(-vx1+vx2    )*(-vx1+vx2    )-cu_sq);
            (D.f[dP0P])[kte  ] =   c1o54* (drho+c3o1*( vx1    +vx3)+c9o2*( vx1    +vx3)*( vx1    +vx3)-cu_sq);
            (D.f[dM0M])[kbw  ] =   c1o54* (drho+c3o1*(-vx1    -vx3)+c9o2*(-vx1    -vx3)*(-vx1    -vx3)-cu_sq);
            (D.f[dP0M])[kbe  ] =   c1o54* (drho+c3o1*( vx1    -vx3)+c9o2*( vx1    -vx3)*( vx1    -vx3)-cu_sq);
            (D.f[dM0P])[ktw  ] =   c1o54* (drho+c3o1*(-vx1    +vx3)+c9o2*(-vx1    +vx3)*(-vx1    +vx3)-cu_sq);
            (D.f[d0PP])[ktn  ] =   c1o54* (drho+c3o1*(     vx2+vx3)+c9o2*(     vx2+vx3)*(     vx2+vx3)-cu_sq);
            (D.f[d0MM])[kbs  ] =   c1o54* (drho+c3o1*(    -vx2-vx3)+c9o2*(    -vx2-vx3)*(    -vx2-vx3)-cu_sq);
            (D.f[d0PM])[kbn  ] =   c1o54* (drho+c3o1*(     vx2-vx3)+c9o2*(     vx2-vx3)*(     vx2-vx3)-cu_sq);
            (D.f[d0MP])[kts  ] =   c1o54* (drho+c3o1*(    -vx2+vx3)+c9o2*(    -vx2+vx3)*(    -vx2+vx3)-cu_sq);
            (D.f[dPPP])[ktne ] =   c1o216*(drho+c3o1*( vx1+vx2+vx3)+c9o2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cu_sq);
            (D.f[dMMM])[kbsw ] =   c1o216*(drho+c3o1*(-vx1-vx2-vx3)+c9o2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cu_sq);
            (D.f[dPPM])[kbne ] =   c1o216*(drho+c3o1*( vx1+vx2-vx3)+c9o2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cu_sq);
            (D.f[dMMP])[ktsw ] =   c1o216*(drho+c3o1*(-vx1-vx2+vx3)+c9o2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cu_sq);
            (D.f[dPMP])[ktse ] =   c1o216*(drho+c3o1*( vx1-vx2+vx3)+c9o2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cu_sq);
            (D.f[dMPM])[kbnw ] =   c1o216*(drho+c3o1*(-vx1+vx2-vx3)+c9o2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cu_sq);
            (D.f[dPMM])[kbse ] =   c1o216*(drho+c3o1*( vx1-vx2-vx3)+c9o2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cu_sq);
            (D.f[dMPP])[ktnw ] =   c1o216*(drho+c3o1*(-vx1+vx2+vx3)+c9o2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cu_sq);

            //////////////////////////////////////////////////////////////////////////

            (D.f[d000])[kzero] += f_ZERO;
            (D.f[dP00])[ke   ] += f_E   ;
            (D.f[dM00])[kw   ] += f_E   ;
            (D.f[d0P0])[kn   ] += f_N   ;
            (D.f[d0M0])[ks   ] += f_N   ;
            (D.f[d00P])[kt   ] += f_T   ;
            (D.f[d00M])[kb   ] += f_T   ;
            (D.f[dPP0])[kne  ] += f_NE  ;
            (D.f[dMM0])[ksw  ] += f_NE  ;
            (D.f[dPM0])[kse  ] += f_SE  ;
            (D.f[dMP0])[knw  ] += f_SE  ;
            (D.f[dP0P])[kte  ] += f_TE  ;
            (D.f[dM0M])[kbw  ] += f_TE  ;
            (D.f[dP0M])[kbe  ] += f_BE  ;
            (D.f[dM0P])[ktw  ] += f_BE  ;
            (D.f[d0PP])[ktn  ] += f_TN  ;
            (D.f[d0MM])[kbs  ] += f_TN  ;
            (D.f[d0PM])[kbn  ] += f_BN  ;
            (D.f[d0MP])[kts  ] += f_BN  ;
            (D.f[dPPP])[ktne ] += f_TNE ;
            (D.f[dMMM])[kbsw ] += f_TNE ;
            (D.f[dPPM])[kbne ] += f_TSW ;
            (D.f[dMMP])[ktsw ] += f_TSW ;
            (D.f[dPMP])[ktse ] += f_TSE ;
            (D.f[dMPM])[kbnw ] += f_TSE ;
            (D.f[dPMM])[kbse ] += f_TNW ;
            (D.f[dMPP])[ktnw ] += f_TNW ;

            //////////////////////////////////////////////////////////////////////////
        }
        else
        {
            //////////////////////////////////////////////////////////////////////////
            Distributions27 D;
            D.f[d000] = &DD[d000 * numberOfLBnodes];
            //////////////////////////////////////////////////////////////////////////
            (D.f[d000])[k] = c96o1;
            //////////////////////////////////////////////////////////////////////////
        }
   }
}











































