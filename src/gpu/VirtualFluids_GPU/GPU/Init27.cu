/* Device code */
#include "LBM/LB.h" 
#include "LBM/D3Q27.h"
#include "Core/RealConstants.h"

////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void LBInit27( int myid,
                                     int numprocs,
                                     real u0,
                                     unsigned int* geoD,
                                     unsigned int* neighborX,
                                     unsigned int* neighborY,
                                     unsigned int* neighborZ,
                                     real* vParabel,
                                     unsigned int size_Mat,
                                     unsigned int grid_nx, 
                                     unsigned int grid_ny, 
                                     unsigned int grid_nz, 
                                     real* DD,
                                     int lev,
                                     int maxlev)
{
   Distributions27 D;
   D.f[dirE   ] = &DD[dirE   *size_Mat];
   D.f[dirW   ] = &DD[dirW   *size_Mat];
   D.f[dirN   ] = &DD[dirN   *size_Mat];
   D.f[dirS   ] = &DD[dirS   *size_Mat];
   D.f[dirT   ] = &DD[dirT   *size_Mat];
   D.f[dirB   ] = &DD[dirB   *size_Mat];
   D.f[dirNE  ] = &DD[dirNE  *size_Mat];
   D.f[dirSW  ] = &DD[dirSW  *size_Mat];
   D.f[dirSE  ] = &DD[dirSE  *size_Mat];
   D.f[dirNW  ] = &DD[dirNW  *size_Mat];
   D.f[dirTE  ] = &DD[dirTE  *size_Mat];
   D.f[dirBW  ] = &DD[dirBW  *size_Mat];
   D.f[dirBE  ] = &DD[dirBE  *size_Mat];
   D.f[dirTW  ] = &DD[dirTW  *size_Mat];
   D.f[dirTN  ] = &DD[dirTN  *size_Mat];
   D.f[dirBS  ] = &DD[dirBS  *size_Mat];
   D.f[dirBN  ] = &DD[dirBN  *size_Mat];
   D.f[dirTS  ] = &DD[dirTS  *size_Mat];
   D.f[dirZERO] = &DD[dirZERO*size_Mat];
   D.f[dirTNE ] = &DD[dirTNE *size_Mat];
   D.f[dirTSW ] = &DD[dirTSW *size_Mat];
   D.f[dirTSE ] = &DD[dirTSE *size_Mat];
   D.f[dirTNW ] = &DD[dirTNW *size_Mat];
   D.f[dirBNE ] = &DD[dirBNE *size_Mat];
   D.f[dirBSW ] = &DD[dirBSW *size_Mat];
   D.f[dirBSE ] = &DD[dirBSE *size_Mat];
   D.f[dirBNW ] = &DD[dirBNW *size_Mat];
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

   (D.f[dirZERO])[kzero] =   c8o27* (drho-cu_sq);
   (D.f[dirE   ])[ke   ] =   c2o27* (drho+c3o1*( vx1        )+c9o2*( vx1        )*( vx1        )-cu_sq);
   (D.f[dirW   ])[kw   ] =   c2o27* (drho+c3o1*(-vx1        )+c9o2*(-vx1        )*(-vx1        )-cu_sq);
   (D.f[dirN   ])[kn   ] =   c2o27* (drho+c3o1*(    vx2     )+c9o2*(     vx2    )*(     vx2    )-cu_sq);
   (D.f[dirS   ])[ks   ] =   c2o27* (drho+c3o1*(   -vx2     )+c9o2*(    -vx2    )*(    -vx2    )-cu_sq);
   (D.f[dirT   ])[kt   ] =   c2o27* (drho+c3o1*(         vx3)+c9o2*(         vx3)*(         vx3)-cu_sq);
   (D.f[dirB   ])[kb   ] =   c2o27* (drho+c3o1*(        -vx3)+c9o2*(        -vx3)*(        -vx3)-cu_sq);
   (D.f[dirNE  ])[kne  ] =   c1o54* (drho+c3o1*( vx1+vx2    )+c9o2*( vx1+vx2    )*( vx1+vx2    )-cu_sq);
   (D.f[dirSW  ])[ksw  ] =   c1o54* (drho+c3o1*(-vx1-vx2    )+c9o2*(-vx1-vx2    )*(-vx1-vx2    )-cu_sq);
   (D.f[dirSE  ])[kse  ] =   c1o54* (drho+c3o1*( vx1-vx2    )+c9o2*( vx1-vx2    )*( vx1-vx2    )-cu_sq);
   (D.f[dirNW  ])[knw  ] =   c1o54* (drho+c3o1*(-vx1+vx2    )+c9o2*(-vx1+vx2    )*(-vx1+vx2    )-cu_sq);
   (D.f[dirTE  ])[kte  ] =   c1o54* (drho+c3o1*( vx1    +vx3)+c9o2*( vx1    +vx3)*( vx1    +vx3)-cu_sq);
   (D.f[dirBW  ])[kbw  ] =   c1o54* (drho+c3o1*(-vx1    -vx3)+c9o2*(-vx1    -vx3)*(-vx1    -vx3)-cu_sq);
   (D.f[dirBE  ])[kbe  ] =   c1o54* (drho+c3o1*( vx1    -vx3)+c9o2*( vx1    -vx3)*( vx1    -vx3)-cu_sq);
   (D.f[dirTW  ])[ktw  ] =   c1o54* (drho+c3o1*(-vx1    +vx3)+c9o2*(-vx1    +vx3)*(-vx1    +vx3)-cu_sq);
   (D.f[dirTN  ])[ktn  ] =   c1o54* (drho+c3o1*(     vx2+vx3)+c9o2*(     vx2+vx3)*(     vx2+vx3)-cu_sq);
   (D.f[dirBS  ])[kbs  ] =   c1o54* (drho+c3o1*(    -vx2-vx3)+c9o2*(    -vx2-vx3)*(    -vx2-vx3)-cu_sq);
   (D.f[dirBN  ])[kbn  ] =   c1o54* (drho+c3o1*(     vx2-vx3)+c9o2*(     vx2-vx3)*(     vx2-vx3)-cu_sq);
   (D.f[dirTS  ])[kts  ] =   c1o54* (drho+c3o1*(    -vx2+vx3)+c9o2*(    -vx2+vx3)*(    -vx2+vx3)-cu_sq);
   (D.f[dirTNE ])[ktne ] =   c1o216*(drho+c3o1*( vx1+vx2+vx3)+c9o2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cu_sq);
   (D.f[dirBSW ])[kbsw ] =   c1o216*(drho+c3o1*(-vx1-vx2-vx3)+c9o2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cu_sq);
   (D.f[dirBNE ])[kbne ] =   c1o216*(drho+c3o1*( vx1+vx2-vx3)+c9o2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cu_sq);
   (D.f[dirTSW ])[ktsw ] =   c1o216*(drho+c3o1*(-vx1-vx2+vx3)+c9o2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cu_sq);
   (D.f[dirTSE ])[ktse ] =   c1o216*(drho+c3o1*( vx1-vx2+vx3)+c9o2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cu_sq);
   (D.f[dirBNW ])[kbnw ] =   c1o216*(drho+c3o1*(-vx1+vx2-vx3)+c9o2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cu_sq);
   (D.f[dirBSE ])[kbse ] =   c1o216*(drho+c3o1*( vx1-vx2-vx3)+c9o2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cu_sq);
   (D.f[dirTNW ])[ktnw ] =   c1o216*(drho+c3o1*(-vx1+vx2+vx3)+c9o2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cu_sq);

}
////////////////////////////////////////////////////////////////////////////////










////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void LBInitNonEqPartSP27( unsigned int* neighborX,
                                                unsigned int* neighborY,
                                                unsigned int* neighborZ,
                                                unsigned int* neighborWSB,
                                                unsigned int* geoD,
                                                real* rho,
                                                real* ux,
                                                real* uy,
                                                real* uz,
                                                unsigned int size_Mat,
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
    
    if(k<size_Mat)
    {
        ////////////////////////////////////////////////////////////////////////////////
        unsigned int BC;
        BC = geoD[k];

        if( BC != GEO_SOLID &&  BC != GEO_VOID)
        {
            Distributions27 D;
            if (EvenOrOdd==true)
            {
                D.f[dirE   ] = &DD[dirE   *size_Mat];
                D.f[dirW   ] = &DD[dirW   *size_Mat];
                D.f[dirN   ] = &DD[dirN   *size_Mat];
                D.f[dirS   ] = &DD[dirS   *size_Mat];
                D.f[dirT   ] = &DD[dirT   *size_Mat];
                D.f[dirB   ] = &DD[dirB   *size_Mat];
                D.f[dirNE  ] = &DD[dirNE  *size_Mat];
                D.f[dirSW  ] = &DD[dirSW  *size_Mat];
                D.f[dirSE  ] = &DD[dirSE  *size_Mat];
                D.f[dirNW  ] = &DD[dirNW  *size_Mat];
                D.f[dirTE  ] = &DD[dirTE  *size_Mat];
                D.f[dirBW  ] = &DD[dirBW  *size_Mat];
                D.f[dirBE  ] = &DD[dirBE  *size_Mat];
                D.f[dirTW  ] = &DD[dirTW  *size_Mat];
                D.f[dirTN  ] = &DD[dirTN  *size_Mat];
                D.f[dirBS  ] = &DD[dirBS  *size_Mat];
                D.f[dirBN  ] = &DD[dirBN  *size_Mat];
                D.f[dirTS  ] = &DD[dirTS  *size_Mat];
                D.f[dirZERO] = &DD[dirZERO*size_Mat];
                D.f[dirTNE ] = &DD[dirTNE *size_Mat];
                D.f[dirTSW ] = &DD[dirTSW *size_Mat];
                D.f[dirTSE ] = &DD[dirTSE *size_Mat];
                D.f[dirTNW ] = &DD[dirTNW *size_Mat];
                D.f[dirBNE ] = &DD[dirBNE *size_Mat];
                D.f[dirBSW ] = &DD[dirBSW *size_Mat];
                D.f[dirBSE ] = &DD[dirBSE *size_Mat];
                D.f[dirBNW ] = &DD[dirBNW *size_Mat];
            }
            else
            {
                D.f[dirW   ] = &DD[dirE   *size_Mat];
                D.f[dirE   ] = &DD[dirW   *size_Mat];
                D.f[dirS   ] = &DD[dirN   *size_Mat];
                D.f[dirN   ] = &DD[dirS   *size_Mat];
                D.f[dirB   ] = &DD[dirT   *size_Mat];
                D.f[dirT   ] = &DD[dirB   *size_Mat];
                D.f[dirSW  ] = &DD[dirNE  *size_Mat];
                D.f[dirNE  ] = &DD[dirSW  *size_Mat];
                D.f[dirNW  ] = &DD[dirSE  *size_Mat];
                D.f[dirSE  ] = &DD[dirNW  *size_Mat];
                D.f[dirBW  ] = &DD[dirTE  *size_Mat];
                D.f[dirTE  ] = &DD[dirBW  *size_Mat];
                D.f[dirTW  ] = &DD[dirBE  *size_Mat];
                D.f[dirBE  ] = &DD[dirTW  *size_Mat];
                D.f[dirBS  ] = &DD[dirTN  *size_Mat];
                D.f[dirTN  ] = &DD[dirBS  *size_Mat];
                D.f[dirTS  ] = &DD[dirBN  *size_Mat];
                D.f[dirBN  ] = &DD[dirTS  *size_Mat];
                D.f[dirZERO] = &DD[dirZERO*size_Mat];
                D.f[dirBSW ] = &DD[dirTNE *size_Mat];
                D.f[dirBNE ] = &DD[dirTSW *size_Mat];
                D.f[dirBNW ] = &DD[dirTSE *size_Mat];
                D.f[dirBSE ] = &DD[dirTNW *size_Mat];
                D.f[dirTSW ] = &DD[dirBNE *size_Mat];
                D.f[dirTNE ] = &DD[dirBSW *size_Mat];
                D.f[dirTNW ] = &DD[dirBSE *size_Mat];
                D.f[dirTSE ] = &DD[dirBNW *size_Mat];
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

            // the following code is copy and pasted from VirtualFluidsCore/Visitors/InitDistributionsBlockVisitor.cpp
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
            
            (D.f[dirZERO])[kzero] =   c8o27* (drho-cu_sq);
            (D.f[dirE   ])[ke   ] =   c2o27* (drho+c3o1*( vx1        )+c9o2*( vx1        )*( vx1        )-cu_sq);
            (D.f[dirW   ])[kw   ] =   c2o27* (drho+c3o1*(-vx1        )+c9o2*(-vx1        )*(-vx1        )-cu_sq);
            (D.f[dirN   ])[kn   ] =   c2o27* (drho+c3o1*(    vx2     )+c9o2*(     vx2    )*(     vx2    )-cu_sq);
            (D.f[dirS   ])[ks   ] =   c2o27* (drho+c3o1*(   -vx2     )+c9o2*(    -vx2    )*(    -vx2    )-cu_sq);
            (D.f[dirT   ])[kt   ] =   c2o27* (drho+c3o1*(         vx3)+c9o2*(         vx3)*(         vx3)-cu_sq);
            (D.f[dirB   ])[kb   ] =   c2o27* (drho+c3o1*(        -vx3)+c9o2*(        -vx3)*(        -vx3)-cu_sq);
            (D.f[dirNE  ])[kne  ] =   c1o54* (drho+c3o1*( vx1+vx2    )+c9o2*( vx1+vx2    )*( vx1+vx2    )-cu_sq);
            (D.f[dirSW  ])[ksw  ] =   c1o54* (drho+c3o1*(-vx1-vx2    )+c9o2*(-vx1-vx2    )*(-vx1-vx2    )-cu_sq);
            (D.f[dirSE  ])[kse  ] =   c1o54* (drho+c3o1*( vx1-vx2    )+c9o2*( vx1-vx2    )*( vx1-vx2    )-cu_sq);
            (D.f[dirNW  ])[knw  ] =   c1o54* (drho+c3o1*(-vx1+vx2    )+c9o2*(-vx1+vx2    )*(-vx1+vx2    )-cu_sq);
            (D.f[dirTE  ])[kte  ] =   c1o54* (drho+c3o1*( vx1    +vx3)+c9o2*( vx1    +vx3)*( vx1    +vx3)-cu_sq);
            (D.f[dirBW  ])[kbw  ] =   c1o54* (drho+c3o1*(-vx1    -vx3)+c9o2*(-vx1    -vx3)*(-vx1    -vx3)-cu_sq);
            (D.f[dirBE  ])[kbe  ] =   c1o54* (drho+c3o1*( vx1    -vx3)+c9o2*( vx1    -vx3)*( vx1    -vx3)-cu_sq);
            (D.f[dirTW  ])[ktw  ] =   c1o54* (drho+c3o1*(-vx1    +vx3)+c9o2*(-vx1    +vx3)*(-vx1    +vx3)-cu_sq);
            (D.f[dirTN  ])[ktn  ] =   c1o54* (drho+c3o1*(     vx2+vx3)+c9o2*(     vx2+vx3)*(     vx2+vx3)-cu_sq);
            (D.f[dirBS  ])[kbs  ] =   c1o54* (drho+c3o1*(    -vx2-vx3)+c9o2*(    -vx2-vx3)*(    -vx2-vx3)-cu_sq);
            (D.f[dirBN  ])[kbn  ] =   c1o54* (drho+c3o1*(     vx2-vx3)+c9o2*(     vx2-vx3)*(     vx2-vx3)-cu_sq);
            (D.f[dirTS  ])[kts  ] =   c1o54* (drho+c3o1*(    -vx2+vx3)+c9o2*(    -vx2+vx3)*(    -vx2+vx3)-cu_sq);
            (D.f[dirTNE ])[ktne ] =   c1o216*(drho+c3o1*( vx1+vx2+vx3)+c9o2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cu_sq);
            (D.f[dirBSW ])[kbsw ] =   c1o216*(drho+c3o1*(-vx1-vx2-vx3)+c9o2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cu_sq);
            (D.f[dirBNE ])[kbne ] =   c1o216*(drho+c3o1*( vx1+vx2-vx3)+c9o2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cu_sq);
            (D.f[dirTSW ])[ktsw ] =   c1o216*(drho+c3o1*(-vx1-vx2+vx3)+c9o2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cu_sq);
            (D.f[dirTSE ])[ktse ] =   c1o216*(drho+c3o1*( vx1-vx2+vx3)+c9o2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cu_sq);
            (D.f[dirBNW ])[kbnw ] =   c1o216*(drho+c3o1*(-vx1+vx2-vx3)+c9o2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cu_sq);
            (D.f[dirBSE ])[kbse ] =   c1o216*(drho+c3o1*( vx1-vx2-vx3)+c9o2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cu_sq);
            (D.f[dirTNW ])[ktnw ] =   c1o216*(drho+c3o1*(-vx1+vx2+vx3)+c9o2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cu_sq);

            //////////////////////////////////////////////////////////////////////////

            (D.f[dirZERO])[kzero] += f_ZERO;
            (D.f[dirE   ])[ke   ] += f_E   ;
            (D.f[dirW   ])[kw   ] += f_E   ;
            (D.f[dirN   ])[kn   ] += f_N   ;
            (D.f[dirS   ])[ks   ] += f_N   ;
            (D.f[dirT   ])[kt   ] += f_T   ;
            (D.f[dirB   ])[kb   ] += f_T   ;
            (D.f[dirNE  ])[kne  ] += f_NE  ;
            (D.f[dirSW  ])[ksw  ] += f_NE  ;
            (D.f[dirSE  ])[kse  ] += f_SE  ;
            (D.f[dirNW  ])[knw  ] += f_SE  ;
            (D.f[dirTE  ])[kte  ] += f_TE  ;
            (D.f[dirBW  ])[kbw  ] += f_TE  ;
            (D.f[dirBE  ])[kbe  ] += f_BE  ;
            (D.f[dirTW  ])[ktw  ] += f_BE  ;
            (D.f[dirTN  ])[ktn  ] += f_TN  ;
            (D.f[dirBS  ])[kbs  ] += f_TN  ;
            (D.f[dirBN  ])[kbn  ] += f_BN  ;
            (D.f[dirTS  ])[kts  ] += f_BN  ;
            (D.f[dirTNE ])[ktne ] += f_TNE ;
            (D.f[dirBSW ])[kbsw ] += f_TNE ;
            (D.f[dirBNE ])[kbne ] += f_TSW ;
            (D.f[dirTSW ])[ktsw ] += f_TSW ;
            (D.f[dirTSE ])[ktse ] += f_TSE ;
            (D.f[dirBNW ])[kbnw ] += f_TSE ;
            (D.f[dirBSE ])[kbse ] += f_TNW ;
            (D.f[dirTNW ])[ktnw ] += f_TNW ;

            //////////////////////////////////////////////////////////////////////////
        }
	    else
	    {
		    //////////////////////////////////////////////////////////////////////////
		    Distributions27 D;
		    D.f[dirZERO] = &DD[dirZERO*size_Mat];
		    //////////////////////////////////////////////////////////////////////////
		    (D.f[dirZERO])[k] = c96o1;
		    //////////////////////////////////////////////////////////////////////////
	    }
   }
}





















////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void LBInitThS7( unsigned int* neighborX,
                                       unsigned int* neighborY,
                                       unsigned int* neighborZ,
                                       unsigned int* geoD,
                                       real* Conc,
                                       real* ux,
                                       real* uy,
                                       real* uz,
                                       unsigned int size_Mat,
                                       real* DD7,
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

   if(k<size_Mat)
   {
      ////////////////////////////////////////////////////////////////////////////////
      unsigned int BC;
      BC        =   geoD[k];

      if( BC != GEO_SOLID && BC != GEO_VOID)
      {
         Distributions7 D7;
         if (EvenOrOdd==true)
         {
            D7.f[0] = &DD7[0*size_Mat];
            D7.f[1] = &DD7[1*size_Mat];
            D7.f[2] = &DD7[2*size_Mat];
            D7.f[3] = &DD7[3*size_Mat];
            D7.f[4] = &DD7[4*size_Mat];
            D7.f[5] = &DD7[5*size_Mat];
            D7.f[6] = &DD7[6*size_Mat];
         }
         else
         {
            D7.f[0] = &DD7[0*size_Mat];
            D7.f[2] = &DD7[1*size_Mat];
            D7.f[1] = &DD7[2*size_Mat];
            D7.f[4] = &DD7[3*size_Mat];
            D7.f[3] = &DD7[4*size_Mat];
            D7.f[6] = &DD7[5*size_Mat];
            D7.f[5] = &DD7[6*size_Mat];
         }
         //////////////////////////////////////////////////////////////////////////
         real ConcD = Conc[k];
         real   vx1 = ux[k];
         real   vx2 = uy[k];
         real   vx3 = uz[k];
         real lambdaD     = -c3o1 + sqrt(c3o1);
         real Diffusivity = c1o20;
         real Lam         = -(c1o2+c1o1/lambdaD);
         real nue_d       = Lam/c3o1;
         real ae          = Diffusivity/nue_d - c1o1;
         real ux_sq       = vx1 * vx1;
         real uy_sq       = vx2 * vx2;
         real uz_sq       = vx3 * vx3;
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
         //////////////////////////////////////////////////////////////////////////

         (D7.f[0])[kzero] = ConcD*(c1o3*(ae*(-c3o1))-(ux_sq+uy_sq+uz_sq));
         (D7.f[1])[ke   ] = ConcD*(c1o6*(ae+c1o1)+c1o2*(ux_sq)+vx1*c1o2);
         (D7.f[2])[kw   ] = ConcD*(c1o6*(ae+c1o1)+c1o2*(ux_sq)-vx1*c1o2);
         (D7.f[3])[kn   ] = ConcD*(c1o6*(ae+c1o1)+c1o2*(uy_sq)+vx2*c1o2);
         (D7.f[4])[ks   ] = ConcD*(c1o6*(ae+c1o1)+c1o2*(uy_sq)-vx2*c1o2);
         (D7.f[5])[kt   ] = ConcD*(c1o6*(ae+c1o1)+c1o2*(uz_sq)+vx3*c1o2);
         (D7.f[6])[kb   ] = ConcD*(c1o6*(ae+c1o1)+c1o2*(uz_sq)-vx3*c1o2);
      }
   }
}












////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void LBInitThS27(unsigned int* neighborX,
                                       unsigned int* neighborY,
                                       unsigned int* neighborZ,
                                       unsigned int* geoD,
                                       real* Conc,
                                       real* ux,
                                       real* uy,
                                       real* uz,
                                       unsigned int size_Mat,
                                       real* DD27,
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

   if(k<size_Mat)
   {
      ////////////////////////////////////////////////////////////////////////////////
      unsigned int BC;
      BC        =   geoD[k];

      if( BC != GEO_SOLID && BC != GEO_VOID)
      {
         Distributions27 D27;
         if (EvenOrOdd==true)
         {
            D27.f[dirE   ] = &DD27[dirE   *size_Mat];
            D27.f[dirW   ] = &DD27[dirW   *size_Mat];
            D27.f[dirN   ] = &DD27[dirN   *size_Mat];
            D27.f[dirS   ] = &DD27[dirS   *size_Mat];
            D27.f[dirT   ] = &DD27[dirT   *size_Mat];
            D27.f[dirB   ] = &DD27[dirB   *size_Mat];
            D27.f[dirNE  ] = &DD27[dirNE  *size_Mat];
            D27.f[dirSW  ] = &DD27[dirSW  *size_Mat];
            D27.f[dirSE  ] = &DD27[dirSE  *size_Mat];
            D27.f[dirNW  ] = &DD27[dirNW  *size_Mat];
            D27.f[dirTE  ] = &DD27[dirTE  *size_Mat];
            D27.f[dirBW  ] = &DD27[dirBW  *size_Mat];
            D27.f[dirBE  ] = &DD27[dirBE  *size_Mat];
            D27.f[dirTW  ] = &DD27[dirTW  *size_Mat];
            D27.f[dirTN  ] = &DD27[dirTN  *size_Mat];
            D27.f[dirBS  ] = &DD27[dirBS  *size_Mat];
            D27.f[dirBN  ] = &DD27[dirBN  *size_Mat];
            D27.f[dirTS  ] = &DD27[dirTS  *size_Mat];
            D27.f[dirZERO] = &DD27[dirZERO*size_Mat];
            D27.f[dirTNE ] = &DD27[dirTNE *size_Mat];
            D27.f[dirTSW ] = &DD27[dirTSW *size_Mat];
            D27.f[dirTSE ] = &DD27[dirTSE *size_Mat];
            D27.f[dirTNW ] = &DD27[dirTNW *size_Mat];
            D27.f[dirBNE ] = &DD27[dirBNE *size_Mat];
            D27.f[dirBSW ] = &DD27[dirBSW *size_Mat];
            D27.f[dirBSE ] = &DD27[dirBSE *size_Mat];
            D27.f[dirBNW ] = &DD27[dirBNW *size_Mat];
         }
         else
         {
            D27.f[dirW   ] = &DD27[dirE   *size_Mat];
            D27.f[dirE   ] = &DD27[dirW   *size_Mat];
            D27.f[dirS   ] = &DD27[dirN   *size_Mat];
            D27.f[dirN   ] = &DD27[dirS   *size_Mat];
            D27.f[dirB   ] = &DD27[dirT   *size_Mat];
            D27.f[dirT   ] = &DD27[dirB   *size_Mat];
            D27.f[dirSW  ] = &DD27[dirNE  *size_Mat];
            D27.f[dirNE  ] = &DD27[dirSW  *size_Mat];
            D27.f[dirNW  ] = &DD27[dirSE  *size_Mat];
            D27.f[dirSE  ] = &DD27[dirNW  *size_Mat];
            D27.f[dirBW  ] = &DD27[dirTE  *size_Mat];
            D27.f[dirTE  ] = &DD27[dirBW  *size_Mat];
            D27.f[dirTW  ] = &DD27[dirBE  *size_Mat];
            D27.f[dirBE  ] = &DD27[dirTW  *size_Mat];
            D27.f[dirBS  ] = &DD27[dirTN  *size_Mat];
            D27.f[dirTN  ] = &DD27[dirBS  *size_Mat];
            D27.f[dirTS  ] = &DD27[dirBN  *size_Mat];
            D27.f[dirBN  ] = &DD27[dirTS  *size_Mat];
            D27.f[dirZERO] = &DD27[dirZERO*size_Mat];
            D27.f[dirBSW ] = &DD27[dirTNE *size_Mat];
            D27.f[dirBNE ] = &DD27[dirTSW *size_Mat];
            D27.f[dirBNW ] = &DD27[dirTSE *size_Mat];
            D27.f[dirBSE ] = &DD27[dirTNW *size_Mat];
            D27.f[dirTSW ] = &DD27[dirBNE *size_Mat];
            D27.f[dirTNE ] = &DD27[dirBSW *size_Mat];
            D27.f[dirTNW ] = &DD27[dirBSE *size_Mat];
            D27.f[dirTSE ] = &DD27[dirBNW *size_Mat];
         }
         //////////////////////////////////////////////////////////////////////////
         real ConcD = Conc[k];
         real   vx1 = ux[k];
         real   vx2 = uy[k];
         real   vx3 = uz[k];
         //real lambdaD     = -three + sqrt(three);
         //real Diffusivity = c1o20;
         //real Lam         = -(c1o2+one/lambdaD);
         //real nue_d       = Lam/three;
         //real ae          = Diffusivity/nue_d - one;
         //real ux_sq       = vx1 * vx1;
         //real uy_sq       = vx2 * vx2;
         //real uz_sq       = vx3 * vx3;
         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         //D3Q7
         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         //index
         //unsigned int kzero= k;
         //unsigned int ke   = k;
         //unsigned int kw   = neighborX[k];
         //unsigned int kn   = k;
         //unsigned int ks   = neighborY[k];
         //unsigned int kt   = k;
         //unsigned int kb   = neighborZ[k];
         //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         //(D7.f[0])[kzero] = ConcD*(c1o3*(ae*(-three))-(ux_sq+uy_sq+uz_sq));
         //(D7.f[1])[ke   ] = ConcD*(c1o6*(ae+one)+c1o2*(ux_sq)+vx1*c1o2);
         //(D7.f[2])[kw   ] = ConcD*(c1o6*(ae+one)+c1o2*(ux_sq)-vx1*c1o2);
         //(D7.f[3])[kn   ] = ConcD*(c1o6*(ae+one)+c1o2*(uy_sq)+vx2*c1o2);
         //(D7.f[4])[ks   ] = ConcD*(c1o6*(ae+one)+c1o2*(uy_sq)-vx2*c1o2);
         //(D7.f[5])[kt   ] = ConcD*(c1o6*(ae+one)+c1o2*(uz_sq)+vx3*c1o2);
         //(D7.f[6])[kb   ] = ConcD*(c1o6*(ae+one)+c1o2*(uz_sq)-vx3*c1o2);
         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         //D3Q27
         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         //index
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
         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
         real cu_sq=c3o2*(vx1*vx1+vx2*vx2+vx3*vx3);

         (D27.f[dirZERO])[kzero] =   c8o27* ConcD*(c1o1-cu_sq);
         (D27.f[dirE   ])[ke   ] =   c2o27* ConcD*(c1o1+c3o1*( vx1        )+c9o2*( vx1        )*( vx1        )-cu_sq);
         (D27.f[dirW   ])[kw   ] =   c2o27* ConcD*(c1o1+c3o1*(-vx1        )+c9o2*(-vx1        )*(-vx1        )-cu_sq);
         (D27.f[dirN   ])[kn   ] =   c2o27* ConcD*(c1o1+c3o1*(    vx2     )+c9o2*(     vx2    )*(     vx2    )-cu_sq);
         (D27.f[dirS   ])[ks   ] =   c2o27* ConcD*(c1o1+c3o1*(   -vx2     )+c9o2*(    -vx2    )*(    -vx2    )-cu_sq);
         (D27.f[dirT   ])[kt   ] =   c2o27* ConcD*(c1o1+c3o1*(         vx3)+c9o2*(         vx3)*(         vx3)-cu_sq);
         (D27.f[dirB   ])[kb   ] =   c2o27* ConcD*(c1o1+c3o1*(        -vx3)+c9o2*(        -vx3)*(        -vx3)-cu_sq);
         (D27.f[dirNE  ])[kne  ] =   c1o54* ConcD*(c1o1+c3o1*( vx1+vx2    )+c9o2*( vx1+vx2    )*( vx1+vx2    )-cu_sq);
         (D27.f[dirSW  ])[ksw  ] =   c1o54* ConcD*(c1o1+c3o1*(-vx1-vx2    )+c9o2*(-vx1-vx2    )*(-vx1-vx2    )-cu_sq);
         (D27.f[dirSE  ])[kse  ] =   c1o54* ConcD*(c1o1+c3o1*( vx1-vx2    )+c9o2*( vx1-vx2    )*( vx1-vx2    )-cu_sq);
         (D27.f[dirNW  ])[knw  ] =   c1o54* ConcD*(c1o1+c3o1*(-vx1+vx2    )+c9o2*(-vx1+vx2    )*(-vx1+vx2    )-cu_sq);
         (D27.f[dirTE  ])[kte  ] =   c1o54* ConcD*(c1o1+c3o1*( vx1    +vx3)+c9o2*( vx1    +vx3)*( vx1    +vx3)-cu_sq);
         (D27.f[dirBW  ])[kbw  ] =   c1o54* ConcD*(c1o1+c3o1*(-vx1    -vx3)+c9o2*(-vx1    -vx3)*(-vx1    -vx3)-cu_sq);
         (D27.f[dirBE  ])[kbe  ] =   c1o54* ConcD*(c1o1+c3o1*( vx1    -vx3)+c9o2*( vx1    -vx3)*( vx1    -vx3)-cu_sq);
         (D27.f[dirTW  ])[ktw  ] =   c1o54* ConcD*(c1o1+c3o1*(-vx1    +vx3)+c9o2*(-vx1    +vx3)*(-vx1    +vx3)-cu_sq);
         (D27.f[dirTN  ])[ktn  ] =   c1o54* ConcD*(c1o1+c3o1*(     vx2+vx3)+c9o2*(     vx2+vx3)*(     vx2+vx3)-cu_sq);
         (D27.f[dirBS  ])[kbs  ] =   c1o54* ConcD*(c1o1+c3o1*(    -vx2-vx3)+c9o2*(    -vx2-vx3)*(    -vx2-vx3)-cu_sq);
         (D27.f[dirBN  ])[kbn  ] =   c1o54* ConcD*(c1o1+c3o1*(     vx2-vx3)+c9o2*(     vx2-vx3)*(     vx2-vx3)-cu_sq);
         (D27.f[dirTS  ])[kts  ] =   c1o54* ConcD*(c1o1+c3o1*(    -vx2+vx3)+c9o2*(    -vx2+vx3)*(    -vx2+vx3)-cu_sq);
         (D27.f[dirTNE ])[ktne ] =   c1o216*ConcD*(c1o1+c3o1*( vx1+vx2+vx3)+c9o2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cu_sq);
         (D27.f[dirBSW ])[kbsw ] =   c1o216*ConcD*(c1o1+c3o1*(-vx1-vx2-vx3)+c9o2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cu_sq);
         (D27.f[dirBNE ])[kbne ] =   c1o216*ConcD*(c1o1+c3o1*( vx1+vx2-vx3)+c9o2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cu_sq);
         (D27.f[dirTSW ])[ktsw ] =   c1o216*ConcD*(c1o1+c3o1*(-vx1-vx2+vx3)+c9o2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cu_sq);
         (D27.f[dirTSE ])[ktse ] =   c1o216*ConcD*(c1o1+c3o1*( vx1-vx2+vx3)+c9o2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cu_sq);
         (D27.f[dirBNW ])[kbnw ] =   c1o216*ConcD*(c1o1+c3o1*(-vx1+vx2-vx3)+c9o2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cu_sq);
         (D27.f[dirBSE ])[kbse ] =   c1o216*ConcD*(c1o1+c3o1*( vx1-vx2-vx3)+c9o2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cu_sq);
         (D27.f[dirTNW ])[ktnw ] =   c1o216*ConcD*(c1o1+c3o1*(-vx1+vx2+vx3)+c9o2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cu_sq);
         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      }
   }
}










//test