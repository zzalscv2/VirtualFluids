#include "LBM/D3Q27.h"
#include "math.h"
#include "GPU/constant.h"

extern "C" __global__ void LB_Kernel_AD_Incomp_27(real diffusivity,
	unsigned int* bcMatD,
	unsigned int* neighborX,
	unsigned int* neighborY,
	unsigned int* neighborZ,
	real* DDStart,
	real* DD27,
	int size_Mat,
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

	if (k<size_Mat)
	{
		////////////////////////////////////////////////////////////////////////////////
		unsigned int BC;
		BC = bcMatD[k];

		if ((BC != GEO_SOLID) && (BC != GEO_VOID))
		{
			Distributions27 D;
			if (EvenOrOdd == true)
			{
				D.f[dirE] = &DDStart[dirE   *size_Mat];
				D.f[dirW] = &DDStart[dirW   *size_Mat];
				D.f[dirN] = &DDStart[dirN   *size_Mat];
				D.f[dirS] = &DDStart[dirS   *size_Mat];
				D.f[dirT] = &DDStart[dirT   *size_Mat];
				D.f[dirB] = &DDStart[dirB   *size_Mat];
				D.f[dirNE] = &DDStart[dirNE  *size_Mat];
				D.f[dirSW] = &DDStart[dirSW  *size_Mat];
				D.f[dirSE] = &DDStart[dirSE  *size_Mat];
				D.f[dirNW] = &DDStart[dirNW  *size_Mat];
				D.f[dirTE] = &DDStart[dirTE  *size_Mat];
				D.f[dirBW] = &DDStart[dirBW  *size_Mat];
				D.f[dirBE] = &DDStart[dirBE  *size_Mat];
				D.f[dirTW] = &DDStart[dirTW  *size_Mat];
				D.f[dirTN] = &DDStart[dirTN  *size_Mat];
				D.f[dirBS] = &DDStart[dirBS  *size_Mat];
				D.f[dirBN] = &DDStart[dirBN  *size_Mat];
				D.f[dirTS] = &DDStart[dirTS  *size_Mat];
				D.f[dirZERO] = &DDStart[dirZERO*size_Mat];
				D.f[dirTNE] = &DDStart[dirTNE *size_Mat];
				D.f[dirTSW] = &DDStart[dirTSW *size_Mat];
				D.f[dirTSE] = &DDStart[dirTSE *size_Mat];
				D.f[dirTNW] = &DDStart[dirTNW *size_Mat];
				D.f[dirBNE] = &DDStart[dirBNE *size_Mat];
				D.f[dirBSW] = &DDStart[dirBSW *size_Mat];
				D.f[dirBSE] = &DDStart[dirBSE *size_Mat];
				D.f[dirBNW] = &DDStart[dirBNW *size_Mat];
			}
			else
			{
				D.f[dirW] = &DDStart[dirE   *size_Mat];
				D.f[dirE] = &DDStart[dirW   *size_Mat];
				D.f[dirS] = &DDStart[dirN   *size_Mat];
				D.f[dirN] = &DDStart[dirS   *size_Mat];
				D.f[dirB] = &DDStart[dirT   *size_Mat];
				D.f[dirT] = &DDStart[dirB   *size_Mat];
				D.f[dirSW] = &DDStart[dirNE  *size_Mat];
				D.f[dirNE] = &DDStart[dirSW  *size_Mat];
				D.f[dirNW] = &DDStart[dirSE  *size_Mat];
				D.f[dirSE] = &DDStart[dirNW  *size_Mat];
				D.f[dirBW] = &DDStart[dirTE  *size_Mat];
				D.f[dirTE] = &DDStart[dirBW  *size_Mat];
				D.f[dirTW] = &DDStart[dirBE  *size_Mat];
				D.f[dirBE] = &DDStart[dirTW  *size_Mat];
				D.f[dirBS] = &DDStart[dirTN  *size_Mat];
				D.f[dirTN] = &DDStart[dirBS  *size_Mat];
				D.f[dirTS] = &DDStart[dirBN  *size_Mat];
				D.f[dirBN] = &DDStart[dirTS  *size_Mat];
				D.f[dirZERO] = &DDStart[dirZERO*size_Mat];
				D.f[dirBSW] = &DDStart[dirTNE *size_Mat];
				D.f[dirBNE] = &DDStart[dirTSW *size_Mat];
				D.f[dirBNW] = &DDStart[dirTSE *size_Mat];
				D.f[dirBSE] = &DDStart[dirTNW *size_Mat];
				D.f[dirTSW] = &DDStart[dirBNE *size_Mat];
				D.f[dirTNE] = &DDStart[dirBSW *size_Mat];
				D.f[dirTNW] = &DDStart[dirBSE *size_Mat];
				D.f[dirTSE] = &DDStart[dirBNW *size_Mat];
			}

			Distributions27 D27;
			if (EvenOrOdd == true)
			{
				D27.f[dirE] = &DD27[dirE   *size_Mat];
				D27.f[dirW] = &DD27[dirW   *size_Mat];
				D27.f[dirN] = &DD27[dirN   *size_Mat];
				D27.f[dirS] = &DD27[dirS   *size_Mat];
				D27.f[dirT] = &DD27[dirT   *size_Mat];
				D27.f[dirB] = &DD27[dirB   *size_Mat];
				D27.f[dirNE] = &DD27[dirNE  *size_Mat];
				D27.f[dirSW] = &DD27[dirSW  *size_Mat];
				D27.f[dirSE] = &DD27[dirSE  *size_Mat];
				D27.f[dirNW] = &DD27[dirNW  *size_Mat];
				D27.f[dirTE] = &DD27[dirTE  *size_Mat];
				D27.f[dirBW] = &DD27[dirBW  *size_Mat];
				D27.f[dirBE] = &DD27[dirBE  *size_Mat];
				D27.f[dirTW] = &DD27[dirTW  *size_Mat];
				D27.f[dirTN] = &DD27[dirTN  *size_Mat];
				D27.f[dirBS] = &DD27[dirBS  *size_Mat];
				D27.f[dirBN] = &DD27[dirBN  *size_Mat];
				D27.f[dirTS] = &DD27[dirTS  *size_Mat];
				D27.f[dirZERO] = &DD27[dirZERO*size_Mat];
				D27.f[dirTNE] = &DD27[dirTNE *size_Mat];
				D27.f[dirTSW] = &DD27[dirTSW *size_Mat];
				D27.f[dirTSE] = &DD27[dirTSE *size_Mat];
				D27.f[dirTNW] = &DD27[dirTNW *size_Mat];
				D27.f[dirBNE] = &DD27[dirBNE *size_Mat];
				D27.f[dirBSW] = &DD27[dirBSW *size_Mat];
				D27.f[dirBSE] = &DD27[dirBSE *size_Mat];
				D27.f[dirBNW] = &DD27[dirBNW *size_Mat];
			}
			else
			{
				D27.f[dirW] = &DD27[dirE   *size_Mat];
				D27.f[dirE] = &DD27[dirW   *size_Mat];
				D27.f[dirS] = &DD27[dirN   *size_Mat];
				D27.f[dirN] = &DD27[dirS   *size_Mat];
				D27.f[dirB] = &DD27[dirT   *size_Mat];
				D27.f[dirT] = &DD27[dirB   *size_Mat];
				D27.f[dirSW] = &DD27[dirNE  *size_Mat];
				D27.f[dirNE] = &DD27[dirSW  *size_Mat];
				D27.f[dirNW] = &DD27[dirSE  *size_Mat];
				D27.f[dirSE] = &DD27[dirNW  *size_Mat];
				D27.f[dirBW] = &DD27[dirTE  *size_Mat];
				D27.f[dirTE] = &DD27[dirBW  *size_Mat];
				D27.f[dirTW] = &DD27[dirBE  *size_Mat];
				D27.f[dirBE] = &DD27[dirTW  *size_Mat];
				D27.f[dirBS] = &DD27[dirTN  *size_Mat];
				D27.f[dirTN] = &DD27[dirBS  *size_Mat];
				D27.f[dirTS] = &DD27[dirBN  *size_Mat];
				D27.f[dirBN] = &DD27[dirTS  *size_Mat];
				D27.f[dirZERO] = &DD27[dirZERO*size_Mat];
				D27.f[dirBSW] = &DD27[dirTNE *size_Mat];
				D27.f[dirBNE] = &DD27[dirTSW *size_Mat];
				D27.f[dirBNW] = &DD27[dirTSE *size_Mat];
				D27.f[dirBSE] = &DD27[dirTNW *size_Mat];
				D27.f[dirTSW] = &DD27[dirBNE *size_Mat];
				D27.f[dirTNE] = &DD27[dirBSW *size_Mat];
				D27.f[dirTNW] = &DD27[dirBSE *size_Mat];
				D27.f[dirTSE] = &DD27[dirBNW *size_Mat];
			}

			////////////////////////////////////////////////////////////////////////////////
			//index
			unsigned int kw = neighborX[k];
			unsigned int ks = neighborY[k];
			unsigned int kb = neighborZ[k];
			unsigned int ksw = neighborY[kw];
			unsigned int kbw = neighborZ[kw];
			unsigned int kbs = neighborZ[ks];
			unsigned int kbsw = neighborZ[ksw];
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			real fW = (D.f[dirE])[k];//ke
			real fE = (D.f[dirW])[kw];
			real fS = (D.f[dirN])[k];//kn
			real fN = (D.f[dirS])[ks];
			real fB = (D.f[dirT])[k];//kt
			real fT = (D.f[dirB])[kb];
			real fSW = (D.f[dirNE])[k];//kne
			real fNE = (D.f[dirSW])[ksw];
			real fNW = (D.f[dirSE])[ks];//kse
			real fSE = (D.f[dirNW])[kw];//knw
			real fBW = (D.f[dirTE])[k];//kte
			real fTE = (D.f[dirBW])[kbw];
			real fTW = (D.f[dirBE])[kb];//kbe
			real fBE = (D.f[dirTW])[kw];//ktw
			real fBS = (D.f[dirTN])[k];//ktn
			real fTN = (D.f[dirBS])[kbs];
			real fTS = (D.f[dirBN])[kb];//kbn
			real fBN = (D.f[dirTS])[ks];//kts
			real fZERO = (D.f[dirZERO])[k];//kzero
			real fBSW = (D.f[dirTNE])[k];//ktne
			real fBNE = (D.f[dirTSW])[ksw];//ktsw
			real fBNW = (D.f[dirTSE])[ks];//ktse
			real fBSE = (D.f[dirTNW])[kw];//ktnw
			real fTSW = (D.f[dirBNE])[kb];//kbne
			real fTNE = (D.f[dirBSW])[kbsw];
			real fTNW = (D.f[dirBSE])[kbs];//kbse
			real fTSE = (D.f[dirBNW])[kbw];//kbnw
										   ////////////////////////////////////////////////////////////////////////////////
										   //real f27E    =  (D27.f[dirE   ])[k  ];//ke
										   //real f27W    =  (D27.f[dirW   ])[kw ];
										   //real f27N    =  (D27.f[dirN   ])[k  ];//kn
										   //real f27S    =  (D27.f[dirS   ])[ks ];
										   //real f27T    =  (D27.f[dirT   ])[k  ];//kt
										   //real f27B    =  (D27.f[dirB   ])[kb ];
										   //real f27NE   =  (D27.f[dirNE  ])[k  ];//kne
										   //real f27SW   =  (D27.f[dirSW  ])[ksw];
										   //real f27SE   =  (D27.f[dirSE  ])[ks ];//kse
										   //real f27NW   =  (D27.f[dirNW  ])[kw ];//knw
										   //real f27TE   =  (D27.f[dirTE  ])[k  ];//kte
										   //real f27BW   =  (D27.f[dirBW  ])[kbw];
										   //real f27BE   =  (D27.f[dirBE  ])[kb ];//kbe
										   //real f27TW   =  (D27.f[dirTW  ])[kw ];//ktw
										   //real f27TN   =  (D27.f[dirTN  ])[k  ];//ktn
										   //real f27BS   =  (D27.f[dirBS  ])[kbs];
										   //real f27BN   =  (D27.f[dirBN  ])[kb ];//kbn
										   //real f27TS   =  (D27.f[dirTS  ])[ks ];//kts
										   //real f27ZERO =  (D27.f[dirZERO])[k  ];//kzero
										   //real f27TNE  =  (D27.f[dirTNE ])[k  ];//ktne
										   //real f27TSW  =  (D27.f[dirTSW ])[ksw];//ktsw
										   //real f27TSE  =  (D27.f[dirTSE ])[ks ];//ktse
										   //real f27TNW  =  (D27.f[dirTNW ])[kw ];//ktnw
										   //real f27BNE  =  (D27.f[dirBNE ])[kb ];//kbne
										   //real f27BSW  =  (D27.f[dirBSW ])[kbsw];
										   //real f27BSE  =  (D27.f[dirBSE ])[kbs];//kbse
										   //real f27BNW  =  (D27.f[dirBNW ])[kbw];//kbnw
										   ////////////////////////////////////////////////////////////////////////////////
										   //real vx1     =  ((fTNE-fBSW)+(fBNE-fTSW)+(fTSE-fBNW)+(fBSE-fTNW) +(fNE-fSW)+(fSE-fNW)+(fTE-fBW)+(fBE-fTW)+(fE-fW));
										   //real vx2     =  ((fTNE-fBSW)+(fBNE-fTSW)+(fBNW-fTSE)+(fTNW-fBSE) +(fNE-fSW)+(fNW-fSE)+(fTN-fBS)+(fBN-fTS)+(fN-fS));
										   //real vx3     =  ((fTNE-fBSW)+(fTSW-fBNE)+(fTSE-fBNW)+(fTNW-fBSE) +(fTE-fBW)+(fTW-fBE)+(fTN-fBS)+(fTS-fBN)+(fT-fB));
										   ////////////////////////////////////////////////////////////////////////////////


			real mfcbb = (D27.f[dirE])[k];
			real mfabb = (D27.f[dirW])[kw];
			real mfbcb = (D27.f[dirN])[k];
			real mfbab = (D27.f[dirS])[ks];
			real mfbbc = (D27.f[dirT])[k];
			real mfbba = (D27.f[dirB])[kb];
			real mfccb = (D27.f[dirNE])[k];
			real mfaab = (D27.f[dirSW])[ksw];
			real mfcab = (D27.f[dirSE])[ks];
			real mfacb = (D27.f[dirNW])[kw];
			real mfcbc = (D27.f[dirTE])[k];
			real mfaba = (D27.f[dirBW])[kbw];
			real mfcba = (D27.f[dirBE])[kb];
			real mfabc = (D27.f[dirTW])[kw];
			real mfbcc = (D27.f[dirTN])[k];
			real mfbaa = (D27.f[dirBS])[kbs];
			real mfbca = (D27.f[dirBN])[kb];
			real mfbac = (D27.f[dirTS])[ks];
			real mfbbb = (D27.f[dirZERO])[k];
			real mfccc = (D27.f[dirTNE])[k];
			real mfaac = (D27.f[dirTSW])[ksw];
			real mfcac = (D27.f[dirTSE])[ks];
			real mfacc = (D27.f[dirTNW])[kw];
			real mfcca = (D27.f[dirBNE])[kb];
			real mfaaa = (D27.f[dirBSW])[kbsw];
			real mfcaa = (D27.f[dirBSE])[kbs];
			real mfaca = (D27.f[dirBNW])[kbw];
			////////////////////////////////////////////////////////////////////////////////////
			//Conc
			real drho = ((((mfccc + mfaaa) + (mfaca + mfcac)) + ((mfacc + mfcaa) + (mfaac + mfcca))) +
				(((mfbac + mfbca) + (mfbaa + mfbcc)) + ((mfabc + mfcba) + (mfaba + mfcbc)) + ((mfacb + mfcab) + (mfaab + mfccb))) +
				((mfabb + mfcbb) + (mfbab + mfbcb)) + (mfbba + mfbbc)) + mfbbb;
			real rho = one + drho;
			////////////////////////////////////////////////////////////////////////////////////

			real vvx = ((fTNE - fBSW) + (fBNE - fTSW) + (fTSE - fBNW) + (fBSE - fTNW) + (fNE - fSW) + (fSE - fNW) + (fTE - fBW) + (fBE - fTW) + (fE - fW));
			real vvy = ((fTNE - fBSW) + (fBNE - fTSW) + (fBNW - fTSE) + (fTNW - fBSE) + (fNE - fSW) + (fNW - fSE) + (fTN - fBS) + (fBN - fTS) + (fN - fS));
			real vvz = ((fTNE - fBSW) + (fTSW - fBNE) + (fTSE - fBNW) + (fTNW - fBSE) + (fTE - fBW) + (fTW - fBE) + (fTN - fBS) + (fTS - fBN) + (fT - fB));
			////////////////////////////////////////////////////////////////////////////////
			real omegaD = two / (six * diffusivity + one);
			////real omegaD     = -three + sqrt(three);
			////real Lam         = -(c1o2+one/omegaD);
			////real nue_d       = Lam/three;
			//real ae          = zero;
			////real ae          = diffusivity/nue_d - one;
			//real ux_sq       = vx * vx;
			//real uy_sq       = vy * vy;
			//real uz_sq       = vz * vz;


			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//D3Q7
			//real ConcD       = f7ZERO+f7E+f7W+f7N+f7S+f7T+f7B;
			//(D7.f[0])[k  ] = f7ZERO*(one+omegaD)-omegaD*ConcD*(c1o3*(ae*(-three))-(ux_sq+uy_sq+uz_sq));
			//(D7.f[2])[kw ] = f7E   *(one+omegaD)-omegaD*ConcD*(c1o6*(ae+one)+c1o2*(ux_sq)+vx*c1o2);
			//(D7.f[1])[k  ] = f7W   *(one+omegaD)-omegaD*ConcD*(c1o6*(ae+one)+c1o2*(ux_sq)-vx*c1o2);
			//(D7.f[4])[ks ] = f7N   *(one+omegaD)-omegaD*ConcD*(c1o6*(ae+one)+c1o2*(uy_sq)+vy*c1o2);
			//(D7.f[3])[k  ] = f7S   *(one+omegaD)-omegaD*ConcD*(c1o6*(ae+one)+c1o2*(uy_sq)-vy*c1o2);
			//(D7.f[6])[kb ] = f7T   *(one+omegaD)-omegaD*ConcD*(c1o6*(ae+one)+c1o2*(uz_sq)+vz*c1o2);
			//(D7.f[5])[k  ] = f7B   *(one+omegaD)-omegaD*ConcD*(c1o6*(ae+one)+c1o2*(uz_sq)-vz*c1o2);
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			////D3Q27
			//real ConcD   = (f27TNE+f27BSW)+(f27TSW+f27BNE)+(f27TSE+f27BNW)+(f27TNW+f27BSE)+
			//                  (f27NE+f27SW)+(f27NW+f27SE)+(f27TE+f27BW)+(f27BE+f27TW)+(f27TN+f27BS)+(f27BN+f27TS)+
			//                  (f27E+f27W)+(f27N+f27S)+(f27T+f27B)+f27ZERO;
			//real cusq    =  c3o2*(vx1*vx1+vx2*vx2+vx3*vx3);

			//(D27.f[ dirE   ])[k   ] = f27W    *(one-omegaD)+omegaD* c2over27* ConcD*(one+three*(-vx1        )+c9over2*(-vx1        )*(-vx1        )-cusq);                                                                     
			//(D27.f[ dirW   ])[kw  ] = f27E    *(one-omegaD)+omegaD* c2over27* ConcD*(one+three*( vx1        )+c9over2*( vx1        )*( vx1        )-cusq);                                                                     
			//(D27.f[ dirN   ])[k   ] = f27S    *(one-omegaD)+omegaD* c2over27* ConcD*(one+three*(   -vx2     )+c9over2*(    -vx2    )*(    -vx2    )-cusq);
			//(D27.f[ dirS   ])[ks  ] = f27N    *(one-omegaD)+omegaD* c2over27* ConcD*(one+three*(    vx2     )+c9over2*(     vx2    )*(     vx2    )-cusq);
			//(D27.f[ dirT   ])[k   ] = f27B    *(one-omegaD)+omegaD* c2over27* ConcD*(one+three*(        -vx3)+c9over2*(        -vx3)*(        -vx3)-cusq);
			//(D27.f[ dirB   ])[kb  ] = f27T    *(one-omegaD)+omegaD* c2over27* ConcD*(one+three*(         vx3)+c9over2*(         vx3)*(         vx3)-cusq);
			//(D27.f[ dirNE  ])[k   ] = f27SW   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*(-vx1-vx2    )+c9over2*(-vx1-vx2    )*(-vx1-vx2    )-cusq);
			//(D27.f[ dirSW  ])[ksw ] = f27NE   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*( vx1+vx2    )+c9over2*( vx1+vx2    )*( vx1+vx2    )-cusq);
			//(D27.f[ dirSE  ])[ks  ] = f27NW   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*(-vx1+vx2    )+c9over2*(-vx1+vx2    )*(-vx1+vx2    )-cusq);
			//(D27.f[ dirNW  ])[kw  ] = f27SE   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*( vx1-vx2    )+c9over2*( vx1-vx2    )*( vx1-vx2    )-cusq);
			//(D27.f[ dirTE  ])[k   ] = f27BW   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*(-vx1    -vx3)+c9over2*(-vx1    -vx3)*(-vx1    -vx3)-cusq);
			//(D27.f[ dirBW  ])[kbw ] = f27TE   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*( vx1    +vx3)+c9over2*( vx1    +vx3)*( vx1    +vx3)-cusq);
			//(D27.f[ dirBE  ])[kb  ] = f27TW   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*(-vx1    +vx3)+c9over2*(-vx1    +vx3)*(-vx1    +vx3)-cusq);
			//(D27.f[ dirTW  ])[kw  ] = f27BE   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*( vx1    -vx3)+c9over2*( vx1    -vx3)*( vx1    -vx3)-cusq);
			//(D27.f[ dirTN  ])[k   ] = f27BS   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*(    -vx2-vx3)+c9over2*(    -vx2-vx3)*(    -vx2-vx3)-cusq);
			//(D27.f[ dirBS  ])[kbs ] = f27TN   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*(     vx2+vx3)+c9over2*(     vx2+vx3)*(     vx2+vx3)-cusq);
			//(D27.f[ dirBN  ])[kb  ] = f27TS   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*(    -vx2+vx3)+c9over2*(    -vx2+vx3)*(    -vx2+vx3)-cusq);
			//(D27.f[ dirTS  ])[ks  ] = f27BN   *(one-omegaD)+omegaD* c1over54* ConcD*(one+three*(     vx2-vx3)+c9over2*(     vx2-vx3)*(     vx2-vx3)-cusq);
			//(D27.f[ dirZERO])[k   ] = f27ZERO *(one-omegaD)+omegaD* c8over27* ConcD*(one-cusq);
			//(D27.f[ dirTNE ])[k   ] = f27BSW  *(one-omegaD)+omegaD* c1over216*ConcD*(one+three*(-vx1-vx2-vx3)+c9over2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cusq);
			//(D27.f[ dirTSE ])[ks  ] = f27BNW  *(one-omegaD)+omegaD* c1over216*ConcD*(one+three*(-vx1+vx2-vx3)+c9over2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cusq);
			//(D27.f[ dirBNE ])[kb  ] = f27TSW  *(one-omegaD)+omegaD* c1over216*ConcD*(one+three*(-vx1-vx2+vx3)+c9over2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cusq);
			//(D27.f[ dirBSE ])[kbs ] = f27TNW  *(one-omegaD)+omegaD* c1over216*ConcD*(one+three*(-vx1+vx2+vx3)+c9over2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cusq);
			//(D27.f[ dirTNW ])[kw  ] = f27BSE  *(one-omegaD)+omegaD* c1over216*ConcD*(one+three*( vx1-vx2-vx3)+c9over2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cusq);
			//(D27.f[ dirTSW ])[ksw ] = f27BNE  *(one-omegaD)+omegaD* c1over216*ConcD*(one+three*( vx1+vx2-vx3)+c9over2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cusq);
			//(D27.f[ dirBNW ])[kbw ] = f27TSE  *(one-omegaD)+omegaD* c1over216*ConcD*(one+three*( vx1-vx2+vx3)+c9over2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cusq);
			//(D27.f[ dirBSW ])[kbsw] = f27TNE  *(one-omegaD)+omegaD* c1over216*ConcD*(one+three*( vx1+vx2+vx3)+c9over2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cusq);
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			real oMdrho = zero;//one; // comp special
			real m0, m1, m2;
			real vx2 = vvx*vvx;
			real vy2 = vvy*vvy;
			real vz2 = vvz*vvz;

			////////////////////////////////////////////////////////////////////////////////////
			//Hin
			////////////////////////////////////////////////////////////////////////////////////
			// mit 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36  Konditionieren
			////////////////////////////////////////////////////////////////////////////////////
			// Z - Dir
			m2 = mfaaa + mfaac;
			m1 = mfaac - mfaaa;
			m0 = m2 + mfaab;
			mfaaa = m0;
			m0 += c1o36 * oMdrho;
			mfaab = m1 - m0 * vvz;
			mfaac = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaba + mfabc;
			m1 = mfabc - mfaba;
			m0 = m2 + mfabb;
			mfaba = m0;
			m0 += c1o9 * oMdrho;
			mfabb = m1 - m0 * vvz;
			mfabc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaca + mfacc;
			m1 = mfacc - mfaca;
			m0 = m2 + mfacb;
			mfaca = m0;
			m0 += c1o36 * oMdrho;
			mfacb = m1 - m0 * vvz;
			mfacc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbaa + mfbac;
			m1 = mfbac - mfbaa;
			m0 = m2 + mfbab;
			mfbaa = m0;
			m0 += c1o9 * oMdrho;
			mfbab = m1 - m0 * vvz;
			mfbac = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbba + mfbbc;
			m1 = mfbbc - mfbba;
			m0 = m2 + mfbbb;
			mfbba = m0;
			m0 += c4o9 * oMdrho;
			mfbbb = m1 - m0 * vvz;
			mfbbc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbca + mfbcc;
			m1 = mfbcc - mfbca;
			m0 = m2 + mfbcb;
			mfbca = m0;
			m0 += c1o9 * oMdrho;
			mfbcb = m1 - m0 * vvz;
			mfbcc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcaa + mfcac;
			m1 = mfcac - mfcaa;
			m0 = m2 + mfcab;
			mfcaa = m0;
			m0 += c1o36 * oMdrho;
			mfcab = m1 - m0 * vvz;
			mfcac = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcba + mfcbc;
			m1 = mfcbc - mfcba;
			m0 = m2 + mfcbb;
			mfcba = m0;
			m0 += c1o9 * oMdrho;
			mfcbb = m1 - m0 * vvz;
			mfcbc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcca + mfccc;
			m1 = mfccc - mfcca;
			m0 = m2 + mfccb;
			mfcca = m0;
			m0 += c1o36 * oMdrho;
			mfccb = m1 - m0 * vvz;
			mfccc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			// mit  1/6, 0, 1/18, 2/3, 0, 2/9, 1/6, 0, 1/18 Konditionieren
			////////////////////////////////////////////////////////////////////////////////////
			// Y - Dir
			m2 = mfaaa + mfaca;
			m1 = mfaca - mfaaa;
			m0 = m2 + mfaba;
			mfaaa = m0;
			m0 += c1o6 * oMdrho;
			mfaba = m1 - m0 * vvy;
			mfaca = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaab + mfacb;
			m1 = mfacb - mfaab;
			m0 = m2 + mfabb;
			mfaab = m0;
			mfabb = m1 - m0 * vvy;
			mfacb = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaac + mfacc;
			m1 = mfacc - mfaac;
			m0 = m2 + mfabc;
			mfaac = m0;
			m0 += c1o18 * oMdrho;
			mfabc = m1 - m0 * vvy;
			mfacc = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbaa + mfbca;
			m1 = mfbca - mfbaa;
			m0 = m2 + mfbba;
			mfbaa = m0;
			m0 += c2o3 * oMdrho;
			mfbba = m1 - m0 * vvy;
			mfbca = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbab + mfbcb;
			m1 = mfbcb - mfbab;
			m0 = m2 + mfbbb;
			mfbab = m0;
			mfbbb = m1 - m0 * vvy;
			mfbcb = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbac + mfbcc;
			m1 = mfbcc - mfbac;
			m0 = m2 + mfbbc;
			mfbac = m0;
			m0 += c2o9 * oMdrho;
			mfbbc = m1 - m0 * vvy;
			mfbcc = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcaa + mfcca;
			m1 = mfcca - mfcaa;
			m0 = m2 + mfcba;
			mfcaa = m0;
			m0 += c1o6 * oMdrho;
			mfcba = m1 - m0 * vvy;
			mfcca = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcab + mfccb;
			m1 = mfccb - mfcab;
			m0 = m2 + mfcbb;
			mfcab = m0;
			mfcbb = m1 - m0 * vvy;
			mfccb = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcac + mfccc;
			m1 = mfccc - mfcac;
			m0 = m2 + mfcbc;
			mfcac = m0;
			m0 += c1o18 * oMdrho;
			mfcbc = m1 - m0 * vvy;
			mfccc = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			// mit     1, 0, 1/3, 0, 0, 0, 1/3, 0, 1/9		Konditionieren
			////////////////////////////////////////////////////////////////////////////////////
			// X - Dir
			m2 = mfaaa + mfcaa;
			m1 = mfcaa - mfaaa;
			m0 = m2 + mfbaa;
			mfaaa = m0;
			m0 += one* oMdrho;
			mfbaa = m1 - m0 * vvx;
			mfcaa = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaba + mfcba;
			m1 = mfcba - mfaba;
			m0 = m2 + mfbba;
			mfaba = m0;
			mfbba = m1 - m0 * vvx;
			mfcba = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaca + mfcca;
			m1 = mfcca - mfaca;
			m0 = m2 + mfbca;
			mfaca = m0;
			m0 += c1o3 * oMdrho;
			mfbca = m1 - m0 * vvx;
			mfcca = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaab + mfcab;
			m1 = mfcab - mfaab;
			m0 = m2 + mfbab;
			mfaab = m0;
			mfbab = m1 - m0 * vvx;
			mfcab = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfabb + mfcbb;
			m1 = mfcbb - mfabb;
			m0 = m2 + mfbbb;
			mfabb = m0;
			mfbbb = m1 - m0 * vvx;
			mfcbb = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfacb + mfccb;
			m1 = mfccb - mfacb;
			m0 = m2 + mfbcb;
			mfacb = m0;
			mfbcb = m1 - m0 * vvx;
			mfccb = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaac + mfcac;
			m1 = mfcac - mfaac;
			m0 = m2 + mfbac;
			mfaac = m0;
			m0 += c1o3 * oMdrho;
			mfbac = m1 - m0 * vvx;
			mfcac = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfabc + mfcbc;
			m1 = mfcbc - mfabc;
			m0 = m2 + mfbbc;
			mfabc = m0;
			mfbbc = m1 - m0 * vvx;
			mfcbc = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfacc + mfccc;
			m1 = mfccc - mfacc;
			m0 = m2 + mfbcc;
			mfacc = m0;
			m0 += c1o9 * oMdrho;
			mfbcc = m1 - m0 * vvx;
			mfccc = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////

			//if(mfaaa < zero) omegaD = one;
			real limit = nine*omegaD*omegaD*(mfbaa*mfbaa + mfaba*mfaba + mfaab*mfaab);
			//real CC=c1o2;
			//if ((two*mfaaa*mfaaa<limit)) omegaD=two / (six * (diffusivity+((limit/(1.0e-10f+two*mfaaa*mfaaa)-one)*(c1o6-diffusivity))*c1o2) + one);
			if ((two*mfaaa*mfaaa<limit)) omegaD = one;
			//omegaD = two / (six * (diffusivity+CC*limit) + one);

			//mfaaa = c1o2;
			//trans 3.
			real Mabc = mfabc - mfaba*c1o3;
			real Mbca = mfbca - mfbaa*c1o3;
			real Macb = mfacb - mfaab*c1o3;
			real Mcba = mfcba - mfaba*c1o3;
			real Mcab = mfcab - mfaab*c1o3;
			real Mbac = mfbac - mfbaa*c1o3;
			//trans 5.
			real Mcbc = mfcbc - mfaba*c1o9;
			real Mbcc = mfbcc - mfbaa*c1o9;
			real Mccb = mfccb - mfaab*c1o9;

			//1.
			mfbaa *= one - omegaD;
			mfaba *= one - omegaD;
			mfaab *= one - omegaD;

			//3.
			//mfbca *= one - omegaD;
			//mfbac *= one - omegaD;
			//mfcba *= one - omegaD;
			//mfabc *= one - omegaD;
			//mfcab *= one - omegaD;
			//mfacb *= one - omegaD;

			//mfbbb *= one - omegaD; 
			Mabc = zero;
			Mbca = zero;
			Macb = zero;
			Mcba = zero;
			Mcab = zero;
			Mbac = zero;
			mfbbb = zero;

			//5.
			//mfbcc *= one - omegaD;
			//mfcbc *= one - omegaD;
			//mfccb *= one - omegaD;
			Mcbc = zero;
			Mbcc = zero;
			Mccb = zero;

			//2.
			mfbba = zero;
			mfbab = zero;
			mfabb = zero;

			mfcaa = c1o3 * drho;
			mfaca = c1o3 * drho;
			mfaac = c1o3 * drho;

			//4.
			mfacc = c1o9 * drho;
			mfcac = c1o9 * drho;
			mfcca = c1o9 * drho;

			mfcbb = zero;
			mfbcb = zero;
			mfbbc = zero;

			//6.
			mfccc = c1o27 * drho;

			//3.
			mfabc = Mabc + mfaba*c1o3;
			mfbca = Mbca + mfbaa*c1o3;
			mfacb = Macb + mfaab*c1o3;
			mfcba = Mcba + mfaba*c1o3;
			mfcab = Mcab + mfaab*c1o3;
			mfbac = Mbac + mfbaa*c1o3;
			//5.	  
			mfcbc = Mcbc + mfaba*c1o9;
			mfbcc = Mbcc + mfbaa*c1o9;
			mfccb = Mccb + mfaab*c1o9;

			////////////////////////////////////////////////////////////////////////////////////
			//back
			////////////////////////////////////////////////////////////////////////////////////
			//mit 1, 0, 1/3, 0, 0, 0, 1/3, 0, 1/9   Konditionieren
			////////////////////////////////////////////////////////////////////////////////////
			// Z - Dir
			m0 = mfaac * c1o2 + mfaab * (vvz - c1o2) + (mfaaa + one* oMdrho) * (vz2 - vvz) * c1o2;
			m1 = -mfaac - two* mfaab *  vvz + mfaaa                * (one - vz2) - one* oMdrho * vz2;
			m2 = mfaac * c1o2 + mfaab * (vvz + c1o2) + (mfaaa + one* oMdrho) * (vz2 + vvz) * c1o2;
			mfaaa = m0;
			mfaab = m1;
			mfaac = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfabc * c1o2 + mfabb * (vvz - c1o2) + mfaba * (vz2 - vvz) * c1o2;
			m1 = -mfabc - two* mfabb *  vvz + mfaba * (one - vz2);
			m2 = mfabc * c1o2 + mfabb * (vvz + c1o2) + mfaba * (vz2 + vvz) * c1o2;
			mfaba = m0;
			mfabb = m1;
			mfabc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfacc * c1o2 + mfacb * (vvz - c1o2) + (mfaca + c1o3 * oMdrho) * (vz2 - vvz) * c1o2;
			m1 = -mfacc - two* mfacb *  vvz + mfaca                  * (one - vz2) - c1o3 * oMdrho * vz2;
			m2 = mfacc * c1o2 + mfacb * (vvz + c1o2) + (mfaca + c1o3 * oMdrho) * (vz2 + vvz) * c1o2;
			mfaca = m0;
			mfacb = m1;
			mfacc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfbac * c1o2 + mfbab * (vvz - c1o2) + mfbaa * (vz2 - vvz) * c1o2;
			m1 = -mfbac - two* mfbab *  vvz + mfbaa * (one - vz2);
			m2 = mfbac * c1o2 + mfbab * (vvz + c1o2) + mfbaa * (vz2 + vvz) * c1o2;
			mfbaa = m0;
			mfbab = m1;
			mfbac = m2;
			/////////b//////////////////////////////////////////////////////////////////////////
			m0 = mfbbc * c1o2 + mfbbb * (vvz - c1o2) + mfbba * (vz2 - vvz) * c1o2;
			m1 = -mfbbc - two* mfbbb *  vvz + mfbba * (one - vz2);
			m2 = mfbbc * c1o2 + mfbbb * (vvz + c1o2) + mfbba * (vz2 + vvz) * c1o2;
			mfbba = m0;
			mfbbb = m1;
			mfbbc = m2;
			/////////b//////////////////////////////////////////////////////////////////////////
			m0 = mfbcc * c1o2 + mfbcb * (vvz - c1o2) + mfbca * (vz2 - vvz) * c1o2;
			m1 = -mfbcc - two* mfbcb *  vvz + mfbca * (one - vz2);
			m2 = mfbcc * c1o2 + mfbcb * (vvz + c1o2) + mfbca * (vz2 + vvz) * c1o2;
			mfbca = m0;
			mfbcb = m1;
			mfbcc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcac * c1o2 + mfcab * (vvz - c1o2) + (mfcaa + c1o3 * oMdrho) * (vz2 - vvz) * c1o2;
			m1 = -mfcac - two* mfcab *  vvz + mfcaa                  * (one - vz2) - c1o3 * oMdrho * vz2;
			m2 = mfcac * c1o2 + mfcab * (vvz + c1o2) + (mfcaa + c1o3 * oMdrho) * (vz2 + vvz) * c1o2;
			mfcaa = m0;
			mfcab = m1;
			mfcac = m2;
			/////////c//////////////////////////////////////////////////////////////////////////
			m0 = mfcbc * c1o2 + mfcbb * (vvz - c1o2) + mfcba * (vz2 - vvz) * c1o2;
			m1 = -mfcbc - two* mfcbb *  vvz + mfcba * (one - vz2);
			m2 = mfcbc * c1o2 + mfcbb * (vvz + c1o2) + mfcba * (vz2 + vvz) * c1o2;
			mfcba = m0;
			mfcbb = m1;
			mfcbc = m2;
			/////////c//////////////////////////////////////////////////////////////////////////
			m0 = mfccc * c1o2 + mfccb * (vvz - c1o2) + (mfcca + c1o9 * oMdrho) * (vz2 - vvz) * c1o2;
			m1 = -mfccc - two* mfccb *  vvz + mfcca                  * (one - vz2) - c1o9 * oMdrho * vz2;
			m2 = mfccc * c1o2 + mfccb * (vvz + c1o2) + (mfcca + c1o9 * oMdrho) * (vz2 + vvz) * c1o2;
			mfcca = m0;
			mfccb = m1;
			mfccc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			//mit 1/6, 2/3, 1/6, 0, 0, 0, 1/18, 2/9, 1/18   Konditionieren
			////////////////////////////////////////////////////////////////////////////////////
			// Y - Dir
			m0 = mfaca * c1o2 + mfaba * (vvy - c1o2) + (mfaaa + c1o6 * oMdrho) * (vy2 - vvy) * c1o2;
			m1 = -mfaca - two* mfaba *  vvy + mfaaa                  * (one - vy2) - c1o6 * oMdrho * vy2;
			m2 = mfaca * c1o2 + mfaba * (vvy + c1o2) + (mfaaa + c1o6 * oMdrho) * (vy2 + vvy) * c1o2;
			mfaaa = m0;
			mfaba = m1;
			mfaca = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfacb * c1o2 + mfabb * (vvy - c1o2) + (mfaab + c2o3 * oMdrho) * (vy2 - vvy) * c1o2;
			m1 = -mfacb - two* mfabb *  vvy + mfaab                  * (one - vy2) - c2o3 * oMdrho * vy2;
			m2 = mfacb * c1o2 + mfabb * (vvy + c1o2) + (mfaab + c2o3 * oMdrho) * (vy2 + vvy) * c1o2;
			mfaab = m0;
			mfabb = m1;
			mfacb = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfacc * c1o2 + mfabc * (vvy - c1o2) + (mfaac + c1o6 * oMdrho) * (vy2 - vvy) * c1o2;
			m1 = -mfacc - two* mfabc *  vvy + mfaac                  * (one - vy2) - c1o6 * oMdrho * vy2;
			m2 = mfacc * c1o2 + mfabc * (vvy + c1o2) + (mfaac + c1o6 * oMdrho) * (vy2 + vvy) * c1o2;
			mfaac = m0;
			mfabc = m1;
			mfacc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfbca * c1o2 + mfbba * (vvy - c1o2) + mfbaa * (vy2 - vvy) * c1o2;
			m1 = -mfbca - two* mfbba *  vvy + mfbaa * (one - vy2);
			m2 = mfbca * c1o2 + mfbba * (vvy + c1o2) + mfbaa * (vy2 + vvy) * c1o2;
			mfbaa = m0;
			mfbba = m1;
			mfbca = m2;
			/////////b//////////////////////////////////////////////////////////////////////////
			m0 = mfbcb * c1o2 + mfbbb * (vvy - c1o2) + mfbab * (vy2 - vvy) * c1o2;
			m1 = -mfbcb - two* mfbbb *  vvy + mfbab * (one - vy2);
			m2 = mfbcb * c1o2 + mfbbb * (vvy + c1o2) + mfbab * (vy2 + vvy) * c1o2;
			mfbab = m0;
			mfbbb = m1;
			mfbcb = m2;
			/////////b//////////////////////////////////////////////////////////////////////////
			m0 = mfbcc * c1o2 + mfbbc * (vvy - c1o2) + mfbac * (vy2 - vvy) * c1o2;
			m1 = -mfbcc - two* mfbbc *  vvy + mfbac * (one - vy2);
			m2 = mfbcc * c1o2 + mfbbc * (vvy + c1o2) + mfbac * (vy2 + vvy) * c1o2;
			mfbac = m0;
			mfbbc = m1;
			mfbcc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcca * c1o2 + mfcba * (vvy - c1o2) + (mfcaa + c1o18 * oMdrho) * (vy2 - vvy) * c1o2;
			m1 = -mfcca - two* mfcba *  vvy + mfcaa                   * (one - vy2) - c1o18 * oMdrho * vy2;
			m2 = mfcca * c1o2 + mfcba * (vvy + c1o2) + (mfcaa + c1o18 * oMdrho) * (vy2 + vvy) * c1o2;
			mfcaa = m0;
			mfcba = m1;
			mfcca = m2;
			/////////c//////////////////////////////////////////////////////////////////////////
			m0 = mfccb * c1o2 + mfcbb * (vvy - c1o2) + (mfcab + c2o9 * oMdrho) * (vy2 - vvy) * c1o2;
			m1 = -mfccb - two* mfcbb *  vvy + mfcab                  * (one - vy2) - c2o9 * oMdrho * vy2;
			m2 = mfccb * c1o2 + mfcbb * (vvy + c1o2) + (mfcab + c2o9 * oMdrho) * (vy2 + vvy) * c1o2;
			mfcab = m0;
			mfcbb = m1;
			mfccb = m2;
			/////////c//////////////////////////////////////////////////////////////////////////
			m0 = mfccc * c1o2 + mfcbc * (vvy - c1o2) + (mfcac + c1o18 * oMdrho) * (vy2 - vvy) * c1o2;
			m1 = -mfccc - two* mfcbc *  vvy + mfcac                   * (one - vy2) - c1o18 * oMdrho * vy2;
			m2 = mfccc * c1o2 + mfcbc * (vvy + c1o2) + (mfcac + c1o18 * oMdrho) * (vy2 + vvy) * c1o2;
			mfcac = m0;
			mfcbc = m1;
			mfccc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			//mit 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36 Konditionieren
			////////////////////////////////////////////////////////////////////////////////////
			// X - Dir
			m0 = mfcaa * c1o2 + mfbaa * (vvx - c1o2) + (mfaaa + c1o36 * oMdrho) * (vx2 - vvx) * c1o2;
			m1 = -mfcaa - two* mfbaa *  vvx + mfaaa                   * (one - vx2) - c1o36 * oMdrho * vx2;
			m2 = mfcaa * c1o2 + mfbaa * (vvx + c1o2) + (mfaaa + c1o36 * oMdrho) * (vx2 + vvx) * c1o2;
			mfaaa = m0;
			mfbaa = m1;
			mfcaa = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcba * c1o2 + mfbba * (vvx - c1o2) + (mfaba + c1o9 * oMdrho) * (vx2 - vvx) * c1o2;
			m1 = -mfcba - two* mfbba *  vvx + mfaba                  * (one - vx2) - c1o9 * oMdrho * vx2;
			m2 = mfcba * c1o2 + mfbba * (vvx + c1o2) + (mfaba + c1o9 * oMdrho) * (vx2 + vvx) * c1o2;
			mfaba = m0;
			mfbba = m1;
			mfcba = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcca * c1o2 + mfbca * (vvx - c1o2) + (mfaca + c1o36 * oMdrho) * (vx2 - vvx) * c1o2;
			m1 = -mfcca - two* mfbca *  vvx + mfaca                   * (one - vx2) - c1o36 * oMdrho * vx2;
			m2 = mfcca * c1o2 + mfbca * (vvx + c1o2) + (mfaca + c1o36 * oMdrho) * (vx2 + vvx) * c1o2;
			mfaca = m0;
			mfbca = m1;
			mfcca = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcab * c1o2 + mfbab * (vvx - c1o2) + (mfaab + c1o9 * oMdrho) * (vx2 - vvx) * c1o2;
			m1 = -mfcab - two* mfbab *  vvx + mfaab                  * (one - vx2) - c1o9 * oMdrho * vx2;
			m2 = mfcab * c1o2 + mfbab * (vvx + c1o2) + (mfaab + c1o9 * oMdrho) * (vx2 + vvx) * c1o2;
			mfaab = m0;
			mfbab = m1;
			mfcab = m2;
			///////////b////////////////////////////////////////////////////////////////////////
			m0 = mfcbb * c1o2 + mfbbb * (vvx - c1o2) + (mfabb + c4o9 * oMdrho) * (vx2 - vvx) * c1o2;
			m1 = -mfcbb - two* mfbbb *  vvx + mfabb                  * (one - vx2) - c4o9 * oMdrho * vx2;
			m2 = mfcbb * c1o2 + mfbbb * (vvx + c1o2) + (mfabb + c4o9 * oMdrho) * (vx2 + vvx) * c1o2;
			mfabb = m0;
			mfbbb = m1;
			mfcbb = m2;
			///////////b////////////////////////////////////////////////////////////////////////
			m0 = mfccb * c1o2 + mfbcb * (vvx - c1o2) + (mfacb + c1o9 * oMdrho) * (vx2 - vvx) * c1o2;
			m1 = -mfccb - two* mfbcb *  vvx + mfacb                  * (one - vx2) - c1o9 * oMdrho * vx2;
			m2 = mfccb * c1o2 + mfbcb * (vvx + c1o2) + (mfacb + c1o9 * oMdrho) * (vx2 + vvx) * c1o2;
			mfacb = m0;
			mfbcb = m1;
			mfccb = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcac * c1o2 + mfbac * (vvx - c1o2) + (mfaac + c1o36 * oMdrho) * (vx2 - vvx) * c1o2;
			m1 = -mfcac - two* mfbac *  vvx + mfaac                   * (one - vx2) - c1o36 * oMdrho * vx2;
			m2 = mfcac * c1o2 + mfbac * (vvx + c1o2) + (mfaac + c1o36 * oMdrho) * (vx2 + vvx) * c1o2;
			mfaac = m0;
			mfbac = m1;
			mfcac = m2;
			///////////c////////////////////////////////////////////////////////////////////////
			m0 = mfcbc * c1o2 + mfbbc * (vvx - c1o2) + (mfabc + c1o9 * oMdrho) * (vx2 - vvx) * c1o2;
			m1 = -mfcbc - two* mfbbc *  vvx + mfabc                  * (one - vx2) - c1o9 * oMdrho * vx2;
			m2 = mfcbc * c1o2 + mfbbc * (vvx + c1o2) + (mfabc + c1o9 * oMdrho) * (vx2 + vvx) * c1o2;
			mfabc = m0;
			mfbbc = m1;
			mfcbc = m2;
			///////////c////////////////////////////////////////////////////////////////////////
			m0 = mfccc * c1o2 + mfbcc * (vvx - c1o2) + (mfacc + c1o36 * oMdrho) * (vx2 - vvx) * c1o2;
			m1 = -mfccc - two* mfbcc *  vvx + mfacc                   * (one - vx2) - c1o36 * oMdrho * vx2;
			m2 = mfccc * c1o2 + mfbcc * (vvx + c1o2) + (mfacc + c1o36 * oMdrho) * (vx2 + vvx) * c1o2;
			mfacc = m0;
			mfbcc = m1;
			mfccc = m2;
			////////////////////////////////////////////////////////////////////////////////////


			////////////////////////////////////////////////////////////////////////////////////
			(D27.f[dirE])[k] = mfabb;
			(D27.f[dirW])[kw] = mfcbb;
			(D27.f[dirN])[k] = mfbab;
			(D27.f[dirS])[ks] = mfbcb;
			(D27.f[dirT])[k] = mfbba;
			(D27.f[dirB])[kb] = mfbbc;
			(D27.f[dirNE])[k] = mfaab;
			(D27.f[dirSW])[ksw] = mfccb;
			(D27.f[dirSE])[ks] = mfacb;
			(D27.f[dirNW])[kw] = mfcab;
			(D27.f[dirTE])[k] = mfaba;
			(D27.f[dirBW])[kbw] = mfcbc;
			(D27.f[dirBE])[kb] = mfabc;
			(D27.f[dirTW])[kw] = mfcba;
			(D27.f[dirTN])[k] = mfbaa;
			(D27.f[dirBS])[kbs] = mfbcc;
			(D27.f[dirBN])[kb] = mfbac;
			(D27.f[dirTS])[ks] = mfbca;
			(D27.f[dirZERO])[k] = mfbbb;
			(D27.f[dirTNE])[k] = mfaaa;
			(D27.f[dirTSE])[ks] = mfaca;
			(D27.f[dirBNE])[kb] = mfaac;
			(D27.f[dirBSE])[kbs] = mfacc;
			(D27.f[dirTNW])[kw] = mfcaa;
			(D27.f[dirTSW])[ksw] = mfcca;
			(D27.f[dirBNW])[kbw] = mfcac;
			(D27.f[dirBSW])[kbsw] = mfccc;
			////////////////////////////////////////////////////////////////////////////////////

		}
	}
}