#include "LBM/D3Q27.h"
#include "math.h"
#include "GPU/constant.h"

extern "C" __global__ void LB_Kernel_Cum_Comp_SP_27(real omega,
	unsigned int* bcMatD,
	unsigned int* neighborX,
	unsigned int* neighborY,
	unsigned int* neighborZ,
	real* DDStart,
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

			////////////////////////////////////////////////////////////////////////////////
			//index
			unsigned int kzero = k;
			unsigned int ke = k;
			unsigned int kw = neighborX[k];
			unsigned int kn = k;
			unsigned int ks = neighborY[k];
			unsigned int kt = k;
			unsigned int kb = neighborZ[k];
			unsigned int ksw = neighborY[kw];
			unsigned int kne = k;
			unsigned int kse = ks;
			unsigned int knw = kw;
			unsigned int kbw = neighborZ[kw];
			unsigned int kte = k;
			unsigned int kbe = kb;
			unsigned int ktw = kw;
			unsigned int kbs = neighborZ[ks];
			unsigned int ktn = k;
			unsigned int kbn = kb;
			unsigned int kts = ks;
			unsigned int ktse = ks;
			unsigned int kbnw = kbw;
			unsigned int ktnw = kw;
			unsigned int kbse = kbs;
			unsigned int ktsw = ksw;
			unsigned int kbne = kb;
			unsigned int ktne = k;
			unsigned int kbsw = neighborZ[ksw];
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			real E = (D.f[dirE])[ke];// +  c2over27 ;
			real W = (D.f[dirW])[kw];// +  c2over27 ;
			real N = (D.f[dirN])[kn];// +  c2over27 ;
			real S = (D.f[dirS])[ks];// +  c2over27 ;
			real F = (D.f[dirT])[kt];// +  c2over27 ;
			real B = (D.f[dirB])[kb];// +  c2over27 ;
			real Ne = (D.f[dirNE])[kne];// +  c1over54 ;
			real Sw = (D.f[dirSW])[ksw];// +  c1over54 ;
			real Se = (D.f[dirSE])[kse];// +  c1over54 ;
			real Nw = (D.f[dirNW])[knw];// +  c1over54 ;
			real Ef = (D.f[dirTE])[kte];// +  c1over54 ;
			real Wb = (D.f[dirBW])[kbw];// +  c1over54 ;
			real Eb = (D.f[dirBE])[kbe];// +  c1over54 ;
			real Wf = (D.f[dirTW])[ktw];// +  c1over54 ;
			real Nf = (D.f[dirTN])[ktn];// +  c1over54 ;
			real Sb = (D.f[dirBS])[kbs];// +  c1over54 ;
			real Nb = (D.f[dirBN])[kbn];// +  c1over54 ;
			real Sf = (D.f[dirTS])[kts];// +  c1over54 ;
			real R = (D.f[dirZERO])[kzero];// +  c8over27 ;
			real Nef = (D.f[dirTNE])[ktne];// +  c1over216;
			real Swf = (D.f[dirTSW])[ktsw];// +  c1over216;
			real Sef = (D.f[dirTSE])[ktse];// +  c1over216;
			real Nwf = (D.f[dirTNW])[ktnw];// +  c1over216;
			real Neb = (D.f[dirBNE])[kbne];// +  c1over216;
			real Swb = (D.f[dirBSW])[kbsw];// +  c1over216;
			real Seb = (D.f[dirBSE])[kbse];// +  c1over216;
			real Nwb = (D.f[dirBNW])[kbnw];// +  c1over216;
										   ////////////////////////////////////////////////////////////////////////////////////
			real fx = zero;
			real fy = zero;
			real fz = zero;
			////////////////////////////////////////////////////////////////////////////////////
			real rho = Nw + W + Sw + S + Se + E + Ne + N + R + Nf + Nb + Sf + Sb + Ef + Eb + Wf + Wb + Nwf + Nwb + Nef + Neb + Swf + Swb + Sef + Seb + F + B + one;// ACHTUNG ne EINS !!!!!!!!
			real pix = (Ne + E + Se + Ef + Eb - Nw - W - Sw - Wf - Wb + Nef + Neb + Sef + Seb - Nwf - Nwb - Swf - Swb);
			real piy = (Ne + N + Nw + Nf + Nb - Se - S - Sw - Sf - Sb + Nef + Neb + Nwf + Nwb - Sef - Seb - Swf - Swb);
			real piz = (Nf + Sf + Wf + Ef + F - Nb - Sb - Wb - Eb - B + Nef + Nwf + Sef + Swf - Neb - Nwb - Seb - Swb);
			real vvx = pix / rho + fx;
			real vvy = piy / rho + fy;
			real vvz = piz / rho + fz;
			real vx2 = vvx*vvx;
			real vy2 = vvy*vvy;
			real vz2 = vvz*vvz;
			////////////////////////////////////////////////////////////////////////////////////
			real mfaaa = Swb;
			real mfaab = Sw;
			real mfaac = Swf;
			real mfaba = Wb;
			real mfabb = W;
			real mfabc = Wf;
			real mfbaa = Sb;
			real mfbab = S;
			real mfbac = Sf;
			real mfbba = B;
			real mfbbb = R;
			real mfbbc = F;
			real mfaca = Nwb;
			real mfacb = Nw;
			real mfacc = Nwf;
			real mfcaa = Seb;
			real mfcab = Se;
			real mfcac = Sef;
			real mfcca = Neb;
			real mfccb = Ne;
			real mfccc = Nef;
			real mfbca = Nb;
			real mfbcb = N;
			real mfbcc = Nf;
			real mfcba = Eb;
			real mfcbb = E;
			real mfcbc = Ef;
			real m0, m1, m2;
			real wadjust;
			real qudricLimit = c1o100;
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
			m0 += c1o36;
			mfaab = m1 - m0 * vvz;
			mfaac = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaba + mfabc;
			m1 = mfabc - mfaba;
			m0 = m2 + mfabb;
			mfaba = m0;
			m0 += c1o9;
			mfabb = m1 - m0 * vvz;
			mfabc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaca + mfacc;
			m1 = mfacc - mfaca;
			m0 = m2 + mfacb;
			mfaca = m0;
			m0 += c1o36;
			mfacb = m1 - m0 * vvz;
			mfacc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbaa + mfbac;
			m1 = mfbac - mfbaa;
			m0 = m2 + mfbab;
			mfbaa = m0;
			m0 += c1o9;
			mfbab = m1 - m0 * vvz;
			mfbac = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbba + mfbbc;
			m1 = mfbbc - mfbba;
			m0 = m2 + mfbbb;
			mfbba = m0;
			m0 += c4o9;
			mfbbb = m1 - m0 * vvz;
			mfbbc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbca + mfbcc;
			m1 = mfbcc - mfbca;
			m0 = m2 + mfbcb;
			mfbca = m0;
			m0 += c1o9;
			mfbcb = m1 - m0 * vvz;
			mfbcc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcaa + mfcac;
			m1 = mfcac - mfcaa;
			m0 = m2 + mfcab;
			mfcaa = m0;
			m0 += c1o36;
			mfcab = m1 - m0 * vvz;
			mfcac = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcba + mfcbc;
			m1 = mfcbc - mfcba;
			m0 = m2 + mfcbb;
			mfcba = m0;
			m0 += c1o9;
			mfcbb = m1 - m0 * vvz;
			mfcbc = m2 - two*	m1 * vvz + vz2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcca + mfccc;
			m1 = mfccc - mfcca;
			m0 = m2 + mfccb;
			mfcca = m0;
			m0 += c1o36;
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
			m0 += c1o6;
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
			m0 += c1o18;
			mfabc = m1 - m0 * vvy;
			mfacc = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbaa + mfbca;
			m1 = mfbca - mfbaa;
			m0 = m2 + mfbba;
			mfbaa = m0;
			m0 += c2o3;
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
			m0 += c2o9;
			mfbbc = m1 - m0 * vvy;
			mfbcc = m2 - two*	m1 * vvy + vy2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcaa + mfcca;
			m1 = mfcca - mfcaa;
			m0 = m2 + mfcba;
			mfcaa = m0;
			m0 += c1o6;
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
			m0 += c1o18;
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
			m0 += one;
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
			m0 += c1o3;
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
			m0 += c1o3;
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
			m0 += c1o9;
			mfbcc = m1 - m0 * vvx;
			mfccc = m2 - two*	m1 * vvx + vx2 * m0;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////


			//////////////////////////////////////////////////////////////////////////////////////
			//// BGK
			//////////////////////////////////////////////////////////////////////////////////////
			////2.
			//mfabb += -(-omega) * (-mfabb);
			//mfbab += -(-omega) * (-mfbab);
			//mfbba += -(-omega) * (-mfbba);
			//
			//mfcaa += -(-omega) * (c1o3 * mfaaa - mfcaa);
			//mfaca += -(-omega) * (c1o3 * mfaaa - mfaca);
			//mfaac += -(-omega) * (c1o3 * mfaaa - mfaac);
			//
			////3.
			//mfabc += -(-omega) * (-mfabc);
			//mfbac += -(-omega) * (-mfbac);
			//
			//mfacb += -(-omega) * (-mfacb);
			//mfbca += -(-omega) * (-mfbca);

			//mfcab += -(-omega) * (-mfcab);
			//mfcba += -(-omega) * (-mfcba);

			//mfbbb += -(-omega) * (-mfbbb);

			////4.
			//mfacc += -(-omega) * (c1o9 * mfaaa - mfacc);
			//mfcac += -(-omega) * (c1o9 * mfaaa - mfcac);
			//mfcca += -(-omega) * (c1o9 * mfaaa - mfcca);

			//mfbbc += -(-omega) * (-mfbbc);
			//mfbcb += -(-omega) * (-mfbcb);
			//mfcbb += -(-omega) * (-mfcbb);

			////5.
			//mfbcc += -(-omega) * (-mfbcc);
			//mfcbc += -(-omega) * (-mfcbc);
			//mfccb += -(-omega) * (-mfccb);

			////6.
			//mfccc += -(-omega) * (c1o27 * mfaaa - mfccc);
			//////////////////////////////////////////////////////////////////////////////////////



			////////////////////////////////////////////////////////////////////////////////////
			// Cumulants
			////////////////////////////////////////////////////////////////////////////////////
			real OxxPyyPzz = one;
			real OxyyPxzz = one;//two+(-omega);//one;
			real OxyyMxzz = one;//two+(-omega);//one;
			real O4 = one;
			real O5 = one;
			real O6 = one;

			//Cum 4.
			real CUMcbb = mfcbb - ((mfcaa + c1o3 * rho) * mfabb + two* mfbba * mfbab) / rho;
			real CUMbcb = mfbcb - ((mfaca + c1o3 * rho) * mfbab + two* mfbba * mfabb) / rho;
			real CUMbbc = mfbbc - ((mfaac + c1o3 * rho) * mfbba + two* mfbab * mfabb) / rho;

			real CUMcca = mfcca - (mfcaa * mfaca + two* mfbba * mfbba) / rho - c1o3 * (mfcaa + mfaca);
			real CUMcac = mfcac - (mfcaa * mfaac + two* mfbab * mfbab) / rho - c1o3 * (mfcaa + mfaac);
			real CUMacc = mfacc - (mfaac * mfaca + two* mfabb * mfabb) / rho - c1o3 * (mfaac + mfaca);

			//Cum 5.
			real CUMbcc = mfbcc - (mfaac * mfbca + mfaca * mfbac + four* mfabb * mfbbb + two* (mfbab * mfacb + mfbba * mfabc)) / rho - c1o3 * (mfbca + mfbac);
			real CUMcbc = mfcbc - (mfaac * mfcba + mfcaa * mfabc + four* mfbab * mfbbb + two* (mfabb * mfcab + mfbba * mfbac)) / rho - c1o3 * (mfcba + mfabc);
			real CUMccb = mfccb - (mfcaa * mfacb + mfaca * mfcab + four* mfbba * mfbbb + two* (mfbab * mfbca + mfabb * mfcba)) / rho - c1o3 * (mfacb + mfcab);

			//Cum 6.
			real CUMccc = mfccc + (-four*  mfbbb * mfbbb
				- (mfcaa * mfacc + mfaca * mfcac + mfaac * mfcca)
				- four* (mfabb * mfcbb + mfbab * mfbcb + mfbba * mfbbc)
				- two* (mfbca * mfbac + mfcba * mfabc + mfcab * mfacb)) / rho
				+ (four* (mfbab * mfbab * mfaca + mfabb * mfabb * mfcaa + mfbba * mfbba * mfaac)
					+ two* (mfcaa * mfaca * mfaac)
					+ sixteen*  mfbba * mfbab * mfabb) / (rho * rho)
				- c1o3* (mfacc + mfcac + mfcca)
				+ c1o9* (mfcaa + mfaca + mfaac)
				+ (two* (mfbab * mfbab + mfabb * mfabb + mfbba * mfbba)
					+ (mfaac * mfaca + mfaac * mfcaa + mfaca * mfcaa)) * c2o3 / rho;


			//2.
			// linear combinations
			real mxxPyyPzz = mfcaa + mfaca + mfaac;
			real mxxMyy = mfcaa - mfaca;
			real mxxMzz = mfcaa - mfaac;

			//relax
			//hat noch nicht so gut funktioniert...Optimierungsbedarf
			//{
			//	real dxux = c1o2 * (-omega) *(mxxMyy + mxxMzz) +  OxxPyyPzz * (mfaaa - mxxPyyPzz);
			//	real dyuy = dxux + omega * c3o2 * mxxMyy;
			//	real dzuz = dxux + omega * c3o2 * mxxMzz;

			//	//relax
			//	mxxPyyPzz += OxxPyyPzz*(mfaaa  - mxxPyyPzz)- three * (one - c1o2 * OxxPyyPzz) * (vx2 * dxux + vy2 * dyuy + vz2 * dzuz);
			//	mxxMyy    += omega * (-mxxMyy) - three * (one + c1o2 * (-omega)) * (vx2 * dxux + vy2 * dyuy);
			//	mxxMzz    += omega * (-mxxMzz) - three * (one + c1o2 * (-omega)) * (vx2 * dxux + vz2 * dzuz);

			//	//////////////////////////////////////////////////////////////////////////
			//	//limiter-Scheise Teil 2
			//	//mxxMyy    += oxxyy * (-mxxMyy) - three * (one + c1o2 * (-omega)) * (vx2 * dxux + vy2 * dyuy);
			//	//mxxMzz    += oxxzz * (-mxxMzz) - three * (one + c1o2 * (-omega)) * (vx2 * dxux + vz2 * dzuz);
			//	//////////////////////////////////////////////////////////////////////////

			//}

			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//incl. correction		(hat noch nicht so gut funktioniert...Optimierungsbedarf??)
			{
				real dxux = c1o2 * (-omega) *(mxxMyy + mxxMzz) + c1o2 *  OxxPyyPzz * (mfaaa - mxxPyyPzz);
				real dyuy = dxux + omega * c3o2 * mxxMyy;
				real dzuz = dxux + omega * c3o2 * mxxMzz;

				//relax
				mxxPyyPzz += OxxPyyPzz*(mfaaa - mxxPyyPzz) - three * (one - c1o2 * OxxPyyPzz) * (vx2 * dxux + vy2 * dyuy + vz2 * dzuz);//-magicBulk*OxxPyyPzz;
				mxxMyy += omega * (-mxxMyy) - three * (one + c1o2 * (-omega)) * (vx2 * dxux - vy2 * dyuy);
				mxxMzz += omega * (-mxxMzz) - three * (one + c1o2 * (-omega)) * (vx2 * dxux - vz2 * dzuz);

				//////////////////////////////////////////////////////////////////////////
				//limiter-Scheise Teil 2
				//mxxMyy    += oxxyy * (-mxxMyy) - three * (one + c1o2 * (-omega)) * (vx2 * dxux + vy2 * dyuy);
				//mxxMzz    += oxxzz * (-mxxMzz) - three * (one + c1o2 * (-omega)) * (vx2 * dxux + vz2 * dzuz);
				//////////////////////////////////////////////////////////////////////////

			}
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			////no correction
			//mxxPyyPzz += OxxPyyPzz*(mfaaa-mxxPyyPzz);
			//mxxMyy    += -(-omega) * (-mxxMyy);
			//mxxMzz    += -(-omega) * (-mxxMzz);
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			mfabb += -(-omega) * (-mfabb);
			mfbab += -(-omega) * (-mfbab);
			mfbba += -(-omega) * (-mfbba);

			// linear combinations back
			mfcaa = c1o3 * (mxxMyy + mxxMzz + mxxPyyPzz);
			mfaca = c1o3 * (-two * mxxMyy + mxxMzz + mxxPyyPzz);
			mfaac = c1o3 * (mxxMyy - two * mxxMzz + mxxPyyPzz);

			//3.
			// linear combinations
			real mxxyPyzz = mfcba + mfabc;
			real mxxyMyzz = mfcba - mfabc;

			real mxxzPyyz = mfcab + mfacb;
			real mxxzMyyz = mfcab - mfacb;

			real mxyyPxzz = mfbca + mfbac;
			real mxyyMxzz = mfbca - mfbac;

			//relax
			wadjust = OxyyMxzz + (one - OxyyMxzz)*abs(mfbbb) / (abs(mfbbb) + qudricLimit);
			mfbbb += wadjust * (-mfbbb);
			wadjust = OxyyPxzz + (one - OxyyPxzz)*abs(mxxyPyzz) / (abs(mxxyPyzz) + qudricLimit);
			mxxyPyzz += wadjust * (-mxxyPyzz);
			wadjust = OxyyMxzz + (one - OxyyMxzz)*abs(mxxyMyzz) / (abs(mxxyMyzz) + qudricLimit);
			mxxyMyzz += wadjust * (-mxxyMyzz);
			wadjust = OxyyPxzz + (one - OxyyPxzz)*abs(mxxzPyyz) / (abs(mxxzPyyz) + qudricLimit);
			mxxzPyyz += wadjust * (-mxxzPyyz);
			wadjust = OxyyMxzz + (one - OxyyMxzz)*abs(mxxzMyyz) / (abs(mxxzMyyz) + qudricLimit);
			mxxzMyyz += wadjust * (-mxxzMyyz);
			wadjust = OxyyPxzz + (one - OxyyPxzz)*abs(mxyyPxzz) / (abs(mxyyPxzz) + qudricLimit);
			mxyyPxzz += wadjust * (-mxyyPxzz);
			wadjust = OxyyMxzz + (one - OxyyMxzz)*abs(mxyyMxzz) / (abs(mxyyMxzz) + qudricLimit);
			mxyyMxzz += wadjust * (-mxyyMxzz);

			// linear combinations back
			mfcba = (mxxyMyzz + mxxyPyzz) * c1o2;
			mfabc = (-mxxyMyzz + mxxyPyzz) * c1o2;
			mfcab = (mxxzMyyz + mxxzPyyz) * c1o2;
			mfacb = (-mxxzMyyz + mxxzPyyz) * c1o2;
			mfbca = (mxyyMxzz + mxyyPxzz) * c1o2;
			mfbac = (-mxyyMxzz + mxyyPxzz) * c1o2;

			//4.
			CUMacc += O4 * (-CUMacc);
			CUMcac += O4 * (-CUMcac);
			CUMcca += O4 * (-CUMcca);

			CUMbbc += O4 * (-CUMbbc);
			CUMbcb += O4 * (-CUMbcb);
			CUMcbb += O4 * (-CUMcbb);

			//5.
			CUMbcc += O5 * (-CUMbcc);
			CUMcbc += O5 * (-CUMcbc);
			CUMccb += O5 * (-CUMccb);

			//6.
			CUMccc += O6 * (-CUMccc);

			//back cumulants to central moments
			//4.
			mfcbb = CUMcbb + ((mfcaa + c1o3 * rho) * mfabb + two* mfbba * mfbab) / rho;
			mfbcb = CUMbcb + ((mfaca + c1o3 * rho) * mfbab + two* mfbba * mfabb) / rho;
			mfbbc = CUMbbc + ((mfaac + c1o3 * rho) * mfbba + two* mfbab * mfabb) / rho;

			mfcca = CUMcca + (mfcaa * mfaca + two* mfbba * mfbba) / rho + c1o3 * (mfcaa + mfaca);
			mfcac = CUMcac + (mfcaa * mfaac + two* mfbab * mfbab) / rho + c1o3 * (mfcaa + mfaac);
			mfacc = CUMacc + (mfaac * mfaca + two* mfabb * mfabb) / rho + c1o3 * (mfaac + mfaca);

			//5.
			mfbcc = CUMbcc + (mfaac * mfbca + mfaca * mfbac + four* mfabb * mfbbb + two* (mfbab * mfacb + mfbba * mfabc)) / rho + c1o3 * (mfbca + mfbac);
			mfcbc = CUMcbc + (mfaac * mfcba + mfcaa * mfabc + four* mfbab * mfbbb + two* (mfabb * mfcab + mfbba * mfbac)) / rho + c1o3 * (mfcba + mfabc);
			mfccb = CUMccb + (mfcaa * mfacb + mfaca * mfcab + four* mfbba * mfbbb + two* (mfbab * mfbca + mfabb * mfcba)) / rho + c1o3 * (mfacb + mfcab);

			//6.
			mfccc = CUMccc - ((-four*  mfbbb * mfbbb
				- (mfcaa * mfacc + mfaca * mfcac + mfaac * mfcca)
				- four* (mfabb * mfcbb + mfbab * mfbcb + mfbba * mfbbc)
				- two* (mfbca * mfbac + mfcba * mfabc + mfcab * mfacb)) / rho
				+ (four* (mfbab * mfbab * mfaca + mfabb * mfabb * mfcaa + mfbba * mfbba * mfaac)
					+ two* (mfcaa * mfaca * mfaac)
					+ sixteen*  mfbba * mfbab * mfabb) / (rho * rho)
				- c1o3* (mfacc + mfcac + mfcca)
				+ c1o9* (mfcaa + mfaca + mfaac)
				+ (two* (mfbab * mfbab + mfabb * mfabb + mfbba * mfbba)
					+ (mfaac * mfaca + mfaac * mfcaa + mfaca * mfcaa)) * c2o3 / rho);
			////////////////////////////////////////////////////////////////////////////////////

			////////////////////////////////////////////////////////////////////////////////////
			//the force be with you
			mfbaa = -mfbaa;
			mfaba = -mfaba;
			mfaab = -mfaab;
			////////////////////////////////////////////////////////////////////////////////////

			////////////////////////////////////////////////////////////////////////////////////
			//back
			////////////////////////////////////////////////////////////////////////////////////
			//mit 1, 0, 1/3, 0, 0, 0, 1/3, 0, 1/9   Konditionieren
			////////////////////////////////////////////////////////////////////////////////////
			// Z - Dir
			m0 = mfaac * c1o2 + mfaab * (vvz - c1o2) + (mfaaa + 1.) * (vz2 - vvz) * c1o2;
			m1 = -mfaac - two* mfaab *  vvz + mfaaa       * (one - vz2) - one* vz2;
			m2 = mfaac * c1o2 + mfaab * (vvz + c1o2) + (mfaaa + 1.) * (vz2 + vvz) * c1o2;
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
			m0 = mfacc * c1o2 + mfacb * (vvz - c1o2) + (mfaca + c1o3) * (vz2 - vvz) * c1o2;
			m1 = -mfacc - two* mfacb *  vvz + mfaca         * (one - vz2) - c1o3 * vz2;
			m2 = mfacc * c1o2 + mfacb * (vvz + c1o2) + (mfaca + c1o3) * (vz2 + vvz) * c1o2;
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
			m0 = mfcac * c1o2 + mfcab * (vvz - c1o2) + (mfcaa + c1o3) * (vz2 - vvz) * c1o2;
			m1 = -mfcac - two* mfcab *  vvz + mfcaa         * (one - vz2) - c1o3 * vz2;
			m2 = mfcac * c1o2 + mfcab * (vvz + c1o2) + (mfcaa + c1o3) * (vz2 + vvz) * c1o2;
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
			m0 = mfccc * c1o2 + mfccb * (vvz - c1o2) + (mfcca + c1o9) * (vz2 - vvz) * c1o2;
			m1 = -mfccc - two* mfccb *  vvz + mfcca         * (one - vz2) - c1o9 * vz2;
			m2 = mfccc * c1o2 + mfccb * (vvz + c1o2) + (mfcca + c1o9) * (vz2 + vvz) * c1o2;
			mfcca = m0;
			mfccb = m1;
			mfccc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			//mit 1/6, 2/3, 1/6, 0, 0, 0, 1/18, 2/9, 1/18   Konditionieren
			////////////////////////////////////////////////////////////////////////////////////
			// Y - Dir
			m0 = mfaca * c1o2 + mfaba * (vvy - c1o2) + (mfaaa + c1o6) * (vy2 - vvy) * c1o2;
			m1 = -mfaca - two* mfaba *  vvy + mfaaa         * (one - vy2) - c1o6 * vy2;
			m2 = mfaca * c1o2 + mfaba * (vvy + c1o2) + (mfaaa + c1o6) * (vy2 + vvy) * c1o2;
			mfaaa = m0;
			mfaba = m1;
			mfaca = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfacb * c1o2 + mfabb * (vvy - c1o2) + (mfaab + c2o3) * (vy2 - vvy) * c1o2;
			m1 = -mfacb - two* mfabb *  vvy + mfaab         * (one - vy2) - c2o3 * vy2;
			m2 = mfacb * c1o2 + mfabb * (vvy + c1o2) + (mfaab + c2o3) * (vy2 + vvy) * c1o2;
			mfaab = m0;
			mfabb = m1;
			mfacb = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfacc * c1o2 + mfabc * (vvy - c1o2) + (mfaac + c1o6) * (vy2 - vvy) * c1o2;
			m1 = -mfacc - two* mfabc *  vvy + mfaac         * (one - vy2) - c1o6 * vy2;
			m2 = mfacc * c1o2 + mfabc * (vvy + c1o2) + (mfaac + c1o6) * (vy2 + vvy) * c1o2;
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
			m0 = mfcca * c1o2 + mfcba * (vvy - c1o2) + (mfcaa + c1o18) * (vy2 - vvy) * c1o2;
			m1 = -mfcca - two* mfcba *  vvy + mfcaa          * (one - vy2) - c1o18 * vy2;
			m2 = mfcca * c1o2 + mfcba * (vvy + c1o2) + (mfcaa + c1o18) * (vy2 + vvy) * c1o2;
			mfcaa = m0;
			mfcba = m1;
			mfcca = m2;
			/////////c//////////////////////////////////////////////////////////////////////////
			m0 = mfccb * c1o2 + mfcbb * (vvy - c1o2) + (mfcab + c2o9) * (vy2 - vvy) * c1o2;
			m1 = -mfccb - two* mfcbb *  vvy + mfcab         * (one - vy2) - c2o9 * vy2;
			m2 = mfccb * c1o2 + mfcbb * (vvy + c1o2) + (mfcab + c2o9) * (vy2 + vvy) * c1o2;
			mfcab = m0;
			mfcbb = m1;
			mfccb = m2;
			/////////c//////////////////////////////////////////////////////////////////////////
			m0 = mfccc * c1o2 + mfcbc * (vvy - c1o2) + (mfcac + c1o18) * (vy2 - vvy) * c1o2;
			m1 = -mfccc - two* mfcbc *  vvy + mfcac          * (one - vy2) - c1o18 * vy2;
			m2 = mfccc * c1o2 + mfcbc * (vvy + c1o2) + (mfcac + c1o18) * (vy2 + vvy) * c1o2;
			mfcac = m0;
			mfcbc = m1;
			mfccc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			//mit 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36 Konditionieren
			////////////////////////////////////////////////////////////////////////////////////
			// X - Dir
			m0 = mfcaa * c1o2 + mfbaa * (vvx - c1o2) + (mfaaa + c1o36) * (vx2 - vvx) * c1o2;
			m1 = -mfcaa - two* mfbaa *  vvx + mfaaa          * (one - vx2) - c1o36 * vx2;
			m2 = mfcaa * c1o2 + mfbaa * (vvx + c1o2) + (mfaaa + c1o36) * (vx2 + vvx) * c1o2;
			mfaaa = m0;
			mfbaa = m1;
			mfcaa = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcba * c1o2 + mfbba * (vvx - c1o2) + (mfaba + c1o9) * (vx2 - vvx) * c1o2;
			m1 = -mfcba - two* mfbba *  vvx + mfaba         * (one - vx2) - c1o9 * vx2;
			m2 = mfcba * c1o2 + mfbba * (vvx + c1o2) + (mfaba + c1o9) * (vx2 + vvx) * c1o2;
			mfaba = m0;
			mfbba = m1;
			mfcba = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcca * c1o2 + mfbca * (vvx - c1o2) + (mfaca + c1o36) * (vx2 - vvx) * c1o2;
			m1 = -mfcca - two* mfbca *  vvx + mfaca          * (one - vx2) - c1o36 * vx2;
			m2 = mfcca * c1o2 + mfbca * (vvx + c1o2) + (mfaca + c1o36) * (vx2 + vvx) * c1o2;
			mfaca = m0;
			mfbca = m1;
			mfcca = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcab * c1o2 + mfbab * (vvx - c1o2) + (mfaab + c1o9) * (vx2 - vvx) * c1o2;
			m1 = -mfcab - two* mfbab *  vvx + mfaab         * (one - vx2) - c1o9 * vx2;
			m2 = mfcab * c1o2 + mfbab * (vvx + c1o2) + (mfaab + c1o9) * (vx2 + vvx) * c1o2;
			mfaab = m0;
			mfbab = m1;
			mfcab = m2;
			///////////b////////////////////////////////////////////////////////////////////////
			m0 = mfcbb * c1o2 + mfbbb * (vvx - c1o2) + (mfabb + c4o9) * (vx2 - vvx) * c1o2;
			m1 = -mfcbb - two* mfbbb *  vvx + mfabb         * (one - vx2) - c4o9 * vx2;
			m2 = mfcbb * c1o2 + mfbbb * (vvx + c1o2) + (mfabb + c4o9) * (vx2 + vvx) * c1o2;
			mfabb = m0;
			mfbbb = m1;
			mfcbb = m2;
			///////////b////////////////////////////////////////////////////////////////////////
			m0 = mfccb * c1o2 + mfbcb * (vvx - c1o2) + (mfacb + c1o9) * (vx2 - vvx) * c1o2;
			m1 = -mfccb - two* mfbcb *  vvx + mfacb         * (one - vx2) - c1o9 * vx2;
			m2 = mfccb * c1o2 + mfbcb * (vvx + c1o2) + (mfacb + c1o9) * (vx2 + vvx) * c1o2;
			mfacb = m0;
			mfbcb = m1;
			mfccb = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m0 = mfcac * c1o2 + mfbac * (vvx - c1o2) + (mfaac + c1o36) * (vx2 - vvx) * c1o2;
			m1 = -mfcac - two* mfbac *  vvx + mfaac          * (one - vx2) - c1o36 * vx2;
			m2 = mfcac * c1o2 + mfbac * (vvx + c1o2) + (mfaac + c1o36) * (vx2 + vvx) * c1o2;
			mfaac = m0;
			mfbac = m1;
			mfcac = m2;
			///////////c////////////////////////////////////////////////////////////////////////
			m0 = mfcbc * c1o2 + mfbbc * (vvx - c1o2) + (mfabc + c1o9) * (vx2 - vvx) * c1o2;
			m1 = -mfcbc - two* mfbbc *  vvx + mfabc         * (one - vx2) - c1o9 * vx2;
			m2 = mfcbc * c1o2 + mfbbc * (vvx + c1o2) + (mfabc + c1o9) * (vx2 + vvx) * c1o2;
			mfabc = m0;
			mfbbc = m1;
			mfcbc = m2;
			///////////c////////////////////////////////////////////////////////////////////////
			m0 = mfccc * c1o2 + mfbcc * (vvx - c1o2) + (mfacc + c1o36) * (vx2 - vvx) * c1o2;
			m1 = -mfccc - two* mfbcc *  vvx + mfacc          * (one - vx2) - c1o36 * vx2;
			m2 = mfccc * c1o2 + mfbcc * (vvx + c1o2) + (mfacc + c1o36) * (vx2 + vvx) * c1o2;
			mfacc = m0;
			mfbcc = m1;
			mfccc = m2;
			////////////////////////////////////////////////////////////////////////////////////


			////////////////////////////////////////////////////////////////////////////////////
			(D.f[dirE])[ke] = mfabb;// -  c2over27 ;//                                                                     
			(D.f[dirW])[kw] = mfcbb;// -  c2over27 ;                                                                     
			(D.f[dirN])[kn] = mfbab;// -  c2over27 ;
			(D.f[dirS])[ks] = mfbcb;// -  c2over27 ;
			(D.f[dirT])[kt] = mfbba;// -  c2over27 ;
			(D.f[dirB])[kb] = mfbbc;// -  c2over27 ;
			(D.f[dirNE])[kne] = mfaab;// -  c1over54 ;
			(D.f[dirSW])[ksw] = mfccb;// -  c1over54 ;
			(D.f[dirSE])[kse] = mfacb;// -  c1over54 ;
			(D.f[dirNW])[knw] = mfcab;// -  c1over54 ;
			(D.f[dirTE])[kte] = mfaba;// -  c1over54 ;
			(D.f[dirBW])[kbw] = mfcbc;// -  c1over54 ;
			(D.f[dirBE])[kbe] = mfabc;// -  c1over54 ;
			(D.f[dirTW])[ktw] = mfcba;// -  c1over54 ;
			(D.f[dirTN])[ktn] = mfbaa;// -  c1over54 ;
			(D.f[dirBS])[kbs] = mfbcc;// -  c1over54 ;
			(D.f[dirBN])[kbn] = mfbac;// -  c1over54 ;
			(D.f[dirTS])[kts] = mfbca;// -  c1over54 ;
			(D.f[dirZERO])[kzero] = mfbbb;// -  c8over27 ;
			(D.f[dirTNE])[ktne] = mfaaa;// -  c1over216;
			(D.f[dirTSE])[ktse] = mfaca;// -  c1over216;
			(D.f[dirBNE])[kbne] = mfaac;// -  c1over216;
			(D.f[dirBSE])[kbse] = mfacc;// -  c1over216;
			(D.f[dirTNW])[ktnw] = mfcaa;// -  c1over216;
			(D.f[dirTSW])[ktsw] = mfcca;// -  c1over216;
			(D.f[dirBNW])[kbnw] = mfcac;// -  c1over216;
			(D.f[dirBSW])[kbsw] = mfccc;// -  c1over216;
										////////////////////////////////////////////////////////////////////////////////////
		}
	}
}