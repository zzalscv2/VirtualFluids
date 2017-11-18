/* Device code */
#include "LBM/D3Q27.h"
#include "math.h"
#include "GPU/constant.h"

////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void LB_Kernel_BGK_Plus_SP_27(doubflo omega,
													unsigned int* bcMatD,
													unsigned int* neighborX,
													unsigned int* neighborY,
													unsigned int* neighborZ,
													doubflo* DDStart,
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

	if (k < size_Mat)
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
			//unsigned int kzero= k;
			//unsigned int ke   = k;
			unsigned int kw = neighborX[k];
			//unsigned int kn   = k;
			unsigned int ks = neighborY[k];
			//unsigned int kt   = k;
			unsigned int kb = neighborZ[k];
			unsigned int ksw = neighborY[kw];
			//unsigned int kne  = k;
			//unsigned int kse  = ks;
			//unsigned int knw  = kw;
			unsigned int kbw = neighborZ[kw];
			//unsigned int kte  = k;
			//unsigned int kbe  = kb;
			//unsigned int ktw  = kw;
			unsigned int kbs = neighborZ[ks];
			//unsigned int ktn  = k;
			//unsigned int kbn  = kb;
			//unsigned int kts  = ks;
			//unsigned int ktse = ks;
			//unsigned int kbnw = kbw;
			//unsigned int ktnw = kw;
			//unsigned int kbse = kbs;
			//unsigned int ktsw = ksw;
			//unsigned int kbne = kb;
			//unsigned int ktne = k;
			unsigned int kbsw = neighborZ[ksw];
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			doubflo mfcbb = (D.f[dirE])[k];//[ke   ];// +  c2over27 ;(D.f[dirE   ])[k  ];//ke
			doubflo mfabb = (D.f[dirW])[kw];//[kw   ];// +  c2over27 ;(D.f[dirW   ])[kw ];
			doubflo mfbcb = (D.f[dirN])[k];//[kn   ];// +  c2over27 ;(D.f[dirN   ])[k  ];//kn
			doubflo mfbab = (D.f[dirS])[ks];//[ks   ];// +  c2over27 ;(D.f[dirS   ])[ks ];
			doubflo mfbbc = (D.f[dirT])[k];//[kt   ];// +  c2over27 ;(D.f[dirT   ])[k  ];//kt
			doubflo mfbba = (D.f[dirB])[kb];//[kb   ];// +  c2over27 ;(D.f[dirB   ])[kb ];
			doubflo mfccb = (D.f[dirNE])[k];//[kne  ];// +  c1over54 ;(D.f[dirNE  ])[k  ];//kne
			doubflo mfaab = (D.f[dirSW])[ksw];//[ksw  ];// +  c1over54 ;(D.f[dirSW  ])[ksw];
			doubflo mfcab = (D.f[dirSE])[ks];//[kse  ];// +  c1over54 ;(D.f[dirSE  ])[ks ];//kse
			doubflo mfacb = (D.f[dirNW])[kw];//[knw  ];// +  c1over54 ;(D.f[dirNW  ])[kw ];//knw
			doubflo mfcbc = (D.f[dirTE])[k];//[kte  ];// +  c1over54 ;(D.f[dirTE  ])[k  ];//kte
			doubflo mfaba = (D.f[dirBW])[kbw];//[kbw  ];// +  c1over54 ;(D.f[dirBW  ])[kbw];
			doubflo mfcba = (D.f[dirBE])[kb];//[kbe  ];// +  c1over54 ;(D.f[dirBE  ])[kb ];//kbe
			doubflo mfabc = (D.f[dirTW])[kw];//[ktw  ];// +  c1over54 ;(D.f[dirTW  ])[kw ];//ktw
			doubflo mfbcc = (D.f[dirTN])[k];//[ktn  ];// +  c1over54 ;(D.f[dirTN  ])[k  ];//ktn
			doubflo mfbaa = (D.f[dirBS])[kbs];//[kbs  ];// +  c1over54 ;(D.f[dirBS  ])[kbs];
			doubflo mfbca = (D.f[dirBN])[kb];//[kbn  ];// +  c1over54 ;(D.f[dirBN  ])[kb ];//kbn
			doubflo mfbac = (D.f[dirTS])[ks];//[kts  ];// +  c1over54 ;(D.f[dirTS  ])[ks ];//kts
			doubflo mfbbb = (D.f[dirZERO])[k];//[kzero];// +  c8over27 ;(D.f[dirZERO])[k  ];//kzero
			doubflo mfccc = (D.f[dirTNE])[k];//[ktne ];// +  c1over216;(D.f[dirTNE ])[k  ];//ktne
			doubflo mfaac = (D.f[dirTSW])[ksw];//[ktsw ];// +  c1over216;(D.f[dirTSW ])[ksw];//ktsw
			doubflo mfcac = (D.f[dirTSE])[ks];//[ktse ];// +  c1over216;(D.f[dirTSE ])[ks ];//ktse
			doubflo mfacc = (D.f[dirTNW])[kw];//[ktnw ];// +  c1over216;(D.f[dirTNW ])[kw ];//ktnw
			doubflo mfcca = (D.f[dirBNE])[kb];//[kbne ];// +  c1over216;(D.f[dirBNE ])[kb ];//kbne
			doubflo mfaaa = (D.f[dirBSW])[kbsw];//[kbsw ];// +  c1over216;(D.f[dirBSW ])[kbsw];
			doubflo mfcaa = (D.f[dirBSE])[kbs];//[kbse ];// +  c1over216;(D.f[dirBSE ])[kbs];//kbse
			doubflo mfaca = (D.f[dirBNW])[kbw];//[kbnw ];// +  c1over216;(D.f[dirBNW ])[kbw];//kbnw
			////////////////////////////////////////////////////////////////////////////////////
			//slow
			//doubflo oMdrho = one - ((((mfccc+mfaaa) + (mfaca+mfcac)) + ((mfacc+mfcaa) + (mfaac+mfcca))) + 
			//					   (((mfbac+mfbca) + (mfbaa+mfbcc)) + ((mfabc+mfcba) + (mfaba+mfcbc)) + ((mfacb+mfcab) + (mfaab+mfccb))) +
			//						((mfabb+mfcbb) + (mfbab+mfbcb)  +  (mfbba+mfbbc)));//fehlt mfbbb
			doubflo vvx = ((((mfccc - mfaaa) + (mfcac - mfaca)) + ((mfcaa - mfacc) + (mfcca - mfaac))) +
				(((mfcba - mfabc) + (mfcbc - mfaba)) + ((mfcab - mfacb) + (mfccb - mfaab))) +
				(mfcbb - mfabb));
			doubflo vvy = ((((mfccc - mfaaa) + (mfaca - mfcac)) + ((mfacc - mfcaa) + (mfcca - mfaac))) +
				(((mfbca - mfbac) + (mfbcc - mfbaa)) + ((mfacb - mfcab) + (mfccb - mfaab))) +
				(mfbcb - mfbab));
			doubflo vvz = ((((mfccc - mfaaa) + (mfcac - mfaca)) + ((mfacc - mfcaa) + (mfaac - mfcca))) +
				(((mfbac - mfbca) + (mfbcc - mfbaa)) + ((mfabc - mfcba) + (mfcbc - mfaba))) +
				(mfbbc - mfbba));
			////////////////////////////////////////////////////////////////////////////////////
			//fast
			doubflo oMdrho = one - (mfccc + mfaaa + mfaca + mfcac + mfacc + mfcaa + mfaac + mfcca +
				mfbac + mfbca + mfbaa + mfbcc + mfabc + mfcba + mfaba + mfcbc + mfacb + mfcab + mfaab + mfccb +
				mfabb + mfcbb + mfbab + mfbcb + mfbba + mfbbc + mfbbb);//fehlt mfbbb nicht mehr
////////////////////////////////////////////////////////////////////////////////////
			doubflo m0, m1, m2;
			doubflo vx2;
			doubflo vy2;
			doubflo vz2;
			vx2 = vvx*vvx;
			vy2 = vvy*vvy;
			vz2 = vvz*vvz;
			////////////////////////////////////////////////////////////////////////////////////
			doubflo wadjust;
			doubflo qudricLimit = 0.01f;
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
			mfaab = m1;
			mfaac = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaba + mfabc;
			m1 = mfabc - mfaba;
			m0 = m2 + mfabb;
			mfaba = m0;
			m0 += c1o9 * oMdrho;
			mfabb = m1;
			mfabc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaca + mfacc;
			m1 = mfacc - mfaca;
			m0 = m2 + mfacb;
			mfaca = m0;
			m0 += c1o36 * oMdrho;
			mfacb = m1;
			mfacc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbaa + mfbac;
			m1 = mfbac - mfbaa;
			m0 = m2 + mfbab;
			mfbaa = m0;
			m0 += c1o9 * oMdrho;
			mfbab = m1;
			mfbac = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbba + mfbbc;
			m1 = mfbbc - mfbba;
			m0 = m2 + mfbbb;
			mfbba = m0;
			m0 += c4o9 * oMdrho;
			mfbbb = m1;
			mfbbc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbca + mfbcc;
			m1 = mfbcc - mfbca;
			m0 = m2 + mfbcb;
			mfbca = m0;
			m0 += c1o9 * oMdrho;
			mfbcb = m1;
			mfbcc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcaa + mfcac;
			m1 = mfcac - mfcaa;
			m0 = m2 + mfcab;
			mfcaa = m0;
			m0 += c1o36 * oMdrho;
			mfcab = m1;
			mfcac = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcba + mfcbc;
			m1 = mfcbc - mfcba;
			m0 = m2 + mfcbb;
			mfcba = m0;
			m0 += c1o9 * oMdrho;
			mfcbb = m1;
			mfcbc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcca + mfccc;
			m1 = mfccc - mfcca;
			m0 = m2 + mfccb;
			mfcca = m0;
			m0 += c1o36 * oMdrho;
			mfccb = m1;
			mfccc = m2;
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
			mfaba = m1;
			mfaca = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaab + mfacb;
			m1 = mfacb - mfaab;
			m0 = m2 + mfabb;
			mfaab = m0;
			mfabb = m1;
			mfacb = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaac + mfacc;
			m1 = mfacc - mfaac;
			m0 = m2 + mfabc;
			mfaac = m0;
			m0 += c1o18 * oMdrho;
			mfabc = m1;
			mfacc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbaa + mfbca;
			m1 = mfbca - mfbaa;
			m0 = m2 + mfbba;
			mfbaa = m0;
			m0 += c2o3 * oMdrho;
			mfbba = m1;
			mfbca = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbab + mfbcb;
			m1 = mfbcb - mfbab;
			m0 = m2 + mfbbb;
			mfbab = m0;
			mfbbb = m1;
			mfbcb = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfbac + mfbcc;
			m1 = mfbcc - mfbac;
			m0 = m2 + mfbbc;
			mfbac = m0;
			m0 += c2o9 * oMdrho;
			mfbbc = m1;
			mfbcc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcaa + mfcca;
			m1 = mfcca - mfcaa;
			m0 = m2 + mfcba;
			mfcaa = m0;
			m0 += c1o6 * oMdrho;
			mfcba = m1;
			mfcca = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcab + mfccb;
			m1 = mfccb - mfcab;
			m0 = m2 + mfcbb;
			mfcab = m0;
			mfcbb = m1;
			mfccb = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfcac + mfccc;
			m1 = mfccc - mfcac;
			m0 = m2 + mfcbc;
			mfcac = m0;
			m0 += c1o18 * oMdrho;
			mfcbc = m1;
			mfccc = m2;
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
			mfbaa = m1;
			mfcaa = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaba + mfcba;
			m1 = mfcba - mfaba;
			m0 = m2 + mfbba;
			mfaba = m0;
			mfbba = m1;
			mfcba = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaca + mfcca;
			m1 = mfcca - mfaca;
			m0 = m2 + mfbca;
			mfaca = m0;
			m0 += c1o3 * oMdrho;
			mfbca = m1;
			mfcca = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaab + mfcab;
			m1 = mfcab - mfaab;
			m0 = m2 + mfbab;
			mfaab = m0;
			mfbab = m1;
			mfcab = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfabb + mfcbb;
			m1 = mfcbb - mfabb;
			m0 = m2 + mfbbb;
			mfabb = m0;
			mfbbb = m1;
			mfcbb = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfacb + mfccb;
			m1 = mfccb - mfacb;
			m0 = m2 + mfbcb;
			mfacb = m0;
			mfbcb = m1;
			mfccb = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfaac + mfcac;
			m1 = mfcac - mfaac;
			m0 = m2 + mfbac;
			mfaac = m0;
			m0 += c1o3 * oMdrho;
			mfbac = m1;
			mfcac = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfabc + mfcbc;
			m1 = mfcbc - mfabc;
			m0 = m2 + mfbbc;
			mfabc = m0;
			mfbbc = m1;
			mfcbc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			m2 = mfacc + mfccc;
			m1 = mfccc - mfacc;
			m0 = m2 + mfbcc;
			mfacc = m0;
			m0 += c1o9 * oMdrho;
			mfbcc = m1;
			mfccc = m2;
			////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////////


			////////////////////////////////////////////////////////////////////////////////////
			// BGK
			////////////////////////////////////////////////////////////////////////////////////
			doubflo OxxPyyPzz = omega;
			doubflo OxyyPxzz = omega;//two-omega;//eight*(two-omega)/(eight -omega);//one;//omega;//two-omega;//
			doubflo OxyyMxzz = omega;//omega;//one;//eight*(two-omega)/(eight -omega);//one;//two-omega;//one;// 
			doubflo O4 = omega;
			doubflo O5 = omega;
			doubflo O6 = omega;

			doubflo mxxPyyPzz = mfcaa + mfaca + mfaac;
			doubflo mxxMyy = mfcaa - mfaca;
			doubflo mxxMzz = mfcaa - mfaac;

			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//incl. correction
			{
				doubflo dxux = c1o2 * (-omega) *(mxxMyy + mxxMzz - two*vx2 + vy2 + vz2) + c1o2 * OxxPyyPzz * (mfaaa + vx2 + vy2 + vz2 - mxxPyyPzz);
				doubflo dyuy = dxux + omega * c3o2 * (mxxMyy - vx2 + vy2);
				doubflo dzuz = dxux + omega * c3o2 * (mxxMzz - vx2 + vz2);

				//relax
				mxxPyyPzz += OxxPyyPzz*(mfaaa + vx2 + vy2 + vz2 - mxxPyyPzz) - three * (one - c1o2 * OxxPyyPzz) * (vx2 * dxux + vy2 * dyuy + vz2 * dzuz);
				mxxMyy += omega * (vx2 - vy2 - mxxMyy) - three * (one + c1o2 * (-omega)) * (vx2 * dxux - vy2 * dyuy);
				mxxMzz += omega * (vx2 - vz2 - mxxMzz) - three * (one + c1o2 * (-omega)) * (vx2 * dxux - vz2 * dzuz);
			}
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// 			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 			//no correction
// 			mxxPyyPzz += OxxPyyPzz*(mfaaa+vx2+vy2+vz2-mxxPyyPzz);
// 			mxxMyy    += -(-omega) * (vx2-vy2-mxxMyy);
// 			mxxMzz    += -(-omega) * (vx2-vz2-mxxMzz);
// 			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			mfabb += omega * (vvy*vvz - mfabb);
			mfbab += omega * (vvx*vvz - mfbab);
			mfbba += omega * (vvx*vvy - mfbba);

			// linear combinations back
			mfcaa = c1o3 * (mxxMyy + mxxMzz + mxxPyyPzz);
			mfaca = c1o3 * (-two*  mxxMyy + mxxMzz + mxxPyyPzz);
			mfaac = c1o3 * (mxxMyy - two* mxxMzz + mxxPyyPzz);

			//3.
			// linear combinations

			doubflo mxxyPyzz = mfcba + mfabc;
			doubflo mxxyMyzz = mfcba - mfabc;

			doubflo mxxzPyyz = mfcab + mfacb;
			doubflo mxxzMyyz = mfcab - mfacb;

			doubflo mxyyPxzz = mfbca + mfbac;
			doubflo mxyyMxzz = mfbca - mfbac;

			mxxyMyzz += OxyyMxzz*((vx2 - vz2)*vvy - mxxyMyzz);
			mxxzMyyz += OxyyMxzz*((vx2 - vy2)*vvz - mxxzMyyz);
			mxyyMxzz += OxyyMxzz*((vy2 - vz2)*vvx - mxyyMxzz);

			mxxyPyzz += OxyyPxzz*((c2o3 + vx2 + vz2)*vvy - mxxyPyzz);
			mxxzPyyz += OxyyPxzz*((c2o3 + vx2 + vy2)*vvz - mxxzPyyz);
			mxyyPxzz += OxyyPxzz*((c2o3 + vy2 + vz2)*vvx - mxyyPxzz);

			mfbbb += OxyyMxzz * (vvx*vvy*vvz - mfbbb);

			mfcba = (mxxyMyzz + mxxyPyzz) * c1o2;
			mfabc = (-mxxyMyzz + mxxyPyzz) * c1o2;
			mfcab = (mxxzMyyz + mxxzPyyz) * c1o2;
			mfacb = (-mxxzMyyz + mxxzPyyz) * c1o2;
			mfbca = (mxyyMxzz + mxyyPxzz) * c1o2;
			mfbac = (-mxyyMxzz + mxyyPxzz) * c1o2;

			//4.
			mfacc += O4*((c1o3 + vy2)*(c1o3 + vz2) + c1o9*(mfaaa - one) - mfacc);
			mfcac += O4*((c1o3 + vx2)*(c1o3 + vz2) + c1o9*(mfaaa - one) - mfcac);
			mfcca += O4*((c1o3 + vx2)*(c1o3 + vy2) + c1o9*(mfaaa - one) - mfcca);

			mfcbb += O4*((c1o3 + vx2)*vvy*vvz - mfcbb);
			mfbcb += O4*((c1o3 + vy2)*vvx*vvz - mfbcb);
			mfbbc += O4*((c1o3 + vz2)*vvx*vvy - mfbbc);

			//5.
			mfbcc += O5*((c1o3 + vy2)*(c1o3 + vz2)*vvx - mfbcc);
			mfcbc += O5*((c1o3 + vx2)*(c1o3 + vz2)*vvy - mfcbc);
			mfccb += O5*((c1o3 + vx2)*(c1o3 + vy2)*vvz - mfccb);

			//6.
			mfccc += O6*((c1o3 + vx2)*(c1o3 + vy2)*(c1o3 + vz2) + c1o27*(mfaaa - one) - mfccc);


			//bad fix
			vvx = zero;
			vvy = zero;
			vvz = zero;
			vx2 = zero;
			vy2 = zero;
			vz2 = zero;
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
			(D.f[dirE])[k] = mfabb;//(D.f[ dirE   ])[ke   ] = mfabb;// -  c2over27 ;  (D.f[ dirE   ])[k   ]                                                                     
			(D.f[dirW])[kw] = mfcbb;//(D.f[ dirW   ])[kw   ] = mfcbb;// -  c2over27 ;  (D.f[ dirW   ])[kw  ]                                                                   
			(D.f[dirN])[k] = mfbab;//(D.f[ dirN   ])[kn   ] = mfbab;// -  c2over27 ;	 (D.f[ dirN   ])[k   ]
			(D.f[dirS])[ks] = mfbcb;//(D.f[ dirS   ])[ks   ] = mfbcb;// -  c2over27 ;	 (D.f[ dirS   ])[ks  ]
			(D.f[dirT])[k] = mfbba;//(D.f[ dirT   ])[kt   ] = mfbba;// -  c2over27 ;	 (D.f[ dirT   ])[k   ]
			(D.f[dirB])[kb] = mfbbc;//(D.f[ dirB   ])[kb   ] = mfbbc;// -  c2over27 ;	 (D.f[ dirB   ])[kb  ]
			(D.f[dirNE])[k] = mfaab;//(D.f[ dirNE  ])[kne  ] = mfaab;// -  c1over54 ;	 (D.f[ dirNE  ])[k   ]
			(D.f[dirSW])[ksw] = mfccb;//(D.f[ dirSW  ])[ksw  ] = mfccb;// -  c1over54 ;	 (D.f[ dirSW  ])[ksw ]
			(D.f[dirSE])[ks] = mfacb;//(D.f[ dirSE  ])[kse  ] = mfacb;// -  c1over54 ;	 (D.f[ dirSE  ])[ks  ]
			(D.f[dirNW])[kw] = mfcab;//(D.f[ dirNW  ])[knw  ] = mfcab;// -  c1over54 ;	 (D.f[ dirNW  ])[kw  ]
			(D.f[dirTE])[k] = mfaba;//(D.f[ dirTE  ])[kte  ] = mfaba;// -  c1over54 ;	 (D.f[ dirTE  ])[k   ]
			(D.f[dirBW])[kbw] = mfcbc;//(D.f[ dirBW  ])[kbw  ] = mfcbc;// -  c1over54 ;	 (D.f[ dirBW  ])[kbw ]
			(D.f[dirBE])[kb] = mfabc;//(D.f[ dirBE  ])[kbe  ] = mfabc;// -  c1over54 ;	 (D.f[ dirBE  ])[kb  ]
			(D.f[dirTW])[kw] = mfcba;//(D.f[ dirTW  ])[ktw  ] = mfcba;// -  c1over54 ;	 (D.f[ dirTW  ])[kw  ]
			(D.f[dirTN])[k] = mfbaa;//(D.f[ dirTN  ])[ktn  ] = mfbaa;// -  c1over54 ;	 (D.f[ dirTN  ])[k   ]
			(D.f[dirBS])[kbs] = mfbcc;//(D.f[ dirBS  ])[kbs  ] = mfbcc;// -  c1over54 ;	 (D.f[ dirBS  ])[kbs ]
			(D.f[dirBN])[kb] = mfbac;//(D.f[ dirBN  ])[kbn  ] = mfbac;// -  c1over54 ;	 (D.f[ dirBN  ])[kb  ]
			(D.f[dirTS])[ks] = mfbca;//(D.f[ dirTS  ])[kts  ] = mfbca;// -  c1over54 ;	 (D.f[ dirTS  ])[ks  ]
			(D.f[dirZERO])[k] = mfbbb;//(D.f[ dirZERO])[kzero] = mfbbb;// -  c8over27 ;	 (D.f[ dirZERO])[k   ]
			(D.f[dirTNE])[k] = mfaaa;//(D.f[ dirTNE ])[ktne ] = mfaaa;// -  c1over216;	 (D.f[ dirTNE ])[k   ]
			(D.f[dirTSE])[ks] = mfaca;//(D.f[ dirTSE ])[ktse ] = mfaca;// -  c1over216;	 (D.f[ dirTSE ])[ks  ]
			(D.f[dirBNE])[kb] = mfaac;//(D.f[ dirBNE ])[kbne ] = mfaac;// -  c1over216;	 (D.f[ dirBNE ])[kb  ]
			(D.f[dirBSE])[kbs] = mfacc;//(D.f[ dirBSE ])[kbse ] = mfacc;// -  c1over216;	 (D.f[ dirBSE ])[kbs ]
			(D.f[dirTNW])[kw] = mfcaa;//(D.f[ dirTNW ])[ktnw ] = mfcaa;// -  c1over216;	 (D.f[ dirTNW ])[kw  ]
			(D.f[dirTSW])[ksw] = mfcca;//(D.f[ dirTSW ])[ktsw ] = mfcca;// -  c1over216;	 (D.f[ dirTSW ])[ksw ]
			(D.f[dirBNW])[kbw] = mfcac;//(D.f[ dirBNW ])[kbnw ] = mfcac;// -  c1over216;	 (D.f[ dirBNW ])[kbw ]
			(D.f[dirBSW])[kbsw] = mfccc;//(D.f[ dirBSW ])[kbsw ] = mfccc;// -  c1over216;	 (D.f[ dirBSW ])[kbsw]
			////////////////////////////////////////////////////////////////////////////////////
		}
	}
}
////////////////////////////////////////////////////////////////////////////////







































////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void LB_Kernel_BGK_Plus_Comp_SP_27(doubflo omega,
														 unsigned int* bcMatD,
														 unsigned int* neighborX,
														 unsigned int* neighborY,
														 unsigned int* neighborZ,
														 doubflo* DDStart,
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

	if (k < size_Mat)
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
			//unsigned int kzero= k;
			//unsigned int ke   = k;
			unsigned int kw = neighborX[k];
			//unsigned int kn   = k;
			unsigned int ks = neighborY[k];
			//unsigned int kt   = k;
			unsigned int kb = neighborZ[k];
			unsigned int ksw = neighborY[kw];
			//unsigned int kne  = k;
			//unsigned int kse  = ks;
			//unsigned int knw  = kw;
			unsigned int kbw = neighborZ[kw];
			//unsigned int kte  = k;
			//unsigned int kbe  = kb;
			//unsigned int ktw  = kw;
			unsigned int kbs = neighborZ[ks];
			//unsigned int ktn  = k;
			//unsigned int kbn  = kb;
			//unsigned int kts  = ks;
			//unsigned int ktse = ks;
			//unsigned int kbnw = kbw;
			//unsigned int ktnw = kw;
			//unsigned int kbse = kbs;
			//unsigned int ktsw = ksw;
			//unsigned int kbne = kb;
			//unsigned int ktne = k;
			unsigned int kbsw = neighborZ[ksw];
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			doubflo mfcbb = (D.f[dirE])[k];//[ke   ];// +  c2over27 ;(D.f[dirE   ])[k  ];//ke
			doubflo mfabb = (D.f[dirW])[kw];//[kw   ];// +  c2over27 ;(D.f[dirW   ])[kw ];
			doubflo mfbcb = (D.f[dirN])[k];//[kn   ];// +  c2over27 ;(D.f[dirN   ])[k  ];//kn
			doubflo mfbab = (D.f[dirS])[ks];//[ks   ];// +  c2over27 ;(D.f[dirS   ])[ks ];
			doubflo mfbbc = (D.f[dirT])[k];//[kt   ];// +  c2over27 ;(D.f[dirT   ])[k  ];//kt
			doubflo mfbba = (D.f[dirB])[kb];//[kb   ];// +  c2over27 ;(D.f[dirB   ])[kb ];
			doubflo mfccb = (D.f[dirNE])[k];//[kne  ];// +  c1over54 ;(D.f[dirNE  ])[k  ];//kne
			doubflo mfaab = (D.f[dirSW])[ksw];//[ksw  ];// +  c1over54 ;(D.f[dirSW  ])[ksw];
			doubflo mfcab = (D.f[dirSE])[ks];//[kse  ];// +  c1over54 ;(D.f[dirSE  ])[ks ];//kse
			doubflo mfacb = (D.f[dirNW])[kw];//[knw  ];// +  c1over54 ;(D.f[dirNW  ])[kw ];//knw
			doubflo mfcbc = (D.f[dirTE])[k];//[kte  ];// +  c1over54 ;(D.f[dirTE  ])[k  ];//kte
			doubflo mfaba = (D.f[dirBW])[kbw];//[kbw  ];// +  c1over54 ;(D.f[dirBW  ])[kbw];
			doubflo mfcba = (D.f[dirBE])[kb];//[kbe  ];// +  c1over54 ;(D.f[dirBE  ])[kb ];//kbe
			doubflo mfabc = (D.f[dirTW])[kw];//[ktw  ];// +  c1over54 ;(D.f[dirTW  ])[kw ];//ktw
			doubflo mfbcc = (D.f[dirTN])[k];//[ktn  ];// +  c1over54 ;(D.f[dirTN  ])[k  ];//ktn
			doubflo mfbaa = (D.f[dirBS])[kbs];//[kbs  ];// +  c1over54 ;(D.f[dirBS  ])[kbs];
			doubflo mfbca = (D.f[dirBN])[kb];//[kbn  ];// +  c1over54 ;(D.f[dirBN  ])[kb ];//kbn
			doubflo mfbac = (D.f[dirTS])[ks];//[kts  ];// +  c1over54 ;(D.f[dirTS  ])[ks ];//kts
			doubflo mfbbb = (D.f[dirZERO])[k];//[kzero];// +  c8over27 ;(D.f[dirZERO])[k  ];//kzero
			doubflo mfccc = (D.f[dirTNE])[k];//[ktne ];// +  c1over216;(D.f[dirTNE ])[k  ];//ktne
			doubflo mfaac = (D.f[dirTSW])[ksw];//[ktsw ];// +  c1over216;(D.f[dirTSW ])[ksw];//ktsw
			doubflo mfcac = (D.f[dirTSE])[ks];//[ktse ];// +  c1over216;(D.f[dirTSE ])[ks ];//ktse
			doubflo mfacc = (D.f[dirTNW])[kw];//[ktnw ];// +  c1over216;(D.f[dirTNW ])[kw ];//ktnw
			doubflo mfcca = (D.f[dirBNE])[kb];//[kbne ];// +  c1over216;(D.f[dirBNE ])[kb ];//kbne
			doubflo mfaaa = (D.f[dirBSW])[kbsw];//[kbsw ];// +  c1over216;(D.f[dirBSW ])[kbsw];
			doubflo mfcaa = (D.f[dirBSE])[kbs];//[kbse ];// +  c1over216;(D.f[dirBSE ])[kbs];//kbse
			doubflo mfaca = (D.f[dirBNW])[kbw];//[kbnw ];// +  c1over216;(D.f[dirBNW ])[kbw];//kbnw
			////////////////////////////////////////////////////////////////////////////////////
			//slow
			//doubflo oMdrho = one - ((((mfccc+mfaaa) + (mfaca+mfcac)) + ((mfacc+mfcaa) + (mfaac+mfcca))) + 
			//					   (((mfbac+mfbca) + (mfbaa+mfbcc)) + ((mfabc+mfcba) + (mfaba+mfcbc)) + ((mfacb+mfcab) + (mfaab+mfccb))) +
			//						((mfabb+mfcbb) + (mfbab+mfbcb)  +  (mfbba+mfbbc)));//fehlt mfbbb
			////////////////////////////////////////////////////////////////////////////////////
			doubflo rho = (mfccc + mfaaa + mfaca + mfcac + mfacc + mfcaa + mfaac + mfcca +
				mfbac + mfbca + mfbaa + mfbcc + mfabc + mfcba + mfaba + mfcbc + mfacb + mfcab + mfaab + mfccb +
				mfabb + mfcbb + mfbab + mfbcb + mfbba + mfbbc + mfbbb + one);//!!!!Achtung + one
////////////////////////////////////////////////////////////////////////////////////
			doubflo vvx = ((((mfccc - mfaaa) + (mfcac - mfaca)) + ((mfcaa - mfacc) + (mfcca - mfaac))) +
				(((mfcba - mfabc) + (mfcbc - mfaba)) + ((mfcab - mfacb) + (mfccb - mfaab))) +
				(mfcbb - mfabb)) / rho;
			doubflo vvy = ((((mfccc - mfaaa) + (mfaca - mfcac)) + ((mfacc - mfcaa) + (mfcca - mfaac))) +
				(((mfbca - mfbac) + (mfbcc - mfbaa)) + ((mfacb - mfcab) + (mfccb - mfaab))) +
				(mfbcb - mfbab)) / rho;
			doubflo vvz = ((((mfccc - mfaaa) + (mfcac - mfaca)) + ((mfacc - mfcaa) + (mfaac - mfcca))) +
				(((mfbac - mfbca) + (mfbcc - mfbaa)) + ((mfabc - mfcba) + (mfcbc - mfaba))) +
				(mfbbc - mfbba)) / rho;
			////////////////////////////////////////////////////////////////////////////////////
			doubflo vx2 = vvx * vvx;
			doubflo vy2 = vvy * vvy;
			doubflo vz2 = vvz * vvz;
			////////////////////////////////////////////////////////////////////////////////////
			doubflo m200 = (mfccc + mfaaa + mfaca + mfcac + mfacc + mfcaa + mfaac + mfcca +
				mfabc + mfcba + mfaba + mfcbc + mfacb + mfcab + mfaab + mfccb +
				mfabb + mfcbb);
			doubflo m020 = (mfccc + mfaaa + mfaca + mfcac + mfacc + mfcaa + mfaac + mfcca +
				mfbac + mfbca + mfbaa + mfbcc + mfacb + mfcab + mfaab + mfccb +
				mfbab + mfbcb);
			doubflo m002 = (mfccc + mfaaa + mfaca + mfcac + mfacc + mfcaa + mfaac + mfcca +
				mfbac + mfbca + mfbaa + mfbcc + mfabc + mfcba + mfaba + mfcbc +
				mfbba + mfbbc);
			////////////////////////////////////////////////////////////////////////////////////
			//Galilei Korrektur
			doubflo Gx = -three * vx2 * (-c1o2 * (three * m200 / rho + one / rho - one - three * vx2)) * (one - omega * c1o2);
			doubflo Gy = -three * vy2 * (-c1o2 * (three * m020 / rho + one / rho - one - three * vy2)) * (one - omega * c1o2);
			doubflo Gz = -three * vz2 * (-c1o2 * (three * m002 / rho + one / rho - one - three * vz2)) * (one - omega * c1o2);
			//doubflo Gx     = zero;
			//doubflo Gy     = zero;
			//doubflo Gz     = zero;
			////////////////////////////////////////////////////////////////////////////////////
			doubflo XXb = -c2o3 + vx2 + Gx;
			doubflo XXc = -c1o2 * (XXb + one + vvx);
			doubflo XXa = XXc + vvx;
			doubflo YYb = -c2o3 + vy2 + Gy;
			doubflo YYc = -c1o2 * (YYb + one + vvy);
			doubflo YYa = YYc + vvy;
			doubflo ZZb = -c2o3 + vz2 + Gz;
			doubflo ZZc = -c1o2 * (ZZb + one + vvz);
			doubflo ZZa = ZZc + vvz;
			////////////////////////////////////////////////////////////////////////////////////
			mfcbb = mfcbb * (one - omega) + omega * (-rho * XXc * YYb * ZZb - c2over27);
			mfabb = mfabb * (one - omega) + omega * (-rho * XXa * YYb * ZZb - c2over27);
			mfbcb = mfbcb * (one - omega) + omega * (-rho * XXb * YYc * ZZb - c2over27);
			mfbab = mfbab * (one - omega) + omega * (-rho * XXb * YYa * ZZb - c2over27);
			mfbbc = mfbbc * (one - omega) + omega * (-rho * XXb * YYb * ZZc - c2over27);
			mfbba = mfbba * (one - omega) + omega * (-rho * XXb * YYb * ZZa - c2over27);
			mfccb = mfccb * (one - omega) + omega * (-rho * XXc * YYc * ZZb - c1over54);
			mfaab = mfaab * (one - omega) + omega * (-rho * XXa * YYa * ZZb - c1over54);
			mfcab = mfcab * (one - omega) + omega * (-rho * XXc * YYa * ZZb - c1over54);
			mfacb = mfacb * (one - omega) + omega * (-rho * XXa * YYc * ZZb - c1over54);
			mfcbc = mfcbc * (one - omega) + omega * (-rho * XXc * YYb * ZZc - c1over54);
			mfaba = mfaba * (one - omega) + omega * (-rho * XXa * YYb * ZZa - c1over54);
			mfcba = mfcba * (one - omega) + omega * (-rho * XXc * YYb * ZZa - c1over54);
			mfabc = mfabc * (one - omega) + omega * (-rho * XXa * YYb * ZZc - c1over54);
			mfbcc = mfbcc * (one - omega) + omega * (-rho * XXb * YYc * ZZc - c1over54);
			mfbaa = mfbaa * (one - omega) + omega * (-rho * XXb * YYa * ZZa - c1over54);
			mfbca = mfbca * (one - omega) + omega * (-rho * XXb * YYc * ZZa - c1over54);
			mfbac = mfbac * (one - omega) + omega * (-rho * XXb * YYa * ZZc - c1over54);
			mfbbb = mfbbb * (one - omega) + omega * (-rho * XXb * YYb * ZZb - c8over27);
			mfccc = mfccc * (one - omega) + omega * (-rho * XXc * YYc * ZZc - c1over216);
			mfaac = mfaac * (one - omega) + omega * (-rho * XXa * YYa * ZZc - c1over216);
			mfcac = mfcac * (one - omega) + omega * (-rho * XXc * YYa * ZZc - c1over216);
			mfacc = mfacc * (one - omega) + omega * (-rho * XXa * YYc * ZZc - c1over216);
			mfcca = mfcca * (one - omega) + omega * (-rho * XXc * YYc * ZZa - c1over216);
			mfaaa = mfaaa * (one - omega) + omega * (-rho * XXa * YYa * ZZa - c1over216);
			mfcaa = mfcaa * (one - omega) + omega * (-rho * XXc * YYa * ZZa - c1over216);
			mfaca = mfaca * (one - omega) + omega * (-rho * XXa * YYc * ZZa - c1over216);
			//			////////////////////////////////////////////////////////////////////////////////////
			//			//fast
			//			doubflo oMdrho = one; //comp special
			//			//doubflo oMdrho = one - (mfccc+mfaaa + mfaca+mfcac + mfacc+mfcaa + mfaac+mfcca + 
			//			//					   mfbac+mfbca + mfbaa+mfbcc + mfabc+mfcba + mfaba+mfcbc + mfacb+mfcab + mfaab+mfccb +
			//			//					   mfabb+mfcbb + mfbab+mfbcb + mfbba+mfbbc + mfbbb + one);//fehlt mfbbb nicht mehr !!!!Achtung + one
			//			////////////////////////////////////////////////////////////////////////////////////
			//			doubflo m0, m1, m2;	
			//			doubflo vx2;
			//			doubflo vy2;
			//			doubflo vz2;
			//			vx2=vvx*vvx;
			//			vy2=vvy*vvy;
			//			vz2=vvz*vvz;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			doubflo wadjust;
			//			doubflo qudricLimit = 0.01f;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			//Hin
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// mit 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36  Konditionieren
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// Z - Dir
			//			m2    = mfaaa	+ mfaac;
			//			m1    = mfaac	- mfaaa;
			//			m0    = m2		+ mfaab;
			//			mfaaa = m0;
			//			m0   += c1o36 * oMdrho;	
			//			mfaab = m1 ;
			//			mfaac = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfaba  + mfabc;
			//			m1    = mfabc  - mfaba;
			//			m0    = m2		+ mfabb;
			//			mfaba = m0;
			//			m0   += c1o9 * oMdrho;
			//			mfabb = m1 ;
			//			mfabc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfaca  + mfacc;
			//			m1    = mfacc  - mfaca;
			//			m0    = m2		+ mfacb;
			//			mfaca = m0;
			//			m0   += c1o36 * oMdrho;
			//			mfacb = m1 ;
			//			mfacc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfbaa	+ mfbac;
			//			m1    = mfbac	- mfbaa;
			//			m0    = m2		+ mfbab;
			//			mfbaa = m0;
			//			m0   += c1o9 * oMdrho;
			//			mfbab = m1 ;
			//			mfbac = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfbba  + mfbbc;
			//			m1    = mfbbc  - mfbba;
			//			m0    = m2		+ mfbbb;
			//			mfbba = m0;
			//			m0   += c4o9 * oMdrho;
			//			mfbbb = m1 ;
			//			mfbbc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfbca  + mfbcc;
			//			m1    = mfbcc  - mfbca;
			//			m0    = m2		+ mfbcb;
			//			mfbca = m0;
			//			m0   += c1o9 * oMdrho;
			//			mfbcb = m1 ;
			//			mfbcc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfcaa	+ mfcac;
			//			m1    = mfcac	- mfcaa;
			//			m0    = m2		+ mfcab;
			//			mfcaa = m0;
			//			m0   += c1o36 * oMdrho;
			//			mfcab = m1 ;
			//			mfcac = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfcba  + mfcbc;
			//			m1    = mfcbc  - mfcba;
			//			m0    = m2		+ mfcbb;
			//			mfcba = m0;
			//			m0   += c1o9 * oMdrho;
			//			mfcbb = m1 ;
			//			mfcbc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfcca  + mfccc;
			//			m1    = mfccc  - mfcca;
			//			m0    = m2		+ mfccb;
			//			mfcca = m0;
			//			m0   += c1o36 * oMdrho;
			//			mfccb = m1 ;
			//			mfccc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// mit  1/6, 0, 1/18, 2/3, 0, 2/9, 1/6, 0, 1/18 Konditionieren
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// Y - Dir
			//			m2    = mfaaa	+ mfaca;
			//			m1    = mfaca	- mfaaa;
			//			m0    = m2		+ mfaba;
			//			mfaaa = m0;
			//			m0   += c1o6 * oMdrho;
			//			mfaba = m1 ;
			//			mfaca = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfaab  + mfacb;
			//			m1    = mfacb  - mfaab;
			//			m0    = m2		+ mfabb;
			//			mfaab = m0;
			//			mfabb = m1 ;
			//			mfacb = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfaac  + mfacc;
			//			m1    = mfacc  - mfaac;
			//			m0    = m2		+ mfabc;
			//			mfaac = m0;
			//			m0   += c1o18 * oMdrho;
			//			mfabc = m1 ;
			//			mfacc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfbaa	+ mfbca;
			//			m1    = mfbca	- mfbaa;
			//			m0    = m2		+ mfbba;
			//			mfbaa = m0;
			//			m0   += c2o3 * oMdrho;
			//			mfbba = m1 ;
			//			mfbca = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfbab  + mfbcb;
			//			m1    = mfbcb  - mfbab;
			//			m0    = m2		+ mfbbb;
			//			mfbab = m0;
			//			mfbbb = m1 ;
			//			mfbcb = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfbac  + mfbcc;
			//			m1    = mfbcc  - mfbac;
			//			m0    = m2		+ mfbbc;
			//			mfbac = m0;
			//			m0   += c2o9 * oMdrho;
			//			mfbbc = m1 ;
			//			mfbcc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfcaa	+ mfcca;
			//			m1    = mfcca	- mfcaa;
			//			m0    = m2		+ mfcba;
			//			mfcaa = m0;
			//			m0   += c1o6 * oMdrho;
			//			mfcba = m1 ;
			//			mfcca = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfcab  + mfccb;
			//			m1    = mfccb  - mfcab;
			//			m0    = m2		+ mfcbb;
			//			mfcab = m0;
			//			mfcbb = m1 ;
			//			mfccb = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfcac  + mfccc;
			//			m1    = mfccc  - mfcac;
			//			m0    = m2		+ mfcbc;
			//			mfcac = m0;
			//			m0   += c1o18 * oMdrho;
			//			mfcbc = m1 ;
			//			mfccc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// mit     1, 0, 1/3, 0, 0, 0, 1/3, 0, 1/9		Konditionieren
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// X - Dir
			//			m2    = mfaaa	+ mfcaa;
			//			m1    = mfcaa	- mfaaa;
			//			m0    = m2		+ mfbaa;
			//			mfaaa = m0;
			//			m0   += one* oMdrho;
			//			mfbaa = m1 ;
			//			mfcaa = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfaba  + mfcba;
			//			m1    = mfcba  - mfaba;
			//			m0    = m2		+ mfbba;
			//			mfaba = m0;
			//			mfbba = m1 ;
			//			mfcba = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfaca  + mfcca;
			//			m1    = mfcca  - mfaca;
			//			m0    = m2		+ mfbca;
			//			mfaca = m0;
			//			m0   += c1o3 * oMdrho;
			//			mfbca = m1 ;
			//			mfcca = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfaab	+ mfcab;
			//			m1    = mfcab	- mfaab;
			//			m0    = m2		+ mfbab;
			//			mfaab = m0;
			//			mfbab = m1 ;
			//			mfcab = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfabb  + mfcbb;
			//			m1    = mfcbb  - mfabb;
			//			m0    = m2		+ mfbbb;
			//			mfabb = m0;
			//			mfbbb = m1 ;
			//			mfcbb = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfacb  + mfccb;
			//			m1    = mfccb  - mfacb;
			//			m0    = m2		+ mfbcb;
			//			mfacb = m0;
			//			mfbcb = m1 ;
			//			mfccb = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfaac	+ mfcac;
			//			m1    = mfcac	- mfaac;
			//			m0    = m2		+ mfbac;
			//			mfaac = m0;
			//			m0   += c1o3 * oMdrho;
			//			mfbac = m1 ;
			//			mfcac = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfabc  + mfcbc;
			//			m1    = mfcbc  - mfabc;
			//			m0    = m2		+ mfbbc;
			//			mfabc = m0;
			//			mfbbc = m1 ;
			//			mfcbc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m2    = mfacc  + mfccc;
			//			m1    = mfccc  - mfacc;
			//			m0    = m2		+ mfbcc;
			//			mfacc = m0;
			//			m0   += c1o9 * oMdrho;
			//			mfbcc = m1 ;
			//			mfccc = m2 ;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//
			//
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// BGK
			//			////////////////////////////////////////////////////////////////////////////////////
			//			doubflo OxxPyyPzz = omega;
			//			doubflo OxyyPxzz  = omega;//two-omega;//eight*(two-omega)/(eight -omega);//one;//omega;//two-omega;//
			//			doubflo OxyyMxzz  = omega;//omega;//one;//eight*(two-omega)/(eight -omega);//one;//two-omega;//one;// 
			//			doubflo O4        = omega;
			//			doubflo O5        = omega;
			//			doubflo O6        = omega;
			//
			//			doubflo mxxPyyPzz = mfcaa + mfaca + mfaac;
			//			doubflo mxxMyy    = mfcaa - mfaca;
			//			doubflo mxxMzz	  = mfcaa - mfaac;
			//
			//			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//			//incl. correction
			//			{
			//				doubflo dxux = c1o2 * (-omega) *(mxxMyy + mxxMzz+(-two*vx2+vy2+vz2)*rho) + c1o2 * OxxPyyPzz * (mfaaa+(vx2+vy2+vz2)*rho - mxxPyyPzz);
			//				doubflo dyuy = dxux + omega * c3o2 * (mxxMyy+(-vx2+vy2)*rho);
			//				doubflo dzuz = dxux + omega * c3o2 * (mxxMzz+(-vx2+vz2)*rho);
			//
			//				//relax
			//				mxxPyyPzz += OxxPyyPzz*(mfaaa +(vx2+vy2+vz2)*rho - mxxPyyPzz)- three * (one - c1o2 * OxxPyyPzz) * (vx2 * dxux + vy2 * dyuy + vz2 * dzuz);
			//				mxxMyy    += omega * ((vx2-vy2)*rho-mxxMyy) - three * (one + c1o2 * (-omega)) * (vx2 * dxux - vy2 * dyuy);
			//				mxxMzz    += omega * ((vx2-vz2)*rho-mxxMzz) - three * (one + c1o2 * (-omega)) * (vx2 * dxux - vz2 * dzuz);
			//			}
			//			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//
			//// 			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//// 			//no correction
			//// 			mxxPyyPzz += OxxPyyPzz*(mfaaa+(vx2+vy2+vz2)*rho-mxxPyyPzz);
			//// 			mxxMyy    += -(-omega) * ((vx2-vy2)*rho-mxxMyy);
			//// 			mxxMzz    += -(-omega) * ((vx2-vz2)*rho-mxxMzz);
			//// 			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//			mfabb     += omega * ((vvy*vvz)*rho-mfabb);
			//			mfbab     += omega * ((vvx*vvz)*rho-mfbab);
			//			mfbba     += omega * ((vvx*vvy)*rho-mfbba);
			//
			//			// linear combinations back
			//			mfcaa = c1o3 * (       mxxMyy +      mxxMzz + mxxPyyPzz);
			//			mfaca = c1o3 * (-two*  mxxMyy +      mxxMzz + mxxPyyPzz);
			//			mfaac = c1o3 * (       mxxMyy - two* mxxMzz + mxxPyyPzz);
			//
			//			//3.
			//			// linear combinations
			//
			//			doubflo mxxyPyzz = mfcba + mfabc;
			//			doubflo mxxyMyzz = mfcba - mfabc;
			//
			//			doubflo mxxzPyyz = mfcab + mfacb;
			//			doubflo mxxzMyyz = mfcab - mfacb;
			//
			//			doubflo mxyyPxzz = mfbca + mfbac;
			//			doubflo mxyyMxzz = mfbca - mfbac;
			//
			//			mxxyMyzz += OxyyMxzz*((vx2-vz2)*vvy*rho-mxxyMyzz);
			//			mxxzMyyz += OxyyMxzz*((vx2-vy2)*vvz*rho-mxxzMyyz);
			//			mxyyMxzz += OxyyMxzz*((vy2-vz2)*vvx*rho-mxyyMxzz);
			//
			//			mxxyPyzz += OxyyPxzz*((c2o3+vx2+vz2)*vvy*rho-mxxyPyzz);
			//			mxxzPyyz += OxyyPxzz*((c2o3+vx2+vy2)*vvz*rho-mxxzPyyz);
			//			mxyyPxzz += OxyyPxzz*((c2o3+vy2+vz2)*vvx*rho-mxyyPxzz);
			//
			//			mfbbb += OxyyMxzz * (vvx*vvy*vvz*rho - mfbbb);
			//			
			//			mfcba = ( mxxyMyzz + mxxyPyzz) * c1o2;
			//			mfabc = (-mxxyMyzz + mxxyPyzz) * c1o2;
			//			mfcab = ( mxxzMyyz + mxxzPyyz) * c1o2;
			//			mfacb = (-mxxzMyyz + mxxzPyyz) * c1o2;
			//			mfbca = ( mxyyMxzz + mxyyPxzz) * c1o2;
			//			mfbac = (-mxyyMxzz + mxyyPxzz) * c1o2;
			//
			//			//4.
			//			//mfacc += O4*((c1o3+vy2)*(c1o3+vz2)*rho+c1o9*(mfaaa-one)-mfacc);
			//			//mfcac += O4*((c1o3+vx2)*(c1o3+vz2)*rho+c1o9*(mfaaa-one)-mfcac);
			//			//mfcca += O4*((c1o3+vx2)*(c1o3+vy2)*rho+c1o9*(mfaaa-one)-mfcca);
			//			mfacc += O4*((c1o3+vy2)*(c1o3+vz2)*rho-c1o9-mfacc);
			//			mfcac += O4*((c1o3+vx2)*(c1o3+vz2)*rho-c1o9-mfcac);
			//			mfcca += O4*((c1o3+vx2)*(c1o3+vy2)*rho-c1o9-mfcca);
			//			
			//			mfcbb += O4*((c1o3+vx2)*vvy*vvz*rho-mfcbb);
			//			mfbcb += O4*((c1o3+vy2)*vvx*vvz*rho-mfbcb);
			//			mfbbc += O4*((c1o3+vz2)*vvx*vvy*rho-mfbbc);
			//
			//			//5.
			//			mfbcc += O5*((c1o3+vy2)*(c1o3+vz2)*vvx*rho-mfbcc);
			//			mfcbc += O5*((c1o3+vx2)*(c1o3+vz2)*vvy*rho-mfcbc);
			//			mfccb += O5*((c1o3+vx2)*(c1o3+vy2)*vvz*rho-mfccb);
			//
			//			//6.
			//			mfccc += O6*((c1o3+vx2)*(c1o3+vy2)*(c1o3+vz2)*rho-c1o27-mfccc);
			//
			//
			//			//bad fix
			//			vvx = zero;
			//			vvy = zero;
			//			vvz = zero;
			//			vx2 = zero;
			//			vy2 = zero;
			//			vz2 = zero;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			//back
			//			////////////////////////////////////////////////////////////////////////////////////
			//			//mit 1, 0, 1/3, 0, 0, 0, 1/3, 0, 1/9   Konditionieren
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// Z - Dir
			//			m0 =  mfaac * c1o2 +      mfaab * (vvz - c1o2) + (mfaaa + one* oMdrho) * (     vz2 - vvz) * c1o2; 
			//			m1 = -mfaac        - two* mfaab *  vvz         +  mfaaa                * (one- vz2)              - one* oMdrho * vz2; 
			//			m2 =  mfaac * c1o2 +      mfaab * (vvz + c1o2) + (mfaaa + one* oMdrho) * (     vz2 + vvz) * c1o2;
			//			mfaaa = m0;
			//			mfaab = m1;
			//			mfaac = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfabc * c1o2 +      mfabb * (vvz - c1o2) + mfaba * (     vz2 - vvz) * c1o2; 
			//			m1 = -mfabc        - two* mfabb *  vvz         + mfaba * (one- vz2); 
			//			m2 =  mfabc * c1o2 +      mfabb * (vvz + c1o2) + mfaba * (     vz2 + vvz) * c1o2;
			//			mfaba = m0;
			//			mfabb = m1;
			//			mfabc = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfacc * c1o2 +      mfacb * (vvz - c1o2) + (mfaca + c1o3 * oMdrho) * (     vz2 - vvz) * c1o2; 
			//			m1 = -mfacc        - two* mfacb *  vvz         +  mfaca                  * (one- vz2)              - c1o3 * oMdrho * vz2; 
			//			m2 =  mfacc * c1o2 +      mfacb * (vvz + c1o2) + (mfaca + c1o3 * oMdrho) * (     vz2 + vvz) * c1o2;
			//			mfaca = m0;
			//			mfacb = m1;
			//			mfacc = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfbac * c1o2 +      mfbab * (vvz - c1o2) + mfbaa * (     vz2 - vvz) * c1o2; 
			//			m1 = -mfbac        - two* mfbab *  vvz         + mfbaa * (one- vz2); 
			//			m2 =  mfbac * c1o2 +      mfbab * (vvz + c1o2) + mfbaa * (     vz2 + vvz) * c1o2;
			//			mfbaa = m0;
			//			mfbab = m1;
			//			mfbac = m2;
			//			/////////b//////////////////////////////////////////////////////////////////////////
			//			m0 =  mfbbc * c1o2 +      mfbbb * (vvz - c1o2) + mfbba * (     vz2 - vvz) * c1o2; 
			//			m1 = -mfbbc        - two* mfbbb *  vvz         + mfbba * (one- vz2); 
			//			m2 =  mfbbc * c1o2 +      mfbbb * (vvz + c1o2) + mfbba * (     vz2 + vvz) * c1o2;
			//			mfbba = m0;
			//			mfbbb = m1;
			//			mfbbc = m2;
			//			/////////b//////////////////////////////////////////////////////////////////////////
			//			m0 =  mfbcc * c1o2 +      mfbcb * (vvz - c1o2) + mfbca * (     vz2 - vvz) * c1o2; 
			//			m1 = -mfbcc        - two* mfbcb *  vvz         + mfbca * (one- vz2); 
			//			m2 =  mfbcc * c1o2 +      mfbcb * (vvz + c1o2) + mfbca * (     vz2 + vvz) * c1o2;
			//			mfbca = m0;
			//			mfbcb = m1;
			//			mfbcc = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfcac * c1o2 +      mfcab * (vvz - c1o2) + (mfcaa + c1o3 * oMdrho) * (     vz2 - vvz) * c1o2; 
			//			m1 = -mfcac        - two* mfcab *  vvz         +  mfcaa                  * (one- vz2)              - c1o3 * oMdrho * vz2; 
			//			m2 =  mfcac * c1o2 +      mfcab * (vvz + c1o2) + (mfcaa + c1o3 * oMdrho) * (     vz2 + vvz) * c1o2;
			//			mfcaa = m0;
			//			mfcab = m1;
			//			mfcac = m2;
			//			/////////c//////////////////////////////////////////////////////////////////////////
			//			m0 =  mfcbc * c1o2 +      mfcbb * (vvz - c1o2) + mfcba * (     vz2 - vvz) * c1o2; 
			//			m1 = -mfcbc        - two* mfcbb *  vvz         + mfcba * (one- vz2); 
			//			m2 =  mfcbc * c1o2 +      mfcbb * (vvz + c1o2) + mfcba * (     vz2 + vvz) * c1o2;
			//			mfcba = m0;
			//			mfcbb = m1;
			//			mfcbc = m2;
			//			/////////c//////////////////////////////////////////////////////////////////////////
			//			m0 =  mfccc * c1o2 +      mfccb * (vvz - c1o2) + (mfcca + c1o9 * oMdrho) * (     vz2 - vvz) * c1o2; 
			//			m1 = -mfccc        - two* mfccb *  vvz         +  mfcca                  * (one- vz2)              - c1o9 * oMdrho * vz2; 
			//			m2 =  mfccc * c1o2 +      mfccb * (vvz + c1o2) + (mfcca + c1o9 * oMdrho) * (     vz2 + vvz) * c1o2;
			//			mfcca = m0;
			//			mfccb = m1;
			//			mfccc = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			//mit 1/6, 2/3, 1/6, 0, 0, 0, 1/18, 2/9, 1/18   Konditionieren
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// Y - Dir
			//			m0 =  mfaca * c1o2 +      mfaba * (vvy - c1o2) + (mfaaa + c1o6 * oMdrho) * (     vy2 - vvy) * c1o2; 
			//			m1 = -mfaca        - two* mfaba *  vvy         +  mfaaa                  * (one- vy2)              - c1o6 * oMdrho * vy2; 
			//			m2 =  mfaca * c1o2 +      mfaba * (vvy + c1o2) + (mfaaa + c1o6 * oMdrho) * (     vy2 + vvy) * c1o2;
			//			mfaaa = m0;
			//			mfaba = m1;
			//			mfaca = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfacb * c1o2 +      mfabb * (vvy - c1o2) + (mfaab + c2o3 * oMdrho) * (     vy2 - vvy) * c1o2; 
			//			m1 = -mfacb        - two* mfabb *  vvy         +  mfaab                  * (one- vy2)              - c2o3 * oMdrho * vy2; 
			//			m2 =  mfacb * c1o2 +      mfabb * (vvy + c1o2) + (mfaab + c2o3 * oMdrho) * (     vy2 + vvy) * c1o2;
			//			mfaab = m0;
			//			mfabb = m1;
			//			mfacb = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfacc * c1o2 +      mfabc * (vvy - c1o2) + (mfaac + c1o6 * oMdrho) * (     vy2 - vvy) * c1o2; 
			//			m1 = -mfacc        - two* mfabc *  vvy         +  mfaac                  * (one- vy2)              - c1o6 * oMdrho * vy2; 
			//			m2 =  mfacc * c1o2 +      mfabc * (vvy + c1o2) + (mfaac + c1o6 * oMdrho) * (     vy2 + vvy) * c1o2;
			//			mfaac = m0;
			//			mfabc = m1;
			//			mfacc = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfbca * c1o2 +      mfbba * (vvy - c1o2) + mfbaa * (     vy2 - vvy) * c1o2; 
			//			m1 = -mfbca        - two* mfbba *  vvy         + mfbaa * (one- vy2); 
			//			m2 =  mfbca * c1o2 +      mfbba * (vvy + c1o2) + mfbaa * (     vy2 + vvy) * c1o2;
			//			mfbaa = m0;
			//			mfbba = m1;
			//			mfbca = m2;
			//			/////////b//////////////////////////////////////////////////////////////////////////
			//			m0 =  mfbcb * c1o2 +      mfbbb * (vvy - c1o2) + mfbab * (     vy2 - vvy) * c1o2; 
			//			m1 = -mfbcb        - two* mfbbb *  vvy         + mfbab * (one- vy2); 
			//			m2 =  mfbcb * c1o2 +      mfbbb * (vvy + c1o2) + mfbab * (     vy2 + vvy) * c1o2;
			//			mfbab = m0;
			//			mfbbb = m1;
			//			mfbcb = m2;
			//			/////////b//////////////////////////////////////////////////////////////////////////
			//			m0 =  mfbcc * c1o2 +      mfbbc * (vvy - c1o2) + mfbac * (     vy2 - vvy) * c1o2; 
			//			m1 = -mfbcc        - two* mfbbc *  vvy         + mfbac * (one- vy2); 
			//			m2 =  mfbcc * c1o2 +      mfbbc * (vvy + c1o2) + mfbac * (     vy2 + vvy) * c1o2;
			//			mfbac = m0;
			//			mfbbc = m1;
			//			mfbcc = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfcca * c1o2 +      mfcba * (vvy - c1o2) + (mfcaa + c1o18 * oMdrho) * (     vy2 - vvy) * c1o2; 
			//			m1 = -mfcca        - two* mfcba *  vvy         +  mfcaa                   * (one- vy2)              - c1o18 * oMdrho * vy2; 
			//			m2 =  mfcca * c1o2 +      mfcba * (vvy + c1o2) + (mfcaa + c1o18 * oMdrho) * (     vy2 + vvy) * c1o2;
			//			mfcaa = m0;
			//			mfcba = m1;
			//			mfcca = m2;
			//			/////////c//////////////////////////////////////////////////////////////////////////
			//			m0 =  mfccb * c1o2 +      mfcbb * (vvy - c1o2) + (mfcab + c2o9 * oMdrho) * (     vy2 - vvy) * c1o2; 
			//			m1 = -mfccb        - two* mfcbb *  vvy         +  mfcab                  * (one- vy2)              - c2o9 * oMdrho * vy2; 
			//			m2 =  mfccb * c1o2 +      mfcbb * (vvy + c1o2) + (mfcab + c2o9 * oMdrho) * (     vy2 + vvy) * c1o2;
			//			mfcab = m0;
			//			mfcbb = m1;
			//			mfccb = m2;
			//			/////////c//////////////////////////////////////////////////////////////////////////
			//			m0 =  mfccc * c1o2 +      mfcbc * (vvy - c1o2) + (mfcac + c1o18 * oMdrho) * (     vy2 - vvy) * c1o2; 
			//			m1 = -mfccc        - two* mfcbc *  vvy         +  mfcac                   * (one- vy2)              - c1o18 * oMdrho * vy2; 
			//			m2 =  mfccc * c1o2 +      mfcbc * (vvy + c1o2) + (mfcac + c1o18 * oMdrho) * (     vy2 + vvy) * c1o2;
			//			mfcac = m0;
			//			mfcbc = m1;
			//			mfccc = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			//mit 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36 Konditionieren
			//			////////////////////////////////////////////////////////////////////////////////////
			//			// X - Dir
			//			m0 =  mfcaa * c1o2 +      mfbaa * (vvx - c1o2) + (mfaaa + c1o36 * oMdrho) * (     vx2 - vvx) * c1o2; 
			//			m1 = -mfcaa        - two* mfbaa *  vvx         +  mfaaa                   * (one- vx2)              - c1o36 * oMdrho * vx2; 
			//			m2 =  mfcaa * c1o2 +      mfbaa * (vvx + c1o2) + (mfaaa + c1o36 * oMdrho) * (     vx2 + vvx) * c1o2;
			//			mfaaa = m0;
			//			mfbaa = m1;
			//			mfcaa = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfcba * c1o2 +      mfbba * (vvx - c1o2) + (mfaba + c1o9 * oMdrho) * (     vx2 - vvx) * c1o2; 
			//			m1 = -mfcba        - two* mfbba *  vvx         +  mfaba                  * (one- vx2)              - c1o9 * oMdrho * vx2; 
			//			m2 =  mfcba * c1o2 +      mfbba * (vvx + c1o2) + (mfaba + c1o9 * oMdrho) * (     vx2 + vvx) * c1o2;
			//			mfaba = m0;
			//			mfbba = m1;
			//			mfcba = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfcca * c1o2 +      mfbca * (vvx - c1o2) + (mfaca + c1o36 * oMdrho) * (     vx2 - vvx) * c1o2; 
			//			m1 = -mfcca        - two* mfbca *  vvx         +  mfaca                   * (one- vx2)              - c1o36 * oMdrho * vx2; 
			//			m2 =  mfcca * c1o2 +      mfbca * (vvx + c1o2) + (mfaca + c1o36 * oMdrho) * (     vx2 + vvx) * c1o2;
			//			mfaca = m0;
			//			mfbca = m1;
			//			mfcca = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfcab * c1o2 +      mfbab * (vvx - c1o2) + (mfaab + c1o9 * oMdrho) * (     vx2 - vvx) * c1o2; 
			//			m1 = -mfcab        - two* mfbab *  vvx         +  mfaab                  * (one- vx2)              - c1o9 * oMdrho * vx2; 
			//			m2 =  mfcab * c1o2 +      mfbab * (vvx + c1o2) + (mfaab + c1o9 * oMdrho) * (     vx2 + vvx) * c1o2;
			//			mfaab = m0;
			//			mfbab = m1;
			//			mfcab = m2;
			//			///////////b////////////////////////////////////////////////////////////////////////
			//			m0 =  mfcbb * c1o2 +      mfbbb * (vvx - c1o2) + (mfabb + c4o9 * oMdrho) * (     vx2 - vvx) * c1o2; 
			//			m1 = -mfcbb        - two* mfbbb *  vvx         +  mfabb                  * (one- vx2)              - c4o9 * oMdrho * vx2; 
			//			m2 =  mfcbb * c1o2 +      mfbbb * (vvx + c1o2) + (mfabb + c4o9 * oMdrho) * (     vx2 + vvx) * c1o2;
			//			mfabb = m0;
			//			mfbbb = m1;
			//			mfcbb = m2;
			//			///////////b////////////////////////////////////////////////////////////////////////
			//			m0 =  mfccb * c1o2 +      mfbcb * (vvx - c1o2) + (mfacb + c1o9 * oMdrho) * (     vx2 - vvx) * c1o2; 
			//			m1 = -mfccb        - two* mfbcb *  vvx         +  mfacb                  * (one- vx2)              - c1o9 * oMdrho * vx2; 
			//			m2 =  mfccb * c1o2 +      mfbcb * (vvx + c1o2) + (mfacb + c1o9 * oMdrho) * (     vx2 + vvx) * c1o2;
			//			mfacb = m0;
			//			mfbcb = m1;
			//			mfccb = m2;
			//			////////////////////////////////////////////////////////////////////////////////////
			//			////////////////////////////////////////////////////////////////////////////////////
			//			m0 =  mfcac * c1o2 +      mfbac * (vvx - c1o2) + (mfaac + c1o36 * oMdrho) * (     vx2 - vvx) * c1o2; 
			//			m1 = -mfcac        - two* mfbac *  vvx         +  mfaac                   * (one- vx2)              - c1o36 * oMdrho * vx2; 
			//			m2 =  mfcac * c1o2 +      mfbac * (vvx + c1o2) + (mfaac + c1o36 * oMdrho) * (     vx2 + vvx) * c1o2;
			//			mfaac = m0;
			//			mfbac = m1;
			//			mfcac = m2;
			//			///////////c////////////////////////////////////////////////////////////////////////
			//			m0 =  mfcbc * c1o2 +      mfbbc * (vvx - c1o2) + (mfabc + c1o9 * oMdrho) * (     vx2 - vvx) * c1o2; 
			//			m1 = -mfcbc        - two* mfbbc *  vvx         +  mfabc                  * (one- vx2)              - c1o9 * oMdrho * vx2; 
			//			m2 =  mfcbc * c1o2 +      mfbbc * (vvx + c1o2) + (mfabc + c1o9 * oMdrho) * (     vx2 + vvx) * c1o2;
			//			mfabc = m0;
			//			mfbbc = m1;
			//			mfcbc = m2;
			//			///////////c////////////////////////////////////////////////////////////////////////
			//			m0 =  mfccc * c1o2 +      mfbcc * (vvx - c1o2) + (mfacc + c1o36 * oMdrho) * (     vx2 - vvx) * c1o2; 
			//			m1 = -mfccc        - two* mfbcc *  vvx         +  mfacc                   * (one- vx2)              - c1o36 * oMdrho * vx2; 
			//			m2 =  mfccc * c1o2 +      mfbcc * (vvx + c1o2) + (mfacc + c1o36 * oMdrho) * (     vx2 + vvx) * c1o2;
			//			mfacc = m0;
			//			mfbcc = m1;
			//			mfccc = m2;
			//			////////////////////////////////////////////////////////////////////////////////////


						////////////////////////////////////////////////////////////////////////////////////
			(D.f[dirE])[k] = mfabb;//(D.f[ dirE   ])[ke   ] = mfabb;// -  c2over27 ;  (D.f[ dirE   ])[k   ]                                                                     
			(D.f[dirW])[kw] = mfcbb;//(D.f[ dirW   ])[kw   ] = mfcbb;// -  c2over27 ;  (D.f[ dirW   ])[kw  ]                                                                   
			(D.f[dirN])[k] = mfbab;//(D.f[ dirN   ])[kn   ] = mfbab;// -  c2over27 ;	 (D.f[ dirN   ])[k   ]
			(D.f[dirS])[ks] = mfbcb;//(D.f[ dirS   ])[ks   ] = mfbcb;// -  c2over27 ;	 (D.f[ dirS   ])[ks  ]
			(D.f[dirT])[k] = mfbba;//(D.f[ dirT   ])[kt   ] = mfbba;// -  c2over27 ;	 (D.f[ dirT   ])[k   ]
			(D.f[dirB])[kb] = mfbbc;//(D.f[ dirB   ])[kb   ] = mfbbc;// -  c2over27 ;	 (D.f[ dirB   ])[kb  ]
			(D.f[dirNE])[k] = mfaab;//(D.f[ dirNE  ])[kne  ] = mfaab;// -  c1over54 ;	 (D.f[ dirNE  ])[k   ]
			(D.f[dirSW])[ksw] = mfccb;//(D.f[ dirSW  ])[ksw  ] = mfccb;// -  c1over54 ;	 (D.f[ dirSW  ])[ksw ]
			(D.f[dirSE])[ks] = mfacb;//(D.f[ dirSE  ])[kse  ] = mfacb;// -  c1over54 ;	 (D.f[ dirSE  ])[ks  ]
			(D.f[dirNW])[kw] = mfcab;//(D.f[ dirNW  ])[knw  ] = mfcab;// -  c1over54 ;	 (D.f[ dirNW  ])[kw  ]
			(D.f[dirTE])[k] = mfaba;//(D.f[ dirTE  ])[kte  ] = mfaba;// -  c1over54 ;	 (D.f[ dirTE  ])[k   ]
			(D.f[dirBW])[kbw] = mfcbc;//(D.f[ dirBW  ])[kbw  ] = mfcbc;// -  c1over54 ;	 (D.f[ dirBW  ])[kbw ]
			(D.f[dirBE])[kb] = mfabc;//(D.f[ dirBE  ])[kbe  ] = mfabc;// -  c1over54 ;	 (D.f[ dirBE  ])[kb  ]
			(D.f[dirTW])[kw] = mfcba;//(D.f[ dirTW  ])[ktw  ] = mfcba;// -  c1over54 ;	 (D.f[ dirTW  ])[kw  ]
			(D.f[dirTN])[k] = mfbaa;//(D.f[ dirTN  ])[ktn  ] = mfbaa;// -  c1over54 ;	 (D.f[ dirTN  ])[k   ]
			(D.f[dirBS])[kbs] = mfbcc;//(D.f[ dirBS  ])[kbs  ] = mfbcc;// -  c1over54 ;	 (D.f[ dirBS  ])[kbs ]
			(D.f[dirBN])[kb] = mfbac;//(D.f[ dirBN  ])[kbn  ] = mfbac;// -  c1over54 ;	 (D.f[ dirBN  ])[kb  ]
			(D.f[dirTS])[ks] = mfbca;//(D.f[ dirTS  ])[kts  ] = mfbca;// -  c1over54 ;	 (D.f[ dirTS  ])[ks  ]
			(D.f[dirZERO])[k] = mfbbb;//(D.f[ dirZERO])[kzero] = mfbbb;// -  c8over27 ;	 (D.f[ dirZERO])[k   ]
			(D.f[dirTNE])[k] = mfaaa;//(D.f[ dirTNE ])[ktne ] = mfaaa;// -  c1over216;	 (D.f[ dirTNE ])[k   ]
			(D.f[dirTSE])[ks] = mfaca;//(D.f[ dirTSE ])[ktse ] = mfaca;// -  c1over216;	 (D.f[ dirTSE ])[ks  ]
			(D.f[dirBNE])[kb] = mfaac;//(D.f[ dirBNE ])[kbne ] = mfaac;// -  c1over216;	 (D.f[ dirBNE ])[kb  ]
			(D.f[dirBSE])[kbs] = mfacc;//(D.f[ dirBSE ])[kbse ] = mfacc;// -  c1over216;	 (D.f[ dirBSE ])[kbs ]
			(D.f[dirTNW])[kw] = mfcaa;//(D.f[ dirTNW ])[ktnw ] = mfcaa;// -  c1over216;	 (D.f[ dirTNW ])[kw  ]
			(D.f[dirTSW])[ksw] = mfcca;//(D.f[ dirTSW ])[ktsw ] = mfcca;// -  c1over216;	 (D.f[ dirTSW ])[ksw ]
			(D.f[dirBNW])[kbw] = mfcac;//(D.f[ dirBNW ])[kbnw ] = mfcac;// -  c1over216;	 (D.f[ dirBNW ])[kbw ]
			(D.f[dirBSW])[kbsw] = mfccc;//(D.f[ dirBSW ])[kbsw ] = mfccc;// -  c1over216;	 (D.f[ dirBSW ])[kbsw]
			////////////////////////////////////////////////////////////////////////////////////
		}
	}
}
////////////////////////////////////////////////////////////////////////////////












































////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void LB_Kernel_BGK_Comp_SP_27(doubflo omega,
													unsigned int* bcMatD,
													unsigned int* neighborX,
													unsigned int* neighborY,
													unsigned int* neighborZ,
													doubflo* DDStart,
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
	///////////////////////////////////////////////////////////////////////////////

	if (k < size_Mat)
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
			//unsigned int kzero= k;
			//unsigned int ke   = k;
			unsigned int kw = neighborX[k];
			//unsigned int kn   = k;
			unsigned int ks = neighborY[k];
			//unsigned int kt   = k;
			unsigned int kb = neighborZ[k];
			unsigned int ksw = neighborY[kw];
			//unsigned int kne  = k;
			//unsigned int kse  = ks;
			//unsigned int knw  = kw;
			unsigned int kbw = neighborZ[kw];
			//unsigned int kte  = k;
			//unsigned int kbe  = kb;
			//unsigned int ktw  = kw;
			unsigned int kbs = neighborZ[ks];
			//unsigned int ktn  = k;
			//unsigned int kbn  = kb;
			//unsigned int kts  = ks;
			//unsigned int ktse = ks;
			//unsigned int kbnw = kbw;
			//unsigned int ktnw = kw;
			//unsigned int kbse = kbs;
			//unsigned int ktsw = ksw;
			//unsigned int kbne = kb;
			//unsigned int ktne = k;
			unsigned int kbsw = neighborZ[ksw];
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			doubflo fE = (D.f[dirE])[k];//ke
			doubflo fW = (D.f[dirW])[kw];
			doubflo fN = (D.f[dirN])[k];//kn
			doubflo fS = (D.f[dirS])[ks];
			doubflo fT = (D.f[dirT])[k];//kt
			doubflo fB = (D.f[dirB])[kb];
			doubflo fNE = (D.f[dirNE])[k];//kne
			doubflo fSW = (D.f[dirSW])[ksw];
			doubflo fSE = (D.f[dirSE])[ks];//kse
			doubflo fNW = (D.f[dirNW])[kw];//knw
			doubflo fTE = (D.f[dirTE])[k];//kte
			doubflo fBW = (D.f[dirBW])[kbw];
			doubflo fBE = (D.f[dirBE])[kb];//kbe
			doubflo fTW = (D.f[dirTW])[kw];//ktw
			doubflo fTN = (D.f[dirTN])[k];//ktn
			doubflo fBS = (D.f[dirBS])[kbs];
			doubflo fBN = (D.f[dirBN])[kb];//kbn
			doubflo fTS = (D.f[dirTS])[ks];//kts
			doubflo fZERO = (D.f[dirZERO])[k];//kzero
			doubflo fTNE = (D.f[dirTNE])[k];//ktne
			doubflo fTSW = (D.f[dirTSW])[ksw];//ktsw
			doubflo fTSE = (D.f[dirTSE])[ks];//ktse
			doubflo fTNW = (D.f[dirTNW])[kw];//ktnw
			doubflo fBNE = (D.f[dirBNE])[kb];//kbne
			doubflo fBSW = (D.f[dirBSW])[kbsw];
			doubflo fBSE = (D.f[dirBSE])[kbs];//kbse
			doubflo fBNW = (D.f[dirBNW])[kbw];//kbnw
			////////////////////////////////////////////////////////////////////////////////
			doubflo drho = (fTNE + fBSW) + (fTSW + fBNE) + (fTSE + fBNW) + (fTNW + fBSE) + (fNE + fSW) + (fNW + fSE) + (fTE + fBW) + (fBE + fTW) + (fTN + fBS) + (fBN + fTS) + (fE + fW) + (fN + fS) + (fT + fB) + fZERO;
			doubflo rho = drho + one;
			doubflo OORho = one / rho;
			doubflo vx1 = OORho*((fTNE - fBSW) + (fBNE - fTSW) + (fTSE - fBNW) + (fBSE - fTNW) + (fNE - fSW) + (fSE - fNW) + (fTE - fBW) + (fBE - fTW) + (fE - fW));
			doubflo vx2 = OORho*((fTNE - fBSW) + (fBNE - fTSW) + (fBNW - fTSE) + (fTNW - fBSE) + (fNE - fSW) + (fNW - fSE) + (fTN - fBS) + (fBN - fTS) + (fN - fS));
			doubflo vx3 = OORho*((fTNE - fBSW) + (fTSW - fBNE) + (fTSE - fBNW) + (fTNW - fBSE) + (fTE - fBW) + (fTW - fBE) + (fTN - fBS) + (fTS - fBN) + (fT - fB));
			////////////////////////////////////////////////////////////////////////////////







			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//BGK comp
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			doubflo cusq = c3o2*(vx1*vx1 + vx2*vx2 + vx3*vx3);
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			fZERO = fZERO *(one + (-omega)) - (-omega)*   c8over27*  (drho - rho * cusq);
			fE    = fE    *(one + (-omega)) - (-omega)*   c2over27*  (drho + rho * (three*(vx1)+c9over2*(vx1)*(vx1)-cusq));
			fW    = fW    *(one + (-omega)) - (-omega)*   c2over27*  (drho + rho * (three*(-vx1) + c9over2*(-vx1)*(-vx1) - cusq));
			fN    = fN    *(one + (-omega)) - (-omega)*   c2over27*  (drho + rho * (three*(vx2)+c9over2*(vx2)*(vx2)-cusq));
			fS    = fS    *(one + (-omega)) - (-omega)*   c2over27*  (drho + rho * (three*(-vx2) + c9over2*(-vx2)*(-vx2) - cusq));
			fT    = fT    *(one + (-omega)) - (-omega)*   c2over27*  (drho + rho * (three*(vx3)+c9over2*(vx3)*(vx3)-cusq));
			fB    = fB    *(one + (-omega)) - (-omega)*   c2over27*  (drho + rho * (three*(-vx3) + c9over2*(-vx3)*(-vx3) - cusq));
			fNE   = fNE   *(one + (-omega)) - (-omega)*   c1over54*  (drho + rho * (three*(vx1 + vx2) + c9over2*(vx1 + vx2)*(vx1 + vx2) - cusq));
			fSW   = fSW   *(one + (-omega)) - (-omega)*   c1over54*  (drho + rho * (three*(-vx1 - vx2) + c9over2*(-vx1 - vx2)*(-vx1 - vx2) - cusq));
			fSE   = fSE   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(vx1 - vx2) + c9over2*(vx1 - vx2)*(vx1 - vx2) - cusq));
			fNW   = fNW   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(-vx1 + vx2) + c9over2*(-vx1 + vx2)*(-vx1 + vx2) - cusq));
			fTE   = fTE   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(vx1 + vx3) + c9over2*(vx1 + vx3)*(vx1 + vx3) - cusq));
			fBW   = fBW   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(-vx1 - vx3) + c9over2*(-vx1 - vx3)*(-vx1 - vx3) - cusq));
			fBE   = fBE   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(vx1 - vx3) + c9over2*(vx1 - vx3)*(vx1 - vx3) - cusq));
			fTW   = fTW   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(-vx1 + vx3) + c9over2*(-vx1 + vx3)*(-vx1 + vx3) - cusq));
			fTN   = fTN   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(vx2 + vx3) + c9over2*(vx2 + vx3)*(vx2 + vx3) - cusq));
			fBS   = fBS   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(-vx2 - vx3) + c9over2*(-vx2 - vx3)*(-vx2 - vx3) - cusq));
			fBN   = fBN   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(vx2 - vx3) + c9over2*(vx2 - vx3)*(vx2 - vx3) - cusq));
			fTS   = fTS   *(one + (-omega)) - (-omega)*    c1over54* (drho + rho * (three*(-vx2 + vx3) + c9over2*(-vx2 + vx3)*(-vx2 + vx3) - cusq));
			fTNE  = fTNE  *(one + (-omega)) - (-omega)*    c1over216*(drho + rho * (three*(vx1 + vx2 + vx3) + c9over2*(vx1 + vx2 + vx3)*(vx1 + vx2 + vx3) - cusq));
			fBSW  = fBSW  *(one + (-omega)) - (-omega)*    c1over216*(drho + rho * (three*(-vx1 - vx2 - vx3) + c9over2*(-vx1 - vx2 - vx3)*(-vx1 - vx2 - vx3) - cusq));
			fBNE  = fBNE  *(one + (-omega)) - (-omega)*    c1over216*(drho + rho * (three*(vx1 + vx2 - vx3) + c9over2*(vx1 + vx2 - vx3)*(vx1 + vx2 - vx3) - cusq));
			fTSW  = fTSW  *(one + (-omega)) - (-omega)*    c1over216*(drho + rho * (three*(-vx1 - vx2 + vx3) + c9over2*(-vx1 - vx2 + vx3)*(-vx1 - vx2 + vx3) - cusq));
			fTSE  = fTSE  *(one + (-omega)) - (-omega)*    c1over216*(drho + rho * (three*(vx1 - vx2 + vx3) + c9over2*(vx1 - vx2 + vx3)*(vx1 - vx2 + vx3) - cusq));
			fBNW  = fBNW  *(one + (-omega)) - (-omega)*    c1over216*(drho + rho * (three*(-vx1 + vx2 - vx3) + c9over2*(-vx1 + vx2 - vx3)*(-vx1 + vx2 - vx3) - cusq));
			fBSE  = fBSE  *(one + (-omega)) - (-omega)*    c1over216*(drho + rho * (three*(vx1 - vx2 - vx3) + c9over2*(vx1 - vx2 - vx3)*(vx1 - vx2 - vx3) - cusq));
			fTNW  = fTNW  *(one + (-omega)) - (-omega)*    c1over216*(drho + rho * (three*(-vx1 + vx2 + vx3) + c9over2*(-vx1 + vx2 + vx3)*(-vx1 + vx2 + vx3) - cusq));
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////







			//////////////////////////////////////////////////////////////////////////                            
			(D.f[ dirE   ])[k   ] = fW    ;                                                                     
			(D.f[ dirW   ])[kw  ] = fE    ;                                                                     
			(D.f[ dirN   ])[k   ] = fS    ;
			(D.f[ dirS   ])[ks  ] = fN    ;
			(D.f[ dirT   ])[k   ] = fB    ;
			(D.f[ dirB   ])[kb  ] = fT    ;
			(D.f[ dirNE  ])[k   ] = fSW   ;
			(D.f[ dirSW  ])[ksw ] = fNE   ;
			(D.f[ dirSE  ])[ks  ] = fNW   ;
			(D.f[ dirNW  ])[kw  ] = fSE   ;
			(D.f[ dirTE  ])[k   ] = fBW   ;
			(D.f[ dirBW  ])[kbw ] = fTE   ;
			(D.f[ dirBE  ])[kb  ] = fTW   ;
			(D.f[ dirTW  ])[kw  ] = fBE   ;
			(D.f[ dirTN  ])[k   ] = fBS   ;
			(D.f[ dirBS  ])[kbs ] = fTN   ;
			(D.f[ dirBN  ])[kb  ] = fTS   ;
			(D.f[ dirTS  ])[ks  ] = fBN   ;
			(D.f[ dirZERO])[k   ] = fZERO ;
			(D.f[ dirTNE ])[k   ] = fBSW  ;
			(D.f[ dirTSE ])[ks  ] = fBNW  ;
			(D.f[ dirBNE ])[kb  ] = fTSW  ;
			(D.f[ dirBSE ])[kbs ] = fTNW  ;
			(D.f[ dirTNW ])[kw  ] = fBSE  ;
			(D.f[ dirTSW ])[ksw ] = fBNE  ;
			(D.f[ dirBNW ])[kbw ] = fTSE  ;
			(D.f[ dirBSW ])[kbsw] = fTNE  ;
			//////////////////////////////////////////////////////////////////////////                            
		}
	}
}
////////////////////////////////////////////////////////////////////////////////











































////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void LB_Kernel_BGK_SP_27( doubflo omega,
												unsigned int* bcMatD,
												unsigned int* neighborX,
												unsigned int* neighborY,
												unsigned int* neighborZ,
												doubflo* DDStart,
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

	if (k < size_Mat)
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
			//unsigned int kzero= k;
			//unsigned int ke   = k;
			unsigned int kw = neighborX[k];
			//unsigned int kn   = k;
			unsigned int ks = neighborY[k];
			//unsigned int kt   = k;
			unsigned int kb = neighborZ[k];
			unsigned int ksw = neighborY[kw];
			//unsigned int kne  = k;
			//unsigned int kse  = ks;
			//unsigned int knw  = kw;
			unsigned int kbw = neighborZ[kw];
			//unsigned int kte  = k;
			//unsigned int kbe  = kb;
			//unsigned int ktw  = kw;
			unsigned int kbs = neighborZ[ks];
			//unsigned int ktn  = k;
			//unsigned int kbn  = kb;
			//unsigned int kts  = ks;
			//unsigned int ktse = ks;
			//unsigned int kbnw = kbw;
			//unsigned int ktnw = kw;
			//unsigned int kbse = kbs;
			//unsigned int ktsw = ksw;
			//unsigned int kbne = kb;
			//unsigned int ktne = k;
			unsigned int kbsw = neighborZ[ksw];
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			doubflo fE = (D.f[dirE])[k];//ke
			doubflo fW = (D.f[dirW])[kw];
			doubflo fN = (D.f[dirN])[k];//kn
			doubflo fS = (D.f[dirS])[ks];
			doubflo fT = (D.f[dirT])[k];//kt
			doubflo fB = (D.f[dirB])[kb];
			doubflo fNE = (D.f[dirNE])[k];//kne
			doubflo fSW = (D.f[dirSW])[ksw];
			doubflo fSE = (D.f[dirSE])[ks];//kse
			doubflo fNW = (D.f[dirNW])[kw];//knw
			doubflo fTE = (D.f[dirTE])[k];//kte
			doubflo fBW = (D.f[dirBW])[kbw];
			doubflo fBE = (D.f[dirBE])[kb];//kbe
			doubflo fTW = (D.f[dirTW])[kw];//ktw
			doubflo fTN = (D.f[dirTN])[k];//ktn
			doubflo fBS = (D.f[dirBS])[kbs];
			doubflo fBN = (D.f[dirBN])[kb];//kbn
			doubflo fTS = (D.f[dirTS])[ks];//kts
			doubflo fZERO = (D.f[dirZERO])[k];//kzero
			doubflo fTNE = (D.f[dirTNE])[k];//ktne
			doubflo fTSW = (D.f[dirTSW])[ksw];//ktsw
			doubflo fTSE = (D.f[dirTSE])[ks];//ktse
			doubflo fTNW = (D.f[dirTNW])[kw];//ktnw
			doubflo fBNE = (D.f[dirBNE])[kb];//kbne
			doubflo fBSW = (D.f[dirBSW])[kbsw];
			doubflo fBSE = (D.f[dirBSE])[kbs];//kbse
			doubflo fBNW = (D.f[dirBNW])[kbw];//kbnw
			////////////////////////////////////////////////////////////////////////////////







			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//BGK incomp
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			doubflo drho    = (fTNE + fBSW) + (fTSW + fBNE) + (fTSE + fBNW) + (fTNW + fBSE) + (fNE + fSW) + (fNW + fSE) + (fTE + fBW) + (fBE + fTW) + (fTN + fBS) + (fBN + fTS) + (fE + fW) + (fN + fS) + (fT + fB) + fZERO;
			doubflo vx1		= (fTNE - fBSW) + (fBNE - fTSW) + (fTSE - fBNW) + (fBSE - fTNW) + (fNE - fSW) + (fSE - fNW) + (fTE - fBW) + (fBE - fTW) + (fE - fW);
			doubflo vx2		= (fTNE - fBSW) + (fBNE - fTSW) + (fBNW - fTSE) + (fTNW - fBSE) + (fNE - fSW) + (fNW - fSE) + (fTN - fBS) + (fBN - fTS) + (fN - fS);
			doubflo vx3		= (fTNE - fBSW) + (fTSW - fBNE) + (fTSE - fBNW) + (fTNW - fBSE) + (fTE - fBW) + (fTW - fBE) + (fTN - fBS) + (fTS - fBN) + (fT - fB);
			doubflo cusq    =  c3o2*(vx1*vx1+vx2*vx2+vx3*vx3);
			//////////////////////////////////////////////////////////////////////////                            
			fZERO = fZERO *(one+(-omega))-(-omega)*   c8over27*  (drho-cusq);
			fE    = fE    *(one+(-omega))-(-omega)*   c2over27*  (drho+three*( vx1        )+c9over2*( vx1        )*( vx1        )-cusq);
			fW    = fW    *(one+(-omega))-(-omega)*   c2over27*  (drho+three*(-vx1        )+c9over2*(-vx1        )*(-vx1        )-cusq);
			fN    = fN    *(one+(-omega))-(-omega)*   c2over27*  (drho+three*(    vx2     )+c9over2*(     vx2    )*(     vx2    )-cusq);
			fS    = fS    *(one+(-omega))-(-omega)*   c2over27*  (drho+three*(   -vx2     )+c9over2*(    -vx2    )*(    -vx2    )-cusq);
			fT    = fT    *(one+(-omega))-(-omega)*   c2over27*  (drho+three*(         vx3)+c9over2*(         vx3)*(         vx3)-cusq);
			fB    = fB    *(one+(-omega))-(-omega)*   c2over27*  (drho+three*(        -vx3)+c9over2*(        -vx3)*(        -vx3)-cusq);
			fNE   = fNE   *(one+(-omega))-(-omega)*   c1over54*  (drho+three*( vx1+vx2    )+c9over2*( vx1+vx2    )*( vx1+vx2    )-cusq);
			fSW   = fSW   *(one+(-omega))-(-omega)*   c1over54*  (drho+three*(-vx1-vx2    )+c9over2*(-vx1-vx2    )*(-vx1-vx2    )-cusq);
			fSE   = fSE   *(one+(-omega))-(-omega)*    c1over54* (drho+three*( vx1-vx2    )+c9over2*( vx1-vx2    )*( vx1-vx2    )-cusq);
			fNW   = fNW   *(one+(-omega))-(-omega)*    c1over54* (drho+three*(-vx1+vx2    )+c9over2*(-vx1+vx2    )*(-vx1+vx2    )-cusq);
			fTE   = fTE   *(one+(-omega))-(-omega)*    c1over54* (drho+three*( vx1    +vx3)+c9over2*( vx1    +vx3)*( vx1    +vx3)-cusq);
			fBW   = fBW   *(one+(-omega))-(-omega)*    c1over54* (drho+three*(-vx1    -vx3)+c9over2*(-vx1    -vx3)*(-vx1    -vx3)-cusq);
			fBE   = fBE   *(one+(-omega))-(-omega)*    c1over54* (drho+three*( vx1    -vx3)+c9over2*( vx1    -vx3)*( vx1    -vx3)-cusq);
			fTW   = fTW   *(one+(-omega))-(-omega)*    c1over54* (drho+three*(-vx1    +vx3)+c9over2*(-vx1    +vx3)*(-vx1    +vx3)-cusq);
			fTN   = fTN   *(one+(-omega))-(-omega)*    c1over54* (drho+three*(     vx2+vx3)+c9over2*(     vx2+vx3)*(     vx2+vx3)-cusq);
			fBS   = fBS   *(one+(-omega))-(-omega)*    c1over54* (drho+three*(    -vx2-vx3)+c9over2*(    -vx2-vx3)*(    -vx2-vx3)-cusq);
			fBN   = fBN   *(one+(-omega))-(-omega)*    c1over54* (drho+three*(     vx2-vx3)+c9over2*(     vx2-vx3)*(     vx2-vx3)-cusq);
			fTS   = fTS   *(one+(-omega))-(-omega)*    c1over54* (drho+three*(    -vx2+vx3)+c9over2*(    -vx2+vx3)*(    -vx2+vx3)-cusq);
			fTNE  = fTNE  *(one+(-omega))-(-omega)*    c1over216*(drho+three*( vx1+vx2+vx3)+c9over2*( vx1+vx2+vx3)*( vx1+vx2+vx3)-cusq);
			fBSW  = fBSW  *(one+(-omega))-(-omega)*    c1over216*(drho+three*(-vx1-vx2-vx3)+c9over2*(-vx1-vx2-vx3)*(-vx1-vx2-vx3)-cusq);
			fBNE  = fBNE  *(one+(-omega))-(-omega)*    c1over216*(drho+three*( vx1+vx2-vx3)+c9over2*( vx1+vx2-vx3)*( vx1+vx2-vx3)-cusq);
			fTSW  = fTSW  *(one+(-omega))-(-omega)*    c1over216*(drho+three*(-vx1-vx2+vx3)+c9over2*(-vx1-vx2+vx3)*(-vx1-vx2+vx3)-cusq);
			fTSE  = fTSE  *(one+(-omega))-(-omega)*    c1over216*(drho+three*( vx1-vx2+vx3)+c9over2*( vx1-vx2+vx3)*( vx1-vx2+vx3)-cusq);
			fBNW  = fBNW  *(one+(-omega))-(-omega)*    c1over216*(drho+three*(-vx1+vx2-vx3)+c9over2*(-vx1+vx2-vx3)*(-vx1+vx2-vx3)-cusq);
			fBSE  = fBSE  *(one+(-omega))-(-omega)*    c1over216*(drho+three*( vx1-vx2-vx3)+c9over2*( vx1-vx2-vx3)*( vx1-vx2-vx3)-cusq);
			fTNW  = fTNW  *(one+(-omega))-(-omega)*    c1over216*(drho+three*(-vx1+vx2+vx3)+c9over2*(-vx1+vx2+vx3)*(-vx1+vx2+vx3)-cusq);
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////







			//////////////////////////////////////////////////////////////////////////                            
			(D.f[dirE])[k] = fW;
			(D.f[dirW])[kw] = fE;
			(D.f[dirN])[k] = fS;
			(D.f[dirS])[ks] = fN;
			(D.f[dirT])[k] = fB;
			(D.f[dirB])[kb] = fT;
			(D.f[dirNE])[k] = fSW;
			(D.f[dirSW])[ksw] = fNE;
			(D.f[dirSE])[ks] = fNW;
			(D.f[dirNW])[kw] = fSE;
			(D.f[dirTE])[k] = fBW;
			(D.f[dirBW])[kbw] = fTE;
			(D.f[dirBE])[kb] = fTW;
			(D.f[dirTW])[kw] = fBE;
			(D.f[dirTN])[k] = fBS;
			(D.f[dirBS])[kbs] = fTN;
			(D.f[dirBN])[kb] = fTS;
			(D.f[dirTS])[ks] = fBN;
			(D.f[dirZERO])[k] = fZERO;
			(D.f[dirTNE])[k] = fBSW;
			(D.f[dirTSE])[ks] = fBNW;
			(D.f[dirBNE])[kb] = fTSW;
			(D.f[dirBSE])[kbs] = fTNW;
			(D.f[dirTNW])[kw] = fBSE;
			(D.f[dirTSW])[ksw] = fBNE;
			(D.f[dirBNW])[kbw] = fTSE;
			(D.f[dirBSW])[kbsw] = fTNE;
			//////////////////////////////////////////////////////////////////////////                            
		}
	}
}
////////////////////////////////////////////////////////////////////////////////


