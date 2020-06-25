/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/* This file was automatically generated --- DO NOT EDIT */
/* Generated on Sun Oct 29 08:17:33 EDT 2017 */

#include "dft/codelet-dft.h"

#if defined(ARCH_PREFERS_FMA) || defined(ISA_EXTENSION_PREFERS_FMA)

/* Generated by: ../../../genfft/gen_twiddle.native -fma -simd -compact -variables 4 -pipeline-latency 8 -n 4 -name t1sv_4 -include dft/simd/ts.h */

/*
 * This function contains 22 FP additions, 12 FP multiplications,
 * (or, 16 additions, 6 multiplications, 6 fused multiply/add),
 * 15 stack variables, 0 constants, and 16 memory accesses
 */
#include "dft/simd/ts.h"

static void t1sv_4(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms)
{
     {
	  INT m;
	  for (m = mb, W = W + (mb * 6); m < me; m = m + (2 * VL), ri = ri + ((2 * VL) * ms), ii = ii + ((2 * VL) * ms), W = W + ((2 * VL) * 6), MAKE_VOLATILE_STRIDE(8, rs)) {
	       V T1, Tv, T7, Tu, Te, To, Tk, Tq;
	       T1 = LD(&(ri[0]), ms, &(ri[0]));
	       Tv = LD(&(ii[0]), ms, &(ii[0]));
	       {
		    V T3, T6, T4, Tt, T2, T5;
		    T3 = LD(&(ri[WS(rs, 2)]), ms, &(ri[0]));
		    T6 = LD(&(ii[WS(rs, 2)]), ms, &(ii[0]));
		    T2 = LDW(&(W[TWVL * 2]));
		    T4 = VMUL(T2, T3);
		    Tt = VMUL(T2, T6);
		    T5 = LDW(&(W[TWVL * 3]));
		    T7 = VFMA(T5, T6, T4);
		    Tu = VFNMS(T5, T3, Tt);
	       }
	       {
		    V Ta, Td, Tb, Tn, T9, Tc;
		    Ta = LD(&(ri[WS(rs, 1)]), ms, &(ri[WS(rs, 1)]));
		    Td = LD(&(ii[WS(rs, 1)]), ms, &(ii[WS(rs, 1)]));
		    T9 = LDW(&(W[0]));
		    Tb = VMUL(T9, Ta);
		    Tn = VMUL(T9, Td);
		    Tc = LDW(&(W[TWVL * 1]));
		    Te = VFMA(Tc, Td, Tb);
		    To = VFNMS(Tc, Ta, Tn);
	       }
	       {
		    V Tg, Tj, Th, Tp, Tf, Ti;
		    Tg = LD(&(ri[WS(rs, 3)]), ms, &(ri[WS(rs, 1)]));
		    Tj = LD(&(ii[WS(rs, 3)]), ms, &(ii[WS(rs, 1)]));
		    Tf = LDW(&(W[TWVL * 4]));
		    Th = VMUL(Tf, Tg);
		    Tp = VMUL(Tf, Tj);
		    Ti = LDW(&(W[TWVL * 5]));
		    Tk = VFMA(Ti, Tj, Th);
		    Tq = VFNMS(Ti, Tg, Tp);
	       }
	       {
		    V T8, Tl, Ts, Tw;
		    T8 = VADD(T1, T7);
		    Tl = VADD(Te, Tk);
		    ST(&(ri[WS(rs, 2)]), VSUB(T8, Tl), ms, &(ri[0]));
		    ST(&(ri[0]), VADD(T8, Tl), ms, &(ri[0]));
		    Ts = VADD(To, Tq);
		    Tw = VADD(Tu, Tv);
		    ST(&(ii[0]), VADD(Ts, Tw), ms, &(ii[0]));
		    ST(&(ii[WS(rs, 2)]), VSUB(Tw, Ts), ms, &(ii[0]));
	       }
	       {
		    V Tm, Tr, Tx, Ty;
		    Tm = VSUB(T1, T7);
		    Tr = VSUB(To, Tq);
		    ST(&(ri[WS(rs, 3)]), VSUB(Tm, Tr), ms, &(ri[WS(rs, 1)]));
		    ST(&(ri[WS(rs, 1)]), VADD(Tm, Tr), ms, &(ri[WS(rs, 1)]));
		    Tx = VSUB(Tv, Tu);
		    Ty = VSUB(Te, Tk);
		    ST(&(ii[WS(rs, 1)]), VSUB(Tx, Ty), ms, &(ii[WS(rs, 1)]));
		    ST(&(ii[WS(rs, 3)]), VADD(Ty, Tx), ms, &(ii[WS(rs, 1)]));
	       }
	  }
     }
     VLEAVE();
}

static const tw_instr twinstr[] = {
     VTW(0, 1),
     VTW(0, 2),
     VTW(0, 3),
     {TW_NEXT, (2 * VL), 0}
};

static const ct_desc desc = { 4, XSIMD_STRING("t1sv_4"), twinstr, &GENUS, {16, 6, 6, 0}, 0, 0, 0 };

void XSIMD(codelet_t1sv_4) (planner *p) {
     X(kdft_dit_register) (p, t1sv_4, &desc);
}
#else

/* Generated by: ../../../genfft/gen_twiddle.native -simd -compact -variables 4 -pipeline-latency 8 -n 4 -name t1sv_4 -include dft/simd/ts.h */

/*
 * This function contains 22 FP additions, 12 FP multiplications,
 * (or, 16 additions, 6 multiplications, 6 fused multiply/add),
 * 13 stack variables, 0 constants, and 16 memory accesses
 */
#include "dft/simd/ts.h"

static void t1sv_4(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms)
{
     {
	  INT m;
	  for (m = mb, W = W + (mb * 6); m < me; m = m + (2 * VL), ri = ri + ((2 * VL) * ms), ii = ii + ((2 * VL) * ms), W = W + ((2 * VL) * 6), MAKE_VOLATILE_STRIDE(8, rs)) {
	       V T1, Tp, T6, To, Tc, Tk, Th, Tl;
	       T1 = LD(&(ri[0]), ms, &(ri[0]));
	       Tp = LD(&(ii[0]), ms, &(ii[0]));
	       {
		    V T3, T5, T2, T4;
		    T3 = LD(&(ri[WS(rs, 2)]), ms, &(ri[0]));
		    T5 = LD(&(ii[WS(rs, 2)]), ms, &(ii[0]));
		    T2 = LDW(&(W[TWVL * 2]));
		    T4 = LDW(&(W[TWVL * 3]));
		    T6 = VFMA(T2, T3, VMUL(T4, T5));
		    To = VFNMS(T4, T3, VMUL(T2, T5));
	       }
	       {
		    V T9, Tb, T8, Ta;
		    T9 = LD(&(ri[WS(rs, 1)]), ms, &(ri[WS(rs, 1)]));
		    Tb = LD(&(ii[WS(rs, 1)]), ms, &(ii[WS(rs, 1)]));
		    T8 = LDW(&(W[0]));
		    Ta = LDW(&(W[TWVL * 1]));
		    Tc = VFMA(T8, T9, VMUL(Ta, Tb));
		    Tk = VFNMS(Ta, T9, VMUL(T8, Tb));
	       }
	       {
		    V Te, Tg, Td, Tf;
		    Te = LD(&(ri[WS(rs, 3)]), ms, &(ri[WS(rs, 1)]));
		    Tg = LD(&(ii[WS(rs, 3)]), ms, &(ii[WS(rs, 1)]));
		    Td = LDW(&(W[TWVL * 4]));
		    Tf = LDW(&(W[TWVL * 5]));
		    Th = VFMA(Td, Te, VMUL(Tf, Tg));
		    Tl = VFNMS(Tf, Te, VMUL(Td, Tg));
	       }
	       {
		    V T7, Ti, Tn, Tq;
		    T7 = VADD(T1, T6);
		    Ti = VADD(Tc, Th);
		    ST(&(ri[WS(rs, 2)]), VSUB(T7, Ti), ms, &(ri[0]));
		    ST(&(ri[0]), VADD(T7, Ti), ms, &(ri[0]));
		    Tn = VADD(Tk, Tl);
		    Tq = VADD(To, Tp);
		    ST(&(ii[0]), VADD(Tn, Tq), ms, &(ii[0]));
		    ST(&(ii[WS(rs, 2)]), VSUB(Tq, Tn), ms, &(ii[0]));
	       }
	       {
		    V Tj, Tm, Tr, Ts;
		    Tj = VSUB(T1, T6);
		    Tm = VSUB(Tk, Tl);
		    ST(&(ri[WS(rs, 3)]), VSUB(Tj, Tm), ms, &(ri[WS(rs, 1)]));
		    ST(&(ri[WS(rs, 1)]), VADD(Tj, Tm), ms, &(ri[WS(rs, 1)]));
		    Tr = VSUB(Tp, To);
		    Ts = VSUB(Tc, Th);
		    ST(&(ii[WS(rs, 1)]), VSUB(Tr, Ts), ms, &(ii[WS(rs, 1)]));
		    ST(&(ii[WS(rs, 3)]), VADD(Ts, Tr), ms, &(ii[WS(rs, 1)]));
	       }
	  }
     }
     VLEAVE();
}

static const tw_instr twinstr[] = {
     VTW(0, 1),
     VTW(0, 2),
     VTW(0, 3),
     {TW_NEXT, (2 * VL), 0}
};

static const ct_desc desc = { 4, XSIMD_STRING("t1sv_4"), twinstr, &GENUS, {16, 6, 6, 0}, 0, 0, 0 };

void XSIMD(codelet_t1sv_4) (planner *p) {
     X(kdft_dit_register) (p, t1sv_4, &desc);
}
#endif
