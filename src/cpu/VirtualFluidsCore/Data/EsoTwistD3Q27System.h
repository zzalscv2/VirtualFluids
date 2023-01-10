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
//! \file EsoTwistD3Q27System.h
//! \ingroup Data
//! \author Konstantin Kutscher
//=======================================================================================

#ifndef ESOTWISTD3Q27SYSTEM_H
#define ESOTWISTD3Q27SYSTEM_H

#include "D3Q27System.h"

//!
struct EsoTwistD3Q27System {
    const static int FSTARTDIR = D3Q27System::FSTARTDIR;
    const static int FENDDIR = D3Q27System::FENDDIR; // gellerstyle: meint alle frichtungen OHNE f0

    const static int STARTF = D3Q27System::STARTF;
    const static int ENDF   = D3Q27System::ENDF;

    const static int STARTDIR = D3Q27System::STARTDIR;
    const static int ENDDIR   = D3Q27System::ENDDIR;

    static const int REST = D3Q27System::DIR_000; /*f0 */
    static const int E    = D3Q27System::DIR_P00;    /*f1 */
    static const int W    = D3Q27System::DIR_M00;    /*f2 */
    static const int N    = D3Q27System::DIR_0P0;    /*f3 */
    static const int S    = D3Q27System::DIR_0M0;    /*f4 */
    static const int T    = D3Q27System::DIR_00P;    /*f5 */
    static const int B    = D3Q27System::DIR_00M;    /*f6 */
    static const int NE   = D3Q27System::DIR_PP0;   /*f7 */
    static const int SW   = D3Q27System::DIR_MM0;   /*f8 */
    static const int SE   = D3Q27System::DIR_PM0;   /*f9 */
    static const int NW   = D3Q27System::DIR_MP0;   /*f10*/
    static const int TE   = D3Q27System::DIR_P0P;   /*f11*/
    static const int BW   = D3Q27System::DIR_M0M;   /*f12*/
    static const int BE   = D3Q27System::DIR_P0M;   /*f13*/
    static const int TW   = D3Q27System::DIR_M0P;   /*f14*/
    static const int TN   = D3Q27System::DIR_0PP;   /*f15*/
    static const int BS   = D3Q27System::DIR_0MM;   /*f16*/
    static const int BN   = D3Q27System::DIR_0PM;   /*f17*/
    static const int TS   = D3Q27System::DIR_0MP;   /*f18*/
    static const int TNE  = D3Q27System::DIR_PPP;
    static const int TNW  = D3Q27System::DIR_MPP;
    static const int TSE  = D3Q27System::DIR_PMP;
    static const int TSW  = D3Q27System::DIR_MMP;
    static const int BNE  = D3Q27System::DIR_PPM;
    static const int BNW  = D3Q27System::DIR_MPM;
    static const int BSE  = D3Q27System::DIR_PMM;
    static const int BSW  = D3Q27System::DIR_MMM;

    static const int INV_E   = D3Q27System::DIR_M00;
    static const int INV_W   = D3Q27System::DIR_P00;
    static const int INV_N   = D3Q27System::DIR_0M0;
    static const int INV_S   = D3Q27System::DIR_0P0;
    static const int INV_T   = D3Q27System::DIR_00M;
    static const int INV_B   = D3Q27System::DIR_00P;
    static const int INV_NE  = D3Q27System::DIR_MM0;
    static const int INV_SW  = D3Q27System::DIR_PP0;
    static const int INV_SE  = D3Q27System::DIR_MP0;
    static const int INV_NW  = D3Q27System::DIR_PM0;
    static const int INV_TE  = D3Q27System::DIR_M0M;
    static const int INV_BW  = D3Q27System::DIR_P0P;
    static const int INV_BE  = D3Q27System::DIR_M0P;
    static const int INV_TW  = D3Q27System::DIR_P0M;
    static const int INV_TN  = D3Q27System::DIR_0MM;
    static const int INV_BS  = D3Q27System::DIR_0PP;
    static const int INV_BN  = D3Q27System::DIR_0MP;
    static const int INV_TS  = D3Q27System::DIR_0PM;
    static const int INV_TNE = D3Q27System::DIR_MMM;
    static const int INV_TNW = D3Q27System::DIR_PMM;
    static const int INV_TSE = D3Q27System::DIR_MPM;
    static const int INV_TSW = D3Q27System::DIR_PPM;
    static const int INV_BNE = D3Q27System::DIR_MMP;
    static const int INV_BNW = D3Q27System::DIR_PMP;
    static const int INV_BSE = D3Q27System::DIR_MPP;
    static const int INV_BSW = D3Q27System::DIR_PPP;

    static const unsigned long int etZERO; // 1;/*f0 */
    static const unsigned long int etE;    //  2;    /*f1 */
    static const unsigned long int etW;    //  4;    /*f2 */
    static const unsigned long int etN;    //  8;    /*f3 */
    static const unsigned long int etS;    //  16;   /*f4 */
    static const unsigned long int etT;    //  32;    /*f5 */
    static const unsigned long int etB;    //  64;   /*f6 */
    static const unsigned long int etNE;   // 128;  /*f7 */
    static const unsigned long int etSW;   // 256;  /*f8 */
    static const unsigned long int etSE;   // 512;  /*f9 */
    static const unsigned long int etNW;   // 1024;  /*f10*/
    static const unsigned long int etTE;   // 2048;  /*f11*/
    static const unsigned long int etBW;   // 4096;  /*f12*/
    static const unsigned long int etBE;   // 8192;  /*f13*/
    static const unsigned long int etTW;   // 16384;  /*f14*/
    static const unsigned long int etTN;   // 32768;  /*f15*/
    static const unsigned long int etBS;   // 65536;  /*f16*/
    static const unsigned long int etBN;   // 131072;  /*f17*/
    static const unsigned long int etTS;   // 262144;  /*f18*/
    static const unsigned long int etTNE;  // 524288;
    static const unsigned long int etTNW;  // 1048576;
    static const unsigned long int etTSE;  // 2097152;
    static const unsigned long int etTSW;  // 4194304;
    static const unsigned long int etBNE;  // 8388608;
    static const unsigned long int etBNW;  // 16777216;
    static const unsigned long int etBSE;  // 33554432;
    static const unsigned long int etBSW;  // = 67108864;

    const static int ETX1[ENDF + 1];
    const static int ETX2[ENDF + 1];
    const static int ETX3[ENDF + 1];
    const static int etINVDIR[ENDF + 1];
    const static unsigned long int etDIR[ENDF + 1];
};

#endif
