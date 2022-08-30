#ifndef LBM_D3Q27_H
#define LBM_D3Q27_H

namespace vf::lbm::dir
{

static constexpr int STARTDIR = 0;
static constexpr int ENDDIR   = 26;

// used in the CPU and the GPU version
static constexpr int DIR_000 = 0;	 // REST
static constexpr int DIR_P00 = 1;	 // E
static constexpr int DIR_M00 = 2;	 // W
static constexpr int DIR_0P0 = 3;	 // N
static constexpr int DIR_0M0 = 4;	 // S
static constexpr int DIR_00P = 5;	 // T
static constexpr int DIR_00M = 6;	 // B

static constexpr int DIR_PP0 = 7;	 // NE
static constexpr int DIR_MM0 = 8;	 // SW
static constexpr int DIR_PM0 = 9;	 // SE
static constexpr int DIR_MP0 = 10;	 // NW
static constexpr int DIR_P0P = 11;	 // TE
static constexpr int DIR_M0M = 12;	 // BW
static constexpr int DIR_P0M = 13;	 // BE
static constexpr int DIR_M0P = 14;	 // TW
static constexpr int DIR_0PP = 15;	 // TN
static constexpr int DIR_0MM = 16;	 // BS
static constexpr int DIR_0PM = 17;	 // BN
static constexpr int DIR_0MP = 18;	 // TS

static constexpr int DIR_PPP = 19;	 // TNE
static constexpr int DIR_MPP = 20;	 // TNW
static constexpr int DIR_PMP = 21;	 // TSE
static constexpr int DIR_MMP = 22;	 // TSW
static constexpr int DIR_PPM = 23;	 // BNE
static constexpr int DIR_MPM = 24;	 // BNW
static constexpr int DIR_PMM = 25;	 // BSE
static constexpr int DIR_MMM = 26;	 // BSW 

// used in the CPU version
// static constexpr int INV_P00 = DIR_M00;
// static constexpr int INV_M00 = DIR_P00;
// static constexpr int INV_0P0 = DIR_0M0;
// static constexpr int INV_0M0 = DIR_0P0;
// static constexpr int INV_00P = DIR_00M;
// static constexpr int INV_00M = DIR_00P;
// static constexpr int INV_PP0 = DIR_MM0;
// static constexpr int INV_MM0 = DIR_PP0;
// static constexpr int INV_PM0 = DIR_MP0;
// static constexpr int INV_MP0 = DIR_PM0;
// static constexpr int INV_P0P = DIR_M0M;
// static constexpr int INV_M0M = DIR_P0P;
// static constexpr int INV_P0M = DIR_M0P;
// static constexpr int INV_M0P = DIR_P0M;
// static constexpr int INV_0PP = DIR_0MM;
// static constexpr int INV_0MM = DIR_0PP;
// static constexpr int INV_0PM = DIR_0MP;
// static constexpr int INV_0MP = DIR_0PM;
// static constexpr int INV_PPP = DIR_MMM;
// static constexpr int INV_MPP = DIR_PMM;
// static constexpr int INV_PMP = DIR_MPM;
// static constexpr int INV_MMP = DIR_PPM;
// static constexpr int INV_PPM = DIR_MMP;
// static constexpr int INV_MPM = DIR_PMP;
// static constexpr int INV_PMM = DIR_MPP;
// static constexpr int INV_MMM = DIR_PPP;

// static constexpr int SGD_P00 = 0;
// static constexpr int SGD_M00 = 1;
// static constexpr int SGD_0P0 = 2;
// static constexpr int SGD_0M0 = 3;
// static constexpr int SGD_00P = 4;
// static constexpr int SGD_00M = 5;
// static constexpr int SGD_PP0 = 6;
// static constexpr int SGD_MM0 = 7;
// static constexpr int SGD_PM0 = 8;
// static constexpr int SGD_MP0 = 9;
// static constexpr int SGD_P0P = 10;
// static constexpr int SGD_M0M = 11;
// static constexpr int SGD_P0M = 12;
// static constexpr int SGD_M0P = 13;
// static constexpr int SGD_0PP = 14;
// static constexpr int SGD_0MM = 15;
// static constexpr int SGD_0PM = 16;
// static constexpr int SGD_0MP = 17;
// static constexpr int SGD_PPP = 18;
// static constexpr int SGD_MPP = 19;
// static constexpr int SGD_PMP = 20;
// static constexpr int SGD_MMP = 21;
// static constexpr int SGD_PPM = 22;
// static constexpr int SGD_MPM = 23;
// static constexpr int SGD_PMM = 24;
// static constexpr int SGD_MMM = 25;


// DEPRECATED
static constexpr int ZZZ = DIR_000;
static constexpr int PZZ = DIR_P00;
static constexpr int MZZ = DIR_M00;
static constexpr int ZPZ = DIR_0P0;
static constexpr int ZMZ = DIR_0M0;
static constexpr int ZZP = DIR_00P;
static constexpr int ZZM = DIR_00M;
static constexpr int PPZ = DIR_PP0;
static constexpr int MMZ = DIR_MM0;
static constexpr int PMZ = DIR_PM0;
static constexpr int MPZ = DIR_MP0;
static constexpr int PZP = DIR_P0P;
static constexpr int MZM = DIR_M0M;
static constexpr int PZM = DIR_P0M;
static constexpr int MZP = DIR_M0P;
static constexpr int ZPP = DIR_0PP;
static constexpr int ZMM = DIR_0MM;
static constexpr int ZPM = DIR_0PM;
static constexpr int ZMP = DIR_0MP;
static constexpr int PPP = DIR_PPP;
static constexpr int MPP = DIR_MPP;
static constexpr int PMP = DIR_PMP;
static constexpr int MMP = DIR_MMP;
static constexpr int PPM = DIR_PPM;
static constexpr int MPM = DIR_MPM;
static constexpr int PMM = DIR_PMM;
static constexpr int MMM = DIR_MMM;
}
#endif
