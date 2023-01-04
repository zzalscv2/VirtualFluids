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
//! \file AdvectionDiffusion27chim.cu
//! \ingroup GPU
//! \author Martin Schoenherr
//=======================================================================================
/* Device code */
#include "LBM/LB.h"
#include "lbm/constants/D3Q27.h"

#include <lbm/constants/NumericConstants.h>

using namespace vf::lbm::constant;
using namespace vf::lbm::dir;

////////////////////////////////////////////////////////////////////////////////
//! \brief forward chimera transformation \ref forwardChimera
//! - Chimera transform from distributions to central moments as defined in Eq. (43)-(45) in \ref
//! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
inline __device__ void forwardChimera(real &mfa, real &mfb, real &mfc, real vv, real v2) {
	real m1 = (mfa + mfc) + mfb;
	real m2 = mfc - mfa;
	mfc     = (mfc + mfa) + (v2*m1 - c2o1*vv*m2);
	mfb     = m2 - vv*m1;
	mfa     = m1;
}


////////////////////////////////////////////////////////////////////////////////
//! \brief backward chimera transformation \ref backwardChimera
//! - Chimera transform from  central moments to distributions as defined in Eq. (88)-(96) in \ref
//! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
inline __device__ void backwardChimera(real &mfa, real &mfb, real &mfc, real vv, real v2) {
	real ma = (mfc + mfa*(v2 - vv))*c1o2 + mfb*(vv - c1o2);
	real mb = ((mfa - mfc) - mfa*v2) - c2o1*mfb*vv;
	mfc     = (mfc + mfa*(v2 + vv))*c1o2 + mfb*(vv + c1o2);
	mfb     = mb;
	mfa     = ma;
}


////////////////////////////////////////////////////////////////////////////////
__global__ void Factorized_Central_Moments_Advection_Diffusion_Device_Kernel(
	real omegaDiffusivity,
	uint* typeOfGridNode,
	uint* neighborX,
	uint* neighborY,
	uint* neighborZ,
	real* distributions,
	real* distributionsAD,
	unsigned long long numberOfLBnodes,
	real* forces,
	bool isEvenTimestep)
{
	//////////////////////////////////////////////////////////////////////////
	//! Cumulant K17 Kernel is based on \ref
	//! <a href="https://doi.org/10.1016/j.jcp.2017.05.040"><b>[ M. Geier et al. (2017), DOI:10.1016/j.jcp.2017.05.040 ]</b></a>
	//! and \ref
	//! <a href="https://doi.org/10.1016/j.jcp.2017.07.004"><b>[ M. Geier et al. (2017), DOI:10.1016/j.jcp.2017.07.004 ]</b></a>
	//!
	//! The cumulant kernel is executed in the following steps
	//!
	////////////////////////////////////////////////////////////////////////////////
	//! - Get node index coordinates from threadIdx, blockIdx, blockDim and gridDim.
	//!
	const unsigned  x = threadIdx.x;
	const unsigned  y = blockIdx.x;
	const unsigned  z = blockIdx.y;

	const unsigned nx = blockDim.x;
	const unsigned ny = gridDim.x;

	const unsigned k = nx*(ny*z + y) + x;

	//////////////////////////////////////////////////////////////////////////
	// run for all indices in size_Mat and fluid nodes
	if ((k < numberOfLBnodes) && (typeOfGridNode[k] == GEO_FLUID))
	{
		//////////////////////////////////////////////////////////////////////////
		//! - Read distributions: style of reading and writing the distributions from/to stored arrays dependent on timestep is based on the esoteric twist algorithm \ref
		//! <a href="https://doi.org/10.3390/computation5020019"><b>[ M. Geier et al. (2017), DOI:10.3390/computation5020019 ]</b></a>
		//!
		Distributions27 dist;
		if (isEvenTimestep)
		{
			dist.f[DIR_P00] = &distributions[DIR_P00 * numberOfLBnodes];
			dist.f[DIR_M00] = &distributions[DIR_M00 * numberOfLBnodes];
			dist.f[DIR_0P0] = &distributions[DIR_0P0 * numberOfLBnodes];
			dist.f[DIR_0M0] = &distributions[DIR_0M0 * numberOfLBnodes];
			dist.f[DIR_00P] = &distributions[DIR_00P * numberOfLBnodes];
			dist.f[DIR_00M] = &distributions[DIR_00M * numberOfLBnodes];
			dist.f[DIR_PP0] = &distributions[DIR_PP0 * numberOfLBnodes];
			dist.f[DIR_MM0] = &distributions[DIR_MM0 * numberOfLBnodes];
			dist.f[DIR_PM0] = &distributions[DIR_PM0 * numberOfLBnodes];
			dist.f[DIR_MP0] = &distributions[DIR_MP0 * numberOfLBnodes];
			dist.f[DIR_P0P] = &distributions[DIR_P0P * numberOfLBnodes];
			dist.f[DIR_M0M] = &distributions[DIR_M0M * numberOfLBnodes];
			dist.f[DIR_P0M] = &distributions[DIR_P0M * numberOfLBnodes];
			dist.f[DIR_M0P] = &distributions[DIR_M0P * numberOfLBnodes];
			dist.f[DIR_0PP] = &distributions[DIR_0PP * numberOfLBnodes];
			dist.f[DIR_0MM] = &distributions[DIR_0MM * numberOfLBnodes];
			dist.f[DIR_0PM] = &distributions[DIR_0PM * numberOfLBnodes];
			dist.f[DIR_0MP] = &distributions[DIR_0MP * numberOfLBnodes];
			dist.f[DIR_000] = &distributions[DIR_000 * numberOfLBnodes];
			dist.f[DIR_PPP] = &distributions[DIR_PPP * numberOfLBnodes];
			dist.f[DIR_MMP] = &distributions[DIR_MMP * numberOfLBnodes];
			dist.f[DIR_PMP] = &distributions[DIR_PMP * numberOfLBnodes];
			dist.f[DIR_MPP] = &distributions[DIR_MPP * numberOfLBnodes];
			dist.f[DIR_PPM] = &distributions[DIR_PPM * numberOfLBnodes];
			dist.f[DIR_MMM] = &distributions[DIR_MMM * numberOfLBnodes];
			dist.f[DIR_PMM] = &distributions[DIR_PMM * numberOfLBnodes];
			dist.f[DIR_MPM] = &distributions[DIR_MPM * numberOfLBnodes];
		}
		else
		{
			dist.f[DIR_M00] = &distributions[DIR_P00 * numberOfLBnodes];
			dist.f[DIR_P00] = &distributions[DIR_M00 * numberOfLBnodes];
			dist.f[DIR_0M0] = &distributions[DIR_0P0 * numberOfLBnodes];
			dist.f[DIR_0P0] = &distributions[DIR_0M0 * numberOfLBnodes];
			dist.f[DIR_00M] = &distributions[DIR_00P * numberOfLBnodes];
			dist.f[DIR_00P] = &distributions[DIR_00M * numberOfLBnodes];
			dist.f[DIR_MM0] = &distributions[DIR_PP0 * numberOfLBnodes];
			dist.f[DIR_PP0] = &distributions[DIR_MM0 * numberOfLBnodes];
			dist.f[DIR_MP0] = &distributions[DIR_PM0 * numberOfLBnodes];
			dist.f[DIR_PM0] = &distributions[DIR_MP0 * numberOfLBnodes];
			dist.f[DIR_M0M] = &distributions[DIR_P0P * numberOfLBnodes];
			dist.f[DIR_P0P] = &distributions[DIR_M0M * numberOfLBnodes];
			dist.f[DIR_M0P] = &distributions[DIR_P0M * numberOfLBnodes];
			dist.f[DIR_P0M] = &distributions[DIR_M0P * numberOfLBnodes];
			dist.f[DIR_0MM] = &distributions[DIR_0PP * numberOfLBnodes];
			dist.f[DIR_0PP] = &distributions[DIR_0MM * numberOfLBnodes];
			dist.f[DIR_0MP] = &distributions[DIR_0PM * numberOfLBnodes];
			dist.f[DIR_0PM] = &distributions[DIR_0MP * numberOfLBnodes];
			dist.f[DIR_000] = &distributions[DIR_000 * numberOfLBnodes];
			dist.f[DIR_MMM] = &distributions[DIR_PPP * numberOfLBnodes];
			dist.f[DIR_PPM] = &distributions[DIR_MMP * numberOfLBnodes];
			dist.f[DIR_MPM] = &distributions[DIR_PMP * numberOfLBnodes];
			dist.f[DIR_PMM] = &distributions[DIR_MPP * numberOfLBnodes];
			dist.f[DIR_MMP] = &distributions[DIR_PPM * numberOfLBnodes];
			dist.f[DIR_PPP] = &distributions[DIR_MMM * numberOfLBnodes];
			dist.f[DIR_MPP] = &distributions[DIR_PMM * numberOfLBnodes];
			dist.f[DIR_PMP] = &distributions[DIR_MPM * numberOfLBnodes];
		}
		////////////////////////////////////////////////////////////////////////////////
		Distributions27 distAD;
		if (isEvenTimestep)
		{
			distAD.f[DIR_P00] = &distributionsAD[DIR_P00 * numberOfLBnodes];
			distAD.f[DIR_M00] = &distributionsAD[DIR_M00 * numberOfLBnodes];
			distAD.f[DIR_0P0] = &distributionsAD[DIR_0P0 * numberOfLBnodes];
			distAD.f[DIR_0M0] = &distributionsAD[DIR_0M0 * numberOfLBnodes];
			distAD.f[DIR_00P] = &distributionsAD[DIR_00P * numberOfLBnodes];
			distAD.f[DIR_00M] = &distributionsAD[DIR_00M * numberOfLBnodes];
			distAD.f[DIR_PP0] = &distributionsAD[DIR_PP0 * numberOfLBnodes];
			distAD.f[DIR_MM0] = &distributionsAD[DIR_MM0 * numberOfLBnodes];
			distAD.f[DIR_PM0] = &distributionsAD[DIR_PM0 * numberOfLBnodes];
			distAD.f[DIR_MP0] = &distributionsAD[DIR_MP0 * numberOfLBnodes];
			distAD.f[DIR_P0P] = &distributionsAD[DIR_P0P * numberOfLBnodes];
			distAD.f[DIR_M0M] = &distributionsAD[DIR_M0M * numberOfLBnodes];
			distAD.f[DIR_P0M] = &distributionsAD[DIR_P0M * numberOfLBnodes];
			distAD.f[DIR_M0P] = &distributionsAD[DIR_M0P * numberOfLBnodes];
			distAD.f[DIR_0PP] = &distributionsAD[DIR_0PP * numberOfLBnodes];
			distAD.f[DIR_0MM] = &distributionsAD[DIR_0MM * numberOfLBnodes];
			distAD.f[DIR_0PM] = &distributionsAD[DIR_0PM * numberOfLBnodes];
			distAD.f[DIR_0MP] = &distributionsAD[DIR_0MP * numberOfLBnodes];
			distAD.f[DIR_000] = &distributionsAD[DIR_000 * numberOfLBnodes];
			distAD.f[DIR_PPP] = &distributionsAD[DIR_PPP * numberOfLBnodes];
			distAD.f[DIR_MMP] = &distributionsAD[DIR_MMP * numberOfLBnodes];
			distAD.f[DIR_PMP] = &distributionsAD[DIR_PMP * numberOfLBnodes];
			distAD.f[DIR_MPP] = &distributionsAD[DIR_MPP * numberOfLBnodes];
			distAD.f[DIR_PPM] = &distributionsAD[DIR_PPM * numberOfLBnodes];
			distAD.f[DIR_MMM] = &distributionsAD[DIR_MMM * numberOfLBnodes];
			distAD.f[DIR_PMM] = &distributionsAD[DIR_PMM * numberOfLBnodes];
			distAD.f[DIR_MPM] = &distributionsAD[DIR_MPM * numberOfLBnodes];
		}
		else
		{
			distAD.f[DIR_M00] = &distributionsAD[DIR_P00 * numberOfLBnodes];
			distAD.f[DIR_P00] = &distributionsAD[DIR_M00 * numberOfLBnodes];
			distAD.f[DIR_0M0] = &distributionsAD[DIR_0P0 * numberOfLBnodes];
			distAD.f[DIR_0P0] = &distributionsAD[DIR_0M0 * numberOfLBnodes];
			distAD.f[DIR_00M] = &distributionsAD[DIR_00P * numberOfLBnodes];
			distAD.f[DIR_00P] = &distributionsAD[DIR_00M * numberOfLBnodes];
			distAD.f[DIR_MM0] = &distributionsAD[DIR_PP0 * numberOfLBnodes];
			distAD.f[DIR_PP0] = &distributionsAD[DIR_MM0 * numberOfLBnodes];
			distAD.f[DIR_MP0] = &distributionsAD[DIR_PM0 * numberOfLBnodes];
			distAD.f[DIR_PM0] = &distributionsAD[DIR_MP0 * numberOfLBnodes];
			distAD.f[DIR_M0M] = &distributionsAD[DIR_P0P * numberOfLBnodes];
			distAD.f[DIR_P0P] = &distributionsAD[DIR_M0M * numberOfLBnodes];
			distAD.f[DIR_M0P] = &distributionsAD[DIR_P0M * numberOfLBnodes];
			distAD.f[DIR_P0M] = &distributionsAD[DIR_M0P * numberOfLBnodes];
			distAD.f[DIR_0MM] = &distributionsAD[DIR_0PP * numberOfLBnodes];
			distAD.f[DIR_0PP] = &distributionsAD[DIR_0MM * numberOfLBnodes];
			distAD.f[DIR_0MP] = &distributionsAD[DIR_0PM * numberOfLBnodes];
			distAD.f[DIR_0PM] = &distributionsAD[DIR_0MP * numberOfLBnodes];
			distAD.f[DIR_000] = &distributionsAD[DIR_000 * numberOfLBnodes];
			distAD.f[DIR_MMM] = &distributionsAD[DIR_PPP * numberOfLBnodes];
			distAD.f[DIR_PPM] = &distributionsAD[DIR_MMP * numberOfLBnodes];
			distAD.f[DIR_MPM] = &distributionsAD[DIR_PMP * numberOfLBnodes];
			distAD.f[DIR_PMM] = &distributionsAD[DIR_MPP * numberOfLBnodes];
			distAD.f[DIR_MMP] = &distributionsAD[DIR_PPM * numberOfLBnodes];
			distAD.f[DIR_PPP] = &distributionsAD[DIR_MMM * numberOfLBnodes];
			distAD.f[DIR_MPP] = &distributionsAD[DIR_PMM * numberOfLBnodes];
			distAD.f[DIR_PMP] = &distributionsAD[DIR_MPM * numberOfLBnodes];
		}
		////////////////////////////////////////////////////////////////////////////////
		//! - Set neighbor indices (necessary for indirect addressing)
		uint kw   = neighborX[k];
		uint ks   = neighborY[k];
		uint kb   = neighborZ[k];
		uint ksw  = neighborY[kw];
		uint kbw  = neighborZ[kw];
		uint kbs  = neighborZ[ks];
		uint kbsw = neighborZ[ksw];
		////////////////////////////////////////////////////////////////////////////////////
		//! - Set local distributions Fluid
		//!
		real fcbb = (dist.f[DIR_P00])[k];
		real fabb = (dist.f[DIR_M00])[kw];
		real fbcb = (dist.f[DIR_0P0])[k];
		real fbab = (dist.f[DIR_0M0])[ks];
		real fbbc = (dist.f[DIR_00P])[k];
		real fbba = (dist.f[DIR_00M])[kb];
		real fccb = (dist.f[DIR_PP0])[k];
		real faab = (dist.f[DIR_MM0])[ksw];
		real fcab = (dist.f[DIR_PM0])[ks];
		real facb = (dist.f[DIR_MP0])[kw];
		real fcbc = (dist.f[DIR_P0P])[k];
		real faba = (dist.f[DIR_M0M])[kbw];
		real fcba = (dist.f[DIR_P0M])[kb];
		real fabc = (dist.f[DIR_M0P])[kw];
		real fbcc = (dist.f[DIR_0PP])[k];
		real fbaa = (dist.f[DIR_0MM])[kbs];
		real fbca = (dist.f[DIR_0PM])[kb];
		real fbac = (dist.f[DIR_0MP])[ks];
		real fbbb = (dist.f[DIR_000])[k];
		real fccc = (dist.f[DIR_PPP])[k];
		real faac = (dist.f[DIR_MMP])[ksw];
		real fcac = (dist.f[DIR_PMP])[ks];
		real facc = (dist.f[DIR_MPP])[kw];
		real fcca = (dist.f[DIR_PPM])[kb];
		real faaa = (dist.f[DIR_MMM])[kbsw];
		real fcaa = (dist.f[DIR_PMM])[kbs];
		real faca = (dist.f[DIR_MPM])[kbw];
		////////////////////////////////////////////////////////////////////////////////////
		//! - Set local distributions Advection Diffusion
		//!
		real mfcbb = (distAD.f[DIR_P00])[k];
		real mfabb = (distAD.f[DIR_M00])[kw];
		real mfbcb = (distAD.f[DIR_0P0])[k];
		real mfbab = (distAD.f[DIR_0M0])[ks];
		real mfbbc = (distAD.f[DIR_00P])[k];
		real mfbba = (distAD.f[DIR_00M])[kb];
		real mfccb = (distAD.f[DIR_PP0])[k];
		real mfaab = (distAD.f[DIR_MM0])[ksw];
		real mfcab = (distAD.f[DIR_PM0])[ks];
		real mfacb = (distAD.f[DIR_MP0])[kw];
		real mfcbc = (distAD.f[DIR_P0P])[k];
		real mfaba = (distAD.f[DIR_M0M])[kbw];
		real mfcba = (distAD.f[DIR_P0M])[kb];
		real mfabc = (distAD.f[DIR_M0P])[kw];
		real mfbcc = (distAD.f[DIR_0PP])[k];
		real mfbaa = (distAD.f[DIR_0MM])[kbs];
		real mfbca = (distAD.f[DIR_0PM])[kb];
		real mfbac = (distAD.f[DIR_0MP])[ks];
		real mfbbb = (distAD.f[DIR_000])[k];
		real mfccc = (distAD.f[DIR_PPP])[k];
		real mfaac = (distAD.f[DIR_MMP])[ksw];
		real mfcac = (distAD.f[DIR_PMP])[ks];
		real mfacc = (distAD.f[DIR_MPP])[kw];
		real mfcca = (distAD.f[DIR_PPM])[kb];
		real mfaaa = (distAD.f[DIR_MMM])[kbsw];
		real mfcaa = (distAD.f[DIR_PMM])[kbs];
		real mfaca = (distAD.f[DIR_MPM])[kbw];
		////////////////////////////////////////////////////////////////////////////////////
		//! - Calculate density and velocity using pyramid summation for low round-off errors as in Eq. (J1)-(J3) \ref
		//! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
		//!
		////////////////////////////////////////////////////////////////////////////////////
		// fluid component
		real drhoFluid =
			((((fccc + faaa) + (faca + fcac)) + ((facc + fcaa) + (faac + fcca))) +
			(((fbac + fbca) + (fbaa + fbcc)) + ((fabc + fcba) + (faba + fcbc)) + ((facb + fcab) + (faab + fccb))) +
			((fabb + fcbb) + (fbab + fbcb) + (fbba + fbbc))) + fbbb;

		real rhoFluid = c1o1 + drhoFluid;
		real OOrhoFluid = c1o1 / rhoFluid;

        real vvx =
			((((fccc - faaa) + (fcac - faca)) + ((fcaa - facc) + (fcca - faac))) +
			(((fcba - fabc) + (fcbc - faba)) + ((fcab - facb) + (fccb - faab))) +
			(fcbb - fabb)) * OOrhoFluid;
		real vvy =
			((((fccc - faaa) + (faca - fcac)) + ((facc - fcaa) + (fcca - faac))) +
			(((fbca - fbac) + (fbcc - fbaa)) + ((facb - fcab) + (fccb - faab))) +
			(fbcb - fbab)) * OOrhoFluid;
		real vvz =
			((((fccc - faaa) + (fcac - faca)) + ((facc - fcaa) + (faac - fcca))) +
			(((fbac - fbca) + (fbcc - fbaa)) + ((fabc - fcba) + (fcbc - faba))) +
			(fbbc - fbba)) * OOrhoFluid;
		////////////////////////////////////////////////////////////////////////////////////
		// second component
		real rho =
			((((mfccc + mfaaa) + (mfaca + mfcac)) + ((mfacc + mfcaa) + (mfaac + mfcca))) +
			(((mfbac + mfbca) + (mfbaa + mfbcc)) + ((mfabc + mfcba) + (mfaba + mfcbc)) + ((mfacb + mfcab) + (mfaab + mfccb))) +
				((mfabb + mfcbb) + (mfbab + mfbcb) + (mfbba + mfbbc))) + mfbbb;

        ////////////////////////////////////////////////////////////////////////////////////
        //! - Add half of the acceleration (body force) to the velocity as in Eq. (42) \ref
        //! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
        //!
        real fx = forces[0];
        real fy = forces[1];
        real fz = -rho*forces[2];
        vvx += fx*c1o2;
        vvy += fy*c1o2;
        vvz += fz*c1o2;
        ////////////////////////////////////////////////////////////////////////////////////
		// calculate the square of velocities for this lattice node
		real vx2 = vvx*vvx;
		real vy2 = vvy*vvy;
		real vz2 = vvz*vvz;
		////////////////////////////////////////////////////////////////////////////////////
		//real omegaDiffusivity = c2o1 / (c6o1 * diffusivity + c1o1);
		////////////////////////////////////////////////////////////////////////////////////
		//! - Chimera transform from distributions to central moments as defined in Eq. (43)-(45) in \ref
		//! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
		//!
		////////////////////////////////////////////////////////////////////////////////////
		// Z - Dir
		forwardChimera(mfaaa, mfaab, mfaac, vvz, vz2);
		forwardChimera(mfaba, mfabb, mfabc, vvz, vz2);
		forwardChimera(mfaca, mfacb, mfacc, vvz, vz2);
		forwardChimera(mfbaa, mfbab, mfbac, vvz, vz2);
		forwardChimera(mfbba, mfbbb, mfbbc, vvz, vz2);
		forwardChimera(mfbca, mfbcb, mfbcc, vvz, vz2);
		forwardChimera(mfcaa, mfcab, mfcac, vvz, vz2);
		forwardChimera(mfcba, mfcbb, mfcbc, vvz, vz2);
		forwardChimera(mfcca, mfccb, mfccc, vvz, vz2);

		////////////////////////////////////////////////////////////////////////////////////
		// Y - Dir
		forwardChimera(mfaaa, mfaba, mfaca, vvy, vy2);
		forwardChimera(mfaab, mfabb, mfacb, vvy, vy2);
		forwardChimera(mfaac, mfabc, mfacc, vvy, vy2);
		forwardChimera(mfbaa, mfbba, mfbca, vvy, vy2);
		forwardChimera(mfbab, mfbbb, mfbcb, vvy, vy2);
		forwardChimera(mfbac, mfbbc, mfbcc, vvy, vy2);
		forwardChimera(mfcaa, mfcba, mfcca, vvy, vy2);
		forwardChimera(mfcab, mfcbb, mfccb, vvy, vy2);
		forwardChimera(mfcac, mfcbc, mfccc, vvy, vy2);

		////////////////////////////////////////////////////////////////////////////////////
		// X - Dir
		forwardChimera(mfaaa, mfbaa, mfcaa, vvx, vx2);
		forwardChimera(mfaba, mfbba, mfcba, vvx, vx2);
		forwardChimera(mfaca, mfbca, mfcca, vvx, vx2);
		forwardChimera(mfaab, mfbab, mfcab, vvx, vx2);
		forwardChimera(mfabb, mfbbb, mfcbb, vvx, vx2);
		forwardChimera(mfacb, mfbcb, mfccb, vvx, vx2);
		forwardChimera(mfaac, mfbac, mfcac, vvx, vx2);
		forwardChimera(mfabc, mfbbc, mfcbc, vvx, vx2);
		forwardChimera(mfacc, mfbcc, mfccc, vvx, vx2);

		////////////////////////////////////////////////////////////////////////////////////
		//! - Factorized central moments for Advection Diffusion Equation - Eq. (15)-(16) in \ref
		//! <a href="https://doi.org/10.1016/j.advwatres.2015.09.015"><b>[ X. Yang et al. (2016), DOI: 10.1016/j.advwatres.2015.09.015]</b></a>
		//!

		// linearized orthogonalization of 3rd order central moments
		real Mabc = mfabc - mfaba*c1o3;
		real Mbca = mfbca - mfbaa*c1o3;
		real Macb = mfacb - mfaab*c1o3;
		real Mcba = mfcba - mfaba*c1o3;
		real Mcab = mfcab - mfaab*c1o3;
		real Mbac = mfbac - mfbaa*c1o3;
		// linearized orthogonalization of 5th order central moments
		real Mcbc = mfcbc - mfaba*c1o9;
		real Mbcc = mfbcc - mfbaa*c1o9;
		real Mccb = mfccb - mfaab*c1o9;

		// collision of 1st order moments
		mfbaa *= c1o1 - omegaDiffusivity;
		mfaba *= c1o1 - omegaDiffusivity;
		mfaab *= c1o1 - omegaDiffusivity;

		// equilibration of 3rd order moments
		Mabc = c0o1;
		Mbca = c0o1;
		Macb = c0o1;
		Mcba = c0o1;
		Mcab = c0o1;
		Mbac = c0o1;
		mfbbb = c0o1;

		// equilibration of 5th order moments
		Mcbc = c0o1;
		Mbcc = c0o1;
		Mccb = c0o1;

		// equilibration of 2nd order moments
		mfbba = c0o1;
		mfbab = c0o1;
		mfabb = c0o1;

		mfcaa = c1o3 * rho;
		mfaca = c1o3 * rho;
		mfaac = c1o3 * rho;

		// equilibration of 4th order moments
		mfacc = c1o9 * rho;
		mfcac = c1o9 * rho;
		mfcca = c1o9 * rho;

		mfcbb = c0o1;
		mfbcb = c0o1;
		mfbbc = c0o1;

		// equilibration of 6th order moment
		mfccc = c1o27 * rho;

		// from linearized orthogonalization 3rd order central moments to central moments
		mfabc = Mabc + mfaba*c1o3;
		mfbca = Mbca + mfbaa*c1o3;
		mfacb = Macb + mfaab*c1o3;
		mfcba = Mcba + mfaba*c1o3;
		mfcab = Mcab + mfaab*c1o3;
		mfbac = Mbac + mfbaa*c1o3;

		// from linearized orthogonalization 5th order central moments to central moments
		mfcbc = Mcbc + mfaba*c1o9;
		mfbcc = Mbcc + mfbaa*c1o9;
		mfccb = Mccb + mfaab*c1o9;

		////////////////////////////////////////////////////////////////////////////////////
		//! - Chimera transform from  central moments to distributions as defined in Eq. (88)-(96) in \ref
		//! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015), DOI:10.1016/j.camwa.2015.05.001 ]</b></a>
		//!
		////////////////////////////////////////////////////////////////////////////////////
		// X - Dir
		backwardChimera(mfaaa, mfbaa, mfcaa, vvx, vx2);
		backwardChimera(mfaba, mfbba, mfcba, vvx, vx2);
		backwardChimera(mfaca, mfbca, mfcca, vvx, vx2);
		backwardChimera(mfaab, mfbab, mfcab, vvx, vx2);
		backwardChimera(mfabb, mfbbb, mfcbb, vvx, vx2);
		backwardChimera(mfacb, mfbcb, mfccb, vvx, vx2);
		backwardChimera(mfaac, mfbac, mfcac, vvx, vx2);
		backwardChimera(mfabc, mfbbc, mfcbc, vvx, vx2);
		backwardChimera(mfacc, mfbcc, mfccc, vvx, vx2);

		////////////////////////////////////////////////////////////////////////////////////
		// Y - Dir
		backwardChimera(mfaaa, mfaba, mfaca, vvy, vy2);
		backwardChimera(mfaab, mfabb, mfacb, vvy, vy2);
		backwardChimera(mfaac, mfabc, mfacc, vvy, vy2);
		backwardChimera(mfbaa, mfbba, mfbca, vvy, vy2);
		backwardChimera(mfbab, mfbbb, mfbcb, vvy, vy2);
		backwardChimera(mfbac, mfbbc, mfbcc, vvy, vy2);
		backwardChimera(mfcaa, mfcba, mfcca, vvy, vy2);
		backwardChimera(mfcab, mfcbb, mfccb, vvy, vy2);
		backwardChimera(mfcac, mfcbc, mfccc, vvy, vy2);

		////////////////////////////////////////////////////////////////////////////////////
		// Z - Dir
		backwardChimera(mfaaa, mfaab, mfaac, vvz, vz2);
		backwardChimera(mfaba, mfabb, mfabc, vvz, vz2);
		backwardChimera(mfaca, mfacb, mfacc, vvz, vz2);
		backwardChimera(mfbaa, mfbab, mfbac, vvz, vz2);
		backwardChimera(mfbba, mfbbb, mfbbc, vvz, vz2);
		backwardChimera(mfbca, mfbcb, mfbcc, vvz, vz2);
		backwardChimera(mfcaa, mfcab, mfcac, vvz, vz2);
		backwardChimera(mfcba, mfcbb, mfcbc, vvz, vz2);
		backwardChimera(mfcca, mfccb, mfccc, vvz, vz2);

		////////////////////////////////////////////////////////////////////////////////////
		//! - Write distributions: style of reading and writing the distributions from/to
		//! stored arrays dependent on timestep is based on the esoteric twist algorithm
		//! <a href="https://doi.org/10.3390/computation5020019"><b>[ M. Geier et al. (2017), DOI:10.3390/computation5020019 ]</b></a>
		//!
		(distAD.f[DIR_P00])[k   ] = mfabb;
		(distAD.f[DIR_M00])[kw  ] = mfcbb;
		(distAD.f[DIR_0P0])[k   ] = mfbab;
		(distAD.f[DIR_0M0])[ks  ] = mfbcb;
		(distAD.f[DIR_00P])[k   ] = mfbba;
		(distAD.f[DIR_00M])[kb  ] = mfbbc;
		(distAD.f[DIR_PP0])[k   ] = mfaab;
		(distAD.f[DIR_MM0])[ksw ] = mfccb;
		(distAD.f[DIR_PM0])[ks  ] = mfacb;
		(distAD.f[DIR_MP0])[kw  ] = mfcab;
		(distAD.f[DIR_P0P])[k   ] = mfaba;
		(distAD.f[DIR_M0M])[kbw ] = mfcbc;
		(distAD.f[DIR_P0M])[kb  ] = mfabc;
		(distAD.f[DIR_M0P])[kw  ] = mfcba;
		(distAD.f[DIR_0PP])[k   ] = mfbaa;
		(distAD.f[DIR_0MM])[kbs ] = mfbcc;
		(distAD.f[DIR_0PM])[kb  ] = mfbac;
		(distAD.f[DIR_0MP])[ks  ] = mfbca;
		(distAD.f[DIR_000])[k   ] = mfbbb;
		(distAD.f[DIR_PPP])[k   ] = mfaaa;
		(distAD.f[DIR_PMP])[ks  ] = mfaca;
		(distAD.f[DIR_PPM])[kb  ] = mfaac;
		(distAD.f[DIR_PMM])[kbs ] = mfacc;
		(distAD.f[DIR_MPP])[kw  ] = mfcaa;
		(distAD.f[DIR_MMP])[ksw ] = mfcca;
		(distAD.f[DIR_MPM])[kbw ] = mfcac;
		(distAD.f[DIR_MMM])[kbsw] = mfccc;
	}
}
////////////////////////////////////////////////////////////////////////////////

