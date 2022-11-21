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
//! \file scaleCF_compressible.cu
//! \ingroup GPU/GridScaling
//! \author Martin Schoenherr, Anna Wellmann
//=======================================================================================

#include "DataTypes.h"
#include "Kernel/Utilities/DistributionHelper.cuh"
#include "Kernel/Utilities/ChimeraTransformation.h"
#include "Kernel/Utilities/ScalingHelperFunctions.h"

using namespace vf::lbm::constant;
using namespace vf::lbm::dir;

//////////////////////////////////////////////////////////////////////////
//! \brief Calculate the interpolated distributions on the fine destination nodes
//! \details Used in scaling from coarse to fine
//! The function is executed in the following steps:
//!
__device__ __inline__ void interpolateDistributions(
    const real& x, const real& y, const real& z,
    real& m_000, 
    real& m_100, real& m_010, real& m_001,
    real& m_011, real& m_101, real& m_110, real& m_200, real& m_020, real& m_002,
    real& m_111, real& m_210, real& m_012, real& m_201, real& m_021, real& m_120, real& m_102,
    real& m_022, real& m_202, real& m_220, real& m_211, real& m_121, real& m_112,
    real& m_122, real& m_212, real& m_221,
    real& m_222,
    const real& a_000, const real& a_100, const real& a_010, const real& a_001, const real& a_200, const real& a_020, const real& a_002, const real& a_110, const real&  a_101, const real& a_011, const real& a_111,
    const real& b_000, const real& b_100, const real& b_010, const real& b_001, const real& b_200, const real& b_020, const real& b_002, const real& b_110, const real&  b_101, const real& b_011, const real& b_111,
    const real& c_000, const real& c_100, const real& c_010, const real& c_001, const real& c_200, const real& c_020, const real& c_002, const real& c_110, const real&  c_101, const real& c_011, const real& c_111,
    const real& d_000, const real& d_100, const real& d_010, const real& d_001, const real& d_110, const real& d_101, const real& d_011, const real& d_111,
    const real& LaplaceRho, const real& eps_new, const real& omegaF, 
    const real& kxxMyyAverage, const real& kxxMzzAverage, const real& kyzAverage, const real& kxzAverage, const real& kxyAverage
)
{
    real useNEQ = c1o1;//zero;//one;   //.... one = on ..... zero = off 

    //////////////////////////////////////////////////////////////////////////
    // - Reset all moments to zero
    //
    m_111 = c0o1;
    m_211 = c0o1;
    m_011 = c0o1;
    m_121 = c0o1;
    m_101 = c0o1;
    m_112 = c0o1;
    m_110 = c0o1;
    m_221 = c0o1;
    m_001 = c0o1;
    m_201 = c0o1;
    m_021 = c0o1;
    m_212 = c0o1;
    m_010 = c0o1;
    m_210 = c0o1;
    m_012 = c0o1;
    m_122 = c0o1;
    m_100 = c0o1;
    m_120 = c0o1;
    m_102 = c0o1;
    m_222 = c0o1;
    m_022 = c0o1;
    m_202 = c0o1;
    m_002 = c0o1;
    m_220 = c0o1;
    m_020 = c0o1;
    m_200 = c0o1;
    m_000 = c0o1;

    ////////////////////////////////////////////////////////////////////////////////
    //! - Set macroscopic values on destination node (zeroth and first order moments)
    //!
    real press = d_000 + x * d_100 + y * d_010 + z * d_001 +
                 x * y * d_110 + x * z * d_101 + y * z * d_011 + x * y * z * d_111 + c3o1 * x * x * LaplaceRho;
    real vvx   = a_000 + x * a_100 + y * a_010 + z * a_001 +
                 x * x * a_200 + y * y * a_020 + z * z * a_002 +
                 x * y * a_110 + x * z * a_101 + y * z * a_011 + x * y * z * a_111;
    real vvy   = b_000 + x * b_100 + y * b_010 + z * b_001 +
                 x * x * b_200 + y * y * b_020 + z * z * b_002 +
                 x * y * b_110 + x * z * b_101 + y * z * b_011 + x * y * z * b_111;
    real vvz   = c_000 + x * c_100 + y * c_010 + z * c_001 +
                 x * x * c_200 + y * y * c_020 + z * z * c_002 +
                 x * y * c_110 + x * z * c_101 + y * z * c_011 + x * y * z * c_111;

    m_000 = press; // m_000 is press, if drho is interpolated directly

    ////////////////////////////////////////////////////////////////////////////////
    //! - Set moments (second to sixth order) on destination node
    //!
    // linear combinations for second order moments
    real mxxPyyPzz = m_000;

    real mxxMyy = -c2o3 * (a_100 - b_010 + kxxMyyAverage + c2o1 * a_200 * x - b_110 * x + a_110 * y
                  -c2o1 * b_020 * y + a_101 * z - b_011 * z - b_111 * x * z + a_111 * y * z) * eps_new/ omegaF * (c1o1 + press);
    real mxxMzz = -c2o3 * (a_100 - c_001 + kxxMzzAverage + c2o1 * a_200 * x - c_101 * x + a_110 * y
                  -c_011 * y - c_111 * x * y + a_101 * z - c2o1 * c_002 * z + a_111 * y * z) * eps_new/ omegaF * (c1o1 + press);

    m_011 = -c1o3 * (b_001 + c_010 + kyzAverage + b_101 * x + c_110 * x + b_011 * y + c2o1 * c_020 * y
            + b_111 * x * y + c2o1 * b_002 * z + c_011 * z + c_111 * x * z) * eps_new / omegaF * (c1o1 + press);
    m_101 = -c1o3 * (a_001 + c_100 + kxzAverage + a_101 * x + c2o1 * c_200 * x + a_011 * y + c_110 * y
            + a_111 * x * y + c2o1 * a_002 * z + c_101 * z + c_111 * y * z) * eps_new / omegaF * (c1o1 + press);
    m_110 = -c1o3 * (a_010 + b_100 + kxyAverage + a_110 * x + c2o1 * b_200 * x + c2o1 * a_020 * y
            + b_110 * y + a_011 * z + b_101 * z + a_111 * x * z + b_111 * y * z) * eps_new / omegaF * (c1o1 + press);

    m_200 = c1o3 * (        mxxMyy +        mxxMzz + mxxPyyPzz) * useNEQ;
    m_020 = c1o3 * (-c2o1 * mxxMyy +        mxxMzz + mxxPyyPzz) * useNEQ;
    m_002 = c1o3 * (        mxxMyy - c2o1 * mxxMzz + mxxPyyPzz) * useNEQ;

    // linear combinations for third order moments
    m_111 = c0o1;

    real mxxyPyzz = c0o1;
    real mxxyMyzz = c0o1;
    real mxxzPyyz = c0o1;
    real mxxzMyyz = c0o1;
    real mxyyPxzz = c0o1;
    real mxyyMxzz = c0o1;

    m_210 = ( mxxyMyzz + mxxyPyzz) * c1o2;
    m_012 = (-mxxyMyzz + mxxyPyzz) * c1o2;
    m_201 = ( mxxzMyyz + mxxzPyyz) * c1o2;
    m_021 = (-mxxzMyyz + mxxzPyyz) * c1o2;
    m_120 = ( mxyyMxzz + mxyyPxzz) * c1o2;
    m_102 = (-mxyyMxzz + mxyyPxzz) * c1o2;

    // fourth order moments
    m_022 = m_000 * c1o9;
    m_202 = m_022;
    m_220 = m_022;

    // fifth order moments

    // sixth order moment
    m_222 = m_000 * c1o27;

    real vx_sq = vvx * vvx;
    real vy_sq = vvy * vvy;
    real vz_sq = vvz * vvz;

    ////////////////////////////////////////////////////////////////////////////////////
    //! - Chimera transform from central moments to well conditioned distributions as defined in Appendix J in
    //! <a href="https://doi.org/10.1016/j.camwa.2015.05.001"><b>[ M. Geier et al. (2015),
    //! DOI:10.1016/j.camwa.2015.05.001 ]</b></a> see also Eq. (88)-(96) in <a
    //! href="https://doi.org/10.1016/j.jcp.2017.05.040"><b>[ M. Geier et al. (2017), DOI:10.1016/j.jcp.2017.05.040
    //! ]</b></a>
    //!

    ////////////////////////////////////////////////////////////////////////////////////
    // X - Dir
    backwardInverseChimeraWithK(m_000, m_100, m_200, vvx, vx_sq, c1o1, c1o1);
    backwardChimera(            m_010, m_110, m_210, vvx, vx_sq);
    backwardInverseChimeraWithK(m_020, m_120, m_220, vvx, vx_sq, c3o1, c1o3);
    backwardChimera(            m_001, m_101, m_201, vvx, vx_sq);
    backwardChimera(            m_011, m_111, m_211, vvx, vx_sq);
    backwardChimera(            m_021, m_121, m_221, vvx, vx_sq);
    backwardInverseChimeraWithK(m_002, m_102, m_202, vvx, vx_sq, c3o1, c1o3);
    backwardChimera(            m_012, m_112, m_212, vvx, vx_sq);
    backwardInverseChimeraWithK(m_022, m_122, m_222, vvx, vx_sq, c9o1, c1o9);

    ////////////////////////////////////////////////////////////////////////////////////
    // Y - Dir
    backwardInverseChimeraWithK(m_000, m_010, m_020, vvy, vy_sq, c6o1, c1o6);
    backwardChimera(            m_001, m_011, m_021, vvy, vy_sq);
    backwardInverseChimeraWithK(m_002, m_012, m_022, vvy, vy_sq, c18o1, c1o18);
    backwardInverseChimeraWithK(m_100, m_110, m_120, vvy, vy_sq, c3o2, c2o3);
    backwardChimera(            m_101, m_111, m_121, vvy, vy_sq);
    backwardInverseChimeraWithK(m_102, m_112, m_122, vvy, vy_sq, c9o2, c2o9);
    backwardInverseChimeraWithK(m_200, m_210, m_220, vvy, vy_sq, c6o1, c1o6);
    backwardChimera(            m_201, m_211, m_221, vvy, vy_sq);
    backwardInverseChimeraWithK(m_202, m_212, m_222, vvy, vy_sq, c18o1, c1o18);

    ////////////////////////////////////////////////////////////////////////////////////
    // Z - Dir
    backwardInverseChimeraWithK(m_000, m_001, m_002, vvz, vz_sq, c36o1, c1o36);
    backwardInverseChimeraWithK(m_010, m_011, m_012, vvz, vz_sq, c9o1,  c1o9);
    backwardInverseChimeraWithK(m_020, m_021, m_022, vvz, vz_sq, c36o1, c1o36);
    backwardInverseChimeraWithK(m_100, m_101, m_102, vvz, vz_sq, c9o1,  c1o9);
    backwardInverseChimeraWithK(m_110, m_111, m_112, vvz, vz_sq, c9o4,  c4o9);
    backwardInverseChimeraWithK(m_120, m_121, m_122, vvz, vz_sq, c9o1,  c1o9);
    backwardInverseChimeraWithK(m_200, m_201, m_202, vvz, vz_sq, c36o1, c1o36);
    backwardInverseChimeraWithK(m_210, m_211, m_212, vvz, vz_sq, c9o1,  c1o9);
    backwardInverseChimeraWithK(m_220, m_221, m_222, vvz, vz_sq, c36o1, c1o36);
}

//////////////////////////////////////////////////////////////////////////
//! \brief Interpolate from coarse to fine nodes
//! \details This scaling function is designed for the Cumulant K17 Kernel chimera collision kernel.
//!
//! The function is executed in the following steps:
//!

// based on scaleCF_RhoSq_comp_27
__global__ void scaleCF_compressible(
    real* distributionsCoarse, 
    real* distributionsFine, 
    unsigned int* neighborXcoarse,
    unsigned int* neighborYcoarse,
    unsigned int* neighborZcoarse,
    unsigned int* neighborXfine,
    unsigned int* neighborYfine,
    unsigned int* neighborZfine,
    unsigned int* typeOfGridNode,
    unsigned int numberOfLBnodesCoarse, 
    unsigned int numberOfLBnodesFine, 
    bool isEvenTimestep,
    unsigned int* indicesCoarseMMM, 
    unsigned int* indicesFineMMM, 
    unsigned int numberOfInterfaceNodes, 
    real omegaCoarse, 
    real omegaFine, 
    OffCF offsetCF)
{
    ////////////////////////////////////////////////////////////////////////////////
    //! - Get the thread index coordinates from threadId_100, blockId_100, blockDim and gridDim.
    //!
    const unsigned k_thread = vf::gpu::getNodeIndex();

    //////////////////////////////////////////////////////////////////////////
    //! - Return for non-interface node
    if (k_thread >= numberOfInterfaceNodes)
        return;

    //////////////////////////////////////////////////////////////////////////
    //! - Read distributions: style of reading and writing the distributions from/to stored arrays dependent on
    //! timestep is based on the esoteric twist algorithm \ref <a
    //! href="https://doi.org/10.3390/computation5020019"><b>[ M. Geier et al. (2017),
    //! DOI:10.3390/computation5020019 ]</b></a>
    //!
    Distributions27 distFine   = vf::gpu::getDistributionReferences27(distributionsFine,   numberOfLBnodesFine,   true);
    Distributions27 distCoarse = vf::gpu::getDistributionReferences27(distributionsCoarse, numberOfLBnodesCoarse, isEvenTimestep);

    ////////////////////////////////////////////////////////////////////////////////
    //! - declare local variables for source nodes
    //!
    real eps_new = c1o2; // ratio of grid resolutions
    real omegaC  = omegaCoarse;
    real omegaF  = omegaFine;

    // zeroth and first order moments at the source nodes
    real drho_PPP, vx1_PPP, vx2_PPP, vx3_PPP;
    real drho_MPP, vx1_MPP, vx2_MPP, vx3_MPP;
    real drho_PMP, vx1_PMP, vx2_PMP, vx3_PMP;
    real drho_MMP, vx1_MMP, vx2_MMP, vx3_MMP;
    real drho_PPM, vx1_PPM, vx2_PPM, vx3_PPM;
    real drho_MPM, vx1_MPM, vx2_MPM, vx3_MPM;
    real drho_PMM, vx1_PMM, vx2_PMM, vx3_PMM;
    real drho_MMM, vx1_MMM, vx2_MMM, vx3_MMM;

    // second order moments at the source nodes
    real kxyFromfcNEQ_PPP, kyzFromfcNEQ_PPP, kxzFromfcNEQ_PPP, kxxMyyFromfcNEQ_PPP, kxxMzzFromfcNEQ_PPP;
    real kxyFromfcNEQ_MPP, kyzFromfcNEQ_MPP, kxzFromfcNEQ_MPP, kxxMyyFromfcNEQ_MPP, kxxMzzFromfcNEQ_MPP;
    real kxyFromfcNEQ_PMP, kyzFromfcNEQ_PMP, kxzFromfcNEQ_PMP, kxxMyyFromfcNEQ_PMP, kxxMzzFromfcNEQ_PMP;
    real kxyFromfcNEQ_MMP, kyzFromfcNEQ_MMP, kxzFromfcNEQ_MMP, kxxMyyFromfcNEQ_MMP, kxxMzzFromfcNEQ_MMP;
    real kxyFromfcNEQ_PPM, kyzFromfcNEQ_PPM, kxzFromfcNEQ_PPM, kxxMyyFromfcNEQ_PPM, kxxMzzFromfcNEQ_PPM;
    real kxyFromfcNEQ_MPM, kyzFromfcNEQ_MPM, kxzFromfcNEQ_MPM, kxxMyyFromfcNEQ_MPM, kxxMzzFromfcNEQ_MPM;
    real kxyFromfcNEQ_PMM, kyzFromfcNEQ_PMM, kxzFromfcNEQ_PMM, kxxMyyFromfcNEQ_PMM, kxxMzzFromfcNEQ_PMM;
    real kxyFromfcNEQ_MMM, kyzFromfcNEQ_MMM, kxzFromfcNEQ_MMM, kxxMyyFromfcNEQ_MMM, kxxMzzFromfcNEQ_MMM;

    ////////////////////////////////////////////////////////////////////////////////
    //! - Calculate moments for each source node 
    //!
    ////////////////////////////////////////////////////////////////////////////////
    // source node BSW = MMM
    ////////////////////////////////////////////////////////////////////////////////
    // index of the base node and its neighbors
    unsigned int k_base_000 = indicesCoarseMMM[k_thread];
    unsigned int k_base_M00 = neighborXcoarse [k_base_000];
    unsigned int k_base_0M0 = neighborYcoarse [k_base_000];
    unsigned int k_base_00M = neighborZcoarse [k_base_000];
    unsigned int k_base_MM0 = neighborYcoarse [k_base_M00];
    unsigned int k_base_M0M = neighborZcoarse [k_base_M00];
    unsigned int k_base_0MM = neighborZcoarse [k_base_0M0];
    unsigned int k_base_MMM = neighborZcoarse [k_base_MM0];
    ////////////////////////////////////////////////////////////////////////////////
    // Set neighbor indices
    unsigned int k_000 = k_base_000;
    unsigned int k_M00 = k_base_M00;
    unsigned int k_0M0 = k_base_0M0;
    unsigned int k_00M = k_base_00M;
    unsigned int k_MM0 = k_base_MM0;
    unsigned int k_M0M = k_base_M0M;
    unsigned int k_0MM = k_base_0MM;
    unsigned int k_MMM = k_base_MMM;

    calculateMomentsOnSourceNodes( distCoarse, omegaC,
        k_000, k_M00, k_0M0, k_00M, k_MM0, k_M0M, k_0MM, k_MMM, drho_MMM, vx1_MMM, vx2_MMM, vx3_MMM,
        kxyFromfcNEQ_MMM, kyzFromfcNEQ_MMM, kxzFromfcNEQ_MMM, kxxMyyFromfcNEQ_MMM, kxxMzzFromfcNEQ_MMM);

    //////////////////////////////////////////////////////////////////////////
    // source node TSW = MMP
    //////////////////////////////////////////////////////////////////////////
    // Set neighbor indices - has to be recalculated for the new source node
    k_000 = k_00M;
    k_M00 = k_M0M;
    k_0M0 = k_0MM;
    k_00M = neighborZcoarse[k_00M];
    k_MM0 = k_MMM;
    k_M0M = neighborZcoarse[k_M0M];
    k_0MM = neighborZcoarse[k_0MM];
    k_MMM = neighborZcoarse[k_MMM];

    calculateMomentsOnSourceNodes( distCoarse, omegaC,
        k_000, k_M00, k_0M0, k_00M, k_MM0, k_M0M, k_0MM, k_MMM, drho_MMP, vx1_MMP, vx2_MMP, vx3_MMP,
        kxyFromfcNEQ_MMP, kyzFromfcNEQ_MMP, kxzFromfcNEQ_MMP, kxxMyyFromfcNEQ_MMP, kxxMzzFromfcNEQ_MMP);

    //////////////////////////////////////////////////////////////////////////
    // source node TSE = PMP
    //////////////////////////////////////////////////////////////////////////
    // index
    k_000 = k_M00;
    k_M00 = neighborXcoarse[k_M00];
    k_0M0 = k_MM0;
    k_00M = k_M0M;
    k_MM0 = neighborXcoarse[k_MM0];
    k_M0M = neighborXcoarse[k_M0M];
    k_0MM = k_MMM;
    k_MMM = neighborXcoarse[k_MMM];

    calculateMomentsOnSourceNodes( distCoarse, omegaC,
        k_000, k_M00, k_0M0, k_00M, k_MM0, k_M0M, k_0MM, k_MMM, drho_PMP, vx1_PMP, vx2_PMP, vx3_PMP,
        kxyFromfcNEQ_PMP, kyzFromfcNEQ_PMP, kxzFromfcNEQ_PMP, kxxMyyFromfcNEQ_PMP, kxxMzzFromfcNEQ_PMP);

    //////////////////////////////////////////////////////////////////////////
    // source node BSE = PMM 
    //////////////////////////////////////////////////////////////////////////
    // index
    k_00M = k_000;
    k_M0M = k_M00;
    k_0MM = k_0M0;
    k_MMM = k_MM0;
    k_000 = k_base_M00;
    k_M00 = neighborXcoarse[k_base_M00];
    k_0M0 = k_base_MM0;
    k_MM0 = neighborXcoarse[k_base_MM0];

    calculateMomentsOnSourceNodes( distCoarse, omegaC,
        k_000, k_M00, k_0M0, k_00M, k_MM0, k_M0M, k_0MM, k_MMM, drho_PMM, vx1_PMM, vx2_PMM, vx3_PMM,
        kxyFromfcNEQ_PMM, kyzFromfcNEQ_PMM, kxzFromfcNEQ_PMM, kxxMyyFromfcNEQ_PMM, kxxMzzFromfcNEQ_PMM);

    //////////////////////////////////////////////////////////////////////////
    // source node BNW = MPM
    //////////////////////////////////////////////////////////////////////////
    // index of the base node and its neighbors --> indices of all source nodes
    k_base_000 = k_base_0M0;
    k_base_M00 = k_base_MM0;
    k_base_0M0 = neighborYcoarse[k_base_0M0];
    k_base_00M = k_base_0MM;
    k_base_MM0 = neighborYcoarse[k_base_MM0];
    k_base_M0M = k_base_MMM;
    k_base_0MM = neighborYcoarse[k_base_0MM];
    k_base_MMM = neighborYcoarse[k_base_MMM];
    //////////////////////////////////////////////////////////////////////////
    // index
    k_000 = k_base_000;
    k_M00 = k_base_M00;
    k_0M0 = k_base_0M0;
    k_00M = k_base_00M;
    k_MM0 = k_base_MM0;
    k_M0M = k_base_M0M;
    k_0MM = k_base_0MM;
    k_MMM = k_base_MMM;

    calculateMomentsOnSourceNodes( distCoarse, omegaC,
        k_000, k_M00, k_0M0, k_00M, k_MM0, k_M0M, k_0MM, k_MMM, drho_MPM, vx1_MPM, vx2_MPM, vx3_MPM,
        kxyFromfcNEQ_MPM, kyzFromfcNEQ_MPM, kxzFromfcNEQ_MPM, kxxMyyFromfcNEQ_MPM, kxxMzzFromfcNEQ_MPM);

    //////////////////////////////////////////////////////////////////////////
    // source node TNW = MPP
    //////////////////////////////////////////////////////////////////////////
    // index
    k_000 = k_00M;
    k_M00 = k_M0M;
    k_0M0 = k_0MM;
    k_00M = neighborZcoarse[k_00M];
    k_MM0 = k_MMM;
    k_M0M = neighborZcoarse[k_M0M];
    k_0MM = neighborZcoarse[k_0MM];
    k_MMM = neighborZcoarse[k_MMM];
    
    calculateMomentsOnSourceNodes( distCoarse, omegaC,
        k_000, k_M00, k_0M0, k_00M, k_MM0, k_M0M, k_0MM, k_MMM, drho_MPP, vx1_MPP, vx2_MPP, vx3_MPP,
        kxyFromfcNEQ_MPP, kyzFromfcNEQ_MPP, kxzFromfcNEQ_MPP, kxxMyyFromfcNEQ_MPP, kxxMzzFromfcNEQ_MPP);

    //////////////////////////////////////////////////////////////////////////
    // source node TNE = PPP
    //////////////////////////////////////////////////////////////////////////
    // index
    // index
    k_000 = k_M00;
    k_M00 = neighborXcoarse[k_M00];
    k_0M0 = k_MM0;
    k_00M = k_M0M;
    k_MM0 = neighborXcoarse[k_MM0];
    k_M0M = neighborXcoarse[k_M0M];
    k_0MM = k_MMM;
    k_MMM = neighborXcoarse[k_MMM];

    calculateMomentsOnSourceNodes( distCoarse, omegaC,
        k_000, k_M00, k_0M0, k_00M, k_MM0, k_M0M, k_0MM, k_MMM, drho_PPP, vx1_PPP, vx2_PPP, vx3_PPP,
        kxyFromfcNEQ_PPP, kyzFromfcNEQ_PPP, kxzFromfcNEQ_PPP, kxxMyyFromfcNEQ_PPP, kxxMzzFromfcNEQ_PPP);


    //////////////////////////////////////////////////////////////////////////
    // source node BNE = PPM
    //////////////////////////////////////////////////////////////////////////
    // index
    k_00M = k_000;
    k_M0M = k_M00;
    k_0MM = k_0M0;
    k_MMM = k_MM0;
    k_000 = k_base_M00;
    k_M00 = neighborXcoarse[k_base_M00];
    k_0M0 = k_base_MM0;
    k_MM0 = neighborXcoarse[k_base_MM0];
    
    calculateMomentsOnSourceNodes( distCoarse, omegaC,
        k_000, k_M00, k_0M0, k_00M, k_MM0, k_M0M, k_0MM, k_MMM, drho_PPM, vx1_PPM, vx2_PPM, vx3_PPM,
        kxyFromfcNEQ_PPM, kyzFromfcNEQ_PPM, kxzFromfcNEQ_PPM, kxxMyyFromfcNEQ_PPM, kxxMzzFromfcNEQ_PPM);

    //////////////////////////////////////////////////////////////////////////
    //! - Calculate coefficients for polynomial interpolation
    //!
    // example: a_110: derivation in x and y direction
    real a_000, a_100, a_010, a_001, a_200, a_020, a_002, a_110, a_101, a_011, a_111;
    real b_000, b_100, b_010, b_001, b_200, b_020, b_002, b_110, b_101, b_011, b_111;
    real c_000, c_100, c_010, c_001, c_200, c_020, c_002, c_110, c_101, c_011, c_111;
    real d_000, d_100, d_010, d_001, d_110, d_101, d_011, d_111;

    //! Workaround for precursorBC
    bool isEdgeNode = typeOfGridNode[k_base_000]!=GEO_FLUID? true: false;
    // isEdgeNode =false;
    if(false)
    {
        printf("drho_MMM: %f \t drho_PMM: %f \t vx1_MMM: %f \t vx1_PMM: %f \n", drho_MMM, drho_PMM, vx3_MMM , vx3_PMM);
        // printf("kxyFromfcNEQ_MMM: %f \t kxyFromfcNEQ_PMM: %f \t kxxMyyFromfcNEQ_MMM: %f \t kxxMyyFromfcNEQ_PMM: %f \n", kxyFromfcNEQ_MMM, kxyFromfcNEQ_PMM, kxxMyyFromfcNEQ_MMM , kxxMyyFromfcNEQ_PMM);
    }
    if(false)
    {
        drho_MMM = drho_PMM;
        vx1_MMM  = vx1_PMM;
        vx2_MMM  = vx2_PMM;
        vx3_MMM  = vx3_PMM;
        kxyFromfcNEQ_MMM = kxyFromfcNEQ_PMM;
        kyzFromfcNEQ_MMM = kyzFromfcNEQ_PMM;
        kxzFromfcNEQ_MMM = kxzFromfcNEQ_PMM;
        kxxMyyFromfcNEQ_MMM = kxxMyyFromfcNEQ_PMM;
        kxxMzzFromfcNEQ_MMM = kxxMzzFromfcNEQ_PMM;

        drho_MPM = drho_PPM;
        vx1_MPM  = vx1_PPM;
        vx2_MPM  = vx2_PPM;
        vx3_MPM  = vx3_PPM;
        kxyFromfcNEQ_MPM = kxyFromfcNEQ_PPM;
        kyzFromfcNEQ_MPM = kyzFromfcNEQ_PPM;
        kxzFromfcNEQ_MPM = kxzFromfcNEQ_PPM;
        kxxMyyFromfcNEQ_MPM = kxxMyyFromfcNEQ_PPM;
        kxxMzzFromfcNEQ_MPM = kxxMzzFromfcNEQ_PPM;
        kxxMzzFromfcNEQ_MPM = kxxMzzFromfcNEQ_PPM;

        drho_MMP = drho_PMP;
        vx1_MMP  = vx1_PMP;
        vx2_MMP  = vx2_PMP;
        vx3_MMP  = vx3_PMP;
        kxyFromfcNEQ_MMP = kxyFromfcNEQ_PMP;
        kyzFromfcNEQ_MMP = kyzFromfcNEQ_PMP;
        kxzFromfcNEQ_MMP = kxzFromfcNEQ_PMP;
        kxxMyyFromfcNEQ_MMP = kxxMyyFromfcNEQ_PMP;
        kxxMzzFromfcNEQ_MMP = kxxMzzFromfcNEQ_PMP;

        drho_MPP = drho_PPP;
        vx1_MPP  = vx1_PPP;
        vx2_MPP  = vx2_PPP;
        vx3_MPP  = vx3_PPP;
        kxyFromfcNEQ_MPP = kxyFromfcNEQ_PPP;
        kyzFromfcNEQ_MPP = kyzFromfcNEQ_PPP;
        kxzFromfcNEQ_MPP = kxzFromfcNEQ_PPP;
        kxxMyyFromfcNEQ_MPP = kxxMyyFromfcNEQ_PPP;
        kxxMzzFromfcNEQ_MPP = kxxMzzFromfcNEQ_PPP;
        kxxMzzFromfcNEQ_MPP = kxxMzzFromfcNEQ_PPP;
    }

    a_000 = (-kxxMyyFromfcNEQ_PPM - kxxMyyFromfcNEQ_PPP + kxxMyyFromfcNEQ_MPM + kxxMyyFromfcNEQ_MPP -
            kxxMyyFromfcNEQ_PMM - kxxMyyFromfcNEQ_PMP + kxxMyyFromfcNEQ_MMM + kxxMyyFromfcNEQ_MMP -
            kxxMzzFromfcNEQ_PPM - kxxMzzFromfcNEQ_PPP + kxxMzzFromfcNEQ_MPM + kxxMzzFromfcNEQ_MPP -
            kxxMzzFromfcNEQ_PMM - kxxMzzFromfcNEQ_PMP + kxxMzzFromfcNEQ_MMM + kxxMzzFromfcNEQ_MMP -
            c2o1 * kxyFromfcNEQ_PPM - c2o1 * kxyFromfcNEQ_PPP - c2o1 * kxyFromfcNEQ_MPM - c2o1 * kxyFromfcNEQ_MPP +
            c2o1 * kxyFromfcNEQ_PMM + c2o1 * kxyFromfcNEQ_PMP + c2o1 * kxyFromfcNEQ_MMM + c2o1 * kxyFromfcNEQ_MMP +
            c2o1 * kxzFromfcNEQ_PPM - c2o1 * kxzFromfcNEQ_PPP + c2o1 * kxzFromfcNEQ_MPM - c2o1 * kxzFromfcNEQ_MPP +
            c2o1 * kxzFromfcNEQ_PMM - c2o1 * kxzFromfcNEQ_PMP + c2o1 * kxzFromfcNEQ_MMM - c2o1 * kxzFromfcNEQ_MMP +
            c8o1 * vx1_PPM + c8o1 * vx1_PPP + c8o1 * vx1_MPM + c8o1 * vx1_MPP + c8o1 * vx1_PMM + c8o1 * vx1_PMP +
            c8o1 * vx1_MMM + c8o1 * vx1_MMP + c2o1 * vx2_PPM + c2o1 * vx2_PPP - c2o1 * vx2_MPM - c2o1 * vx2_MPP -
            c2o1 * vx2_PMM - c2o1 * vx2_PMP + c2o1 * vx2_MMM + c2o1 * vx2_MMP - c2o1 * vx3_PPM + c2o1 * vx3_PPP +
            c2o1 * vx3_MPM - c2o1 * vx3_MPP - c2o1 * vx3_PMM + c2o1 * vx3_PMP + c2o1 * vx3_MMM - c2o1 * vx3_MMP) /
            c64o1;
    b_000 = (c2o1 * kxxMyyFromfcNEQ_PPM + c2o1 * kxxMyyFromfcNEQ_PPP + c2o1 * kxxMyyFromfcNEQ_MPM +
            c2o1 * kxxMyyFromfcNEQ_MPP - c2o1 * kxxMyyFromfcNEQ_PMM - c2o1 * kxxMyyFromfcNEQ_PMP -
            c2o1 * kxxMyyFromfcNEQ_MMM - c2o1 * kxxMyyFromfcNEQ_MMP - kxxMzzFromfcNEQ_PPM - kxxMzzFromfcNEQ_PPP -
            kxxMzzFromfcNEQ_MPM - kxxMzzFromfcNEQ_MPP + kxxMzzFromfcNEQ_PMM + kxxMzzFromfcNEQ_PMP +
            kxxMzzFromfcNEQ_MMM + kxxMzzFromfcNEQ_MMP - c2o1 * kxyFromfcNEQ_PPM - c2o1 * kxyFromfcNEQ_PPP +
            c2o1 * kxyFromfcNEQ_MPM + c2o1 * kxyFromfcNEQ_MPP - c2o1 * kxyFromfcNEQ_PMM - c2o1 * kxyFromfcNEQ_PMP +
            c2o1 * kxyFromfcNEQ_MMM + c2o1 * kxyFromfcNEQ_MMP + c2o1 * kyzFromfcNEQ_PPM - c2o1 * kyzFromfcNEQ_PPP +
            c2o1 * kyzFromfcNEQ_MPM - c2o1 * kyzFromfcNEQ_MPP + c2o1 * kyzFromfcNEQ_PMM - c2o1 * kyzFromfcNEQ_PMP +
            c2o1 * kyzFromfcNEQ_MMM - c2o1 * kyzFromfcNEQ_MMP + c2o1 * vx1_PPM + c2o1 * vx1_PPP - c2o1 * vx1_MPM -
            c2o1 * vx1_MPP - c2o1 * vx1_PMM - c2o1 * vx1_PMP + c2o1 * vx1_MMM + c2o1 * vx1_MMP + c8o1 * vx2_PPM +
            c8o1 * vx2_PPP + c8o1 * vx2_MPM + c8o1 * vx2_MPP + c8o1 * vx2_PMM + c8o1 * vx2_PMP + c8o1 * vx2_MMM +
            c8o1 * vx2_MMP - c2o1 * vx3_PPM + c2o1 * vx3_PPP - c2o1 * vx3_MPM + c2o1 * vx3_MPP + c2o1 * vx3_PMM -
            c2o1 * vx3_PMP + c2o1 * vx3_MMM - c2o1 * vx3_MMP) /
            c64o1;
    c_000 = (kxxMyyFromfcNEQ_PPM - kxxMyyFromfcNEQ_PPP + kxxMyyFromfcNEQ_MPM - kxxMyyFromfcNEQ_MPP +
            kxxMyyFromfcNEQ_PMM - kxxMyyFromfcNEQ_PMP + kxxMyyFromfcNEQ_MMM - kxxMyyFromfcNEQ_MMP -
            c2o1 * kxxMzzFromfcNEQ_PPM + c2o1 * kxxMzzFromfcNEQ_PPP - c2o1 * kxxMzzFromfcNEQ_MPM +
            c2o1 * kxxMzzFromfcNEQ_MPP - c2o1 * kxxMzzFromfcNEQ_PMM + c2o1 * kxxMzzFromfcNEQ_PMP -
            c2o1 * kxxMzzFromfcNEQ_MMM + c2o1 * kxxMzzFromfcNEQ_MMP - c2o1 * kxzFromfcNEQ_PPM -
            c2o1 * kxzFromfcNEQ_PPP + c2o1 * kxzFromfcNEQ_MPM + c2o1 * kxzFromfcNEQ_MPP - c2o1 * kxzFromfcNEQ_PMM -
            c2o1 * kxzFromfcNEQ_PMP + c2o1 * kxzFromfcNEQ_MMM + c2o1 * kxzFromfcNEQ_MMP - c2o1 * kyzFromfcNEQ_PPM -
            c2o1 * kyzFromfcNEQ_PPP - c2o1 * kyzFromfcNEQ_MPM - c2o1 * kyzFromfcNEQ_MPP + c2o1 * kyzFromfcNEQ_PMM +
            c2o1 * kyzFromfcNEQ_PMP + c2o1 * kyzFromfcNEQ_MMM + c2o1 * kyzFromfcNEQ_MMP - c2o1 * vx1_PPM +
            c2o1 * vx1_PPP + c2o1 * vx1_MPM - c2o1 * vx1_MPP - c2o1 * vx1_PMM + c2o1 * vx1_PMP + c2o1 * vx1_MMM -
            c2o1 * vx1_MMP - c2o1 * vx2_PPM + c2o1 * vx2_PPP - c2o1 * vx2_MPM + c2o1 * vx2_MPP + c2o1 * vx2_PMM -
            c2o1 * vx2_PMP + c2o1 * vx2_MMM - c2o1 * vx2_MMP + c8o1 * vx3_PPM + c8o1 * vx3_PPP + c8o1 * vx3_MPM +
            c8o1 * vx3_MPP + c8o1 * vx3_PMM + c8o1 * vx3_PMP + c8o1 * vx3_MMM + c8o1 * vx3_MMP) /
            c64o1;
    a_100  = (vx1_PPM + vx1_PPP - vx1_MPM - vx1_MPP + vx1_PMM + vx1_PMP - vx1_MMM - vx1_MMP) / c4o1;
    b_100  = (vx2_PPM + vx2_PPP - vx2_MPM - vx2_MPP + vx2_PMM + vx2_PMP - vx2_MMM - vx2_MMP) / c4o1;
    c_100  = (vx3_PPM + vx3_PPP - vx3_MPM - vx3_MPP + vx3_PMM + vx3_PMP - vx3_MMM - vx3_MMP) / c4o1;
    a_200 = (kxxMyyFromfcNEQ_PPM + kxxMyyFromfcNEQ_PPP - kxxMyyFromfcNEQ_MPM - kxxMyyFromfcNEQ_MPP +
            kxxMyyFromfcNEQ_PMM + kxxMyyFromfcNEQ_PMP - kxxMyyFromfcNEQ_MMM - kxxMyyFromfcNEQ_MMP +
            kxxMzzFromfcNEQ_PPM + kxxMzzFromfcNEQ_PPP - kxxMzzFromfcNEQ_MPM - kxxMzzFromfcNEQ_MPP +
            kxxMzzFromfcNEQ_PMM + kxxMzzFromfcNEQ_PMP - kxxMzzFromfcNEQ_MMM - kxxMzzFromfcNEQ_MMP + c2o1 * vx2_PPM +
            c2o1 * vx2_PPP - c2o1 * vx2_MPM - c2o1 * vx2_MPP - c2o1 * vx2_PMM - c2o1 * vx2_PMP + c2o1 * vx2_MMM +
            c2o1 * vx2_MMP - c2o1 * vx3_PPM + c2o1 * vx3_PPP + c2o1 * vx3_MPM - c2o1 * vx3_MPP - c2o1 * vx3_PMM +
            c2o1 * vx3_PMP + c2o1 * vx3_MMM - c2o1 * vx3_MMP) /
            c16o1;
    b_200 = (kxyFromfcNEQ_PPM + kxyFromfcNEQ_PPP - kxyFromfcNEQ_MPM - kxyFromfcNEQ_MPP + kxyFromfcNEQ_PMM +
            kxyFromfcNEQ_PMP - kxyFromfcNEQ_MMM - kxyFromfcNEQ_MMP - c2o1 * vx1_PPM - c2o1 * vx1_PPP +
            c2o1 * vx1_MPM + c2o1 * vx1_MPP + c2o1 * vx1_PMM + c2o1 * vx1_PMP - c2o1 * vx1_MMM - c2o1 * vx1_MMP) /
            c8o1;
    c_200 = (kxzFromfcNEQ_PPM + kxzFromfcNEQ_PPP - kxzFromfcNEQ_MPM - kxzFromfcNEQ_MPP + kxzFromfcNEQ_PMM +
            kxzFromfcNEQ_PMP - kxzFromfcNEQ_MMM - kxzFromfcNEQ_MMP + c2o1 * vx1_PPM - c2o1 * vx1_PPP -
            c2o1 * vx1_MPM + c2o1 * vx1_MPP + c2o1 * vx1_PMM - c2o1 * vx1_PMP - c2o1 * vx1_MMM + c2o1 * vx1_MMP) /
            c8o1;
    a_010  = (vx1_PPM + vx1_PPP + vx1_MPM + vx1_MPP - vx1_PMM - vx1_PMP - vx1_MMM - vx1_MMP) / c4o1;
    b_010  = (vx2_PPM + vx2_PPP + vx2_MPM + vx2_MPP - vx2_PMM - vx2_PMP - vx2_MMM - vx2_MMP) / c4o1;
    c_010  = (vx3_PPM + vx3_PPP + vx3_MPM + vx3_MPP - vx3_PMM - vx3_PMP - vx3_MMM - vx3_MMP) / c4o1;
    a_020 = (kxyFromfcNEQ_PPM + kxyFromfcNEQ_PPP + kxyFromfcNEQ_MPM + kxyFromfcNEQ_MPP - kxyFromfcNEQ_PMM -
            kxyFromfcNEQ_PMP - kxyFromfcNEQ_MMM - kxyFromfcNEQ_MMP - c2o1 * vx2_PPM - c2o1 * vx2_PPP +
            c2o1 * vx2_MPM + c2o1 * vx2_MPP + c2o1 * vx2_PMM + c2o1 * vx2_PMP - c2o1 * vx2_MMM - c2o1 * vx2_MMP) /
            c8o1;
    b_020 = (-c2o1 * kxxMyyFromfcNEQ_PPM - c2o1 * kxxMyyFromfcNEQ_PPP - c2o1 * kxxMyyFromfcNEQ_MPM -
            c2o1 * kxxMyyFromfcNEQ_MPP + c2o1 * kxxMyyFromfcNEQ_PMM + c2o1 * kxxMyyFromfcNEQ_PMP +
            c2o1 * kxxMyyFromfcNEQ_MMM + c2o1 * kxxMyyFromfcNEQ_MMP + kxxMzzFromfcNEQ_PPM + kxxMzzFromfcNEQ_PPP +
            kxxMzzFromfcNEQ_MPM + kxxMzzFromfcNEQ_MPP - kxxMzzFromfcNEQ_PMM - kxxMzzFromfcNEQ_PMP -
            kxxMzzFromfcNEQ_MMM - kxxMzzFromfcNEQ_MMP + c2o1 * vx1_PPM + c2o1 * vx1_PPP - c2o1 * vx1_MPM -
            c2o1 * vx1_MPP - c2o1 * vx1_PMM - c2o1 * vx1_PMP + c2o1 * vx1_MMM + c2o1 * vx1_MMP - c2o1 * vx3_PPM +
            c2o1 * vx3_PPP - c2o1 * vx3_MPM + c2o1 * vx3_MPP + c2o1 * vx3_PMM - c2o1 * vx3_PMP + c2o1 * vx3_MMM -
            c2o1 * vx3_MMP) /
            c16o1;
    c_020 = (kyzFromfcNEQ_PPM + kyzFromfcNEQ_PPP + kyzFromfcNEQ_MPM + kyzFromfcNEQ_MPP - kyzFromfcNEQ_PMM -
            kyzFromfcNEQ_PMP - kyzFromfcNEQ_MMM - kyzFromfcNEQ_MMP + c2o1 * vx2_PPM - c2o1 * vx2_PPP +
            c2o1 * vx2_MPM - c2o1 * vx2_MPP - c2o1 * vx2_PMM + c2o1 * vx2_PMP - c2o1 * vx2_MMM + c2o1 * vx2_MMP) /
            c8o1;
    a_001  = (-vx1_PPM + vx1_PPP - vx1_MPM + vx1_MPP - vx1_PMM + vx1_PMP - vx1_MMM + vx1_MMP) / c4o1;
    b_001  = (-vx2_PPM + vx2_PPP - vx2_MPM + vx2_MPP - vx2_PMM + vx2_PMP - vx2_MMM + vx2_MMP) / c4o1;
    c_001  = (-vx3_PPM + vx3_PPP - vx3_MPM + vx3_MPP - vx3_PMM + vx3_PMP - vx3_MMM + vx3_MMP) / c4o1;
    a_002 = (-kxzFromfcNEQ_PPM + kxzFromfcNEQ_PPP - kxzFromfcNEQ_MPM + kxzFromfcNEQ_MPP - kxzFromfcNEQ_PMM +
            kxzFromfcNEQ_PMP - kxzFromfcNEQ_MMM + kxzFromfcNEQ_MMP + c2o1 * vx3_PPM - c2o1 * vx3_PPP -
            c2o1 * vx3_MPM + c2o1 * vx3_MPP + c2o1 * vx3_PMM - c2o1 * vx3_PMP - c2o1 * vx3_MMM + c2o1 * vx3_MMP) /
            c8o1;
    b_002 = (-kyzFromfcNEQ_PPM + kyzFromfcNEQ_PPP - kyzFromfcNEQ_MPM + kyzFromfcNEQ_MPP - kyzFromfcNEQ_PMM +
            kyzFromfcNEQ_PMP - kyzFromfcNEQ_MMM + kyzFromfcNEQ_MMP + c2o1 * vx3_PPM - c2o1 * vx3_PPP +
            c2o1 * vx3_MPM - c2o1 * vx3_MPP - c2o1 * vx3_PMM + c2o1 * vx3_PMP - c2o1 * vx3_MMM + c2o1 * vx3_MMP) /
            c8o1;
    c_002 = (-kxxMyyFromfcNEQ_PPM + kxxMyyFromfcNEQ_PPP - kxxMyyFromfcNEQ_MPM + kxxMyyFromfcNEQ_MPP -
            kxxMyyFromfcNEQ_PMM + kxxMyyFromfcNEQ_PMP - kxxMyyFromfcNEQ_MMM + kxxMyyFromfcNEQ_MMP +
            c2o1 * kxxMzzFromfcNEQ_PPM - c2o1 * kxxMzzFromfcNEQ_PPP + c2o1 * kxxMzzFromfcNEQ_MPM -
            c2o1 * kxxMzzFromfcNEQ_MPP + c2o1 * kxxMzzFromfcNEQ_PMM - c2o1 * kxxMzzFromfcNEQ_PMP +
            c2o1 * kxxMzzFromfcNEQ_MMM - c2o1 * kxxMzzFromfcNEQ_MMP - c2o1 * vx1_PPM + c2o1 * vx1_PPP +
            c2o1 * vx1_MPM - c2o1 * vx1_MPP - c2o1 * vx1_PMM + c2o1 * vx1_PMP + c2o1 * vx1_MMM - c2o1 * vx1_MMP -
            c2o1 * vx2_PPM + c2o1 * vx2_PPP - c2o1 * vx2_MPM + c2o1 * vx2_MPP + c2o1 * vx2_PMM - c2o1 * vx2_PMP +
            c2o1 * vx2_MMM - c2o1 * vx2_MMP) /
            c16o1;
    a_110 = (vx1_PPM + vx1_PPP - vx1_MPM - vx1_MPP - vx1_PMM - vx1_PMP + vx1_MMM + vx1_MMP) / c2o1;
    b_110 = (vx2_PPM + vx2_PPP - vx2_MPM - vx2_MPP - vx2_PMM - vx2_PMP + vx2_MMM + vx2_MMP) / c2o1;
    c_110 = (vx3_PPM + vx3_PPP - vx3_MPM - vx3_MPP - vx3_PMM - vx3_PMP + vx3_MMM + vx3_MMP) / c2o1;
    a_101 = (-vx1_PPM + vx1_PPP + vx1_MPM - vx1_MPP - vx1_PMM + vx1_PMP + vx1_MMM - vx1_MMP) / c2o1;
    b_101 = (-vx2_PPM + vx2_PPP + vx2_MPM - vx2_MPP - vx2_PMM + vx2_PMP + vx2_MMM - vx2_MMP) / c2o1;
    c_101 = (-vx3_PPM + vx3_PPP + vx3_MPM - vx3_MPP - vx3_PMM + vx3_PMP + vx3_MMM - vx3_MMP) / c2o1;
    a_011 = (-vx1_PPM + vx1_PPP - vx1_MPM + vx1_MPP + vx1_PMM - vx1_PMP + vx1_MMM - vx1_MMP) / c2o1;
    b_011 = (-vx2_PPM + vx2_PPP - vx2_MPM + vx2_MPP + vx2_PMM - vx2_PMP + vx2_MMM - vx2_MMP) / c2o1;
    c_011 = (-vx3_PPM + vx3_PPP - vx3_MPM + vx3_MPP + vx3_PMM - vx3_PMP + vx3_MMM - vx3_MMP) / c2o1;

    a_111 = -vx1_PPM + vx1_PPP + vx1_MPM - vx1_MPP + vx1_PMM - vx1_PMP - vx1_MMM + vx1_MMP;
    b_111 = -vx2_PPM + vx2_PPP + vx2_MPM - vx2_MPP + vx2_PMM - vx2_PMP - vx2_MMM + vx2_MMP;
    c_111 = -vx3_PPM + vx3_PPP + vx3_MPM - vx3_MPP + vx3_PMM - vx3_PMP - vx3_MMM + vx3_MMP;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    real kxyAverage    = c0o1;
    real kyzAverage    = c0o1;
    real kxzAverage    = c0o1;
    real kxxMyyAverage = c0o1;
    real kxxMzzAverage = c0o1;

    // real kxyAverage    = (kxyFromfcNEQ_MMM+
    //                       kxyFromfcNEQ_MMP+
    //                       kxyFromfcNEQ_PMP+
    //                       kxyFromfcNEQ_PMM+
    //                       kxyFromfcNEQ_MPM+
    //                       kxyFromfcNEQ_MPP+
    //                       kxyFromfcNEQ_PPP+
    //                       kxyFromfcNEQ_PPM) * c1o8 - (a_010 + b_100);
    // real kyzAverage    = (kyzFromfcNEQ_MMM+
    //                       kyzFromfcNEQ_MMP+
    //                       kyzFromfcNEQ_PMP+
    //                       kyzFromfcNEQ_PMM+
    //                       kyzFromfcNEQ_MPM+
    //                       kyzFromfcNEQ_MPP+
    //                       kyzFromfcNEQ_PPP+
    //                       kyzFromfcNEQ_PPM) * c1o8 - (b_001 + c_010);
    // real kxzAverage    = (kxzFromfcNEQ_MMM+
    //                       kxzFromfcNEQ_MMP+
    //                       kxzFromfcNEQ_PMP+
    //                       kxzFromfcNEQ_PMM+
    //                       kxzFromfcNEQ_MPM+
    //                       kxzFromfcNEQ_MPP+
    //                       kxzFromfcNEQ_PPP+
    //                       kxzFromfcNEQ_PPM) * c1o8 - (a_001 + c_100);
    // real kxxMyyAverage = (kxxMyyFromfcNEQ_MMM+
    //                       kxxMyyFromfcNEQ_MMP+
    //                       kxxMyyFromfcNEQ_PMP+
    //                       kxxMyyFromfcNEQ_PMM+
    //                       kxxMyyFromfcNEQ_MPM+
    //                       kxxMyyFromfcNEQ_MPP+
    //                       kxxMyyFromfcNEQ_PPP+
    //                       kxxMyyFromfcNEQ_PPM) * c1o8 - (a_100 - b_010);
    // real kxxMzzAverage = (kxxMzzFromfcNEQ_MMM+
    //                       kxxMzzFromfcNEQ_MMP+
    //                       kxxMzzFromfcNEQ_PMP+
    //                       kxxMzzFromfcNEQ_PMM+
    //                       kxxMzzFromfcNEQ_MPM+
    //                       kxxMzzFromfcNEQ_MPP+
    //                       kxxMzzFromfcNEQ_PPP+
    //                       kxxMzzFromfcNEQ_PPM) * c1o8 - (a_100 - c_001);

    ////////////////////////////////////////////////////////////////////////////////
    //! - Set the relative position of the offset cell {-1, 0, 1}
    //!
    real xoff    = offsetCF.xOffCF[k_thread];
    real yoff    = offsetCF.yOffCF[k_thread];
    real zoff    = offsetCF.zOffCF[k_thread];

    real xoff_sq = xoff * xoff;
    real yoff_sq = yoff * yoff;
    real zoff_sq = zoff * zoff;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //drho
    real LaplaceRho = 
        ((xoff != c0o1) || (yoff != c0o1) || (zoff != c0o1))
        ? c0o1
        : -c3o1 * (a_100 * a_100 + b_010 * b_010 + c_001 * c_001) - c6o1 * (b_100 * a_010 + c_100 * a_001 + c_010 * b_001);
    d_000 = ( drho_PPM + drho_PPP + drho_MPM + drho_MPP + drho_PMM + drho_PMP + drho_MMM + drho_MMP) * c1o8;
    d_100 = ( drho_PPM + drho_PPP - drho_MPM - drho_MPP + drho_PMM + drho_PMP - drho_MMM - drho_MMP) * c1o4;
    d_010 = ( drho_PPM + drho_PPP + drho_MPM + drho_MPP - drho_PMM - drho_PMP - drho_MMM - drho_MMP) * c1o4;
    d_001 = (-drho_PPM + drho_PPP - drho_MPM + drho_MPP - drho_PMM + drho_PMP - drho_MMM + drho_MMP) * c1o4;
    d_110 = ( drho_PPM + drho_PPP - drho_MPM - drho_MPP - drho_PMM - drho_PMP + drho_MMM + drho_MMP) * c1o2;
    d_101 = (-drho_PPM + drho_PPP + drho_MPM - drho_MPP - drho_PMM + drho_PMP + drho_MMM - drho_MMP) * c1o2;
    d_011 = (-drho_PPM + drho_PPP - drho_MPM + drho_MPP + drho_PMM - drho_PMP + drho_MMM - drho_MMP) * c1o2;
    d_111 =  -drho_PPM + drho_PPP + drho_MPM - drho_MPP + drho_PMM - drho_PMP - drho_MMM + drho_MMP;

    //////////////////////////////////////////////////////////////////////////
    //! - Extrapolation for refinement in to the wall (polynomial coefficients)
    //!
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // X------X
    // |      | x---x
    // |   ---+-+-> |    ----> offset-vector
    // |      | x---x 
    // X------X
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    a_000 = a_000 + xoff * a_100 + yoff * a_010 + zoff * a_001 + xoff_sq * a_200 + yoff_sq * a_020 + zoff_sq * a_002 +
            xoff * yoff * a_110 + xoff * zoff * a_101 + yoff * zoff * a_011;
    a_100 = a_100 + c2o1 * xoff * a_200 + yoff * a_110 + zoff * a_101;
    a_010 = a_010 + c2o1 * yoff * a_020 + xoff * a_110 + zoff * a_011;
    a_001 = a_001 + c2o1 * zoff * a_002 + xoff * a_101 + yoff * a_011;
    b_000 = b_000 + xoff * b_100 + yoff * b_010 + zoff * b_001 + xoff_sq * b_200 + yoff_sq * b_020 + zoff_sq * b_002 +
            xoff * yoff * b_110 + xoff * zoff * b_101 + yoff * zoff * b_011;
    b_100 = b_100 + c2o1 * xoff * b_200 + yoff * b_110 + zoff * b_101;
    b_010 = b_010 + c2o1 * yoff * b_020 + xoff * b_110 + zoff * b_011;
    b_001 = b_001 + c2o1 * zoff * b_002 + xoff * b_101 + yoff * b_011;
    c_000 = c_000 + xoff * c_100 + yoff * c_010 + zoff * c_001 + xoff_sq * c_200 + yoff_sq * c_020 + zoff_sq * c_002 +
            xoff * yoff * c_110 + xoff * zoff * c_101 + yoff * zoff * c_011;
    c_100 = c_100 + c2o1 * xoff * c_200 + yoff * c_110 + zoff * c_101;
    c_010 = c_010 + c2o1 * yoff * c_020 + xoff * c_110 + zoff * c_011;
    c_001 = c_001 + c2o1 * zoff * c_002 + xoff * c_101 + yoff * c_011;
    d_000 = d_000 + xoff * d_100 + yoff * d_010 + zoff * d_001 + 
            xoff * yoff * d_110 + xoff * zoff * d_101 + yoff * zoff * d_011;
    d_100 = d_100 + yoff * d_110 + zoff * d_101;
    d_010 = d_010 + xoff * d_110 + zoff * d_011;
    d_001 = d_001 + xoff * d_101 + yoff * d_011;

    ////////////////////////////////////////////////////////////////////////////////////
    //! - Set all moments to zero
    //!      
    real m_111 = c0o1;
    real m_211 = c0o1;
    real m_011 = c0o1;
    real m_121 = c0o1;
    real m_101 = c0o1;
    real m_112 = c0o1;
    real m_110 = c0o1;
    real m_221 = c0o1;
    real m_001 = c0o1;
    real m_201 = c0o1;
    real m_021 = c0o1;
    real m_212 = c0o1;
    real m_010 = c0o1;
    real m_210 = c0o1;
    real m_012 = c0o1;
    real m_122 = c0o1;
    real m_100 = c0o1;
    real m_120 = c0o1;
    real m_102 = c0o1;
    real m_222 = c0o1;
    real m_022 = c0o1;
    real m_202 = c0o1;
    real m_002 = c0o1;
    real m_220 = c0o1;
    real m_020 = c0o1;
    real m_200 = c0o1;
    real m_000 = c0o1;

    ////////////////////////////////////////////////////////////////////////////////////
    //! - Define aliases to use the same variable for the distributions (f's):
    //!
    real& f_000 = m_111;
    real& f_P00 = m_211;
    real& f_M00 = m_011;
    real& f_0P0 = m_121;
    real& f_0M0 = m_101;
    real& f_00P = m_112;
    real& f_00M = m_110;
    real& f_PP0 = m_221;
    real& f_MM0 = m_001;
    real& f_PM0 = m_201;
    real& f_MP0 = m_021;
    real& f_P0P = m_212;
    real& f_M0M = m_010;
    real& f_P0M = m_210;
    real& f_M0P = m_012;
    real& f_0PP = m_122;
    real& f_0MM = m_100;
    real& f_0PM = m_120;
    real& f_0MP = m_102;
    real& f_PPP = m_222;
    real& f_MPP = m_022;
    real& f_PMP = m_202;
    real& f_MMP = m_002;
    real& f_PPM = m_220;
    real& f_MPM = m_020;
    real& f_PMM = m_200;
    real& f_MMM = m_000;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Position BSW = MMM: -0.25, -0.25, -0.25
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    real x = -c1o4;
    real y = -c1o4;
    real z = -c1o4;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    //! - Set moments (zeroth to sixth order) on destination node
    //!
    interpolateDistributions(
        x, y, z,
        m_000, 
        m_100, m_010, m_001,
        m_011, m_101, m_110, m_200, m_020, m_002,
        m_111, m_210, m_012, m_201, m_021, m_120, m_102,
        m_022, m_202, m_220, m_211, m_121, m_112,
        m_122, m_212, m_221,
        m_222,
        a_000, a_100, a_010, a_001, a_200, a_020, a_002, a_110,  a_101, a_011, a_111,
        b_000, b_100, b_010, b_001, b_200, b_020, b_002, b_110,  b_101, b_011, b_111,
        c_000, c_100, c_010, c_001, c_200, c_020, c_002, c_110,  c_101, c_011, c_111,
        d_000, d_100, d_010, d_001, d_110, d_101, d_011, d_111,
        LaplaceRho, eps_new, omegaF, 
        kxxMyyAverage, kxxMzzAverage, kyzAverage, kxzAverage, kxyAverage
    );

    //////////////////////////////////////////////////////////////////////////
    // index of the base node and its neighbors
    k_base_000 = indicesFineMMM[k_thread];
    k_base_M00 = neighborXfine [k_base_000];
    k_base_0M0 = neighborYfine [k_base_000];
    k_base_00M = neighborZfine [k_base_000];
    k_base_MM0 = neighborYfine [k_base_M00];
    k_base_M0M = neighborZfine [k_base_M00];
    k_base_0MM = neighborZfine [k_base_0M0];
    k_base_MMM = neighborZfine [k_base_MM0];
    //////////////////////////////////////////////////////////////////////////
    // Set neighbor indices
    k_000 = k_base_000;
    k_M00 = k_base_M00;
    k_0M0 = k_base_0M0;
    k_00M = k_base_00M;
    k_MM0 = k_base_MM0;
    k_M0M = k_base_M0M;
    k_0MM = k_base_0MM;
    k_MMM = k_base_MMM;

    //////////////////////////////////////////////////////////////////////////
    //! - Write distributions: style of reading and writing the distributions from/to
    //! stored arrays dependent on timestep is based on the esoteric twist algorithm
    //! <a href="https://doi.org/10.3390/computation5020019"><b>[ M. Geier et al. (2017),
    //! DOI:10.3390/computation5020019 ]</b></a>
    //!
    (distFine.f[DIR_000])[k_000] = f_000;
    (distFine.f[DIR_P00])[k_000] = f_P00;
    (distFine.f[DIR_M00])[k_M00] = f_M00;
    (distFine.f[DIR_0P0])[k_000] = f_0P0;
    (distFine.f[DIR_0M0])[k_0M0] = f_0M0;
    (distFine.f[DIR_00P])[k_000] = f_00P;
    (distFine.f[DIR_00M])[k_00M] = f_00M;
    (distFine.f[DIR_PP0])[k_000] = f_PP0;
    (distFine.f[DIR_MM0])[k_MM0] = f_MM0;
    (distFine.f[DIR_PM0])[k_0M0] = f_PM0;
    (distFine.f[DIR_MP0])[k_M00] = f_MP0;
    (distFine.f[DIR_P0P])[k_000] = f_P0P;
    (distFine.f[DIR_M0M])[k_M0M] = f_M0M;
    (distFine.f[DIR_P0M])[k_00M] = f_P0M;
    (distFine.f[DIR_M0P])[k_M00] = f_M0P;
    (distFine.f[DIR_0PP])[k_000] = f_0PP;
    (distFine.f[DIR_0MM])[k_0MM] = f_0MM;
    (distFine.f[DIR_0PM])[k_00M] = f_0PM;
    (distFine.f[DIR_0MP])[k_0M0] = f_0MP;
    (distFine.f[DIR_PPP])[k_000] = f_PPP;
    (distFine.f[DIR_MPP])[k_M00] = f_MPP;
    (distFine.f[DIR_PMP])[k_0M0] = f_PMP;
    (distFine.f[DIR_MMP])[k_MM0] = f_MMP;
    (distFine.f[DIR_PPM])[k_00M] = f_PPM;
    (distFine.f[DIR_MPM])[k_M0M] = f_MPM;
    (distFine.f[DIR_PMM])[k_0MM] = f_PMM;
    (distFine.f[DIR_MMM])[k_MMM] = f_MMM;
    //////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Position TSW = MMP: -0.25, -0.25, 0.25
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    x = -c1o4;
    y = -c1o4;
    z =  c1o4;

    ////////////////////////////////////////////////////////////////////////////////
    // Set moments (zeroth to sixth orders) on destination node
    interpolateDistributions(
        x, y, z,
        m_000, 
        m_100, m_010, m_001,
        m_011, m_101, m_110, m_200, m_020, m_002,
        m_111, m_210, m_012, m_201, m_021, m_120, m_102,
        m_022, m_202, m_220, m_211, m_121, m_112,
        m_122, m_212, m_221,
        m_222,
        a_000, a_100, a_010, a_001, a_200, a_020, a_002, a_110,  a_101, a_011, a_111,
        b_000, b_100, b_010, b_001, b_200, b_020, b_002, b_110,  b_101, b_011, b_111,
        c_000, c_100, c_010, c_001, c_200, c_020, c_002, c_110,  c_101, c_011, c_111,
        d_000, d_100, d_010, d_001, d_110, d_101, d_011, d_111,
        LaplaceRho, eps_new, omegaF, 
        kxxMyyAverage, kxxMzzAverage, kyzAverage, kxzAverage, kxyAverage
    );

    ////////////////////////////////////////////////////////////////////////////////////
    // Set neighbor indices
    k_000 = k_00M;
    k_M00 = k_M0M;
    k_0M0 = k_0MM;
    k_00M = neighborZfine[k_00M];
    k_MM0 = k_MMM;
    k_M0M = neighborZfine[k_M0M];
    k_0MM = neighborZfine[k_0MM];
    k_MMM = neighborZfine[k_MMM];

    //////////////////////////////////////////////////////////////////////////
    // Write distributions
    (distFine.f[DIR_000])[k_000] = f_000;
    (distFine.f[DIR_P00])[k_000] = f_P00;
    (distFine.f[DIR_M00])[k_M00] = f_M00;
    (distFine.f[DIR_0P0])[k_000] = f_0P0;
    (distFine.f[DIR_0M0])[k_0M0] = f_0M0;
    (distFine.f[DIR_00P])[k_000] = f_00P;
    (distFine.f[DIR_00M])[k_00M] = f_00M;
    (distFine.f[DIR_PP0])[k_000] = f_PP0;
    (distFine.f[DIR_MM0])[k_MM0] = f_MM0;
    (distFine.f[DIR_PM0])[k_0M0] = f_PM0;
    (distFine.f[DIR_MP0])[k_M00] = f_MP0;
    (distFine.f[DIR_P0P])[k_000] = f_P0P;
    (distFine.f[DIR_M0M])[k_M0M] = f_M0M;
    (distFine.f[DIR_P0M])[k_00M] = f_P0M;
    (distFine.f[DIR_M0P])[k_M00] = f_M0P;
    (distFine.f[DIR_0PP])[k_000] = f_0PP;
    (distFine.f[DIR_0MM])[k_0MM] = f_0MM;
    (distFine.f[DIR_0PM])[k_00M] = f_0PM;
    (distFine.f[DIR_0MP])[k_0M0] = f_0MP;
    (distFine.f[DIR_PPP])[k_000] = f_PPP;
    (distFine.f[DIR_MPP])[k_M00] = f_MPP;
    (distFine.f[DIR_PMP])[k_0M0] = f_PMP;
    (distFine.f[DIR_MMP])[k_MM0] = f_MMP;
    (distFine.f[DIR_PPM])[k_00M] = f_PPM;
    (distFine.f[DIR_MPM])[k_M0M] = f_MPM;
    (distFine.f[DIR_PMM])[k_0MM] = f_PMM;
    (distFine.f[DIR_MMM])[k_MMM] = f_MMM;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Position TSE = PMP: 0.25, -0.25, 0.25
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    x =  c1o4;
    y = -c1o4;
    z =  c1o4;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // Set moments (zeroth to sixth orders) on destination node
    interpolateDistributions(
        x, y, z,
        m_000, 
        m_100, m_010, m_001,
        m_011, m_101, m_110, m_200, m_020, m_002,
        m_111, m_210, m_012, m_201, m_021, m_120, m_102,
        m_022, m_202, m_220, m_211, m_121, m_112,
        m_122, m_212, m_221,
        m_222,
        a_000, a_100, a_010, a_001, a_200, a_020, a_002, a_110,  a_101, a_011, a_111,
        b_000, b_100, b_010, b_001, b_200, b_020, b_002, b_110,  b_101, b_011, b_111,
        c_000, c_100, c_010, c_001, c_200, c_020, c_002, c_110,  c_101, c_011, c_111,
        d_000, d_100, d_010, d_001, d_110, d_101, d_011, d_111,
        LaplaceRho, eps_new, omegaF, 
        kxxMyyAverage, kxxMzzAverage, kyzAverage, kxzAverage, kxyAverage
    );

    ////////////////////////////////////////////////////////////////////////////////////
    // Set neighbor indices
    k_000 = k_M00;
    k_M00 = neighborXfine[k_M00];
    k_0M0 = k_MM0;
    k_00M = k_M0M;
    k_MM0 = neighborXfine[k_MM0];
    k_M0M = neighborXfine[k_M0M];
    k_0MM = k_MMM;
    k_MMM = neighborXfine[k_MMM];

    //////////////////////////////////////////////////////////////////////////
    // Write distributions
    (distFine.f[DIR_000])[k_000] = f_000;
    (distFine.f[DIR_P00])[k_000] = f_P00;
    (distFine.f[DIR_M00])[k_M00] = f_M00;
    (distFine.f[DIR_0P0])[k_000] = f_0P0;
    (distFine.f[DIR_0M0])[k_0M0] = f_0M0;
    (distFine.f[DIR_00P])[k_000] = f_00P;
    (distFine.f[DIR_00M])[k_00M] = f_00M;
    (distFine.f[DIR_PP0])[k_000] = f_PP0;
    (distFine.f[DIR_MM0])[k_MM0] = f_MM0;
    (distFine.f[DIR_PM0])[k_0M0] = f_PM0;
    (distFine.f[DIR_MP0])[k_M00] = f_MP0;
    (distFine.f[DIR_P0P])[k_000] = f_P0P;
    (distFine.f[DIR_M0M])[k_M0M] = f_M0M;
    (distFine.f[DIR_P0M])[k_00M] = f_P0M;
    (distFine.f[DIR_M0P])[k_M00] = f_M0P;
    (distFine.f[DIR_0PP])[k_000] = f_0PP;
    (distFine.f[DIR_0MM])[k_0MM] = f_0MM;
    (distFine.f[DIR_0PM])[k_00M] = f_0PM;
    (distFine.f[DIR_0MP])[k_0M0] = f_0MP;
    (distFine.f[DIR_PPP])[k_000] = f_PPP;
    (distFine.f[DIR_MPP])[k_M00] = f_MPP;
    (distFine.f[DIR_PMP])[k_0M0] = f_PMP;
    (distFine.f[DIR_MMP])[k_MM0] = f_MMP;
    (distFine.f[DIR_PPM])[k_00M] = f_PPM;
    (distFine.f[DIR_MPM])[k_M0M] = f_MPM;
    (distFine.f[DIR_PMM])[k_0MM] = f_PMM;
    (distFine.f[DIR_MMM])[k_MMM] = f_MMM;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Position BSE = PMM: 0.25, -0.25, -0.25
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    x =  c1o4;
    y = -c1o4;
    z = -c1o4;

    ////////////////////////////////////////////////////////////////////////////////
    // Set moments (zeroth to sixth orders) on destination node
    interpolateDistributions(
        x, y, z,
        m_000, 
        m_100, m_010, m_001,
        m_011, m_101, m_110, m_200, m_020, m_002,
        m_111, m_210, m_012, m_201, m_021, m_120, m_102,
        m_022, m_202, m_220, m_211, m_121, m_112,
        m_122, m_212, m_221,
        m_222,
        a_000, a_100, a_010, a_001, a_200, a_020, a_002, a_110,  a_101, a_011, a_111,
        b_000, b_100, b_010, b_001, b_200, b_020, b_002, b_110,  b_101, b_011, b_111,
        c_000, c_100, c_010, c_001, c_200, c_020, c_002, c_110,  c_101, c_011, c_111,
        d_000, d_100, d_010, d_001, d_110, d_101, d_011, d_111,
        LaplaceRho, eps_new, omegaF, 
        kxxMyyAverage, kxxMzzAverage, kyzAverage, kxzAverage, kxyAverage
    );

    ////////////////////////////////////////////////////////////////////////////////////
    // Set neighbor indices
    k_00M = k_000;
    k_M0M = k_M00;
    k_0MM = k_0M0;
    k_MMM = k_MM0;
    k_000 = k_base_M00;
    k_M00 = neighborXfine[k_base_M00];
    k_0M0 = k_base_MM0;
    k_MM0 = neighborXfine[k_base_MM0];

    //////////////////////////////////////////////////////////////////////////
    // Write distributions
    (distFine.f[DIR_000])[k_000] = f_000;
    (distFine.f[DIR_P00])[k_000] = f_P00;
    (distFine.f[DIR_M00])[k_M00] = f_M00;
    (distFine.f[DIR_0P0])[k_000] = f_0P0;
    (distFine.f[DIR_0M0])[k_0M0] = f_0M0;
    (distFine.f[DIR_00P])[k_000] = f_00P;
    (distFine.f[DIR_00M])[k_00M] = f_00M;
    (distFine.f[DIR_PP0])[k_000] = f_PP0;
    (distFine.f[DIR_MM0])[k_MM0] = f_MM0;
    (distFine.f[DIR_PM0])[k_0M0] = f_PM0;
    (distFine.f[DIR_MP0])[k_M00] = f_MP0;
    (distFine.f[DIR_P0P])[k_000] = f_P0P;
    (distFine.f[DIR_M0M])[k_M0M] = f_M0M;
    (distFine.f[DIR_P0M])[k_00M] = f_P0M;
    (distFine.f[DIR_M0P])[k_M00] = f_M0P;
    (distFine.f[DIR_0PP])[k_000] = f_0PP;
    (distFine.f[DIR_0MM])[k_0MM] = f_0MM;
    (distFine.f[DIR_0PM])[k_00M] = f_0PM;
    (distFine.f[DIR_0MP])[k_0M0] = f_0MP;
    (distFine.f[DIR_PPP])[k_000] = f_PPP;
    (distFine.f[DIR_MPP])[k_M00] = f_MPP;
    (distFine.f[DIR_PMP])[k_0M0] = f_PMP;
    (distFine.f[DIR_MMP])[k_MM0] = f_MMP;
    (distFine.f[DIR_PPM])[k_00M] = f_PPM;
    (distFine.f[DIR_MPM])[k_M0M] = f_MPM;
    (distFine.f[DIR_PMM])[k_0MM] = f_PMM;
    (distFine.f[DIR_MMM])[k_MMM] = f_MMM;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Position BNW = MPM: -0.25, 0.25, -0.25
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    x = -c1o4;
    y =  c1o4;
    z = -c1o4;
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set moments (zeroth to sixth orders) on destination node
    interpolateDistributions(
        x, y, z,
        m_000, 
        m_100, m_010, m_001,
        m_011, m_101, m_110, m_200, m_020, m_002,
        m_111, m_210, m_012, m_201, m_021, m_120, m_102,
        m_022, m_202, m_220, m_211, m_121, m_112,
        m_122, m_212, m_221,
        m_222,
        a_000, a_100, a_010, a_001, a_200, a_020, a_002, a_110,  a_101, a_011, a_111,
        b_000, b_100, b_010, b_001, b_200, b_020, b_002, b_110,  b_101, b_011, b_111,
        c_000, c_100, c_010, c_001, c_200, c_020, c_002, c_110,  c_101, c_011, c_111,
        d_000, d_100, d_010, d_001, d_110, d_101, d_011, d_111,
        LaplaceRho, eps_new, omegaF, 
        kxxMyyAverage, kxxMzzAverage, kyzAverage, kxzAverage, kxyAverage
    );

    //////////////////////////////////////////////////////////////////////////
    // index of the base node and its neighbors
    k_base_000 = k_base_0M0;
    k_base_M00 = k_base_MM0;
    k_base_0M0 = neighborYfine[k_base_0M0];
    k_base_00M = k_base_0MM;
    k_base_MM0 = neighborYfine[k_base_MM0];
    k_base_M0M = k_base_MMM;
    k_base_0MM = neighborYfine[k_base_0MM];
    k_base_MMM = neighborYfine[k_base_MMM];

    //////////////////////////////////////////////////////////////////////////
    // Set neighbor indices
    k_000 = k_base_000;
    k_M00 = k_base_M00;
    k_0M0 = k_base_0M0;
    k_00M = k_base_00M;
    k_MM0 = k_base_MM0;
    k_M0M = k_base_M0M;
    k_0MM = k_base_0MM;
    k_MMM = k_base_MMM;

    //////////////////////////////////////////////////////////////////////////
    // Write distributions
    (distFine.f[DIR_000])[k_000] = f_000;
    (distFine.f[DIR_P00])[k_000] = f_P00;
    (distFine.f[DIR_M00])[k_M00] = f_M00;
    (distFine.f[DIR_0P0])[k_000] = f_0P0;
    (distFine.f[DIR_0M0])[k_0M0] = f_0M0;
    (distFine.f[DIR_00P])[k_000] = f_00P;
    (distFine.f[DIR_00M])[k_00M] = f_00M;
    (distFine.f[DIR_PP0])[k_000] = f_PP0;
    (distFine.f[DIR_MM0])[k_MM0] = f_MM0;
    (distFine.f[DIR_PM0])[k_0M0] = f_PM0;
    (distFine.f[DIR_MP0])[k_M00] = f_MP0;
    (distFine.f[DIR_P0P])[k_000] = f_P0P;
    (distFine.f[DIR_M0M])[k_M0M] = f_M0M;
    (distFine.f[DIR_P0M])[k_00M] = f_P0M;
    (distFine.f[DIR_M0P])[k_M00] = f_M0P;
    (distFine.f[DIR_0PP])[k_000] = f_0PP;
    (distFine.f[DIR_0MM])[k_0MM] = f_0MM;
    (distFine.f[DIR_0PM])[k_00M] = f_0PM;
    (distFine.f[DIR_0MP])[k_0M0] = f_0MP;
    (distFine.f[DIR_PPP])[k_000] = f_PPP;
    (distFine.f[DIR_MPP])[k_M00] = f_MPP;
    (distFine.f[DIR_PMP])[k_0M0] = f_PMP;
    (distFine.f[DIR_MMP])[k_MM0] = f_MMP;
    (distFine.f[DIR_PPM])[k_00M] = f_PPM;
    (distFine.f[DIR_MPM])[k_M0M] = f_MPM;
    (distFine.f[DIR_PMM])[k_0MM] = f_PMM;
    (distFine.f[DIR_MMM])[k_MMM] = f_MMM;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Position TNW = MPP: -0.25, 0.25, 0.25
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    x = -c1o4;
    y =  c1o4;
    z =  c1o4;

    ////////////////////////////////////////////////////////////////////////////////
    // Set moments (zeroth to sixth orders) on destination node
    interpolateDistributions(
        x, y, z,
        m_000, 
        m_100, m_010, m_001,
        m_011, m_101, m_110, m_200, m_020, m_002,
        m_111, m_210, m_012, m_201, m_021, m_120, m_102,
        m_022, m_202, m_220, m_211, m_121, m_112,
        m_122, m_212, m_221,
        m_222,
        a_000, a_100, a_010, a_001, a_200, a_020, a_002, a_110,  a_101, a_011, a_111,
        b_000, b_100, b_010, b_001, b_200, b_020, b_002, b_110,  b_101, b_011, b_111,
        c_000, c_100, c_010, c_001, c_200, c_020, c_002, c_110,  c_101, c_011, c_111,
        d_000, d_100, d_010, d_001, d_110, d_101, d_011, d_111,
        LaplaceRho, eps_new, omegaF, 
        kxxMyyAverage, kxxMzzAverage, kyzAverage, kxzAverage, kxyAverage
    );

    ////////////////////////////////////////////////////////////////////////////////////
    // Set neighbor indices
    k_000 = k_00M;
    k_M00 = k_M0M;
    k_0M0 = k_0MM;
    k_00M = neighborZfine[k_00M];
    k_MM0 = k_MMM;
    k_M0M = neighborZfine[k_M0M];
    k_0MM = neighborZfine[k_0MM];
    k_MMM = neighborZfine[k_MMM];

    // if(k_000==522209 || k_000==497639)
    // {
    //     printf("CF: node MPP \t %u \t fP0M %f \n", k_000, f_P0M);
    //     printf("GEO node: %u \n", typeOfGridNode[k_000]);
    //     printf("GEO base: %u \n", typeOfGridNode[k_base_000]);
    // } 

    //////////////////////////////////////////////////////////////////////////
    // Write distributions
    (distFine.f[DIR_000])[k_000] = f_000;
    (distFine.f[DIR_P00])[k_000] = f_P00;
    (distFine.f[DIR_M00])[k_M00] = f_M00;
    (distFine.f[DIR_0P0])[k_000] = f_0P0;
    (distFine.f[DIR_0M0])[k_0M0] = f_0M0;
    (distFine.f[DIR_00P])[k_000] = f_00P;
    (distFine.f[DIR_00M])[k_00M] = f_00M;
    (distFine.f[DIR_PP0])[k_000] = f_PP0;
    (distFine.f[DIR_MM0])[k_MM0] = f_MM0;
    (distFine.f[DIR_PM0])[k_0M0] = f_PM0;
    (distFine.f[DIR_MP0])[k_M00] = f_MP0;
    (distFine.f[DIR_P0P])[k_000] = f_P0P;
    (distFine.f[DIR_M0M])[k_M0M] = f_M0M;
    (distFine.f[DIR_P0M])[k_00M] = f_P0M;
    (distFine.f[DIR_M0P])[k_M00] = f_M0P;
    (distFine.f[DIR_0PP])[k_000] = f_0PP;
    (distFine.f[DIR_0MM])[k_0MM] = f_0MM;
    (distFine.f[DIR_0PM])[k_00M] = f_0PM;
    (distFine.f[DIR_0MP])[k_0M0] = f_0MP;
    (distFine.f[DIR_PPP])[k_000] = f_PPP;
    (distFine.f[DIR_MPP])[k_M00] = f_MPP;
    (distFine.f[DIR_PMP])[k_0M0] = f_PMP;
    (distFine.f[DIR_MMP])[k_MM0] = f_MMP;
    (distFine.f[DIR_PPM])[k_00M] = f_PPM;
    (distFine.f[DIR_MPM])[k_M0M] = f_MPM;
    (distFine.f[DIR_PMM])[k_0MM] = f_PMM;
    (distFine.f[DIR_MMM])[k_MMM] = f_MMM;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Position TNE = PPP: 0.25, 0.25, 0.25
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    x = c1o4;
    y = c1o4;
    z = c1o4;

    ////////////////////////////////////////////////////////////////////////////////
    // Set moments (zeroth to sixth orders) on destination node
    interpolateDistributions(
        x, y, z,
        m_000, 
        m_100, m_010, m_001,
        m_011, m_101, m_110, m_200, m_020, m_002,
        m_111, m_210, m_012, m_201, m_021, m_120, m_102,
        m_022, m_202, m_220, m_211, m_121, m_112,
        m_122, m_212, m_221,
        m_222,
        a_000, a_100, a_010, a_001, a_200, a_020, a_002, a_110,  a_101, a_011, a_111,
        b_000, b_100, b_010, b_001, b_200, b_020, b_002, b_110,  b_101, b_011, b_111,
        c_000, c_100, c_010, c_001, c_200, c_020, c_002, c_110,  c_101, c_011, c_111,
        d_000, d_100, d_010, d_001, d_110, d_101, d_011, d_111,
        LaplaceRho, eps_new, omegaF, 
        kxxMyyAverage, kxxMzzAverage, kyzAverage, kxzAverage, kxyAverage
    );

    ////////////////////////////////////////////////////////////////////////////////////
    // Set neighbor indices
    k_000 = k_M00;
    k_M00 = neighborXfine[k_M00];
    k_0M0 = k_MM0;
    k_00M = k_M0M;
    k_MM0 = neighborXfine[k_MM0];
    k_M0M = neighborXfine[k_M0M];
    k_0MM = k_MMM;
    k_MMM = neighborXfine[k_MMM];

    //////////////////////////////////////////////////////////////////////////
    // Write distributions
    (distFine.f[DIR_000])[k_000] = f_000;
    (distFine.f[DIR_P00])[k_000] = f_P00;
    (distFine.f[DIR_M00])[k_M00] = f_M00;
    (distFine.f[DIR_0P0])[k_000] = f_0P0;
    (distFine.f[DIR_0M0])[k_0M0] = f_0M0;
    (distFine.f[DIR_00P])[k_000] = f_00P;
    (distFine.f[DIR_00M])[k_00M] = f_00M;
    (distFine.f[DIR_PP0])[k_000] = f_PP0;
    (distFine.f[DIR_MM0])[k_MM0] = f_MM0;
    (distFine.f[DIR_PM0])[k_0M0] = f_PM0;
    (distFine.f[DIR_MP0])[k_M00] = f_MP0;
    (distFine.f[DIR_P0P])[k_000] = f_P0P;
    (distFine.f[DIR_M0M])[k_M0M] = f_M0M;
    (distFine.f[DIR_P0M])[k_00M] = f_P0M;
    (distFine.f[DIR_M0P])[k_M00] = f_M0P;
    (distFine.f[DIR_0PP])[k_000] = f_0PP;
    (distFine.f[DIR_0MM])[k_0MM] = f_0MM;
    (distFine.f[DIR_0PM])[k_00M] = f_0PM;
    (distFine.f[DIR_0MP])[k_0M0] = f_0MP;
    (distFine.f[DIR_PPP])[k_000] = f_PPP;
    (distFine.f[DIR_MPP])[k_M00] = f_MPP;
    (distFine.f[DIR_PMP])[k_0M0] = f_PMP;
    (distFine.f[DIR_MMP])[k_MM0] = f_MMP;
    (distFine.f[DIR_PPM])[k_00M] = f_PPM;
    (distFine.f[DIR_MPM])[k_M0M] = f_MPM;
    (distFine.f[DIR_PMM])[k_0MM] = f_PMM;
    (distFine.f[DIR_MMM])[k_MMM] = f_MMM;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //Position BNE = PPM: 0.25, 0.25, -0.25
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    x =  c1o4;
    y =  c1o4;
    z = -c1o4;

    ////////////////////////////////////////////////////////////////////////////////
    // Set moments (zeroth to sixth orders) on destination node
    interpolateDistributions(
        x, y, z,
        m_000, 
        m_100, m_010, m_001,
        m_011, m_101, m_110, m_200, m_020, m_002,
        m_111, m_210, m_012, m_201, m_021, m_120, m_102,
        m_022, m_202, m_220, m_211, m_121, m_112,
        m_122, m_212, m_221,
        m_222,
        a_000, a_100, a_010, a_001, a_200, a_020, a_002, a_110,  a_101, a_011, a_111,
        b_000, b_100, b_010, b_001, b_200, b_020, b_002, b_110,  b_101, b_011, b_111,
        c_000, c_100, c_010, c_001, c_200, c_020, c_002, c_110,  c_101, c_011, c_111,
        d_000, d_100, d_010, d_001, d_110, d_101, d_011, d_111,
        LaplaceRho, eps_new, omegaF, 
        kxxMyyAverage, kxxMzzAverage, kyzAverage, kxzAverage, kxyAverage
    );

    ////////////////////////////////////////////////////////////////////////////////////
    // Set neighbor indices
    k_00M = k_000;
    k_M0M = k_M00;
    k_0MM = k_0M0;
    k_MMM = k_MM0;
    k_000 = k_base_M00;
    k_M00 = neighborXfine[k_base_M00];
    k_0M0 = k_base_MM0;
    k_MM0 = neighborXfine[k_base_MM0];

    //////////////////////////////////////////////////////////////////////////
    // Write distributions
    (distFine.f[DIR_000])[k_000] = f_000;
    (distFine.f[DIR_P00])[k_000] = f_P00;
    (distFine.f[DIR_M00])[k_M00] = f_M00;
    (distFine.f[DIR_0P0])[k_000] = f_0P0;
    (distFine.f[DIR_0M0])[k_0M0] = f_0M0;
    (distFine.f[DIR_00P])[k_000] = f_00P;
    (distFine.f[DIR_00M])[k_00M] = f_00M;
    (distFine.f[DIR_PP0])[k_000] = f_PP0;
    (distFine.f[DIR_MM0])[k_MM0] = f_MM0;
    (distFine.f[DIR_PM0])[k_0M0] = f_PM0;
    (distFine.f[DIR_MP0])[k_M00] = f_MP0;
    (distFine.f[DIR_P0P])[k_000] = f_P0P;
    (distFine.f[DIR_M0M])[k_M0M] = f_M0M;
    (distFine.f[DIR_P0M])[k_00M] = f_P0M;
    (distFine.f[DIR_M0P])[k_M00] = f_M0P;
    (distFine.f[DIR_0PP])[k_000] = f_0PP;
    (distFine.f[DIR_0MM])[k_0MM] = f_0MM;
    (distFine.f[DIR_0PM])[k_00M] = f_0PM;
    (distFine.f[DIR_0MP])[k_0M0] = f_0MP;
    (distFine.f[DIR_PPP])[k_000] = f_PPP;
    (distFine.f[DIR_MPP])[k_M00] = f_MPP;
    (distFine.f[DIR_PMP])[k_0M0] = f_PMP;
    (distFine.f[DIR_MMP])[k_MM0] = f_MMP;
    (distFine.f[DIR_PPM])[k_00M] = f_PPM;
    (distFine.f[DIR_MPM])[k_M0M] = f_MPM;
    (distFine.f[DIR_PMM])[k_0MM] = f_PMM;
    (distFine.f[DIR_MMM])[k_MMM] = f_MMM;
}
