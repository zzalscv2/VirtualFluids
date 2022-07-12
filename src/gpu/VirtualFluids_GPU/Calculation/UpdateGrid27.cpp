#include "UpdateGrid27.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "Communication/ExchangeData27.h"
#include "Parameter/CudaStreamManager.h"
#include "GPU/TurbulentViscosity.h"
#include "KernelManager/BCKernelManager.h"
#include "KernelManager/ADKernelManager.h"
#include "KernelManager/GridScalingKernelManager.h"
#include "Kernel/Kernel.h"

#include "CollisionStrategy.h"

void UpdateGrid27::updateGrid(int level, unsigned int t)
{
    //////////////////////////////////////////////////////////////////////////

    if (level != para->getFine()) {
        updateGrid(level + 1, t);
        updateGrid(level + 1, t);
    }

    //////////////////////////////////////////////////////////////////////////

    collision(this, para.get(), level, t);

    //////////////////////////////////////////////////////////////////////////

    postCollisionBC(level);

    //////////////////////////////////////////////////////////////////////////

    swapBetweenEvenAndOddTimestep(level);

    //////////////////////////////////////////////////////////////////////////

    if (para->getUseWale())
        calcMacroscopicQuantities(level);

    if (para->getUseTurbulentViscosity())
        calcTurbulentViscosity(level);

    //////////////////////////////////////////////////////////////////////////

    preCollisionBC(level, t);

    //////////////////////////////////////////////////////////////////////////
    if( level != para->getFine() )
    {
        (this->*refinementAndExchange)(level);
    }

    interactWithActuators(level, t);

    interactWithProbes(level, t);
}

void UpdateGrid27::refinementAndExchange_noRefinementAndExchange(int level) {}

void UpdateGrid27::refinementAndExchange_streams_onlyExchangeInterface(int level)
{
    int borderStreamIndex = para->getStreamManager()->getBorderStreamIndex();
    int bulkStreamIndex = para->getStreamManager()->getBulkStreamIndex();

    // fine to coarse border
    fineToCoarse(level, para->getParD(level)->intFCBorder.ICellFCC, para->getParD(level)->intFCBorder.ICellFCF,
                 para->getParD(level)->intFCBorder.kFC, borderStreamIndex);

    // prepare exchange and trigger bulk kernel when finished
    prepareExchangeMultiGPUAfterFtoC(level, borderStreamIndex);
    if (para->getUseStreams())
        para->getStreamManager()->triggerStartBulkKernel(borderStreamIndex);

    // launch bulk kernels (f to c and c to f)
    para->getStreamManager()->waitOnStartBulkKernelEvent(bulkStreamIndex);
    fineToCoarse(level, para->getParD(level)->intFCBulk.ICellFCC, para->getParD(level)->intFCBulk.ICellFCF,
                 para->getParD(level)->intFCBulk.kFC, bulkStreamIndex);
    coarseToFine(level, para->getParD(level)->intCFBulk.ICellCFC, para->getParD(level)->intCFBulk.ICellCFF,
                 para->getParD(level)->intCFBulk.kCF, para->getParD(level)->offCFBulk, bulkStreamIndex);

    // exchange
    exchangeMultiGPUAfterFtoC(level, borderStreamIndex);

    // coarse to fine border
    coarseToFine(level, para->getParD(level)->intCFBorder.ICellCFC, para->getParD(level)->intCFBorder.ICellCFF,
                 para->getParD(level)->intCFBorder.kCF, para->getParD(level)->offCF, borderStreamIndex);
    cudaDeviceSynchronize();
}

void UpdateGrid27::refinementAndExchange_streams_completeExchange(int level)
{
    int borderStreamIndex = para->getStreamManager()->getBorderStreamIndex();
    int bulkStreamIndex = para->getStreamManager()->getBulkStreamIndex();

    // fine to coarse border
    fineToCoarse(level, para->getParD(level)->intFCBorder.ICellFCC, para->getParD(level)->intFCBorder.ICellFCF,
                 para->getParD(level)->intFCBorder.kFC, borderStreamIndex);

    // prepare exchange and trigger bulk kernel when finished
    prepareExchangeMultiGPU(level, borderStreamIndex);
    if (para->getUseStreams())
        para->getStreamManager()->triggerStartBulkKernel(borderStreamIndex);

    // launch bulk kernels (f to c and c to f)
    para->getStreamManager()->waitOnStartBulkKernelEvent(bulkStreamIndex);
    fineToCoarse(level, para->getParD(level)->intFCBulk.ICellFCC, para->getParD(level)->intFCBulk.ICellFCF,
                 para->getParD(level)->intFCBulk.kFC, bulkStreamIndex);
    coarseToFine(level, para->getParD(level)->intCFBulk.ICellCFC, para->getParD(level)->intCFBulk.ICellCFF,
                 para->getParD(level)->intCFBulk.kCF, para->getParD(level)->offCFBulk, bulkStreamIndex);

    // exchange
    exchangeMultiGPU(level, borderStreamIndex);

    // coarse to fine border
    coarseToFine(level, para->getParD(level)->intCFBorder.ICellCFC, para->getParD(level)->intCFBorder.ICellCFF,
                 para->getParD(level)->intCFBorder.kCF, para->getParD(level)->offCF, borderStreamIndex);
    cudaDeviceSynchronize();
}

void UpdateGrid27::refinementAndExchange_noStreams_onlyExchangeInterface(int level)
{
    fineToCoarse(level, para->getParD(level)->intFC.ICellFCC, para->getParD(level)->intFC.ICellFCF, para->getParD(level)->K_FC, -1);

    exchangeMultiGPU_noStreams_withPrepare(level, true);

    coarseToFine(level, para->getParD(level)->intCF.ICellCFC, para->getParD(level)->intCF.ICellCFF, para->getParD(level)->K_CF,
                 para->getParD(level)->offCF, -1);
}

void UpdateGrid27::refinementAndExchange_noStreams_completeExchange(int level)
{
    fineToCoarse(level, para->getParD(level)->intFC.ICellFCC, para->getParD(level)->intFC.ICellFCF, para->getParD(level)->K_FC, -1);

    exchangeMultiGPU_noStreams_withPrepare(level, false);

    coarseToFine(level, para->getParD(level)->intCF.ICellCFC, para->getParD(level)->intCF.ICellCFF, para->getParD(level)->K_CF,
                 para->getParD(level)->offCF, -1);
}

void UpdateGrid27::refinementAndExchange_noExchange(int level)
{
    fineToCoarse(level, para->getParD(level)->intFC.ICellFCC, para->getParD(level)->intFC.ICellFCF, para->getParD(level)->K_FC, -1);
    coarseToFine(level, para->getParD(level)->intCF.ICellCFC, para->getParD(level)->intCF.ICellCFF, para->getParD(level)->K_CF,
                 para->getParD(level)->offCF, -1);
}

void UpdateGrid27::collisionAllNodes(int level, unsigned int t)
{
    kernels.at(level)->run();

    //////////////////////////////////////////////////////////////////////////

    if (para->getSimulatePorousMedia())
        collisionPorousMedia(level);

    //////////////////////////////////////////////////////////////////////////

    if (para->getDiffOn())
        collisionAdvectionDiffusion(level);
}

void UpdateGrid27::collisionUsingIndex(int level, unsigned int t, uint *fluidNodeIndices, uint numberOfFluidNodes, int stream)
{
    if (fluidNodeIndices != nullptr && numberOfFluidNodes != 0)
        kernels.at(level)->runOnIndices(fluidNodeIndices, numberOfFluidNodes, stream);
    else
        std::cout << "In collision: fluidNodeIndices or numberOfFluidNodes not definded"
                      << std::endl;

    //////////////////////////////////////////////////////////////////////////

    if (para->getSimulatePorousMedia())
        collisionPorousMedia(level);

    //////////////////////////////////////////////////////////////////////////

    if (para->getDiffOn())
        collisionAdvectionDiffusion(level);
}

void UpdateGrid27::collisionPorousMedia(int level)
{
    for( std::size_t i = 0; i < pm.size(); i++ )
    {
        KernelPMCumOneCompSP27(para->getParD(level)->numberofthreads,
                               para->getParD(level)->omega,
                               para->getParD(level)->neighborX,
                               para->getParD(level)->neighborY,
                               para->getParD(level)->neighborZ,
                               para->getParD(level)->distributions.f[0],
                               para->getParD(level)->numberOfNodes,
                               level,
                               para->getForcesDev(),
                               pm[i]->getPorosity(),
                               pm[i]->getDarcyLBM(),
                               pm[i]->getForchheimerLBM(),
                               pm[i]->getSizePM(),
                               pm[i]->getHostNodeIDsPM(),
                               para->getParD(level)->isEvenTimestep);
        getLastCudaError("KernelPMCumOneCompSP27 execution failed");
    }
}

void UpdateGrid27::collisionAdvectionDiffusion(int level)
{
    this->adKernelManager->runADcollisionKernel(level);
}

void UpdateGrid27::prepareExchangeMultiGPU(int level, int streamIndex)
{
    prepareExchangeCollDataXGPU27AllNodes(para.get(), level, streamIndex);
    prepareExchangeCollDataYGPU27AllNodes(para.get(), level, streamIndex);
    prepareExchangeCollDataZGPU27AllNodes(para.get(), level, streamIndex);
}

void UpdateGrid27::prepareExchangeMultiGPUAfterFtoC(int level, int streamIndex)
{
    prepareExchangeCollDataXGPU27AfterFtoC(para.get(), level, streamIndex);
    prepareExchangeCollDataYGPU27AfterFtoC(para.get(), level, streamIndex);
    prepareExchangeCollDataZGPU27AfterFtoC(para.get(), level, streamIndex);
}

void UpdateGrid27::exchangeMultiGPU(int level, int streamIndex)
{
    //////////////////////////////////////////////////////////////////////////
    // 3D domain decomposition
    exchangeCollDataXGPU27AllNodes(para.get(), comm, cudaMemoryManager.get(), level, streamIndex);
    exchangeCollDataYGPU27AllNodes(para.get(), comm, cudaMemoryManager.get(), level, streamIndex);
    exchangeCollDataZGPU27AllNodes(para.get(), comm, cudaMemoryManager.get(), level, streamIndex);

    scatterNodesFromRecvBufferXGPU27AllNodes(para.get(), level, streamIndex);
    scatterNodesFromRecvBufferYGPU27AllNodes(para.get(), level, streamIndex);
    scatterNodesFromRecvBufferZGPU27AllNodes(para.get(), level, streamIndex);

    //////////////////////////////////////////////////////////////////////////
    // 3D domain decomposition convection diffusion
    if (para->getDiffOn()) {
        if (para->getUseStreams())
            std::cout << "Warning: Cuda streams not yet implemented for convection diffusion" << std::endl;
        exchangePostCollDataADXGPU27(para.get(), comm, cudaMemoryManager.get(), level);
        exchangePostCollDataADYGPU27(para.get(), comm, cudaMemoryManager.get(), level);
        exchangePostCollDataADZGPU27(para.get(), comm, cudaMemoryManager.get(), level);
    }

    //////////////////////////////////////////////////////////////////////////
    // D E P R E C A T E D
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 1D domain decomposition
    // exchangePostCollDataGPU27(para, comm, level);
}
void UpdateGrid27::exchangeMultiGPU_noStreams_withPrepare(int level, bool useReducedComm)
{
    //////////////////////////////////////////////////////////////////////////
    // 3D domain decomposition
    if (useReducedComm) {
        // X
        prepareExchangeCollDataXGPU27AfterFtoC(para.get(), level, -1);
        exchangeCollDataXGPU27AfterFtoC(para.get(), comm, cudaMemoryManager.get(), level, -1);
        scatterNodesFromRecvBufferXGPU27AfterFtoC(para.get(), level, -1);
        // Y
        prepareExchangeCollDataYGPU27AfterFtoC(para.get(), level, -1);
        exchangeCollDataYGPU27AfterFtoC(para.get(), comm, cudaMemoryManager.get(), level, -1);
        scatterNodesFromRecvBufferYGPU27AfterFtoC(para.get(), level, -1);
        // Z
        prepareExchangeCollDataZGPU27AfterFtoC(para.get(), level, -1);
        exchangeCollDataZGPU27AfterFtoC(para.get(), comm, cudaMemoryManager.get(), level, -1);
        scatterNodesFromRecvBufferZGPU27AfterFtoC(para.get(), level, -1);
    } else {
        // X
        prepareExchangeCollDataXGPU27AllNodes(para.get(), level, -1);
        exchangeCollDataXGPU27AllNodes(para.get(), comm, cudaMemoryManager.get(), level, -1);
        scatterNodesFromRecvBufferXGPU27AllNodes(para.get(), level, -1);
        // Y
        prepareExchangeCollDataYGPU27AllNodes(para.get(), level, -1);
        exchangeCollDataYGPU27AllNodes(para.get(), comm, cudaMemoryManager.get(), level, -1);
        scatterNodesFromRecvBufferYGPU27AllNodes(para.get(), level, -1);
        // Z
        prepareExchangeCollDataZGPU27AllNodes(para.get(), level, -1);
        exchangeCollDataZGPU27AllNodes(para.get(), comm, cudaMemoryManager.get(), level, -1);
        scatterNodesFromRecvBufferZGPU27AllNodes(para.get(), level, -1);
    }

    //////////////////////////////////////////////////////////////////////////
    // 3D domain decomposition convection diffusion
    if (para->getDiffOn()) {
        if (para->getUseStreams())
            std::cout << "Warning: Cuda streams not yet implemented for convection diffusion" << std::endl;
        exchangePostCollDataADXGPU27(para.get(), comm, cudaMemoryManager.get(), level);
        exchangePostCollDataADYGPU27(para.get(), comm, cudaMemoryManager.get(), level);
        exchangePostCollDataADZGPU27(para.get(), comm, cudaMemoryManager.get(), level);
    }
}
void UpdateGrid27::exchangeMultiGPUAfterFtoC(int level, int streamIndex)
{
    //////////////////////////////////////////////////////////////////////////
    // 3D domain decomposition
    exchangeCollDataXGPU27AfterFtoC(para.get(), comm, cudaMemoryManager.get(), level, streamIndex);
    exchangeCollDataYGPU27AfterFtoC(para.get(), comm, cudaMemoryManager.get(), level, streamIndex);
    exchangeCollDataZGPU27AfterFtoC(para.get(), comm, cudaMemoryManager.get(), level, streamIndex);

    scatterNodesFromRecvBufferXGPU27AfterFtoC(para.get(), level, streamIndex);
    scatterNodesFromRecvBufferYGPU27AfterFtoC(para.get(), level, streamIndex);
    scatterNodesFromRecvBufferZGPU27AfterFtoC(para.get(), level, streamIndex);

    //////////////////////////////////////////////////////////////////////////
    // 3D domain decomposition convection diffusion
    if (para->getDiffOn()) {
        if (para->getUseStreams())
            std::cout << "Warning: Cuda streams not yet implemented for convection diffusion" << std::endl;
        exchangePostCollDataADXGPU27(para.get(), comm, cudaMemoryManager.get(), level);
        exchangePostCollDataADYGPU27(para.get(), comm, cudaMemoryManager.get(), level);
        exchangePostCollDataADZGPU27(para.get(), comm, cudaMemoryManager.get(), level);
    }
}

void UpdateGrid27::postCollisionBC(int level)
{
    //////////////////////////////////////////////////////////////////////////
    // V E L O C I T Y (I N F L O W)
    this->bcKernelManager->runVelocityBCKernelPost(level);

    //////////////////////////////////////////////////////////////////////////
    // N O - S L I P
    this->bcKernelManager->runNoSlipBCKernelPost(level);

    //////////////////////////////////////////////////////////////////////////
    // S L I P
    this->bcKernelManager->runSlipBCKernelPost(level);

    //////////////////////////////////////////////////////////////////////////
    // S T R E S S (wall model)
    this->bcKernelManager->runStressWallModelKernelPost(level);

    //////////////////////////////////////////////////////////////////////////
    // G E O M E T R Y
    this->bcKernelManager->runGeoBCKernelPost(level);

    //////////////////////////////////////////////////////////////////////////
    // O U T F L O W
    this->bcKernelManager->runOutflowBCKernelPre(level);

    //////////////////////////////////////////////////////////////////////////
    // P R E S S U R E
    this->bcKernelManager->runPressureBCKernelPost(level);

    //////////////////////////////////////////////////////////////////////////
    // A D V E C T I O N    D I F F U S I O N
    if (para->getDiffOn())
    {
        this->adKernelManager->runADgeometryBCKernel(level);
        this->adKernelManager->runADveloBCKernel(level);
        this->adKernelManager->runADslipBCKernel(level);
        this->adKernelManager->runADpressureBCKernel(level);
    }
}

void UpdateGrid27::swapBetweenEvenAndOddTimestep(int level)
{
    if (para->getParD(level)->isEvenTimestep==true)  para->getParD(level)->isEvenTimestep=false;
    else                                        para->getParD(level)->isEvenTimestep=true;
}

void UpdateGrid27::calcMacroscopicQuantities(int level)
{
    CalcMacCompSP27(para->getParD(level)->velocityX,
                    para->getParD(level)->velocityY,
                    para->getParD(level)->velocityZ,
                    para->getParD(level)->rho,
                    para->getParD(level)->pressure,
                    para->getParD(level)->typeOfGridNode,
                    para->getParD(level)->neighborX,
                    para->getParD(level)->neighborY,
                    para->getParD(level)->neighborZ,
                    para->getParD(level)->numberOfNodes,
                    para->getParD(level)->numberofthreads,
                    para->getParD(level)->distributions.f[0],
                    para->getParD(level)->isEvenTimestep);
    getLastCudaError("CalcMacSP27 execution failed");
}

void UpdateGrid27::preCollisionBC(int level, unsigned int t)
{
    //////////////////////////////////////////////////////////////////////////
    // V E L O C I T Y (I N F L O W)
    this->bcKernelManager->runVelocityBCKernelPre(level);

    //////////////////////////////////////////////////////////////////////////
    // G E O M E T R Y
    this->bcKernelManager->runGeoBCKernelPre(level, t, cudaMemoryManager.get());

    //////////////////////////////////////////////////////////////////////////
    // P R E S S U R E
    this->bcKernelManager->runPressureBCKernelPre(level);

    //////////////////////////////////////////////////////////////////////////
    // O U T F L O W
    this->bcKernelManager->runOutflowBCKernelPre(level);

    //////////////////////////////////////////////////////////////////////////////////
    ////only for a round off error test
    //para->cudaCopyTestREtoHost(0,para->getParH(0)->pressureBC.numberOfBCnodes);
    //printRE(para, t);
    //////////////////////////////////////////////////////////////////////////////////
}

void UpdateGrid27::fineToCoarse(int level, uint *iCellFCC, uint *iCellFCF, uint k_FC, int streamIndex)
{
    gridScalingKernelManager->runFineToCoarseKernelLB(level, iCellFCC, iCellFCF, k_FC, streamIndex);

    if (para->getDiffOn()) {
        if (streamIndex != -1) {
            printf("fineToCoarse Advection Diffusion not implemented"); // TODO
            return;
        }
        gridScalingKernelManager->runFineToCoarseKernelAD(level);
    }
}

void UpdateGrid27::coarseToFine(int level, uint *iCellCFC, uint *iCellCFF, uint k_CF, OffCF &offCF,
                                int streamIndex)
{
    this->gridScalingKernelManager->runCoarseToFineKernelLB(level, iCellCFC, iCellCFF, k_CF, offCF, streamIndex);

    if (para->getDiffOn())
    {
        if (streamIndex != -1){
            printf("CoarseToFineWithStream Advection Diffusion not implemented"); // TODO
            return;
        }
        this->gridScalingKernelManager->runCoarseToFineKernelAD(level);
    }
}

void UpdateGrid27::interactWithActuators(int level, unsigned int t)
{
    for( SPtr<PreCollisionInteractor> actuator: para->getActuators() )
    {
        actuator->interact(para.get(), cudaMemoryManager.get(), level, t);
    }
}

void  UpdateGrid27::interactWithProbes(int level, unsigned int t)
{
    for( SPtr<PreCollisionInteractor> probe: para->getProbes() )
    {
        probe->interact(para.get(), cudaMemoryManager.get(), level, t);
    }
}

void  UpdateGrid27::calcTurbulentViscosity(int level)
{
    if(para->getUseAMD())
        calcTurbulentViscosityAMD(para.get(), level);
}


UpdateGrid27::UpdateGrid27(SPtr<Parameter> para, vf::gpu::Communicator &comm, SPtr<CudaMemoryManager> cudaMemoryManager,
                           std::vector<std::shared_ptr<PorousMedia>> &pm, std::vector<SPtr<Kernel>> &kernels , BoundaryConditionFactory* bcFactory)
    : para(para), comm(comm), cudaMemoryManager(cudaMemoryManager), pm(pm), kernels(kernels)
{
    this->collision = getFunctionForCollisionAndExchange(para->getUseStreams(), para->getNumprocs(), para->getKernelNeedsFluidNodeIndicesToRun());
    chooseFunctionForRefinementAndExchange();
    this->bcKernelManager = std::make_shared<BCKernelManager>(para, bcFactory);
    this->adKernelManager = std::make_shared<ADKernelManager>(para);
    this->gridScalingKernelManager =  std::make_shared<GridScalingKernelManager>(para);
}

void UpdateGrid27::chooseFunctionForRefinementAndExchange()
{
    std::cout << "Function used for refinementAndExchange: ";
    if (para->getMaxLevel() == 0) {
        this->refinementAndExchange = &UpdateGrid27::refinementAndExchange_noRefinementAndExchange;
        std::cout << "only one level - no function needed." << std::endl;

    } else if (para->getNumprocs() == 1) {
        this->refinementAndExchange = &UpdateGrid27::refinementAndExchange_noExchange;
        std::cout << "refinementAndExchange_noExchange()" << std::endl;

    } else if (para->getNumprocs() > 1 && para->getUseStreams() && para->useReducedCommunicationAfterFtoC) {
        this->refinementAndExchange = &UpdateGrid27::refinementAndExchange_streams_onlyExchangeInterface;
        std::cout << "refinementAndExchange_streams_onlyExchangeInterface()" << std::endl;

    } else if(para->getNumprocs() > 1 && para->getUseStreams() && !para->useReducedCommunicationAfterFtoC){
        this->refinementAndExchange = &UpdateGrid27::refinementAndExchange_streams_completeExchange;
        std::cout << "refinementAndExchange_streams_completeExchange()" << std::endl;

    } else if (para->getNumprocs() > 1 && !para->getUseStreams() && para->useReducedCommunicationAfterFtoC) {
        this->refinementAndExchange = &UpdateGrid27::refinementAndExchange_noStreams_onlyExchangeInterface;
        std::cout << "refinementAndExchange_noStreams_onlyExchangeInterface()" << std::endl;

    } else {
        this->refinementAndExchange = &UpdateGrid27::refinementAndExchange_noStreams_completeExchange;
        std::cout << "refinementAndExchange_noStreams_completeExchange()" << std::endl;
    }
}


void UpdateGrid27::exchangeData(int level)
{
    exchangeMultiGPU_noStreams_withPrepare(level, false);
}