#ifndef KERNEL_IMP_H
#define KERNEL_IMP_H

#include "Kernel.h"

#include <memory>

#include <cuda/CudaGrid.h>

class CheckParameterStrategy;
class Parameter;
class CudaStreamManager; 
class KernelImp : public Kernel
{
public:
    virtual void run() = 0;
    virtual void runOnIndices(const unsigned int *indices, unsigned int size_indices, CudaStreamIndex streamIndex=CudaStreamIndex::Legacy);

    bool checkParameter();
    std::vector<PreProcessorType> getPreProcessorTypes();
    KernelGroup getKernelGroup();

    void setCheckParameterStrategy(std::shared_ptr<CheckParameterStrategy> strategy);
    bool getKernelUsesFluidNodeIndices();

protected:
    KernelImp(std::shared_ptr<Parameter> para, int level);
    KernelImp();

    std::shared_ptr<Parameter> para;
    std::shared_ptr<CheckParameterStrategy> checkStrategy;
    int level;
    std::vector<PreProcessorType> myPreProcessorTypes;
    KernelGroup myKernelGroup;

    vf::cuda::CudaGrid cudaGrid;

    bool kernelUsesFluidNodeIndices = false;
};

#endif
