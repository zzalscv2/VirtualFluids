#ifndef SAMPLER_H_
#define SAMPLER_H_
#include <basics/PointerDefinitions.h>

class Parameter;
class GridProvider;
class CudaMemoryManager;

class Sampler
{
public:
    Sampler(SPtr<Parameter> para, SPtr<CudaMemoryManager> cudaMemoryManager) : para(para), cudaMemoryManager(cudaMemoryManager) {}
    virtual ~Sampler() = default;

    virtual void init()=0;
    virtual void sample(int level, uint t)=0;
    virtual void getTaggedFluidNodes(GridProvider* gridProvider)=0;
protected:
    SPtr<Parameter> para;
    SPtr<CudaMemoryManager> cudaMemoryManager;
};

#endif