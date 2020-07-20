#ifndef KERNEL_FACTORY_H
#define KERNEL_FACTORY_H

#include "Kernel/Utilities/KernelType.h"

#include <memory>
#include <string>
#include <vector>

class Kernel;
class ADKernel;
class Parameter;
class PorousMedia;

class VIRTUALFLUIDS_GPU_EXPORT KernelFactory
{
public:
	virtual std::vector< std::shared_ptr< Kernel>> makeKernels(std::shared_ptr<Parameter> para) = 0;
	virtual std::vector< std::shared_ptr< ADKernel>> makeAdvDifKernels(std::shared_ptr<Parameter> para) = 0;
	
	virtual void setPorousMedia(std::vector<std::shared_ptr<PorousMedia>> pm) = 0;
};
#endif