#include "KernelFactoryImp.h"

#include "Parameter/Parameter.h"

#include "Kernel/Kernels/BasicKernels/Advection/Compressible/BGK/BGKCompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/BGKPlus/BGKPlusCompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/Cascade/CascadeCompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/Cumulant/CumulantCompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/CumulantAA2016/CumulantAA2016CompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/CumulantAA2016Bulk/CumulantAA2016CompBulkSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/CumulantAll4/CumulantAll4CompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/CumulantF3/CumulantF3CompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/CumulantF32018/CumulantF32018CompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/CumulantOne/CumulantOneCompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/CumulantOneBulk/CumulantOneCompBulkSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/CumulantOneSponge/CumulantOneCompSpongeSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Compressible/MRT/MRTCompSP27.h"

#include "Kernel/Kernels/BasicKernels/Advection/Incompressible/BGK/BGKIncompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Incompressible/BGKPlus/BGKPlusIncompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Incompressible/Cascade/CascadeIncompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Incompressible/Cumulant1hSP27/Cumulant1hIncompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Incompressible/CumulantIsoSP27/CumulantIsoIncompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Incompressible/CumulantOne/CumulantOneIncompSP27.h"
#include "Kernel/Kernels/BasicKernels/Advection/Incompressible/MRT/MRTIncompSP27.h"

#include "Kernel/Kernels/BasicKernels/AdvectionDiffusion/Compressible/Mod27/ADComp27/ADComp27.h"
#include "Kernel/Kernels/BasicKernels/AdvectionDiffusion/Compressible/Mod7/ADComp7/ADComp7.h"

#include "Kernel/Kernels/BasicKernels/AdvectionDiffusion/Incompressible/Mod27/ADIncomp27/ADIncomp27.h"
#include "Kernel/Kernels/BasicKernels/AdvectionDiffusion/Incompressible/Mod7/ADIncomp7/ADIncomp7.h"

#include "Kernel/Kernels/PorousMediaKernels/Advection/Compressible/CumulantOne/PMCumulantOneCompSP27.h"

#include "Kernel/Kernels/WaleKernels/Advection/Compressible/CumulantAA2016/WaleCumulantAA2016CompSP27.h"
#include "Kernel/Kernels/WaleKernels/Advection/Compressible/CumulantAA2016Debug/WaleCumulantAA2016DebugCompSP27.h"
#include "Kernel/Kernels/WaleKernels/Advection/Compressible/CumulantOne/WaleCumulantOneCompSP27.h"
#include "Kernel/Kernels/WaleKernels/Advection/Compressible/CumulantOneBySoniMalav/WaleBySoniMalavCumulantOneCompSP27.h"

#include "Kernel/Kernels/BasicKernels/Advection/Compressible/AdvecCompStrategy.h"
#include "Kernel/Kernels/BasicKernels/Advection/Incompressible/AdvecIncompStrategy.h"
#include "Kernel/Kernels/BasicKernels/AdvectionDiffusion/Compressible/Mod27/ADMod27CompStrategy.h"
#include "Kernel/Kernels/BasicKernels/AdvectionDiffusion/Compressible/Mod7/ADMod7CompStrategy.h"
#include "Kernel/Kernels/BasicKernels/AdvectionDiffusion/Incompressible/Mod27/ADMod27IncompStrategy.h"
#include "Kernel/Kernels/BasicKernels/AdvectionDiffusion/Incompressible/Mod7/ADMod7IncompStrategy.h"
#include "Kernel/Kernels/PorousMediaKernels/Advection/Compressible/PMAdvecCompStrategy.h"
#include "Kernel/Kernels/WaleKernels/Advection/Compressible/WaleAdvecCompStrategy.h"


std::shared_ptr<KernelFactoryImp> KernelFactoryImp::getInstance()
{
	static std::shared_ptr<KernelFactoryImp> uniqueInstance;
	if (!uniqueInstance)
		uniqueInstance = std::shared_ptr<KernelFactoryImp>(new KernelFactoryImp());
	return uniqueInstance;
}

std::vector<std::shared_ptr<Kernel>> KernelFactoryImp::makeKernels(std::shared_ptr<Parameter> para)
{
	std::vector< std::shared_ptr< Kernel>> kernels;
	for (int level = 0; level <= para->getMaxLevel(); level++)
		kernels.push_back(makeKernel(para, para->getMainKernel(), level));

	if (para->getMaxLevel() > 0)
		if (para->getMultiKernelOn())
			for (int i = 0; i < para->getMultiKernelLevel().size(); i++)
				setKernelAtLevel(kernels, para, para->getMultiKernel().at(i), para->getMultiKernelLevel().at(i));
	return kernels;
}

std::vector<std::shared_ptr<ADKernel>> KernelFactoryImp::makeAdvDifKernels(std::shared_ptr<Parameter> para)
{
	std::vector< std::shared_ptr< ADKernel>> aDKernels;
	for (int level = 0; level <= para->getMaxLevel(); level++)
		aDKernels.push_back(makeAdvDifKernel(para, para->getADKernel(), level));
	return aDKernels;
}

void KernelFactoryImp::setPorousMedia(std::vector<std::shared_ptr<PorousMedia>> pm)
{
	this->pm = pm;
}

void KernelFactoryImp::setKernelAtLevel(std::vector<std::shared_ptr<Kernel>> kernels, std::shared_ptr<Parameter> para, KernelType kernel, int level)
{
	kernels.at(level) = makeKernel(para, kernel, level);
}

std::shared_ptr<Kernel> KernelFactoryImp::makeKernel(std::shared_ptr<Parameter> para, KernelType kernel, int level)
{
	std::shared_ptr<KernelImp> newKernel;
	std::shared_ptr<CheckParameterStrategy> checkStrategy;

	switch (kernel)
	{
	case LB_BGKCompSP27:
		newKernel = BGKCompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_BGKPlusCompSP27:
		newKernel = BGKPlusCompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CascadeCompSP27:
		newKernel = CascadeCompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CumulantCompSP27:
		newKernel = CumulantCompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CumulantAA2016CompSP27:
		newKernel = CumulantAA2016CompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CumulantAA2016CompBulkSP27:
		newKernel = CumulantAA2016CompBulkSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CumulantAll4CompSP27:
		newKernel = CumulantAll4CompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CumulantF3CompSP27:
		newKernel = CumulantF3CompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CumulantF32018CompSP27:
		newKernel = CumulantF32018CompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CumulantOneCompSP27:
		newKernel = CumulantOneCompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CumulantOneCompBulkSP27:
		newKernel = CumulantOneCompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_CumulantOneCompSpongeSP27:
		newKernel = CumulantOneCompSpongeSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;
	case LB_MRTCompSP27:
		newKernel = MRTCompSP27::getNewInstance(para, level);
		checkStrategy = AdvecCompStrategy::getInstance();
		break;


	case LB_BGKIncompSP27:
		newKernel = BGKIncompSP27::getNewInstance(para, level);
		checkStrategy = AdvecIncompStrategy::getInstance();
		break;
	case LB_BGKPlusIncompSP27:
		newKernel = BGKPlusIncompSP27::getNewInstance(para, level);
		checkStrategy = AdvecIncompStrategy::getInstance();
		break;
	case LB_CascadeIncompSP27:
		newKernel = CascadeIncompSP27::getNewInstance(para, level);
		checkStrategy = AdvecIncompStrategy::getInstance();
		break;
	case LB_Cumulant1hIncompSP27:
		newKernel = Cumulant1hIncompSP27::getNewInstance(para, level);
		checkStrategy = AdvecIncompStrategy::getInstance();
		break;
	case LB_CumulantIsoIncompSP27:
		newKernel = CumulantIsoIncompSP27::getNewInstance(para, level);
		checkStrategy = AdvecIncompStrategy::getInstance();
		break;
	case LB_CumulantOneIncompSP27:
		newKernel = CumulantOneIncompSP27::getNewInstance(para, level);
		checkStrategy = AdvecIncompStrategy::getInstance();
		break;
	case LB_MRTIncompSP27:
		newKernel = MRTIncompSP27::getNewInstance(para, level);
		checkStrategy = AdvecIncompStrategy::getInstance();
		break;

	case LB_PMCumulantOneCompSP27:
		newKernel = PMCumulantOneCompSP27::getNewInstance(para, pm, level);
		checkStrategy = PMAdvecCompStrategy::getInstance();
		break;



	case LB_WaleCumulantAA2016CompSP27:
		newKernel = WaleCumulantAA2016CompSP27::getNewInstance(para, level);
		checkStrategy = WaleAdvecCompStrategy::getInstance();
		break;
	case LB_WaleCumulantAA2016DebugCompSP27:
		newKernel = WaleCumulantAA2016DebugCompSP27::getNewInstance(para, level);
		checkStrategy = WaleAdvecCompStrategy::getInstance();
		break;
	case LB_WaleCumulantOneCompSP27:
		newKernel = WaleCumulantOneCompSP27::getNewInstance(para, level);
		checkStrategy = WaleAdvecCompStrategy::getInstance();
		break;
	case LB_WaleBySoniMalavCumulantOneCompSP27:
		newKernel = WaleBySoniMalavCumulantOneCompSP27::getNewInstance(para, level);
		checkStrategy = WaleAdvecCompStrategy::getInstance();
		break;
	default:
		break;
	}

	if (newKernel) {
		newKernel->setCheckParameterStrategy(checkStrategy);
		return newKernel;
	}
	else
		throw  std::exception("Kernelfactory does not know the KernelType.");

	
}

std::shared_ptr<ADKernel> KernelFactoryImp::makeAdvDifKernel(std::shared_ptr<Parameter> para, ADKernelType kernel, int level)
{
	std::shared_ptr<ADKernel> newKernel;
	std::shared_ptr<CheckParameterStrategy> checkStrategy;

	switch (kernel)
	{
	case LB_ADComp27:
		newKernel = ADComp27::getNewInstance(para, level);
		checkStrategy = ADMod27CompStrategy::getInstance();
		break;
	case LB_ADComp7:
		newKernel = ADComp7::getNewInstance(para, level);
		checkStrategy = ADMod7CompStrategy::getInstance();
		break;
	case LB_ADIncomp27:
		newKernel = ADIncomp27::getNewInstance(para, level);
		checkStrategy = ADMod27IncompStrategy::getInstance();
		break;
	case LB_ADIncomp7:
		newKernel = ADIncomp7::getNewInstance(para, level);
		checkStrategy = ADMod7IncompStrategy::getInstance();
		break;
	default:
		break;
	}

	if (newKernel) {
		newKernel->setCheckParameterStrategy(checkStrategy);
		return newKernel;
	}
	else
		throw  std::exception("Kernelfactory does not know the KernelType.");
}

KernelFactoryImp::KernelFactoryImp()
{

}