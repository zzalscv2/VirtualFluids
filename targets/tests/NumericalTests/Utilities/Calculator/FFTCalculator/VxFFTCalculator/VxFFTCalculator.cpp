#include "VxFFTCalculator.h"

#include "Utilities/SimulationResults/SimulationResults.h"

std::shared_ptr<VxFFTCalculator> VxFFTCalculator::getNewInstance(double viscosity, std::shared_ptr<PhiAndNuTest> testResults)
{
	return std::shared_ptr<VxFFTCalculator>(new VxFFTCalculator(viscosity, testResults));
}

void VxFFTCalculator::setVectorToCalc()
{
	data = simResults->getVx();
}

VxFFTCalculator::VxFFTCalculator(double viscosity, std::shared_ptr<PhiAndNuTest> testResults) : FFTCalculator(viscosity, testResults)
{
	
}