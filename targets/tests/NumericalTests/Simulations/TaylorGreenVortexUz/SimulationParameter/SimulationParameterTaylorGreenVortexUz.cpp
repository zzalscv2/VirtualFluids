#include "SimulationParameterTaylorGreenVortexUz.h"

#include "Simulations/TaylorGreenVortexUz/InitialConditions/InitialConditionTaylorGreenVortexUz.h"
#include "Simulations/TaylorGreenVortexUz/TaylorGreenVortexUzParameterStruct.h"

#include <sstream>

std::shared_ptr<SimulationParameterTaylorGreenUz> SimulationParameterTaylorGreenUz::getNewInstance(std::string kernelName, double viscosity, std::shared_ptr<TaylorGreenVortexUzParameterStruct> tgvParameterStruct, std::shared_ptr<GridInformationStruct> gridInfo)
{
	return std::shared_ptr<SimulationParameterTaylorGreenUz>(new SimulationParameterTaylorGreenUz(kernelName, viscosity, tgvParameterStruct, gridInfo));
}

SimulationParameterTaylorGreenUz::SimulationParameterTaylorGreenUz(std::string kernelName, double viscosity, std::shared_ptr<TaylorGreenVortexUzParameterStruct> tgvParameterStruct, std::shared_ptr<GridInformationStruct> gridInfo)
:SimulationParameterImp(kernelName, viscosity, tgvParameterStruct->basicSimulationParameter, gridInfo)
{
	this->timeStepLength = tgvParameterStruct->basicTimeStepLength * (gridInfo->lz / l0)*(gridInfo->lz / l0);
	this->maxVelocity = tgvParameterStruct->uz / (lz / l0);

	std::ostringstream oss;
	oss << tgvParameterStruct->vtkFilePath << "/TaylorGreenVortex Uz/Viscosity_" << viscosity << "/uz_" << tgvParameterStruct->uz << "_amplitude_" << tgvParameterStruct->amplitude << "/" << kernelName << "/grid" << lx;
	generateFileDirectionInMyStystem(oss.str());
	this->filePath = oss.str();
}