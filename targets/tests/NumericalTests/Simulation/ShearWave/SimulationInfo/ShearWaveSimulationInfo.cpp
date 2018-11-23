#include "ShearWaveSimulationInfo.h"

#include <sstream>

std::shared_ptr<SimulationInfo> ShearWaveSimulationInfo::getNewInstance(double u0, double v0, int l0, int lx, double viscosity, std::string kernelName, int numberOfSimulations)
{
	return std::shared_ptr<SimulationInfo>(new ShearWaveSimulationInfo(u0, v0, l0, lx, viscosity, kernelName, numberOfSimulations));
}

ShearWaveSimulationInfo::ShearWaveSimulationInfo(double u0, double v0, int l0, int lx, double viscosity, std::string kernelName, int numberOfSimulations) : SimulationInfoImp(lx, viscosity, kernelName, numberOfSimulations)
{
	std::ostringstream oss;
	oss << " u0: " << u0 / (lx / l0) << " v0: " << v0 / (lx / l0);
	this->simulationParameterString = oss.str();

	simulationName = "ShearWave";
}