#include "ConfigFileReader.h"

#include <fstream>
#include <iostream>

#include "utilities/input/Input.h"
#include "utilities/StringUtil/StringUtil.h"

#include "Utilities/TestInformation/TestInformationImp.h"
#include "Utilities/TestCout/TestCoutImp.h"
#include "Utilities/SimulationInfo/SimulationInfoImp.h"

#include "Simulation/TaylorGreenVortex/SimulationParameter/TaylorGreenSimulationParameter.h"
#include "Simulation/ShearWave/SimulationParameter/ShearWaveSimulationParameter.h"
#include "Simulation/ShearWave/LogFileInformation/ShearWaveLogFileInformation.h"
#include "Simulation/TaylorGreenVortex/LogFileInformation/TaylorGreenLogFileInformation.h"


#include "Tests/PhiAndNuTest/PhiAndNuTest.h"

#include "Utilities/LogFileInformation/LogFileInformation.h"
#include "Utilities/LogFileInformation/LogFileInformationOutput/LogFileInformationOutput.h"
#include "Utilities/LogFileInformation/BasicSimulationInfo/BasicSimulationInfo.h"
#include "Utilities/LogFileInformation/SimulationTimeInformation/SimulationTimeInformation.h"

bool ConfigFileReader::testShouldRun(std::vector<bool> test)
{
	for (int i = 0; i < test.size(); i++) {
		if (test.at(i))
			return true;
	}
	return false;
}

std::shared_ptr<ConfigFileReader> ConfigFileReader::getNewInstance()
{
	return std::shared_ptr<ConfigFileReader>(new ConfigFileReader());
}

void ConfigFileReader::readConfigFile(const std::string aFilePath)
{
	std::ifstream stream;
	stream.open(aFilePath.c_str(), std::ios::in);
	if (stream.fail())
		throw "can not open config file!\n";

	std::unique_ptr<input::Input> input = input::Input::makeInput(stream, "config");

	devices = StringUtil::toVector(input->getValue("Devices"));
	kernelsToTest = StringUtil::toStringVector(input->getValue("KernelsToTest"));

	viscosity = StringUtil::toDouble(input->getValue("Viscosity"));
	minOrderOfAccuracy = StringUtil::toDouble(input->getValue("MinOrderOfAccuracy"));

	amplitudeTGV = StringUtil::toDouble(input->getValue("Amplitude_TGV"));
	u0TGV = StringUtil::toDouble(input->getValue("u0_TGV"));

	v0SW = StringUtil::toDouble(input->getValue("v0_SW"));
	u0SW = StringUtil::toDouble(input->getValue("u0_SW"));

	numberOfTimeSteps = StringUtil::toInt(input->getValue("NumberOfTimeSteps"));
	basisTimeStepLength = StringUtil::toInt(input->getValue("BasisTimeStepLength"));
	startStepCalculation = StringUtil::toInt(input->getValue("StartStepCalculation"));

	grids.resize(5);
	grids.at(0) = input->getValue("GridPath32");
	grids.at(1) = input->getValue("GridPath64");
	grids.at(2) = input->getValue("GridPath128");
	grids.at(3) = input->getValue("GridPath256");
	grids.at(4) = input->getValue("GridPath512");

	ySliceForCalculation = StringUtil::toInt(input->getValue("ySliceForCalculation"));

	writeFiles = StringUtil::toBool(input->getValue("WriteFiles"));
	filePath = input->getValue("PathForFileWriting");
	startStepFileWriter = StringUtil::toInt(input->getValue("StartStepFileWriter"));
	logFilePath = input->getValue("PathLogFile");;

	tgv.resize(5);
	tgv.at(0) = StringUtil::toBool(input->getValue("TaylorGreenVortex32"));
	tgv.at(1) = StringUtil::toBool(input->getValue("TaylorGreenVortex64"));
	tgv.at(2) = StringUtil::toBool(input->getValue("TaylorGreenVortex128"));
	tgv.at(3) = StringUtil::toBool(input->getValue("TaylorGreenVortex256"));
	tgv.at(4) = StringUtil::toBool(input->getValue("TaylorGreenVortex512"));

	sw.resize(5);
	sw.at(0) = StringUtil::toBool(input->getValue("ShearWave32"));
	sw.at(1) = StringUtil::toBool(input->getValue("ShearWave64"));
	sw.at(2) = StringUtil::toBool(input->getValue("ShearWave128"));
	sw.at(3) = StringUtil::toBool(input->getValue("ShearWave256"));
	sw.at(4) = StringUtil::toBool(input->getValue("ShearWave512"));

	stream.close();

	
	makeSimulationParameter();
	makeTestInformation();
}

std::shared_ptr<TestInformation> ConfigFileReader::getTestInformation()
{
	return testInfo;
}

std::vector<std::shared_ptr<SimulationParameter>> ConfigFileReader::getSimulationParameter()
{
	return simParameter;
}

ConfigFileReader::ConfigFileReader()
{
	tests.resize(0);
	simParameter.resize(0);
	simInfo.resize(0);
	logInfo.resize(0);
	testResults.resize(0);

	lx.resize(5);
	lx.at(0) = 32.0;	
	lx.at(1) = 64.0;
	lx.at(2) = 128.0;
	lx.at(3) = 256.0;
	lx.at(4) = 512.0;

	lz.resize(5);
	lz.at(0) = lx.at(0) * 3.0 / 2.0; 
	lz.at(1) = lx.at(1) * 3.0 / 2.0;
	lz.at(2) = lx.at(2) * 3.0 / 2.0;
	lz.at(3) = lx.at(3) * 3.0 / 2.0;
	lz.at(4) = lx.at(4) * 3.0 / 2.0;

	l0 = 32.0;
	rho0 = 1.0;

	maxLevel = 0; //wird nicht ben�tigt
	numberOfGridLevels = 1;
	
}

void ConfigFileReader::makeTestInformation()
{
	testInfo = TestInformationImp::getNewInstance();

	testInfo->setColorOutput(testOutput);

	makeSimulationInfo();
	testInfo->setSimulationInfo(simInfo);

	makeLogFileInformation();
	testInfo->setLogFileInformation(logInfo);
	testInfo->setLogFilePath(logFilePath);

	makeTestResults();
	testInfo->setTestResults(testResults);
}

void ConfigFileReader::makeSimulationParameter()
{
	testOutput = TestCoutImp::getNewInstance();

	if (testShouldRun(tgv)) {
		std::shared_ptr< PhiAndNuTest> tgvTestResults = PhiAndNuTest::getNewInstance("TaylorGreenVortex", minOrderOfAccuracy, testOutput);
		tests.push_back(tgvTestResults);
		for (int i = 0; i < tgv.size(); i++) {
			if (tgv.at(i)) {
				simParameter.push_back(TaylorGreenSimulationParameter::getNewInstance(u0TGV, amplitudeTGV, viscosity, rho0, lx.at(i), lz.at(i), l0, numberOfTimeSteps, basisTimeStepLength, startStepCalculation, ySliceForCalculation, grids.at(i), maxLevel, numberOfGridLevels, writeFiles, startStepFileWriter, filePath, tgvTestResults, devices));
			}
		}
	}

	if (testShouldRun(sw)) {
		std::shared_ptr< PhiAndNuTest> swTestResults = PhiAndNuTest::getNewInstance("ShearWave", minOrderOfAccuracy, testOutput);
		tests.push_back(swTestResults);

		for (int i = 0; i < sw.size(); i++) {
			if (sw.at(i)) {
				simParameter.push_back(ShearWaveSimulationParameter::getNewInstance(u0SW, v0SW, viscosity, rho0, lx.at(i), lz.at(i), l0, numberOfTimeSteps, basisTimeStepLength, startStepCalculation, ySliceForCalculation, grids.at(i), maxLevel, numberOfGridLevels, writeFiles, startStepFileWriter, filePath, swTestResults, devices));
			}
		}
	}
}

void ConfigFileReader::makeSimulationInfo()
{
	for (int i = 0; i < tgv.size(); i++) {
		if (tgv.at(i)) {
			simInfo.push_back(SimulationInfoImp::getNewInstance(testOutput, "TaylorGreenVortex", lx.at(i)));
		}
	}
	for (int i = 0; i < sw.size(); i++) {
		if (sw.at(i)) {
			simInfo.push_back(SimulationInfoImp::getNewInstance(testOutput, "ShearWave", lx.at(i)));
		}
	}
}

void ConfigFileReader::makeLogFileInformation()
{
	
	logInfo.push_back(LogFileInformationOutput::getNewInstance(devices));
	logInfo.push_back(BasicSimulationInfo::getNewInstance(numberOfTimeSteps, basisTimeStepLength, startStepCalculation, viscosity));

	if (testShouldRun(tgv))
		logInfo.push_back(TaylorGreenInformation::getNewInstance(u0TGV, amplitudeTGV, tgv, lx));
	if (testShouldRun(sw))
		logInfo.push_back(ShearWaveInformation::getNewInstance(u0SW, v0SW, sw, lx));

	logInfo.push_back(SimulationTimeInformation::getNewInstance(simInfo, writeFiles));

	for (int i = 0; i < tests.size(); i++) {
		logInfo.push_back(tests.at(i));
	}
}

void ConfigFileReader::makeTestResults()
{
	for (int i = 0; i < tests.size(); i++) {
		testResults.push_back(tests.at(i));
	}
}
