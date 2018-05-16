#ifndef READER_H
#define READER_H

#include "LBM\LB.h"

#include <memory>
#include <vector>
#include <string>

class TestInformation;
class TestParameter;
class TestResults;
class TestCout;
class TestInformationImp;
class PhiAndNuTest;
class SimulationInfo;
class LogFileInformation;

class Reader
{
public:
	static std::shared_ptr< Reader> getNewInstance(const std::string aFilePath);

	std::shared_ptr <TestInformation> getTestInformation();
	std::vector <std::shared_ptr< TestParameter> > getTestParameter();
	std::vector <std::shared_ptr< TestResults> > getAllTestResults();

protected:
	Reader() {};
	Reader(const std::string aFilePath);
	
private:
	void makeTestInformation();
	void makeTestParameter();
	void makeSimulationInfo();
	void makeLogFileInformation();
	bool testShouldRun(std::vector<bool> test);

	std::vector<int> devices;
	real viscosity;
	double minOrderOfAccuracy;
	unsigned int numberOfTimeSteps, basisTimeStepLength, startStepCalculation;
	unsigned int ySliceForCalculation;
	std::vector<real> l;
	std::vector<std::string> grids;
	bool writeFiles;
	std::string filePath;
	unsigned int startStepFileWriter;
	std::string logFilePath;
	std::vector<bool> tgv;
	std::vector<bool> sw;
	real u0SW, v0SW;
	real amplitudeTGV, u0TGV;

	std::vector< std::shared_ptr< PhiAndNuTest> > testResults;
	std::shared_ptr< TestCout> testOutput;
	std::vector< std::shared_ptr< TestParameter> > testParameter;
	std::shared_ptr< TestInformationImp> testInfo;
	std::vector < std::shared_ptr< SimulationInfo> > simInfo;
	std::vector< std::shared_ptr<LogFileInformation> > logInfo;
};
#endif