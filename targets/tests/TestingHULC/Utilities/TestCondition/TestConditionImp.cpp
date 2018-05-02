#include "TestConditionImp.h"

#include "VirtualFluids_GPU/Parameter/Parameter.h"
#include "VirtualFluids_GPU\Output\FileWriter.h"

#include "Utilities\GridReaderforTesting\GridReaderforTesting.h"
#include "Utilities\Results\Results.h"
#include "Utilities\DataWriter\Y2dSliceToResults\Y2dSliceToResults.h"
#include "Utilities\InitialCondition\InitialCondition.h"

#include <sstream>

std::shared_ptr<Parameter> TestConditionImp::getParameter()
{
	return para;
}

std::shared_ptr<GridProvider> TestConditionImp::getGrid()
{
	return grid;
}

std::shared_ptr<DataWriter> TestConditionImp::getDataWriter()
{
	return writeToVector;
}

std::shared_ptr<Results> TestConditionImp::getSimulationResults()
{
	return simResults;
}


std::shared_ptr<TestConditionImp> TestConditionImp::getNewInstance()
{
	return std::shared_ptr<TestConditionImp>(new TestConditionImp());
}

void TestConditionImp::initParameter(real viscosity, std::string aGridPath, std::string filePath, int numberOfGridLevels, unsigned int endTime, unsigned int timeStepLength)
{
	para = Parameter::make();

	para->setMaxDev(1);
	std::vector<int> devices;
	devices.resize(1);
	devices[0] = 1;
	para->setDevices(devices);
	para->setNumprocs(1);

	std::string _prefix = "cells";
	std::string gridPath = aGridPath + "\\";
	para->setFName(filePath + "/" + _prefix);
	para->setPrintFiles(true);

	para->setD3Qxx(27);
	para->setMaxLevel(numberOfGridLevels);

	para->setTEnd(endTime);
	para->setTOut(timeStepLength);

	para->setViscosity(viscosity);
	para->setVelocity(0.096);
	para->setViscosityRatio(1.0);
	para->setVelocityRatio(1.0);
	para->setDensityRatio(1.0);
	para->setFactorPressBC(100000.0);

	para->setgeoVec(gridPath + "geoVec.dat");
	para->setcoordX(gridPath + "coordX.dat");
	para->setcoordY(gridPath + "coordY.dat");
	para->setcoordZ(gridPath + "coordZ.dat");
	para->setneighborX(gridPath + "neighborX.dat");
	para->setneighborY(gridPath + "neighborY.dat");
	para->setneighborZ(gridPath + "neighborZ.dat");
	para->setgeomBoundaryBcQs(gridPath + "geomBoundaryQs.dat");
	para->setgeomBoundaryBcValues(gridPath + "geomBoundaryValues.dat");
	para->setinletBcQs(gridPath + "inletBoundaryQs.dat");
	para->setinletBcValues(gridPath + "inletBoundaryValues.dat");
	para->setoutletBcQs(gridPath + "outletBoundaryQs.dat");
	para->setoutletBcValues(gridPath + "outletBoundaryValues.dat");
	para->settopBcQs(gridPath + "topBoundaryQs.dat");
	para->settopBcValues(gridPath + "topBoundaryValues.dat");
	para->setbottomBcQs(gridPath + "bottomBoundaryQs.dat");
	para->setbottomBcValues(gridPath + "bottomBoundaryValues.dat");
	para->setfrontBcQs(gridPath + "frontBoundaryQs.dat");
	para->setfrontBcValues(gridPath + "frontBoundaryValues.dat");
	para->setbackBcQs(gridPath + "backBoundaryQs.dat");
	para->setbackBcValues(gridPath + "backBoundaryValues.dat");
	para->setnumberNodes(gridPath + "numberNodes.dat");
	para->setLBMvsSI(gridPath + "LBMvsSI.dat");

	para->setForcing(0.0, 0.0, 0.0);

	std::vector<int> dist;
	dist.resize(1);
	dist[0] = 0;
	para->setDistX(dist);
	para->setDistY(dist);
	para->setDistZ(dist);
}

void TestConditionImp::initInitialConditions(std::shared_ptr<InitialCondition> initialCondition)
{
	this->initialCondition = initialCondition;
	this->initialCondition->setParameter(para);
}

void TestConditionImp::initGridProvider()
{
	grid = std::shared_ptr<GridProvider>(new GridReaderforTesting(para, initialCondition));
}

void TestConditionImp::initResults(unsigned int lx, unsigned int lz, unsigned int timeStepLength)
{
	simResults = std::shared_ptr<Results>(new Results(lx,lz,timeStepLength));
}

void TestConditionImp::initDataWriter(unsigned int ySliceForCalculation, unsigned int startTimeCalculation, unsigned int endTime, unsigned int timeStepLength, bool writeFiles, unsigned int startTimeDataWriter)
{
	fileWriter = std::shared_ptr<FileWriter>(new FileWriter());
	writeToVector = std::shared_ptr<ToVectorWriter>(new Y2dSliceToResults(simResults, ySliceForCalculation, startTimeCalculation, endTime, timeStepLength, writeFiles, fileWriter, startTimeDataWriter));
}

