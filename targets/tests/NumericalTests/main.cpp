#include <gmock/gmock.h>
#include "mpi.h"

#include "VirtualFluids_GPU/LBM/Simulation.h"

#include "Utilities/ConfigFileReader/ConfigFileReader.h"
#include "Utilities/TestCondition/TestCondition.h"
#include "Utilities/TestConditionFactory/TestConditionFactoryImp.h"
#include "Utilities/Calculator/Calculator.h"
#include "Utilities/TestInformation/TestInformation.h"

static void startNumericalTests(const std::string &configFile)
{
	std::shared_ptr< ConfigFileReader> configReader = ConfigFileReader::getNewInstance();
	configReader->readConfigFile(configFile);

	std::vector< std::shared_ptr< SimulationParameter> > simPara = configReader->getSimulationParameter();
	std::shared_ptr< TestInformation> testInfo = configReader->getTestInformation();

	std::shared_ptr< TestConditionFactory> factory = TestConditionFactoryImp::getNewInstance();
	std::vector< std::shared_ptr< TestCondition> > testConditions = factory->makeTestConditions(simPara);

	for (int i = 0; i < testConditions.size(); i++)
	{
		testInfo->makeSimulationHeadOutput(i);
		testInfo->setSimulationStartTime(i);
		Simulation sim;
		sim.init(testConditions.at(i)->getParameter(), testConditions.at(i)->getGrid(), testConditions.at(i)->getDataWriter());
		sim.run();
		testInfo->setSimulationEndTime(i);

		testConditions.at(i)->getCalculator()->calcAndCopyToTestResults();
	}

	testInfo->makeFinalTestOutput();

	testInfo->writeLogFile();
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	if (argc > 1)
		startNumericalTests(argv[1]);
	else
		std::cout << "Configuration file must be set!: lbmgm <config file>" << std::endl << std::flush;

    MPI_Finalize();

	return 0;
}