#include <iostream>
#include <vector>
#include <memory>
#include <ctime>

#include "Core/DataTypes.h"
#include "Core/Logger/Logger.h"

#include "Traffic/TrafficMovementFactory.h"
#include "Traffic/TrafficMovementFactory - Kopie.h"

int main()
{

	//////Basel
	{
		uint numberOfTimesteps = 1000;
		
		//Stephans Logger
		logging::Logger::addStream(&std::cout);
		logging::Logger::setDebugLevel(logging::Logger::Level::INFO_LOW);
		logging::Logger::timeStamp(logging::Logger::ENABLE);
		logging::Logger::enablePrintedRankNumbers(logging::Logger::ENABLE);


		TrafficMovementFactory * factory = new TrafficMovementFactory();
		std::string path = "C:/Users/hiwi/BaselDokumente/";
		factory->initTrafficMovement(path);

		//clock
		std::clock_t start;
		double duration;
		start = std::clock();

		for (uint step = 1; step <= numberOfTimesteps; step++) {
			factory->calculateTimestep(step);
			factory->writeTimestep(step);
		}
			
	
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		factory->endSimulation(numberOfTimesteps, duration);

		std::cout << "Dauer: " << duration << '\n';

		//factory->writeTimestep(numberOfTimesteps);
	
	}




	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	////Testcases

	//{uint numberOfTimesteps = 20;

	//TrafficMovementFactoryTest * factory = new TrafficMovementFactoryTest();
	//factory->initTrafficMovement();
	//factory->loopThroughTimesteps(numberOfTimesteps);

	//std::cout << std::endl << std::endl; }


}

