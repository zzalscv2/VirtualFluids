#include "TrafficMovementFactory - Kopie.h"

#include <iostream>

#include "GridGenerator/StreetPointFinder/JunctionReader.h"
#include "GridGenerator/StreetPointFinder/SourceReader.h"
#include "GridGenerator/StreetPointFinder/SinkReader.h"

#include "RoadNetwork/RoadMaker.h"
#include "TrafficMovement.h"
#include "Source/SourceRandom.h"
#include "Junction/JunctionRandom.h"
#include "Sink/SinkRandom.h"
#include "Output/ConcentrationByPosition.h"
#include "Output/ConcBySpeedAndAcceleration.h"
#include "Utilities/safe_casting.h"


void TrafficMovementFactoryTest::initTrafficMovement(real * pconcArrayStart)
{
	//Variables

	uint roadLength = 20;

	real vehicleDensity = 0.1f;

	uint vehicleLength = 2;
	uint maxVelocity = 5;
	uint maxAcceleration = 1;

	real dawdlePossibility = (real) 0.2; //typical value: 0.2
	real slowToStartPossibility = (real) 0.4;

	bool useGPU = true;
	bool useSlowToStart = true;


	//make RoadNetwork
	std::vector<int> road(20);
	std::fill(road.begin(), road.end(), -1);
	road[9] = 5;
	auto roadNetwork = std::make_unique<RoadMaker>(road, maxVelocity, vehicleLength);
	//RoadMaker(const uint roadLength, const uint maxVelocity, uint vehicleLength, const real vehicleDensity); //random vehicle Distribution
	//RoadMaker(const std::vector<int> vehicleDistribution, const uint maxVelocity, uint vehicleLength); //given vehicle distribution
	//RoadMaker(const uint roadLength, const uint maxVelocity, uint vehicleLength);//empty road

	//Sources
	std::unique_ptr<Source> source = std::make_unique <SourceRandom>(SourceRandom(0, 0.7f, maxVelocity));
	roadNetwork->addSource(source);

	//Sinks
	std::unique_ptr<Sink> s = std::make_unique <SinkRandom>(SinkRandom(roadLength-1, 0.5f));
	roadNetwork->addSink(move(s));

	//Junctions
	std::vector<uint> inCellIndices = { 9 };
	std::vector<uint> outCellIndices = { 11 };
	
	std::unique_ptr<Junction> j = std::make_unique<JunctionRandom>(JunctionRandom(inCellIndices, outCellIndices));
	roadNetwork->addJunction(std::move(j));

	//init TrafficMovement
	this->simulator = std::make_shared<TrafficMovement>(std::move(roadNetwork), dawdlePossibility);
	if (useSlowToStart) simulator->setSlowToStart(slowToStartPossibility);	
	simulator->setMaxAcceleration(maxAcceleration);
	if (useGPU) simulator->setUseGPU();
	////init ConcentrationOutwriter
	//std::unique_ptr<ConcentrationOutwriter> writer = std::make_unique<ConcBySpeedAndAcceleration>(ConcBySpeedAndAcceleration(simulator->getRoadLength(), pconcArrayStart));
	//simulator->setConcentrationOutwriter(move(writer));
	//Variables

}


void TrafficMovementFactoryTest::calculateTimestep(uint step, uint stepForVTK)
{
	simulator->calculateTimestep(step);
}

void TrafficMovementFactoryTest::loopThroughTimesteps(uint timeSteps)
{
	simulator->setSaveResultsTrue(timeSteps);
	simulator->loopTroughTimesteps(timeSteps);
	//std::cout << "Number of Cars: " << simulator->getNumberOfCars() << std::endl;
}
