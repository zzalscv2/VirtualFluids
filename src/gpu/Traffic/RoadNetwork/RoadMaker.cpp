#include "RoadMaker.h"

#include <iostream>

#include "Utilities/VectorHelper.h"
#include "Utilities/invalidInput_error.h"
#include "Utilities/safe_casting.h"


//random vehicle Distribution
RoadMaker::RoadMaker(const uint roadLength, const uint maxVelocity, uint vehicleLength, const real vehicleDensity)
{
	std::uniform_int_distribution<uint> distInt2{ 0, maxVelocity };
	distInt = distInt2;

	this->roadLength = roadLength;
	this->maxVelocity = maxVelocity;
	initVehicleLength(vehicleLength);

	//init vectors
	//this->conc.resize(this->roadLength);
	initCurrentAsEmpty();
	initNext();
	initNeighbors();
	initCurrentWithLongVehicles();

	initVehicleDensity(vehicleDensity);
}


//given vehicle distribution
RoadMaker::RoadMaker(const std::vector<int> vehicleDistribution, const uint maxVelocity, uint vehicleLength)
{
	this->roadLength = castSizeT_Uint(vehicleDistribution.size());

	this->maxVelocity = maxVelocity;
	initVehicleLength(vehicleLength);

	//init vectors
	//this->conc.resize(this->roadLength);
	current = vehicleDistribution;
	initNext();
	initNeighbors();
	initCurrentWithLongVehicles();
}


//empty road
RoadMaker::RoadMaker(const uint roadLength, const uint maxVelocity, uint vehicleLength)
{
	this->roadLength = roadLength;
	this->maxVelocity = maxVelocity;
	initVehicleLength(vehicleLength);

	//init vectors
	//this->conc.resize(this->roadLength);
	initCurrentAsEmpty();
	initNext();
	initNeighbors();
	initCurrentWithLongVehicles();
}


RoadMaker::~RoadMaker()
{
}


void RoadMaker::initNext()
{
	next.resize(roadLength);
	VectorHelper::fillVector(next, -1);
}


void RoadMaker::initNeighbors()
{
	neighbors.resize(roadLength);
	for (uint i = 0; i < roadLength - 1; i++) {
		neighbors[i] = i + 1;
	}
	neighbors[roadLength - 1] = 0;
}


void RoadMaker::initCurrentAsEmpty()
{
	current.resize(roadLength);
	VectorHelper::fillVector(current, -1);
}


void RoadMaker::initCurrentWithLongVehicles()
{
	currentWithLongVehicles.resize(roadLength);
}


void RoadMaker::initVehicleDensity(const real vehicleDensity)
{
	try {
		if (vehicleDensity > 0 && vehicleDensity < 1) {
			initRandomCars(vehicleDensity);
		}
		else {
			throw invalidInput_error("The vehicleDensity should be between 0 and 1");
		}
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}


void RoadMaker::initRandomCars(const real vehicleDensity)
{
	//this method doesn't fill the first cells, so that the safetyDistance isn't violated in a periodic road
	for (uint i = safetyDistance; i < roadLength; i++) {
		double randomNumber = distFloat(engine);
		if (randomNumber <= vehicleDensity) {
			current[i] = randomSpeed();
			i += safetyDistance;
		}
	}
}


int RoadMaker::randomSpeed()
{
	return distInt(engine);
}


void RoadMaker::initVehicleLength(const uint vehicleLength)
{
	try {
		if (vehicleLength == 0) throw  invalidInput_error("The vehicleLength has to be greater than 0");
		this->vehicleLength = vehicleLength;
		this->safetyDistance = vehicleLength - 1;
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}


void RoadMaker::setJunctions(std::vector<std::shared_ptr<Junction> > & junctions)
{
	for (auto& junc : junctions)
		addJunction(junc);
}


void RoadMaker::addJunction(std::shared_ptr<Junction>& junction)
{
	try {

		junction->checkOutCellIndices(roadLength);
		setJunctionAsNeighbor(junction);
		this->junctions.push_back(junction);

		if (junctions.size() > 999) throw std::runtime_error("too many junctions");

	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}


void RoadMaker::setJunctionAsNeighbor(std::shared_ptr<Junction> & junction)
{
	//set the junction as neighbor of the incoming cells

	int junctionIndex = -1000 - castSizeT_Int(junctions.size()); //value range: -1000 to -1999
	std::vector<uint> inCells = junction->getInCellIndices();

	try {

		for (auto cell : inCells) {
			if (cell >= roadLength) throw invalidInput_error("The index of an incoming cell to a junction ist greater than the roadLength.");
			if (neighbors[cell] < 0)				
				std::cout << "The neighboring cell of cell " << cell << " was already definded as sink or junction, no new junction added." << std::endl;
			else
				neighbors[cell] = junctionIndex;
		}

	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}


void RoadMaker::setSinks(std::vector<std::shared_ptr<Sink> > & sinks)
{
	for (auto& sink : sinks)
		addSink(sink);
}


void RoadMaker::addSink(std::shared_ptr<Sink>& sink)
{

	try {

		setSinkAsNeighbor(sink);
		this->sinks.push_back(sink);
		if (sinks.size() > 999) throw std::runtime_error("too many sinks");


	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}

}


void RoadMaker::setSinkAsNeighbor(std::shared_ptr<Sink> & sink)
{
	//set the sink as neighbor of the incoming cell

	int sinkIndex = -2000 - castSizeT_Int(sinks.size()); //value range: -2000 to -2999
	uint sinkCell = sink->getIndex();

	if (sinkCell >= roadLength) throw invalidInput_error("The index of a sink ist greater than the roadLength.");

	if (neighbors[sinkCell] < 0) {
		std::cout << "The neighboring cell of cell " << sinkCell << " was already definded as sink or junction, no new sink added." << std::endl;
	}
	else
	{
		neighbors[sinkCell] = sinkIndex;
	}
}



void RoadMaker::setSources(std::vector< std::shared_ptr<Source> > & sources)
{
	for (auto& source : sources)
		addSource(source);
}


void RoadMaker::addSource(std::shared_ptr<Source>& source)
{
	try {
		if (source->getIndex() >= roadLength) throw invalidInput_error("Source index is greater than roadlength");
		this->sources.push_back(source);
	}

	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}


void RoadMaker::setNeighbor(uint index, uint neighbor)
{
	this->neighbors[index] = neighbor;
}

void RoadMaker::setNeighborForCurve(uint index, uint neighbor)
{
	this->neighbors[index] = neighbor;
	for (uint i = 0; i < this->vehicleLength; i++) {
		if (neighbor < 0) break;
		this->current[neighbor] = -1;		
		neighbor = neighbors[neighbor];
	}
}


uint RoadMaker::getMaxVelocity()
{
	return maxVelocity;
}
