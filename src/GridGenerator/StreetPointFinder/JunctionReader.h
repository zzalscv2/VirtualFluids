#ifndef JUNCTIOREADER_H
#define JUNCTIONREADER_H

#include <vector>

#include "Core/DataTypes.h"
#include "Core/Logger/Logger.h"

#include "StreetPointFinder.h"

#include <VirtualFluidsDefinitions.h>

struct VF_PUBLIC JunctionReaderData
{
	std::vector<uint> inCells;
	std::vector<uint> outCells;
	std::vector<int> carCanNotEnterThisOutCell;
	uint trafficLightSwitchTime;

	JunctionReaderData(std::vector<uint> inCells, std::vector<uint> outCells, std::vector<int> carCanNotEnterThisOutCell, uint trafficLightSwitchTime);
};


struct VF_PUBLIC Neighbors
{
	std::vector<int> cells;
	std::vector<int> neighbors;
};



struct VF_PUBLIC JunctionReader
{
	std::vector<JunctionReaderData> junctions;
	Neighbors specialNeighbors;
	StreetPointFinder streetPointFinder;

	void readJunctions(std::string filename, StreetPointFinder streetPointFinder);


private:
	unsigned int getCellIndex(unsigned int streetIndex, char startOrEnd);
};
#endif
