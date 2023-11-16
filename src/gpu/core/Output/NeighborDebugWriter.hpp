#ifndef NEIGHBORDEBUG_HPP
#define NEIGHBORDEBUG_HPP

#include "LBM/LB.h"
#include "Logger.h"
#include "Parameter/Parameter.h"
#include "basics/utilities/UbSystem.h"
#include "gpu/GridGenerator/grid/NodeValues.h"
#include "lbm/constants/D3Q27.h"
#include <basics/writer/WbWriterVtkXmlBinary.h>

#include "StringUtilities/StringUtil.h"
#include "Utilities/FindNeighbors.h"

namespace NeighborDebugWriter
{

inline void writeNeighborLinkLines(LBMSimulationParameter *parH, int direction, const std::string &name,
                                   WbWriter *writer)
{
    VF_LOG_INFO("Write node links in direction {}.", direction);

    const unsigned long long numberOfNodes = parH->numberOfNodes;
    std::vector<UbTupleFloat3> nodes;
    nodes.reserve(numberOfNodes);
    std::vector<UbTupleInt2> cells;
    cells.reserve(numberOfNodes/2);

    for (size_t position = 0; position < numberOfNodes; position++) {
        if (parH->typeOfGridNode[position] != GEO_FLUID)
            continue;

        const double x1 = parH->coordinateX[position];
        const double x2 = parH->coordinateY[position];
        const double x3 = parH->coordinateZ[position];

        const uint positionNeighbor = getNeighborIndex(parH, (uint)position, direction);

        const double x1Neighbor = parH->coordinateX[positionNeighbor];
        const double x2Neighbor = parH->coordinateY[positionNeighbor];
        const double x3Neighbor = parH->coordinateZ[positionNeighbor];

        nodes.emplace_back(float(x1), float(x2), float(x3));
        nodes.emplace_back(float(x1Neighbor), float(x2Neighbor), float(x3Neighbor));

        cells.emplace_back((int)nodes.size() - 2, (int)nodes.size() - 1);
    }
    writer->writeLines(name, nodes, cells);
}

inline void writeNeighborLinkLinesDebug(Parameter *para)
{
    for (int level = 0; level <= para->getMaxLevel(); level++) {
        for (size_t direction = vf::lbm::dir::STARTDIR; direction <= vf::lbm::dir::ENDDIR; direction++) {
            const std::string fileName = para->getFName() + "_" + StringUtil::toString<int>(level) + "_Link_" +
                                         std::to_string(direction) + "_Debug.vtk";
            writeNeighborLinkLines(para->getParH(level).get(), (int)direction, fileName, WbWriterVtkXmlBinary::getInstance());
        }
    }
}

} // namespace NeighborDebugWriter

#endif