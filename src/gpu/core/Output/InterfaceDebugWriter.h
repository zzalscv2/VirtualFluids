#ifndef INTERFACEDEBUG_HPP
#define INTERFACEDEBUG_HPP

#include <basics/DataTypes.h>

class Parameter;
namespace InterfaceDebugWriter
{
//! \brief write lines that connect the cell centers for the interpolation from coarse to fine cells
void writeInterfaceLinesDebugCF(Parameter* para, uint timeStep = 0);
//! \brief write lines that connect the cell centers for the interpolation from fine to coarse cells
void writeInterfaceLinesDebugFC(Parameter* para, uint timeStep = 0);

void writeInterfaceLinesDebugCFCneighbor(Parameter *para);
void writeInterfaceLinesDebugCFFneighbor(Parameter *para);
void writeInterfaceLinesDebugFCCneighbor(Parameter *para);
void writeInterfaceLinesDebugFCFneighbor(Parameter *para);

void writeInterfaceLinesDebugOff(Parameter *para);
void writeInterfacePointsDebugCFC(Parameter *para);

void writeBcPointsDebug(Parameter *para);
void writePressPointsDebug(Parameter *para);
void writePressNeighborPointsDebug(Parameter *para);

void writeNeighborXPointsDebug(Parameter *para);
void writeNeighborXLinesDebug(Parameter *para);
void writeNeighborYPointsDebug(Parameter *para);
void writeNeighborYLinesDebug(Parameter *para);
void writeNeighborZPointsDebug(Parameter *para);
void writeNeighborZLinesDebug(Parameter *para);

void writeInterfaceCellsDebugCFC(Parameter *para);
void writeInterfaceCellsDebugCFF(Parameter *para);

void writeInterfaceFCC_Send(Parameter *para, int processID = 0);
void writeInterfaceCFC_Recv(Parameter *para, int processID = 0);

void writeSendNodesStream(Parameter *para, int processID = 0);
void writeRecvNodesStream(Parameter *para, int processID = 0);
} // namespace InterfaceDebugWriter

#endif