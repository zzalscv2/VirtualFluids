#include "IndexRearrangementForStreams.h"

#include "Parameter/Parameter.h"
#include <GridGenerator/grid/GridBuilder/GridBuilder.h>
#include "Communication/Communicator.h"

#include <iostream>
#include <algorithm>

IndexRearrangementForStreams::IndexRearrangementForStreams(std::shared_ptr<Parameter> para,
                                                           std::shared_ptr<GridBuilder> builder)
    : para(para), builder(builder)
{ }

void IndexRearrangementForStreams::initCommunicationArraysForCommAfterFinetoCoarseX(const uint &level, int j,
                                                                                    int direction)
{
    // init send indices for communication after coarse to fine
    std::cout << "communication: reorder send indices X ";
    para->initNumberOfProcessNeighborsAfterFtoCX(level);
    std::vector<uint> sendIndicesForCommAfterFtoCPositions;
    reorderSendIndicesForCommAfterFtoCX(direction, level, j, sendIndicesForCommAfterFtoCPositions);
    para->setSendProcessNeighborsAfterFtoCX(para->getParH(level)->sendProcessNeighborsAfterFtoCX[j].numberOfNodes,
                                            level, j);

    // send sendIndicesForCommAfterFtoCPositions to receiving process and receive recvIndicesForCommAfterFtoCPositions from sending process
    std::cout << "mpi send and receive ";
    std::vector<uint> recvIndicesForCommAfterFtoCPositions;
    recvIndicesForCommAfterFtoCPositions.resize(
        (size_t)para->getParH(level)->sendProcessNeighborsAfterFtoCX[j].numberOfNodes *
        2); // give vector an arbitraty size (larger than needed) // TODO: This is stupid! Find a better way
    auto comm = vf::gpu::Communicator::getInstanz();
    comm->exchangeIndices(recvIndicesForCommAfterFtoCPositions.data(), (int)recvIndicesForCommAfterFtoCPositions.size(),
                          para->getParH(level)->recvProcessNeighborX[j].rankNeighbor,
                          sendIndicesForCommAfterFtoCPositions.data(), (int)sendIndicesForCommAfterFtoCPositions.size(),
                          para->getParH(level)->sendProcessNeighborX[j].rankNeighbor);
    // resize receiving vector to correct size
    auto it = std::unique(recvIndicesForCommAfterFtoCPositions.begin(), recvIndicesForCommAfterFtoCPositions.end());
    recvIndicesForCommAfterFtoCPositions.erase(std::prev(it, 1), recvIndicesForCommAfterFtoCPositions.end());

    // init receive indices for communication after coarse to fine
    std::cout << "reorder receive indices ";
    reorderRecvIndicesForCommAfterFtoCX(direction, level, j, recvIndicesForCommAfterFtoCPositions);
    para->setRecvProcessNeighborsAfterFtoCX(para->getParH(level)->recvProcessNeighborsAfterFtoCX[j].numberOfNodes,
                                            level, j);
    copyProcessNeighborToAfterFtoCX(level, j);

    std::cout << "done." << std::endl;
}

void IndexRearrangementForStreams::initCommunicationArraysForCommAfterFinetoCoarseY(const uint &level, int j, int direction)
{
    // init send indices for communication after coarse to fine
    std::cout << "communication: reorder send indices Y ";
    para->initNumberOfProcessNeighborsAfterFtoCY(level);
    std::vector<uint> sendIndicesForCommAfterFtoCPositions;
    reorderSendIndicesForCommAfterFtoCY(direction, level, j, sendIndicesForCommAfterFtoCPositions);
    para->setSendProcessNeighborsAfterFtoCY(para->getParH(level)->sendProcessNeighborsAfterFtoCY[j].numberOfNodes,
                                            level, j);

    // send sendIndicesForCommAfterFtoCPositions to receiving process and receive recvIndicesForCommAfterFtoCPositions from sending process
    std::cout << "mpi send and receive ";
    std::vector<uint> recvIndicesForCommAfterFtoCPositions; 
    recvIndicesForCommAfterFtoCPositions.resize((size_t) para->getParH(level)->sendProcessNeighborsAfterFtoCY[j].numberOfNodes *
                                                2); // give vector an arbitraty size (larger than needed) // TODO: This is stupid! Find a better way
    auto comm = vf::gpu::Communicator::getInstanz();
    comm->exchangeIndices(recvIndicesForCommAfterFtoCPositions.data(), (int)recvIndicesForCommAfterFtoCPositions.size(),
                          para->getParH(level)->recvProcessNeighborY[j].rankNeighbor,
                          sendIndicesForCommAfterFtoCPositions.data(), (int)sendIndicesForCommAfterFtoCPositions.size(),
                          para->getParH(level)->sendProcessNeighborY[j].rankNeighbor);
    // resize receiving vector to correct size
    auto it = std::unique(recvIndicesForCommAfterFtoCPositions.begin(), recvIndicesForCommAfterFtoCPositions.end());
    recvIndicesForCommAfterFtoCPositions.erase(std::prev(it, 1), recvIndicesForCommAfterFtoCPositions.end());

    // init receive indices for communication after coarse to fine
    std::cout << "reorder receive indices ";
    reorderRecvIndicesForCommAfterFtoCY(direction, level, j, recvIndicesForCommAfterFtoCPositions);
    para->setRecvProcessNeighborsAfterFtoCY(para->getParH(level)->recvProcessNeighborsAfterFtoCY[j].numberOfNodes,
                                            level, j);

    copyProcessNeighborToAfterFtoCY(level, j);

    std::cout << "done." << std::endl;
}

void IndexRearrangementForStreams::initCommunicationArraysForCommAfterFinetoCoarseZ(const uint &level, int j, int direction)
{
    // init send indices for communication after coarse to fine
    std::cout << "communication: reorder send indices Z ";
    para->initNumberOfProcessNeighborsAfterFtoCZ(level);
    std::vector<uint> sendIndicesForCommAfterFtoCPositions;
    reorderSendIndicesForCommAfterFtoCZ(direction, level, j, sendIndicesForCommAfterFtoCPositions);
    para->setSendProcessNeighborsAfterFtoCZ(para->getParH(level)->sendProcessNeighborsAfterFtoCZ[j].numberOfNodes,
                                            level, j);

    // send sendIndicesForCommAfterFtoCPositions to receiving process and receive recvIndicesForCommAfterFtoCPositions from sending process
    std::cout << "mpi send and receive ";
    std::vector<uint> recvIndicesForCommAfterFtoCPositions; 
    recvIndicesForCommAfterFtoCPositions.resize((size_t) para->getParH(level)->sendProcessNeighborsAfterFtoCZ[j].numberOfNodes *
                                                2); // give vector an arbitraty size (larger than needed) // TODO: This is stupid! Find a better way
    auto comm = vf::gpu::Communicator::getInstanz();
    comm->exchangeIndices(recvIndicesForCommAfterFtoCPositions.data(), (int)recvIndicesForCommAfterFtoCPositions.size(),
                          para->getParH(level)->recvProcessNeighborZ[j].rankNeighbor,
                          sendIndicesForCommAfterFtoCPositions.data(), (int)sendIndicesForCommAfterFtoCPositions.size(),
                          para->getParH(level)->sendProcessNeighborZ[j].rankNeighbor);
    // resize receiving vector to correct size
    auto it = std::unique(recvIndicesForCommAfterFtoCPositions.begin(), recvIndicesForCommAfterFtoCPositions.end());
    recvIndicesForCommAfterFtoCPositions.erase(std::prev(it, 1), recvIndicesForCommAfterFtoCPositions.end());

    // init receive indices for communication after coarse to fine
    std::cout << "reorder receive indices ";
    reorderRecvIndicesForCommAfterFtoCZ(direction, level, j, recvIndicesForCommAfterFtoCPositions);
    para->setRecvProcessNeighborsAfterFtoCZ(para->getParH(level)->recvProcessNeighborsAfterFtoCZ[j].numberOfNodes,
                                            level, j);

    copyProcessNeighborToAfterFtoCZ(level, j);

    std::cout << "done." << std::endl;
}

void IndexRearrangementForStreams::copyProcessNeighborToAfterFtoCX(const uint &level, int j)
{
    // init f[0]*
    para->getParD(level)->sendProcessNeighborsAfterFtoCX[j].f[0] = para->getParD(level)->sendProcessNeighborX[j].f[0];
    para->getParH(level)->sendProcessNeighborsAfterFtoCX[j].f[0] = para->getParH(level)->sendProcessNeighborX[j].f[0];
    para->getParD(level)->recvProcessNeighborsAfterFtoCX[j].f[0] = para->getParD(level)->recvProcessNeighborX[j].f[0];
    para->getParH(level)->recvProcessNeighborsAfterFtoCX[j].f[0] = para->getParH(level)->recvProcessNeighborX[j].f[0];

    // init index*
    para->getParD(level)->sendProcessNeighborsAfterFtoCX[j].index = para->getParD(level)->sendProcessNeighborX[j].index;
    para->getParH(level)->sendProcessNeighborsAfterFtoCX[j].index = para->getParH(level)->sendProcessNeighborX[j].index;
    para->getParD(level)->recvProcessNeighborsAfterFtoCX[j].index = para->getParD(level)->recvProcessNeighborX[j].index;
    para->getParH(level)->recvProcessNeighborsAfterFtoCX[j].index = para->getParH(level)->recvProcessNeighborX[j].index;

    // rank neighbor
    para->getParH(level)->sendProcessNeighborsAfterFtoCX[j].rankNeighbor = para->getParH(level)->sendProcessNeighborX[j].rankNeighbor;
    para->getParH(level)->recvProcessNeighborsAfterFtoCX[j].rankNeighbor = para->getParH(level)->recvProcessNeighborX[j].rankNeighbor;
}

void IndexRearrangementForStreams::copyProcessNeighborToAfterFtoCY(const uint &level, int j)
{
    // init f[0]*
    para->getParD(level)->sendProcessNeighborsAfterFtoCY[j].f[0] = para->getParD(level)->sendProcessNeighborY[j].f[0];
    para->getParH(level)->sendProcessNeighborsAfterFtoCY[j].f[0] = para->getParH(level)->sendProcessNeighborY[j].f[0];
    para->getParD(level)->recvProcessNeighborsAfterFtoCY[j].f[0] = para->getParD(level)->recvProcessNeighborY[j].f[0];
    para->getParH(level)->recvProcessNeighborsAfterFtoCY[j].f[0] = para->getParH(level)->recvProcessNeighborY[j].f[0];

    // init index*
    para->getParD(level)->sendProcessNeighborsAfterFtoCY[j].index = para->getParD(level)->sendProcessNeighborY[j].index;
    para->getParH(level)->sendProcessNeighborsAfterFtoCY[j].index = para->getParH(level)->sendProcessNeighborY[j].index;
    para->getParD(level)->recvProcessNeighborsAfterFtoCY[j].index = para->getParD(level)->recvProcessNeighborY[j].index;
    para->getParH(level)->recvProcessNeighborsAfterFtoCY[j].index = para->getParH(level)->recvProcessNeighborY[j].index;

    // rank neighbor
    para->getParH(level)->sendProcessNeighborsAfterFtoCY[j].rankNeighbor = para->getParH(level)->sendProcessNeighborY[j].rankNeighbor;
    para->getParH(level)->recvProcessNeighborsAfterFtoCY[j].rankNeighbor = para->getParH(level)->recvProcessNeighborY[j].rankNeighbor;
}

void IndexRearrangementForStreams::copyProcessNeighborToAfterFtoCZ(const uint &level, int j)
{
    // init f[0]*
    para->getParD(level)->sendProcessNeighborsAfterFtoCZ[j].f[0] = para->getParD(level)->sendProcessNeighborZ[j].f[0];
    para->getParH(level)->sendProcessNeighborsAfterFtoCZ[j].f[0] = para->getParH(level)->sendProcessNeighborZ[j].f[0];
    para->getParD(level)->recvProcessNeighborsAfterFtoCZ[j].f[0] = para->getParD(level)->recvProcessNeighborZ[j].f[0];
    para->getParH(level)->recvProcessNeighborsAfterFtoCZ[j].f[0] = para->getParH(level)->recvProcessNeighborZ[j].f[0];

    // init index*
    para->getParD(level)->sendProcessNeighborsAfterFtoCZ[j].index = para->getParD(level)->sendProcessNeighborZ[j].index;
    para->getParH(level)->sendProcessNeighborsAfterFtoCZ[j].index = para->getParH(level)->sendProcessNeighborZ[j].index;
    para->getParD(level)->recvProcessNeighborsAfterFtoCZ[j].index = para->getParD(level)->recvProcessNeighborZ[j].index;
    para->getParH(level)->recvProcessNeighborsAfterFtoCZ[j].index = para->getParH(level)->recvProcessNeighborZ[j].index;

    // rank neighbor
    para->getParH(level)->sendProcessNeighborsAfterFtoCZ[j].rankNeighbor = para->getParH(level)->sendProcessNeighborZ[j].rankNeighbor;
    para->getParH(level)->recvProcessNeighborsAfterFtoCZ[j].rankNeighbor = para->getParH(level)->recvProcessNeighborZ[j].rankNeighbor;
}

void IndexRearrangementForStreams::reorderSendIndicesForCommAfterFtoCX(int direction, int level, int j,
                                                        std::vector<uint> &sendIndicesForCommAfterFtoCPositions)
{
    int *sendIndices                    = para->getParH(level)->sendProcessNeighborX[j].index;
    int &numberOfSendNeighborsAfterFtoC = para->getParH(level)->sendProcessNeighborsAfterFtoCX[j].numberOfNodes;
    reorderSendIndicesForCommAfterFtoC(sendIndices, numberOfSendNeighborsAfterFtoC, direction, level, j,
                                       sendIndicesForCommAfterFtoCPositions);
}

void IndexRearrangementForStreams::reorderSendIndicesForCommAfterFtoCY(int direction, int level, int j,
                                                        std::vector<uint> &sendIndicesForCommAfterFtoCPositions)
{
    int *sendIndices                    = para->getParH(level)->sendProcessNeighborY[j].index;
    int &numberOfSendNeighborsAfterFtoC = para->getParH(level)->sendProcessNeighborsAfterFtoCY[j].numberOfNodes;
    reorderSendIndicesForCommAfterFtoC(sendIndices, numberOfSendNeighborsAfterFtoC, direction, level, j,
                                       sendIndicesForCommAfterFtoCPositions);
}

void IndexRearrangementForStreams::reorderSendIndicesForCommAfterFtoCZ(int direction, int level, int j,
                                                        std::vector<uint> &sendIndicesForCommAfterFtoCPositions) 
{
    int *sendIndices                    = para->getParH(level)->sendProcessNeighborZ[j].index;
    int &numberOfSendNeighborsAfterFtoC = para->getParH(level)->sendProcessNeighborsAfterFtoCZ[j].numberOfNodes;
    reorderSendIndicesForCommAfterFtoC(sendIndices, numberOfSendNeighborsAfterFtoC, direction, level, j,
                                       sendIndicesForCommAfterFtoCPositions);
}

void IndexRearrangementForStreams::reorderSendIndicesForCommAfterFtoC(int *sendIndices, int &numberOfSendNeighborsAfterFtoC,
                                                       int direction, int level, int j,
                                                       std::vector<uint> &sendIndicesForCommAfterFtoCPositions)
{
    *logging::out << logging::Logger::INFO_INTERMEDIATE
                  << "reorder send indices for communication after fine to coarse: level: " << level
                  << " direction: " << direction;
    if (para->getParH(level)->intCF.kCF == 0 || para->getParH(level)->intFC.kFC == 0)
        *logging::out << logging::Logger::LOGGER_ERROR
                      << "reorderSendIndicesForCommAfterFtoC(): iCellFCC needs to be inititalized before calling "
                         "this function "
                      << "\n";

    int sparseIndexSend;
    std::vector<int> sendIndicesAfterFtoC;
    std::vector<int> sendIndicesOther;
    std::array<int, 7> neighbors;
    uint numberOfSendIndices = builder->getNumberOfSendIndices(direction, level);

    //iCellFCC
    for (uint posInSendIndices = 0; posInSendIndices < numberOfSendIndices; posInSendIndices++) {
        neighbors.fill(-1);
        sparseIndexSend = sendIndices[posInSendIndices];  
        if (isSparseIndexInICellFCC(para->getParH(level)->intFC.kFC, sparseIndexSend, level))
            addUniqueIndexToCommunicationVectors(sendIndicesAfterFtoC, sparseIndexSend,
                                                 sendIndicesForCommAfterFtoCPositions, posInSendIndices);
    }

    // iCellCFC
    std::vector<uint> nodesCFC;
    aggregateNodesInICellCFC(level, nodesCFC);
    for (auto sparseIndex : nodesCFC)
        findIfSparseIndexIsInSendIndicesAndAddToCommVectors(sparseIndex, sendIndices, numberOfSendIndices,
                                                            sendIndicesAfterFtoC, sendIndicesForCommAfterFtoCPositions);

    numberOfSendNeighborsAfterFtoC = (int)sendIndicesAfterFtoC.size();

    findIndicesNotInCommAfterFtoC(numberOfSendIndices, sendIndices, sendIndicesAfterFtoC, sendIndicesOther);

    // copy new vectors back to sendIndices array
    for (int i = 0; i < numberOfSendNeighborsAfterFtoC; i++)
        sendIndices[i] = sendIndicesAfterFtoC[i];
    for (uint i = 0; i < (uint)sendIndicesOther.size(); i++)
        sendIndices[i + numberOfSendNeighborsAfterFtoC] = sendIndicesOther[i];

    *logging::out << logging::Logger::INFO_INTERMEDIATE
                  << "... numberOfSendNeighborsAfterFtoC: " << numberOfSendNeighborsAfterFtoC << "\n";

    if (numberOfSendNeighborsAfterFtoC + sendIndicesOther.size() != numberOfSendIndices) {
        *logging::out << logging::Logger::LOGGER_ERROR
                      << "reorderSendIndicesForCommAfterFtoC(): incorrect number of nodes"
                      << "\n";
        std::cout << "numberOfSendNeighborsAfterFtoC = " << numberOfSendNeighborsAfterFtoC
                  << ", sendOrIndicesOther.size() = " << sendIndicesOther.size()
                  << ", numberOfSendOrRecvIndices = " << numberOfSendIndices << std::endl;
    }
}

bool IndexRearrangementForStreams::isSparseIndexInICellFCC(uint sizeOfICellFCC, int sparseIndex, int level)
{
    for (uint j = 0; j < sizeOfICellFCC; j++) {
        if (sparseIndex < 0)
            return false;
        if (para->getParH(level)->intFC.ICellFCC[j] == (uint)sparseIndex) {
            return true;
        }
    }
    return false;
}

void IndexRearrangementForStreams::aggregateNodesInICellCFC(int level, std::vector<uint> &nodesCFC)
{
    uint sparseIndex;
    uint *neighborX = para->getParH(level)->neighborX_SP;
    uint *neighborY = para->getParH(level)->neighborY_SP;
    uint *neighborZ = para->getParH(level)->neighborZ_SP;

    for (uint x = 0; x < para->getParH(level)->intCF.kCF; x++) {
        sparseIndex = para->getParH(level)->intCF.ICellCFC[x];
        nodesCFC.push_back(sparseIndex);
        nodesCFC.push_back(neighborX[sparseIndex]);
        nodesCFC.push_back(neighborY[sparseIndex]);
        nodesCFC.push_back(neighborZ[sparseIndex]);
        nodesCFC.push_back(neighborY[neighborX[sparseIndex]]);
        nodesCFC.push_back(neighborZ[neighborX[sparseIndex]]);
        nodesCFC.push_back(neighborZ[neighborY[sparseIndex]]);
        nodesCFC.push_back(neighborZ[neighborY[neighborX[sparseIndex]]]);           
    }
    std::sort(nodesCFC.begin(), nodesCFC.end());
    auto iterator = std::unique(nodesCFC.begin(), nodesCFC.end());
    nodesCFC.erase(iterator, nodesCFC.end());
}

void IndexRearrangementForStreams::addUniqueIndexToCommunicationVectors(
    std::vector<int> &sendIndicesAfterFtoC, int &sparseIndexSend,
    std::vector<unsigned int> &sendIndicesForCommAfterFtoCPositions, uint &posInSendIndices) const
{
    // add index to corresponding vectors but omit indices which are already in sendIndicesAfterFtoC
    if (std::find(sendIndicesAfterFtoC.begin(), sendIndicesAfterFtoC.end(), sparseIndexSend) == sendIndicesAfterFtoC.end()) {
        sendIndicesAfterFtoC.push_back(sparseIndexSend);
        sendIndicesForCommAfterFtoCPositions.push_back(posInSendIndices);
    }
}

void IndexRearrangementForStreams::findIfSparseIndexIsInSendIndicesAndAddToCommVectors(
    int sparseIndex, int *sendIndices, uint numberOfSendIndices, std::vector<int> &sendIndicesAfterFtoC,
    std::vector<uint> &sendIndicesForCommAfterFtoCPositions) const
{
    int sparseIndexSend;
    for (uint posInSendIndices = 0; posInSendIndices < numberOfSendIndices; posInSendIndices++) {
        sparseIndexSend = sendIndices[posInSendIndices];
        if (sparseIndex == sparseIndexSend) {
            addUniqueIndexToCommunicationVectors(sendIndicesAfterFtoC, sparseIndex,
                                                 sendIndicesForCommAfterFtoCPositions, posInSendIndices);
            break;
        }
    }
}

void IndexRearrangementForStreams::findIndicesNotInCommAfterFtoC(const uint &numberOfSendOrRecvIndices, 
                                                      int *sendOrReceiveIndices, std::vector<int> &sendOrReceiveIndicesAfterFtoC,
                                                      std::vector<int> &sendOrIndicesOther)
{
    int sparseIndexSend;
    for (uint posInSendIndices = 0; posInSendIndices < numberOfSendOrRecvIndices; posInSendIndices++) {
        sparseIndexSend = sendOrReceiveIndices[posInSendIndices];
        if (std::find(sendOrReceiveIndicesAfterFtoC.begin(), sendOrReceiveIndicesAfterFtoC.end(), sparseIndexSend) ==
            sendOrReceiveIndicesAfterFtoC.end())
            sendOrIndicesOther.push_back(sparseIndexSend);
    }
}

void IndexRearrangementForStreams::reorderRecvIndicesForCommAfterFtoCX(int direction, int level, int j,
                                                        std::vector<uint> &sendIndicesForCommAfterFtoCPositions)
{
    int *recvIndices                    = para->getParH(level)->recvProcessNeighborX[j].index;
    int &numberOfRecvNeighborsAfterFtoC = para->getParH(level)->recvProcessNeighborsAfterFtoCX[j].numberOfNodes;
    reorderRecvIndicesForCommAfterFtoC(recvIndices, numberOfRecvNeighborsAfterFtoC, direction, level, j,
                                       sendIndicesForCommAfterFtoCPositions);
}

void IndexRearrangementForStreams::reorderRecvIndicesForCommAfterFtoCY(int direction, int level, int j,
                                                       std::vector<uint> &sendIndicesForCommAfterFtoCPositions)
{
    int *recvIndices                    = para->getParH(level)->recvProcessNeighborY[j].index;
    int &numberOfRecvNeighborsAfterFtoC = para->getParH(level)->recvProcessNeighborsAfterFtoCY[j].numberOfNodes;
    reorderRecvIndicesForCommAfterFtoC(recvIndices, numberOfRecvNeighborsAfterFtoC, direction, level, j,
                                       sendIndicesForCommAfterFtoCPositions);
}

void IndexRearrangementForStreams::reorderRecvIndicesForCommAfterFtoCZ(int direction, int level, int j,
                                                        std::vector<uint> &sendIndicesForCommAfterFtoCPositions)
{
    int *recvIndices                    = para->getParH(level)->recvProcessNeighborZ[j].index;
    int &numberOfRecvNeighborsAfterFtoC = para->getParH(level)->recvProcessNeighborsAfterFtoCZ[j].numberOfNodes;
    reorderRecvIndicesForCommAfterFtoC(recvIndices, numberOfRecvNeighborsAfterFtoC, direction, level, j,
                                       sendIndicesForCommAfterFtoCPositions);
}

void IndexRearrangementForStreams::reorderRecvIndicesForCommAfterFtoC(int *recvIndices,
                                                       int &numberOfRecvNeighborsAfterFtoC, int direction, int level,
                                                       int j,
                                                       std::vector<uint> &sendIndicesForCommAfterFtoCPositions)
{
    *logging::out << logging::Logger::INFO_INTERMEDIATE
                  << "reorder receive indices for communication after fine to coarse: level: " << level
                  << " direction: " << direction;
    if (sendIndicesForCommAfterFtoCPositions.size() == 0)
        *logging::out << logging::Logger::LOGGER_ERROR
                      << "reorderRecvIndicesForCommAfterFtoC(): sendIndicesForCommAfterFtoCPositions is empty."
                      << "\n";

    uint numberOfRecvIndices = builder->getNumberOfReceiveIndices(direction, level);
    std::vector<int> recvIndicesAfterFtoC;
    std::vector<int> recvIndicesOther;

    // find recvIndices for Communication after fine to coarse
    for (uint vectorPos : sendIndicesForCommAfterFtoCPositions)
        recvIndicesAfterFtoC.push_back(recvIndices[vectorPos]);

    findIndicesNotInCommAfterFtoC(numberOfRecvIndices, recvIndices, recvIndicesAfterFtoC, recvIndicesOther);

    numberOfRecvNeighborsAfterFtoC = (int)recvIndicesAfterFtoC.size();

    // copy new vectors back to sendIndices array
    for (int i = 0; i < numberOfRecvNeighborsAfterFtoC; i++)
        recvIndices[i] = recvIndicesAfterFtoC[i];
    for (uint i = 0; i < (uint)recvIndicesOther.size(); i++)
        recvIndices[i + numberOfRecvNeighborsAfterFtoC] = recvIndicesOther[i];

    *logging::out << logging::Logger::INFO_INTERMEDIATE
                  << "... numberOfRecvNeighborsAfterFtoC: " << numberOfRecvNeighborsAfterFtoC << "\n";

    if (numberOfRecvNeighborsAfterFtoC + recvIndicesOther.size() != numberOfRecvIndices) {
        *logging::out << logging::Logger::LOGGER_ERROR
                      << "reorderRecvIndicesForCommAfterFtoC(): incorrect number of nodes"
                      << "\n";
        std::cout << "numberOfRecvNeighborsAfterFtoC = " << numberOfRecvNeighborsAfterFtoC
                  << ", recvIndicesOther.size() = " << recvIndicesOther.size()
                  << ", numberOfRecvIndices = " << numberOfRecvIndices << std::endl;
    }
}


void IndexRearrangementForStreams::splitCoarseToFineIntoBorderAndBulk(const uint &level)
{
    para->getParH(level)->intCFBorder.ICellCFC = para->getParH(level)->intCF.ICellCFC;
    para->getParH(level)->intCFBorder.ICellCFF = para->getParH(level)->intCF.ICellCFF;

    builder->getGridInterfaceIndicesBorderBulkCF(
        para->getParH(level)->intCFBorder.ICellCFC, para->getParH(level)->intCFBulk.ICellCFC,
        para->getParH(level)->intCFBorder.ICellCFF, para->getParH(level)->intCFBulk.ICellCFF,
        para->getParH(level)->intCFBorder.kCF, para->getParH(level)->intCFBulk.kCF,
        this->para->getParH(level)->neighborX_SP, this->para->getParH(level)->neighborY_SP,
        this->para->getParH(level)->neighborZ_SP, level);

    para->getParD(level)->intCFBorder.kCF      = para->getParH(level)->intCFBorder.kCF;
    para->getParD(level)->intCFBulk.kCF        = para->getParH(level)->intCFBulk.kCF;
    para->getParD(level)->intCFBorder.ICellCFC = para->getParD(level)->intCF.ICellCFC;
    para->getParD(level)->intCFBulk.ICellCFC =
        para->getParD(level)->intCFBorder.ICellCFC + para->getParD(level)->intCFBorder.kCF;
    para->getParD(level)->intCFBorder.ICellCFF = para->getParD(level)->intCF.ICellCFF;
    para->getParD(level)->intCFBulk.ICellCFF =
        para->getParD(level)->intCFBorder.ICellCFF + para->getParD(level)->intCFBorder.kCF;
}

void IndexRearrangementForStreams::splitFineToCoarseIntoBorderAndBulk(const uint &level)
{
    para->getParH(level)->intFCBorder.ICellFCC = para->getParH(level)->intFC.ICellFCC;
    para->getParH(level)->intFCBorder.ICellFCF = para->getParH(level)->intFC.ICellFCF;

    builder->getGridInterfaceIndicesBorderBulkFC(
        para->getParH(level)->intFCBorder.ICellFCC, para->getParH(level)->intFCBulk.ICellFCC,
        para->getParH(level)->intFCBorder.ICellFCF, para->getParH(level)->intFCBulk.ICellFCF,
        para->getParH(level)->intFCBorder.kFC, para->getParH(level)->intFCBulk.kFC, level);

    para->getParD(level)->intFCBorder.kFC      = para->getParH(level)->intFCBorder.kFC;
    para->getParD(level)->intFCBulk.kFC        = para->getParH(level)->intFCBulk.kFC;
    para->getParD(level)->intFCBorder.ICellFCC = para->getParD(level)->intFC.ICellFCC;
    para->getParD(level)->intFCBulk.ICellFCC =
        para->getParD(level)->intFCBorder.ICellFCC + para->getParD(level)->intFCBorder.kFC;
    para->getParD(level)->intFCBorder.ICellFCF = para->getParD(level)->intFC.ICellFCF;
    para->getParD(level)->intFCBulk.ICellFCF =
        para->getParD(level)->intFCBorder.ICellFCF + para->getParD(level)->intFCBorder.kFC;
}

