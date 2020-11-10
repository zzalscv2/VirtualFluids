#include "GridReader.h"

#include <iostream>

#include "Parameter/Parameter.h"

#include "CoordNeighborGeoV.h"
#include "BoundaryQs.h"
#include "BoundaryValues.h"

#include <GPU/CudaMemoryManager.h>
#include "OffsetScale.h"

GridReader::GridReader(FILEFORMAT format, std::shared_ptr<Parameter> para, std::shared_ptr<CudaMemoryManager> cudaManager)
{
    this->para = para;
    this->cudaMemoryManager = cudaManager;

	if (format == FILEFORMAT::ASCII)
		this->binaer = false;
	else
		this->binaer = true;

	channelDirections.resize(6);
	channelBoundaryConditions.resize(6);
	BC_Values.resize(6);

	channelDirections[0] = "inlet";
	channelDirections[1] = "outlet";
	channelDirections[2] = "front";
	channelDirections[3] = "back";
	channelDirections[4] = "top";
	channelDirections[5] = "bottom";
}

GridReader::~GridReader()
{

}

bool GridReader::getBinaer()
{
	return binaer;
}

void rearrangeGeometry(Parameter* para, int lev)
{
    for (uint index = 0; index < para->getParH(lev)->size_Mat_SP; index++)
    {
        if (para->getParH(lev)->geoSP[index] == GEO_FLUID_OLD)
        {
            para->getParH(lev)->geoSP[index] = GEO_FLUID;
        }
    }
}

void GridReader::allocArrays_CoordNeighborGeo()
{
	std::cout << "-----Config Arrays Coord, Neighbor, Geo------" << std::endl;

	CoordNeighborGeoV coordX(para->getcoordX(), binaer, true);
	CoordNeighborGeoV coordY(para->getcoordY(), binaer, true);
	CoordNeighborGeoV coordZ(para->getcoordZ(), binaer, true);
	neighX = std::shared_ptr<CoordNeighborGeoV>(new CoordNeighborGeoV(para->getneighborX(), binaer, false));
	neighY = std::shared_ptr<CoordNeighborGeoV>(new CoordNeighborGeoV(para->getneighborY(), binaer, false));
	neighZ = std::shared_ptr<CoordNeighborGeoV>(new CoordNeighborGeoV(para->getneighborZ(), binaer, false));
	CoordNeighborGeoV geoV(para->getgeoVec(), binaer, false);

	uint maxLevel = coordX.getLevel();
	std::cout << "Number of Level: " << maxLevel + 1 << std::endl;
	uint numberOfNodesGlobal = 0;
	std::cout << "Number of Nodes: " << std::endl;

	for (uint level = 0; level <= maxLevel; level++) 
	{		
		int numberOfNodesPerLevel = coordX.getSize(level) + 1;
		numberOfNodesGlobal += numberOfNodesPerLevel;
		std::cout << "Level " << level << " = " << numberOfNodesPerLevel << " Nodes" << std::endl;

		setNumberOfNodes(numberOfNodesPerLevel, level);

        cudaMemoryManager->cudaAllocCoord(level);
		cudaMemoryManager->cudaAllocSP(level);
        cudaMemoryManager->cudaAllocF3SP(level);
        cudaMemoryManager->cudaAllocNeighborWSB(level);

        if (para->getUseWale())
			cudaMemoryManager->cudaAllocTurbulentViscosity(level);

		coordX.initalCoords(para->getParH(level)->coordX_SP, level);
		coordY.initalCoords(para->getParH(level)->coordY_SP, level);
		coordZ.initalCoords(para->getParH(level)->coordZ_SP, level);
		neighX->initalNeighbors(para->getParH(level)->neighborX_SP, level);
		neighY->initalNeighbors(para->getParH(level)->neighborY_SP, level);
		neighZ->initalNeighbors(para->getParH(level)->neighborZ_SP, level);
		geoV.initalNeighbors(para->getParH(level)->geoSP, level);
        rearrangeGeometry(para.get(), level);
		setInitalNodeValues(numberOfNodesPerLevel, level);

        cudaMemoryManager->cudaCopyNeighborWSB(level);
        cudaMemoryManager->cudaCopySP(level);
        cudaMemoryManager->cudaCopyCoord(level);
	}
	std::cout << "Number of Nodes: " << numberOfNodesGlobal << std::endl;
	std::cout << "-----finish Coord, Neighbor, Geo------" <<std::endl;
}

void GridReader::allocArrays_BoundaryValues()
{
	std::cout << "------read BoundaryValues------" <<std::endl;

	this->makeReader(para);
	this->setChannelBoundaryCondition();
	int level = BC_Values[0]->getLevel();

    for (uint i = 0; i < channelBoundaryConditions.size(); i++)
    {
        setVelocityValues(i);
        setPressureValues(i);
        setOutflowValues(i);
    }

	initalValuesDomainDecompostion(level);
}

void GridReader::allocArrays_OffsetScale()
{
    std::cout << "-----Config Arrays OffsetScale------" << std::endl;
    OffsetScale *obj_offCF = new OffsetScale(para->getscaleOffsetCF(), true);
    OffsetScale *obj_offFC = new OffsetScale(para->getscaleOffsetFC(), true);
    OffsetScale *obj_scaleCFC = new OffsetScale(para->getscaleCFC(), false);
    OffsetScale *obj_scaleCFF = new OffsetScale(para->getscaleCFF(), false);
    OffsetScale *obj_scaleFCC = new OffsetScale(para->getscaleFCC(), false);
    OffsetScale *obj_scaleFCF = new OffsetScale(para->getscaleFCF(), false);

    int level = obj_offCF->getLevel();

    int AnzahlKnotenGesCF = 0;
    int AnzahlKnotenGesFC = 0;

    for (int i = 0; i<level; i++) {
        unsigned int tempCF = obj_offCF->getSize(i);
        std::cout << "Groesse der Daten CF vom Level " << i << " : " << tempCF << std::endl;
        unsigned int tempFC = obj_offFC->getSize(i);
        std::cout << "Groesse der Daten FC vom Level " << i << " : " << tempFC << std::endl;

        AnzahlKnotenGesCF += tempCF;
        AnzahlKnotenGesFC += tempFC;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //size + memsize CF
        para->getParH(i)->K_CF = tempCF;
        para->getParD(i)->K_CF = para->getParH(i)->K_CF;
        para->getParH(i)->intCF.kCF = para->getParH(i)->K_CF;
        para->getParD(i)->intCF.kCF = para->getParH(i)->K_CF;
        para->getParH(i)->mem_size_kCF = sizeof(unsigned int)* para->getParH(i)->K_CF;
        para->getParD(i)->mem_size_kCF = sizeof(unsigned int)* para->getParD(i)->K_CF;
        para->getParH(i)->mem_size_kCF_off = sizeof(real)* para->getParH(i)->K_CF;
        para->getParD(i)->mem_size_kCF_off = sizeof(real)* para->getParD(i)->K_CF;
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //size + memsize FC
        para->getParH(i)->K_FC = tempFC;
        para->getParD(i)->K_FC = para->getParH(i)->K_FC;
        para->getParH(i)->intFC.kFC = para->getParH(i)->K_FC;
        para->getParD(i)->intFC.kFC = para->getParH(i)->K_FC;
        para->getParH(i)->mem_size_kFC = sizeof(unsigned int)* para->getParH(i)->K_FC;
        para->getParD(i)->mem_size_kFC = sizeof(unsigned int)* para->getParD(i)->K_FC;
        para->getParH(i)->mem_size_kFC_off = sizeof(real)* para->getParH(i)->K_FC;
        para->getParD(i)->mem_size_kFC_off = sizeof(real)* para->getParD(i)->K_FC;
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //alloc
		cudaMemoryManager->cudaAllocInterfaceCF(i);
		cudaMemoryManager->cudaAllocInterfaceFC(i);
		cudaMemoryManager->cudaAllocInterfaceOffCF(i);
		cudaMemoryManager->cudaAllocInterfaceOffFC(i);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //init
        obj_offCF->initArrayOffset(para->getParH(i)->offCF.xOffCF, para->getParH(i)->offCF.yOffCF, para->getParH(i)->offCF.zOffCF, i);
        obj_offFC->initArrayOffset(para->getParH(i)->offFC.xOffFC, para->getParH(i)->offFC.yOffFC, para->getParH(i)->offFC.zOffFC, i);
        obj_scaleCFC->initScale(para->getParH(i)->intCF.ICellCFC, i);
        obj_scaleCFF->initScale(para->getParH(i)->intCF.ICellCFF, i);
        obj_scaleFCC->initScale(para->getParH(i)->intFC.ICellFCC, i);
        obj_scaleFCF->initScale(para->getParH(i)->intFC.ICellFCF, i);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //copy
		cudaMemoryManager->cudaCopyInterfaceCF(i);
		cudaMemoryManager->cudaCopyInterfaceFC(i);
		cudaMemoryManager->cudaCopyInterfaceOffCF(i);
		cudaMemoryManager->cudaCopyInterfaceOffFC(i);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
    std::cout << "Gesamtanzahl Knoten CF = " << AnzahlKnotenGesCF << std::endl;
    std::cout << "Gesamtanzahl Knoten FC = " << AnzahlKnotenGesFC << std::endl;

    delete obj_offCF;
    delete obj_offFC;
    delete obj_scaleCFC;
    delete obj_scaleCFF;
    delete obj_scaleFCC;
    delete obj_scaleFCF;
    std::cout << "-----Ende OffsetScale------" << std::endl;
}


void GridReader::setPressureValues(int channelSide) const
{
	for (unsigned int level = 0; level <= BC_Values[channelSide]->getLevel(); level++)
	{
		int sizePerLevel = BC_Values[channelSide]->getSize(level);
        setPressSizePerLevel(level, sizePerLevel);

		if (sizePerLevel > 0)
		{
			std::cout << "size pressure level " << level << " : " << sizePerLevel << std::endl;

            cudaMemoryManager->cudaAllocPress(level);

			setPressRhoBC(sizePerLevel, level, channelSide);
            cudaMemoryManager->cudaCopyPress(level);
		}
	}
}

void GridReader::setPressRhoBC(int sizePerLevel, int level, int channelSide) const
{
	BC_Values[channelSide]->setPressValues(para->getParH(level)->QPress.RhoBC, para->getParH(level)->QPress.kN, level);
	for (int m = 0; m < sizePerLevel; m++)
		para->getParH(level)->QPress.RhoBC[m] = (para->getParH(level)->QPress.RhoBC[m] / para->getFactorPressBC());
}


void GridReader::setVelocityValues(int channelSide) const
{
	for (unsigned int level = 0; level <= BC_Values[channelSide]->getLevel(); level++)
	{
		int sizePerLevel = BC_Values[channelSide]->getSize(level);
        setVelocitySizePerLevel(level, sizePerLevel);

		if (sizePerLevel > 1)
		{
			std::cout << "size velocity level " << level << " : " << sizePerLevel << std::endl;

            cudaMemoryManager->cudaAllocVeloBC(level);

			setVelocity(level, sizePerLevel, channelSide);
            cudaMemoryManager->cudaCopyVeloBC(level);
		}
	}
}

void GridReader::setVelocity(int level, int sizePerLevel, int channelSide) const
{
	BC_Values[channelSide]->setVelocityValues(para->getParH(level)->Qinflow.Vx, para->getParH(level)->Qinflow.Vy, para->getParH(level)->Qinflow.Vz, level);

	for (int index = 0; index < sizePerLevel; index++)
	{
		para->getParH(level)->Qinflow.Vx[index] = para->getParH(level)->Qinflow.Vx[index] / para->getVelocityRatio();
		para->getParH(level)->Qinflow.Vy[index] = para->getParH(level)->Qinflow.Vy[index] / para->getVelocityRatio();
		para->getParH(level)->Qinflow.Vz[index] = para->getParH(level)->Qinflow.Vz[index] / para->getVelocityRatio();
		//para->getParH(level)->Qinflow.Vx[index] = para->getVelocity();//0.035;
		//para->getParH(level)->Qinflow.Vy[index] = 0.0;//para->getVelocity();//0.0;
		//para->getParH(level)->Qinflow.Vz[index] = 0.0;
	}
}


void GridReader::setOutflowValues(int channelSide) const
{
	for (unsigned int level = 0; level <= BC_Values[channelSide]->getLevel(); level++)
	{
		int sizePerLevel = BC_Values[channelSide]->getSize(level);
        setOutflowSizePerLevel(level, sizePerLevel);

		if (sizePerLevel > 1)
		{
			std::cout << "size outflow level " << level << " : " << sizePerLevel << std::endl;

            cudaMemoryManager->cudaAllocOutflowBC(level);

			setOutflow(level, sizePerLevel, channelSide);
            cudaMemoryManager->cudaCopyOutflowBC(level);

		}
	}
}

void GridReader::setOutflow(int level, int sizePerLevel, int channelSide) const
{
	BC_Values[channelSide]->setOutflowValues(para->getParH(level)->Qoutflow.RhoBC, para->getParH(level)->Qoutflow.kN, level);
	for (int index = 0; index < sizePerLevel; index++)
		para->getParH(level)->Qoutflow.RhoBC[index] = (para->getParH(level)->Qoutflow.RhoBC[index] / para->getFactorPressBC()) * (real)0.0;
}


void GridReader::initalValuesDomainDecompostion(int level)
{
	////////////////////////////////////////////////////////////////////////
	//3D domain decomposition
	std::vector< std::shared_ptr<BoundaryValues> > procNeighborsSendX, procNeighborsSendY, procNeighborsSendZ;
	std::vector< std::shared_ptr<BoundaryValues> > procNeighborsRecvX, procNeighborsRecvY, procNeighborsRecvZ;
	std::vector< int >             neighborRankX, neighborRankY, neighborRankZ;

	if (para->getNumprocs() > 1)
	{
		for (int process = 0; process < para->getNumprocs(); process++)
		{
			std::shared_ptr<BoundaryValues> pnXsend = std::shared_ptr<BoundaryValues> (new BoundaryValues(process, para, "send", "X"));
			std::shared_ptr<BoundaryValues> pnYsend = std::shared_ptr<BoundaryValues> (new BoundaryValues(process, para, "send", "Y"));
			std::shared_ptr<BoundaryValues> pnZsend = std::shared_ptr<BoundaryValues> (new BoundaryValues(process, para, "send", "Z"));
			std::shared_ptr<BoundaryValues> pnXrecv = std::shared_ptr<BoundaryValues> (new BoundaryValues(process, para, "recv", "X"));
			std::shared_ptr<BoundaryValues> pnYrecv = std::shared_ptr<BoundaryValues> (new BoundaryValues(process, para, "recv", "Y"));
			std::shared_ptr<BoundaryValues> pnZrecv = std::shared_ptr<BoundaryValues> (new BoundaryValues(process, para, "recv", "Z"));
			if (para->getIsNeighborX())
			{
				procNeighborsSendX.push_back(pnXsend);
				procNeighborsRecvX.push_back(pnXrecv);
				neighborRankX.push_back(process);
				std::cout << "MyID: " << para->getMyID() << ", neighborRankX: " << process << std::endl;
			}
			if (para->getIsNeighborY())
			{
				procNeighborsSendY.push_back(pnYsend);
				procNeighborsRecvY.push_back(pnYrecv);
				neighborRankY.push_back(process);
				std::cout << "MyID: " << para->getMyID() << ", neighborRankY: " << process << std::endl;
			}
			if (para->getIsNeighborZ())
			{
				procNeighborsSendZ.push_back(pnZsend);
				procNeighborsRecvZ.push_back(pnZrecv);
				neighborRankZ.push_back(process);
				std::cout << "MyID: " << para->getMyID() << ", neighborRankZ: " << process << std::endl;
			}
		}
		std::cout << "MyID: " << para->getMyID() << ", size of neighborRankX: " << neighborRankX.size() << ", size of neighborRankY: " << neighborRankY.size() << ", size of neighborRankZ: " << neighborRankZ.size() << std::endl;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//3D domain decomposition
	// X
	if ((para->getNumprocs() > 1) && (procNeighborsSendX.size() == procNeighborsRecvX.size()))
	{
		for (int j = 0; j < procNeighborsSendX.size(); j++)
		{
			for (int i = 0; i <= level; i++) {
				int tempSend = procNeighborsSendX[j]->getSize(i);
				int tempRecv = procNeighborsRecvX[j]->getSize(i);
				if (tempSend > 0)
				{
					////////////////////////////////////////////////////////////////////////////////////////
					//send
					std::cout << "size of Data for X send buffer, Level " << i << " : " << tempSend << std::endl;
					////////////////////////////////////////////////////////////////////////////////////////
					para->setNumberOfProcessNeighborsX((unsigned int)procNeighborsSendX.size(), i, "send");
					para->getParH(i)->sendProcessNeighborX[j].rankNeighbor = neighborRankX[j];
					////////////////////////////////////////////////////////////////////////////////////////
					para->getParH(i)->sendProcessNeighborX[j].numberOfNodes = tempSend;
					para->getParD(i)->sendProcessNeighborX[j].numberOfNodes = tempSend;
					para->getParH(i)->sendProcessNeighborX[j].numberOfFs = para->getD3Qxx() * tempSend;
					para->getParD(i)->sendProcessNeighborX[j].numberOfFs = para->getD3Qxx() * tempSend;
					para->getParH(i)->sendProcessNeighborX[j].memsizeIndex = sizeof(unsigned int)*tempSend;
					para->getParD(i)->sendProcessNeighborX[j].memsizeIndex = sizeof(unsigned int)*tempSend;
					para->getParH(i)->sendProcessNeighborX[j].memsizeFs = sizeof(real)     *tempSend;
					para->getParD(i)->sendProcessNeighborX[j].memsizeFs = sizeof(real)     *tempSend;
					////////////////////////////////////////////////////////////////////////////////////////
					//recv
					std::cout << "size of Data for X receive buffer, Level " << i << " : " << tempRecv << std::endl;
					////////////////////////////////////////////////////////////////////////////////////////
					para->setNumberOfProcessNeighborsX((unsigned int)procNeighborsRecvX.size(), i, "recv");
					para->getParH(i)->recvProcessNeighborX[j].rankNeighbor = neighborRankX[j];
					////////////////////////////////////////////////////////////////////////////////////////
					para->getParH(i)->recvProcessNeighborX[j].numberOfNodes = tempRecv;
					para->getParD(i)->recvProcessNeighborX[j].numberOfNodes = tempRecv;
					para->getParH(i)->recvProcessNeighborX[j].numberOfFs = para->getD3Qxx() * tempRecv;
					para->getParD(i)->recvProcessNeighborX[j].numberOfFs = para->getD3Qxx() * tempRecv;
					para->getParH(i)->recvProcessNeighborX[j].memsizeIndex = sizeof(unsigned int)*tempRecv;
					para->getParD(i)->recvProcessNeighborX[j].memsizeIndex = sizeof(unsigned int)*tempRecv;
					para->getParH(i)->recvProcessNeighborX[j].memsizeFs = sizeof(real)     *tempRecv;
					para->getParD(i)->recvProcessNeighborX[j].memsizeFs = sizeof(real)     *tempRecv;
					////////////////////////////////////////////////////////////////////////////////////////
					//malloc on host and device
                    cudaMemoryManager->cudaAllocProcessNeighborX(i, j);
					////////////////////////////////////////////////////////////////////////////////////////
					//init index arrays
					procNeighborsSendX[j]->initIndex(para->getParH(i)->sendProcessNeighborX[j].index, i);
					procNeighborsRecvX[j]->initIndex(para->getParH(i)->recvProcessNeighborX[j].index, i);
					////////////////////////////////////////////////////////////////////////////////////////
                    cudaMemoryManager->cudaCopyProcessNeighborXIndex(i, j);
					////////////////////////////////////////////////////////////////////////////////////////
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Y
	if ((para->getNumprocs() > 1) && (procNeighborsSendY.size() == procNeighborsRecvY.size()))
	{
		for (int j = 0; j < procNeighborsSendY.size(); j++)
		{
			for (int i = 0; i <= level; i++) {
				int tempSend = procNeighborsSendY[j]->getSize(i);
				int tempRecv = procNeighborsRecvY[j]->getSize(i);
				if (tempSend > 0)
				{
					////////////////////////////////////////////////////////////////////////////////////////
					//send
					std::cout << "size of Data for Y send buffer Level " << i << " : " << tempSend << std::endl;
					////////////////////////////////////////////////////////////////////////////////////////
					para->setNumberOfProcessNeighborsY((unsigned int)procNeighborsSendY.size(), i, "send");
					para->getParH(i)->sendProcessNeighborY[j].rankNeighbor = neighborRankY[j];
					////////////////////////////////////////////////////////////////////////////////////////
					para->getParH(i)->sendProcessNeighborY[j].numberOfNodes = tempSend;
					para->getParD(i)->sendProcessNeighborY[j].numberOfNodes = tempSend;
					para->getParH(i)->sendProcessNeighborY[j].numberOfFs = para->getD3Qxx() * tempSend;
					para->getParD(i)->sendProcessNeighborY[j].numberOfFs = para->getD3Qxx() * tempSend;
					para->getParH(i)->sendProcessNeighborY[j].memsizeIndex = sizeof(unsigned int)*tempSend;
					para->getParD(i)->sendProcessNeighborY[j].memsizeIndex = sizeof(unsigned int)*tempSend;
					para->getParH(i)->sendProcessNeighborY[j].memsizeFs = sizeof(real)     *tempSend;
					para->getParD(i)->sendProcessNeighborY[j].memsizeFs = sizeof(real)     *tempSend;
					////////////////////////////////////////////////////////////////////////////////////////
					//recv
					std::cout << "size of Data for Y receive buffer, Level " << i << " : " << tempRecv << std::endl;
					////////////////////////////////////////////////////////////////////////////////////////
					para->setNumberOfProcessNeighborsY((unsigned int)procNeighborsRecvY.size(), i, "recv");
					para->getParH(i)->recvProcessNeighborY[j].rankNeighbor = neighborRankY[j];
					////////////////////////////////////////////////////////////////////////////////////////
					para->getParH(i)->recvProcessNeighborY[j].numberOfNodes = tempRecv;
					para->getParD(i)->recvProcessNeighborY[j].numberOfNodes = tempRecv;
					para->getParH(i)->recvProcessNeighborY[j].numberOfFs = para->getD3Qxx() * tempRecv;
					para->getParD(i)->recvProcessNeighborY[j].numberOfFs = para->getD3Qxx() * tempRecv;
					para->getParH(i)->recvProcessNeighborY[j].memsizeIndex = sizeof(unsigned int)*tempRecv;
					para->getParD(i)->recvProcessNeighborY[j].memsizeIndex = sizeof(unsigned int)*tempRecv;
					para->getParH(i)->recvProcessNeighborY[j].memsizeFs = sizeof(real)     *tempRecv;
					para->getParD(i)->recvProcessNeighborY[j].memsizeFs = sizeof(real)     *tempRecv;
					////////////////////////////////////////////////////////////////////////////////////////
					//malloc on host and device
                    cudaMemoryManager->cudaAllocProcessNeighborY(i, j);
					////////////////////////////////////////////////////////////////////////////////////////
					//init index arrays
					procNeighborsSendY[j]->initIndex(para->getParH(i)->sendProcessNeighborY[j].index, i);
					procNeighborsRecvY[j]->initIndex(para->getParH(i)->recvProcessNeighborY[j].index, i);
					////////////////////////////////////////////////////////////////////////////////////////
                    cudaMemoryManager->cudaCopyProcessNeighborYIndex(i, j);
					////////////////////////////////////////////////////////////////////////////////////////
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Z
	if ((para->getNumprocs() > 1) && (procNeighborsSendZ.size() == procNeighborsRecvZ.size()))
	{
		for (int j = 0; j < procNeighborsSendZ.size(); j++)
		{
			for (int i = 0; i <= level; i++) {
				int tempSend = procNeighborsSendZ[j]->getSize(i);
				int tempRecv = procNeighborsRecvZ[j]->getSize(i);
				if (tempSend > 0)
				{
					////////////////////////////////////////////////////////////////////////////////////////
					//send
					std::cout << "size of Data for Z send buffer, Level " << i << " : " << tempSend << std::endl;
					////////////////////////////////////////////////////////////////////////////////////////
					para->setNumberOfProcessNeighborsZ((unsigned int)procNeighborsSendZ.size(), i, "send");
					para->getParH(i)->sendProcessNeighborZ[j].rankNeighbor = neighborRankZ[j];
					////////////////////////////////////////////////////////////////////////////////////////
					para->getParH(i)->sendProcessNeighborZ[j].numberOfNodes = tempSend;
					para->getParD(i)->sendProcessNeighborZ[j].numberOfNodes = tempSend;
					para->getParH(i)->sendProcessNeighborZ[j].numberOfFs = para->getD3Qxx() * tempSend;
					para->getParD(i)->sendProcessNeighborZ[j].numberOfFs = para->getD3Qxx() * tempSend;
					para->getParH(i)->sendProcessNeighborZ[j].memsizeIndex = sizeof(unsigned int)*tempSend;
					para->getParD(i)->sendProcessNeighborZ[j].memsizeIndex = sizeof(unsigned int)*tempSend;
					para->getParH(i)->sendProcessNeighborZ[j].memsizeFs = sizeof(real)     *tempSend;
					para->getParD(i)->sendProcessNeighborZ[j].memsizeFs = sizeof(real)     *tempSend;
					////////////////////////////////////////////////////////////////////////////////////////
					//recv
					std::cout << "size of Data for Z receive buffer, Level " << i << " : " << tempRecv << std::endl;
					////////////////////////////////////////////////////////////////////////////////////////
					para->setNumberOfProcessNeighborsZ((unsigned int)procNeighborsRecvZ.size(), i, "recv");
					para->getParH(i)->recvProcessNeighborZ[j].rankNeighbor = neighborRankZ[j];
					////////////////////////////////////////////////////////////////////////////////////////
					para->getParH(i)->recvProcessNeighborZ[j].numberOfNodes = tempRecv;
					para->getParD(i)->recvProcessNeighborZ[j].numberOfNodes = tempRecv;
					para->getParH(i)->recvProcessNeighborZ[j].numberOfFs = para->getD3Qxx() * tempRecv;
					para->getParD(i)->recvProcessNeighborZ[j].numberOfFs = para->getD3Qxx() * tempRecv;
					para->getParH(i)->recvProcessNeighborZ[j].memsizeIndex = sizeof(unsigned int)*tempRecv;
					para->getParD(i)->recvProcessNeighborZ[j].memsizeIndex = sizeof(unsigned int)*tempRecv;
					para->getParH(i)->recvProcessNeighborZ[j].memsizeFs = sizeof(real)     *tempRecv;
					para->getParD(i)->recvProcessNeighborZ[j].memsizeFs = sizeof(real)     *tempRecv;
					////////////////////////////////////////////////////////////////////////////////////////
					//malloc on host and device
                    cudaMemoryManager->cudaAllocProcessNeighborZ(i, j);
					////////////////////////////////////////////////////////////////////////////////////////
					//init index arrays
					procNeighborsSendZ[j]->initIndex(para->getParH(i)->sendProcessNeighborZ[j].index, i);
					procNeighborsRecvZ[j]->initIndex(para->getParH(i)->recvProcessNeighborZ[j].index, i);
					////////////////////////////////////////////////////////////////////////////////////////
                    cudaMemoryManager->cudaCopyProcessNeighborZIndex(i, j);
					////////////////////////////////////////////////////////////////////////////////////////
				}
			}
		}
	}
}

void GridReader::allocArrays_BoundaryQs()
{
	std::cout << "------read BoundaryQs-------" <<std::endl;

	std::vector<std::shared_ptr<BoundaryQs> > BC_Qs(channelDirections.size());
	this->makeReader(BC_Qs, para);

	for (int i = 0; i < channelBoundaryConditions.size(); i++)
	{
		if (this->channelBoundaryConditions[i] == "noSlip") { setNoSlipQs(BC_Qs[i]); }
		else if (this->channelBoundaryConditions[i] == "velocity") { setVelocityQs(BC_Qs[i]); }
		else if (this->channelBoundaryConditions[i] == "pressure") { setPressQs(BC_Qs[i]); }
		else if (this->channelBoundaryConditions[i] == "outflow") { setOutflowQs(BC_Qs[i]); }
	}

	std::shared_ptr<BoundaryQs> obj_geomQ = std::shared_ptr<BoundaryQs>(new BoundaryQs(para->getgeomBoundaryBcQs(), para, "geo", false));
	if (para->getIsGeo())
		setGeoQs(obj_geomQ);

	std::cout << "-----finish BoundaryQs------" <<std::endl;
}


/*------------------------------------------------------------------------------------------------*/
/*----------------------------------------q setter methods----------------------------------------*/
/*------------------------------------------------------------------------------------------------*/
void GridReader::setPressQs(std::shared_ptr<BoundaryQs> boundaryQ) const
{
	for (unsigned int level = 0; level <= boundaryQ->getLevel(); level++)
	{
		if (hasQs(boundaryQ, level))
		{
			this->printQSize("pressure", boundaryQ, level);
			this->initalQStruct(para->getParH(level)->QPress, boundaryQ, level);
            cudaMemoryManager->cudaCopyPress(level);
		}
	}
}

void GridReader::setVelocityQs(std::shared_ptr<BoundaryQs> boundaryQ) const
{
	for (unsigned int level = 0; level <= boundaryQ->getLevel(); level++)
	{
		if (hasQs(boundaryQ, level))
		{
			this->printQSize("velocity", boundaryQ, level);
			this->initalQStruct(para->getParH(level)->Qinflow, boundaryQ, level);
            cudaMemoryManager->cudaCopyVeloBC(level);
		}
	}
}

void GridReader::setOutflowQs(std::shared_ptr<BoundaryQs> boundaryQ) const
{
	for (unsigned int level = 0; level <= boundaryQ->getLevel(); level++)
	{
		if (hasQs(boundaryQ, level))
		{
			this->printQSize("outflow", boundaryQ, level);
			this->initalQStruct(para->getParH(level)->Qoutflow, boundaryQ, level);
            cudaMemoryManager->cudaCopyOutflowBC(level);
		}
	}
}

void GridReader::setNoSlipQs(std::shared_ptr<BoundaryQs> boundaryQ) const
{
	for (unsigned int level = 0; level <= boundaryQ->getLevel(); level++)
	{
		if (hasQs(boundaryQ, level))
		{
			this->printQSize("no slip", boundaryQ, level);
			this->setSizeNoSlip(boundaryQ, level);
			this->initalQStruct(para->getParH(level)->QWall, boundaryQ, level);
            cudaMemoryManager->cudaCopyWallBC(level);
		}
	}
}

void GridReader::setGeoQs(std::shared_ptr<BoundaryQs> boundaryQ) const
{
	for (unsigned int level = 0; level <= boundaryQ->getLevel(); level++)
	{
		if (hasQs(boundaryQ, level))
		{
			this->printQSize("geo Qs", boundaryQ, level);
			this->setSizeGeoQs(boundaryQ, level);
			this->initalQStruct(para->getParH(level)->QGeom, boundaryQ, level);

			modifyQElement(boundaryQ, level);

            cudaMemoryManager->cudaCopyGeomBC(level);
		}
	}
}

void GridReader::modifyQElement(std::shared_ptr<BoundaryQs> boundaryQ, unsigned int level) const
{
	QforBoundaryConditions Q;
	real* QQ = para->getParH(level)->QGeom.q27[0];
	Q.q27[dirZERO] = &QQ[dirZERO * para->getParH(level)->QGeom.kQ];
	for (unsigned int i = 0; i < boundaryQ->getSize(level); i++)
		Q.q27[dirZERO][i] = 0.0f;
}

/*------------------------------------------------------------------------------------------------*/
/*---------------------------------------private q methods----------------------------------------*/
/*------------------------------------------------------------------------------------------------*/
void GridReader::initalQStruct(QforBoundaryConditions& Q, std::shared_ptr<BoundaryQs> boundaryQ, unsigned int level) const
{
	QforBoundaryConditions qTemp;
	this->setQ27Size(qTemp, Q.q27[0], Q.kQ);
	boundaryQ->setValues(qTemp.q27, level);
	boundaryQ->setIndex(Q.k, level);
}

bool GridReader::hasQs(std::shared_ptr<BoundaryQs> boundaryQ, unsigned int level) const
{
	return boundaryQ->getSize(level) > 0;
}

void GridReader::initalGridInformations()
{

}

void GridReader::setQ27Size(QforBoundaryConditions &Q, real* QQ, unsigned int sizeQ) const
{
	Q.q27[dirE] = &QQ[dirE   *sizeQ];
	Q.q27[dirW] = &QQ[dirW   *sizeQ];
	Q.q27[dirN] = &QQ[dirN   *sizeQ];
	Q.q27[dirS] = &QQ[dirS   *sizeQ];
	Q.q27[dirT] = &QQ[dirT   *sizeQ];
	Q.q27[dirB] = &QQ[dirB   *sizeQ];
	Q.q27[dirNE] = &QQ[dirNE  *sizeQ];
	Q.q27[dirSW] = &QQ[dirSW  *sizeQ];
	Q.q27[dirSE] = &QQ[dirSE  *sizeQ];
	Q.q27[dirNW] = &QQ[dirNW  *sizeQ];
	Q.q27[dirTE] = &QQ[dirTE  *sizeQ];
	Q.q27[dirBW] = &QQ[dirBW  *sizeQ];
	Q.q27[dirBE] = &QQ[dirBE  *sizeQ];
	Q.q27[dirTW] = &QQ[dirTW  *sizeQ];
	Q.q27[dirTN] = &QQ[dirTN  *sizeQ];
	Q.q27[dirBS] = &QQ[dirBS  *sizeQ];
	Q.q27[dirBN] = &QQ[dirBN  *sizeQ];
	Q.q27[dirTS] = &QQ[dirTS  *sizeQ];
	Q.q27[dirZERO] = &QQ[dirZERO*sizeQ];
	Q.q27[dirTNE] = &QQ[dirTNE *sizeQ];
	Q.q27[dirTSW] = &QQ[dirTSW *sizeQ];
	Q.q27[dirTSE] = &QQ[dirTSE *sizeQ];
	Q.q27[dirTNW] = &QQ[dirTNW *sizeQ];
	Q.q27[dirBNE] = &QQ[dirBNE *sizeQ];
	Q.q27[dirBSW] = &QQ[dirBSW *sizeQ];
	Q.q27[dirBSE] = &QQ[dirBSE *sizeQ];
	Q.q27[dirBNW] = &QQ[dirBNW *sizeQ];
}

void GridReader::setSizeNoSlip(std::shared_ptr<BoundaryQs> boundaryQ, unsigned int level) const
{
	para->getParH(level)->QWall.kQ = boundaryQ->getSize(level);
	para->getParD(level)->QWall.kQ = para->getParH(level)->QWall.kQ;
	para->getParH(level)->kQ = para->getParH(level)->QWall.kQ;
	para->getParD(level)->kQ = para->getParH(level)->QWall.kQ;
    cudaMemoryManager->cudaAllocWallBC(level);
}

void GridReader::setSizeGeoQs(std::shared_ptr<BoundaryQs> boundaryQ, unsigned int level) const
{
	para->getParH(level)->QGeom.kQ = boundaryQ->getSize(level);
	para->getParD(level)->QGeom.kQ = para->getParH(level)->QGeom.kQ;

    cudaMemoryManager->cudaAllocGeomBC(level);
}

void GridReader::printQSize(std::string bc, std::shared_ptr<BoundaryQs> boundaryQ, unsigned int level) const
{
	std::cout << "level " << level << ", " << bc << "-size: " << boundaryQ->getSize(level) << std::endl;
}


void GridReader::setDimensions()
{
	std::ifstream numberNodes;
	numberNodes.open(para->getnumberNodes().c_str(), std::ios::in);
	if (!numberNodes) {
		std::cerr << "can't open file NumberNodes: " << para->getnumberNodes() << std::endl;
		exit(1);
	}

	std::string buffer;
	int bufferInt;
	std::vector<int> localGridNX;
	std::vector<int> localGridNY;
	std::vector<int> localGridNZ;

	for (/*unsigned*/ int i = 0; i <= para->getMaxLevel(); i++) {
		numberNodes >> buffer;
		numberNodes >> bufferInt;
		localGridNX.push_back(bufferInt);
		numberNodes >> bufferInt;
		localGridNY.push_back(bufferInt);
		numberNodes >> bufferInt;
		localGridNZ.push_back(bufferInt);
	}
	para->setGridX(localGridNX);
	para->setGridY(localGridNY);
	para->setGridZ(localGridNZ);
}

void GridReader::setBoundingBox()
{
	std::ifstream numberNodes;
	numberNodes.open(para->getLBMvsSI().c_str(), std::ios::in);
	if (!numberNodes) {
		std::cerr << "can't open file LBMvsSI" << std::endl;
		exit(1);
	}
	real bufferreal;
	std::vector<real> minX, maxX, minY, maxY, minZ, maxZ;

	for (int i = 0; i <= para->getMaxLevel(); i++) {
		numberNodes >> bufferreal;
		minX.push_back(bufferreal);
		numberNodes >> bufferreal;
		minY.push_back(bufferreal);
		numberNodes >> bufferreal;
		minZ.push_back(bufferreal);
		numberNodes >> bufferreal;
		maxX.push_back(bufferreal);
		numberNodes >> bufferreal;
		maxY.push_back(bufferreal);
		numberNodes >> bufferreal;
		maxZ.push_back(bufferreal);
	}
	para->setMinCoordX(minX);
	para->setMinCoordY(minY);
	para->setMinCoordZ(minZ);
	para->setMaxCoordX(maxX);
	para->setMaxCoordY(maxY);
	para->setMaxCoordZ(maxZ);
}

void GridReader::initPeriodicNeigh(std::vector<std::vector<std::vector<unsigned int> > > periodV, std::vector<std::vector<unsigned int> > periodIndex,  std::string boundaryCondition)
{
	std::vector<unsigned int>neighVec;
	std::vector<unsigned int>indexVec;
	
	int counter = 0;

	for(unsigned int i=0; i<neighX->getLevel();i++) {
		if(boundaryCondition =="periodic_y"){
			neighVec = neighY->getVec(i);
		} 
		else if(boundaryCondition =="periodic_x"){
			neighVec = neighX->getVec(i);
		}
		else if(boundaryCondition =="periodic_z"){
			neighVec = neighZ->getVec(i);
		}
		else {
			std::cout << "wrong String in periodicValue" << std::endl;
			exit(1);
		}

		for (std::vector<unsigned int>::iterator it = periodIndex[i].begin(); it != periodIndex[i].end(); it++) {
			if(periodV[i][0][counter] != 0) {
				neighVec[*it]=periodV[i][0][counter];
			}

			counter++;
		}


		if(boundaryCondition =="periodic_y"){
			neighY->setVec(i, neighVec);
		} 
		else if(boundaryCondition =="periodic_x"){
			neighX->setVec(i, neighVec);
		}
		else if(boundaryCondition =="periodic_z"){
			neighZ->setVec(i, neighVec);
		}

	}
}

void GridReader::makeReader(std::shared_ptr<Parameter> para)
{
	for (int i = 0; i < BC_Values.size(); i++)
	{
		if (channelDirections[i].compare("inlet") == 0){ BC_Values[i]  = std::shared_ptr<BoundaryValues>(new BoundaryValues(para->getinletBcValues())); }
		if (channelDirections[i].compare("outlet") == 0){ BC_Values[i] = std::shared_ptr<BoundaryValues>(new BoundaryValues(para->getoutletBcValues())); }
		if (channelDirections[i].compare("back") == 0){ BC_Values[i]   = std::shared_ptr<BoundaryValues>(new BoundaryValues(para->getbackBcValues())); }
		if (channelDirections[i].compare("front") == 0){ BC_Values[i]  = std::shared_ptr<BoundaryValues>(new BoundaryValues(para->getfrontBcValues())); }
		if (channelDirections[i].compare("top") == 0){ BC_Values[i]    = std::shared_ptr<BoundaryValues>(new BoundaryValues(para->gettopBcValues())); }
		if (channelDirections[i].compare("bottom") == 0){ BC_Values[i] = std::shared_ptr<BoundaryValues>(new BoundaryValues(para->getbottomBcValues()));}
	}
}

void GridReader::makeReader(std::vector<std::shared_ptr<BoundaryQs> > &BC_Qs, std::shared_ptr<Parameter> para)
{
	for (int i = 0; i < BC_Qs.size(); i++)
	{
		if (channelDirections[i].compare("inlet") == 0){ BC_Qs[i]  = std::shared_ptr<BoundaryQs>(new BoundaryQs(para->getinletBcQs(), false)); }
		if (channelDirections[i].compare("outlet") == 0){ BC_Qs[i] = std::shared_ptr<BoundaryQs>(new BoundaryQs(para->getoutletBcQs(), false)); }
		if (channelDirections[i].compare("back") == 0){ BC_Qs[i]   = std::shared_ptr<BoundaryQs>(new BoundaryQs(para->getbackBcQs(), false)); }
		if (channelDirections[i].compare("front") == 0){ BC_Qs[i]  = std::shared_ptr<BoundaryQs>(new BoundaryQs(para->getfrontBcQs(), false)); }
		if (channelDirections[i].compare("top") == 0){ BC_Qs[i]    = std::shared_ptr<BoundaryQs>(new BoundaryQs(para->gettopBcQs(), false)); }
		if (channelDirections[i].compare("bottom") == 0){ BC_Qs[i] = std::shared_ptr<BoundaryQs>(new BoundaryQs(para->getbottomBcQs(), false)); }
	}
}

void GridReader::setChannelBoundaryCondition()
{
	for (int i = 0; i < channelDirections.size(); i++)
	{
		this->channelBoundaryConditions[i] = BC_Values[i]->getBoundaryCondition();
		std::cout << this->channelDirections[i] << " Boundary: " << channelBoundaryConditions[i] << std::endl;
	}
}