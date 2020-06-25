#ifndef INTERFACEDEBUG_HPP
#define INTERFACEDEBUG_HPP

#include <stdio.h>
#include <fstream>
#include <sstream>
// #include <math.h>
#include <cmath>
#include "LBM/LB.h"
#include "LBM/D3Q27.h"
#include "Parameter/Parameter.h"
#include "basics/utilities/UbSystem.h"
#include "Utilities/StringUtil.hpp"
#include <basics/writer/WbWriterVtkXmlBinary.h>


using namespace std;

namespace InterfaceDebugWriter
{

    void writeGridInterfaceLines(Parameter* para, int level, const uint* coarse, const uint* fine, uint numberOfNodes, const std::string& name)
    {
        vector<UbTupleFloat3> nodes(numberOfNodes * 2);
        vector<UbTupleInt2> cells(numberOfNodes);

        int actualNodeNumber = 0;
        for (uint u = 0; u < numberOfNodes; u++)
        {
            const int posCoarse = coarse[u];
            const double x1Coarse = para->getParH(level)->coordX_SP[posCoarse];
            const double x2Coarse = para->getParH(level)->coordY_SP[posCoarse];
            const double x3Coarse = para->getParH(level)->coordZ_SP[posCoarse];

            const int posFine = fine[u];
            const double x1Fine = para->getParH(level + 1)->coordX_SP[posFine];
            const double x2Fine = para->getParH(level + 1)->coordY_SP[posFine];
            const double x3Fine = para->getParH(level + 1)->coordZ_SP[posFine];

            nodes[actualNodeNumber++] = makeUbTuple(float(x1Coarse), float(x2Coarse), float(x3Coarse));
            nodes[actualNodeNumber++] = makeUbTuple(float(x1Fine), float(x2Fine), float(x3Fine));

            cells[u] = makeUbTuple(actualNodeNumber - 2, actualNodeNumber - 1);
        }
        WbWriterVtkXmlBinary::getInstance()->writeLines(name, nodes, cells);
    }


    void writeInterfaceLinesDebugCF(Parameter* para)
    {
		for (int level = 0; level < para->getMaxLevel(); level++)
		{
            const std::string fileName = para->getFName() + "_" + StringUtil::toString<int>(level) + "_OffDebugCF.vtk";
            writeGridInterfaceLines(para, level, para->getParH(level)->intCF.ICellCFC, para->getParH(level)->intCF.ICellCFF, para->getParH(level)->K_CF, fileName);
		}
	}


	void writeInterfaceLinesDebugFC(Parameter* para)
    {
        for (int level = 0; level < para->getMaxLevel(); level++)
        {
            const std::string fileName = para->getFName() + "_" + StringUtil::toString<int>(level) + "_OffDebugFC.vtk";
            writeGridInterfaceLines(para, level, para->getParH(level)->intFC.ICellFCC, para->getParH(level)->intFC.ICellFCF, para->getParH(level)->K_FC, fileName);
        }
	}


	//////////////////////////////////////////////////////////////////////////
    void writeGridInterfaceLinesNeighbors(Parameter* para, int level, const uint* interfaceIndices, uint numberOfNodes, const std::string& name)
    {
        vector<UbTupleFloat3> nodes(numberOfNodes * 2);
        vector<UbTupleInt2> cells(numberOfNodes);

        int actualNodeNumber = 0;
        for (uint u = 0; u < numberOfNodes; u++)
        {
            const int pos = interfaceIndices[u];
            const double x1 = para->getParH(level)->coordX_SP[pos];
            const double x2 = para->getParH(level)->coordY_SP[pos];
            const double x3 = para->getParH(level)->coordZ_SP[pos];
	
            const double x1Neighbor = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];
            const double x2Neighbor = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];
            const double x3Neighbor = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];

            nodes[actualNodeNumber++] = (makeUbTuple(float(x1), float(x2), float(x3)));
            nodes[actualNodeNumber++] = (makeUbTuple(float(x1Neighbor), float(x2Neighbor), float(x3Neighbor)));

            cells[u] = makeUbTuple(actualNodeNumber - 2, actualNodeNumber - 1);
        }
        WbWriterVtkXmlBinary::getInstance()->writeLines(name, nodes, cells);
    }

	void writeInterfaceLinesDebugCFCneighbor(Parameter* para)
    {
		for (int level = 0; level < para->getMaxLevel(); level++)
		{
            std::string filename = para->getFName() + "_" + StringUtil::toString<int>(level) + "_CFCneighbor.vtk";
            writeGridInterfaceLinesNeighbors(para, level, para->getParH(level)->intCF.ICellCFC, para->getParH(level)->K_CF, filename);
		}
	}


	//////////////////////////////////////////////////////////////////////////
	void writeInterfaceLinesDebugCFFneighbor(Parameter* para)
    {
        for (int level = 0; level < para->getMaxLevel(); level++)
        {
            std::string filename = para->getFName() + "_" + StringUtil::toString<int>(level) + "_CFFneighbor.vtk";
            writeGridInterfaceLinesNeighbors(para, level + 1, para->getParH(level)->intCF.ICellCFF, para->getParH(level)->K_CF, filename);
        }
	}


	//////////////////////////////////////////////////////////////////////////
	void writeInterfaceLinesDebugFCCneighbor(Parameter* para)
    {
        for (int level = 0; level < para->getMaxLevel(); level++)
        {
            std::string filename = para->getFName() + "_" + StringUtil::toString<int>(level) + "_FCCneighbor.vtk";
            writeGridInterfaceLinesNeighbors(para, level, para->getParH(level)->intFC.ICellFCC, para->getParH(level)->K_FC, filename);
        }
	}


	//////////////////////////////////////////////////////////////////////////
	void writeInterfaceLinesDebugFCFneighbor(Parameter* para)
    {
        for (int level = 0; level < para->getMaxLevel(); level++)
        {
            std::string filename = para->getFName() + "_" + StringUtil::toString<int>(level) + "_FCFneighbor.vtk";
            writeGridInterfaceLinesNeighbors(para, level + 1, para->getParH(level)->intFC.ICellFCF, para->getParH(level)->K_FC, filename);
        }
	}


	//////////////////////////////////////////////////////////////////////////
	void writeInterfaceLinesDebugOff(Parameter* para){
		vector< UbTupleFloat3 > nodesVec;
		vector< UbTupleInt2 > cellsVec;
		int nodeNumberVec = 0;

		for (int level = 0; level < para->getMaxLevel(); level++) //evtl. Maxlevel + 1
		{
			nodeNumberVec += (int)para->getParH(level)->K_CF;
		}
		nodesVec.resize(nodeNumberVec*8);
		int nodeCount = 0;
		for (int level = 0; level < para->getMaxLevel(); level++)
		{
			double nodeDeltaLevel = para->getParH(level)->dx;
			double nodeDeltaLevelFine = para->getParH(level+1)->dx;
			double halfNodeDeltaLevel = 0.5*nodeDeltaLevel;
			double halfNodeDeltaLevelFine = 0.5*nodeDeltaLevelFine;

			for(unsigned int u=0;u<para->getParH(level)->K_CF;u++)
			{
				double xoff = para->getParH(level)->offCF.xOffCF[u];
				double yoff = para->getParH(level)->offCF.yOffCF[u];
				double zoff = para->getParH(level)->offCF.zOffCF[u];

				int posFine = para->getParH(level)->intCF.ICellCFF[u];

				double x1Fine = para->getParH(level+1)->coordX_SP[posFine];
				double x2Fine = para->getParH(level+1)->coordY_SP[posFine];
				double x3Fine = para->getParH(level+1)->coordZ_SP[posFine];

				double x1 = x1Fine + xoff;
				double x2 = x2Fine + yoff;
				double x3 = x3Fine + zoff;

				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1Fine),(float)(x2Fine),(float)(x3Fine) ) );

				cellsVec.push_back( makeUbTuple(nodeCount-2,nodeCount-1) );

			}
			std::string filenameVec = para->getFName()+"_"+StringUtil::toString<int>(level)+"_OffDebugCF_Offs.vtk";
			WbWriterVtkXmlBinary::getInstance()->writeLines(filenameVec,nodesVec,cellsVec);
            cellsVec.clear();
            nodesVec.clear();
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeInterfacePointsDebugCFC(Parameter* para){
		vector< UbTupleFloat3 > nodesVec2;
		int nodeNumberVec = 0;

		for (int level = 0; level < para->getMaxLevel(); level++) //evtl. Maxlevel + 1
		{
			nodeNumberVec += (int)para->getParH(level)->K_CF;
		}
		nodesVec2.resize(nodeNumberVec*8); 
		int nodeCount2 = 0; 
		for (int level = 0; level < para->getMaxLevel(); level++)
		{
			double nodeDeltaLevel = para->getParH(level)->dx;
			double halfNodeDeltaLevel = 0.5*nodeDeltaLevel;

			for(unsigned int u=0;u<para->getParH(level)->K_CF;u++)
			{
				int pos = para->getParH(level)->intCF.ICellCFC[u];

				double x1 = para->getParH(level)->coordX_SP[pos];
				double x2 = para->getParH(level)->coordY_SP[pos];
				double x3 = para->getParH(level)->coordZ_SP[pos];

				nodesVec2[nodeCount2++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );

			}
			std::string filenameVec2 = para->getFName()+"_"+StringUtil::toString<int>(level)+"_OffDebugPointsCF.vtk";
			WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec2,nodesVec2);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeBcPointsDebug(Parameter* para){
		vector< UbTupleFloat3 > nodesVec2;
		int nodeNumberVec = 0;

		for (int level = 0; level <= para->getMaxLevel(); level++) //evtl. Maxlevel + 1
		{
			nodeNumberVec += (int)para->getParH(level)->QWall.kQ;
		}
		nodesVec2.resize(nodeNumberVec*8); 
		int nodeCount2 = 0; 
		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			double nodeDeltaLevel = para->getParH(level)->dx;
			double halfNodeDeltaLevel = 0.5*nodeDeltaLevel;

			for(int u=0;u<para->getParH(level)->QWall.kQ;u++)
			{
				int pos = para->getParH(level)->QWall.k[u];

				double x1 = para->getParH(level)->coordX_SP[pos];
				double x2 = para->getParH(level)->coordY_SP[pos];
				double x3 = para->getParH(level)->coordZ_SP[pos];

				nodesVec2[nodeCount2++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );

			}
			std::string filenameVec2 = para->getFName()+"_PointsBc_"+StringUtil::toString<int>(level);
			WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec2,nodesVec2);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writePressPointsDebug(Parameter* para){
		vector< UbTupleFloat3 > nodesVec;
		int nodeNumberVec = 0;

		for (int level = 0; level <= para->getMaxLevel(); level++) //evtl. Maxlevel + 1
		{
			nodeNumberVec += (int)para->getParH(level)->QPress.kQ;
		}
		nodesVec.resize(nodeNumberVec); 
		int nodeCount2 = 0; 
		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			for(int u=0;u<para->getParH(level)->QPress.kQ;u++)
			{
				int pos = para->getParH(level)->QPress.k[u];

				double x1 = para->getParH(level)->coordX_SP[pos];
				double x2 = para->getParH(level)->coordY_SP[pos];
				double x3 = para->getParH(level)->coordZ_SP[pos];

				nodesVec[nodeCount2++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );

			}
			std::string filenameVec = para->getFName()+"_PointsPress_"+StringUtil::toString<int>(level);
			WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec,nodesVec);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writePressNeighborPointsDebug(Parameter* para){
		vector< UbTupleFloat3 > nodesVec;
		int nodeNumberVec = 0;

		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			nodeNumberVec += (int)para->getParH(level)->QPress.kQ;
		}
		nodesVec.resize(nodeNumberVec); 
		int nodeCount2 = 0; 
		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			for(int u=0;u<para->getParH(level)->QPress.kQ;u++)
			{
				int pos = para->getParH(level)->QPress.kN[u];

				real x1 = para->getParH(level)->coordX_SP[pos];
				real x2 = para->getParH(level)->coordY_SP[pos];
				real x3 = para->getParH(level)->coordZ_SP[pos];

				nodesVec[nodeCount2++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );
			}
			std::string filenameVec = para->getFName()+"_PointsPressNeighbor_"+StringUtil::toString<int>(level);
			WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec,nodesVec);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeNeighborXPointsDebug(Parameter* para){
		vector< UbTupleFloat3 > nodesVec;
		int nodeNumberVec = 0;

		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			nodeNumberVec += (int)para->getParH(level)->size_Mat_SP;
		}
		nodesVec.resize(nodeNumberVec); 
		int nodeCount2 = 0;
		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			for(unsigned int u=0;u<para->getParH(level)->size_Mat_SP;u++)
			{
				real x1 = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[u]];
				real x2 = para->getParH(level)->coordY_SP[para->getParH(level)->neighborX_SP[u]];
				real x3 = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborX_SP[u]];

				nodesVec[nodeCount2++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );
			}
			std::string filenameVec = para->getFName()+"_PointsNeighborX_"+StringUtil::toString<int>(level);
			WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec,nodesVec);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeNeighborXLinesDebug(Parameter* para){
		vector< UbTupleFloat3 > nodesVec;
		vector< UbTupleInt2 > cellsVec;
		int nodeNumberVec = 0;

		for (int level = 0; level < para->getMaxLevel(); level++) //evtl. Maxlevel + 1
		{
			nodeNumberVec += (int)para->getParH(level)->size_Mat_SP;
		}
		nodesVec.resize(nodeNumberVec*2);
		int nodeCount = 0;
		for (int level = 0; level < para->getMaxLevel(); level++)
		{
			for(unsigned int u=0;u<para->getParH(level)->size_Mat_SP;u++)
			{
				real x1 = para->getParH(level)->coordX_SP[u];
				real x2 = para->getParH(level)->coordY_SP[u];
				real x3 = para->getParH(level)->coordZ_SP[u];
				real x1N = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[u]];
				real x2N = para->getParH(level)->coordY_SP[para->getParH(level)->neighborX_SP[u]];
				real x3N = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborX_SP[u]];

				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1N),(float)(x2N),(float)(x3N) ) );

				if (para->getParH(level)->geoSP[u]==GEO_FLUID)
				{
					cellsVec.push_back( makeUbTuple(nodeCount-2,nodeCount-1) );
				}

			}
			std::string filenameVec = para->getFName()+"_"+StringUtil::toString<int>(level)+"_NeighborX_Lines.vtk";
			WbWriterVtkXmlBinary::getInstance()->writeLines(filenameVec,nodesVec,cellsVec);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeNeighborYPointsDebug(Parameter* para){
		vector< UbTupleFloat3 > nodesVec;
		int nodeNumberVec = 0;

		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			nodeNumberVec += (int)para->getParH(level)->size_Mat_SP;
		}
		nodesVec.resize(nodeNumberVec); 
		int nodeCount2 = 0;
		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			for(unsigned int u=0;u<para->getParH(level)->size_Mat_SP;u++)
			{
				real x1 = para->getParH(level)->coordX_SP[para->getParH(level)->neighborY_SP[u]];
				real x2 = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[u]];
				real x3 = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborY_SP[u]];

				nodesVec[nodeCount2++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );
			}
			std::string filenameVec = para->getFName()+"_PointsNeighborY_"+StringUtil::toString<int>(level);
			WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec,nodesVec);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeNeighborYLinesDebug(Parameter* para){
		vector< UbTupleFloat3 > nodesVec;
		vector< UbTupleInt2 > cellsVec;
		int nodeNumberVec = 0;

		for (int level = 0; level < para->getMaxLevel(); level++) //evtl. Maxlevel + 1
		{
			nodeNumberVec += (int)para->getParH(level)->size_Mat_SP;
		}
		nodesVec.resize(nodeNumberVec*2);
		int nodeCount = 0;
		for (int level = 0; level < para->getMaxLevel(); level++)
		{
			for(unsigned int u=0;u<para->getParH(level)->size_Mat_SP;u++)
			{
				real x1 = para->getParH(level)->coordX_SP[u];
				real x2 = para->getParH(level)->coordY_SP[u];
				real x3 = para->getParH(level)->coordZ_SP[u];
				real x1N = para->getParH(level)->coordX_SP[para->getParH(level)->neighborY_SP[u]];
				real x2N = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[u]];
				real x3N = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborY_SP[u]];

				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1N),(float)(x2N),(float)(x3N) ) );

				if (para->getParH(level)->geoSP[u]==GEO_FLUID)
				{
					cellsVec.push_back( makeUbTuple(nodeCount-2,nodeCount-1) );
				}

			}
			std::string filenameVec = para->getFName()+"_"+StringUtil::toString<int>(level)+"_NeighborY_Lines.vtk";
			WbWriterVtkXmlBinary::getInstance()->writeLines(filenameVec,nodesVec,cellsVec);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeNeighborZPointsDebug(Parameter* para){
		vector< UbTupleFloat3 > nodesVec;
		int nodeNumberVec = 0;

		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			nodeNumberVec += (int)para->getParH(level)->size_Mat_SP;
		}
		nodesVec.resize(nodeNumberVec); 
		int nodeCount2 = 0;
		for (int level = 0; level <= para->getMaxLevel(); level++)
		{
			for(unsigned int u=0;u<para->getParH(level)->size_Mat_SP;u++)
			{
				real x1 = para->getParH(level)->coordX_SP[para->getParH(level)->neighborZ_SP[u]];
				real x2 = para->getParH(level)->coordY_SP[para->getParH(level)->neighborZ_SP[u]];
				real x3 = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[u]];

				nodesVec[nodeCount2++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );
			}
			std::string filenameVec = para->getFName()+"_PointsNeighborZ_"+StringUtil::toString<int>(level);
			WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec,nodesVec);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeNeighborZLinesDebug(Parameter* para){
		vector< UbTupleFloat3 > nodesVec;
		vector< UbTupleInt2 > cellsVec;
		int nodeNumberVec = 0;

		for (int level = 0; level < para->getMaxLevel(); level++) //evtl. Maxlevel + 1
		{
			nodeNumberVec += (int)para->getParH(level)->size_Mat_SP;
		}
		nodesVec.resize(nodeNumberVec*2);
		int nodeCount = 0;
		for (int level = 0; level < para->getMaxLevel(); level++)
		{
			for(unsigned int u=0;u<para->getParH(level)->size_Mat_SP;u++)
			{
				real x1 = para->getParH(level)->coordX_SP[u];
				real x2 = para->getParH(level)->coordY_SP[u];
				real x3 = para->getParH(level)->coordZ_SP[u];
				real x1N = para->getParH(level)->coordX_SP[para->getParH(level)->neighborZ_SP[u]];
				real x2N = para->getParH(level)->coordY_SP[para->getParH(level)->neighborZ_SP[u]];
				real x3N = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[u]];

				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1),(float)(x2),(float)(x3) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1N),(float)(x2N),(float)(x3N) ) );

				if (para->getParH(level)->geoSP[u]==GEO_FLUID)
				{
					cellsVec.push_back( makeUbTuple(nodeCount-2,nodeCount-1) );
				}

			}
			std::string filenameVec = para->getFName()+"_"+StringUtil::toString<int>(level)+"_NeighborZ_Lines.vtk";
			WbWriterVtkXmlBinary::getInstance()->writeLines(filenameVec,nodesVec,cellsVec);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeInterfaceCellsDebugCFC(Parameter* para){

		vector< UbTupleFloat3 > nodesVec;
		vector< UbTupleInt8 > cellsVec;
		int nodeNumberVec = 0;
		for (int level = 0; level < para->getMaxLevel(); level++) //evtl. Maxlevel + 1
		{
			nodeNumberVec += (int)para->getParH(level)->K_CF;
		}
		nodesVec.resize(nodeNumberVec*8);
		int nodeCount = 0;
		for (int level = 0; level < para->getMaxLevel(); level++)
		{
			double nodeDeltaLevel = para->getParH(level)->dx;
			double halfNodeDeltaLevel = 0.5*nodeDeltaLevel;

			for(unsigned int u=0;u<para->getParH(level)->K_CF;u++)
			{
				int pos  = para->getParH(level)->intCF.ICellCFC[u];

				double x1  = para->getParH(level)->coordX_SP[pos];
				double x2  = para->getParH(level)->coordY_SP[pos];
				double x3  = para->getParH(level)->coordZ_SP[pos];
				double x1P = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];
				double x2P = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];
				double x3P = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1P),(float)(x2 ),(float)(x3 ) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1P),(float)(x2P),(float)(x3 ) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1 ),(float)(x2P),(float)(x3 ) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3P) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1P),(float)(x2 ),(float)(x3P) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1P),(float)(x2P),(float)(x3P) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1 ),(float)(x2P),(float)(x3P) ) );

				cellsVec.push_back( makeUbTuple(nodeCount-8,nodeCount-7,nodeCount-6,nodeCount-5,nodeCount-4,nodeCount-3,nodeCount-2,nodeCount-1) );

			}
			std::string filenameVec = para->getFName()+"_CellsCFC_"+StringUtil::toString<int>(level);
			WbWriterVtkXmlBinary::getInstance()->writeOcts(filenameVec,nodesVec,cellsVec);
		}
	}


	//////////////////////////////////////////////////////////////////////////


	void writeInterfaceCellsDebugCFF(Parameter* para){

		vector< UbTupleFloat3 > nodesVec;
		vector< UbTupleInt8 > cellsVec;
		int nodeNumberVec = 0;
		for (int level = 0; level < para->getMaxLevel(); level++) //evtl. Maxlevel + 1
		{
			nodeNumberVec += (int)para->getParH(level)->K_CF;
		}
		nodesVec.resize(nodeNumberVec*8);
		int nodeCount = 0;
		for (int level = 0; level < para->getMaxLevel(); level++)
		{
			double nodeDeltaLevel = para->getParH(level+1)->dx;
			double halfNodeDeltaLevel = 0.5*nodeDeltaLevel;

			for(unsigned int u=0;u<para->getParH(level)->K_CF;u++)
			{
				int pos  = para->getParH(level  )->intCF.ICellCFF[u];

				double x1  = para->getParH(level+1)->coordX_SP[pos];
				double x2  = para->getParH(level+1)->coordY_SP[pos];
				double x3  = para->getParH(level+1)->coordZ_SP[pos];
				double x1P = para->getParH(level+1)->coordX_SP[para->getParH(level+1)->neighborX_SP[pos]];
				double x2P = para->getParH(level+1)->coordY_SP[para->getParH(level+1)->neighborY_SP[pos]];
				double x3P = para->getParH(level+1)->coordZ_SP[para->getParH(level+1)->neighborZ_SP[pos]];
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1P),(float)(x2 ),(float)(x3 ) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1P),(float)(x2P),(float)(x3 ) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1 ),(float)(x2P),(float)(x3 ) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3P) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1P),(float)(x2 ),(float)(x3P) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1P),(float)(x2P),(float)(x3P) ) );
				nodesVec[nodeCount++]=( makeUbTuple( (float)(x1 ),(float)(x2P),(float)(x3P) ) );

				cellsVec.push_back( makeUbTuple(nodeCount-8,nodeCount-7,nodeCount-6,nodeCount-5,nodeCount-4,nodeCount-3,nodeCount-2,nodeCount-1) );

			}
			std::string filenameVec = para->getFName()+"_CellsCFF_"+StringUtil::toString<int>(level);
			WbWriterVtkXmlBinary::getInstance()->writeOcts(filenameVec,nodesVec,cellsVec);
		}
	}
}

#endif
