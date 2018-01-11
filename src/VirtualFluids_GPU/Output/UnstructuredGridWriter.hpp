#ifndef UNSTRUCTUREDGRID_HPP
#define UNSTRUCTUREDGRID_HPP

#include <stdio.h>
#include <fstream>
#include <sstream>
// #include <math.h>
#include <cmath>
#include "LBM/LB.h"
#include "LBM/D3Q27.h"
#include "Parameter/Parameter.h"
#include "VirtualFluidsBasics/basics/utilities/UbSystem.h"
#include <VirtualFluidsBasics/basics/writer/WbWriterVtkXmlBinary.h>
#include <VirtualFluidsBasics/basics/writer/WbWriterVtkXmlASCII.h>
#include <VirtualFluidsBasics/basics/utilities/UbTuple.h>

using namespace std;

namespace UnstrucuredGridWriter
{


	void writeUnstrucuredGrid(Parameter* para, int level, std::string& fname, std::string& filenameVec2) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		vector< string > nodedatanames;
		nodedatanames.push_back("rho");
		nodedatanames.push_back("vx1");
		nodedatanames.push_back("vx2");
		nodedatanames.push_back("vx3");
		nodedatanames.push_back("geo");
		//int number1,number2,number3,number4,number5,number6,number7,number8;
		vector< vector< double > > nodedata(nodedatanames.size());

		bool neighborsFluid;

		unsigned int allnodes = para->getParH(level)->size_Mat_SP * 8;

		nodes.resize(allnodes);
		nodedata[0].resize(allnodes);
		nodedata[1].resize(allnodes);
		nodedata[2].resize(allnodes);
		nodedata[3].resize(allnodes);
		nodedata[4].resize(allnodes);

		unsigned int nodeCount = 0;
		double nodeDeltaLevel = para->getParH(level)->dx;

		for (unsigned int pos=0;pos<para->getParH(level)->size_Mat_SP;pos++)
		{
			if (para->getParH(level)->geoSP[pos] == GEO_FLUID /*!= GEO_VOID*/)
			{
				//////////////////////////////////////////////////////////////////////////
				double ix1  = para->getParH(level)->coordX_SP[pos];//-STARTOFFX;
				double ix2  = para->getParH(level)->coordY_SP[pos];//-STARTOFFY;
				double ix3  = para->getParH(level)->coordZ_SP[pos];//-STARTOFFZ;
				double ix1P = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];//-STARTOFFX;
				double ix2P = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];//-STARTOFFY;
				double ix3P = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];//-STARTOFFZ;
				//////////////////////////////////////////////////////////////////////////
				double x1  = ix1;  // para->getParH(level)->distX + ix1 *nodeDeltaLevel;// + tmpDist;
				double x2  = ix2;  // para->getParH(level)->distY + ix2 *nodeDeltaLevel;// + tmpDist;
				double x3  = ix3;  // para->getParH(level)->distZ + ix3 *nodeDeltaLevel;// + tmpDist;
				double x1P = ix1P; // para->getParH(level)->distX + ix1P*nodeDeltaLevel;// + tmpDist;
				double x2P = ix2P; // para->getParH(level)->distY + ix2P*nodeDeltaLevel;// + tmpDist;
				double x3P = ix3P; // para->getParH(level)->distZ + ix3P*nodeDeltaLevel;// + tmpDist;
				//////////////////////////////////////////////////////////////////////////
				neighborsFluid = true;
				//////////////////////////////////////////////////////////////////////////
				//1
				nodes[nodeCount]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
				nodedata[0][nodeCount] = para->getParH(level)->rho_SP[pos] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[1][nodeCount] = para->getParH(level)->vx_SP[pos] * para->getVelocityRatio();
				nodedata[2][nodeCount] = para->getParH(level)->vy_SP[pos] * para->getVelocityRatio();
				nodedata[3][nodeCount] = para->getParH(level)->vz_SP[pos] * para->getVelocityRatio();
				nodedata[4][nodeCount] = para->getParH(level)->geoSP[pos];
				//if(para->getParH(level)->geoSP[pos]==GEO_VOID) neighborsFluid = false;
				nodeCount++;
				//////////////////////////////////////////////////////////////////////////
				//2
				nodes[nodeCount]=( makeUbTuple( (float)(x1P),(float)(x2 ),(float)(x3 ) ) );
				nodedata[0][nodeCount] = para->getParH(level)->rho_SP[para->getParH(level)->neighborX_SP[pos]] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[1][nodeCount] = para->getParH(level)->vx_SP[para->getParH(level)->neighborX_SP[pos]] * para->getVelocityRatio();
				nodedata[2][nodeCount] = para->getParH(level)->vy_SP[para->getParH(level)->neighborX_SP[pos]] * para->getVelocityRatio();
				nodedata[3][nodeCount] = para->getParH(level)->vz_SP[para->getParH(level)->neighborX_SP[pos]] * para->getVelocityRatio();
				nodedata[4][nodeCount] = para->getParH(level)->geoSP[para->getParH(level)->neighborX_SP[pos]];
				//if(para->getParH(level)->geoSP[para->getParH(level)->neighborX_SP[pos]]==GEO_VOID) neighborsFluid = false;
				nodeCount++;
				//////////////////////////////////////////////////////////////////////////
				//3
				nodes[nodeCount]=( makeUbTuple( (float)(x1P),(float)(x2P),(float)(x3 ) ) );
				nodedata[0][nodeCount] = para->getParH(level)->rho_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[1][nodeCount] = para->getParH(level)->vx_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]] * para->getVelocityRatio();
				nodedata[2][nodeCount] = para->getParH(level)->vy_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]] * para->getVelocityRatio();
				nodedata[3][nodeCount] = para->getParH(level)->vz_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]] * para->getVelocityRatio();
				nodedata[4][nodeCount] = para->getParH(level)->geoSP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]];
				//if(para->getParH(level)->geoSP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]]==GEO_VOID) neighborsFluid = false;
				nodeCount++;
				//////////////////////////////////////////////////////////////////////////
				//4
				nodes[nodeCount]=( makeUbTuple( (float)(x1 ),(float)(x2P),(float)(x3 ) ) );
				nodedata[0][nodeCount] = para->getParH(level)->rho_SP[para->getParH(level)->neighborY_SP[pos]] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[1][nodeCount] = para->getParH(level)->vx_SP[para->getParH(level)->neighborY_SP[pos]] * para->getVelocityRatio();
				nodedata[2][nodeCount] = para->getParH(level)->vy_SP[para->getParH(level)->neighborY_SP[pos]] * para->getVelocityRatio();
				nodedata[3][nodeCount] = para->getParH(level)->vz_SP[para->getParH(level)->neighborY_SP[pos]] * para->getVelocityRatio();
				nodedata[4][nodeCount] = para->getParH(level)->geoSP[para->getParH(level)->neighborY_SP[pos]];
				//if(para->getParH(level)->geoSP[para->getParH(level)->neighborY_SP[pos]]==GEO_VOID) neighborsFluid = false;
				//if((para->getParH(level)->neighborY_SP[pos]<=pos) && ((para->getParH(level)->coordY_SP[pos]) > (para->getParH(level)->gridNY-2))) neighborsFluid = false;
				nodeCount++;
				//////////////////////////////////////////////////////////////////////////
				//5
				nodes[nodeCount]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3P) ) );
				nodedata[0][nodeCount] = para->getParH(level)->rho_SP[para->getParH(level)->neighborZ_SP[pos]] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[1][nodeCount] = para->getParH(level)->vx_SP[para->getParH(level)->neighborZ_SP[pos]] * para->getVelocityRatio();
				nodedata[2][nodeCount] = para->getParH(level)->vy_SP[para->getParH(level)->neighborZ_SP[pos]] * para->getVelocityRatio();
				nodedata[3][nodeCount] = para->getParH(level)->vz_SP[para->getParH(level)->neighborZ_SP[pos]] * para->getVelocityRatio();
				nodedata[4][nodeCount] = para->getParH(level)->geoSP[para->getParH(level)->neighborZ_SP[pos]];
				//if(para->getParH(level)->geoSP[para->getParH(level)->neighborZ_SP[pos]]==GEO_VOID) neighborsFluid = false;
				nodeCount++;
				//////////////////////////////////////////////////////////////////////////
				//6
				nodes[nodeCount]=( makeUbTuple( (float)(x1P),(float)(x2 ),(float)(x3P) ) );
				nodedata[0][nodeCount] = para->getParH(level)->rho_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborX_SP[pos]]] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[1][nodeCount] = para->getParH(level)->vx_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborX_SP[pos]]] * para->getVelocityRatio();
				nodedata[2][nodeCount] = para->getParH(level)->vy_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborX_SP[pos]]] * para->getVelocityRatio();
				nodedata[3][nodeCount] = para->getParH(level)->vz_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborX_SP[pos]]] * para->getVelocityRatio();
				nodedata[4][nodeCount] = para->getParH(level)->geoSP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborX_SP[pos]]];
				//if(para->getParH(level)->geoSP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborX_SP[pos]]]==GEO_VOID) neighborsFluid = false;
				nodeCount++;
				//////////////////////////////////////////////////////////////////////////
				//7
				nodes[nodeCount]=( makeUbTuple( (float)(x1P),(float)(x2P),(float)(x3P) ) );
				nodedata[0][nodeCount] = para->getParH(level)->rho_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]]] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[1][nodeCount] = para->getParH(level)->vx_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]]] * para->getVelocityRatio();
				nodedata[2][nodeCount] = para->getParH(level)->vy_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]]] * para->getVelocityRatio();
				nodedata[3][nodeCount] = para->getParH(level)->vz_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]]] * para->getVelocityRatio();
				nodedata[4][nodeCount] = para->getParH(level)->geoSP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]]];
				//if(para->getParH(level)->geoSP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[para->getParH(level)->neighborX_SP[pos]]]]==GEO_VOID) neighborsFluid = false;
				nodeCount++;
				//////////////////////////////////////////////////////////////////////////
				//8
				nodes[nodeCount]=( makeUbTuple( (float)(x1 ),(float)(x2P),(float)(x3P) ) );
				nodedata[0][nodeCount] = para->getParH(level)->rho_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[pos]]] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[1][nodeCount] = para->getParH(level)->vx_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[pos]]] * para->getVelocityRatio();
				nodedata[2][nodeCount] = para->getParH(level)->vy_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[pos]]] * para->getVelocityRatio();
				nodedata[3][nodeCount] = para->getParH(level)->vz_SP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[pos]]] * para->getVelocityRatio();
				nodedata[4][nodeCount] = para->getParH(level)->geoSP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[pos]]];
				//if(para->getParH(level)->geoSP[para->getParH(level)->neighborZ_SP[para->getParH(level)->neighborY_SP[pos]]]==GEO_VOID) neighborsFluid = false;
				nodeCount++;

				if(neighborsFluid)
				{
					cells.push_back( makeUbTuple(nodeCount-8,nodeCount-7,nodeCount-6,nodeCount-5,nodeCount-4,nodeCount-3,nodeCount-2,nodeCount-1) );		
				}
			}
		}
		WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname,nodes,cells,nodedatanames,nodedata);
		//WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec2,nodes);
	}
	//////////////////////////////////////////////////////////////////////////



	bool isPeriodicCell(Parameter* para, int level, unsigned int number2, unsigned int number1, unsigned int number3, unsigned int number5)
	{
		return (para->getParH(level)->coordX_SP[number2] < para->getParH(level)->coordX_SP[number1]) ||
			(para->getParH(level)->coordY_SP[number3] < para->getParH(level)->coordY_SP[number1]) ||
			(para->getParH(level)->coordZ_SP[number5] < para->getParH(level)->coordZ_SP[number1]);
	}


	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridLT(Parameter* para, int level, vector<string >& fname) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		//vector< UbTupleUInt8 > cells2;
		vector< string > nodedatanames;
		nodedatanames.push_back("press");
		nodedatanames.push_back("rho");
		nodedatanames.push_back("vx1");
		nodedatanames.push_back("vx2");
		nodedatanames.push_back("vx3");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		unsigned int dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8;
		bool neighborsFluid;
		double vxmax = 0;
		unsigned int startpos = 0;
		unsigned int endpos = 0;
		unsigned int sizeOfNodes = 0;
		vector< vector< double > > nodedata(nodedatanames.size());


		//printf("\n test for if... \n");
		for (unsigned int part=0; part < fname.size(); part++)
		{
			vxmax = 0;
			//printf("\n test in if I... \n");
			//////////////////////////////////////////////////////////////////////////
			if ( ((part+1)*para->getlimitOfNodesForVTK()) > para->getParH(level)->size_Mat_SP)
			{
				sizeOfNodes = para->getParH(level)->size_Mat_SP - (part * para->getlimitOfNodesForVTK());
			}
			else
			{
				sizeOfNodes = para->getlimitOfNodesForVTK();
			}
			//////////////////////////////////////////////////////////////////////////
			startpos = part * para->getlimitOfNodesForVTK();
			endpos = startpos + sizeOfNodes;
			//////////////////////////////////////////////////////////////////////////
			cells.clear();
			nodes.resize(sizeOfNodes);
			nodedata[0].resize(sizeOfNodes);
			nodedata[1].resize(sizeOfNodes);
			nodedata[2].resize(sizeOfNodes);
			nodedata[3].resize(sizeOfNodes);
			nodedata[4].resize(sizeOfNodes);
			nodedata[5].resize(sizeOfNodes);
			//////////////////////////////////////////////////////////////////////////
			//int counter = 0;
			//////////////////////////////////////////////////////////////////////////
			//printf("\n test in if II... \n");

			for (unsigned int pos=startpos;pos<endpos;pos++)
			{
				if (/*para->getParH(level)->geoSP[pos] >= GEO_FLUID*/true)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1  = para->getParH(level)->coordX_SP[pos];
					double x2  = para->getParH(level)->coordY_SP[pos];
					double x3  = para->getParH(level)->coordZ_SP[pos];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					dn1 = pos - startpos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor node data... \n");
					nodes[dn1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
					nodedata[0][dn1] = (double)para->getParH(level)->press_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[1][dn1] = (double)para->getParH(level)->rho_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[2][dn1] = (double)para->getParH(level)->vx_SP[pos] * (double)para->getVelocityRatio();
					nodedata[3][dn1] = (double)para->getParH(level)->vy_SP[pos] * (double)para->getVelocityRatio();
					nodedata[4][dn1] = (double)para->getParH(level)->vz_SP[pos] * (double)para->getVelocityRatio();
					nodedata[5][dn1] = (double)para->getParH(level)->geoSP[pos];
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor numbers... \n");
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor neighborsFluid... \n");
					if (para->getParH(level)->geoSP[number2] < GEO_FLUID ||
						para->getParH(level)->geoSP[number3] < GEO_FLUID ||
						para->getParH(level)->geoSP[number4] < GEO_FLUID ||
						para->getParH(level)->geoSP[number5] < GEO_FLUID ||
						para->getParH(level)->geoSP[number6] < GEO_FLUID ||
						para->getParH(level)->geoSP[number7] < GEO_FLUID ||
						para->getParH(level)->geoSP[number8] < GEO_FLUID )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					//if(neighborsFluid==false) counter++;
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor numbers and neighborsFluid... \n");
					if (number2 > endpos ||
						number3 > endpos ||
						number4 > endpos ||
						number5 > endpos ||
						number6 > endpos ||
						number7 > endpos ||
						number8 > endpos )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					//if(neighborsFluid==false) counter++;
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor dn... \n");
					dn2 = number2 - startpos;
					dn3 = number3 - startpos;
					dn4 = number4 - startpos;
					dn5 = number5 - startpos;
					dn6 = number6 - startpos;
					dn7 = number7 - startpos;
					dn8 = number8 - startpos;
					//////////////////////////////////////////////////////////////////////////
					//if( std::fabs(nodedata[2][dn1]) > std::fabs(vxmax) ) vxmax = nodedata[2][dn1];
					//////////////////////////////////////////////////////////////////////////

					if (isPeriodicCell(para, level, number2, number1, number3, number5))
						continue;

					//counter++;
					if (neighborsFluid) cells.push_back( makeUbTuple(dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8) );
					//////////////////////////////////////////////////////////////////////////
				}
				//printf("\n test II... \n");
			}
			//printf("\n number of cells: %d at level: %d\n", cells.size(), level);
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname[part],nodes,cells,nodedatanames,nodedata);
			//WbWriterVtkXmlBinary::getInstance()->writeNodesWithNodeData(fname[part], nodes, nodedatanames, nodedata);
			//////////////////////////////////////////////////////////////////////////
			//printf("\n vx max: %.1f at level: %d\n", vxmax, level);
			//printf("\n counter: %d at level: %d\n", counter, level);
		} 
	}
	//////////////////////////////////////////////////////////////////////////






	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridLTwithTurbulentViscosity(Parameter* para, int level, vector<string >& fname)
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		//vector< UbTupleUInt8 > cells2;
		vector< string > nodedatanames;
		nodedatanames.push_back("press");
		nodedatanames.push_back("rho");
		nodedatanames.push_back("vx1");
		nodedatanames.push_back("vx2");
		nodedatanames.push_back("vx3");
		nodedatanames.push_back("geo");
		nodedatanames.push_back("turbVis");
		unsigned int number1, number2, number3, number4, number5, number6, number7, number8;
		unsigned int dn1, dn2, dn3, dn4, dn5, dn6, dn7, dn8;
		bool neighborsFluid;
		double vxmax = 0;
		unsigned int startpos = 0;
		unsigned int endpos = 0;
		unsigned int sizeOfNodes = 0;
		vector< vector< double > > nodedata(nodedatanames.size());


		//printf("\n test for if... \n");
		for (unsigned int part = 0; part < fname.size(); part++)
		{
			vxmax = 0;
			//printf("\n test in if I... \n");
			//////////////////////////////////////////////////////////////////////////
			if (((part + 1)*para->getlimitOfNodesForVTK()) > para->getParH(level)->size_Mat_SP)
			{
				sizeOfNodes = para->getParH(level)->size_Mat_SP - (part * para->getlimitOfNodesForVTK());
			}
			else
			{
				sizeOfNodes = para->getlimitOfNodesForVTK();
			}
			//////////////////////////////////////////////////////////////////////////
			startpos = part * para->getlimitOfNodesForVTK();
			endpos = startpos + sizeOfNodes;
			//////////////////////////////////////////////////////////////////////////
			cells.clear();
			nodes.resize(sizeOfNodes);
			nodedata[0].resize(sizeOfNodes);
			nodedata[1].resize(sizeOfNodes);
			nodedata[2].resize(sizeOfNodes);
			nodedata[3].resize(sizeOfNodes);
			nodedata[4].resize(sizeOfNodes);
			nodedata[5].resize(sizeOfNodes);
			nodedata[6].resize(sizeOfNodes);
			//////////////////////////////////////////////////////////////////////////
			//int counter = 0;
			//////////////////////////////////////////////////////////////////////////
			//printf("\n test in if II... \n");

			for (unsigned int pos = startpos; pos < endpos; pos++)
			{
				if (/*para->getParH(level)->geoSP[pos] >= GEO_FLUID*/true)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1 = para->getParH(level)->coordX_SP[pos];
					double x2 = para->getParH(level)->coordY_SP[pos];
					double x3 = para->getParH(level)->coordZ_SP[pos];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					dn1 = pos - startpos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					nodes[dn1] = (makeUbTuple((float)(x1), (float)(x2), (float)(x3)));
					nodedata[0][dn1] = (double)para->getParH(level)->press_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[1][dn1] = (double)para->getParH(level)->rho_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[2][dn1] = (double)para->getParH(level)->vx_SP[pos] * (double)para->getVelocityRatio();
					nodedata[3][dn1] = (double)para->getParH(level)->vy_SP[pos] * (double)para->getVelocityRatio();
					nodedata[4][dn1] = (double)para->getParH(level)->vz_SP[pos] * (double)para->getVelocityRatio();
					nodedata[5][dn1] = (double)para->getParH(level)->geoSP[pos];
					nodedata[6][dn1] = (double)para->getParH(level)->turbViscosity[pos];
					//////////////////////////////////////////////////////////////////////////
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					if (para->getParH(level)->geoSP[number2] < GEO_FLUID ||
						para->getParH(level)->geoSP[number3] < GEO_FLUID ||
						para->getParH(level)->geoSP[number4] < GEO_FLUID ||
						para->getParH(level)->geoSP[number5] < GEO_FLUID ||
						para->getParH(level)->geoSP[number6] < GEO_FLUID ||
						para->getParH(level)->geoSP[number7] < GEO_FLUID ||
						para->getParH(level)->geoSP[number8] < GEO_FLUID)  neighborsFluid = false;
					//////////////////////////////////////////////////////////////////////////
					if (number2 > endpos ||
						number3 > endpos ||
						number4 > endpos ||
						number5 > endpos ||
						number6 > endpos ||
						number7 > endpos ||
						number8 > endpos)  neighborsFluid = false;
					//////////////////////////////////////////////////////////////////////////
					dn2 = number2 - startpos;
					dn3 = number3 - startpos;
					dn4 = number4 - startpos;
					dn5 = number5 - startpos;
					dn6 = number6 - startpos;
					dn7 = number7 - startpos;
					dn8 = number8 - startpos;
					//////////////////////////////////////////////////////////////////////////
					if (isPeriodicCell(para, level, number2, number1, number3, number5))
						continue;
					//////////////////////////////////////////////////////////////////////////
					if (neighborsFluid) cells.push_back(makeUbTuple(dn1, dn2, dn3, dn4, dn5, dn6, dn7, dn8));
					//////////////////////////////////////////////////////////////////////////
				}
			}
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname[part], nodes, cells, nodedatanames, nodedata);
			//WbWriterVtkXmlBinary::getInstance()->writeNodesWithNodeData(fname[part], nodes, nodedatanames, nodedata);
		}
	}
	//////////////////////////////////////////////////////////////////////////






	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridPM(Parameter* para, int level, vector<string >& fname)
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		//vector< UbTupleUInt8 > cells2;
		vector< string > nodedatanames;
		nodedatanames.push_back("press");
		nodedatanames.push_back("rho");
		nodedatanames.push_back("vx1");
		nodedatanames.push_back("vx2");
		nodedatanames.push_back("vx3");
		nodedatanames.push_back("geo");
		unsigned int number1, number2, number3, number4, number5, number6, number7, number8;
		unsigned int dn1, dn2, dn3, dn4, dn5, dn6, dn7, dn8;
		bool neighborsFluid;
		double vxmax = 0;
		unsigned int startpos = 0;
		unsigned int endpos = 0;
		unsigned int sizeOfNodes = 0;
		vector< vector< double > > nodedata(nodedatanames.size());


		//printf("\n test for if... \n");
		for (unsigned int part = 0; part < fname.size(); part++)
		{
			vxmax = 0;
			//printf("\n test in if I... \n");
			//////////////////////////////////////////////////////////////////////////
			if (((part + 1)*para->getlimitOfNodesForVTK()) > para->getParH(level)->size_Mat_SP)
			{
				sizeOfNodes = para->getParH(level)->size_Mat_SP - (part * para->getlimitOfNodesForVTK());
			}
			else
			{
				sizeOfNodes = para->getlimitOfNodesForVTK();
			}
			//////////////////////////////////////////////////////////////////////////
			startpos = part * para->getlimitOfNodesForVTK();
			endpos = startpos + sizeOfNodes;
			//////////////////////////////////////////////////////////////////////////
			cells.clear();
			nodes.resize(sizeOfNodes);
			nodedata[0].resize(sizeOfNodes);
			nodedata[1].resize(sizeOfNodes);
			nodedata[2].resize(sizeOfNodes);
			nodedata[3].resize(sizeOfNodes);
			nodedata[4].resize(sizeOfNodes);
			nodedata[5].resize(sizeOfNodes);
			//////////////////////////////////////////////////////////////////////////
			//int counter = 0;
			//////////////////////////////////////////////////////////////////////////
			//printf("\n test in if II... \n");

			for (unsigned int pos = startpos; pos < endpos; pos++)
			{
				if (/*(para->getParH(level)->geoSP[pos] >= GEO_FLUID) || ((para->getParH(level)->geoSP[pos] >= GEO_PM_0) && (para->getParH(level)->geoSP[pos] <= GEO_PM_2))*/true)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1 = para->getParH(level)->coordX_SP[pos];
					double x2 = para->getParH(level)->coordY_SP[pos];
					double x3 = para->getParH(level)->coordZ_SP[pos];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					dn1 = pos - startpos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor node data... \n");
					nodes[dn1] = (makeUbTuple((float)(x1), (float)(x2), (float)(x3)));
					nodedata[0][dn1] = (double)para->getParH(level)->press_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[1][dn1] = (double)para->getParH(level)->rho_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[2][dn1] = (double)para->getParH(level)->vx_SP[pos] * (double)para->getVelocityRatio();
					nodedata[3][dn1] = (double)para->getParH(level)->vy_SP[pos] * (double)para->getVelocityRatio();
					nodedata[4][dn1] = (double)para->getParH(level)->vz_SP[pos] * (double)para->getVelocityRatio();
					nodedata[5][dn1] = (double)para->getParH(level)->geoSP[pos];
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor numbers... \n");
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor neighborsFluid... \n");
					if (((para->getParH(level)->geoSP[number2] != GEO_FLUID) && (para->getParH(level)->geoSP[number2] < GEO_PM_0) && (para->getParH(level)->geoSP[number2] > GEO_PM_2)) ||
						((para->getParH(level)->geoSP[number3] != GEO_FLUID) && (para->getParH(level)->geoSP[number3] < GEO_PM_0) && (para->getParH(level)->geoSP[number3] > GEO_PM_2)) ||
						((para->getParH(level)->geoSP[number4] != GEO_FLUID) && (para->getParH(level)->geoSP[number4] < GEO_PM_0) && (para->getParH(level)->geoSP[number4] > GEO_PM_2)) ||
						((para->getParH(level)->geoSP[number5] != GEO_FLUID) && (para->getParH(level)->geoSP[number5] < GEO_PM_0) && (para->getParH(level)->geoSP[number5] > GEO_PM_2)) ||
						((para->getParH(level)->geoSP[number6] != GEO_FLUID) && (para->getParH(level)->geoSP[number6] < GEO_PM_0) && (para->getParH(level)->geoSP[number6] > GEO_PM_2)) ||
						((para->getParH(level)->geoSP[number7] != GEO_FLUID) && (para->getParH(level)->geoSP[number7] < GEO_PM_0) && (para->getParH(level)->geoSP[number7] > GEO_PM_2)) ||
						((para->getParH(level)->geoSP[number8] != GEO_FLUID) && (para->getParH(level)->geoSP[number8] < GEO_PM_0) && (para->getParH(level)->geoSP[number8] > GEO_PM_2)))  neighborsFluid = false;
					//////////////////////////////////////////////////////////////////////////
					//if(neighborsFluid==false) counter++;
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor numbers and neighborsFluid... \n");
					if (number2 > endpos ||
						number3 > endpos ||
						number4 > endpos ||
						number5 > endpos ||
						number6 > endpos ||
						number7 > endpos ||
						number8 > endpos)  neighborsFluid = false;
					//////////////////////////////////////////////////////////////////////////
					//if(neighborsFluid==false) counter++;
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test vor dn... \n");
					dn2 = number2 - startpos;
					dn3 = number3 - startpos;
					dn4 = number4 - startpos;
					dn5 = number5 - startpos;
					dn6 = number6 - startpos;
					dn7 = number7 - startpos;
					dn8 = number8 - startpos;
					//////////////////////////////////////////////////////////////////////////
					//if( std::fabs(nodedata[2][dn1]) > std::fabs(vxmax) ) vxmax = nodedata[2][dn1];
					//////////////////////////////////////////////////////////////////////////

					if (isPeriodicCell(para, level, number2, number1, number3, number5))
						continue;

					//counter++;
					if (neighborsFluid) cells.push_back(makeUbTuple(dn1, dn2, dn3, dn4, dn5, dn6, dn7, dn8));
					//////////////////////////////////////////////////////////////////////////
				}
				//printf("\n test II... \n");
			}
			//printf("\n number of cells: %d at level: %d\n", cells.size(), level);
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname[part], nodes, cells, nodedatanames, nodedata);
			//WbWriterVtkXmlBinary::getInstance()->writeNodesWithNodeData(fname[part], nodes, nodedatanames, nodedata);
			//////////////////////////////////////////////////////////////////////////
			//printf("\n vx max: %.1f at level: %d\n", vxmax, level);
			//printf("\n counter: %d at level: %d\n", counter, level);
		}
	}
	//////////////////////////////////////////////////////////////////////////






	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridLTConc(Parameter* para, int level, vector<string >& fname) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		//vector< UbTupleUInt8 > cells2;
		vector< string > nodedatanames;
		nodedatanames.push_back("press");
		nodedatanames.push_back("rho");
		nodedatanames.push_back("vx1");
		nodedatanames.push_back("vx2");
		nodedatanames.push_back("vx3");
		nodedatanames.push_back("geo");
		nodedatanames.push_back("Conc");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		unsigned int dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8;
		bool neighborsFluid;
		double vxmax = 0;
		unsigned int startpos = 0;
		unsigned int endpos = 0;
		unsigned int sizeOfNodes = 0;
		vector< vector< double > > nodedata(nodedatanames.size());

		//printf("\n test for if... \n");
		for (unsigned int part=0; part < fname.size(); part++)
		{
			vxmax = 0;
			//printf("\n test in if I... \n");
			//////////////////////////////////////////////////////////////////////////
			if ( ((part+1)*para->getlimitOfNodesForVTK()) > para->getParH(level)->size_Mat_SP)
			{
				sizeOfNodes = para->getParH(level)->size_Mat_SP - (part * para->getlimitOfNodesForVTK());
			}
			else
			{
				sizeOfNodes = para->getlimitOfNodesForVTK();
			}
			//////////////////////////////////////////////////////////////////////////
			startpos = part * para->getlimitOfNodesForVTK();
			endpos = startpos + sizeOfNodes;
			//////////////////////////////////////////////////////////////////////////
			cells.clear();
			nodes.resize(sizeOfNodes);
			nodedata[0].resize(sizeOfNodes);
			nodedata[1].resize(sizeOfNodes);
			nodedata[2].resize(sizeOfNodes);
			nodedata[3].resize(sizeOfNodes);
			nodedata[4].resize(sizeOfNodes);
			nodedata[5].resize(sizeOfNodes);
			nodedata[6].resize(sizeOfNodes);
			//////////////////////////////////////////////////////////////////////////
			//printf("\n test in if II... \n");
			for (unsigned int pos=startpos;pos<endpos;pos++)
			{
				if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1  = para->getParH(level)->coordX_SP[pos];
					double x2  = para->getParH(level)->coordY_SP[pos];
					double x3  = para->getParH(level)->coordZ_SP[pos];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					dn1 = pos - startpos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					nodes[dn1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
					nodedata[0][dn1] = (double)para->getParH(level)->press_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[1][dn1] = (double)para->getParH(level)->rho_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[2][dn1] = (double)para->getParH(level)->vx_SP[pos] * (double)para->getVelocityRatio();
					nodedata[3][dn1] = (double)para->getParH(level)->vy_SP[pos] * (double)para->getVelocityRatio();
					nodedata[4][dn1] = (double)para->getParH(level)->vz_SP[pos] * (double)para->getVelocityRatio();
					nodedata[5][dn1] = (double)para->getParH(level)->geoSP[pos];
					nodedata[6][dn1] = (double)para->getParH(level)->Conc[pos];
					//////////////////////////////////////////////////////////////////////////
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
						para->getParH(level)->geoSP[number3] != GEO_FLUID ||
						para->getParH(level)->geoSP[number4] != GEO_FLUID ||
						para->getParH(level)->geoSP[number5] != GEO_FLUID ||
						para->getParH(level)->geoSP[number6] != GEO_FLUID ||
						para->getParH(level)->geoSP[number7] != GEO_FLUID ||
						para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					if (number2 > endpos ||
						number3 > endpos ||
						number4 > endpos ||
						number5 > endpos ||
						number6 > endpos ||
						number7 > endpos ||
						number8 > endpos )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					dn2 = number2 - startpos;
					dn3 = number3 - startpos;
					dn4 = number4 - startpos;
					dn5 = number5 - startpos;
					dn6 = number6 - startpos;
					dn7 = number7 - startpos;
					dn8 = number8 - startpos;
					//////////////////////////////////////////////////////////////////////////
					if( std::fabs(nodedata[2][dn1]) > std::fabs(vxmax) ) vxmax = nodedata[2][dn1];
					//////////////////////////////////////////////////////////////////////////
					if (neighborsFluid==true) cells.push_back( makeUbTuple(dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8) );		
					//////////////////////////////////////////////////////////////////////////
				}
			}
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname[part],nodes,cells,nodedatanames,nodedata);
			//////////////////////////////////////////////////////////////////////////
			//printf("\n vx max: %.1f at level: %d\n", vxmax, level);
		} 
	}
	//////////////////////////////////////////////////////////////////////////











	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridBig(Parameter* para, int level, std::string& fname, std::string& fname2) 
	{
		unsigned int limitOfNodes = 30000000; //27 Million
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		vector< UbTupleUInt8 > cells2;
		vector< string > nodedatanames;
		nodedatanames.push_back("press");
		nodedatanames.push_back("rho");
		nodedatanames.push_back("vx1");
		nodedatanames.push_back("vx2");
		nodedatanames.push_back("vx3");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		unsigned int dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8;
		bool neighborsFluid;
		//double posmax = 0;
		double vxmax = 0;
		vector< vector< double > > nodedata(nodedatanames.size());

		//printf("\n test for if... \n");
		if (para->getParH(level)->size_Mat_SP > limitOfNodes)
		{
			//printf("\n test in if I... \n");
			unsigned int restOfNodes = para->getParH(level)->size_Mat_SP - limitOfNodes;
			//////////////////////////////////////////////////////////////////////////
			//PART I
			nodes.resize(limitOfNodes);
			nodedata[0].resize(limitOfNodes);
			nodedata[1].resize(limitOfNodes);
			nodedata[2].resize(limitOfNodes);
			nodedata[3].resize(limitOfNodes);
			nodedata[4].resize(limitOfNodes);
			nodedata[5].resize(limitOfNodes);

			//printf("\n test in if II... \n");
			for (unsigned int pos=0;pos<limitOfNodes;pos++)
			{
				if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1  = para->getParH(level)->coordX_SP[pos];
					double x2  = para->getParH(level)->coordY_SP[pos];
					double x3  = para->getParH(level)->coordZ_SP[pos];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					nodes[number1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
					nodedata[0][number1] = (double)para->getParH(level)->press_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[1][number1] = (double)para->getParH(level)->rho_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[2][number1] = (double)para->getParH(level)->vx_SP[pos] * (double)para->getVelocityRatio();
					nodedata[3][number1] = (double)para->getParH(level)->vy_SP[pos] * (double)para->getVelocityRatio();
					nodedata[4][number1] = (double)para->getParH(level)->vz_SP[pos] * (double)para->getVelocityRatio();
					nodedata[5][number1] = (double)para->getParH(level)->geoSP[pos];
					//////////////////////////////////////////////////////////////////////////
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
						para->getParH(level)->geoSP[number3] != GEO_FLUID ||
						para->getParH(level)->geoSP[number4] != GEO_FLUID ||
						para->getParH(level)->geoSP[number5] != GEO_FLUID ||
						para->getParH(level)->geoSP[number6] != GEO_FLUID ||
						para->getParH(level)->geoSP[number7] != GEO_FLUID ||
						para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					if (number2 > limitOfNodes ||
						number3 > limitOfNodes ||
						number4 > limitOfNodes ||
						number5 > limitOfNodes ||
						number6 > limitOfNodes ||
						number7 > limitOfNodes ||
						number8 > limitOfNodes )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					//if (level == 0 &&
					//	(number2 <= number1 ||
					//	number3 <= number1 ||
					//	number4 <= number1 ||
					//	number5 <= number1 ||
					//	number6 <= number1 ||
					//	number7 <= number1 ||
					//	number8 <= number1) )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					if( std::fabs(nodedata[2][number1]) > std::fabs(vxmax) ) vxmax = nodedata[2][number1];
					//////////////////////////////////////////////////////////////////////////
					if (neighborsFluid==true) cells.push_back( makeUbTuple(number1,number2,number3,number4,number5,number6,number7,number8) );		
					//////////////////////////////////////////////////////////////////////////
				}
			}
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname,nodes,cells,nodedatanames,nodedata);

			//printf("\n test in if III... \n");
			//////////////////////////////////////////////////////////////////////////
			//PART II
			nodes.resize(restOfNodes);
			nodedata[0].resize(restOfNodes);
			nodedata[1].resize(restOfNodes);
			nodedata[2].resize(restOfNodes);
			nodedata[3].resize(restOfNodes);
			nodedata[4].resize(restOfNodes);
			nodedata[5].resize(restOfNodes);
			//printf("\n test in if IV... \n");

			for (unsigned int pos=limitOfNodes;pos<para->getParH(level)->size_Mat_SP;pos++)
			{
				if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1  = para->getParH(level)->coordX_SP[pos];
					double x2  = para->getParH(level)->coordY_SP[pos];
					double x3  = para->getParH(level)->coordZ_SP[pos];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					dn1 = pos - limitOfNodes;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					nodes[dn1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
					nodedata[0][dn1] = (double)para->getParH(level)->press_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[1][dn1] = (double)para->getParH(level)->rho_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[2][dn1] = (double)para->getParH(level)->vx_SP[pos] * (double)para->getVelocityRatio();
					nodedata[3][dn1] = (double)para->getParH(level)->vy_SP[pos] * (double)para->getVelocityRatio();
					nodedata[4][dn1] = (double)para->getParH(level)->vz_SP[pos] * (double)para->getVelocityRatio();
					nodedata[5][dn1] = (double)para->getParH(level)->geoSP[pos];
					//////////////////////////////////////////////////////////////////////////
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
						para->getParH(level)->geoSP[number3] != GEO_FLUID ||
						para->getParH(level)->geoSP[number4] != GEO_FLUID ||
						para->getParH(level)->geoSP[number5] != GEO_FLUID ||
						para->getParH(level)->geoSP[number6] != GEO_FLUID ||
						para->getParH(level)->geoSP[number7] != GEO_FLUID ||
						para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					dn2 = number2 - limitOfNodes;
					dn3 = number3 - limitOfNodes;
					dn4 = number4 - limitOfNodes;
					dn5 = number5 - limitOfNodes;
					dn6 = number6 - limitOfNodes;
					dn7 = number7 - limitOfNodes;
					dn8 = number8 - limitOfNodes;
					//////////////////////////////////////////////////////////////////////////
					if( std::fabs(nodedata[2][dn1]) > std::fabs(vxmax) ) vxmax = nodedata[2][dn1];
					//////////////////////////////////////////////////////////////////////////
					if (neighborsFluid==true) cells2.push_back( makeUbTuple(dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8) );		
					//////////////////////////////////////////////////////////////////////////
				}
			}
			//printf("\n test in if V... \n");
			//WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeDataMS(fname,nodes,cells2,nodedatanames,nodedata);
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname2,nodes,cells2,nodedatanames,nodedata);
			//printf("\n test in if VI... \n");
			//////////////////////////////////////////////////////////////////////////
			//printf("pos max: %.1f", posmax);
			printf("\n vx max: %.1f at level: %d\n", vxmax, level);
		} 
		else
		{
			//printf("\n test in else I... \n");
			nodes.resize(para->getParH(level)->size_Mat_SP);
			nodedata[0].resize(para->getParH(level)->size_Mat_SP);
			nodedata[1].resize(para->getParH(level)->size_Mat_SP);
			nodedata[2].resize(para->getParH(level)->size_Mat_SP);
			nodedata[3].resize(para->getParH(level)->size_Mat_SP);
			nodedata[4].resize(para->getParH(level)->size_Mat_SP);
			nodedata[5].resize(para->getParH(level)->size_Mat_SP);

			//printf("\n test in else II... \n");
			for (unsigned int pos=0;pos<para->getParH(level)->size_Mat_SP;pos++)
			{
				if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
				{
					//printf("\n test in else-for I pos = %d \n", pos);
					//////////////////////////////////////////////////////////////////////////
					double x1  = para->getParH(level)->coordX_SP[pos];
					double x2  = para->getParH(level)->coordY_SP[pos];
					double x3  = para->getParH(level)->coordZ_SP[pos];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test in else-for II pos = %d \n", pos);
					nodes[number1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
					nodedata[0][number1] = (double)para->getParH(level)->press_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[1][number1] = (double)para->getParH(level)->rho_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
					nodedata[2][number1] = (double)para->getParH(level)->vx_SP[pos] * (double)para->getVelocityRatio();
					nodedata[3][number1] = (double)para->getParH(level)->vy_SP[pos] * (double)para->getVelocityRatio();
					nodedata[4][number1] = (double)para->getParH(level)->vz_SP[pos] * (double)para->getVelocityRatio();
					nodedata[5][number1] = (double)para->getParH(level)->geoSP[pos];
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test in else-for III pos = %d \n", pos);
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test in else-for VI pos = %d \n", pos);
					if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
						para->getParH(level)->geoSP[number3] != GEO_FLUID ||
						para->getParH(level)->geoSP[number4] != GEO_FLUID ||
						para->getParH(level)->geoSP[number5] != GEO_FLUID ||
						para->getParH(level)->geoSP[number6] != GEO_FLUID ||
						para->getParH(level)->geoSP[number7] != GEO_FLUID ||
						para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					//if (level == 0 &&
					//	(number2 <= number1 ||
					//	number3 <= number1 ||
					//	number4 <= number1 ||
					//	number5 <= number1 ||
					//	number6 <= number1 ||
					//	number7 <= number1 ||
					//	number8 <= number1) )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					if( std::fabs(nodedata[2][number1]) > std::fabs(vxmax) ) vxmax = nodedata[2][number1];
					//////////////////////////////////////////////////////////////////////////
					//printf("\n test in else-for V pos = %d \n", pos);
					if (neighborsFluid==true) cells.push_back( makeUbTuple(number1,number2,number3,number4,number5,number6,number7,number8) );		
					//////////////////////////////////////////////////////////////////////////
				}
			}
			//printf("\n test in else III... \n");
			//WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeDataMS(fname,nodes,cells,nodedatanames,nodedata);
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname,nodes,cells,nodedatanames,nodedata);
			//WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec2,nodes);
			//printf("\n test in else IV... \n");
			//////////////////////////////////////////////////////////////////////////
			//printf("pos max: %.1f", posmax);
			printf("\n vx max: %.1f at level: %d\n", vxmax, level);
		}
	}
	//////////////////////////////////////////////////////////////////////////











	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridEff(Parameter* para, int level, std::string& fname, std::string& filenameVec2) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		vector< string > nodedatanames;
		nodedatanames.push_back("press");
		nodedatanames.push_back("rho");
		nodedatanames.push_back("vx1");
		nodedatanames.push_back("vx2");
		nodedatanames.push_back("vx3");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		bool neighborsFluid;
		double vxmax = 0;
		vector< vector< double > > nodedata(nodedatanames.size());

		nodes.resize(para->getParH(level)->size_Mat_SP);
		nodedata[0].resize(para->getParH(level)->size_Mat_SP);
		nodedata[1].resize(para->getParH(level)->size_Mat_SP);
		nodedata[2].resize(para->getParH(level)->size_Mat_SP);
		nodedata[3].resize(para->getParH(level)->size_Mat_SP);
		nodedata[4].resize(para->getParH(level)->size_Mat_SP);
		nodedata[5].resize(para->getParH(level)->size_Mat_SP);

		for (unsigned int pos=0;pos<para->getParH(level)->size_Mat_SP;pos++)
		{
			if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
			{
				//////////////////////////////////////////////////////////////////////////
				double x1  = para->getParH(level)->coordX_SP[pos];
				double x2  = para->getParH(level)->coordY_SP[pos];
				double x3  = para->getParH(level)->coordZ_SP[pos];
				double x1P = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];
				double x2P = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];
				double x3P = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];
				//////////////////////////////////////////////////////////////////////////
				number1 = pos;
				neighborsFluid = true;
				//////////////////////////////////////////////////////////////////////////
				nodes[number1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
				nodedata[0][number1] = (double)pos;
				//nodedata[0][number1] = (double)para->getParH(level)->press_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
				nodedata[1][number1] = (double)para->getParH(level)->rho_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
				nodedata[2][number1] = (double)para->getParH(level)->vx_SP[pos] * (double)para->getVelocityRatio();
				nodedata[3][number1] = (double)para->getParH(level)->vy_SP[pos] * (double)para->getVelocityRatio();
				nodedata[4][number1] = (double)para->getParH(level)->vz_SP[pos] * (double)para->getVelocityRatio();
				nodedata[5][number1] = (double)para->getParH(level)->geoSP[pos];
				//////////////////////////////////////////////////////////////////////////
				number2 = para->getParH(level)->neighborX_SP[number1];
				number3 = para->getParH(level)->neighborY_SP[number2];
				number4 = para->getParH(level)->neighborY_SP[number1];
				number5 = para->getParH(level)->neighborZ_SP[number1];
				number6 = para->getParH(level)->neighborZ_SP[number2];
				number7 = para->getParH(level)->neighborZ_SP[number3];
				number8 = para->getParH(level)->neighborZ_SP[number4];
				//////////////////////////////////////////////////////////////////////////
				if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
					para->getParH(level)->geoSP[number3] != GEO_FLUID ||
					para->getParH(level)->geoSP[number4] != GEO_FLUID ||
					para->getParH(level)->geoSP[number5] != GEO_FLUID ||
					para->getParH(level)->geoSP[number6] != GEO_FLUID ||
					para->getParH(level)->geoSP[number7] != GEO_FLUID ||
					para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
				//////////////////////////////////////////////////////////////////////////
				//if (level == 0 &&
				//	(number2 <= number1 ||
				//	number3 <= number1 ||
				//	number4 <= number1 ||
				//	number5 <= number1 ||
				//	number6 <= number1 ||
				//	number7 <= number1 ||
				//	number8 <= number1) )  neighborsFluid=false;
				//////////////////////////////////////////////////////////////////////////
				if (neighborsFluid==true) cells.push_back( makeUbTuple(number1,number2,number3,number4,number5,number6,number7,number8) );		
				//////////////////////////////////////////////////////////////////////////
			}
			if( std::fabs(nodedata[2][number1]) > std::fabs(vxmax) ) vxmax = nodedata[2][number1];
		}
		WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname,nodes,cells,nodedatanames,nodedata);
		printf("\n vx max: %.1f at level: %d\n", vxmax, level);
	}




	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridAsciiEff(Parameter* para, int level, std::string& fname, std::string& filenameVec2) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		vector< string > nodedatanames;
		nodedatanames.push_back("press");
		nodedatanames.push_back("rho");
		nodedatanames.push_back("vx1");
		nodedatanames.push_back("vx2");
		nodedatanames.push_back("vx3");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		bool neighborsFluid;
		double posmax = 0;
		vector< vector< double > > nodedata(nodedatanames.size());

		nodes.resize(para->getParH(level)->size_Mat_SP);
		nodedata[0].resize(para->getParH(level)->size_Mat_SP);
		nodedata[1].resize(para->getParH(level)->size_Mat_SP);
		nodedata[2].resize(para->getParH(level)->size_Mat_SP);
		nodedata[3].resize(para->getParH(level)->size_Mat_SP);
		nodedata[4].resize(para->getParH(level)->size_Mat_SP);
		nodedata[5].resize(para->getParH(level)->size_Mat_SP);

		for (unsigned int pos=0;pos<para->getParH(level)->size_Mat_SP;pos++)
		{
			if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
			{
				//////////////////////////////////////////////////////////////////////////
				double x1  = para->getParH(level)->coordX_SP[pos];
				double x2  = para->getParH(level)->coordY_SP[pos];
				double x3  = para->getParH(level)->coordZ_SP[pos];
				double x1P = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];
				double x2P = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];
				double x3P = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];
				//////////////////////////////////////////////////////////////////////////
				number1 = pos;
				neighborsFluid = true;
				//////////////////////////////////////////////////////////////////////////
				nodes[number1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
				nodedata[0][number1] = (double)pos;
				//nodedata[0][number1] = (double)para->getParH(level)->press_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
				nodedata[1][number1] = (double)para->getParH(level)->rho_SP[pos] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio();
				nodedata[2][number1] = (double)para->getParH(level)->vx_SP[pos] * (double)para->getVelocityRatio();
				nodedata[3][number1] = (double)para->getParH(level)->vy_SP[pos] * (double)para->getVelocityRatio();
				nodedata[4][number1] = (double)para->getParH(level)->vz_SP[pos] * (double)para->getVelocityRatio();
				nodedata[5][number1] = (double)para->getParH(level)->geoSP[pos];
				//////////////////////////////////////////////////////////////////////////
				number2 = para->getParH(level)->neighborX_SP[number1];
				number3 = para->getParH(level)->neighborY_SP[number2];
				number4 = para->getParH(level)->neighborY_SP[number1];
				number5 = para->getParH(level)->neighborZ_SP[number1];
				number6 = para->getParH(level)->neighborZ_SP[number2];
				number7 = para->getParH(level)->neighborZ_SP[number3];
				number8 = para->getParH(level)->neighborZ_SP[number4];
				//////////////////////////////////////////////////////////////////////////
				if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
					para->getParH(level)->geoSP[number3] != GEO_FLUID ||
					para->getParH(level)->geoSP[number4] != GEO_FLUID ||
					para->getParH(level)->geoSP[number5] != GEO_FLUID ||
					para->getParH(level)->geoSP[number6] != GEO_FLUID ||
					para->getParH(level)->geoSP[number7] != GEO_FLUID ||
					para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
				//////////////////////////////////////////////////////////////////////////
				//if (level == 0 &&
				//	(number2 <= number1 ||
				//	number3 <= number1 ||
				//	number4 <= number1 ||
				//	number5 <= number1 ||
				//	number6 <= number1 ||
				//	number7 <= number1 ||
				//	number8 <= number1) )  neighborsFluid=false;
				//////////////////////////////////////////////////////////////////////////
				if (neighborsFluid==true) cells.push_back( makeUbTuple(number1,number2,number3,number4,number5,number6,number7,number8) );		
				//////////////////////////////////////////////////////////////////////////
			}
			if( std::fabs((double)pos) > std::fabs(posmax) ) posmax = (double)pos;
		}
		WbWriterVtkXmlASCII::getInstance()->writeOctsWithNodeData(fname,nodes,cells,nodedatanames,nodedata);
		//WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname,nodes,cells,nodedatanames,nodedata);
		////WbWriterVtkXmlBinary::getInstance()->writeNodes(filenameVec2,nodes);
		printf("\ncells: %.1f \n", (double)cells.size());
		printf("nodes: %.1f \n", (double)nodes.size());
		printf("pos max: %.1f \n", posmax);
	}
	//////////////////////////////////////////////////////////////////////////








	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridMedianLT(Parameter* para, int level, vector<string >& fname) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		//vector< UbTupleUInt8 > cells2;
		vector< string > nodedatanames;
		nodedatanames.push_back("pressMed");
		nodedatanames.push_back("rhoMed");
		nodedatanames.push_back("vx1Med");
		nodedatanames.push_back("vx2Med");
		nodedatanames.push_back("vx3Med");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		unsigned int dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8;
		bool neighborsFluid;
		double vxmax = 0;
		unsigned int startpos = 0;
		unsigned int endpos = 0;
		unsigned int sizeOfNodes = 0;
		vector< vector< double > > nodedata(nodedatanames.size());

		//printf("\n test for if... \n");
		for (unsigned int part=0; part < fname.size(); part++)
		{
			vxmax = 0;
			//printf("\n test in if I... \n");
			//////////////////////////////////////////////////////////////////////////
			if ( ((part+1)*para->getlimitOfNodesForVTK()) > para->getParH(level)->size_Mat_SP)
			{
				sizeOfNodes = para->getParH(level)->size_Mat_SP - (part * para->getlimitOfNodesForVTK());
			}
			else
			{
				sizeOfNodes = para->getlimitOfNodesForVTK();
			}
			//////////////////////////////////////////////////////////////////////////
			startpos = part * para->getlimitOfNodesForVTK();
			endpos = startpos + sizeOfNodes;
			//////////////////////////////////////////////////////////////////////////
			cells.clear();
			nodes.resize(sizeOfNodes);
			nodedata[0].resize(sizeOfNodes);
			nodedata[1].resize(sizeOfNodes);
			nodedata[2].resize(sizeOfNodes);
			nodedata[3].resize(sizeOfNodes);
			nodedata[4].resize(sizeOfNodes);
			nodedata[5].resize(sizeOfNodes);
			//////////////////////////////////////////////////////////////////////////
			//printf("\n test in if II... \n");
			for (unsigned int pos=startpos;pos<endpos;pos++)
			{
				if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1  = para->getParH(level)->coordX_SP[pos];
					double x2  = para->getParH(level)->coordY_SP[pos];
					double x3  = para->getParH(level)->coordZ_SP[pos];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					dn1 = pos - startpos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					nodes[dn1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
					nodedata[0][dn1] = para->getParH(level)->press_SP_Med_Out[pos] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
					nodedata[1][dn1] = para->getParH(level)->rho_SP_Med_Out[pos] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
					nodedata[2][dn1] = para->getParH(level)->vx_SP_Med_Out[pos] * para->getVelocityRatio();
					nodedata[3][dn1] = para->getParH(level)->vy_SP_Med_Out[pos] * para->getVelocityRatio();
					nodedata[4][dn1] = para->getParH(level)->vz_SP_Med_Out[pos] * para->getVelocityRatio();
					nodedata[5][dn1] = (double)para->getParH(level)->geoSP[pos];
					//////////////////////////////////////////////////////////////////////////
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
						para->getParH(level)->geoSP[number3] != GEO_FLUID ||
						para->getParH(level)->geoSP[number4] != GEO_FLUID ||
						para->getParH(level)->geoSP[number5] != GEO_FLUID ||
						para->getParH(level)->geoSP[number6] != GEO_FLUID ||
						para->getParH(level)->geoSP[number7] != GEO_FLUID ||
						para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					if (number2 > endpos ||
						number3 > endpos ||
						number4 > endpos ||
						number5 > endpos ||
						number6 > endpos ||
						number7 > endpos ||
						number8 > endpos )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					dn2 = number2 - startpos;
					dn3 = number3 - startpos;
					dn4 = number4 - startpos;
					dn5 = number5 - startpos;
					dn6 = number6 - startpos;
					dn7 = number7 - startpos;
					dn8 = number8 - startpos;
					//////////////////////////////////////////////////////////////////////////
					if( std::fabs(nodedata[2][dn1]) > std::fabs(vxmax) ) vxmax = nodedata[2][dn1];
					//////////////////////////////////////////////////////////////////////////
					if (neighborsFluid==true) cells.push_back( makeUbTuple(dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8) );		
					//////////////////////////////////////////////////////////////////////////
				}
			}
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname[part],nodes,cells,nodedatanames,nodedata);
			//////////////////////////////////////////////////////////////////////////
			printf("\n vx median max: %.1f at level: %d\n", vxmax, level);
		} 
	}
	//////////////////////////////////////////////////////////////////////////







	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridEffMedian(Parameter* para, int level, std::string& fname) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		vector< string > nodedatanames;
		nodedatanames.push_back("pressMed");
		nodedatanames.push_back("rhoMed");
		nodedatanames.push_back("vx1Med");
		nodedatanames.push_back("vx2Med");
		nodedatanames.push_back("vx3Med");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		bool neighborsFluid;
		vector< vector< double > > nodedata(nodedatanames.size());

		nodes.resize(para->getParH(level)->size_Mat_SP);
		nodedata[0].resize(para->getParH(level)->size_Mat_SP);
		nodedata[1].resize(para->getParH(level)->size_Mat_SP);
		nodedata[2].resize(para->getParH(level)->size_Mat_SP);
		nodedata[3].resize(para->getParH(level)->size_Mat_SP);
		nodedata[4].resize(para->getParH(level)->size_Mat_SP);
		nodedata[5].resize(para->getParH(level)->size_Mat_SP);

		for (unsigned int pos=0;pos<para->getParH(level)->size_Mat_SP;pos++)
		{
			if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
			{
				//////////////////////////////////////////////////////////////////////////
				double x1  = para->getParH(level)->coordX_SP[pos];
				double x2  = para->getParH(level)->coordY_SP[pos];
				double x3  = para->getParH(level)->coordZ_SP[pos];
				double x1P = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];
				double x2P = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];
				double x3P = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];
				//////////////////////////////////////////////////////////////////////////
				number1 = pos;
				neighborsFluid = true;
				//////////////////////////////////////////////////////////////////////////
				nodes[number1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
				nodedata[0][number1] = para->getParH(level)->press_SP_Med_Out[pos] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[1][number1] = para->getParH(level)->rho_SP_Med_Out[pos] / 3.0f * para->getDensityRatio() * para->getVelocityRatio() * para->getVelocityRatio();
				nodedata[2][number1] = para->getParH(level)->vx_SP_Med_Out[pos] * para->getVelocityRatio();
				nodedata[3][number1] = para->getParH(level)->vy_SP_Med_Out[pos] * para->getVelocityRatio();
				nodedata[4][number1] = para->getParH(level)->vz_SP_Med_Out[pos] * para->getVelocityRatio();
				nodedata[5][number1] = para->getParH(level)->geoSP[pos];
				//////////////////////////////////////////////////////////////////////////
				number2 = para->getParH(level)->neighborX_SP[number1];
				number3 = para->getParH(level)->neighborY_SP[number2];
				number4 = para->getParH(level)->neighborY_SP[number1];
				number5 = para->getParH(level)->neighborZ_SP[number1];
				number6 = para->getParH(level)->neighborZ_SP[number2];
				number7 = para->getParH(level)->neighborZ_SP[number3];
				number8 = para->getParH(level)->neighborZ_SP[number4];
				//////////////////////////////////////////////////////////////////////////
				if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
					para->getParH(level)->geoSP[number3] != GEO_FLUID ||
					para->getParH(level)->geoSP[number4] != GEO_FLUID ||
					para->getParH(level)->geoSP[number5] != GEO_FLUID ||
					para->getParH(level)->geoSP[number6] != GEO_FLUID ||
					para->getParH(level)->geoSP[number7] != GEO_FLUID ||
					para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
				//////////////////////////////////////////////////////////////////////////
				//if (level == 0 &&
				//	(number2 <= number1 ||
				//	number3 <= number1 ||
				//	number4 <= number1 ||
				//	number5 <= number1 ||
				//	number6 <= number1 ||
				//	number7 <= number1 ||
				//	number8 <= number1) )  neighborsFluid=false;
				//////////////////////////////////////////////////////////////////////////
				if (neighborsFluid==true) cells.push_back( makeUbTuple(number1,number2,number3,number4,number5,number6,number7,number8) );		
				//////////////////////////////////////////////////////////////////////////
			}
		}
		WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname,nodes,cells,nodedatanames,nodedata);
	}
	//////////////////////////////////////////////////////////////////////////








	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridEff2ndMoments(Parameter* para, int level, std::string& fname) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		vector< string > nodedatanames;
		nodedatanames.push_back("kxyFromfcNEQ");
		nodedatanames.push_back("kyzFromfcNEQ");
		nodedatanames.push_back("kxzFromfcNEQ");
		nodedatanames.push_back("kxxMyyFromfcNEQ");
		nodedatanames.push_back("kxxMzzFromfcNEQ");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		bool neighborsFluid;
		vector< vector< double > > nodedata(nodedatanames.size());

		nodes.resize(para->getParH(level)->size_Mat_SP);
		nodedata[0].resize(para->getParH(level)->size_Mat_SP);
		nodedata[1].resize(para->getParH(level)->size_Mat_SP);
		nodedata[2].resize(para->getParH(level)->size_Mat_SP);
		nodedata[3].resize(para->getParH(level)->size_Mat_SP);
		nodedata[4].resize(para->getParH(level)->size_Mat_SP);
		nodedata[5].resize(para->getParH(level)->size_Mat_SP);

		for (unsigned int pos=0;pos<para->getParH(level)->size_Mat_SP;pos++)
		{
			if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
			{
				//////////////////////////////////////////////////////////////////////////
				double x1  = para->getParH(level)->coordX_SP[pos];
				double x2  = para->getParH(level)->coordY_SP[pos];
				double x3  = para->getParH(level)->coordZ_SP[pos];
				double x1P = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];
				double x2P = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];
				double x3P = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];
				//////////////////////////////////////////////////////////////////////////
				number1 = pos;
				neighborsFluid = true;
				//////////////////////////////////////////////////////////////////////////
				nodes[number1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
				nodedata[0][number1] = para->getParH(level)->kxyFromfcNEQ[pos];
				nodedata[1][number1] = para->getParH(level)->kyzFromfcNEQ[pos];
				nodedata[2][number1] = para->getParH(level)->kxzFromfcNEQ[pos];
				nodedata[3][number1] = para->getParH(level)->kxxMyyFromfcNEQ[pos];
				nodedata[4][number1] = para->getParH(level)->kxxMzzFromfcNEQ[pos];
				nodedata[5][number1] = para->getParH(level)->geoSP[pos];
				//////////////////////////////////////////////////////////////////////////
				number2 = para->getParH(level)->neighborX_SP[number1];
				number3 = para->getParH(level)->neighborY_SP[number2];
				number4 = para->getParH(level)->neighborY_SP[number1];
				number5 = para->getParH(level)->neighborZ_SP[number1];
				number6 = para->getParH(level)->neighborZ_SP[number2];
				number7 = para->getParH(level)->neighborZ_SP[number3];
				number8 = para->getParH(level)->neighborZ_SP[number4];
				//////////////////////////////////////////////////////////////////////////
				if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
					para->getParH(level)->geoSP[number3] != GEO_FLUID ||
					para->getParH(level)->geoSP[number4] != GEO_FLUID ||
					para->getParH(level)->geoSP[number5] != GEO_FLUID ||
					para->getParH(level)->geoSP[number6] != GEO_FLUID ||
					para->getParH(level)->geoSP[number7] != GEO_FLUID ||
					para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
				//////////////////////////////////////////////////////////////////////////
				if (neighborsFluid==true) cells.push_back( makeUbTuple(number1,number2,number3,number4,number5,number6,number7,number8) );		
				//////////////////////////////////////////////////////////////////////////
			}
		}
		WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname,nodes,cells,nodedatanames,nodedata);
	}
	//////////////////////////////////////////////////////////////////////////













	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredParticles(Parameter* para, int level, std::string& fname) 
	{
		vector< UbTupleFloat3 > particlePosition;
		vector< string > particleDataNames;
		particleDataNames.push_back("ID");
		particleDataNames.push_back("timestep");
		particleDataNames.push_back("particleVx");
		particleDataNames.push_back("particleVy");
		particleDataNames.push_back("particleVz");
		//particleDataNames.push_back("tauxx");
		//particleDataNames.push_back("tauyy");
		//particleDataNames.push_back("tauzz");
		//particleDataNames.push_back("tauxy");
		//particleDataNames.push_back("tauxz");
		//particleDataNames.push_back("tauyz");
		vector< vector< double > > particleData(particleDataNames.size());
		//////////////////////////////////////////////////////////////////////////
		unsigned int numberOfParticles  = para->getParH(level)->plp.numberOfParticles;
		unsigned int timestep           = para->getParH(level)->plp.numberOfTimestepsParticles;
		//////////////////////////////////////////////////////////////////////////
		unsigned int pos = 0;
		unsigned int sizeOfNodes = numberOfParticles * timestep;
		//////////////////////////////////////////////////////////////////////////
		particlePosition.resize(sizeOfNodes);
		for (unsigned int i = 0; i < particleDataNames.size(); i++)
		{
			particleData[i].resize(sizeOfNodes);
		}
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int i = 0; i < timestep; i++)
		{
			for (unsigned int j = 0; j < numberOfParticles; j++)
			{
				//////////////////////////////////////////////////////////////////////////
				double x1  = (double)para->getParH(level)->plp.coordXabsolut[pos];
				double x2  = (double)para->getParH(level)->plp.coordYabsolut[pos];
				double x3  = (double)para->getParH(level)->plp.coordZabsolut[pos];
				//////////////////////////////////////////////////////////////////////////
				particlePosition[pos]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
				particleData[0][pos] = (double)para->getParH(level)->plp.ID[j];
				particleData[1][pos] = (double)para->getParH(level)->plp.timestep[i];
				particleData[2][pos] = (double)para->getParH(level)->plp.veloX[pos];
				particleData[3][pos] = (double)para->getParH(level)->plp.veloY[pos];
				particleData[4][pos] = (double)para->getParH(level)->plp.veloZ[pos];
				//////////////////////////////////////////////////////////////////////////
				pos++;
			}
		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		WbWriterVtkXmlBinary::getInstance()->writeNodesWithNodeData(fname,particlePosition,particleDataNames,particleData);
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	} 
	//////////////////////////////////////////////////////////////////////////









	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridEff2ndMomentsLT(Parameter* para, int level, vector<string >& fname) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		//vector< UbTupleUInt8 > cells2;
		vector< string > nodedatanames;
		nodedatanames.push_back("kxyFromfcNEQ");
		nodedatanames.push_back("kyzFromfcNEQ");
		nodedatanames.push_back("kxzFromfcNEQ");
		nodedatanames.push_back("kxxMyyFromfcNEQ");
		nodedatanames.push_back("kxxMzzFromfcNEQ");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		unsigned int dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8;
		bool neighborsFluid;
		double vxmax = 0;
		unsigned int startpos = 0;
		unsigned int endpos = 0;
		unsigned int sizeOfNodes = 0;
		vector< vector< double > > nodedata(nodedatanames.size());

		//printf("\n test for if... \n");
		for (unsigned int part=0; part < fname.size(); part++)
		{
			vxmax = 0;
			//printf("\n test in if I... \n");
			//////////////////////////////////////////////////////////////////////////
			if ( ((part+1)*para->getlimitOfNodesForVTK()) > para->getParH(level)->size_Mat_SP)
			{
				sizeOfNodes = para->getParH(level)->size_Mat_SP - (part * para->getlimitOfNodesForVTK());
			}
			else
			{
				sizeOfNodes = para->getlimitOfNodesForVTK();
			}
			//////////////////////////////////////////////////////////////////////////
			startpos = part * para->getlimitOfNodesForVTK();
			endpos = startpos + sizeOfNodes;
			//////////////////////////////////////////////////////////////////////////
			cells.clear();
			nodes.resize(sizeOfNodes);
			nodedata[0].resize(sizeOfNodes);
			nodedata[1].resize(sizeOfNodes);
			nodedata[2].resize(sizeOfNodes);
			nodedata[3].resize(sizeOfNodes);
			nodedata[4].resize(sizeOfNodes);
			nodedata[5].resize(sizeOfNodes);
			//////////////////////////////////////////////////////////////////////////
			//printf("\n test in if II... \n");
			for (unsigned int pos=startpos;pos<endpos;pos++)
			{
				if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1  = para->getParH(level)->coordX_SP[pos];
					double x2  = para->getParH(level)->coordY_SP[pos];
					double x3  = para->getParH(level)->coordZ_SP[pos];
					double x1P = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];
					double x2P = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];
					double x3P = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					nodes[number1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
					nodedata[0][number1] = para->getParH(level)->kxyFromfcNEQ[pos];
					nodedata[1][number1] = para->getParH(level)->kyzFromfcNEQ[pos];
					nodedata[2][number1] = para->getParH(level)->kxzFromfcNEQ[pos];
					nodedata[3][number1] = para->getParH(level)->kxxMyyFromfcNEQ[pos];
					nodedata[4][number1] = para->getParH(level)->kxxMzzFromfcNEQ[pos];
					nodedata[5][number1] = para->getParH(level)->geoSP[pos];
					//////////////////////////////////////////////////////////////////////////
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
						para->getParH(level)->geoSP[number3] != GEO_FLUID ||
						para->getParH(level)->geoSP[number4] != GEO_FLUID ||
						para->getParH(level)->geoSP[number5] != GEO_FLUID ||
						para->getParH(level)->geoSP[number6] != GEO_FLUID ||
						para->getParH(level)->geoSP[number7] != GEO_FLUID ||
						para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					if (neighborsFluid==true) cells.push_back( makeUbTuple(number1,number2,number3,number4,number5,number6,number7,number8) );		
					//////////////////////////////////////////////////////////////////////////
				}
			}
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname[part],nodes,cells,nodedatanames,nodedata);
		} 
	}
	//////////////////////////////////////////////////////////////////////////






	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridEff3rdMomentsLT(Parameter* para, int level, vector<string >& fname) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		//vector< UbTupleUInt8 > cells2;
		vector< string > nodedatanames;
		nodedatanames.push_back("CUMbbb");
		nodedatanames.push_back("CUMabc");
		nodedatanames.push_back("CUMbac");
		nodedatanames.push_back("CUMbca");
		nodedatanames.push_back("CUMcba");
		nodedatanames.push_back("CUMacb");
		nodedatanames.push_back("CUMcab");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		unsigned int dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8;
		bool neighborsFluid;
		double vxmax = 0;
		unsigned int startpos = 0;
		unsigned int endpos = 0;
		unsigned int sizeOfNodes = 0;
		vector< vector< double > > nodedata(nodedatanames.size());

		//printf("\n test for if... \n");
		for (unsigned int part=0; part < fname.size(); part++)
		{
			vxmax = 0;
			//printf("\n test in if I... \n");
			//////////////////////////////////////////////////////////////////////////
			if ( ((part+1)*para->getlimitOfNodesForVTK()) > para->getParH(level)->size_Mat_SP)
			{
				sizeOfNodes = para->getParH(level)->size_Mat_SP - (part * para->getlimitOfNodesForVTK());
			}
			else
			{
				sizeOfNodes = para->getlimitOfNodesForVTK();
			}
			//////////////////////////////////////////////////////////////////////////
			startpos = part * para->getlimitOfNodesForVTK();
			endpos = startpos + sizeOfNodes;
			//////////////////////////////////////////////////////////////////////////
			cells.clear();
			nodes.resize(sizeOfNodes);
			nodedata[0].resize(sizeOfNodes);
			nodedata[1].resize(sizeOfNodes);
			nodedata[2].resize(sizeOfNodes);
			nodedata[3].resize(sizeOfNodes);
			nodedata[4].resize(sizeOfNodes);
			nodedata[5].resize(sizeOfNodes);
			nodedata[6].resize(sizeOfNodes);
			nodedata[7].resize(sizeOfNodes);
			//////////////////////////////////////////////////////////////////////////
			//printf("\n test in if II... \n");
			for (unsigned int pos=startpos;pos<endpos;pos++)
			{
				if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1  = para->getParH(level)->coordX_SP[pos];
					double x2  = para->getParH(level)->coordY_SP[pos];
					double x3  = para->getParH(level)->coordZ_SP[pos];
					double x1P = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];
					double x2P = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];
					double x3P = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					nodes[number1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
					nodedata[0][number1] = para->getParH(level)->CUMbbb[pos];
					nodedata[1][number1] = para->getParH(level)->CUMabc[pos];
					nodedata[2][number1] = para->getParH(level)->CUMbac[pos];
					nodedata[3][number1] = para->getParH(level)->CUMbca[pos];
					nodedata[4][number1] = para->getParH(level)->CUMcba[pos];
					nodedata[5][number1] = para->getParH(level)->CUMacb[pos];
					nodedata[6][number1] = para->getParH(level)->CUMcab[pos];
					nodedata[7][number1] = para->getParH(level)->geoSP[pos];
					//////////////////////////////////////////////////////////////////////////
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
						para->getParH(level)->geoSP[number3] != GEO_FLUID ||
						para->getParH(level)->geoSP[number4] != GEO_FLUID ||
						para->getParH(level)->geoSP[number5] != GEO_FLUID ||
						para->getParH(level)->geoSP[number6] != GEO_FLUID ||
						para->getParH(level)->geoSP[number7] != GEO_FLUID ||
						para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					if (neighborsFluid==true) cells.push_back( makeUbTuple(number1,number2,number3,number4,number5,number6,number7,number8) );		
					//////////////////////////////////////////////////////////////////////////
				}
			}
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname[part],nodes,cells,nodedatanames,nodedata);
		} 
	}
	//////////////////////////////////////////////////////////////////////////






	//////////////////////////////////////////////////////////////////////////
	void writeUnstrucuredGridEffHigherMomentsLT(Parameter* para, int level, vector<string >& fname) 
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleUInt8 > cells;
		//vector< UbTupleUInt8 > cells2;
		vector< string > nodedatanames;
		nodedatanames.push_back("CUMcbb");
		nodedatanames.push_back("CUMbcb");
		nodedatanames.push_back("CUMbbc");
		nodedatanames.push_back("CUMcca");
		nodedatanames.push_back("CUMcac");
		nodedatanames.push_back("CUMacc");
		nodedatanames.push_back("CUMbcc");
		nodedatanames.push_back("CUMcbc");
		nodedatanames.push_back("CUMccb");
		nodedatanames.push_back("CUMccc");
		nodedatanames.push_back("geo");
		unsigned int number1,number2,number3,number4,number5,number6,number7,number8;
		unsigned int dn1,dn2,dn3,dn4,dn5,dn6,dn7,dn8;
		bool neighborsFluid;
		double vxmax = 0;
		unsigned int startpos = 0;
		unsigned int endpos = 0;
		unsigned int sizeOfNodes = 0;
		vector< vector< double > > nodedata(nodedatanames.size());

		//printf("\n test for if... \n");
		for (unsigned int part=0; part < fname.size(); part++)
		{
			vxmax = 0;
			//printf("\n test in if I... \n");
			//////////////////////////////////////////////////////////////////////////
			if ( ((part+1)*para->getlimitOfNodesForVTK()) > para->getParH(level)->size_Mat_SP)
			{
				sizeOfNodes = para->getParH(level)->size_Mat_SP - (part * para->getlimitOfNodesForVTK());
			}
			else
			{
				sizeOfNodes = para->getlimitOfNodesForVTK();
			}
			//////////////////////////////////////////////////////////////////////////
			startpos = part * para->getlimitOfNodesForVTK();
			endpos = startpos + sizeOfNodes;
			//////////////////////////////////////////////////////////////////////////
			cells.clear();
			nodes.resize(sizeOfNodes);
			nodedata[0].resize(sizeOfNodes);
			nodedata[1].resize(sizeOfNodes);
			nodedata[2].resize(sizeOfNodes);
			nodedata[3].resize(sizeOfNodes);
			nodedata[4].resize(sizeOfNodes);
			nodedata[5].resize(sizeOfNodes);
			nodedata[6].resize(sizeOfNodes);
			nodedata[7].resize(sizeOfNodes);
			nodedata[8].resize(sizeOfNodes);
			nodedata[9].resize(sizeOfNodes);
			nodedata[10].resize(sizeOfNodes);
			//////////////////////////////////////////////////////////////////////////
			//printf("\n test in if II... \n");
			for (unsigned int pos=startpos;pos<endpos;pos++)
			{
				if (para->getParH(level)->geoSP[pos] == GEO_FLUID)
				{
					//////////////////////////////////////////////////////////////////////////
					double x1  = para->getParH(level)->coordX_SP[pos];
					double x2  = para->getParH(level)->coordY_SP[pos];
					double x3  = para->getParH(level)->coordZ_SP[pos];
					double x1P = para->getParH(level)->coordX_SP[para->getParH(level)->neighborX_SP[pos]];
					double x2P = para->getParH(level)->coordY_SP[para->getParH(level)->neighborY_SP[pos]];
					double x3P = para->getParH(level)->coordZ_SP[para->getParH(level)->neighborZ_SP[pos]];
					//////////////////////////////////////////////////////////////////////////
					number1 = pos;
					neighborsFluid = true;
					//////////////////////////////////////////////////////////////////////////
					nodes[number1]=( makeUbTuple( (float)(x1 ),(float)(x2 ),(float)(x3 ) ) );
					nodedata[0][number1] = para->getParH(level)->CUMcbb[pos];
					nodedata[1][number1] = para->getParH(level)->CUMbcb[pos];
					nodedata[2][number1] = para->getParH(level)->CUMbbc[pos];
					nodedata[3][number1] = para->getParH(level)->CUMcca[pos];
					nodedata[4][number1] = para->getParH(level)->CUMcac[pos];
					nodedata[5][number1] = para->getParH(level)->CUMacc[pos];
					nodedata[6][number1] = para->getParH(level)->CUMbcc[pos];
					nodedata[7][number1] = para->getParH(level)->CUMcbc[pos];
					nodedata[8][number1] = para->getParH(level)->CUMccb[pos];
					nodedata[9][number1] = para->getParH(level)->CUMccc[pos];
					nodedata[10][number1] = para->getParH(level)->geoSP[pos];
					//////////////////////////////////////////////////////////////////////////
					number2 = para->getParH(level)->neighborX_SP[number1];
					number3 = para->getParH(level)->neighborY_SP[number2];
					number4 = para->getParH(level)->neighborY_SP[number1];
					number5 = para->getParH(level)->neighborZ_SP[number1];
					number6 = para->getParH(level)->neighborZ_SP[number2];
					number7 = para->getParH(level)->neighborZ_SP[number3];
					number8 = para->getParH(level)->neighborZ_SP[number4];
					//////////////////////////////////////////////////////////////////////////
					if (para->getParH(level)->geoSP[number2] != GEO_FLUID ||
						para->getParH(level)->geoSP[number3] != GEO_FLUID ||
						para->getParH(level)->geoSP[number4] != GEO_FLUID ||
						para->getParH(level)->geoSP[number5] != GEO_FLUID ||
						para->getParH(level)->geoSP[number6] != GEO_FLUID ||
						para->getParH(level)->geoSP[number7] != GEO_FLUID ||
						para->getParH(level)->geoSP[number8] != GEO_FLUID )  neighborsFluid=false;
					//////////////////////////////////////////////////////////////////////////
					if (neighborsFluid==true) cells.push_back( makeUbTuple(number1,number2,number3,number4,number5,number6,number7,number8) );		
					//////////////////////////////////////////////////////////////////////////
				}
			}
			WbWriterVtkXmlBinary::getInstance()->writeOctsWithNodeData(fname[part],nodes,cells,nodedatanames,nodedata);
		} 
	}
	//////////////////////////////////////////////////////////////////////////






	//////////////////////////////////////////////////////////////////////////
	void writeQs(Parameter* para, int level, std::string& fname)
	{
		vector< UbTupleFloat3 > nodes;
		vector< UbTupleInt2 > qs;
		unsigned int startpos = 0;
		unsigned int endpos = 0;
		unsigned int sizeOfNodes = 0;
		int node = 0;
		int wall = 1;
		int line = 0;
		double dx = 1.0 / pow(2, level);
		real* QQ;
		QforBoundaryConditions Q;
		double nodeX1, nodeX2, nodeX3, wallX1, wallX2, wallX3, q;
		//////////////////////////////////////////////////////////////////////////
		sizeOfNodes = para->getParH(level)->QGeom.kQ;
		endpos = startpos + sizeOfNodes;
		//////////////////////////////////////////////////////////////////////////
		//qs.clear();
		//nodes.clear();
		unsigned int numberOfLines = para->getD3Qxx() * sizeOfNodes;
		unsigned int numberOfNodes = numberOfLines * 2;
		qs.resize(numberOfLines);
		nodes.resize(numberOfNodes);
		//////////////////////////////////////////////////////////////////////////
		vector< string > nodedatanames;
		nodedatanames.push_back("sizeQ");
		vector< vector< double > > nodedata(nodedatanames.size());
		nodedata[0].resize(numberOfNodes);
		//////////////////////////////////////////////////////////////////////////
		for (unsigned int pos = startpos; pos < endpos; pos++)
		{
			//////////////////////////////////////////////////////////////////////////
			nodeX1 = para->getParH(level)->coordX_SP[para->getParH(level)->QGeom.k[pos]];
			nodeX2 = para->getParH(level)->coordY_SP[para->getParH(level)->QGeom.k[pos]];
			nodeX3 = para->getParH(level)->coordZ_SP[para->getParH(level)->QGeom.k[pos]];
			wallX1 = 0.0;
			wallX2 = 0.0;
			wallX3 = 0.0;
			q      = 0.0;
			//////////////////////////////////////////////////////////////////////////
			for (unsigned int typeOfQ = dirSTART; typeOfQ <= dirEND; typeOfQ++)
			{
				QQ = para->getParH(level)->QGeom.q27[0];
				Q.q27[typeOfQ] = &QQ[typeOfQ*sizeOfNodes];
				q = (double)(Q.q27[typeOfQ][pos]);
				//////////////////////////////////////////////////////////////////////////
				switch (typeOfQ)
				{
					case dirE:   wallX1 = nodeX1 + q*dx; wallX2 = nodeX2;        wallX3 = nodeX3;        break;
					case dirN:   wallX1 = nodeX1;        wallX2 = nodeX2 + q*dx; wallX3 = nodeX3;        break;
					case dirW:   wallX1 = nodeX1 - q*dx; wallX2 = nodeX2;        wallX3 = nodeX3;        break;
					case dirS:   wallX1 = nodeX1;        wallX2 = nodeX2 - q*dx; wallX3 = nodeX3;        break;
					case dirNE:  wallX1 = nodeX1 + q*dx; wallX2 = nodeX2 + q*dx; wallX3 = nodeX3;        break;
					case dirNW:  wallX1 = nodeX1 - q*dx; wallX2 = nodeX2 + q*dx; wallX3 = nodeX3;        break;
					case dirSW:  wallX1 = nodeX1 - q*dx; wallX2 = nodeX2 - q*dx; wallX3 = nodeX3;        break;
					case dirSE:  wallX1 = nodeX1 + q*dx; wallX2 = nodeX2 - q*dx; wallX3 = nodeX3;        break;
					case dirT:   wallX1 = nodeX1;        wallX2 = nodeX2;        wallX3 = nodeX3 + q*dx; break;
					case dirTE:  wallX1 = nodeX1 + q*dx; wallX2 = nodeX2;        wallX3 = nodeX3 + q*dx; break;
					case dirTN:  wallX1 = nodeX1;        wallX2 = nodeX2 + q*dx; wallX3 = nodeX3 + q*dx; break;
					case dirTW:  wallX1 = nodeX1 - q*dx; wallX2 = nodeX2;        wallX3 = nodeX3 + q*dx; break;
					case dirTS:  wallX1 = nodeX1;        wallX2 = nodeX2 - q*dx; wallX3 = nodeX3 + q*dx; break;
					case dirB:   wallX1 = nodeX1;        wallX2 = nodeX2;        wallX3 = nodeX3 - q*dx; break;
					case dirBE:  wallX1 = nodeX1 + q*dx; wallX2 = nodeX2;        wallX3 = nodeX3 - q*dx; break;
					case dirBN:  wallX1 = nodeX1;        wallX2 = nodeX2 + q*dx; wallX3 = nodeX3 - q*dx; break;
					case dirBW:  wallX1 = nodeX1 - q*dx; wallX2 = nodeX2;        wallX3 = nodeX3 - q*dx; break;
					case dirBS:  wallX1 = nodeX1;        wallX2 = nodeX2 - q*dx; wallX3 = nodeX3 - q*dx; break;
					case dirTNE: wallX1 = nodeX1 + q*dx; wallX2 = nodeX2 + q*dx; wallX3 = nodeX3 + q*dx; break;
					case dirBSW: wallX1 = nodeX1 - q*dx; wallX2 = nodeX2 - q*dx; wallX3 = nodeX3 - q*dx; break;
					case dirBNE: wallX1 = nodeX1 + q*dx; wallX2 = nodeX2 + q*dx; wallX3 = nodeX3 - q*dx; break;
					case dirTSW: wallX1 = nodeX1 - q*dx; wallX2 = nodeX2 - q*dx; wallX3 = nodeX3 + q*dx; break;
					case dirTSE: wallX1 = nodeX1 + q*dx; wallX2 = nodeX2 - q*dx; wallX3 = nodeX3 + q*dx; break;
					case dirBNW: wallX1 = nodeX1 - q*dx; wallX2 = nodeX2 + q*dx; wallX3 = nodeX3 - q*dx; break;
					case dirBSE: wallX1 = nodeX1 + q*dx; wallX2 = nodeX2 - q*dx; wallX3 = nodeX3 - q*dx; break;
					case dirTNW: wallX1 = nodeX1 - q*dx; wallX2 = nodeX2 + q*dx; wallX3 = nodeX3 + q*dx; break;
					case dirZERO:wallX1 = nodeX1;        wallX2 = nodeX2;		 wallX3 = nodeX3;        break;
					default: throw UbException(UB_EXARGS, "unknown direction");
				}
				//////////////////////////////////////////////////////////////////////////
				nodes[node] = makeUbTuple((float)(nodeX1), (float)(nodeX2), (float)(nodeX3));
				nodes[wall] = makeUbTuple((float)(wallX1), (float)(wallX2), (float)(wallX3));
				qs[line]    = makeUbTuple(node, wall);
				//////////////////////////////////////////////////////////////////////////
				nodedata[0][node] = q;
				nodedata[0][wall] = q;
				//////////////////////////////////////////////////////////////////////////
				node = node + 2;
				wall = wall + 2;
				line++;
				//////////////////////////////////////////////////////////////////////////
			}
		}
		WbWriterVtkXmlBinary::getInstance()->writeLines(fname, nodes, qs);
		//WbWriterVtkXmlBinary::getInstance()->writeLinesWithNodeData(fname, nodes, qs, nodedatanames, nodedata);
		//WbWriterVtkXmlASCII::getInstance()->writeLinesWithNodeData(fname, nodes, qs, nodedatanames, nodedata);
	}
	//////////////////////////////////////////////////////////////////////////
}

#endif
