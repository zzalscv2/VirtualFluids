#include "Calculation/Cp.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <stdio.h>

#include <fstream>
#include <sstream>

//#include <math.h>
//#include "LB.h"

using namespace std;



void calcCp(Parameter* para, CudaMemoryManager* cudaManager, int lev)
{
	//////////////////////////////////////////////////////////////////////////
	//copy to host
	cudaManager->cudaCopyCpTop(lev);
	cudaManager->cudaCopyCpBottom(lev);
	cudaManager->cudaCopyCpBottom2(lev);
	//////////////////////////////////////////////////////////////////////////
	//Parameter
	double rhoSI = 1.204; // kg/m^3
	double veloSI = (double)para->getVelocity() * (double)para->getVelocityRatio(); // m/s
	double pressSI;
	double cp;
	std::vector< double > cpTopRow;
	std::vector< double > cpBottomRow;
	std::vector< double > cpBottom2Row;
	//////////////////////////////////////////////////////////////////////////
	//calc cp top
	for (unsigned int it = 0; it < para->getParH(lev)->numberOfPointsCpTop; it++)
	{
		pressSI = (double)(para->getParH(lev)->cpPressTop[it] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio());
		cp      = (double) (pressSI / (0.5 * rhoSI * veloSI * veloSI));
		cpTopRow.push_back(cp);
	}
	para->getParH(lev)->cpTop.push_back(cpTopRow);
	//////////////////////////////////////////////////////////////////////////
	//calc cp bottom
	for (int it = 0; it < para->getParH(lev)->numberOfPointsCpBottom; it++)
	{
		pressSI = (double)(para->getParH(lev)->cpPressBottom[it] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio());
		cp      = (double) (pressSI / (0.5 * rhoSI * veloSI * veloSI));
		cpBottomRow.push_back(cp);
	}
	para->getParH(lev)->cpBottom.push_back(cpBottomRow);
	//////////////////////////////////////////////////////////////////////////
	//calc cp bottom 2
	for (int it = 0; it < para->getParH(lev)->numberOfPointsCpBottom2; it++)
	{
		pressSI = (double)(para->getParH(lev)->cpPressBottom2[it] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio());
		cp      = (double) (pressSI / (0.5 * rhoSI * veloSI * veloSI));
		cpBottom2Row.push_back(cp);
	}
	para->getParH(lev)->cpBottom2.push_back(cpBottom2Row);
	//////////////////////////////////////////////////////////////////////////
}



void printCpTopIntermediateStep(Parameter* para, unsigned int t, int lev)
{
	//////////////////////////////////////////////////////////////////////////
	//set filename
	std::string ffname = para->getFName() + StringUtil::toString<int>(para->getMyID()) + "_" + StringUtil::toString<int>(t) + "_cp_top.txt";
	const char* fname = ffname.c_str();
	//////////////////////////////////////////////////////////////////////////
	//set ofstream
	ofstream ostr;
	//////////////////////////////////////////////////////////////////////////
	//open file
	ostr.open(fname);
	//////////////////////////////////////////////////////////////////////////
	//fill file with data
	for (vector< vector<double> >::const_iterator i = para->getParH(lev)->cpTop.begin(); i != para->getParH(lev)->cpTop.end(); ++i)
	{
		for (vector<double>::const_iterator j = i->begin(); j != i->end(); ++j)
		{
			ostr << *j << " ";
		}
		ostr << endl;
	}
	//////////////////////////////////////////////////////////////////////////
	//close file
	ostr.close();
	//////////////////////////////////////////////////////////////////////////
	para->getParH(lev)->cpTop.clear();
	//////////////////////////////////////////////////////////////////////////
}



void printCpTop(Parameter* para, CudaMemoryManager* cudaManager, int lev)
{
	//////////////////////////////////////////////////////////////////////////
	//set filename
	std::string ffname = para->getFName()+StringUtil::toString<int>(para->getMyID())+"_cp_top.txt";
	const char* fname = ffname.c_str();
	//////////////////////////////////////////////////////////////////////////
	//set ofstream
	ofstream ostr;
	//////////////////////////////////////////////////////////////////////////
	//open file
	ostr.open(fname);
	//////////////////////////////////////////////////////////////////////////
	//fill file with data
	for (vector< vector<double> >::const_iterator i = para->getParH(lev)->cpTop.begin() ; i != para->getParH(lev)->cpTop.end(); ++i)
	{
		for (vector<double>::const_iterator j=i->begin(); j!=i->end(); ++j)
		{
			ostr << *j << " ";
		}
		ostr << endl;
	}
	//////////////////////////////////////////////////////////////////////////
	//close file
	ostr.close();
	//////////////////////////////////////////////////////////////////////////
	para->getParH(lev)->cpTop.clear();
	cudaManager->cudaFreeCpTop(lev);
	//////////////////////////////////////////////////////////////////////////
}



void printCpBottom(Parameter* para, CudaMemoryManager* cudaManager)
{
	//////////////////////////////////////////////////////////////////////////
	//set level
	int lev = para->getMaxLevel();
	//////////////////////////////////////////////////////////////////////////
	//set filename
	std::string ffname = para->getFName()+StringUtil::toString<int>(para->getMyID())+"_cp_bottom.txt";
	const char* fname = ffname.c_str();
	//////////////////////////////////////////////////////////////////////////
	//set ofstream
	ofstream ostr;
	//////////////////////////////////////////////////////////////////////////
	//open file
	ostr.open(fname);
	//////////////////////////////////////////////////////////////////////////
	//fill file with data
	for (vector< vector<double> >::const_iterator i = para->getParH(lev)->cpBottom.begin() ; i != para->getParH(lev)->cpBottom.end(); ++i)
	{
		for (vector<double>::const_iterator j=i->begin(); j!=i->end(); ++j)
		{
			ostr << *j << " ";
		}
		ostr << endl;
	}
	//////////////////////////////////////////////////////////////////////////
	//close file
	ostr.close();
	//////////////////////////////////////////////////////////////////////////
	para->getParH(lev)->cpBottom.clear();
	cudaManager->cudaFreeCpBottom(lev);
	//////////////////////////////////////////////////////////////////////////
}



void printCpBottom2(Parameter* para, CudaMemoryManager* cudaManager)
{
	//////////////////////////////////////////////////////////////////////////
	//set level
	int lev = para->getMaxLevel();
	//////////////////////////////////////////////////////////////////////////
	//set filename
	std::string ffname = para->getFName()+StringUtil::toString<int>(para->getMyID())+"_cp_bottom2.txt";
	const char* fname = ffname.c_str();
	//////////////////////////////////////////////////////////////////////////
	//set ofstream
	ofstream ostr;
	//////////////////////////////////////////////////////////////////////////
	//open file
	ostr.open(fname);
	//////////////////////////////////////////////////////////////////////////
	//fill file with data
	for (vector< vector<double> >::const_iterator i = para->getParH(lev)->cpBottom2.begin() ; i != para->getParH(lev)->cpBottom2.end(); ++i)
	{
		for (vector<double>::const_iterator j=i->begin(); j!=i->end(); ++j)
		{
			ostr << *j << " ";
		}
		ostr << endl;
	}
	//////////////////////////////////////////////////////////////////////////
	//close file
	ostr.close();
	//////////////////////////////////////////////////////////////////////////
	para->getParH(lev)->cpBottom2.clear();
	cudaManager->cudaFreeCpBottom2(lev);
	//////////////////////////////////////////////////////////////////////////
}






















void excludeGridInterfaceNodesForMirror(Parameter* para, int lev)
{
	bool tempBool = true;
	para->getParH(lev)->numberOfPointsPressWindow = 0;
	para->getParH(lev + 1)->numberOfPointsPressWindow = 0;
	//////////////////////////////////////////////////////////////////////////
	//define bool vector for nodes outside the interface
	for (unsigned int it = 0; it < para->getParH(lev + 1)->numberOfPointsCpTop; it++)
	{
		for (unsigned int ifit = 0; ifit < para->getParH(lev)->K_CF; ifit++)
		{
			if ((para->getParH(lev + 1)->cpTopIndex[it] == para->getParH(lev)->intCF.ICellCFF[ifit]) ||
				(para->getParH(lev + 1)->cpTopIndex[it] == para->getParH(lev + 1)->neighborX_SP[para->getParH(lev)->intCF.ICellCFF[ifit]]) ||
				(para->getParH(lev + 1)->cpTopIndex[it] == para->getParH(lev + 1)->neighborY_SP[para->getParH(lev)->intCF.ICellCFF[ifit]]) ||
				(para->getParH(lev + 1)->cpTopIndex[it] == para->getParH(lev + 1)->neighborZ_SP[para->getParH(lev)->intCF.ICellCFF[ifit]]) ||
				(para->getParH(lev + 1)->cpTopIndex[it] == para->getParH(lev + 1)->neighborY_SP[para->getParH(lev + 1)->neighborX_SP[para->getParH(lev)->intCF.ICellCFF[ifit]]]) ||
				(para->getParH(lev + 1)->cpTopIndex[it] == para->getParH(lev + 1)->neighborZ_SP[para->getParH(lev + 1)->neighborX_SP[para->getParH(lev)->intCF.ICellCFF[ifit]]]) ||
				(para->getParH(lev + 1)->cpTopIndex[it] == para->getParH(lev + 1)->neighborZ_SP[para->getParH(lev + 1)->neighborY_SP[para->getParH(lev)->intCF.ICellCFF[ifit]]]) ||
				(para->getParH(lev + 1)->cpTopIndex[it] == para->getParH(lev + 1)->neighborZ_SP[para->getParH(lev + 1)->neighborY_SP[para->getParH(lev + 1)->neighborX_SP[para->getParH(lev)->intCF.ICellCFF[ifit]]]]))
			{
				para->getParH(lev + 1)->isOutsideInterface.push_back(false);
				tempBool = false;
				break;
			}
		}
		if (tempBool == true)
		{
			para->getParH(lev + 1)->isOutsideInterface.push_back(true);
			para->getParH(lev + 1)->numberOfPointsPressWindow++;
		}
		tempBool = true;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for (unsigned int it = 0; it < para->getParH(lev)->numberOfPointsCpTop; it++)
	{
		for (unsigned int ifit = 0; ifit < para->getParH(lev)->K_FC; ifit++)
		{
			if (para->getParH(lev)->cpTopIndex[it] == para->getParH(lev)->intFC.ICellFCC[ifit])
			{
				para->getParH(lev)->isOutsideInterface.push_back(false);
				tempBool = false;
				break;
			}
		}
		if (tempBool == true)
		{
			para->getParH(lev)->isOutsideInterface.push_back(true);
			para->getParH(lev)->numberOfPointsPressWindow++;
		}
		tempBool = true;
	}
	////////////////////////////////////////////////////////////////////////////
	std::cout << "number of nodes cp top level 7:" << para->getParH(lev)->numberOfPointsCpTop << endl;
	std::cout << "number of nodes bool level 7:" << para->getParH(lev)->isOutsideInterface.size() << endl;
	std::cout << "number of nodes press window level 7:" << para->getParH(lev)->numberOfPointsPressWindow << endl;
	std::cout << "number of nodes cp top level 8:" << para->getParH(lev + 1)->numberOfPointsCpTop << endl;
	std::cout << "number of nodes bool level 8:" << para->getParH(lev + 1)->isOutsideInterface.size() << endl;
	std::cout << "number of nodes press window level 8:" << para->getParH(lev + 1)->numberOfPointsPressWindow << endl;
}



void calcPressForMirror(Parameter* para, CudaMemoryManager* cudaManager, int lev)
{
	//////////////////////////////////////////////////////////////////////////
	//copy to host
	cudaManager->cudaCopyCpTop(lev);
	//////////////////////////////////////////////////////////////////////////
	//Parameter
	double pressSI;
	//////////////////////////////////////////////////////////////////////////
	//calc press
	for (unsigned int it = 0; it < para->getParH(lev)->numberOfPointsCpTop; it++)
	{
		if (para->getParH(lev)->isOutsideInterface[it])
		{
			pressSI = (double)(para->getParH(lev)->cpPressTop[it] / (double)3.0 * (double)para->getDensityRatio() * (double)para->getVelocityRatio() * (double)para->getVelocityRatio());
			para->getParH(lev)->pressMirror.push_back(pressSI);
		}
	}
	////////////////////////////////////////////////////////////////////////////
	//std::cout << "number of nodes press mirror:" << para->getParH(lev)->pressMirror.size() << ", at level: " << lev << endl;
}



//Ensight Gold
void printCaseFile(Parameter* para)
{
	//////////////////////////////////////////////////////////////////////////
	double deltaXcoarse = 0.256; // [m]
	double deltat = (para->getVelocity() * deltaXcoarse) / (para->getVelocity() * para->getVelocityRatio());
	unsigned int numberOfSteps = (unsigned int)((para->getTEnd() - para->getTStartOut()) * pow(2,5) );
	//cout << "number of nodes:" << numberOfSteps << endl;
	//////////////////////////////////////////////////////////////////////////
	//set filename
	std::string ffname = para->getFName() + "_" + StringUtil::toString<int>(para->getMyID()) + ".case";
	const char* fname = ffname.c_str();
	//////////////////////////////////////////////////////////////////////////
	//set filename geo
	std::string ffnameGeo = para->getOutputPrefix() + "_" + StringUtil::toString<int>(para->getMyID()) + ".geo";
	const char* fnameGeo = ffnameGeo.c_str();
	//////////////////////////////////////////////////////////////////////////
	//set filename scalar
	std::string ffnameScalar = para->getOutputPrefix() + "_" + StringUtil::toString<int>(para->getMyID()) + ".*****.p";
	const char* fnameScalar = ffnameScalar.c_str();
	//////////////////////////////////////////////////////////////////////////
	//set ofstream
	ofstream ostr;
	//////////////////////////////////////////////////////////////////////////
	//open file
	ostr.open(fname);
	//////////////////////////////////////////////////////////////////////////
	//fill file with data
	ostr << "###########################################################" << std::endl;
	ostr << "#### Casefile written by VirtualFluidsGPU" << std::endl;
	ostr << "###########################################################" << std::endl << std::endl;

	ostr << "FORMAT " << std::endl;
	ostr << "type:                  ensight gold" << std::endl << std::endl;

	ostr << "GEOMETRY " << std::endl;
	ostr << "model:                 " << "Data/" << fnameGeo << std::endl << std::endl;

	ostr << "VARIABLE " << std::endl;
	ostr << "scalar per element:    1  Pressure \t" << "Data/" << fnameScalar << std::endl << std::endl;

	ostr << "TIME " << std::endl;
	ostr << "time set:              1" << std::endl;
	ostr << "number of steps:       " << numberOfSteps << "\n";
	ostr << "filename start number: 0" << std::endl;
	ostr << "filename increment:    1" << std::endl;
	ostr << "time values:" << std::endl;
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int t = 0; t < numberOfSteps; t++) 
	{
		ostr << "\t" << t * deltat / (pow(2, 5)) << std::endl;
	}
	//////////////////////////////////////////////////////////////////////////
	//close file
	ostr.close();
	//////////////////////////////////////////////////////////////////////////
}






extern "C" void printGeoFile(Parameter* para, bool fileFormat)
{
	//////////////////////////////////////////////////////////////////////////
	//set filename geo
	std::string ffnameGeo = para->getOutputPrefix() + "_" + StringUtil::toString<int>(para->getMyID());
	const char* fnameGeo = ffnameGeo.c_str();
	//////////////////////////////////////////////////////////////////////////
	char fname[1024];
	sprintf(fname, "%s/Data/%s.geo", para->getOutputPath().c_str(), fnameGeo);
	//////////////////////////////////////////////////////////////////////////
	size_t startlevel = para->getMaxLevel() - 1;
	size_t endlevel   = para->getMaxLevel();
	//////////////////////////////////////////////////////////////////////////
	unsigned int non = 0;
	for (size_t lev = startlevel; lev <= endlevel; lev++)
	{
		non += para->getParH(lev)->numberOfPointsPressWindow;
	}
	//////////////////////////////////////////////////////////////////////////

	switch (fileFormat)
	{
		case 0: //ASCII
		{
			//////////////////////////////////////////////////////////////////////////
			//set ofstream
			ofstream ostr;
			std::ostringstream temp1;
			std::ostringstream temp2;
			std::ostringstream temp3;
			//////////////////////////////////////////////////////////////////////////
			//open file
			ostr.open(fname);
			//////////////////////////////////////////////////////////////////////////
			ostr << "This geometry File was written by VirtualFluidsGPU\n";
			ostr << "#### Casefile written by VirtualFluidsGPU\n";
			//////////////////////////////////////////////////////////////////////////
			ostr << "node id assign \n";
			ostr << "element id assign \n";
			//////////////////////////////////////////////////////////////////////////
			ostr << "part \n \t 1 \n";
			ostr << fnameGeo << "\n";
			ostr << "coordinates \n \t" << non << "\n";
			//////////////////////////////////////////////////////////////////////////
			// X
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (unsigned int i = 0; i < para->getParH(lev)->numberOfPointsCpTop; i++)
				{
					if (para->getParH(lev)->isOutsideInterface[i])
					{
						ostr << (para->getParH(lev)->coordX_SP[para->getParH(lev)->cpTopIndex[i]] * para->getScaleLBMtoSI().at(0) + para->getTranslateLBMtoSI().at(0)) << std::endl;
					}
				}
			}
			//////////////////////////////////////////////////////////////////////////
			// Y
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (unsigned int i = 0; i < para->getParH(lev)->numberOfPointsCpTop; i++)
				{
					if (para->getParH(lev)->isOutsideInterface[i])
					{
						ostr << (para->getParH(lev)->coordY_SP[para->getParH(lev)->cpTopIndex[i]] * para->getScaleLBMtoSI().at(1) + para->getTranslateLBMtoSI().at(1)) << std::endl;
					}
				}
			}
			//////////////////////////////////////////////////////////////////////////
			// Z
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (unsigned int i = 0; i < para->getParH(lev)->numberOfPointsCpTop; i++)
				{
					if (para->getParH(lev)->isOutsideInterface[i])
					{
						ostr << (para->getParH(lev)->coordZ_SP[para->getParH(lev)->cpTopIndex[i]] * para->getScaleLBMtoSI().at(2) + para->getTranslateLBMtoSI().at(2)) << std::endl;
					}
				}
			}
			//////////////////////////////////////////////////////////////////////////
			ostr << "point \n \t" << non << "\n";
			//////////////////////////////////////////////////////////////////////////
			unsigned int j = 0;
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (size_t i = 0; i < para->getParH(lev)->numberOfPointsPressWindow; i++)
				{
					j++;
					ostr << j << "\n";
				}
			}
			//////////////////////////////////////////////////////////////////////////
			//close file
			ostr.close();
			//////////////////////////////////////////////////////////////////////////
			break;
			//////////////////////////////////////////////////////////////////////////
		}

		case 1: //Binary:
		{
			int tempX = 0;
			//////////////////////////////////////////////////////////////////////////
			std::ofstream ostr;
			ostr.open(fname, std::ios::out | std::ios::binary);
			assert(ostr.is_open());
			//////////////////////////////////////////////////////////////////////////
			float tempCoord = 0.0f;
			//////////////////////////////////////////////////////////////////////////
			writeStringToFile("C Binary", ostr);
			writeStringToFile("This geometry File was written by VirtualFluidsGPU", ostr);
			writeStringToFile("#### Casefile written by VirtualFluidsGPU", ostr);
			writeStringToFile("node id assign", ostr);
			writeStringToFile("element id assign", ostr);
			writeStringToFile("part", ostr);
			writeIntToFile(1, ostr);
			writeStringToFile(fnameGeo, ostr);
			writeStringToFile("coordinates", ostr);
			writeIntToFile(non, ostr);
			//////////////////////////////////////////////////////////////////////////
			// X
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (unsigned int i = 0; i < para->getParH(lev)->numberOfPointsCpTop; i++)
				{
					if (para->getParH(lev)->isOutsideInterface[i])
					{
						tempCoord = (para->getParH(lev)->coordX_SP[para->getParH(lev)->cpTopIndex[i]] * para->getScaleLBMtoSI().at(0) + para->getTranslateLBMtoSI().at(0));
						writeFloatToFile(tempCoord, ostr);
						tempX++;
					}
				}
				//std::cout << "tempX in geo: " << tempX << endl;
			}
			//////////////////////////////////////////////////////////////////////////
			// Y
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (unsigned int i = 0; i < para->getParH(lev)->numberOfPointsCpTop; i++)
				{
					if (para->getParH(lev)->isOutsideInterface[i])
					{
						tempCoord = (para->getParH(lev)->coordY_SP[para->getParH(lev)->cpTopIndex[i]] * para->getScaleLBMtoSI().at(1) + para->getTranslateLBMtoSI().at(1));
						writeFloatToFile(tempCoord, ostr);
					}
				}
			}
			//////////////////////////////////////////////////////////////////////////
			// Z
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (unsigned int i = 0; i < para->getParH(lev)->numberOfPointsCpTop; i++)
				{
					if (para->getParH(lev)->isOutsideInterface[i])
					{
						tempCoord = (para->getParH(lev)->coordZ_SP[para->getParH(lev)->cpTopIndex[i]] * para->getScaleLBMtoSI().at(2) + para->getTranslateLBMtoSI().at(2));
						writeFloatToFile(tempCoord, ostr);
					}
				}
			}
			//////////////////////////////////////////////////////////////////////////
			writeStringToFile("point", ostr);
			writeIntToFile(non, ostr);
			//////////////////////////////////////////////////////////////////////////
			unsigned int j = 0;
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (size_t i = 0; i < para->getParH(lev)->numberOfPointsPressWindow; i++)
				{
					j++;
					writeIntToFile(j, ostr);
				}
				//std::cout << "level: " << lev << ", numberOfPointsPressWindow:" << para->getParH(lev)->numberOfPointsPressWindow << endl;
			}
			//////////////////////////////////////////////////////////////////////////
			//close file
			ostr.close();
			//////////////////////////////////////////////////////////////////////////
			break;
			//////////////////////////////////////////////////////////////////////////
		}
	}
}






extern "C" void printScalars(Parameter* para, bool fileFormat)
{
	//////////////////////////////////////////////////////////////////////////
	//set filename scalar
	std::string ffnameScalar = para->getOutputPrefix() + "_" + StringUtil::toString<int>(para->getMyID());
	const char* fnameScalar = ffnameScalar.c_str();
	//////////////////////////////////////////////////////////////////////////
    char fname[1024];
    sprintf(fname, "%s/Data/%s.%05u.p", para->getOutputPath().c_str(), fnameScalar, para->getStepEnsight());
	para->setStepEnsight(para->getStepEnsight()+1);
	//////////////////////////////////////////////////////////////////////////
	size_t startlevel = para->getMaxLevel() - 1;
	size_t endlevel   = para->getMaxLevel();
	//////////////////////////////////////////////////////////////////////////


	switch (fileFormat)
	{
		case 0: //ASCII
		{
			//////////////////////////////////////////////////////////////////////////
			//set ofstream
			ofstream ostr;
			//////////////////////////////////////////////////////////////////////////
			//open file
			ostr.open(fname);
			//////////////////////////////////////////////////////////////////////////
			ostr << fnameScalar << " \n";
			ostr << "part \n\t 1 \n";
			ostr << "point\n";
			//////////////////////////////////////////////////////////////////////////
			//fill file with data
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (vector<double>::const_iterator i = para->getParH(lev)->pressMirror.begin(); i != para->getParH(lev)->pressMirror.end(); ++i)
				{
					ostr << *i << "\n";
				}
			}
			//////////////////////////////////////////////////////////////////////////
			//close file
			ostr.close();
			//////////////////////////////////////////////////////////////////////////
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				para->getParH(lev)->pressMirror.clear();
			}
			//////////////////////////////////////////////////////////////////////////
			break;
			//////////////////////////////////////////////////////////////////////////
		}

		case 1: //Binary:
		{
			//////////////////////////////////////////////////////////////////////////
			std::ofstream ostr;
			ostr.open(fname, std::ios::out | std::ios::binary);
			assert(ostr.is_open());
			//////////////////////////////////////////////////////////////////////////
			writeStringToFile(fnameScalar, ostr);
			writeStringToFile("part", ostr);
			writeIntToFile(1, ostr);
			writeStringToFile("point", ostr);
			//////////////////////////////////////////////////////////////////////////
			//fill file with data
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				for (vector<double>::const_iterator i = para->getParH(lev)->pressMirror.begin(); i != para->getParH(lev)->pressMirror.end(); ++i)
				{
					writeFloatToFile(*i, ostr);
				}
			}
			//////////////////////////////////////////////////////////////////////////
			//close file
			ostr.close();
			//////////////////////////////////////////////////////////////////////////
			for (size_t lev = startlevel; lev <= endlevel; lev++)
			{
				para->getParH(lev)->pressMirror.clear();
			}
			//////////////////////////////////////////////////////////////////////////
			break;
			//////////////////////////////////////////////////////////////////////////
		}
	}
}




//functions to write binary files
void writeIntToFile(const int &i, std::ofstream &ofile)
{
	ofile.write((char*)&i, sizeof(int));
}

void writeFloatToFile(const float &f, std::ofstream &ofile)
{
	ofile.write((char*)&f, sizeof(float));
}

void writeStringToFile(const std::string &s, std::ofstream &ofile)
{
	assert(s.size() <= 80);
	char cbuffer[81];
	// Terminate the buffer to avoid static analyzer warnings about strncpy not
	// NUL-terminating its destination buffer in case the input is too long.
	cbuffer[80] = '\0';
	strncpy(cbuffer, s.c_str(), 80);
	// Write a constant 80 bytes to the file.
	ofile.write(cbuffer, 80);
}