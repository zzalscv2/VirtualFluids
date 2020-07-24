#include "Init/VfReader.h"

#include "GPU/CudaMemoryManager.h"

////////////////////////////////////////////////////////////////////////////////
void readVFkFull(Parameter* para, CudaMemoryManager* cudaManager, const std::string geometryFile)
{
	kFullReader::readFileForAlloc(geometryFile, para);

	for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		cudaManager->cudaAllocFull(lev);
		//////////////////////////////////////////////////////////////////////////
		for(unsigned int ix3=0; ix3<para->getParH(lev)->nz; ix3++)
		{
			for(unsigned int ix2=0; ix2<para->getParH(lev)->ny; ix2++)
			{
				for(unsigned int ix1=0; ix1<para->getParH(lev)->nx; ix1++)
				{
					unsigned int m = para->getParH(lev)->nx*(para->getParH(lev)->ny*ix3 + ix2) + ix1;
					para->getParH(lev)->k[m]   =  0;
					para->getParH(lev)->geo[m] =  16;
				}
			}
		}
	}

	kFullReader::readFile(geometryFile, para);
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readVFgeoFull(Parameter* para, const std::string geometryFile)
{
	kFullReader::readGeoFull(geometryFile, para);

	//////////////////////////////////////////////////////////////////////////
	//for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	//{
	//	for(unsigned int ix3=0; ix3<para->getParH(lev)->nz; ix3++)
	//	{
	//		for(unsigned int ix2=0; ix2<para->getParH(lev)->ny; ix2++)
	//		{
	//			for(unsigned int ix1=0; ix1<para->getParH(lev)->nx; ix1++)
	//			{
	//				unsigned int m = para->getParH(lev)->nx*(para->getParH(lev)->ny*ix3 + ix2) + ix1;
	//				if (para->getParH(lev)->geo[m] == 0 || para->getParH(lev)->geo[m] == 15)
	//				{
	//					para->getParH(lev)->geo[m] = 16;
	//				}
	//			}
	//		}
	//	}
	//}
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readVecSP(Parameter* para, CudaMemoryManager* cudaManager)
{
	PositionReader::readFileForAlloc(para->getgeoVec(), para);

	//alloc
	for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		cudaManager->cudaAllocSP(lev);
	}

	//geoSP
	PositionReader::readFile(para->getgeoVec(), "geoVec", para);
	//neighborX
	PositionReader::readFile(para->getneighborX(), "neighborX", para);
	//neighborY
	PositionReader::readFile(para->getneighborY(), "neighborY", para);
	//neighborZ
	PositionReader::readFile(para->getneighborZ(), "neighborZ", para);

	//Copy Host -> Device
	for(int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		for(unsigned int u=0; u<para->getParH(lev)->size_Mat_SP; u++)
		{
			para->getParH(lev)->rho_SP[u]   = 0.01f;//+ lev/100.f;
			para->getParH(lev)->vx_SP[u]    = 0.0f;//+ lev/100.f;   
			para->getParH(lev)->vy_SP[u]    = 0.0f;//+ lev/100.f;   
			para->getParH(lev)->vz_SP[u]    = 0.0f;//+ lev/100.f;   
			para->getParH(lev)->press_SP[u] = 0.0f;//+ lev/100.f;
			//Median
			para->getParH(lev)->rho_SP_Med[u]   = 0.0f;
			para->getParH(lev)->vx_SP_Med[u]    = 0.0f;
			para->getParH(lev)->vy_SP_Med[u]    = 0.0f;
			para->getParH(lev)->vz_SP_Med[u]    = 0.0f;
			para->getParH(lev)->press_SP_Med[u] = 0.0f;
		}
		cudaManager->cudaCopySP(lev);
	}
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readInterfaceCF(Parameter* para, CudaMemoryManager* cudaManager)
{
	PositionReader::readFileInterfaceForAlloc(para->getscaleCFC(), "CF", para);

	//alloc
	for (int lev = 0; lev < para->getMaxLevel(); lev++)
	{
		cudaManager->cudaAllocInterfaceCF(lev);
	}

	//Scale Coarse to Fine - Coarse
	PositionReader::readFileInterface(para->getscaleCFC(), "CFC", para);
	//Scale Coarse to Fine - Fine
	PositionReader::readFileInterface(para->getscaleCFF(), "CFF", para);

	//Copy Host -> Device
	for (int lev = 0; lev < para->getMaxLevel(); lev++)
	{
		cudaManager->cudaCopyInterfaceCF(lev);
	}
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readInterfaceFC(Parameter* para, CudaMemoryManager* cudaManager)
{
	PositionReader::readFileInterfaceForAlloc(para->getscaleFCC(), "FC", para);

	//alloc
	for (int lev = 0; lev < para->getMaxLevel(); lev++)
	{
		cudaManager->cudaAllocInterfaceFC(lev);
	}

	//Scale Fine to Coarse - Coarse
	PositionReader::readFileInterface(para->getscaleFCC(), "FCC", para);
	//Scale Fine to Coarse - Fine
	PositionReader::readFileInterface(para->getscaleFCF(), "FCF", para);

	//Copy Host -> Device
	for (int lev = 0; lev < para->getMaxLevel(); lev++)
	{
		cudaManager->cudaCopyInterfaceFC(lev);
	}
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readInterfaceOffCF(Parameter* para, CudaMemoryManager* cudaManager, const std::string geometryFile)
{
	PositionReader::readFileInterfaceOffsetForAlloc(geometryFile, "CF", para);

	for (int lev = 0; lev < para->getMaxLevel(); lev++)
	{
		cudaManager->cudaAllocInterfaceOffCF(lev);
	}

	PositionReader::readFileInterfaceOffset(geometryFile, "CF", para);

	for (int lev = 0; lev < para->getMaxLevel(); lev++)
	{
		cudaManager->cudaCopyInterfaceOffCF(lev);
	}
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readInterfaceOffFC(Parameter* para, CudaMemoryManager* cudaManager, const std::string geometryFile)
{
	PositionReader::readFileInterfaceOffsetForAlloc(geometryFile, "FC", para);

	for (int lev = 0; lev < para->getMaxLevel(); lev++)
	{
		cudaManager->cudaAllocInterfaceOffFC(lev);
	}

	PositionReader::readFileInterfaceOffset(geometryFile, "FC", para);

	for (int lev = 0; lev < para->getMaxLevel(); lev++)
	{
		cudaManager->cudaCopyInterfaceOffFC(lev);
	}
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readNoSlipBc(Parameter* para, CudaMemoryManager* cudaManager)
{
	PositionReader::readFileNoSlipBcForAlloc(para->getnoSlipBcPos(), para);
	PositionReader::readFileNoSlipBcQreadForAlloc(para->getnoSlipBcQs(), para);

	for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		cudaManager->cudaAllocWallBC(lev);
	}

	PositionReader::readFileNoSlipBcPos(para->getnoSlipBcPos(), para);
	PositionReader::readFileNoSlipBcValue(para->getnoSlipBcValue(), para);
	PositionReader::readFileNoSlipBcQs(para->getnoSlipBcQs(), para);

	PositionReader::findQs(para);

	for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		cudaManager->cudaCopyWallBC(lev);
	}
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readSlipBc(Parameter* para, CudaMemoryManager* cudaManager)
{
	PositionReader::readFileSlipBcForAlloc(para->getslipBcPos(), para);
	PositionReader::readFileSlipBcQreadForAlloc(para->getslipBcQs(), para);

	for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		cudaManager->cudaAllocSlipBC(lev);
	}

	PositionReader::readFileSlipBcPos(para->getslipBcPos(), para);
	PositionReader::readFileSlipBcValue(para->getslipBcValue(), para);
	PositionReader::readFileSlipBcQs(para->getslipBcQs(), para);

	PositionReader::findSlipQs(para);

	for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		cudaManager->cudaCopySlipBC(lev);
	}
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readPressBc(Parameter* para, CudaMemoryManager* cudaManager)
{
	PositionReader::readFilePressBcForAlloc(para->getpressBcPos(), para);
	PositionReader::readFilePressBcQreadForAlloc(para->getpressBcQs(), para);

	for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		cudaManager->cudaAllocPress(lev);
	}
	//only Coarse
	//para->cudaAllocPress(para->getCoarse());

	PositionReader::readFilePressBcPos(para->getpressBcPos(), para);
	PositionReader::readFilePressBcValue(para->getpressBcValue(), para);
	PositionReader::readFilePressBcQs(para->getpressBcQs(), para);

	PositionReader::findPressQs(para);

	for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		cudaManager->cudaCopyPress(lev);
	}
	//only Coarse
	//para->cudaCopyPress(para->getCoarse());
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readPropellerCylinder(Parameter* para, CudaMemoryManager* cudaManager)
{
	PositionReader::readFilePropellerCylinderForAlloc(para);

	cudaManager->cudaAllocVeloPropeller(para->getFine());

	PositionReader::readFilePropellerCylinder(para);
	//PositionReader::definePropellerQs(para);

	cudaManager->cudaCopyVeloPropeller(para->getFine());
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
void readMeasurePoints(Parameter* para, CudaMemoryManager* cudaManager)
{
	//read measure points from file
	PositionReader::readMeasurePoints(para);
	//printf("done, reading the file...\n");
	//level loop
	for (int lev = 0; lev <= para->getMaxLevel(); lev++)
	{
		//set Memory Size and malloc of the indices and macroscopic values per level
		para->getParH(lev)->numberOfValuesMP = (unsigned int)para->getParH(lev)->MP.size()*(unsigned int)para->getclockCycleForMP()/((unsigned int)para->getTimestepForMP());
		para->getParD(lev)->numberOfValuesMP = para->getParH(lev)->numberOfValuesMP;

		para->getParH(lev)->numberOfPointskMP = (int)para->getParH(lev)->MP.size();
		para->getParD(lev)->numberOfPointskMP = para->getParH(lev)->numberOfPointskMP;

		para->getParH(lev)->memSizeIntkMP = sizeof(unsigned int)*(int)para->getParH(lev)->MP.size();
		para->getParD(lev)->memSizeIntkMP = para->getParH(lev)->memSizeIntkMP;

		para->getParH(lev)->memSizerealkMP = sizeof(real)*para->getParH(lev)->numberOfValuesMP;
		para->getParD(lev)->memSizerealkMP = para->getParH(lev)->memSizerealkMP;		
		
		printf("Level: %d, numberOfValuesMP: %d, memSizeIntkMP: %d, memSizerealkMP: %d\n",lev,para->getParH(lev)->numberOfValuesMP,para->getParH(lev)->memSizeIntkMP, para->getParD(lev)->memSizerealkMP);

		cudaManager->cudaAllocMeasurePointsIndex(lev);

		//loop over all measure points per level 
		for(int index = 0; index < (int)para->getParH(lev)->MP.size(); index++)
		{
			//set indices
			para->getParH(lev)->kMP[index] = para->getParH(lev)->MP[index].k;
		}
		//loop over all measure points per level times MPClockCycle
		for(int index = 0; index < (int)para->getParH(lev)->numberOfValuesMP; index++)
		{
			//init values
			para->getParH(lev)->VxMP[index]  = (real)0.0;
			para->getParH(lev)->VyMP[index]  = (real)0.0;
			para->getParH(lev)->VzMP[index]  = (real)0.0;
			para->getParH(lev)->RhoMP[index] = (real)0.0;
		}

		//copy indices-arrays
		cudaManager->cudaCopyMeasurePointsIndex(lev);
	}
}
////////////////////////////////////////////////////////////////////////////////







