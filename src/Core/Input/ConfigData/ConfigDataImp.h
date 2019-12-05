#ifndef CONFIGDATAIMP_H
#define CONFIGDATAIMP_H

#include "ConfigData.h"

#include <memory>
#include <string>

class  ConfigDataImp : public ConfigData
{
public:
    static std::shared_ptr<ConfigDataImp> getNewInstance();
    virtual ~ConfigDataImp(void);

	real getViscosity();
	uint getNumberOfDevices();
	std::vector<uint> getDevices();
	std::string getOutputPath();
	std::string getPrefix();
	std::string getGridPath();
	bool getPrintOutputFiles();
	bool getGeometryValues();
	bool getCalc2ndOrderMoments();
	bool getCalc3rdOrderMoments();
	bool getCalcHighOrderMoments();
	bool getReadGeo();
	bool getCalcMedian();
	bool getConcFile();
	bool getStreetVelocityFile();
	bool getUseMeasurePoints();
	bool getUseWale();
	bool getUseInitNeq();
	bool getSimulatePorousMedia();
	uint getD3Qxx();
	uint getTEnd();
	uint getTOut();
	uint getTStartOut();
	uint getTimeCalcMedStart();
	uint getTimeCalcMedEnd();
	uint getPressInID();
	uint getPressOutID();
	uint getPressInZ();
	uint getPressOutZ();
	bool getDiffOn();
	uint getDiffMod();
	real getDiffusivity();
	real getTemperatureInit();
	real getTemperatureBC();
	real getVelocity();
	real getViscosityRatio();
	real getVelocityRatio();
	real getDensityRatio();
	real getPressRatio();
	real getRealX();
	real getRealY();
	real getFactorPressBC();
	std::string getGeometryFileC();
	std::string getGeometryFileM();
	std::string getGeometryFileF();
	uint getClockCycleForMP();
	uint getTimestepForMP();
	real getForcingX();
	real getForcingY();
	real getForcingZ();
    real getQuadricLimiterP();
    real getQuadricLimiterM();
    real getQuadricLimiterD();
	bool getCalcParticles();
	int getParticleBasicLevel();
	int getParticleInitLevel();
	int getNumberOfParticles();
	real getStartXHotWall();
	real getEndXHotWall();
	std::vector<std::string> getPossNeighborFilesX();
	std::vector<std::string> getPossNeighborFilesY();
	std::vector<std::string> getPossNeighborFilesZ();
	//std::vector<std::string> getPossNeighborFilesX();
	//std::vector<std::string> getPossNeighborFilesY();
	//std::vector<std::string> getPossNeighborFilesZ();
	int getTimeDoCheckPoint();
	int getTimeDoRestart();
	bool getDoCheckPoint();
	bool getDoRestart();
	uint getMaxLevel();
	std::vector<int> getGridX();
	std::vector<int> getGridY();
	std::vector<int> getGridZ();
	std::vector<int> getDistX();
	std::vector<int> getDistY();
	std::vector<int> getDistZ();
	std::vector<bool> getNeedInterface();
	std::string getMainKernel();
	bool getMultiKernelOn();
	std::vector<int> getMultiKernelLevel();
	std::vector<std::string> getMultiKernelName();

	void setViscosity(real viscosity);
	void setNumberOfDevices(uint numberOfDevices);
	void setDevices(std::vector<uint> devices);
	void setOutputPath(std::string outputPath);
	void setPrefix(std::string prefix);
	void setGridPath(std::string gridPath);
	void setPrintOutputFiles(bool printOutputFiles);
	void setGeometryValues(bool geometryValues);
	void setCalc2ndOrderMoments(bool calc2ndOrderMoments);
	void setCalc3rdOrderMoments(bool calc3rdOrderMoments);
	void setCalcHighOrderMoments(bool calcHighOrderMoment);
	void setReadGeo(bool readGeo);
	void setCalcMedian(bool calcMedian);   
	void setConcFile(bool concFile);
	void setStreetVelocityFile(bool streetVelocityFile);
	void setUseMeasurePoints(bool useMeasurePoints);
	void setUseWale(bool useWale);
	void setUseInitNeq(bool useInitNeq);
	void setSimulatePorousMedia(bool simulatePorousMedia);
	void setD3Qxx(uint d3Qxx);
	void setTEnd(uint tEnd);
	void setTOut(uint tOut);
	void setTStartOut(uint tStartOut);
	void setTimeCalcMedStart(uint timeCalcMedStart);
	void setTimeCalcMedEnd(uint timeCalcMedEnd);   
	void setPressInID(uint pressInID);
	void setPressOutID(uint pressOutID);   
	void setPressInZ(uint pressInZ);
	void setPressOutZ(uint pressOutZ);
	void setDiffOn(bool diffOn);	
	void setDiffMod(uint diffMod);		
	void setDiffusivity(real diffusivity);	
	void setTemperatureInit(real temperatureInit);	
	void setTemperatureBC(real temperatureBC);	
	//void setViscosity(real viscosity);	
	void setVelocity(real velocity);	
	void setViscosityRatio(real viscosityRatio);	
	void setVelocityRatio(real velocityRatio);	
	void setDensityRatio(real fensityRatio);	
	void setPressRatio(real pressRatio);	
	void setRealX(real realX);	
	void setRealY(real realY);	
	void setFactorPressBC(real factorPressBC);	
	void setGeometryFileC(std::string geometryFileC);
	void setGeometryFileM(std::string geometryFileM);
	void setGeometryFileF(std::string geometryFileF); 
	void setClockCycleForMP(uint clockCycleForMP);
	void setTimestepForMP(uint timestepForMP);
	void setForcingX(real forcingX);
	void setForcingY(real forcingY);
	void setForcingZ(real forcingZ);
	void setQuadricLimiterP(real quadricLimiterP);
	void setQuadricLimiterM(real quadricLimiterM);
	void setQuadricLimiterD(real quadricLimiterD);
	void setCalcParticles(bool calcParticles);
	void setParticleBasicLevel(int particleBasicLevel);
	void setParticleInitLevel(int particleInitLevel);
	void setNumberOfParticles(int numberOfParticles);
	void setStartXHotWall(real startXHotWall);
	void setEndXHotWall(real endXHotWall);
	void setPossNeighborFilesX(std::vector<std::string> possNeighborFilesX);
	void setPossNeighborFilesY(std::vector<std::string> possNeighborFilesY);
	void setPossNeighborFilesZ(std::vector<std::string> possNeighborFilesZ);
	//void setPossNeighborFilesX(std::vector<std::string> possNeighborFilesX);
	//void setPossNeighborFilesY(std::vector<std::string> possNeighborFilesY);
	//void setPossNeighborFilesZ(std::vector<std::string> possNeighborFilesZ);
	void setTimeDoCheckPoint(int timeDoCheckPoint);
	void setTimeDoRestart(int timeDoRestart);
	void setDoCheckPoint(bool doCheckPoint);
	void setDoRestart(bool doRestart);
	void setMaxLevel(uint maxLevel);
	void setGridX(std::vector<int> gridX);
	void setGridY(std::vector<int> gridY);
	void setGridZ(std::vector<int> gridZ);
	void setDistX(std::vector<int> distX);
	void setDistY(std::vector<int> distY);
	void setDistZ(std::vector<int> distZ);
	void setNeedInterface(std::vector<bool> needInterface);
	void setMainKernel(std::string mainKernel);
	void setMultiKernelOn(bool multiKernelOn);
	void setMultiKernelLevel(std::vector<int> multiKernelLevel);
	void setMultiKernelName(std::vector<std::string> multiKernelName);

	bool isViscosityInConfigFile();
	bool isNumberOfDevicesInConfigFile();
	bool isDevicesInConfigFile();
	bool isOutputPathInConfigFile();
	bool isPrefixInConfigFile();
	bool isGridPathInConfigFile();
	bool isPrintOutputFilesInConfigFile();
	bool isGeometryValuesInConfigFile();
	bool isCalc2ndOrderMomentsInConfigFile();
	bool isCalc3rdOrderMomentsInConfigFile();
	bool isCalcHighOrderMomentsInConfigFile();
	bool isReadGeoInConfigFile();
	bool isCalcMedianInConfigFile();
	bool isConcFileInConfigFile();
	bool isStreetVelocityFileInConfigFile();
	bool isUseMeasurePointsInConfigFile();
	bool isUseWaleInConfigFile();
	bool isUseInitNeqInConfigFile();
	bool isSimulatePorousMediaInConfigFile();
	bool isD3QxxInConfigFile();
	bool isTEndInConfigFile();
	bool isTOutInConfigFile();
	bool isTStartOutInConfigFile();
	bool isTimeCalcMedStartInConfigFile();
	bool isTimeCalcMedEndInConfigFile();
	bool isPressInIDInConfigFile();
	bool isPressOutIDInConfigFile();
	bool isPressInZInConfigFile();
	bool isPressOutZInConfigFile();
	bool isDiffOnInConfigFile();
	bool isDiffModInConfigFile();
	bool isDiffusivityInConfigFile();
	bool isTemperatureInitInConfigFile();
	bool isTemperatureBCInConfigFile();
	//bool isViscosityInConfigFile();
	bool isVelocityInConfigFile();
	bool isViscosityRatioInConfigFile();
	bool isVelocityRatioInConfigFile();
	bool isDensityRatioInConfigFile();
	bool isPressRatioInConfigFile();
	bool isRealXInConfigFile();
	bool isRealYInConfigFile();
	bool isFactorPressBCInConfigFile();
	bool isGeometryFileCInConfigFile();
	bool isGeometryFileMInConfigFile();
	bool isGeometryFileFInConfigFile();
	bool isClockCycleForMPInConfigFile();
	bool isTimestepForMPInConfigFile();
	bool isForcingXInConfigFile();
	bool isForcingYInConfigFile();
	bool isForcingZInConfigFile();
	bool isQuadricLimiterPInConfigFile();
	bool isQuadricLimiterMInConfigFile();
	bool isQuadricLimiterDInConfigFile();
	bool isCalcParticlesInConfigFile();
	bool isParticleBasicLevelInConfigFile();
	bool isParticleInitLevelInConfigFile();
	bool isNumberOfParticlesInConfigFile();
	bool isNeighborWSBInConfigFile();
	bool isStartXHotWallInConfigFile();
	bool isEndXHotWallInConfigFile();
	bool isPossNeighborFilesXInConfigFile();
	bool isPossNeighborFilesYInConfigFile();
	bool isPossNeighborFilesZInConfigFile();
	bool isTimeDoCheckPointInConfigFile();
	bool isTimeDoRestartInConfigFile();
	bool isDoCheckPointInConfigFile();
	bool isDoRestartInConfigFile();
	bool isMaxLevelInConfigFile();
	bool isGridXInConfigFile();
	bool isGridYInConfigFile();
	bool isGridZInConfigFile();
	bool isDistXInConfigFile();
	bool isDistYInConfigFile();
	bool isDistZInConfigFile();
	bool isNeedInterfaceInConfigFile();
	bool isMainKernelInConfigFile();
	bool isMultiKernelOnInConfigFile();
	bool isMultiKernelLevelInConfigFile();
	bool isMultiKernelNameInConfigFile();


private:
	ConfigDataImp();

	real viscosity;
	uint numberOfDevices;
	std::vector<uint> devices;
	std::string outputPath;
	std::string prefix;
	std::string gridPath;
	bool printOutputFiles;
	bool geometryValues;
	bool calc2ndOrderMoments;
	bool calc3rdOrderMoments;
	bool calcHighOrderMoments;
	bool readGeo;
	bool calcMedian;
	bool concFile;
	bool streetVelocityFile;
	bool useMeasurePoints;
	bool useWale;
	bool useInitNeq;
	bool simulatePorousMedia;
	uint d3Qxx;
	uint tEnd;
	uint tOut;
	uint tStartOut;
	uint timeCalcMedStart;
	uint timeCalcMedEnd;
	uint pressInID;
	uint pressOutID;
	uint pressInZ;
	uint pressOutZ;
	bool diffOn;
	uint diffMod;
	real diffusivity;
	real temperatureInit;
	real temperatureBC;
	//real viscosity;
	real velocity;
	real viscosityRatio;
	real velocityRatio;
	real densityRatio;
	real pressRatio;
	real realX;
	real realY;
	real factorPressBC;
	std::string geometryFileC;
	std::string geometryFileM;
	std::string geometryFileF;
	uint clockCycleForMP;
	uint timestepForMP;
	real forcingX;
	real forcingY;
	real forcingZ;
	real quadricLimiterP;
	real quadricLimiterM;
	real quadricLimiterD;
	bool calcParticles;
	int particleBasicLevel;
	int particleInitLevel;
	int numberOfParticles;
	real startXHotWall;
	real endXHotWall;
	std::vector<std::string> possNeighborFilesX;
	std::vector<std::string> possNeighborFilesY;
	std::vector<std::string> possNeighborFilesZ;
	//std::vector<std::string> possNeighborFilesX;
	//std::vector<std::string> possNeighborFilesY;
	//std::vector<std::string> possNeighborFilesZ;
	int timeDoCheckPoint;
	int timeDoRestart;
	bool doCheckPoint;
	bool doRestart;
	int maxLevel;
	std::vector<int> gridX;
	std::vector<int> gridY;
	std::vector<int> gridZ;
	std::vector<int> distX;
	std::vector<int> distY;
	std::vector<int> distZ;
	std::vector<bool> needInterface;
	std::string mainKernel;
	bool multiKernelOn;
	std::vector<int> multiKernelLevel;
	std::vector<std::string> multiKernelName;

	bool isViscosity;
	bool isNumberOfDevices;
	bool isDevices;
	bool isOutputPath;
	bool isPrefix;
	bool isGridPath;
	bool isPrintOutputFiles;
	bool isGeometryValues;
	bool isCalc2ndOrderMoments;
	bool isCalc3rdOrderMoments;
	bool isCalcHighOrderMoments;
	bool isReadGeo;
	bool isCalcMedian;
	bool isConcFile;
	bool isStreetVelocityFile;
	bool isUseMeasurePoints;
	bool isUseWale;
	bool isUseInitNeq;
	bool isSimulatePorousMedia;
	bool isD3Qxx;
	bool isTEnd;
	bool isTOut;
	bool isTStartOut;
	bool isTimeCalcMedStart;
	bool isTimeCalcMedEnd;
	bool isPressInID;
	bool isPressOutID;
	bool isPressInZ;
	bool isPressOutZ;
	bool isDiffOn;
	bool isDiffMod;
	bool isDiffusivity;
	bool isTemperatureInit;
	bool isTemperatureBC;
	//bool isViscosity;
	bool isVelocity;
	bool isViscosityRatio;
	bool isVelocityRatio;
	bool isDensityRatio;
	bool isPressRatio;
	bool isRealX;
	bool isRealY;
	bool isFactorPressBC;
	bool isGeometryFileC;
	bool isGeometryFileM;
	bool isGeometryFileF;
	bool isClockCycleForMP;
	bool isTimestepForMP;
	bool isForcingX;
	bool isForcingY;
	bool isForcingZ;
	bool isQuadricLimiterP;
	bool isQuadricLimiterM;
	bool isQuadricLimiterD;
	bool isCalcParticles;
	bool isParticleBasicLevel;
	bool isParticleInitLevel;
	bool isNumberOfParticles;
	bool isNeighborWSB;
	bool isStartXHotWall;
	bool isEndXHotWall;
	bool isPossNeighborFilesX;
	bool isPossNeighborFilesY;
	bool isPossNeighborFilesZ;
	bool isTimeDoCheckPoint;
	bool isTimeDoRestart;
	bool isDoCheckPoint;
	bool isDoRestart;
	bool isMaxLevel;
	bool isGridX;
	bool isGridY;
	bool isGridZ;
	bool isDistX;
	bool isDistY;
	bool isDistZ;
	bool isNeedInterface;
	bool isMainKernel;
	bool isMultiKernelOn;
	bool isMultiKernelLevel;
	bool isMultiKernelName;

};
#endif
