//#define MPI_LOGGING

// Stephas Branch

#include <mpi.h>
#if defined( MPI_LOGGING )
	#include <mpe.h>
#endif

#include <string>
#include <iostream>
#include <stdexcept>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>

#include "metis.h"

#include "Core/LbmOrGks.h"
#include "Core/Input/Input.h"
#include "Core/StringUtilities/StringUtil.h"

#include "VirtualFluids_GPU/LBM/Simulation.h"
#include "VirtualFluids_GPU/Communication/Communicator.h"
#include "VirtualFluids_GPU/DataStructureInitializer/GridReaderGenerator/GridGenerator.h"
#include "VirtualFluids_GPU/DataStructureInitializer/GridProvider.h"
#include "VirtualFluids_GPU/DataStructureInitializer/GridReaderFiles/GridReader.h"
#include "VirtualFluids_GPU/Parameter/Parameter.h"
#include "VirtualFluids_GPU/Output/FileWriter.h"

#include "global.h"

#include "geometries/Sphere/Sphere.h"
#include "geometries/VerticalCylinder/VerticalCylinder.h"
#include "geometries/Cuboid/Cuboid.h"
#include "geometries/TriangularMesh/TriangularMesh.h"
#include "geometries/Conglomerate/Conglomerate.h"
#include "geometries/TriangularMesh/TriangularMeshStrategy.h"

#include "grid/GridBuilder/LevelGridBuilder.h"
#include "grid/GridBuilder/MultipleGridBuilder.h"
#include "grid/BoundaryConditions/Side.h"
#include "grid/BoundaryConditions/BoundaryCondition.h"
#include "grid/GridFactory.h"

#include "io/SimulationFileWriter/SimulationFileWriter.h"
#include "io/GridVTKWriter/GridVTKWriter.h"
#include "io/STLReaderWriter/STLReader.h"
#include "io/STLReaderWriter/STLWriter.h"

#include "utilities/math/Math.h"
#include "utilities/communication.h"
#include "utilities/transformator/TransformatorImp.h"

std::string getGridPath(std::shared_ptr<Parameter> para, std::string Gridpath)
{
    if (para->getNumprocs() == 1)
        return Gridpath + "/";
    
    return Gridpath + "/" + StringUtil::toString(para->getMyID()) + "/";
}

void setParameters(std::shared_ptr<Parameter> para, std::unique_ptr<input::Input> &input)
{
	Communicator* comm = Communicator::getInstanz();

	para->setMaxDev(StringUtil::toInt(input->getValue("NumberOfDevices")));
	para->setNumprocs(comm->getNummberOfProcess());
	para->setDevices(StringUtil::toVector(input->getValue("Devices")));
	para->setMyID(comm->getPID());
	
	std::string _path = input->getValue("Path");
    std::string _prefix = input->getValue("Prefix");
    std::string _gridpath = input->getValue("GridPath");
    std::string gridPath = getGridPath(para, _gridpath);
    para->setOutputPath(_path);
    para->setOutputPrefix(_prefix);
    para->setFName(_path + "/" + _prefix);
    para->setPrintFiles(false);
    para->setPrintFiles(StringUtil::toBool(input->getValue("WriteGrid")));
    para->setGeometryValues(StringUtil::toBool(input->getValue("GeometryValues")));
    para->setCalc2ndOrderMoments(StringUtil::toBool(input->getValue("calc2ndOrderMoments")));
    para->setCalc3rdOrderMoments(StringUtil::toBool(input->getValue("calc3rdOrderMoments")));
    para->setCalcHighOrderMoments(StringUtil::toBool(input->getValue("calcHigherOrderMoments")));
    para->setReadGeo(StringUtil::toBool(input->getValue("ReadGeometry")));
    para->setCalcMedian(StringUtil::toBool(input->getValue("calcMedian")));
    para->setConcFile(StringUtil::toBool(input->getValue("UseConcFile")));
    para->setUseMeasurePoints(StringUtil::toBool(input->getValue("UseMeasurePoints")));
    para->setUseWale(StringUtil::toBool(input->getValue("UseWale")));
    para->setSimulatePorousMedia(StringUtil::toBool(input->getValue("SimulatePorousMedia")));
    para->setD3Qxx(StringUtil::toInt(input->getValue("D3Qxx")));
    para->setTEnd(StringUtil::toInt(input->getValue("TimeEnd")));
    para->setTOut(StringUtil::toInt(input->getValue("TimeOut")));
    para->setTStartOut(StringUtil::toInt(input->getValue("TimeStartOut")));
    para->setTimeCalcMedStart(StringUtil::toInt(input->getValue("TimeStartCalcMedian")));
    para->setTimeCalcMedEnd(StringUtil::toInt(input->getValue("TimeEndCalcMedian")));
    para->setPressInID(StringUtil::toInt(input->getValue("PressInID")));
    para->setPressOutID(StringUtil::toInt(input->getValue("PressOutID")));
    para->setPressInZ(StringUtil::toInt(input->getValue("PressInZ")));
    para->setPressOutZ(StringUtil::toInt(input->getValue("PressOutZ")));
    //////////////////////////////////////////////////////////////////////////
    para->setDiffOn(StringUtil::toBool(input->getValue("DiffOn")));
    para->setDiffMod(StringUtil::toInt(input->getValue("DiffMod")));
    para->setDiffusivity(StringUtil::toFloat(input->getValue("Diffusivity")));
    para->setTemperatureInit(StringUtil::toFloat(input->getValue("Temp")));
    para->setTemperatureBC(StringUtil::toFloat(input->getValue("TempBC")));
    //////////////////////////////////////////////////////////////////////////
    para->setViscosity(StringUtil::toFloat(input->getValue("Viscosity_LB")));
    para->setVelocity(StringUtil::toFloat(input->getValue("Velocity_LB")));
    para->setViscosityRatio(StringUtil::toFloat(input->getValue("Viscosity_Ratio_World_to_LB")));
    para->setVelocityRatio(StringUtil::toFloat(input->getValue("Velocity_Ratio_World_to_LB")));
    para->setDensityRatio(StringUtil::toFloat(input->getValue("Density_Ratio_World_to_LB")));
    para->setPressRatio(StringUtil::toFloat(input->getValue("Delta_Press")));
    para->setRealX(StringUtil::toFloat(input->getValue("SliceRealX")));
    para->setRealY(StringUtil::toFloat(input->getValue("SliceRealY")));
    para->setFactorPressBC(StringUtil::toFloat(input->getValue("dfpbc")));
    para->setGeometryFileC(input->getValue("GeometryC"));
    para->setGeometryFileM(input->getValue("GeometryM"));
    para->setGeometryFileF(input->getValue("GeometryF"));
    //////////////////////////////////////////////////////////////////////////
    para->setgeoVec(gridPath + input->getValue("geoVec"));
    para->setcoordX(gridPath + input->getValue("coordX"));
    para->setcoordY(gridPath + input->getValue("coordY"));
    para->setcoordZ(gridPath + input->getValue("coordZ"));
    para->setneighborX(gridPath + input->getValue("neighborX"));
    para->setneighborY(gridPath + input->getValue("neighborY"));
    para->setneighborZ(gridPath + input->getValue("neighborZ"));
    para->setscaleCFC(gridPath + input->getValue("scaleCFC"));
    para->setscaleCFF(gridPath + input->getValue("scaleCFF"));
    para->setscaleFCC(gridPath + input->getValue("scaleFCC"));
    para->setscaleFCF(gridPath + input->getValue("scaleFCF"));
    para->setscaleOffsetCF(gridPath + input->getValue("scaleOffsetCF"));
    para->setscaleOffsetFC(gridPath + input->getValue("scaleOffsetFC"));
    para->setgeomBoundaryBcQs(gridPath + input->getValue("geomBoundaryBcQs"));
    para->setgeomBoundaryBcValues(gridPath + input->getValue("geomBoundaryBcValues"));
    para->setinletBcQs(gridPath + input->getValue("inletBcQs"));
    para->setinletBcValues(gridPath + input->getValue("inletBcValues"));
    para->setoutletBcQs(gridPath + input->getValue("outletBcQs"));
    para->setoutletBcValues(gridPath + input->getValue("outletBcValues"));
    para->settopBcQs(gridPath + input->getValue("topBcQs"));
    para->settopBcValues(gridPath + input->getValue("topBcValues"));
    para->setbottomBcQs(gridPath + input->getValue("bottomBcQs"));
    para->setbottomBcValues(gridPath + input->getValue("bottomBcValues"));
    para->setfrontBcQs(gridPath + input->getValue("frontBcQs"));
    para->setfrontBcValues(gridPath + input->getValue("frontBcValues"));
    para->setbackBcQs(gridPath + input->getValue("backBcQs"));
    para->setbackBcValues(gridPath + input->getValue("backBcValues"));
    para->setnumberNodes(gridPath + input->getValue("numberNodes"));
    para->setLBMvsSI(gridPath + input->getValue("LBMvsSI"));
    //////////////////////////////gridPath + ////////////////////////////////////////////
    para->setmeasurePoints(gridPath + input->getValue("measurePoints"));
    para->setpropellerValues(gridPath + input->getValue("propellerValues"));
    para->setclockCycleForMP(StringUtil::toFloat(input->getValue("measureClockCycle")));
    para->settimestepForMP(StringUtil::toInt(input->getValue("measureTimestep")));
    para->setcpTop(gridPath + input->getValue("cpTop"));
    para->setcpBottom(gridPath + input->getValue("cpBottom"));
    para->setcpBottom2(gridPath + input->getValue("cpBottom2"));
    para->setConcentration(gridPath + input->getValue("Concentration"));
    //////////////////////////////////////////////////////////////////////////
    //Normals - Geometry
    para->setgeomBoundaryNormalX(gridPath + input->getValue("geomBoundaryNormalX"));
    para->setgeomBoundaryNormalY(gridPath + input->getValue("geomBoundaryNormalY"));
    para->setgeomBoundaryNormalZ(gridPath + input->getValue("geomBoundaryNormalZ"));
    //Normals - Inlet
    para->setInflowBoundaryNormalX(gridPath + input->getValue("inletBoundaryNormalX"));
    para->setInflowBoundaryNormalY(gridPath + input->getValue("inletBoundaryNormalY"));
    para->setInflowBoundaryNormalZ(gridPath + input->getValue("inletBoundaryNormalZ"));
    //Normals - Outlet
    para->setOutflowBoundaryNormalX(gridPath + input->getValue("outletBoundaryNormalX"));
    para->setOutflowBoundaryNormalY(gridPath + input->getValue("outletBoundaryNormalY"));
    para->setOutflowBoundaryNormalZ(gridPath + input->getValue("outletBoundaryNormalZ"));
    //////////////////////////////////////////////////////////////////////////
    //Forcing
    para->setForcing(StringUtil::toFloat(input->getValue("ForcingX")), StringUtil::toFloat(input->getValue("ForcingY")), StringUtil::toFloat(input->getValue("ForcingZ")));
    //////////////////////////////////////////////////////////////////////////
    //Particles
    para->setCalcParticles(StringUtil::toBool(input->getValue("calcParticles")));
    para->setParticleBasicLevel(StringUtil::toInt(input->getValue("baseLevel")));
    para->setParticleInitLevel(StringUtil::toInt(input->getValue("initLevel")));
    para->setNumberOfParticles(StringUtil::toInt(input->getValue("numberOfParticles")));
    para->setneighborWSB(gridPath + input->getValue("neighborWSB"));
    para->setStartXHotWall(StringUtil::toDouble(input->getValue("startXHotWall")));
    para->setEndXHotWall(StringUtil::toDouble(input->getValue("endXHotWall")));
    //////////////////////////////////////////////////////////////////////////
    //for Multi GPU
    if (para->getNumprocs() > 1)
    {
        ////////////////////////////////////////////////////////////////////////////
        ////1D domain decomposition
        //std::vector<std::string> sendProcNeighbors;
        //std::vector<std::string> recvProcNeighbors;
        //for (int i = 0; i<para->getNumprocs();i++)
        //{
        // sendProcNeighbors.push_back(gridPath + StringUtil::toString(i) + "s.dat");
        // recvProcNeighbors.push_back(gridPath + StringUtil::toString(i) + "r.dat");
        //}
        //para->setPossNeighborFiles(sendProcNeighbors, "send");
        //para->setPossNeighborFiles(recvProcNeighbors, "recv");
        //////////////////////////////////////////////////////////////////////////
        //3D domain decomposition
        std::vector<std::string> sendProcNeighborsX, sendProcNeighborsY, sendProcNeighborsZ;
        std::vector<std::string> recvProcNeighborsX, recvProcNeighborsY, recvProcNeighborsZ;
        for (int i = 0; i < para->getNumprocs(); i++)
        {
            sendProcNeighborsX.push_back(gridPath + StringUtil::toString(i) + "Xs.dat");
            sendProcNeighborsY.push_back(gridPath + StringUtil::toString(i) + "Ys.dat");
            sendProcNeighborsZ.push_back(gridPath + StringUtil::toString(i) + "Zs.dat");
            recvProcNeighborsX.push_back(gridPath + StringUtil::toString(i) + "Xr.dat");
            recvProcNeighborsY.push_back(gridPath + StringUtil::toString(i) + "Yr.dat");
            recvProcNeighborsZ.push_back(gridPath + StringUtil::toString(i) + "Zr.dat");
        }
        para->setPossNeighborFilesX(sendProcNeighborsX, "send");
        para->setPossNeighborFilesY(sendProcNeighborsY, "send");
        para->setPossNeighborFilesZ(sendProcNeighborsZ, "send");
        para->setPossNeighborFilesX(recvProcNeighborsX, "recv");
        para->setPossNeighborFilesY(recvProcNeighborsY, "recv");
        para->setPossNeighborFilesZ(recvProcNeighborsZ, "recv");
    }
    //////////////////////////////////////////////////////////////////////////
    //para->setkFull(             input->getValue( "kFull" ));
    //para->setgeoFull(           input->getValue( "geoFull" ));
    //para->setnoSlipBcPos(       input->getValue( "noSlipBcPos" ));
    //para->setnoSlipBcQs(          input->getValue( "noSlipBcQs" ));
    //para->setnoSlipBcValues(      input->getValue( "noSlipBcValues" ));
    //para->setnoSlipBcValue(     input->getValue( "noSlipBcValue" ));
    //para->setslipBcPos(         input->getValue( "slipBcPos" ));
    //para->setslipBcQs(          input->getValue( "slipBcQs" ));
    //para->setslipBcValue(       input->getValue( "slipBcValue" ));
    //para->setpressBcPos(        input->getValue( "pressBcPos" ));
    //para->setpressBcQs(           input->getValue( "pressBcQs" ));
    //para->setpressBcValues(       input->getValue( "pressBcValues" ));
    //para->setpressBcValue(      input->getValue( "pressBcValue" ));
    //para->setvelBcQs(             input->getValue( "velBcQs" ));
    //para->setvelBcValues(         input->getValue( "velBcValues" ));
    //para->setpropellerCylinder( input->getValue( "propellerCylinder" ));
    //para->setpropellerQs(		 input->getValue( "propellerQs"      ));
    //para->setwallBcQs(            input->getValue( "wallBcQs"         ));
    //para->setwallBcValues(        input->getValue( "wallBcValues"     ));
    //para->setperiodicBcQs(        input->getValue( "periodicBcQs"     ));
    //para->setperiodicBcValues(    input->getValue( "periodicBcValues" ));
    //cout << "Try this: " << para->getgeomBoundaryBcValues() << endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //Restart
    para->setTimeDoCheckPoint(StringUtil::toInt(input->getValue("TimeDoCheckPoint")));
    para->setTimeDoRestart(StringUtil::toInt(input->getValue("TimeDoRestart")));
    para->setDoCheckPoint(StringUtil::toBool(input->getValue("DoCheckPoint")));
    para->setDoRestart(StringUtil::toBool(input->getValue("DoRestart")));
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    para->setMaxLevel(StringUtil::toInt(input->getValue("NOGL")));
    para->setGridX(StringUtil::toVector(input->getValue("GridX")));                           
    para->setGridY(StringUtil::toVector(input->getValue("GridY")));                           
    para->setGridZ(StringUtil::toVector(input->getValue("GridZ")));                  
    para->setDistX(StringUtil::toVector(input->getValue("DistX")));                  
    para->setDistY(StringUtil::toVector(input->getValue("DistY")));                  
    para->setDistZ(StringUtil::toVector(input->getValue("DistZ")));                

    para->setNeedInterface(std::vector<bool>{true, true, true, true, true, true});
}



void multipleLevel(const std::string& configPath)
{
    //std::ofstream logFile( "F:/Work/Computations/gridGenerator/grid/gridGeneratorLog.txt" );
    std::ofstream logFile( "grid/gridGeneratorLog.txt" );
    logging::Logger::addStream(&logFile);

    logging::Logger::addStream(&std::cout);
    logging::Logger::setDebugLevel(logging::Logger::Level::INFO_LOW);
    logging::Logger::timeStamp(logging::Logger::ENABLE);
    logging::Logger::enablePrintedRankNumbers(logging::Logger::ENABLE);

    //UbLog::reportingLevel() = UbLog::logLevelFromString("DEBUG5");

    auto gridFactory = GridFactory::make();
    gridFactory->setGridStrategy(Device::CPU);
    //gridFactory->setTriangularMeshDiscretizationMethod(TriangularMeshDiscretizationMethod::RAYCASTING);
    gridFactory->setTriangularMeshDiscretizationMethod(TriangularMeshDiscretizationMethod::POINT_IN_OBJECT);
    //gridFactory->setTriangularMeshDiscretizationMethod(TriangularMeshDiscretizationMethod::POINT_UNDER_TRIANGLE);

    auto gridBuilder = MultipleGridBuilder::makeShared(gridFactory);
    
    SPtr<Parameter> para = Parameter::make();
    SPtr<GridProvider> gridGenerator;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool useGridGenerator = true;

    if(useGridGenerator){

        enum testCase{ 
            DrivAer,
            DLC,
            MultiGPU,
            MetisTest
        };

        int testcase = MetisTest;
        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if( testcase == DrivAer )
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        {
            real dx = 0.2;
            real vx = 0.05;

            TriangularMesh* DrivAerSTL = TriangularMesh::make("F:/Work/Computations/gridGenerator/stl/DrivAer_Fastback_Coarse.stl");
            //TriangularMesh* triangularMesh = TriangularMesh::make("M:/TestGridGeneration/STL/DrivAer_NoSTLGroups.stl");
            //TriangularMesh* triangularMesh = TriangularMesh::make("M:/TestGridGeneration/STL/DrivAer_Coarse.stl");
            //TriangularMesh* DrivAerSTL = TriangularMesh::make("stl/DrivAer_Fastback_Coarse.stl");

            TriangularMesh* DrivAerRefBoxSTL = TriangularMesh::make("F:/Work/Computations/gridGenerator/stl/DrivAer_REF_BOX_Adrea.stl");
            //TriangularMesh* DrivAerRefBoxSTL = TriangularMesh::make("stl/DrivAer_REF_BOX_Adrea.stl");

            real z0 = 0.318+0.5*dx;

            gridBuilder->addCoarseGrid(- 5.0, -5.0, 0.0 - z0,
                                        15.0,  5.0, 5.0 - z0, dx);  // DrivAer

            //Object* floorBox = new Cuboid( -0.3, -1, -1, 4.0, 1, 0.2 );
            //Object* wakeBox  = new Cuboid(  3.5, -1, -1, 5.5, 1, 0.8 );

            //Conglomerate* refRegion = new Conglomerate();

            //refRegion->add(floorBox);
            //refRegion->add(wakeBox);
            //refRegion->add(DrivAerRefBoxSTL);

            gridBuilder->setNumberOfLayers(10,8);
            gridBuilder->addGrid(DrivAerRefBoxSTL, 2);
        
            //gridBuilder->setNumberOfLayers(10,8);
            //gridBuilder->addGrid(DrivAerSTL, 5);

            gridBuilder->addGeometry(DrivAerSTL);

            gridBuilder->setPeriodicBoundaryCondition(false, false, false);

            gridBuilder->buildGrids(LBM, true); // buildGrids() has to be called before setting the BCs!!!!

            //////////////////////////////////////////////////////////////////////////

            gridBuilder->setVelocityBoundaryCondition(SideType::PY, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MY, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::PZ, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MZ, vx , 0.0, 0.0);

            gridBuilder->setPressureBoundaryCondition(SideType::PX, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MX, vx, 0.0, 0.0);

            gridBuilder->setVelocityBoundaryCondition(SideType::GEOMETRY, 0.0, 0.0, 0.0);
            
            //////////////////////////////////////////////////////////////////////////

            SPtr<Grid> grid = gridBuilder->getGrid(gridBuilder->getNumberOfLevels() - 1);

            gridBuilder->getGeometryBoundaryCondition(gridBuilder->getNumberOfLevels() - 1)->setTangentialVelocityForPatch( grid, 4, 0.0075, -2.0, 0.0,
                                                                                                                                     0.0075,  2.0, 0.0, -vx, 0.318);
            gridBuilder->getGeometryBoundaryCondition(gridBuilder->getNumberOfLevels() - 1)->setTangentialVelocityForPatch( grid, 3, 2.793 , -2.0, 0.0,
                                                                                                                                     2.793 ,  2.0, 0.0, -vx, 0.318);

            //////////////////////////////////////////////////////////////////////////

            gridBuilder->writeGridsToVtk("F:/Work/Computations/gridGenerator/grid/DrivAer_Grid");
            gridBuilder->writeArrows    ("F:/Work/Computations/gridGenerator/grid/DrivAer_Grid_arrow");

            //SimulationFileWriter::write("D:/GRIDGENERATION/files/", gridBuilder, FILEFORMAT::ASCII);
            //SimulationFileWriter::write("C:/Users/lenz/Desktop/Work/gridGenerator/grid/", gridBuilder, FILEFORMAT::ASCII);
            SimulationFileWriter::write("F:/Work/Computations/gridGenerator/grid/", gridBuilder, FILEFORMAT::BINARY);
            //SimulationFileWriter::write("grid/", gridBuilder, FILEFORMAT::ASCII);

            return;

            gridGenerator = GridGenerator::make(gridBuilder, para);
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if( testcase == DLC )
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        {
			real velocityRatio = 594.093427;

			real dx = 0.2;
			real vx = 0.065272188;

			real z0 = 0.24395 + 0.5*dx;

            std::vector<uint> ignorePatches = { 152, 153, 154 };

            //TriangularMesh* VW370_SERIE_STL = TriangularMesh::make("C:/Users/lenz/Desktop/Work/gridGenerator/stl/VW370_SERIE.stl", ignorePatches);
            TriangularMesh* VW370_SERIE_STL = TriangularMesh::make("stl/VW370_SERIE.stl", ignorePatches);

            //TriangularMesh* DLC_RefBox = TriangularMesh::make("C:/Users/lenz/Desktop/Work/gridGenerator/stl/DLC_RefBox.stl");

            //TriangularMesh* DLC_RefBox_1 = TriangularMesh::make("C:/Users/lenz/Desktop/Work/gridGenerator/stl/DLC_RefBox_withWake/DLC_RefBox_withWake_4m.stl");
            //TriangularMesh* DLC_RefBox_2 = TriangularMesh::make("C:/Users/lenz/Desktop/Work/gridGenerator/stl/DLC_RefBox_withWake/DLC_RefBox_withWake_3m.stl");
            //TriangularMesh* DLC_RefBox_3 = TriangularMesh::make("C:/Users/lenz/Desktop/Work/gridGenerator/stl/DLC_RefBox_withWake/DLC_RefBox_withWake_2m.stl");
            //TriangularMesh* DLC_RefBox_4 = TriangularMesh::make("C:/Users/lenz/Desktop/Work/gridGenerator/stl/DLC_RefBox_withWake/DLC_RefBox_withWake_1m.stl");

            //TriangularMesh* DLC_RefBox_Level_3 = TriangularMesh::make("C:/Users/lenz/Desktop/Work/gridGenerator/stl/DLC/DLC_RefBox_Level_3.stl");
            //TriangularMesh* DLC_RefBox_Level_4 = TriangularMesh::make("C:/Users/lenz/Desktop/Work/gridGenerator/stl/DLC/DLC_RefBox_Level_4.stl");
            //TriangularMesh* DLC_RefBox_Level_5 = TriangularMesh::make("C:/Users/lenz/Desktop/Work/gridGenerator/stl/DLC/DLC_RefBox_Level_5.stl");

            TriangularMesh* DLC_RefBox_Level_3 = TriangularMesh::make("stl/DLC/DLC_RefBox_Level_3.stl");
            TriangularMesh* DLC_RefBox_Level_4 = TriangularMesh::make("stl/DLC/DLC_RefBox_Level_4.stl");
            TriangularMesh* DLC_RefBox_Level_5 = TriangularMesh::make("stl/DLC/DLC_RefBox_Level_5.stl");

            //TriangularMesh* VW370_SERIE_STL = TriangularMesh::make("stl/VW370_SERIE.stl", ignorePatches);
            //TriangularMesh* DLC_RefBox = TriangularMesh::make("stl/DLC_RefBox.lnx.stl");
            //TriangularMesh* DLC_RefBox_4 = TriangularMesh::make("stl/DLC_RefBox_withWake/DLC_RefBox_withWake_1m.lnx.stl");

            gridBuilder->addCoarseGrid(-30.0, -20.0,  0.0 - z0,
                                        50.0,  20.0, 25.0 - z0, dx);
            
            gridBuilder->setNumberOfLayers(10,8);
            gridBuilder->addGrid( new Cuboid( - 6.6, -6, -0.7, 20.6 , 6, 5.3  ), 1 );
            gridBuilder->addGrid( new Cuboid( -3.75, -3, -0.7, 11.75, 3, 2.65 ), 2 );

            gridBuilder->setNumberOfLayers(10,8);
            gridBuilder->addGrid(DLC_RefBox_Level_3, 3);
            gridBuilder->addGrid(DLC_RefBox_Level_4, 4);
        
            Conglomerate* refinement = new Conglomerate();
            refinement->add(DLC_RefBox_Level_5);
            refinement->add(VW370_SERIE_STL);

            gridBuilder->setNumberOfLayers(10,8);
            gridBuilder->addGrid(refinement, 5);

            gridBuilder->addGeometry(VW370_SERIE_STL);

            gridBuilder->setPeriodicBoundaryCondition(false, false, false);

            gridBuilder->buildGrids(LBM, true); // buildGrids() has to be called before setting the BCs!!!!

            //////////////////////////////////////////////////////////////////////////

            gridBuilder->setVelocityBoundaryCondition(SideType::PY, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MY, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::PZ, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MZ, vx , 0.0, 0.0);

            gridBuilder->setPressureBoundaryCondition(SideType::PX, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MX, vx, 0.0, 0.0);

            gridBuilder->setVelocityBoundaryCondition(SideType::GEOMETRY, 0.0, 0.0, 0.0);
            
            //////////////////////////////////////////////////////////////////////////

            SPtr<Grid> grid = gridBuilder->getGrid(gridBuilder->getNumberOfLevels() - 1);

            real wheelsFrontX = -0.081;
            real wheelsRearX  =  2.5486;

            real wheelsFrontZ =  0.0504;
            real wheelsRearZ  =  0.057;

            real wheelsRadius =  0.318;

			real wheelRotationFrequency = 1170.74376 / 60.0;

			real wheelTangentialVelocity = -2.0 * M_PI * wheelsRadius * wheelRotationFrequency / velocityRatio;

            std::vector<uint> frontWheelPatches = { 71, 86, 87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97, 159 };
            std::vector<uint> rearWheelPatches  = { 82, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 160 };

            for( uint patch : frontWheelPatches ){
                gridBuilder->getGeometryBoundaryCondition(gridBuilder->getNumberOfLevels() - 1)->setTangentialVelocityForPatch( grid, patch, wheelsFrontX, -2.0, wheelsFrontZ,
                                                                                                                                             wheelsFrontX,  2.0, wheelsFrontZ, 
					                                                                                                                         wheelTangentialVelocity, wheelsRadius);
            }

            for( uint patch : rearWheelPatches ){
                gridBuilder->getGeometryBoundaryCondition(gridBuilder->getNumberOfLevels() - 1)->setTangentialVelocityForPatch( grid, patch, wheelsRearX , -2.0, wheelsRearZ ,
                                                                                                                                             wheelsRearX ,  2.0, wheelsRearZ , 
					                                                                                                                         wheelTangentialVelocity, wheelsRadius);
            }

            //////////////////////////////////////////////////////////////////////////

            //gridBuilder->writeGridsToVtk("C:/Users/lenz/Desktop/Work/gridGenerator/grid/DLC_Grid");
            //gridBuilder->writeArrows    ("C:/Users/lenz/Desktop/Work/gridGenerator/grid/DLC_Grid_arrow");

            gridBuilder->writeGridsToVtk("grid/DLC_Grid");
            gridBuilder->writeArrows    ("grid/DLC_Grid_arrow");

            //SimulationFileWriter::write("D:/GRIDGENERATION/files/", gridBuilder, FILEFORMAT::ASCII);
            //SimulationFileWriter::write("C:/Users/lenz/Desktop/Work/gridGenerator/grid/", gridBuilder, FILEFORMAT::ASCII);
            SimulationFileWriter::write("grid/", gridBuilder, FILEFORMAT::ASCII);

            gridGenerator = GridGenerator::make(gridBuilder, para);
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if( testcase == MultiGPU )
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        {
            //const uint generatePart = 1;
            const uint generatePart = Communicator::getInstanz()->getPID();
            
            std::ofstream logFile2;
            
            if( generatePart == 0 )
                //logFile2.open( "F:/Work/Computations/gridGenerator/grid/0/gridGeneratorLog.txt" );
                logFile2.open( "grid/0/gridGeneratorLog.txt" );
            
            if( generatePart == 1 )
                //logFile2.open( "F:/Work/Computations/gridGenerator/grid/1/gridGeneratorLog.txt" );
                logFile2.open( "grid/1/gridGeneratorLog.txt" );

            logging::Logger::addStream(&logFile2);

            real dx = 1.0 / 40.0;
            real vx = 0.05;

            //TriangularMesh* triangularMesh = TriangularMesh::make("F:/Work/Computations/gridGenerator/stl/ShpereNotOptimal.stl");
            TriangularMesh* triangularMesh = TriangularMesh::make("stl/ShpereNotOptimal.lnx.stl");

            // all
            //gridBuilder->addCoarseGrid(-2, -2, -2,  
            //                            4,  2,  2, dx);

            real overlap = 10.0 * dx;

            if( generatePart == 0 )
                gridBuilder->addCoarseGrid(-2.0          , -2.0, -2.0,  
                                            0.5 + overlap,  2.0,  2.0, dx);

            if( generatePart == 1 )
                gridBuilder->addCoarseGrid( 0.5 - overlap, -2.0, -2.0,  
                                            4.0          ,  2.0,  2.0, dx);


            gridBuilder->setNumberOfLayers(10,8);
            gridBuilder->addGrid(triangularMesh, 1);

            gridBuilder->addGeometry(triangularMesh);
            
            if( generatePart == 0 )
                gridBuilder->setSubDomainBox( std::make_shared<BoundingBox>( -2.0, 0.5, 
                                                                             -2.0, 2.0, 
                                                                             -2.0, 2.0 ) );
            
            if( generatePart == 1 )
                gridBuilder->setSubDomainBox( std::make_shared<BoundingBox>(  0.5, 4.0, 
                                                                             -2.0, 2.0, 
                                                                             -2.0, 2.0 ) );

            gridBuilder->setPeriodicBoundaryCondition(false, false, false);

            gridBuilder->buildGrids(LBM, true); // buildGrids() has to be called before setting the BCs!!!!
            
            if( generatePart == 0 ){
                gridBuilder->findCommunicationIndices(CommunicationDirections::PX);
                gridBuilder->setCommunicationProcess(CommunicationDirections::PX, 1);
            }
            
            if( generatePart == 1 ){
                gridBuilder->findCommunicationIndices(CommunicationDirections::MX);
                gridBuilder->setCommunicationProcess(CommunicationDirections::MX, 0);
            }

            //////////////////////////////////////////////////////////////////////////

            gridBuilder->setVelocityBoundaryCondition(SideType::PY, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MY, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::PZ, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MZ, vx , 0.0, 0.0);

            if (generatePart == 0) {
                gridBuilder->setVelocityBoundaryCondition(SideType::MX, vx, 0.0, 0.0);
            }
            if (generatePart == 1) {
                gridBuilder->setPressureBoundaryCondition(SideType::PX, 0.0);
            }

            gridBuilder->setVelocityBoundaryCondition(SideType::GEOMETRY, 0.0, 0.0, 0.0);
        
            //////////////////////////////////////////////////////////////////////////

            if (generatePart == 0) {
                //gridBuilder->writeGridsToVtk("F:/Work/Computations/gridGenerator/grid/0/Test_");
                //gridBuilder->writeArrows    ("F:/Work/Computations/gridGenerator/grid/0/Test_Arrow");
            }
            if (generatePart == 1) {
                //gridBuilder->writeGridsToVtk("F:/Work/Computations/gridGenerator/grid/1/Test_");
                //gridBuilder->writeArrows    ("F:/Work/Computations/gridGenerator/grid/1/Test_Arrow");
            }

            if (generatePart == 0)
                //SimulationFileWriter::write("F:/Work/Computations/gridGenerator/grid/0/", gridBuilder, FILEFORMAT::ASCII);
                SimulationFileWriter::write("grid/0/", gridBuilder, FILEFORMAT::ASCII);
            if (generatePart == 1)
                //SimulationFileWriter::write("F:/Work/Computations/gridGenerator/grid/1/", gridBuilder, FILEFORMAT::ASCII);
                SimulationFileWriter::write("grid/1/", gridBuilder, FILEFORMAT::ASCII);

            //return;

            gridGenerator = GridGenerator::make(gridBuilder, para);
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if( testcase == MetisTest )
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        {
            //const uint generatePart = 1;
            const uint generatePart = Communicator::getInstanz()->getPID();
            
            real dx = 1.0 / 20.0;
            real vx = 0.05;

            TriangularMesh* triangularMesh = TriangularMesh::make("F:/Work/Computations/gridGenerator/stl/ShpereNotOptimal.stl");
            //TriangularMesh* triangularMesh = TriangularMesh::make("stl/ShpereNotOptimal.lnx.stl");

            // all
            //gridBuilder->addCoarseGrid(-2, -2, -2,  
            //                            4,  2,  2, dx);

            real overlap = 10.0 * dx;

            gridBuilder->addCoarseGrid(-2.0, -2.0, -2.0,  
                                        4.0,  2.0,  2.0, dx);


            gridBuilder->setNumberOfLayers(10,8);
            gridBuilder->addGrid(triangularMesh, 1);

            gridBuilder->addGeometry(triangularMesh);

            gridBuilder->setPeriodicBoundaryCondition(false, false, false);

            gridBuilder->buildGrids(LBM, true); // buildGrids() has to be called before setting the BCs!!!!

            //////////////////////////////////////////////////////////////////////////

            gridBuilder->setVelocityBoundaryCondition(SideType::PY, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MY, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::PZ, vx , 0.0, 0.0);
            gridBuilder->setVelocityBoundaryCondition(SideType::MZ, vx , 0.0, 0.0);

            gridBuilder->setVelocityBoundaryCondition(SideType::MX, vx, 0.0, 0.0);
            gridBuilder->setPressureBoundaryCondition(SideType::PX, 0.0);

            gridBuilder->setVelocityBoundaryCondition(SideType::GEOMETRY, 0.0, 0.0, 0.0);
        
            //////////////////////////////////////////////////////////////////////////

            gridBuilder->writeGridsToVtk("F:/Work/Computations/gridGenerator/grid/Test_");
            //gridBuilder->writeArrows    ("F:/Work/Computations/gridGenerator/grid/Test_Arrow");

            //SimulationFileWriter::write("F:/Work/Computations/gridGenerator/grid/", gridBuilder, FILEFORMAT::ASCII);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            if(false)
            {

                auto getParentIndex = [&] (uint index, uint level) -> uint
                {
                    SPtr<Grid> grid = gridBuilder->getGrid( level );

                    if( level != 0 )
                    {
                        real x, y, z;
                        grid->transIndexToCoords(index, x, y, z);

                        SPtr<Grid> coarseGrid = gridBuilder->getGrid(level - 1);

                        for (const auto dir : DistributionHelper::getDistribution27())
                        {
                            if (std::abs(dir[0]) < 0.5 || std::abs(dir[1]) < 0.5 || std::abs(dir[2]) < 0.5) continue;

                            real coarseX = x + dir[0] * 0.5 * grid->getDelta();
                            real coarseY = y + dir[1] * 0.5 * grid->getDelta();
                            real coarseZ = z + dir[2] * 0.5 * grid->getDelta();

                            // check if close enough to coarse grid coordinates
                            if( 0.01 * grid->getDelta() < std::abs(         (coarseGrid->getStartX() - coarseX) / grid->getDelta() 
                                                                  - lround( (coarseGrid->getStartX() - coarseX) / grid->getDelta() ) ) ) continue;
                            if( 0.01 * grid->getDelta() < std::abs(         (coarseGrid->getStartY() - coarseY) / grid->getDelta() 
                                                                  - lround( (coarseGrid->getStartY() - coarseY) / grid->getDelta() ) ) ) continue;
                            if( 0.01 * grid->getDelta() < std::abs(         (coarseGrid->getStartZ() - coarseZ) / grid->getDelta() 
                                                                  - lround( (coarseGrid->getStartZ() - coarseZ) / grid->getDelta() ) ) ) continue;

                            uint parentIndex = coarseGrid->transCoordToIndex( coarseX, coarseY, coarseZ);

                            return parentIndex;
                        }
                    }

                    return INVALID_INDEX;
                };


                std::vector<idx_t> xadj;
                std::vector<idx_t> adjncy;

                std::vector<idx_t> vwgt;
                std::vector<idx_t> adjwgt;

                idx_t vertexCounter = 0;
                uint edgeCounter = 0;

                std::cout << "Checkpoint 1:" << std::endl;

                std::vector< std::vector<idx_t> > vertexIndex( gridBuilder->getNumberOfLevels() );

                std::vector< uint > startVerticesPerLevel;;

                for( uint level = 0; level < gridBuilder->getNumberOfLevels(); level++ )
                {
                    SPtr<Grid> grid = gridBuilder->getGrid( level );

                    vertexIndex[level].resize( grid->getSize() );

                    startVerticesPerLevel.push_back(vertexCounter);

                    for (uint index = 0; index < grid->getSize(); index++)
                    {
                        if (grid->getSparseIndex(index) == INVALID_INDEX)
                        {
                            vertexIndex[level][index] = INVALID_INDEX;
                            continue;
                        }

                        uint parentIndex = getParentIndex(index, level);

                        if( parentIndex != INVALID_INDEX )
                        {
                            SPtr<Grid> coarseGrid = gridBuilder->getGrid(level - 1);

                            if( coarseGrid->getFieldEntry(parentIndex) == FLUID_CFC ||
                                coarseGrid->getFieldEntry(parentIndex) == FLUID_FCC ||
                                coarseGrid->getFieldEntry(parentIndex) == STOPPER_COARSE_UNDER_FINE )
                            {
                                //vertexIndex[level][index] = INVALID_INDEX;
                                vertexIndex[level][index] = vertexIndex[level - 1][parentIndex];
                                continue;
                            }
                        }

                        vertexIndex[level][index] = vertexCounter;

                        vwgt.push_back( std::pow(2, level) );
                        //vwgt.push_back( std::pow(2, 2*level) );
                        vertexCounter++;
                    }

                }

                //////////////////////////////////////////////////////////////////////////
                //for( uint level = 0; level < gridBuilder->getNumberOfLevels(); level++ )
                //{
                //    SPtr<Grid> grid = gridBuilder->getGrid( level );

                //    for (uint index = 0; index < grid->getSize(); index++)
                //    {
                //        grid->setFieldEntry(index, vertexIndex[level][index] >= startVerticesPerLevel[level] && vertexIndex[level][index] != INVALID_INDEX);
                //    }
                //}

                //gridBuilder->writeGridsToVtk("F:/Work/Computations/gridGenerator/grid/VertexIndex_");

                //return;
                //////////////////////////////////////////////////////////////////////////


                std::cout << "Checkpoint 2:" << std::endl;
                
                for( uint level = 0; level < gridBuilder->getNumberOfLevels(); level++ )
                {
                    SPtr<Grid> grid = gridBuilder->getGrid( level );

                    for (uint index = 0; index < grid->getSize(); index++)
                    {
                        //if (grid->getSparseIndex(index) == INVALID_INDEX) continue;

                        if( vertexIndex[level][index] == INVALID_INDEX ) continue;

                        if( vertexIndex[level][index] < startVerticesPerLevel[level] ) continue;

                        xadj.push_back(edgeCounter);

                        real x, y, z;
                        grid->transIndexToCoords(index, x, y, z);

                        for (const auto dir : DistributionHelper::getDistribution27())
                        {
                            const uint neighborIndex = grid->transCoordToIndex(x + dir[0] * grid->getDelta(), 
                                                                               y + dir[1] * grid->getDelta(), 
                                                                               z + dir[2] * grid->getDelta());

                            if (neighborIndex == INVALID_INDEX) continue;

                            if (neighborIndex == index) continue;

                            if( vertexIndex[level][neighborIndex] == INVALID_INDEX ) continue;

                            adjncy.push_back( vertexIndex[level][neighborIndex] );
                            adjwgt.push_back( std::pow(2, level) );

                            edgeCounter++;
                        }

                        if( grid->getFieldEntry(index) == FLUID_CFC ||
                            grid->getFieldEntry(index) == FLUID_FCC ||
                            grid->getFieldEntry(index) == STOPPER_COARSE_UNDER_FINE )

                        {
                            SPtr<Grid> fineGrid = gridBuilder->getGrid(level + 1);

                            for (const auto dir : DistributionHelper::getDistribution27())
                            {
                                if (std::abs(dir[0]) < 0.5 || std::abs(dir[1]) < 0.5 || std::abs(dir[2]) < 0.5) continue;

                                real fineX = x + dir[0] * 0.25 * grid->getDelta();
                                real fineY = y + dir[1] * 0.25 * grid->getDelta();
                                real fineZ = z + dir[2] * 0.25 * grid->getDelta();

                                uint childIndex = fineGrid->transCoordToIndex(fineX, fineY, fineZ);

                                if( fineGrid->getFieldEntry(childIndex) == INVALID_INDEX ) continue;
                                if( vertexIndex[level + 1][childIndex]  == INVALID_INDEX ) continue;

                                for (const auto dir : DistributionHelper::getDistribution27())
                                {
                                    const uint neighborIndex = fineGrid->transCoordToIndex( fineX + dir[0] * fineGrid->getDelta(), 
                                                                                            fineY + dir[1] * fineGrid->getDelta(), 
                                                                                            fineZ + dir[2] * fineGrid->getDelta() );

                                    if(neighborIndex == INVALID_INDEX) continue;

                                    if (neighborIndex == childIndex) continue;

                                    if( vertexIndex[level + 1][neighborIndex] == INVALID_INDEX ) continue;

                                    adjncy.push_back( vertexIndex[level + 1][neighborIndex] );
                                    adjwgt.push_back( std::pow(2, level) );

                                    edgeCounter++;
                                }
                            }
                        }
                    }
                }

                xadj.push_back( edgeCounter );

                std::cout << "Checkpoint 3:" << std::endl;
                
                idx_t nWeights  = 1;
                idx_t nParts    = 4;
                idx_t objval    = 0;

                std::vector<idx_t> part( vertexCounter );
                
                std::cout << vertexCounter << std::endl;
                std::cout << edgeCounter << std::endl;
                std::cout << xadj.size()  << std::endl;
                std::cout << adjncy.size() << std::endl;

                //int ret = METIS_PartGraphRecursive(&vertexCounter, &nWeights, xadj.data(), adjncy.data(),
                // 				                   vwgt.data(), NULL, adjwgt.data(), &nParts, 
                //                                   NULL, NULL, NULL, &objval, part.data());

                int ret = METIS_PartGraphKway(&vertexCounter, &nWeights, xadj.data(), adjncy.data(),
                 				              vwgt.data(), NULL, NULL/*adjwgt.data()*/, &nParts, 
                                              NULL, NULL, NULL, &objval, part.data());

                std::cout << "objval:" << objval << std::endl;

                std::cout << "Checkpoint 4:" << std::endl;

                //uint partCounter = 0;
                
                for( uint level = 0; level < gridBuilder->getNumberOfLevels(); level++ )
                {
                    SPtr<Grid> grid = gridBuilder->getGrid( level );

                    for (uint index = 0; index < grid->getSize(); index++)
                    {
                        if (grid->getSparseIndex(index) == INVALID_INDEX) continue;

                        grid->setFieldEntry(index, part[vertexIndex[level][index]]);

                        //partCounter++;
                    }
                }

                std::cout << "Checkpoint 5:" << std::endl;

                gridBuilder->writeGridsToVtk("F:/Work/Computations/gridGenerator/grid/Partition_");

            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            {

                for( int level = gridBuilder->getNumberOfLevels()-1; level >= 0 ; level-- )
                {
                    std::vector< std::vector<idx_t> > vertexIndex( gridBuilder->getNumberOfLevels() );

                    std::vector<idx_t> xadj;
                    std::vector<idx_t> adjncy;

                    std::vector<idx_t> vwgt;
                    std::vector<idx_t> adjwgt;

                    idx_t vertexCounter = 0;
                    uint edgeCounter = 0;

                    SPtr<Grid> grid = gridBuilder->getGrid( level );

                    vertexIndex[level].resize( grid->getSize() );

                    for (uint index = 0; index < grid->getSize(); index++)
                    {
                        if (grid->getSparseIndex(index) == INVALID_INDEX)
                        {
                            vertexIndex[level][index] = INVALID_INDEX;
                            continue;
                        }

                        vertexIndex[level][index] = vertexCounter;

                        vwgt.push_back( std::pow(2, level) );
                        //vwgt.push_back( std::pow(2, 2*level) );
                        vertexCounter++;
                    }

                    for (uint index = 0; index < grid->getSize(); index++)
                    {
                        //if (grid->getSparseIndex(index) == INVALID_INDEX) continue;

                        if( vertexIndex[level][index] == INVALID_INDEX ) continue;

                        xadj.push_back(edgeCounter);

                        real x, y, z;
                        grid->transIndexToCoords(index, x, y, z);

                        for (const auto dir : DistributionHelper::getDistribution27())
                        {
                            const uint neighborIndex = grid->transCoordToIndex(x + dir[0] * grid->getDelta(), 
                                                                               y + dir[1] * grid->getDelta(), 
                                                                               z + dir[2] * grid->getDelta());

                            if (neighborIndex == INVALID_INDEX) continue;

                            if (neighborIndex == index) continue;

                            if( vertexIndex[level][neighborIndex] == INVALID_INDEX ) continue;

                            adjncy.push_back( vertexIndex[level][neighborIndex] );
                            adjwgt.push_back( std::pow(2, level) );

                            edgeCounter++;
                        }
                    }

                    xadj.push_back( edgeCounter );

                    std::cout << "Checkpoint 3:" << std::endl;
                
                    idx_t nWeights  = 1;
                    idx_t nParts    = 4;
                    idx_t objval    = 0;

                    std::vector<idx_t> part( vertexCounter );
                
                    std::cout << vertexCounter << std::endl;
                    std::cout << edgeCounter << std::endl;
                    std::cout << xadj.size()  << std::endl;
                    std::cout << adjncy.size() << std::endl;

                    int ret = METIS_PartGraphRecursive(&vertexCounter, &nWeights, xadj.data(), adjncy.data(),
                     				                   NULL/*vwgt.data()*/, NULL, NULL/*adjwgt.data()*/, &nParts, 
                                                       NULL, NULL, NULL, &objval, part.data());

                    //int ret = METIS_PartGraphKway(&vertexCounter, &nWeights, xadj.data(), adjncy.data(),
                 			//	                  NULL/*vwgt.data()*/, NULL, NULL/*adjwgt.data()*/, &nParts, 
                    //                              NULL, NULL, NULL, &objval, part.data());

                    std::cout << "objval:" << objval << std::endl;

                    std::cout << "Checkpoint 4:" << std::endl;

                    for (uint index = 0; index < grid->getSize(); index++)
                    {
                        if (vertexIndex[level][index] == INVALID_INDEX) continue;

                        if( grid->getFieldEntry(index) == FLUID_CFC ||
                            grid->getFieldEntry(index) == FLUID_FCC ||
                            grid->getFieldEntry(index) == STOPPER_COARSE_UNDER_FINE )
                        {
                            SPtr<Grid> fineGrid = gridBuilder->getGrid(level+1);
                            
                            real x, y, z;
                            grid->transIndexToCoords(index, x, y, z);

                            for (const auto dir : DistributionHelper::getDistribution27())
                            {
                                if (std::abs(dir[0]) < 0.5 || std::abs(dir[1]) < 0.5 || std::abs(dir[2]) < 0.5) continue;

                                real fineX = x + dir[0] * 0.25 * grid->getDelta();
                                real fineY = y + dir[1] * 0.25 * grid->getDelta();
                                real fineZ = z + dir[2] * 0.25 * grid->getDelta();

                                uint childIndex = fineGrid->transCoordToIndex(fineX, fineY, fineZ);

                                if( childIndex == INVALID_INDEX ) continue;

                                fineGrid->setFieldEntry(childIndex, part[vertexIndex[level][index]]);
                                //fineGrid->setFieldEntry(childIndex, grid->getFieldEntry(index));
                            }
                        }

                        grid->setFieldEntry(index, part[vertexIndex[level][index]]);
                    }
                }

                gridBuilder->writeGridsToVtk("F:/Work/Computations/gridGenerator/grid/Partition_");

            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            return;

            gridGenerator = GridGenerator::make(gridBuilder, para);
        }
    }
    else
    {
        gridGenerator = GridReader::make(FileFormat::BINARY, para);
        //gridGenerator = GridReader::make(FileFormat::ASCII, para);
    }

    logFile.close();

    //return;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    std::ifstream stream;
    stream.open(configPath.c_str(), std::ios::in);
    if (stream.fail())
        throw std::runtime_error("can not open config file!");

    UPtr<input::Input> input = input::Input::makeInput(stream, "config");

    setParameters(para, input);

    Simulation sim;
    SPtr<FileWriter> fileWriter = SPtr<FileWriter>(new FileWriter());
    sim.init(para, gridGenerator, fileWriter);
    sim.run();
}


int main( int argc, char* argv[])
{
     MPI_Init(&argc, &argv);
    std::string str, str2; 
    if ( argv != NULL )
    {
        str = static_cast<std::string>(argv[0]);
        if (argc > 1)
        {
            str2 = static_cast<std::string>(argv[1]);
            try
            {
                multipleLevel(str2);
            }
            catch (const std::exception& e)
            {
                *logging::out << logging::Logger::ERROR << e.what() << "\n";
                //MPI_Abort(MPI_COMM_WORLD, -1);
            }
            catch (...)
            {
                std::cout << "unknown exeption" << std::endl;
            }
        }
        else
        {
            try
            {
                multipleLevel("F:/Work/Computations/gridGenerator/inp/configTest.txt");
            }
            catch (const std::exception& e)
            {
                
                *logging::out << logging::Logger::ERROR << e.what() << "\n";
                //std::cout << e.what() << std::flush;
                //MPI_Abort(MPI_COMM_WORLD, -1);
            }
            catch (const std::bad_alloc e)
            {
                
                *logging::out << logging::Logger::ERROR << "Bad Alloc:" << e.what() << "\n";
                //std::cout << e.what() << std::flush;
                //MPI_Abort(MPI_COMM_WORLD, -1);
            }
            catch (...)
            {
                *logging::out << logging::Logger::ERROR << "Unknown exception!\n";
                //std::cout << "unknown exeption" << std::endl;
            }

            std::cout << "\nConfiguration file must be set!: lbmgm <config file>" << std::endl << std::flush;
            //MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }


   /*
   MPE_Init_log() & MPE_Finish_log() are NOT needed when
   liblmpe.a is linked with this program.  In that case,
   MPI_Init() would have called MPE_Init_log() already.
   */
#if defined( MPI_LOGGING )
   MPE_Init_log();
#endif

#if defined( MPI_LOGGING )
   if ( argv != NULL )
      MPE_Finish_log( argv[0] );
   if ( str != "" )
      MPE_Finish_log( str.c_str() );
   else
      MPE_Finish_log( "TestLog" );
#endif

   MPI_Finalize();
   return 0;
}
