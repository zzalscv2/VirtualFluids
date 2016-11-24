#include <iostream>
#include <string>
#include <VirtualFluids.h>

using namespace std;

void changeDP()
{
}
//////////////////////////////////////////////////////////////////////////
void run(string configname)
{
   try
   {
      ConfigurationFile   config;
      config.load(configname);

      string          pathname = config.getString("pathname");
      string          pathGeo = config.getString("pathGeo");
      int             numOfThreads = config.getInt("numOfThreads");
      string          sampleFilename = config.getString("sampleFilename");
      int             pmNX1 = config.getInt("pmNX1");
      int             pmNX2 = config.getInt("pmNX2");
      int             pmNX3 = config.getInt("pmNX3");
      double          lthreshold = config.getDouble("lthreshold");
      double          uthreshold = config.getDouble("uthreshold");
      double          pmL1 = config.getDouble("pmL1");
      double          pmL2 = config.getDouble("pmL2");
      double          pmL3 = config.getDouble("pmL3");
      int             blocknx = config.getInt("blocknx");
      double          nx3 = config.getDouble("nx3");
      double          dp_LB = config.getDouble("dp_LB");
      double          nu_LB = config.getDouble("nu_LB");
      string          timeSeriesFile = config.getString("timeSeriesFile");
      double          restartStep = config.getDouble("restartStep");
      double          restartStepStart = config.getDouble("restartStepStart");
      double          endTime = config.getDouble("endTime");
      double          outTime = config.getDouble("outTime");
      double          availMem = config.getDouble("availMem");
      bool            rawFile = config.getBool("rawFile");
      double          timeSeriesOutTime = config.getDouble("timeSeriesOutTime");
      bool            logToFile = config.getBool("logToFile");
      bool            spongeLayer = config.getBool("spongeLayer");


      CommunicatorPtr comm = MPICommunicator::getInstance();
      int myid = comm->getProcessID();

      if (logToFile)
      {
#if defined(__unix__)
         if (myid == 0)
         {
            const char* str = pathname.c_str();
            int status = mkdir(str, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
         }
#endif 

         if (myid == 0)
         {
            stringstream logFilename;
            logFilename << pathname + "/logfile" + UbSystem::toString(UbSystem::getTimeStamp()) + ".txt";
            UbLog::output_policy::setStream(logFilename.str());
         }
      }

      //Sleep(30000);

      if (myid == 0) UBLOG(logINFO, "Testcase permeability");

      string machinename = UbSystem::getMachineName();
      UBLOG(logINFO, "PID = " << myid << " Hostname: " << machinename);
      UBLOG(logINFO, "PID = " << myid << " Total Physical Memory (RAM): " << Utilities::getTotalPhysMem());
      UBLOG(logINFO, "PID = " << myid << " Physical Memory currently used: " << Utilities::getPhysMemUsed());
      UBLOG(logINFO, "PID = " << myid << " Physical Memory currently used by current process: " << Utilities::getPhysMemUsedByMe());

      int blocknx1 = blocknx;
      int blocknx2 = blocknx;
      int blocknx3 = blocknx;

      LBMReal rho_LB = 0.0;
      double rhoLBinflow = dp_LB*3.0;

      LBMUnitConverterPtr conv = LBMUnitConverterPtr(new LBMUnitConverter());

      const int baseLevel = 0;

      double coord[6];
      double deltax;

      Grid3DPtr grid(new Grid3D(comm));

      //////////////////////////////////////////////////////////////////////////
      //restart
      UbSchedulerPtr rSch(new UbScheduler(restartStep, restartStepStart));
      RestartCoProcessor rp(grid, rSch, comm, pathname, RestartCoProcessor::BINARY);
      //////////////////////////////////////////////////////////////////////////

      if (grid->getTimeStep() == 0)
      {
         if (myid == 0) UBLOG(logINFO, "new start..");

         UBLOG(logINFO, "new start PID = " << myid << " Hostname: " << machinename);
         UBLOG(logINFO, "new start PID = " << myid << " Total Physical Memory (RAM): " << Utilities::getTotalPhysMem());
         UBLOG(logINFO, "new start PID = " << myid << " Physical Memory currently used: " << Utilities::getPhysMemUsed());
         UBLOG(logINFO, "new start PID = " << myid << " Physical Memory currently used by current process: " << Utilities::getPhysMemUsedByMe());

         string samplePathname = pathGeo + sampleFilename;

         double deltaVoxelX1 = pmL1/(double)pmNX1;
         double deltaVoxelX2 = pmL2/(double)pmNX2;
         double deltaVoxelX3 = pmL3/(double)pmNX3;

         GbVoxelMatrix3DPtr sample(new GbVoxelMatrix3D(pmNX1, pmNX2, pmNX3, 0, lthreshold, uthreshold));
         if (rawFile)
         {
            sample->readMatrixFromRawFile<unsigned short>(samplePathname, GbVoxelMatrix3D::BigEndian);
         }
         else
         {
            sample->readMatrixFromVtiASCIIFile(samplePathname);
         }

         sample->setVoxelMatrixDelta((float)deltaVoxelX1, (float)deltaVoxelX2, (float)deltaVoxelX3);
         sample->setVoxelMatrixMininum(0.0, 0.0, 0.0);

         if (myid == 0) sample->writeToVTKImageDataASCII(pathname + "/geo/sample");

         ///////////////////////////////////////////////////////

         ////////////////////////////////////////////////////////////////////////

         double offset1 = sample->getLengthX1()/10.0;
         double offset2 = 2.0*offset1;
         //bounding box
         double g_minX1 = sample->getX1Minimum() - offset1;
         double g_minX2 = sample->getX2Minimum();
         double g_minX3 = sample->getX3Minimum();

         double g_maxX1 = sample->getX1Maximum() + offset2;
         double g_maxX2 = sample->getX2Maximum();
         double g_maxX3 = sample->getX3Maximum();

         deltax = (g_maxX3-g_minX3) /(nx3*blocknx3);

         double blockLength = (double)blocknx1*deltax;

         grid->setPeriodicX1(false);
         grid->setPeriodicX2(false);
         grid->setPeriodicX3(false);
         grid->setDeltaX(deltax);
         grid->setBlockNX(blocknx1, blocknx2, blocknx3);

         GbObject3DPtr gridCube(new GbCuboid3D(g_minX1, g_minX2, g_minX3, g_maxX1, g_maxX2, g_maxX3));
         if (myid == 0) GbSystem3D::writeGeoObject(gridCube.get(), pathname + "/geo/gridCube", WbWriterVtkXmlBinary::getInstance());

         GenBlocksGridVisitor genBlocks(gridCube);
         grid->accept(genBlocks);

         if (myid == 0)
         {
            UBLOG(logINFO, "Parameters:");
            UBLOG(logINFO, "rho_LB = " << rho_LB);
            UBLOG(logINFO, "nu_LB = " << nu_LB);
            UBLOG(logINFO, "dp_LB = " << dp_LB);
            UBLOG(logINFO, "dx = " << deltax << " m");
            UBLOG(logINFO, "numOfThreads = " << numOfThreads);
            UBLOG(logINFO, "path = " << pathname);
            UBLOG(logINFO, "Preprozess - start");
         }

         //walls
         GbCuboid3DPtr addWallYmin(new GbCuboid3D(g_minX1-blockLength, g_minX2-blockLength, g_minX3-blockLength, g_maxX1+blockLength, g_minX2, g_maxX3+blockLength));
         if (myid == 0) GbSystem3D::writeGeoObject(addWallYmin.get(), pathname+"/geo/addWallYmin", WbWriterVtkXmlASCII::getInstance());

         GbCuboid3DPtr addWallZmin(new GbCuboid3D(g_minX1-blockLength, g_minX2-blockLength, g_minX3-blockLength, g_maxX1+blockLength, g_maxX2+blockLength, g_minX3));
         if (myid == 0) GbSystem3D::writeGeoObject(addWallZmin.get(), pathname+"/geo/addWallZmin", WbWriterVtkXmlASCII::getInstance());

         GbCuboid3DPtr addWallYmax(new GbCuboid3D(g_minX1-blockLength, g_maxX2, g_minX3-blockLength, g_maxX1+blockLength, g_maxX2+blockLength, g_maxX3+blockLength));
         if (myid == 0) GbSystem3D::writeGeoObject(addWallYmax.get(), pathname+"/geo/addWallYmax", WbWriterVtkXmlASCII::getInstance());

         GbCuboid3DPtr addWallZmax(new GbCuboid3D(g_minX1-blockLength, g_minX2-blockLength, g_maxX3, g_maxX1+blockLength, g_maxX2+blockLength, g_maxX3+blockLength));
         if (myid == 0) GbSystem3D::writeGeoObject(addWallZmax.get(), pathname+"/geo/addWallZmax", WbWriterVtkXmlASCII::getInstance());


         //inflow
         GbCuboid3DPtr geoInflow(new GbCuboid3D(g_minX1-blockLength, g_minX2-blockLength, g_minX3-blockLength, g_minX1, g_maxX2+blockLength, g_maxX3+blockLength));
         if (myid == 0) GbSystem3D::writeGeoObject(geoInflow.get(), pathname + "/geo/geoInflow", WbWriterVtkXmlASCII::getInstance());

         //outflow
         GbCuboid3DPtr geoOutflow(new GbCuboid3D(g_maxX1, g_minX2-blockLength, g_minX3-blockLength, g_maxX1+blockLength, g_maxX2+blockLength, g_maxX3+blockLength));
         if (myid == 0) GbSystem3D::writeGeoObject(geoOutflow.get(), pathname + "/geo/geoOutflow", WbWriterVtkXmlASCII::getInstance());

         WriteBlocksCoProcessorPtr ppblocks(new WriteBlocksCoProcessor(grid, UbSchedulerPtr(new UbScheduler(1)), pathname, WbWriterVtkXmlBinary::getInstance(), comm));

         //bone interactor
         int bcOptionNoSlip = 1; //0=simple Bounce Back, 1=quadr. BB, 2=thin wall
         D3Q27BoundaryConditionAdapterPtr bcNoSlip(new D3Q27NoSlipBCAdapter(bcOptionNoSlip));
         D3Q27InteractorPtr sampleInt(new D3Q27Interactor(sample, grid, bcNoSlip, Interactor3D::SOLID));

         //wall interactors
         D3Q27InteractorPtr addWallYminInt(new D3Q27Interactor(addWallYmin, grid, bcNoSlip, Interactor3D::SOLID));
         D3Q27InteractorPtr addWallZminInt(new D3Q27Interactor(addWallZmin, grid, bcNoSlip, Interactor3D::SOLID));
         D3Q27InteractorPtr addWallYmaxInt(new D3Q27Interactor(addWallYmax, grid, bcNoSlip, Interactor3D::SOLID));
         D3Q27InteractorPtr addWallZmaxInt(new D3Q27Interactor(addWallZmax, grid, bcNoSlip, Interactor3D::SOLID));

         D3Q27BoundaryConditionAdapterPtr denBCAdapterInflow(new D3Q27DensityBCAdapter(rhoLBinflow));
         denBCAdapterInflow->setSecondaryBcOption(0);
         D3Q27InteractorPtr inflowInt = D3Q27InteractorPtr(new D3Q27Interactor(geoInflow, grid, denBCAdapterInflow, Interactor3D::SOLID));

         //outflow
         D3Q27BoundaryConditionAdapterPtr denBCAdapterOutflow(new D3Q27DensityBCAdapter(rho_LB));
         denBCAdapterOutflow->setSecondaryBcOption(0);
         D3Q27InteractorPtr outflowInt = D3Q27InteractorPtr(new D3Q27Interactor(geoOutflow, grid, denBCAdapterOutflow, Interactor3D::SOLID));

         
         UBLOG(logINFO, "PID = " << myid << " Hostname: " << machinename);
         UBLOG(logINFO, "PID = " << myid << " Total Physical Memory (RAM): " << Utilities::getTotalPhysMem());
         UBLOG(logINFO, "PID = " << myid << " Physical Memory currently used: " << Utilities::getPhysMemUsed());
         UBLOG(logINFO, "PID = " << myid << " Physical Memory currently used by current process: " << Utilities::getPhysMemUsedByMe());

         ////////////////////////////////////////////
         //METIS
         Grid3DVisitorPtr metisVisitor(new MetisPartitioningGridVisitor(comm, MetisPartitioningGridVisitor::LevelBased, D3Q27System::BSW, MetisPartitioner::RECURSIVE));
         ////////////////////////////////////////////
         //Zoltan
         //Grid3DVisitorPtr zoltanVisitor(new ZoltanPartitioningGridVisitor(comm, D3Q27System::BSW, 1));
         //grid->accept(zoltanVisitor);
         /////delete solid blocks
         if (myid == 0) UBLOG(logINFO, "deleteSolidBlocks - start");
         InteractorsHelper intHelper(grid, metisVisitor);
         intHelper.addInteractor(addWallYminInt);
         intHelper.addInteractor(addWallZminInt);
         intHelper.addInteractor(addWallYmaxInt);
         intHelper.addInteractor(addWallZmaxInt);
         intHelper.addInteractor(inflowInt);
         intHelper.addInteractor(outflowInt);
         intHelper.addInteractor(sampleInt);
         intHelper.selectBlocks();
         if (myid == 0) UBLOG(logINFO, "deleteSolidBlocks - end");
         //////////////////////////////////////

         //set connectors
         D3Q27InterpolationProcessorPtr iProcessor(new D3Q27IncompressibleOffsetInterpolationProcessor());
         D3Q27SetConnectorsBlockVisitor setConnsVisitor(comm, true, D3Q27System::ENDDIR, nu_LB, iProcessor);
         grid->accept(setConnsVisitor);

         //domain decomposition for threads
         PQueuePartitioningGridVisitor pqPartVisitor(numOfThreads);
         grid->accept(pqPartVisitor);

         ppblocks->process(0);
         ppblocks.reset();

         unsigned long nob = grid->getNumberOfBlocks();
         int gl = 3;
         unsigned long nodb = (blocknx1)* (blocknx2)* (blocknx3);
         unsigned long nod = nob * (blocknx1)* (blocknx2)* (blocknx3);
         unsigned long nodg = nob * (blocknx1 + gl) * (blocknx2 + gl) * (blocknx3 + gl);
         double needMemAll = double(nodg*(27 * sizeof(double) + sizeof(int) + sizeof(float) * 4));
         double needMem = needMemAll / double(comm->getNumberOfProcesses());

         if (myid == 0)
         {
            UBLOG(logINFO, "Number of blocks = " << nob);
            UBLOG(logINFO, "Number of nodes  = " << nod);
            int minInitLevel = grid->getCoarsestInitializedLevel();
            int maxInitLevel = grid->getFinestInitializedLevel();
            for (int level = minInitLevel; level <= maxInitLevel; level++)
            {
               int nobl = grid->getNumberOfBlocks(level);
               UBLOG(logINFO, "Number of blocks for level " << level << " = " << nobl);
               UBLOG(logINFO, "Number of nodes for level " << level << " = " << nobl*nodb);
            }
            UBLOG(logINFO, "Necessary memory  = " << needMemAll << " bytes");
            UBLOG(logINFO, "Necessary memory per process = " << needMem << " bytes");
            UBLOG(logINFO, "Available memory per process = " << availMem << " bytes");
         }

         LBMKernel3DPtr kernel;
         kernel = LBMKernel3DPtr(new LBMKernelETD3Q27CCLB(blocknx1, blocknx2, blocknx3, LBMKernelETD3Q27CCLB::NORMAL));

         BCProcessorPtr bcProc(new D3Q27ETBCProcessor());
         
         BoundaryConditionPtr densityBC(new NonEqDensityBoundaryCondition());
         BoundaryConditionPtr noSlipBC(new NoSlipBoundaryCondition());

         bcProc->addBC(densityBC);
         bcProc->addBC(noSlipBC);

         kernel->setBCProcessor(bcProc);

         SetKernelBlockVisitor kernelVisitor(kernel, nu_LB, availMem, needMem);
         grid->accept(kernelVisitor);

         //BC
         intHelper.setBC();

         BoundaryConditionBlockVisitor bcVisitor;
         grid->accept(bcVisitor);

         //Press*1.6e8+(14.76-coordsX)/3.5*5000
         //initialization of distributions
         mu::Parser fct;
         fct.SetExpr("(x1max-x1)/l*dp*3.0");
         fct.DefineConst("dp", dp_LB);
         fct.DefineConst("x1max", g_maxX1);
         fct.DefineConst("l", g_maxX1-g_minX1);

         D3Q27ETInitDistributionsBlockVisitor initVisitor(nu_LB, rho_LB);
         initVisitor.setRho(fct);
         grid->accept(initVisitor);

         //Postrozess
         UbSchedulerPtr geoSch(new UbScheduler(1));
         MacroscopicQuantitiesCoProcessorPtr ppgeo(
            new MacroscopicQuantitiesCoProcessor(grid, geoSch, pathname, WbWriterVtkXmlBinary::getInstance(), conv, true));
         ppgeo->process(0);
         ppgeo.reset();

         coord[0] = sample->getX1Minimum();
         coord[1] = sample->getX2Minimum();
         coord[2] = sample->getX3Minimum();
         coord[3] = sample->getX1Maximum();
         coord[4] = sample->getX2Maximum();
         coord[5] = sample->getX3Maximum();

         ////////////////////////////////////////////////////////
         FILE * pFile;
         string str = pathname + "/checkpoints/coord.txt";
         pFile = fopen(str.c_str(), "w");
         fprintf(pFile, "%g\n", deltax);
         fprintf(pFile, "%g\n", coord[0]);
         fprintf(pFile, "%g\n", coord[1]);
         fprintf(pFile, "%g\n", coord[2]);
         fprintf(pFile, "%g\n", coord[3]);
         fprintf(pFile, "%g\n", coord[4]);
         fprintf(pFile, "%g\n", coord[5]);
         fclose(pFile);
         ////////////////////////////////////////////////////////

         grid->addInteractor(inflowInt);

         if (myid == 0) UBLOG(logINFO, "Preprozess - end");
      }
      else
      {
         ////////////////////////////////////////////////////////
         FILE * pFile;
         string str = pathname + "/checkpoints/coord.txt";
         pFile = fopen(str.c_str(), "r");
         fscanf(pFile, "%lg\n", &deltax);
         fscanf(pFile, "%lg\n", &coord[0]);
         fscanf(pFile, "%lg\n", &coord[1]);
         fscanf(pFile, "%lg\n", &coord[2]);
         fscanf(pFile, "%lg\n", &coord[3]);
         fscanf(pFile, "%lg\n", &coord[4]);
         fscanf(pFile, "%lg\n", &coord[5]);
         fclose(pFile);
         ////////////////////////////////////////////////////////

         //new nu
         //ViscosityBlockVisitor nuVisitor(nu_LB);
         //grid->accept(nuVisitor);

         //new dp
         Grid3D::Interactor3DSet interactors = grid->getInteractors();
         interactors[0]->setGrid3D(grid);
         boost::dynamic_pointer_cast<D3Q27Interactor>(interactors[0])->deleteBCAdapter();
         D3Q27BoundaryConditionAdapterPtr denBCAdapterFront(new D3Q27DensityBCAdapter(rhoLBinflow));
         boost::dynamic_pointer_cast<D3Q27Interactor>(interactors[0])->addBCAdapter(denBCAdapterFront);
         interactors[0]->updateInteractor();

         if (myid == 0)
         {
	         UBLOG(logINFO, "Parameters:");
	         UBLOG(logINFO, "rho_LB = " << rho_LB);
	         UBLOG(logINFO, "nu_LB = " << nu_LB);
	         UBLOG(logINFO, "dp_LB = " << dp_LB);
	         UBLOG(logINFO, "dx = " << deltax << " m");
         }

         BoundaryConditionBlockVisitor bcVisitor;
         grid->accept(bcVisitor);

         //set connectors
         D3Q27InterpolationProcessorPtr iProcessor(new D3Q27IncompressibleOffsetInterpolationProcessor());
         D3Q27SetConnectorsBlockVisitor setConnsVisitor(comm, true, D3Q27System::ENDDIR, nu_LB, iProcessor);
         grid->accept(setConnsVisitor);

         if (myid == 0) UBLOG(logINFO, "Restart - end");
      }
      UbSchedulerPtr nupsSch(new UbScheduler(10, 30, 100));
      //nupsSch->addSchedule(nupsStep[0], nupsStep[1], nupsStep[2]);
      NUPSCounterCoProcessor npr(grid, nupsSch, numOfThreads, comm);

      UbSchedulerPtr stepSch(new UbScheduler(outTime));

      MacroscopicQuantitiesCoProcessor pp(grid, stepSch, pathname, WbWriterVtkXmlBinary::getInstance(), conv);

      deltax = grid->getDeltaX(baseLevel);
      double dxd2 = deltax / 2.0;

      D3Q27IntegrateValuesHelperPtr ih1(new D3Q27IntegrateValuesHelper(grid, comm, coord[0] - dxd2*10.0, coord[1] - dxd2, coord[2] - dxd2,
         coord[0] - dxd2*10.0 - 2.0*dxd2, coord[4] + dxd2, coord[5] + dxd2));

      //D3Q27IntegrateValuesHelperPtr ih2(new D3Q27IntegrateValuesHelper(grid, comm, coord[3]/2.0, coord[1] - dxd2, coord[2] - dxd2,
      //   coord[3]/2.0 + 2.0*dxd2, coord[4] + dxd2, coord[5] + dxd2));
      D3Q27IntegrateValuesHelperPtr ih2(new D3Q27IntegrateValuesHelper(grid, comm, coord[0], coord[1], coord[2], coord[3], coord[4], coord[5]));

      D3Q27IntegrateValuesHelperPtr ih3(new D3Q27IntegrateValuesHelper(grid, comm, coord[3] + dxd2*10.0, coord[1] - dxd2, coord[2] - dxd2,
         coord[3] + dxd2*10.0 + 2.0*dxd2, coord[4] + dxd2, coord[5] + dxd2));

      //D3Q27IntegrateValuesHelperPtr ih1(new D3Q27IntegrateValuesHelper(grid, comm, coord[0], coord[1], coord[2], coord[3], coord[4], coord[5]));
      if (myid == 0) GbSystem3D::writeGeoObject(ih1->getBoundingBox().get(), pathname + "/geo/ih1", WbWriterVtkXmlBinary::getInstance());
      if (myid == 0) GbSystem3D::writeGeoObject(ih2->getBoundingBox().get(), pathname + "/geo/ih2", WbWriterVtkXmlBinary::getInstance());
      if (myid == 0) GbSystem3D::writeGeoObject(ih3->getBoundingBox().get(), pathname + "/geo/ih3", WbWriterVtkXmlBinary::getInstance());

      double factorp = 1; // dp_real / dp_LB;
      double factorv = 1;// dx / dt;
      UbSchedulerPtr stepMV(new UbScheduler(timeSeriesOutTime));
      
      TimeseriesCoProcessor tsp1(grid, stepMV, ih1, pathname+timeSeriesFile+"_1", comm);
      TimeseriesCoProcessor tsp2(grid, stepMV, ih2, pathname+timeSeriesFile+"_2", comm);
      TimeseriesCoProcessor tsp3(grid, stepMV, ih3, pathname+timeSeriesFile+"_3", comm);

      if (myid == 0)
      {
         UBLOG(logINFO, "PID = " << myid << " Total Physical Memory (RAM): " << Utilities::getTotalPhysMem());
         UBLOG(logINFO, "PID = " << myid << " Physical Memory currently used: " << Utilities::getPhysMemUsed());
         UBLOG(logINFO, "PID = " << myid << " Physical Memory currently used by current process: " << Utilities::getPhysMemUsedByMe());
      }

      CalculationManagerPtr calculation(new CalculationManager(grid, numOfThreads, endTime, stepMV));
      if (myid == 0) UBLOG(logINFO, "Simulation-start");
      calculation->calculate();
      if (myid == 0) UBLOG(logINFO, "Simulation-end");
   }
   catch (exception& e)
   {
      cerr << e.what() << endl << flush;
   }
   catch (string& s)
   {
      cerr << s << endl;
   }
   catch (...)
   {
      cerr << "unknown exception" << endl;
   }

}
//////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{

   if (argv!=NULL)
   {
      if (argv[1]!=NULL)
      {
         run(string(argv[1]));
      }
      else
      {
         cout<<"Configuration file is missing!"<<endl;
      }
   }

   return 0;
}
