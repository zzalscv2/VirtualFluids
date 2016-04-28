#include <iostream>
#include <string>

#include "VirtualFluids.h"

using namespace std;


void run(string configname)
{
   try
   {
      ConfigurationFile   config;
      config.load(configname);

      string          pathname = config.getString("pathname");
      int             numOfThreads = config.getInt("numOfThreads");
      vector<int>     blocknx = config.getVector<int>("blocknx");
      double          uLB = config.getDouble("uLB");
      double          endTime = config.getDouble("endTime");
      double          outTime = config.getDouble("outTime");
      double          availMem = config.getDouble("availMem");
      int             refineLevel = config.getInt("refineLevel");
      double          Re = config.getDouble("Re");
      double          dx = config.getDouble("dx");
      vector<double>  length = config.getVector<double>("length");
      bool            logToFile = config.getBool("logToFile");

      UbLog::reportingLevel() = UbLog::logLevelFromString("DEBUG3");

      CommunicatorPtr comm = MPICommunicator::getInstance();
      int myid = comm->getProcessID();

      if (logToFile)
      {
#if defined(__unix__)
         if (myid == 0)
         {
            const char* str = pathname.c_str();
            mkdir(str, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
         }
#endif 

         if (myid == 0)
         {
            stringstream logFilename;
            logFilename << pathname + "/logfile" + UbSystem::toString(UbSystem::getTimeStamp()) + ".txt";
            UbLog::output_policy::setStream(logFilename.str());
         }
      }

      LBMReal dLB = length[1] / dx;
      LBMReal rhoLB = 0.0;
      LBMReal nuLB = (uLB*dLB) / Re;

      LBMUnitConverterPtr conv = LBMUnitConverterPtr(new LBMUnitConverter());

      const int baseLevel = 0;

      //bounding box
      double g_minX1 = 0.0;
      double g_minX2 = 0.0;
      double g_minX3 = 0.0;

      double g_maxX1 = length[0];
      double g_maxX2 = length[1];
      double g_maxX3 = length[2];

      //geometry
      GbObject3DPtr box(new GbCuboid3D(g_minX1, g_minX2, g_minX3, g_maxX1, g_maxX2, g_maxX3));
      //if (myid == 0) GbSystem3D::writeGeoObject(box.get(), pathname + "/geo/box", WbWriterVtkXmlBinary::getInstance());

      GbObject3DPtr gridCube(new GbCuboid3D(g_minX1, g_minX2, g_minX3, g_maxX1, g_maxX2, g_maxX3));
      //if (myid == 0) GbSystem3D::writeGeoObject(gridCube.get(), pathname + "/geo/gridCube", WbWriterVtkXmlBinary::getInstance());


      double blockLength = blocknx[0] * dx;

      Grid3DPtr grid(new Grid3D(comm));

      if (myid == 0)
      {
         UBLOG(logINFO, "uLb = " << uLB);
         UBLOG(logINFO, "rho = " << rhoLB);
         UBLOG(logINFO, "nuLb = " << nuLB);
         UBLOG(logINFO, "Re = " << Re);
         UBLOG(logINFO, "dx = " << dx);
         UBLOG(logINFO, "length = " << length[0] << " " << length[1] << " " << length[2]);
         UBLOG(logINFO, "blocknx = " << blocknx[0] << " " << blocknx[1] << " " << blocknx[2]);
         UBLOG(logINFO, "number of levels = " << refineLevel + 1);
         UBLOG(logINFO, "number of processes = " << comm->getNumberOfProcesses());
         UBLOG(logINFO, "number of threads = " << numOfThreads);
         UBLOG(logINFO, "Preprocess - start");
      }

      grid->setDeltaX(dx);
      grid->setBlockNX(blocknx[0], blocknx[1], blocknx[2]);

      //if (myid == 0) GbSystem3D::writeGeoObject(gridCube.get(), pathname + "/geo/gridCube", WbWriterVtkXmlBinary::getInstance());

      GenBlocksGridVisitor genBlocks(gridCube);
      grid->accept(genBlocks);

      WriteBlocksCoProcessorPtr ppblocks(new WriteBlocksCoProcessor(grid, UbSchedulerPtr(new UbScheduler(1)), pathname, WbWriterVtkXmlBinary::getInstance(), comm));

      //int bbOption = 1; //0=simple Bounce Back, 1=quadr. BB
      //D3Q27BoundaryConditionAdapterPtr bcObst(new D3Q27NoSlipBCAdapter(bbOption));
      //D3Q27InteractorPtr boxInt(new D3Q27Interactor(box, grid, bcObst, Interactor3D::INVERSESOLID));

      Grid3DVisitorPtr metisVisitor(new MetisPartitioningGridVisitor(comm, MetisPartitioningGridVisitor::LevelBased, D3Q27System::B));
      InteractorsHelper intHelper(grid, metisVisitor);
      //intHelper.addInteractor(boxInt);
      intHelper.selectBlocks();

      ppblocks->process(0);
      ppblocks.reset();

      //set connectors
      D3Q27InterpolationProcessorPtr iProcessor(new D3Q27IncompressibleOffsetInterpolationProcessor());
      D3Q27SetConnectorsBlockVisitor setConnsVisitor(comm, true, D3Q27System::ENDDIR, nuLB, iProcessor);
      //ConnectorFactoryPtr factory(new Block3DConnectorFactory());
      //ConnectorBlockVisitor setConnsVisitor(comm, nuLB, iProcessor, factory);
      UBLOG(logINFO, "D3Q27SetConnectorsBlockVisitor:start");
      grid->accept(setConnsVisitor);
      UBLOG(logINFO, "D3Q27SetConnectorsBlockVisitor:end");

      //domain decomposition for threads
      PQueuePartitioningGridVisitor pqPartVisitor(numOfThreads);
      grid->accept(pqPartVisitor);


      unsigned long long numberOfBlocks = (unsigned long long)grid->getNumberOfBlocks();
      int ghostLayer = 3;
      unsigned long long numberOfNodesPerBlock = (unsigned long long)(blocknx[0])* (unsigned long long)(blocknx[1])* (unsigned long long)(blocknx[2]);
      unsigned long long numberOfNodes = numberOfBlocks * numberOfNodesPerBlock;
      unsigned long long numberOfNodesPerBlockWithGhostLayer = numberOfBlocks * (blocknx[0] + ghostLayer) * (blocknx[1] + ghostLayer) * (blocknx[2] + ghostLayer);
      double needMemAll = double(numberOfNodesPerBlockWithGhostLayer*(27 * sizeof(double) + sizeof(int) + sizeof(float) * 4));
      double needMem = needMemAll / double(comm->getNumberOfProcesses());

      if (myid == 0)
      {
         UBLOG(logINFO, "Number of blocks = " << numberOfBlocks);
         UBLOG(logINFO, "Number of nodes  = " << numberOfNodes);
         int minInitLevel = grid->getCoarsestInitializedLevel();
         int maxInitLevel = grid->getFinestInitializedLevel();
         for (int level = minInitLevel; level <= maxInitLevel; level++)
         {
            int nobl = grid->getNumberOfBlocks(level);
            UBLOG(logINFO, "Number of blocks for level " << level << " = " << nobl);
            UBLOG(logINFO, "Number of nodes for level " << level << " = " << nobl*numberOfNodesPerBlock);
         }
         UBLOG(logINFO, "Necessary memory  = " << needMemAll << " bytes");
         UBLOG(logINFO, "Necessary memory per process = " << needMem << " bytes");
         UBLOG(logINFO, "Available memory per process = " << availMem << " bytes");
      }

      LBMKernel3DPtr kernel;

      kernel = LBMKernel3DPtr(new LBMKernelETD3Q27CCLB(blocknx[0], blocknx[1], blocknx[2], LBMKernelETD3Q27CCLB::NORMAL));

      //
      BCProcessorPtr bcProc(new D3Q27ETBCProcessor());
      //BCProcessorPtr bcProc(new D3Q27ETForThinWallBCProcessor());
      BoundaryConditionPtr noSlipBC;
      //noSlipBC = BoundaryConditionPtr(new NoSlipBoundaryCondition());
      //noSlipBC = BoundaryConditionPtr(new ThinWallNoSlipBoundaryCondition());
      //bcProc->addBC(noSlipBC);

      kernel->setBCProcessor(bcProc);

      SetKernelBlockVisitor kernelVisitor(kernel, nuLB, availMem, needMem);
      grid->accept(kernelVisitor);

      if (refineLevel > 0)
      {
         D3Q27SetUndefinedNodesBlockVisitor undefNodesVisitor;
         grid->accept(undefNodesVisitor);
      }

      intHelper.setBC();

      //BoundaryConditionBlockVisitor bcVisitor;
      //grid->accept(bcVisitor);

      //initialization of distributions
      D3Q27ETInitDistributionsBlockVisitor initVisitor(nuLB, rhoLB, uLB);
      grid->accept(initVisitor);

      //boundary conditions grid
      //{
      //   UbSchedulerPtr geoSch(new UbScheduler(1));
      //   MacroscopicQuantitiesCoProcessorPtr ppgeo(
      //      new MacroscopicQuantitiesCoProcessor(grid, geoSch, pathname, WbWriterVtkXmlBinary::getInstance(), conv, true));
      //   grid->coProcess(0);
      //}

      if (myid == 0) UBLOG(logINFO, "Preprocess - end");

      if (myid == 0)
      {
         UBLOG(logINFO, "PID = " << myid << " Total Physical Memory (RAM): " << Utilities::getTotalPhysMem());
         UBLOG(logINFO, "PID = " << myid << " Physical Memory currently used: " << Utilities::getPhysMemUsed());
         UBLOG(logINFO, "PID = " << myid << " Physical Memory currently used by current process: " << Utilities::getPhysMemUsedByMe());
      }

      UbSchedulerPtr visSch(new UbScheduler(outTime));
      //MacroscopicQuantitiesCoProcessor pp(grid, visSch, pathname, WbWriterVtkXmlASCII::getInstance(), conv);

      UbSchedulerPtr nupsSch(new UbScheduler(10, 30, 100));
      NUPSCounterCoProcessor npr(grid, nupsSch, numOfThreads, comm);

      CalculationManagerPtr calculation(new CalculationManager(grid, numOfThreads, endTime, visSch));
      if (myid == 0) UBLOG(logINFO, "Simulation-start");
      calculation->calculate();
      if (myid == 0) UBLOG(logINFO, "Simulation-end");
   }
   catch (std::exception& e)
   {
      cerr << e.what() << endl << flush;
   }
   catch (std::string& s)
   {
      cerr << s << endl;
   }
   catch (...)
   {
      cerr << "unknown exception" << endl;
   }

}
int main(int argc, char* argv[])
{
   if (argv != NULL)
   {
      if (argv[1] != NULL)
      {
         run(string(argv[1]));
      }
      else
      {
         cout << "Configuration file is missing!" << endl;
      }
   }

}

