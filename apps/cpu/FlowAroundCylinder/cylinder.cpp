#include <iostream>
#include <string>

#include "VirtualFluids.h"

using namespace std;


//////////////////////////////////////////////////////////////////////////
void run(string configname)
{
   try
   {
      //DEBUG///////////////////////////////////////
      //Sleep(30000);
      /////////////////////////////////////////////
      ConfigurationFile   config;
      config.load(configname);

      string          pathOut = config.getValue<string>("pathOut");
      double          uLB = config.getValue<double>("uLB");
      double          restartStep = config.getValue<double>("restartStep");
      double          cpStart = config.getValue<double>("cpStart");
      double          cpStep = config.getValue<double>("cpStep");
      double          endTime = config.getValue<double>("endTime");
      double          outTime = config.getValue<double>("outTime");
      double          availMem = config.getValue<double>("availMem");
      int             refineLevel = config.getValue<int>("refineLevel");
      bool            logToFile = config.getValue<bool>("logToFile");
      vector<double>  nupsStep = config.getVector<double>("nupsStep");
      bool            newStart = config.getValue<bool>("newStart");
      int             numOfThreads = config.getValue<int>("numOfThreads");
      vector<int>     blockNx = config.getVector<int>("blockNx");
      double          dx = config.getValue<double>("dx");

      SPtr<Communicator> comm = MPICommunicator::getInstance();
      int myid = comm->getProcessID();

      if (logToFile)
      {
#if defined(__unix__)
         if (myid==0)
         {
            const char* str = pathOut.c_str();
            mkdir(str, S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH);
         }
#endif 

         if (myid==0)
         {
            stringstream logFilename;
            logFilename<<pathOut+"/logfile"+UbSystem::toString(UbSystem::getTimeStamp())+".txt";
            UbLog::output_policy::setStream(logFilename.str());
         }
      }

      if (myid == 0) UBLOG(logINFO, "Test case: flow around cylinder");

      

      double L1 = 2.5;
      double L2, L3, H;
      L2 = L3 = H = 0.41;

      LBMReal Re = 20.0;
      LBMReal radius = 0.05;
      LBMReal rhoReal = 1.0; //kg/m^3
      LBMReal uReal = 0.45;//m/s
      LBMReal nueReal = (uReal*radius*2.0)/Re;
      
      LBMReal rhoLB = 0.0;
      LBMReal nueLB = (((4.0/9.0)*uLB)*2.0*(radius/dx))/Re;

      SPtr<LBMUnitConverter> conv = SPtr<LBMUnitConverter>(new LBMUnitConverter());

      const int baseLevel = 0;

      SPtr<Grid3D> grid(new Grid3D(comm));

      //BC
      SPtr<BCAdapter> noSlipAdapter(new NoSlipBCAdapter());
      noSlipAdapter->setBcAlgorithm(SPtr<BCAlgorithm>(new NoSlipBCAlgorithm()));

      mu::Parser fct;
      fct.SetExpr("16*U*x2*x3*(H-x2)*(H-x3)/H^4");
      fct.DefineConst("U", uLB);
      fct.DefineConst("H", H);
      SPtr<BCAdapter> velBCAdapter(new VelocityBCAdapter(true, false, false, fct, 0, BCFunction::INFCONST));
      velBCAdapter->setBcAlgorithm(SPtr<BCAlgorithm>(new VelocityWithDensityBCAlgorithm()));

      SPtr<BCAdapter> denBCAdapter(new DensityBCAdapter(rhoLB));
      denBCAdapter->setBcAlgorithm(SPtr<BCAlgorithm>(new NonReflectingOutflowBCAlgorithm()));
      
      BoundaryConditionsBlockVisitor bcVisitor;
      bcVisitor.addBC(noSlipAdapter);
      bcVisitor.addBC(velBCAdapter);
      bcVisitor.addBC(denBCAdapter);

      //////////////////////////////////////////////////////////////////////////
      //restart
      SPtr<UbScheduler> rSch(new UbScheduler(cpStep, cpStart));
      //RestartCoProcessor rp(grid, rSch, comm, pathOut, RestartCoProcessor::BINARY);
      MPIIORestartCoProcessor rcp(grid, rSch, pathOut, comm);
      //////////////////////////////////////////////////////////////////////////

      ////cylinder
      SPtr<GbObject3D> cylinder(new GbCylinder3D(0.5, 0.2, -0.1, 0.5, 0.2, L3+0.1, radius));
      GbSystem3D::writeGeoObject(cylinder.get(), pathOut+"/geo/cylinder", WbWriterVtkXmlBinary::getInstance());
      
      SPtr<D3Q27Interactor> cylinderInt = SPtr<D3Q27Interactor>(new D3Q27Interactor(cylinder, grid, noSlipAdapter, Interactor3D::SOLID));

      if (newStart)
      {
         if (myid==0)
         {
            UBLOG(logINFO, "Number of processes = "<<comm->getNumberOfProcesses());
            UBLOG(logINFO, "Number of threads = "<<numOfThreads);
            UBLOG(logINFO, "path = "<<pathOut);
            UBLOG(logINFO, "L = "<<L1/dx);
            UBLOG(logINFO, "H = "<<H/dx);
            UBLOG(logINFO, "uReal = "<<uReal<<" m/s");
            UBLOG(logINFO, "rhoReal = "<<rhoReal<<" kg/m^3");
            UBLOG(logINFO, "nueReal = "<<nueReal<<" m^2/s");
            UBLOG(logINFO, "uLB = "<<uLB);
            UBLOG(logINFO, "rhoLB = "<<rhoLB);
            UBLOG(logINFO, "nueLB = "<<nueLB);
            UBLOG(logINFO, "Re = "<<Re);
            UBLOG(logINFO, "dx coarse= "<<dx);
            UBLOG(logINFO, "dx fine = "<<dx/(1<<refineLevel) );
            UBLOG(logINFO, "Number of level = "<<refineLevel+1);
            UBLOG(logINFO, "Preprozess - start");
         }

         SPtr<GbObject3D> refCylinder(new GbCylinder3D(0.5, 0.2, -0.1, 0.5, 0.2, L3+0.1, radius+7.0*dx/(1<<refineLevel)));
         GbSystem3D::writeGeoObject(refCylinder.get(), pathOut+"/geo/refCylinder", WbWriterVtkXmlBinary::getInstance());

         //bounding box
         double g_minX1 = 0.0;
         double g_minX2 = 0.0;
         double g_minX3 = 0.0;

         double g_maxX1 = L1;
         double g_maxX2 = L2;
         double g_maxX3 = L3;

         SPtr<GbObject3D> gridCube(new GbCuboid3D(g_minX1, g_minX2, g_minX3, g_maxX1, g_maxX2, g_maxX3));
         if (myid==0) GbSystem3D::writeGeoObject(gridCube.get(), pathOut+"/geo/gridCube", WbWriterVtkXmlBinary::getInstance());

         const int blocknx1 = blockNx[0];
         const int blocknx2 = blockNx[1];
         const int blocknx3 = blockNx[2];

         double blockLength = blocknx1*dx;

         grid->setDeltaX(dx);
         grid->setBlockNX(blocknx1, blocknx2, blocknx3);

         GenBlocksGridVisitor genBlocks(gridCube);
         grid->accept(genBlocks);

         //walls
         GbCuboid3DPtr addWallYmin(new GbCuboid3D(g_minX1-blockLength, g_minX2-blockLength, g_minX3-blockLength, g_maxX1+blockLength, g_minX2, g_maxX3+blockLength));
         if (myid==0) GbSystem3D::writeGeoObject(addWallYmin.get(), pathOut+"/geo/addWallYmin", WbWriterVtkXmlASCII::getInstance());

         GbCuboid3DPtr addWallYmax(new GbCuboid3D(g_minX1-blockLength, g_maxX2, g_minX3-blockLength, g_maxX1+blockLength, g_maxX2+blockLength, g_maxX3+blockLength));
         if (myid==0) GbSystem3D::writeGeoObject(addWallYmax.get(), pathOut+"/geo/addWallYmax", WbWriterVtkXmlASCII::getInstance());

         GbCuboid3DPtr addWallZmin(new GbCuboid3D(g_minX1-blockLength, g_minX2-blockLength, g_minX3-blockLength, g_maxX1+blockLength, g_maxX2+blockLength, g_minX3));
         if (myid==0) GbSystem3D::writeGeoObject(addWallZmin.get(), pathOut+"/geo/addWallZmin", WbWriterVtkXmlASCII::getInstance());

         GbCuboid3DPtr addWallZmax(new GbCuboid3D(g_minX1-blockLength, g_minX2-blockLength, g_maxX3, g_maxX1+blockLength, g_maxX2+blockLength, g_maxX3+blockLength));
         if (myid==0) GbSystem3D::writeGeoObject(addWallZmax.get(), pathOut+"/geo/addWallZmax", WbWriterVtkXmlASCII::getInstance());

         //inflow
         GbCuboid3DPtr geoInflow(new GbCuboid3D(g_minX1-blockLength, g_minX2-blockLength, g_minX3-blockLength, g_minX1, g_maxX2+blockLength, g_maxX3+blockLength));
         if (myid==0) GbSystem3D::writeGeoObject(geoInflow.get(), pathOut+"/geo/geoInflow", WbWriterVtkXmlASCII::getInstance());

         //outflow
         GbCuboid3DPtr geoOutflow(new GbCuboid3D(g_maxX1, g_minX2-blockLength, g_minX3-blockLength, g_maxX1+blockLength, g_maxX2+blockLength, g_maxX3+blockLength));
         if (myid==0) GbSystem3D::writeGeoObject(geoOutflow.get(), pathOut+"/geo/geoOutflow", WbWriterVtkXmlASCII::getInstance());

         SPtr<CoProcessor> ppblocks(new WriteBlocksCoProcessor(grid, SPtr<UbScheduler>(new UbScheduler(1)), pathOut, WbWriterVtkXmlBinary::getInstance(), comm));

         if (refineLevel>0)
         {
            if (myid==0) UBLOG(logINFO, "Refinement - start");
            RefineCrossAndInsideGbObjectHelper refineHelper(grid, refineLevel, comm);
            refineHelper.addGbObject(refCylinder, refineLevel);
            refineHelper.refine();
            if (myid==0) UBLOG(logINFO, "Refinement - end");
         }

         //walls
         SPtr<D3Q27Interactor> addWallYminInt(new D3Q27Interactor(addWallYmin, grid, noSlipAdapter, Interactor3D::SOLID));
         SPtr<D3Q27Interactor> addWallYmaxInt(new D3Q27Interactor(addWallYmax, grid, noSlipAdapter, Interactor3D::SOLID));
         SPtr<D3Q27Interactor> addWallZminInt(new D3Q27Interactor(addWallZmin, grid, noSlipAdapter, Interactor3D::SOLID));
         SPtr<D3Q27Interactor> addWallZmaxInt(new D3Q27Interactor(addWallZmax, grid, noSlipAdapter, Interactor3D::SOLID));

         //inflow
         SPtr<D3Q27Interactor> inflowInt = SPtr<D3Q27Interactor>(new D3Q27Interactor(geoInflow, grid, velBCAdapter, Interactor3D::SOLID));

         //outflow
         SPtr<D3Q27Interactor> outflowInt = SPtr<D3Q27Interactor>(new D3Q27Interactor(geoOutflow, grid, denBCAdapter, Interactor3D::SOLID));

         
         SPtr<Grid3DVisitor> metisVisitor(new MetisPartitioningGridVisitor(comm, MetisPartitioningGridVisitor::LevelBased, D3Q27System::B));
         InteractorsHelper intHelper(grid, metisVisitor);
         intHelper.addInteractor(cylinderInt);
         intHelper.addInteractor(addWallYminInt);
         intHelper.addInteractor(addWallYmaxInt);
         intHelper.addInteractor(addWallZminInt);
         intHelper.addInteractor(addWallZmaxInt);
         intHelper.addInteractor(inflowInt);
         intHelper.addInteractor(outflowInt);
         intHelper.selectBlocks();


         ppblocks->process(0);
         ppblocks.reset();

         unsigned long long numberOfBlocks = (unsigned long long)grid->getNumberOfBlocks();
         int ghostLayer = 3;
         unsigned long long numberOfNodesPerBlock = (unsigned long long)(blockNx[0])* (unsigned long long)(blockNx[1])* (unsigned long long)(blockNx[2]);
         unsigned long long numberOfNodes = numberOfBlocks * numberOfNodesPerBlock;
         unsigned long long numberOfNodesPerBlockWithGhostLayer = numberOfBlocks * (blockNx[0]+ghostLayer) * (blockNx[1]+ghostLayer) * (blockNx[2]+ghostLayer);
         double needMemAll = double(numberOfNodesPerBlockWithGhostLayer*(27*sizeof(double)+sizeof(int)+sizeof(float)*4));
         double needMem = needMemAll/double(comm->getNumberOfProcesses());

         if (myid==0)
         {
            UBLOG(logINFO, "Number of blocks = "<<numberOfBlocks);
            UBLOG(logINFO, "Number of nodes  = "<<numberOfNodes);
            int minInitLevel = grid->getCoarsestInitializedLevel();
            int maxInitLevel = grid->getFinestInitializedLevel();
            for (int level = minInitLevel; level<=maxInitLevel; level++)
            {
               int nobl = grid->getNumberOfBlocks(level);
               UBLOG(logINFO, "Number of blocks for level "<<level<<" = "<<nobl);
               UBLOG(logINFO, "Number of nodes for level "<<level<<" = "<<nobl*numberOfNodesPerBlock);
            }
            UBLOG(logINFO, "Necessary memory  = "<<needMemAll<<" bytes");
            UBLOG(logINFO, "Necessary memory per process = "<<needMem<<" bytes");
            UBLOG(logINFO, "Available memory per process = "<<availMem<<" bytes");
         }

         SPtr<LBMKernel> kernel(new CompressibleCumulantLBMKernel());

         SPtr<BCProcessor> bcProc(new BCProcessor());
         kernel->setBCProcessor(bcProc);

         SetKernelBlockVisitor kernelVisitor(kernel, nueLB, availMem, needMem);
         grid->accept(kernelVisitor);

         if (refineLevel>0)
         {
            SetUndefinedNodesBlockVisitor undefNodesVisitor;
            grid->accept(undefNodesVisitor);
         }

         intHelper.setBC();

         //domain decomposition
         PQueuePartitioningGridVisitor pqPartVisitor(numOfThreads);
         grid->accept(pqPartVisitor);

         //initialization of distributions
         InitDistributionsBlockVisitor initVisitor;
         initVisitor.setVx1(fct);
         grid->accept(initVisitor);

;
         grid->accept(bcVisitor);

         //Postrozess
         SPtr<UbScheduler> geoSch(new UbScheduler(1));
         SPtr<CoProcessor> ppgeo(
            new WriteBoundaryConditionsCoProcessor(grid, geoSch, pathOut, WbWriterVtkXmlBinary::getInstance(), comm));
         ppgeo->process(0);
         ppgeo.reset();

         if (myid==0) UBLOG(logINFO, "Preprozess - end");
      }
      else
      {
         rcp.restart((int)restartStep);
         grid->setTimeStep(restartStep);

         grid->accept(bcVisitor);
      }

	  //set connectors
	  InterpolationProcessorPtr iProcessor(new CompressibleOffsetMomentsInterpolationProcessor());
	  SetConnectorsBlockVisitor setConnsVisitor(comm, true, D3Q27System::ENDDIR, nueLB, iProcessor);
	  grid->accept(setConnsVisitor);

      SPtr<UbScheduler> stepSch(new UbScheduler(outTime));

	  SPtr<CoProcessor> writeMQCoProcessor(new WriteMacroscopicQuantitiesCoProcessor(grid, stepSch, pathOut, WbWriterVtkXmlBinary::getInstance(), conv, comm));

      //double area = (2.0*radius*H)/(dx*dx);
      //double v    = 4.0*uLB/9.0;
      //CalculateForcesCoProcessor fp(grid, stepSch, pathOut + "/results/forces.txt", comm, v, area);
      //fp.addInteractor(cylinderInt);

	  SPtr<UbScheduler> nupsSch(new UbScheduler(nupsStep[0], nupsStep[1], nupsStep[2]));
	  std::shared_ptr<CoProcessor> nupsCoProcessor(new NUPSCounterCoProcessor(grid, nupsSch, numOfThreads, comm));

	  omp_set_num_threads(numOfThreads);
	  SPtr<UbScheduler> stepGhostLayer(new UbScheduler(1));
	  SPtr<Calculator> calculator(new BasicCalculator(grid, stepGhostLayer, endTime));
	  calculator->addCoProcessor(nupsCoProcessor);
	  calculator->addCoProcessor(writeMQCoProcessor);
      if(myid == 0) UBLOG(logINFO,"Simulation-start");
	  calculator->calculate();
      if(myid == 0) UBLOG(logINFO,"Simulation-end");
   }
   catch(std::exception& e)
   {
      cerr << e.what() << endl << flush;
   }
   catch(std::string& s)
   {
      cerr << s << endl;
   }
   catch(...)
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
         cout<<"Configuration file must be set!: "<<argv[0]<<" <config file>"<<endl<<std::flush;
      }
   }
   return 0;
}
