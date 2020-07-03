#ifdef VF_CATALYST

#ifndef InSituCatalystCoProcessor_h__
#define InSituCatalystCoProcessor_h__


#include <CoProcessor.h>
#include <Grid3D.h>
#include <LBMUnitConverter.h>

#include <string>

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDoubleArray.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkCPPythonScriptPipeline.h>
#include <vtkNew.h>

class InSituCatalystCoProcessor : public CoProcessor
{
public:
   InSituCatalystCoProcessor();
   InSituCatalystCoProcessor(SPtr<Grid3D> grid, SPtr<UbScheduler> s, std::string script);
   virtual ~InSituCatalystCoProcessor(); 
   void process(double step);
protected:
   void collectData(double step);
   void addData(SPtr<Block3D> block);
   void buildVTKGrid();
   void addVTKGridData(SPtr<Block3D> block);
private:
   std::vector<std::vector<SPtr<Block3D>> > blockVector;
   int minInitLevel;
   int maxInitLevel;
   int gridRank;
   vtkSmartPointer<vtkCPProcessor> Processor;
   vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid;
   vtkSmartPointer<vtkPoints> points;
   vtkSmartPointer<vtkDoubleArray> arrays[4];
   std::vector<double> vx1Array;
   std::vector<double> vx2Array;
   std::vector<double> vx3Array;
   std::vector<double> rhoArray;
   int index;
   int numOfPoints;
   typedef void(*CalcMacrosFct)(const LBMReal* const& /*feq[27]*/, LBMReal& /*(d)rho*/, LBMReal& /*vx1*/, LBMReal& /*vx2*/, LBMReal& /*vx3*/);
   CalcMacrosFct calcMacros;
};
#endif // InSituCatalystCoProcessor_h__

#endif