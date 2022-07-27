#ifndef VIRTUAL_FLUID_SIMULATION_IMP_H
#define VIRTUAL_FLUID_SIMULATION_IMP_H

#include <memory>
#include <string>

class CudaMemoryManager;
class InitialCondition;
class DataWriter;
class Parameter;
class GridProvider;
class KernelConfiguration;
class TestSimulation;
class TimeTracking;
class NumericalTestSuite;
class KernelFactory;
class PreProcessorFactory;

class VirtualFluidSimulation
{
public:
    void run();

    void setParameter(std::shared_ptr<Parameter> para);
    void setCudaMemoryManager(std::shared_ptr<CudaMemoryManager> cudaManager);
    void setGridProvider(std::shared_ptr<GridProvider> grid);
    void setDataWriter(std::shared_ptr<DataWriter> dataWriter);
    void setNumericalTestSuite(std::shared_ptr<NumericalTestSuite> numericalTestSuite);
    void setTimeTracking(std::shared_ptr<TimeTracking> timeTracking);

    void setKernelFactory(std::shared_ptr<KernelFactory> kernelFactory);
    void setPreProcessorFactory(std::shared_ptr<PreProcessorFactory> preProcessorFactory);

private:
    std::shared_ptr<Parameter> para;
    std::shared_ptr<CudaMemoryManager> cudaManager;
    std::shared_ptr<InitialCondition> initialCondition;
    std::shared_ptr<GridProvider> grid;
    std::shared_ptr<DataWriter> dataWriter;
    std::shared_ptr<NumericalTestSuite> numericalTestSuite;
    std::shared_ptr<TimeTracking> timeTracking;

    std::shared_ptr<KernelFactory> kernelFactory;
    std::shared_ptr<PreProcessorFactory> preProcessorFactory;
};
#endif