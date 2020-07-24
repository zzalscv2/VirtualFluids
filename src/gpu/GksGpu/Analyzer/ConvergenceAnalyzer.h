#ifndef  ConvergenceAnalyzer_H
#define  ConvergenceAnalyzer_H

#include <vector>

#include "VirtualFluidsDefinitions.h"

#include "Core/PointerDefinitions.h"
#include "Core/DataTypes.h"
#include "Core/Timer/Timer.h"

#include "FlowStateData/FlowStateData.cuh"

namespace GksGpu {

struct DataBase;

class VIRTUALFLUIDS_GPU_EXPORT ConvergenceAnalyzer
{
private:

    SPtr<DataBase> dataBase;

    std::vector<real> dataHostOld;
    std::vector<real> dataHostNew;

    uint outputIter;

    ConservedVariables convergenceThreshold;

public:

    ConvergenceAnalyzer( SPtr<DataBase> dataBase, uint outputIter = 10000, real convergenceThreshold = 1.0e-6 );

    void setConvergenceThreshold( real convergenceThreshold );
    void setConvergenceThreshold( ConservedVariables convergenceThreshold );

    bool run( uint iter );

private:

    void printL2Change( ConservedVariables L2Change );

};

} // namespace GksGpu

#endif