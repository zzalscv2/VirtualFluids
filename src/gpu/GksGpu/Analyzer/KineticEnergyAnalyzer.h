#ifndef  KineticEngergyAnalyzer_H
#define  KineticEngergyAnalyzer_H

#include <vector>
#include <string>

#include "VirtualFluidsDefinitions.h"

#include "PointerDefinitions.h"
#include "Core/DataTypes.h"

#include "FlowStateData/FlowStateData.cuh"

namespace GksGpu {

struct DataBase;

class VIRTUALFLUIDS_GPU_EXPORT KineticEnergyAnalyzer
{
private:

    SPtr<DataBase> dataBase;

    uint outputIter;

    uint analyzeIter;

    std::vector<real> kineticEnergyTimeSeries;

public:

    KineticEnergyAnalyzer( SPtr<DataBase> dataBase, uint analyzeIter, uint outputIter );

    bool run( uint iter );

    void writeToFile( std::string filename );

};

} // namespace GksGpu

#endif
