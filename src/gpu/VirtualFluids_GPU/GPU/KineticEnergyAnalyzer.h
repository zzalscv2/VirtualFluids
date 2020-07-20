#ifndef  KineticEngergyAnalyzer_H
#define  KineticEngergyAnalyzer_H

#include <vector>
#include <string>

#include "VirtualFluidsDefinitions.h"

#include "Core/PointerDefinitions.h"
#include "Core/DataTypes.h"

class Parameter;

class VIRTUALFLUIDS_GPU_EXPORT KineticEnergyAnalyzer
{
private:

	SPtr<Parameter> para;

    uint analyzeIter;

    std::vector<real> kineticEnergyTimeSeries;

public:

    KineticEnergyAnalyzer( SPtr<Parameter> para, uint analyzeIter );

    bool run( uint iter );

    void writeToFile( std::string filename );

};

#endif
