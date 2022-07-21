#ifndef CONFIG_FILE_READER_H
#define CONFIG_FILE_READER_H

#include "Utilities/Structs/ConfigDataStruct.h"

#include <memory>
#include <string>
#include <vector>

namespace vf::basics 
{
class ConfigurationFile;
}

namespace vf::gpu::tests
{
    std::shared_ptr<ConfigDataStruct> readConfigFile(const std::string aFilePath);
}

// private:
//     std::ifstream openConfigFile(const std::string aFilePath);
//     bool checkConfigFile(std::shared_ptr<vf::basics::ConfigurationFile> input);
//     std::vector<std::string> readKernelList(std::shared_ptr<vf::basics::ConfigurationFile> input);

//     int calcNumberOfSimulations(std::shared_ptr<vf::basics::ConfigurationFile> input);
//     int calcNumberOfSimulationGroup(std::shared_ptr<vf::basics::ConfigurationFile> input, std::string simName);
//     unsigned int calcStartStepForToVectorWriter(std::shared_ptr<vf::basics::ConfigurationFile> input);

//     std::vector<std::shared_ptr<TaylorGreenVortexUxParameterStruct>>
//     makeTaylorGreenVortexUxParameter(std::shared_ptr<vf::basics::ConfigurationFile> input,
//                                      std::shared_ptr<BasicSimulationParameterStruct> basicSimParameter);
//     std::vector<std::shared_ptr<TaylorGreenVortexUzParameterStruct>>
//     makeTaylorGreenVortexUzParameter(std::shared_ptr<vf::basics::ConfigurationFile> input,
//                                      std::shared_ptr<BasicSimulationParameterStruct> basicSimParameter);
//     std::vector<std::shared_ptr<ShearWaveParameterStruct>>
//     makeShearWaveParameter(std::shared_ptr<vf::basics::ConfigurationFile> input,
//                            std::shared_ptr<BasicSimulationParameterStruct> basicSimParameter);

//     std::shared_ptr<NyTestParameterStruct> makeNyTestParameter(std::shared_ptr<vf::basics::ConfigurationFile> input);
//     std::shared_ptr<PhiTestParameterStruct> makePhiTestParameter(std::shared_ptr<vf::basics::ConfigurationFile> input);
//     std::shared_ptr<L2NormTestParameterStruct> makeL2NormTestParameter(std::shared_ptr<vf::basics::ConfigurationFile> input);
//     std::shared_ptr<L2NormTestBetweenKernelsParameterStruct>
//     makeL2NormTestBetweenKernelsParameter(std::shared_ptr<vf::basics::ConfigurationFile> input);

//     std::shared_ptr<BasicSimulationParameterStruct> makeBasicSimulationParameter(std::shared_ptr<vf::basics::ConfigurationFile> input);
//     std::vector<std::shared_ptr<GridInformationStruct>> makeGridInformation(std::shared_ptr<vf::basics::ConfigurationFile> input,
//                                                                             std::string simName);

//     std::shared_ptr<VectorWriterInformationStruct>
//     makeVectorWriterInformationStruct(std::shared_ptr<vf::basics::ConfigurationFile> input);
//     std::shared_ptr<LogFileParameterStruct> makeLogFilePara(std::shared_ptr<vf::basics::ConfigurationFile> input);

//     std::string pathNumericalTests;

//     const std::string myFilePath;
//     std::shared_ptr<ConfigDataStruct> configData;
#endif