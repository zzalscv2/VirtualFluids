#include "ConfigFileReaderNT.h"

#include "Core/Input/Input.h"
#include "Core/StringUtilities/StringUtil.h"

#include "VirtualFluids_GPU/Kernel/Utilities/Mapper/KernelMapper/KernelMapper.h"

#include <string>
#include <fstream>

std::shared_ptr<ConfigFileReader> ConfigFileReader::getNewInstance(const std::string aFilePath)
{
	return std::shared_ptr<ConfigFileReader>(new ConfigFileReader(aFilePath));
}

ConfigFileReader::ConfigFileReader(const std::string aFilePath) : myFilePath(aFilePath)
{
	myKernelMapper = KernelMapper::getInstance();
}

void ConfigFileReader::readConfigFile()
{
	configData = std::shared_ptr<ConfigDataStruct>(new ConfigDataStruct);
	std::ifstream stream = openConfigFile(myFilePath);

	std::shared_ptr<input::Input> input = input::Input::makeInput(stream, "config");

	if (!checkConfigFile(input))
		exit(1);

	configData->viscosity = StringUtil::toDoubleVector(input->getValue("Viscosity"));
	configData->kernelsToTest = readKernelList(input);
	configData->writeAnalyticalToVTK = StringUtil::toBool(input->getValue("WriteAnalyResultsToVTK"));
	configData->ySliceForCalculation = StringUtil::toInt(input->getValue("ySliceForCalculation"));;
	configData->logFilePath = input->getValue("PathLogFile");
	configData->numberOfSimulations = calcNumberOfSimulations(input);

	std::shared_ptr<BasicSimulationParameterStruct> basicSimPara = makeBasicSimulationParameter(input);

	configData->taylorGreenVortexUxParameter = makeTaylorGreenVortexUxParameter(input, basicSimPara);
	configData->taylorGreenVortexUxGridInformation = makeGridInformation(input, "TaylorGreenVortexUx");;

	configData->taylorGreenVortexUzParameter = makeTaylorGreenVortexUzParameter(input, basicSimPara);
	configData->taylorGreenVortexUzGridInformation = makeGridInformation(input, "TaylorGreenVortexUz");;

	configData->shearWaveParameter = makeShearWaveParameter(input, basicSimPara);
	configData->shearWaveGridInformation = makeGridInformation(input, "ShearWave");;

	configData->phiTestParameter = makePhiTestParameter(input);
	configData->nyTestParameter = makeNyTestParameter(input);
	configData->l2NormTestParameter = makeL2NormTestParameter(input);
	configData->l2NormTestBetweenKernelsParameter = makeL2NormTestBetweenKernelsParameter(input);

	configData->vectorWriterInfo = makeVectorWriterInformationStruct(input);

	configData->logFilePara = makeLogFilePara(input);

	stream.close();
}

std::ifstream ConfigFileReader::openConfigFile(const std::string aFilePath)
{
	std::ifstream stream;
	stream.open(aFilePath.c_str(), std::ios::in);
	if (stream.fail())
		throw "can not open config file!\n";

	return stream;
}

std::shared_ptr<ConfigDataStruct> ConfigFileReader::getConfigData()
{
	return configData;
}

bool ConfigFileReader::checkConfigFile(std::shared_ptr<input::Input> input)
{
	std::vector<double> u0TGVux = StringUtil::toDoubleVector(input->getValue("ux_TGV_Ux"));
	std::vector<double> amplitudeTGVux = StringUtil::toDoubleVector(input->getValue("Amplitude_TGV_Ux"));
	std::vector<int> basisTimeStepLengthTGVux = StringUtil::toIntVector(input->getValue("BasisTimeStepLength_TGV_Ux"));

	std::vector<double> v0TGVuz = StringUtil::toDoubleVector(input->getValue("uz_TGV_Uz"));
	std::vector<double> amplitudeTGVuz = StringUtil::toDoubleVector(input->getValue("Amplitude_TGV_Uz"));
	std::vector<int> basisTimeStepLengthTGVuz = StringUtil::toIntVector(input->getValue("BasisTimeStepLength_TGV_Uz"));

	std::vector<double> v0SW = StringUtil::toDoubleVector(input->getValue("v0_SW"));
	std::vector<double> u0SW = StringUtil::toDoubleVector(input->getValue("u0_SW"));
	std::vector<int> basisTimeStepLengthSW = StringUtil::toIntVector(input->getValue("BasisTimeStepLength_SW"));

	if (u0TGVux.size() != amplitudeTGVux.size() || u0TGVux.size() != basisTimeStepLengthTGVux.size()) {
		std::cout << "Length u0_TGV_U0 is unequal to Lenght Amplitude_TGV_U0 or BasisTimeStepLength_TGV_U0!" << std::endl << std::flush;
		return false;
	} else if (v0TGVuz.size() != amplitudeTGVuz.size() || v0TGVuz.size() != basisTimeStepLengthTGVuz.size()) {
		std::cout << "Length v0_TGV_V0 is unequal to Lenght Amplitude_TGV_V0 or BasisTimeStepLength_TGV_V0!" << std::endl << std::flush;
		return false;
	} else if (u0SW.size() != v0SW.size() || u0SW.size() != basisTimeStepLengthSW.size()) {
		std::cout << "Length u0_SW is unequal to Lenght v0_SW!" << std::endl << std::flush;
		return false;
	}
	else
	{
		return true;
	}
}

std::shared_ptr<BasicSimulationParameterStruct> ConfigFileReader::makeBasicSimulationParameter(std::shared_ptr<input::Input> input)
{
	std::shared_ptr<BasicSimulationParameterStruct> basicSimPara = std::shared_ptr<BasicSimulationParameterStruct>(new BasicSimulationParameterStruct);

	basicSimPara->numberOfTimeSteps = StringUtil::toInt(input->getValue("NumberOfTimeSteps"));
	basicSimPara->devices = StringUtil::toUintVector(input->getValue("Devices"));
	return basicSimPara;
}


std::vector<std::shared_ptr<TaylorGreenVortexUxParameterStruct> > ConfigFileReader::makeTaylorGreenVortexUxParameter(std::shared_ptr<input::Input> input, std::shared_ptr<BasicSimulationParameterStruct> basicSimParameter)
{
	std::vector<int> basisTimeStepLength = StringUtil::toIntVector(input->getValue("BasisTimeStepLength_TGV_Ux"));
	std::vector<double> amplitude = StringUtil::toDoubleVector(input->getValue("Amplitude_TGV_Ux"));
	std::vector<double> u0 = StringUtil::toDoubleVector(input->getValue("ux_TGV_Ux"));
	int l0 = StringUtil::toInt(input->getValue("l0_TGV_Ux"));
	basicSimParameter->l0 = l0;

	std::vector<std::shared_ptr<TaylorGreenVortexUxParameterStruct> > parameter;
	for (int i = 0; i < u0.size(); i++) {
		std::shared_ptr<TaylorGreenVortexUxParameterStruct> aParameter = std::shared_ptr<TaylorGreenVortexUxParameterStruct>(new TaylorGreenVortexUxParameterStruct);
		aParameter->basicSimulationParameter = basicSimParameter;
		
		aParameter->ux = u0.at(i);
		aParameter->amplitude = amplitude.at(i);
		aParameter->basicTimeStepLength = basisTimeStepLength.at(i);
		aParameter->l0 = l0;
		aParameter->rho0 = StringUtil::toDouble(input->getValue("Rho0"));
		aParameter->vtkFilePath = input->getValue("PathForVTKFileWriting");
		aParameter->dataToCalcTests = StringUtil::toStringVector(input->getValue("DataToCalcTests_TGV_Ux"));
		parameter.push_back(aParameter);
	}
	return parameter;
}

std::vector<std::shared_ptr<TaylorGreenVortexUzParameterStruct> > ConfigFileReader::makeTaylorGreenVortexUzParameter(std::shared_ptr<input::Input> input, std::shared_ptr<BasicSimulationParameterStruct> basicSimParameter)
{
	std::vector<int> basisTimeStepLength = StringUtil::toIntVector(input->getValue("BasisTimeStepLength_TGV_Uz"));
	std::vector<double> amplitude = StringUtil::toDoubleVector(input->getValue("Amplitude_TGV_Uz"));
	std::vector<double> uz = StringUtil::toDoubleVector(input->getValue("uz_TGV_Uz"));
	int l0 = StringUtil::toInt(input->getValue("l0_TGV_Uz"));
	basicSimParameter->l0 = l0;

	std::vector<std::shared_ptr<TaylorGreenVortexUzParameterStruct> > parameter;
	for (int i = 0; i < uz.size(); i++) {
		std::shared_ptr<TaylorGreenVortexUzParameterStruct> aParameter = std::shared_ptr<TaylorGreenVortexUzParameterStruct>(new TaylorGreenVortexUzParameterStruct);
		aParameter->basicSimulationParameter = basicSimParameter;
		aParameter->uz = uz.at(i);
		aParameter->amplitude = amplitude.at(i);
		aParameter->basicTimeStepLength = basisTimeStepLength.at(i);
		aParameter->l0 = l0;
		aParameter->rho0 = StringUtil::toDouble(input->getValue("Rho0"));
		aParameter->vtkFilePath = input->getValue("PathForVTKFileWriting");
		aParameter->dataToCalcTests = StringUtil::toStringVector(input->getValue("DataToCalcTests_TGV_Uz"));
		parameter.push_back(aParameter);
	}
	return parameter;
}
std::vector<std::shared_ptr<ShearWaveParameterStruct> > ConfigFileReader::makeShearWaveParameter(std::shared_ptr<input::Input> input, std::shared_ptr<BasicSimulationParameterStruct> basicSimParameter)
{
	std::vector<int> basisTimeStepLength = StringUtil::toIntVector(input->getValue("BasisTimeStepLength_SW"));
	std::vector<double> uz = StringUtil::toDoubleVector(input->getValue("v0_SW"));
	std::vector<double> ux = StringUtil::toDoubleVector(input->getValue("u0_SW"));
	int l0 = StringUtil::toInt(input->getValue("l0_SW"));
	basicSimParameter->l0 = l0;

	std::vector<std::shared_ptr<ShearWaveParameterStruct> > parameter;
	for (int i = 0; i < uz.size(); i++) {
		std::shared_ptr<ShearWaveParameterStruct> aParameter = std::shared_ptr<ShearWaveParameterStruct>(new ShearWaveParameterStruct);
		aParameter->basicSimulationParameter = basicSimParameter;
		aParameter->uz = uz.at(i);
		aParameter->ux = ux.at(i);
		aParameter->basicTimeStepLength = basisTimeStepLength.at(i);
		aParameter->l0 = l0;
		aParameter->rho0 = StringUtil::toDouble(input->getValue("Rho0"));
		aParameter->vtkFilePath = input->getValue("PathForVTKFileWriting");
		aParameter->dataToCalcTests = StringUtil::toStringVector(input->getValue("DataToCalcTests_SW"));
		parameter.push_back(aParameter);
	}
	return parameter;
}

std::shared_ptr<NyTestParameterStruct> ConfigFileReader::makeNyTestParameter(std::shared_ptr<input::Input> input)
{
	std::shared_ptr<BasicTestParameterStruct> basicTestParameter = std::shared_ptr<BasicTestParameterStruct>(new BasicTestParameterStruct);
	basicTestParameter->runTest = StringUtil::toBool(input->getValue("NyTest"));
	basicTestParameter->ySliceForCalculation = StringUtil::toInt(input->getValue("ySliceForCalculation"));

	std::shared_ptr<NyTestParameterStruct> testParameter = std::shared_ptr<NyTestParameterStruct>(new NyTestParameterStruct);
	testParameter->basicTestParameter = basicTestParameter;
	testParameter->endTimeStepCalculation = StringUtil::toInt(input->getValue("EndTimeStepCalculation_Ny"));
	testParameter->minOrderOfAccuracy = StringUtil::toDouble(input->getValue("MinOrderOfAccuracy_Ny"));
	testParameter->startTimeStepCalculation = StringUtil::toInt(input->getValue("StartTimeStepCalculation_Ny"));

	return testParameter;
}

std::shared_ptr<PhiTestParameterStruct> ConfigFileReader::makePhiTestParameter(std::shared_ptr<input::Input> input)
{
	std::shared_ptr<BasicTestParameterStruct> basicTestParameter = std::shared_ptr<BasicTestParameterStruct>(new BasicTestParameterStruct);
	basicTestParameter->runTest = StringUtil::toBool(input->getValue("PhiTest"));
	basicTestParameter->ySliceForCalculation = StringUtil::toInt(input->getValue("ySliceForCalculation"));

	std::shared_ptr<PhiTestParameterStruct> testParameter = std::shared_ptr<PhiTestParameterStruct>(new PhiTestParameterStruct);
	testParameter->basicTestParameter = basicTestParameter;
	testParameter->endTimeStepCalculation = StringUtil::toInt(input->getValue("EndTimeStepCalculation_Phi"));
	testParameter->minOrderOfAccuracy = StringUtil::toDouble(input->getValue("MinOrderOfAccuracy_Phi"));
	testParameter->startTimeStepCalculation = StringUtil::toInt(input->getValue("StartTimeStepCalculation_Phi"));

	return testParameter;
}

std::shared_ptr<L2NormTestParameterStruct> ConfigFileReader::makeL2NormTestParameter(std::shared_ptr<input::Input> input)
{
	std::shared_ptr<BasicTestParameterStruct> basicTestParameter = std::shared_ptr<BasicTestParameterStruct>(new BasicTestParameterStruct);
	basicTestParameter->runTest = StringUtil::toBool(input->getValue("L2NormTest"));
	basicTestParameter->ySliceForCalculation = StringUtil::toInt(input->getValue("ySliceForCalculation"));

	std::shared_ptr<L2NormTestParameterStruct> testParameter = std::shared_ptr<L2NormTestParameterStruct>(new L2NormTestParameterStruct);
	testParameter->basicTestParameter = basicTestParameter;
	testParameter->basicTimeStep = StringUtil::toInt(input->getValue("BasicTimeStep_L2"));
	testParameter->divergentTimeStep = StringUtil::toInt(input->getValue("DivergentTimeStep_L2"));
	testParameter->normalizeData = StringUtil::toStringVector(input->getValue("NormalizeData_L2Norm"));
	testParameter->maxDiff = StringUtil::toDoubleVector(input->getValue("MaxL2NormDiff"));

	return testParameter;
}

std::shared_ptr<L2NormTestBetweenKernelsParameterStruct> ConfigFileReader::makeL2NormTestBetweenKernelsParameter(std::shared_ptr<input::Input> input)
{
	std::shared_ptr<BasicTestParameterStruct> basicTestParameter = std::shared_ptr<BasicTestParameterStruct>(new BasicTestParameterStruct);
	basicTestParameter->runTest = StringUtil::toBool(input->getValue("L2NormBetweenKernelsTest"));
	basicTestParameter->ySliceForCalculation = StringUtil::toInt(input->getValue("ySliceForCalculation"));

	std::shared_ptr<L2NormTestBetweenKernelsParameterStruct> testParameter = std::shared_ptr<L2NormTestBetweenKernelsParameterStruct>(new L2NormTestBetweenKernelsParameterStruct);
	testParameter->basicTestParameter = basicTestParameter;
	testParameter->basicKernel = myKernelMapper->getEnum(input->getValue("BasicKernel_L2NormBetweenKernels"));
	testParameter->kernelsToTest = readKernelList(input);
	testParameter->timeSteps = StringUtil::toIntVector(input->getValue("Timesteps_L2NormBetweenKernels"));
	testParameter->normalizeData = StringUtil::toStringVector(input->getValue("NormalizeData_L2Norm"));

	bool correct = false;
	for (int i = 0; i < testParameter->normalizeData.size(); i++)
		if (testParameter->normalizeData.at(i) == "Amplitude" || testParameter->normalizeData.at(i) == "BasicData")
			correct = true;
	

	if (!correct) {
		std::cout << "invalid input in ConfigFile." << std::endl << "possible data for NormalizeWith Parameter in L2-Norm Test Between Kernels Parameter:" << std::endl << "Amplitude, BasicData" << std::endl << std::endl;
		exit(1);
	}

	return testParameter;
}

std::vector<std::shared_ptr<GridInformationStruct> > ConfigFileReader::makeGridInformation(std::shared_ptr<input::Input> input, std::string simName)
{
	int number = 32;
	std::vector<std::string> valueNames;
	std::vector<std::string> gridPaths;
	for (int i = 1; i <= 5; i++) {
		std::string aValueName = simName;
		aValueName += std::to_string(number);
		valueNames.push_back(aValueName);
		std::string aGridpath = "GridPath";
		aGridpath += std::to_string(number);
		gridPaths.push_back(aGridpath);
		number *= 2;
	}
	
	std::vector<double> lx;
	std::vector<double> lz;
	std::vector<std::string> gridPath;

	double nextNumber = 32.0;

	for (int i = 0; i < valueNames.size(); i++) {
		if (StringUtil::toBool(input->getValue(valueNames.at(i)))) {
			lx.push_back(nextNumber);
			lz.push_back(nextNumber * 3.0 / 2.0);
			gridPath.push_back(input->getValue(gridPaths.at(i)));
			nextNumber *= 2;
		}
	}

	std::vector<std::shared_ptr<GridInformationStruct> > gridInformation;
	for (int i = 0; i < lx.size(); i++) {
		std::shared_ptr<GridInformationStruct> aGridInformation = std::shared_ptr<GridInformationStruct> (new GridInformationStruct);
		aGridInformation->numberOfGridLevels = StringUtil::toInt(input->getValue("NumberOfGridLevels"));
		aGridInformation->maxLevel = aGridInformation->numberOfGridLevels - 1;
		aGridInformation->gridPath = gridPath.at(i);
		aGridInformation->lx = lx.at(i);
		aGridInformation->lz = lz.at(i);
		gridInformation.push_back(aGridInformation);
	}
	return gridInformation;
}

std::shared_ptr<VectorWriterInformationStruct> ConfigFileReader::makeVectorWriterInformationStruct(std::shared_ptr<input::Input> input)
{
	std::shared_ptr<VectorWriterInformationStruct> vectorWriter = std::shared_ptr<VectorWriterInformationStruct>(new VectorWriterInformationStruct);
	vectorWriter->startTimeVectorWriter = calcStartStepForToVectorWriter(input);
	vectorWriter->startTimeVTKDataWriter = StringUtil::toInt(input->getValue("StartStepFileWriter"));
	vectorWriter->writeVTKFiles = StringUtil::toBool(input->getValue("WriteVTKFiles"));

	return vectorWriter;
}

std::shared_ptr<LogFileParameterStruct> ConfigFileReader::makeLogFilePara(std::shared_ptr<input::Input> input)
{
	std::shared_ptr<LogFileParameterStruct> logFilePara = std::shared_ptr<LogFileParameterStruct>(new LogFileParameterStruct);
	logFilePara->devices = StringUtil::toIntVector(input->getValue("Devices"));
	logFilePara->numberOfTimeSteps = StringUtil::toInt(input->getValue("NumberOfTimeSteps"));
	logFilePara->writeAnalyticalToVTK = StringUtil::toBool(input->getValue("WriteAnalyResultsToVTK"));

	return logFilePara;
}

std::vector<KernelType> ConfigFileReader::readKernelList(std::shared_ptr<input::Input> input)
{
	if (StringUtil::toBool(input->getValue("L2NormBetweenKernelsTest"))) {
		std::vector<std::string> kernelList = StringUtil::toStringVector(input->getValue("KernelsToTest"));
		std::string beginnKernel = input->getValue("BasicKernel_L2NormBetweenKernels");
		bool basicKernelInKernelList = false;
		for (int i = 0; i < kernelList.size(); i++) {
			if (kernelList.at(i) == beginnKernel)
				basicKernelInKernelList = true;
		}
		if (!basicKernelInKernelList)
			kernelList.push_back(beginnKernel);

		std::vector<std::string> kernelNames = kernelList;

		while (kernelNames.at(0) != beginnKernel) {
			kernelNames.push_back(kernelNames.at(0));
			std::vector<std::string>::iterator it = kernelNames.begin();
			kernelNames.erase(it);
		}
		std::vector<KernelType> kernels;
		for (int i = 0; i < kernelNames.size(); i++)
			kernels.push_back(myKernelMapper->getEnum(kernelNames.at(i)));
		return kernels;
	}else {
		std::vector<std::string> kernelList = StringUtil::toStringVector(input->getValue("KernelsToTest"));
		std::vector<KernelType> kernels;
		for (int i = 0; i < kernelList.size(); i++)
			kernels.push_back(myKernelMapper->getEnum(kernelList.at(i)));

		return kernels;
	}	
}

unsigned int ConfigFileReader::calcStartStepForToVectorWriter(std::shared_ptr<input::Input> input)
{
	std::vector<unsigned int> startStepsTests;
	startStepsTests.push_back(StringUtil::toInt(input->getValue("BasicTimeStep_L2")));
	startStepsTests.push_back(StringUtil::toInt(input->getValue("StartTimeStepCalculation_Ny")));
	startStepsTests.push_back(StringUtil::toInt(input->getValue("StartTimeStepCalculation_Phi")));
	std::sort(startStepsTests.begin(), startStepsTests.end());

	return startStepsTests.at(0);
}

int ConfigFileReader::calcNumberOfSimulations(std::shared_ptr<input::Input> input)
{
	int counter = 0;

	int tgvCounterU0 = calcNumberOfSimulationGroup(input, "TaylorGreenVortexUx");
	tgvCounterU0 *= StringUtil::toDoubleVector(input->getValue("ux_TGV_Ux")).size();
	counter += tgvCounterU0;

	int tgvCounterV0 = calcNumberOfSimulationGroup(input, "TaylorGreenVortexUz");;
	tgvCounterV0 *= StringUtil::toDoubleVector(input->getValue("uz_TGV_Uz")).size();
	counter += tgvCounterV0;

	int swCounter = calcNumberOfSimulationGroup(input, "ShearWave");;
	swCounter *= StringUtil::toDoubleVector(input->getValue("u0_SW")).size();
	counter += swCounter;

	counter *= StringUtil::toDoubleVector(input->getValue("Viscosity")).size();
	counter *= configData->kernelsToTest.size();

	return counter;
}

int ConfigFileReader::calcNumberOfSimulationGroup(std::shared_ptr<input::Input> input, std::string simName)
{
	int counter = 0;
	int number = 32;
	std::vector<std::string> valueNames;
	for (int i = 1; i <= 5; i++) {
		std::string aValueName = simName;
		aValueName += std::to_string(number);
		valueNames.push_back(aValueName);
		number *= 2;
	}
	for (int i = 0; i < valueNames.size(); i++) {
		if (StringUtil::toBool(input->getValue(valueNames.at(i))))
			counter++;
	}
	return counter;
}