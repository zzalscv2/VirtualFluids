#ifndef DATA_WRITER_H
#define DATA_WRITER_H

#include "PointerDefinitions.h"
#include "Core/DataTypes.h"
#include "VirtualFluids_GPU_export.h"

#include <memory>
#include <vector>
#include <string>


class Parameter;
class CudaMemoryManager;

class DataWriter
{
public:
	DataWriter() {}
    virtual ~DataWriter() {}

    virtual void writeInit(std::shared_ptr<Parameter> para, std::shared_ptr<CudaMemoryManager> cudaManager) = 0;
    virtual void writeTimestep(std::shared_ptr<Parameter> para, unsigned int timestep) = 0;
	virtual void writeTimestep(std::shared_ptr<Parameter> para, unsigned int timestep, int level) = 0;

    DataWriter(const DataWriter& dataWriter) {}
};
#endif
