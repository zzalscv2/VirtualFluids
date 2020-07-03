#ifndef INIT_SP27_H
#define INIT_SP27_H

#include "PreProcessor/PreProcessorStrategy/PreProcessorStrategy.h"

#include <memory>

class Parameter;

class InitSP27 : public PreProcessorStrategy
{
public:
	static std::shared_ptr<PreProcessorStrategy> getNewInstance(std::shared_ptr< Parameter> para);
	void init(int level);
	bool checkParameter();

private:
	InitSP27();
	InitSP27(std::shared_ptr< Parameter> para);
	std::shared_ptr< Parameter> para;
};

#endif 