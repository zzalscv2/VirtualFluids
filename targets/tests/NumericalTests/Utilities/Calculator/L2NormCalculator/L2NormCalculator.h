#ifndef L2NORM_CALCULATOR_H
#define L2NORM_CALCULATOR_H

#include <vector>
#include <memory>

class L2NormCalculator
{
public:
	static std::shared_ptr< L2NormCalculator> getNewInstance();

	double calc(std::vector<double> basicData, std::vector<double> divergentData, std::vector<unsigned int> level);

private:
	L2NormCalculator();
};
#endif