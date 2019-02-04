#ifndef TIME_INFO_H
#define TIME_INFO_H

#include <string>

class TimeInfo
{
public:
	virtual std::string getSimulationTime() = 0;
	virtual std::string getTestTime() = 0;
	virtual std::string getAnalyticalResultWriteTime() = 0;
};
#endif