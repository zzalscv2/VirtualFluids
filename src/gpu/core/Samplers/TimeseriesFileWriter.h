#ifndef TIMESERIESFILEWRITER_H
#define TIMESERIESFILEWRITER_H
#include <string>
#include <vector>

#include <basics/DataTypes.h>

/*
File Layout:
TimeseriesOutput
Quantities: Quant1 Quant2 Quant3
Positions:
point1.x, point1.y, point1.z
point2.x, point2.y, point2.z
...
t0 point1.quant1 point2.quant1 ... point1.quant2 point2.quant2 ...
t1 point1.quant1 point2.quant1 ... point1.quant2 point2.quant2 ...
*/

class TimeseriesFileWriter{
public: 
    TimeseriesFileWriter() = default;

    void writeHeader(const std::string& fileName, int numberOfPoints, std::vector<std::string>& variableNames, const real* coordsX, const real* coordsY, const real* coordsZ) const;
    void appendData(const std::string& fileName, std::vector<std::vector<real>>& data) const;
};

#endif