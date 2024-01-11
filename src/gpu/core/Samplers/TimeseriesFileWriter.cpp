#include "TimeseriesFileWriter.h"

#include <string>
#include <vector>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <filesystem>


#include <basics/DataTypes.h>

void TimeseriesFileWriter::writeHeader(const std::string& fileName, int numberOfPoints, std::vector<std::string>& variableNames, const real* coordsX, const real* coordsY, const real* coordsZ) const
{
    std::filesystem::create_directories(std::filesystem::path(fileName).parent_path());
    std::ofstream out(fileName.c_str(), std::ios::out | std::ios::binary);

    if (!out.is_open())
        throw std::runtime_error("Could not open timeseries file " + fileName + "!");

    out << "TimeseriesOutput \n";
    out << "Quantities: ";
    for (const std::string& name : variableNames)
        out << name << ", ";
    out << "\n";
    out << "Number of points in this file: \n";
    out << numberOfPoints << "\n";
    out << "Positions: x, y, z\n";
    for (int i = 0; i < numberOfPoints; i++)
        out << coordsX[i] << ", " << coordsY[i] << ", " << coordsZ[i]
    << "\n";

    out.close();
}

void TimeseriesFileWriter::appendData(const std::string& fileName, std::vector<std::vector<real>>& data) const {
    std::ofstream out(fileName.c_str(), std::ios::app | std::ios::binary);
    for (auto& timestepData : data) {
        out.write((char*)timestepData.data(), sizeof(real) * timestepData.size());
    }
    out.close();
}