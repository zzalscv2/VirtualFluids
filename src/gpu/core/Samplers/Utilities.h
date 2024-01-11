#include <basics/StringUtilities/StringUtil.h>
#include <string>

template <typename T>
inline std::string nameComponent(const std::string& name, T value)
{
    return "_" + name + "_" + StringUtil::toString<T>(value);
}

inline std::string makeParallelFileName(const std::string& probeName, int id, int t)
{
    return probeName + "_bin" + nameComponent("ID", id) + nameComponent("t", t) + ".vtk";
}

inline std::string makeGridFileName(const std::string& probeName, int level, int id, int t, uint part)
{
    return probeName + "_bin" + nameComponent("lev", level) + nameComponent("ID", id) + nameComponent<int>("Part", part) +
           nameComponent("t", t) + ".vtk";
}

inline std::string makeTimeseriesFileName(const std::string& probeName, int level, int id)
{
    return probeName + "_timeseries" + nameComponent("lev", level) + nameComponent("ID", id) + ".txt";
}