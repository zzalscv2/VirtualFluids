#ifndef GRID_INTERFACE_H
#define GRID_INTERFACE_H

#include "core/DataTypes.h"
#include "VirtualFluidsDefinitions.h"

struct Grid;

class GridInterface
{
public:
    VF_PUBLIC GridInterface(const Grid* finerGrid);
    VF_PUBLIC ~GridInterface();
    void VF_PUBLIC findCF(const uint& index, const Grid* coarseGrid, const Grid* fineGrid);
    void VF_PUBLIC findFC(const uint& index, const Grid* coarseGrid, const Grid* fineGrid);

private:
    struct Interface
    {
        uint *fine, *coarse;
        uint numberOfEntries = 0;
        real startCoarseX;
        real startCoarseY;
        real startCoarseZ;

        real endCoarseX;
        real endCoarseY;
        real endCoarseZ;

        char coarseEntry, fineEntry;
    } fc, cf;

    void findInterface(Interface& interface, const int& factor, const uint& index, const Grid* coarseGrid, const Grid* fineGrid) const;
    static bool isOnInterface(Interface& interface, const real& x, const real& y, const real& z);
    uint getIndexOnFinerGrid(const int& factor, const Grid* fineGrid, const real& x, const real& y, const  real& z) const;
};


#endif