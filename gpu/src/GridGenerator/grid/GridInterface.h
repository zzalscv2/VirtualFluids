#ifndef GRID_INTERFACE_H
#define GRID_INTERFACE_H

#include "global.h"

class GridImp;

class GridInterface
{
public:
    HOSTDEVICE VF_PUBLIC GridInterface();
    HOSTDEVICE VF_PUBLIC ~GridInterface();

    HOSTDEVICE void VF_PUBLIC findInterfaceCF(const uint& indexOnCoarseGrid, GridImp* coarseGrid, GridImp* fineGrid);
    HOSTDEVICE void VF_PUBLIC findBoundaryGridInterfaceCF(const uint& indexOnCoarseGrid, GridImp* coarseGrid, GridImp* fineGrid);


	HOSTDEVICE void VF_PUBLIC findInterfaceCF_GKS(const uint& indexOnCoarseGrid, GridImp* coarseGrid, GridImp* fineGrid);

	HOSTDEVICE void VF_PUBLIC findInterfaceFC(const uint& indexOnCoarseGrid, GridImp* coarseGrid, GridImp* fineGrid);
    HOSTDEVICE void VF_PUBLIC findOverlapStopper(const uint& indexOnCoarseGrid, GridImp* coarseGrid, GridImp* fineGrid);
    
    HOSTDEVICE void VF_PUBLIC findInvalidBoundaryNodes(const uint& indexOnCoarseGrid, GridImp* coarseGrid);

    HOSTDEVICE void VF_PUBLIC findForGridInterfaceSparseIndexCF(GridImp* coarseGrid, GridImp* fineGrid, uint index);
    HOSTDEVICE void VF_PUBLIC findForGridInterfaceSparseIndexFC(GridImp* coarseGrid, GridImp* fineGrid, uint index);

    HOST void VF_PUBLIC repairGridInterfaceOnMultiGPU(SPtr<GridImp> coarseGrid, SPtr<GridImp> fineGrid);

    HOSTDEVICE void VF_PUBLIC print() const;

    struct Interface
    {
        uint *fine, *coarse;
        uint numberOfEntries = 0;
        uint *offset;
    } fc, cf;


private:
    HOSTDEVICE uint getCoarseToFineIndexOnFineGrid(const uint& indexOnCoarseGrid, const GridImp* coarseGrid, const GridImp* fineGrid);
    HOSTDEVICE bool isNeighborFineInvalid(real x, real y, real z, const GridImp* coarseGrid, const GridImp* fineGrid);

    HOSTDEVICE uint getFineToCoarseIndexOnFineGrid(const uint& indexOnCoarseGrid, const GridImp* coarseGrid, const GridImp* fineGrid);

    HOSTDEVICE static void findSparseIndex(uint* indices, GridImp* grid, uint index);

    HOSTDEVICE uint findOffsetCF( const uint& indexOnCoarseGrid, GridImp* coarseGrid, uint interfaceIndex );

    HOSTDEVICE uint findOffsetFC( const uint& indexOnCoarseGrid, GridImp* coarseGrid, uint interfaceIndex );
};


#endif