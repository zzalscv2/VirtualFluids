#include "GridInterface.h"

#include "GridImp.h"
#include "Field.h"
#include "NodeValues.h"
#include "utilities/math/CudaMath.cuh"

GridInterface::GridInterface()
{

}

GridInterface::~GridInterface()
{

}



void GridInterface::findForGridInterfaceSparseIndexCF(GridImp* coarseGrid, GridImp* fineGrid, uint index)
{
    const uint matrixIndexCoarse = this->cf.coarse[index];
    const uint sparseIndexCoarse = coarseGrid->getSparseIndex(matrixIndexCoarse);
    this->cf.coarse[index] = sparseIndexCoarse;

    const uint matrixIndexFine = this->cf.fine[index];
    const uint sparseIndexFine = fineGrid->getSparseIndex(matrixIndexFine);
    this->cf.fine[index] = sparseIndexFine;
}

void GridInterface::findForGridInterfaceSparseIndexFC(GridImp* coarseGrid, GridImp* fineGrid, uint index)
{
    const uint matrixIndexCoarse = this->fc.coarse[index];
    const uint sparseIndexCoarse = coarseGrid->getSparseIndex(matrixIndexCoarse);
    this->fc.coarse[index] = sparseIndexCoarse;

    const uint matrixIndexFine = this->fc.fine[index];
    const uint sparseIndexFine = fineGrid->getSparseIndex(matrixIndexFine);
    this->fc.fine[index] = sparseIndexFine;
}

void GridInterface::findInterfaceCF(const uint& indexOnCoarseGrid, GridImp* coarseGrid, GridImp* fineGrid)
{
    const bool nodeOnCoarseGridIsFluid = coarseGrid->getField().isFluid(indexOnCoarseGrid);
    if (!nodeOnCoarseGridIsFluid)
        return;

    const int indexOnFineGridCF = getCoarseToFineIndexOnFineGrid(indexOnCoarseGrid, coarseGrid, fineGrid);
    if (indexOnFineGridCF == -1)
        return;

    const bool fineGridNodeIsFluid = fineGrid->getField().isFluid(indexOnFineGridCF);
    if (!fineGridNodeIsFluid)
        return;

    real x, y, z;
    coarseGrid->transIndexToCoords(indexOnCoarseGrid, x, y, z);

    for(const auto dir : coarseGrid->distribution)
    {
        const bool isFineGridNeighborFluid = isNeighborFineFluid(x + dir[0] * coarseGrid->getDelta(), y + dir[1] * coarseGrid->getDelta(), z + dir[2] * coarseGrid->getDelta(), coarseGrid, fineGrid);
        if(!isFineGridNeighborFluid)
        {
            cf.coarse[cf.numberOfEntries] = indexOnCoarseGrid;
            cf.fine[cf.numberOfEntries] = indexOnFineGridCF;

            cf.numberOfEntries++;

            coarseGrid->setCellTo(indexOnCoarseGrid, FLUID_CFC);
            fineGrid->setCellTo(indexOnFineGridCF, FLUID_CFF);
            break;
        }
    }
}



HOSTDEVICE void GridInterface::findInterfaceFC(const uint& indexOnCoarseGrid, GridImp* coarseGrid, GridImp* fineGrid)
{
    const bool nodeOnCoarseGridIsFluid = coarseGrid->getField().isFluid(indexOnCoarseGrid);
    const bool nodeOnCoarseGridIsCoarseToFine = coarseGrid->getField().isCoarseToFineNode(indexOnCoarseGrid);
    if (!nodeOnCoarseGridIsFluid || nodeOnCoarseGridIsCoarseToFine)
        return;

    const int indexOnFineGridFC = getFineToCoarseIndexOnFineGrid(indexOnCoarseGrid, coarseGrid, fineGrid);
    if (indexOnFineGridFC == -1)
        return;

    const bool fineGridNodeIsFluid = fineGrid->getField().isFluid(indexOnFineGridFC);
    if (!fineGridNodeIsFluid)
        return;

    real x, y, z;
    coarseGrid->transIndexToCoords(indexOnCoarseGrid, x, y, z);

    for (const auto dir : coarseGrid->distribution)
    {
        const int neighborIndex = coarseGrid->transCoordToIndex(x + dir[0] * coarseGrid->getDelta(), y + dir[1] * coarseGrid->getDelta(), z + dir[2] * coarseGrid->getDelta());
        const bool neighborBelongsToCoarseToFineInterpolationCell = coarseGrid->getField().isCoarseToFineNode(neighborIndex);
        if (neighborBelongsToCoarseToFineInterpolationCell)
        {
            fc.coarse[fc.numberOfEntries] = indexOnCoarseGrid;
            fc.fine[fc.numberOfEntries] = indexOnFineGridFC;

            fc.numberOfEntries++;

            fineGrid->setCellTo(indexOnFineGridFC, FLUID_FCF);
            coarseGrid->getField().setFieldEntry(indexOnCoarseGrid, FLUID_FCC);
            break;
        }
    }
}

void GridInterface::findOverlapStopper(const uint& indexOnCoarseGrid, GridImp* coarseGrid, GridImp* fineGrid)
{
    const bool nodeOnCoarseGridIsFluid = coarseGrid->getField().isFluid(indexOnCoarseGrid);
    const bool nodeOnCoarseGridIsCoarseToFine = coarseGrid->getField().isCoarseToFineNode(indexOnCoarseGrid);
    const bool nodeOnCoarseGridIsFineToCoarse = coarseGrid->getField().isFineToCoarseNode(indexOnCoarseGrid);
    if (!nodeOnCoarseGridIsFluid || nodeOnCoarseGridIsCoarseToFine || nodeOnCoarseGridIsFineToCoarse)
        return;

    const int indexOnFineGridFC = getFineToCoarseIndexOnFineGrid(indexOnCoarseGrid, coarseGrid, fineGrid);
    if (indexOnFineGridFC == -1)
        return;

    const bool fineGridNodeIsFluid = fineGrid->getField().isFluid(indexOnFineGridFC);
    if (!fineGridNodeIsFluid)
        return;

    real x, y, z;
    coarseGrid->transIndexToCoords(indexOnCoarseGrid, x, y, z);

    bool neighborBelongsToFineToCoarseInterpolationCell = false;
    for (const auto dir : coarseGrid->distribution)
    {
        //if (dir[0] > 0 || dir[1] > 0 || dir[2] > 0)
        //    continue;

        const int neighborIndex = coarseGrid->transCoordToIndex(x + dir[0] * coarseGrid->getDelta(), y + dir[1] * coarseGrid->getDelta(), z + dir[2] * coarseGrid->getDelta());
        neighborBelongsToFineToCoarseInterpolationCell = coarseGrid->getField().isFineToCoarseNode(neighborIndex);
        if (neighborBelongsToFineToCoarseInterpolationCell)
        {
            coarseGrid->getField().setFieldEntryToStopperOverlapGrid(indexOnCoarseGrid);
            break;
        }

    }
    if(!neighborBelongsToFineToCoarseInterpolationCell) //should be inside of fine grid and can be deleted
        coarseGrid->getField().setFieldEntryToInvalid(indexOnCoarseGrid);
}

bool GridInterface::isNeighborFineFluid(real x, real y, real z, const GridImp* coarseGrid, const GridImp* fineGrid)
{
    const int neighbor = coarseGrid->transCoordToIndex(x, y, z);
    const int indexOnFineGrid = getCoarseToFineIndexOnFineGrid(neighbor, coarseGrid, fineGrid);
    if (indexOnFineGrid == -1)
        return false;
    return fineGrid->getField().isFluid(indexOnFineGrid);
}

int GridInterface::getCoarseToFineIndexOnFineGrid(const uint& indexOnCoarseGrid, const GridImp* coarseGrid, const GridImp* fineGrid)
{
    real x, y, z;
    coarseGrid->transIndexToCoords(indexOnCoarseGrid, x, y, z);
    const real xFine = x + (fineGrid->getDelta() * 0.5);
    const real yFine = y + (fineGrid->getDelta() * 0.5);
    const real zFine = z + (fineGrid->getDelta() * 0.5);

    return fineGrid->transCoordToIndex(xFine, yFine, zFine);
}

int GridInterface::getFineToCoarseIndexOnFineGrid(const uint& indexOnCoarseGrid, const GridImp* coarseGrid, const GridImp* fineGrid)
{
    real x, y, z;
    coarseGrid->transIndexToCoords(indexOnCoarseGrid, x, y, z);
    const real xFine = x - (fineGrid->getDelta() * 0.5);
    const real yFine = y - (fineGrid->getDelta() * 0.5);
    const real zFine = z - (fineGrid->getDelta() * 0.5);

    return fineGrid->transCoordToIndex(xFine, yFine, zFine);
}

void GridInterface::print() const
{
    printf("Grid Interface - CF nodes: %d, FC nodes: %d\n", cf.numberOfEntries, fc.numberOfEntries);
}
