#ifndef LEVEL_GRID_BUILDER_H
#define LEVEL_GRID_BUILDER_H

#include <vector>
#include <string>
#include <memory>
#include <array>

#include "global.h"

#include "grid/GridBuilder/GridBuilder.h"
#include "grid/Grid.h"
#include "grid/GridInterface.h"
#include "grid/NodeValues.h"

struct Vertex;
class  Grid;
class Transformator;
class ArrowTransformator;
class PolyDataWriterWrapper;
class BoundingBox;
enum class Device;

class Side;
class VelocityBoundaryCondition;
class PressureBoundaryCondition;
class GeometryBoundaryCondition;
enum class SideType;



class LevelGridBuilder : public GridBuilder
{
protected:
    VIRTUALFLUIDS_GPU_EXPORT LevelGridBuilder(Device device, const std::string& d3qxx);

public:
    VIRTUALFLUIDS_GPU_EXPORT static std::shared_ptr<LevelGridBuilder> makeShared(Device device, const std::string& d3qxx);

    VIRTUALFLUIDS_GPU_EXPORT SPtr<Grid> getGrid(uint level) override;

    VIRTUALFLUIDS_GPU_EXPORT void copyDataFromGpu();
    VIRTUALFLUIDS_GPU_EXPORT virtual ~LevelGridBuilder();

    VIRTUALFLUIDS_GPU_EXPORT void setVelocityBoundaryCondition(SideType sideType, real vx, real vy, real vz);
    VIRTUALFLUIDS_GPU_EXPORT void setPressureBoundaryCondition(SideType sideType, real rho);
    VIRTUALFLUIDS_GPU_EXPORT void setPeriodicBoundaryCondition(bool periodic_X, bool periodic_Y, bool periodic_Z);
    VIRTUALFLUIDS_GPU_EXPORT void setNoSlipBoundaryCondition(SideType sideType);

    VIRTUALFLUIDS_GPU_EXPORT void setEnableFixRefinementIntoTheWall( bool enableFixRefinementIntoTheWall );

    VIRTUALFLUIDS_GPU_EXPORT void setCommunicationProcess(int direction, uint process);

    VIRTUALFLUIDS_GPU_EXPORT uint getCommunicationProcess(int direction) override;

    VIRTUALFLUIDS_GPU_EXPORT virtual std::shared_ptr<Grid> getGrid(int level, int box);


    VIRTUALFLUIDS_GPU_EXPORT virtual unsigned int getNumberOfNodes(unsigned int level) const;


    VIRTUALFLUIDS_GPU_EXPORT virtual void getNodeValues(real *xCoords, real *yCoords, real *zCoords,
                                         uint *neighborX, uint *neighborY, uint *neighborZ, uint *neighborNegative, 
                                         uint *geo, const int level) const override;
    VIRTUALFLUIDS_GPU_EXPORT virtual void getDimensions(int &nx, int &ny, int &nz, const int level) const;


    VIRTUALFLUIDS_GPU_EXPORT uint getVelocitySize(int level) const;
    VIRTUALFLUIDS_GPU_EXPORT virtual void getVelocityValues(real* vx, real* vy, real* vz, int* indices, int level) const;
    VIRTUALFLUIDS_GPU_EXPORT virtual void getVelocityQs(real* qs[27], int level) const;
    VIRTUALFLUIDS_GPU_EXPORT uint getPressureSize(int level) const override;
    VIRTUALFLUIDS_GPU_EXPORT void getPressureValues(real* rho, int* indices, int* neighborIndices, int level) const override;
    VIRTUALFLUIDS_GPU_EXPORT virtual void getPressureQs(real* qs[27], int level) const;

    VIRTUALFLUIDS_GPU_EXPORT virtual void getGeometryQs(real* qs[27], int level) const;
    VIRTUALFLUIDS_GPU_EXPORT virtual uint getGeometrySize(int level) const;
    VIRTUALFLUIDS_GPU_EXPORT virtual void getGeometryIndices(int* indices, int level) const;
    VIRTUALFLUIDS_GPU_EXPORT virtual bool hasGeometryValues() const;
    VIRTUALFLUIDS_GPU_EXPORT virtual void getGeometryValues(real* vx, real* vy, real* vz, int level) const;


    VIRTUALFLUIDS_GPU_EXPORT void writeArrows(std::string fileName) const;

    VIRTUALFLUIDS_GPU_EXPORT SPtr<BoundaryCondition> getBoundaryCondition( SideType side, uint level ) const override;
    VIRTUALFLUIDS_GPU_EXPORT SPtr<GeometryBoundaryCondition> getGeometryBoundaryCondition(uint level) const override;

protected:
    

    struct BoundaryConditions
    {
		BoundaryConditions() : geometryBoundaryCondition(nullptr) {}

        std::vector<SPtr<VelocityBoundaryCondition> > velocityBoundaryConditions;
        std::vector<SPtr<PressureBoundaryCondition> > pressureBoundaryConditions;

		//TODO: add slip BC



        std::vector<SPtr<VelocityBoundaryCondition> > noSlipBoundaryConditions;

        SPtr<GeometryBoundaryCondition> geometryBoundaryCondition;
    };
    bool geometryHasValues = false;

    std::vector<std::shared_ptr<Grid> > grids;
    std::vector<SPtr<BoundaryConditions> > boundaryConditions;

    std::array<uint, 6> communicationProcesses;

    void checkLevel(int level);

protected:
    void setVelocityGeometryBoundaryCondition(real vx, real vy, real vz);

    void createBCVectors();
    void addShortQsToVector(int index);
    void addQsToVector(int index);
    void fillRBForNode(int index, int direction, int directionSign, int rb);

    Vertex getVertex(const int matrixIndex) const;

private:
    Device device;
    std::string d3qxx;

public:
    VIRTUALFLUIDS_GPU_EXPORT void getGridInformations(std::vector<int>& gridX, std::vector<int>& gridY,
                                       std::vector<int>& gridZ, std::vector<int>& distX, std::vector<int>& distY,
                                       std::vector<int>& distZ) override;
    VIRTUALFLUIDS_GPU_EXPORT uint getNumberOfGridLevels() const override;

    VIRTUALFLUIDS_GPU_EXPORT uint getNumberOfNodesCF(int level) override;
    VIRTUALFLUIDS_GPU_EXPORT uint getNumberOfNodesFC(int level) override;

    VIRTUALFLUIDS_GPU_EXPORT void getGridInterfaceIndices(uint* iCellCfc, uint* iCellCff, uint* iCellFcc, uint* iCellFcf, int level) const override;

    VIRTUALFLUIDS_GPU_EXPORT void getOffsetFC(real* xOffCf, real* yOffCf, real* zOffCf, int level) override;
    VIRTUALFLUIDS_GPU_EXPORT void getOffsetCF(real* xOffFc, real* yOffFc, real* zOffFc, int level) override;

    VIRTUALFLUIDS_GPU_EXPORT uint getNumberOfSendIndices( int direction, uint level ) override;
    VIRTUALFLUIDS_GPU_EXPORT uint getNumberOfReceiveIndices( int direction, uint level ) override;
    VIRTUALFLUIDS_GPU_EXPORT void getSendIndices( int* sendIndices, int direction, int level ) override;
    VIRTUALFLUIDS_GPU_EXPORT void getReceiveIndices( int* sendIndices, int direction, int level ) override;

};

#endif
