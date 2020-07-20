#ifndef Geometry_h
#define Geometry_h

#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>

#include "global.h"

#include "geometries/Triangle/Triangle.h"
#include "geometries/BoundingBox/BoundingBox.h"

#include "geometries/Object.h"

class GeometryMemento;
class GbTriFaceMesh3D;

enum class DiscretizationMethod { RAYCASTING, POINT_IN_OBJECT, POINT_UNDER_TRIANGLE };


class TriangularMesh : public Object
{
public:

    VIRTUALFLUIDS_GPU_EXPORT static TriangularMesh* make(const std::string& fileName, const std::vector<uint> ignorePatches = std::vector<uint>());
	VIRTUALFLUIDS_GPU_EXPORT TriangularMesh();
    VIRTUALFLUIDS_GPU_EXPORT TriangularMesh(const std::string& inputPath, const std::vector<uint> ignorePatches = std::vector<uint>());
	VIRTUALFLUIDS_GPU_EXPORT TriangularMesh(const std::string& inputPath, const BoundingBox &box);
	HOSTDEVICE VIRTUALFLUIDS_GPU_EXPORT ~TriangularMesh();

    VIRTUALFLUIDS_GPU_EXPORT uint getNumberOfTriangles() const;

	VIRTUALFLUIDS_GPU_EXPORT void setTriangles(std::vector<Triangle> triangles);
	VIRTUALFLUIDS_GPU_EXPORT void setMinMax(BoundingBox minmax);

	std::vector<Triangle> triangleVec;
	Triangle *triangles;
	long size;
	BoundingBox minmax;

    SPtr<GbTriFaceMesh3D> VF_GbTriFaceMesh3D;

    CUDA_HOST VIRTUALFLUIDS_GPU_EXPORT bool operator==(const TriangularMesh &geometry) const;

    VIRTUALFLUIDS_GPU_EXPORT void findNeighbors();

    HOSTDEVICE VIRTUALFLUIDS_GPU_EXPORT GbTriFaceMesh3D* getGbTriFaceMesh3D() const;

    CUDA_HOST VIRTUALFLUIDS_GPU_EXPORT void generateGbTriFaceMesh3D();

private:
	
	void initalizeDataFromTriangles();

    static std::vector<Vertex> getAverrageNormalsPerVertex(std::vector<std::vector<Triangle> > trianglesPerVertex);
    static void eliminateTriangleswithIdenticialNormal(std::vector<Triangle> &triangles);

public:
    HOSTDEVICE Object* clone() const override;
    double getX1Centroid() override { throw "Not implemented in TriangularMesh"; }
    double getX1Minimum() override { return minmax.minX; }
    double getX1Maximum() override { return minmax.maxX; }
    double getX2Centroid() override { throw "Not implemented in TriangularMesh"; }
    double getX2Minimum() override { return minmax.minY; }
    double getX2Maximum() override { return minmax.maxY; }
    double getX3Centroid() override { throw "Not implemented in TriangularMesh"; }
    double getX3Minimum() override { return minmax.minZ; }
    double getX3Maximum() override { return minmax.maxZ; }
    void scale(double delta) override;
    HOSTDEVICE bool isPointInObject(const double& x1, const double& x2, const double& x3, const double& minOffset,
        const double& maxOffset) override {
        return false;
    }
    
    void findInnerNodes(SPtr<GridImp> grid) override;
};



#endif

