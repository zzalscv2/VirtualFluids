#ifndef D3Q19AMRTRIFACEMESHINTERACTOR_H
#define D3Q19AMRTRIFACEMESHINTERACTOR_H

#include <PointerDefinitions.h>
#include <map>
#include <string>
#include <vector>

#include "CbArray3D.h"
#include "D3Q27Interactor.h"

class GbObject3D;
class Grid3D;
class BC;
class GbTriFaceMesh3D;
class Block3D;

class D3Q27TriFaceMeshInteractor : public D3Q27Interactor
{
public:
    static const int STRESSNORMAL     = 0;
    static const int STRESSALTERNATIV = 1;

    D3Q27TriFaceMeshInteractor();
    D3Q27TriFaceMeshInteractor(SPtr<Grid3D> grid, std::string name = "D3Q27TriFaceMeshInteractor");
    D3Q27TriFaceMeshInteractor(SPtr<GbObject3D> geoObject3D, SPtr<Grid3D> grid, int type);
    D3Q27TriFaceMeshInteractor(SPtr<GbTriFaceMesh3D> triFaceMesh, SPtr<Grid3D> grid, SPtr<BC> BC,
                               int type);
    D3Q27TriFaceMeshInteractor(SPtr<GbTriFaceMesh3D> triFaceMesh, SPtr<Grid3D> grid, SPtr<BC> BC,
                               int type, Interactor3D::Accuracy a);
    // D3Q27TriFaceMeshInteractor(SPtr<GbTriFaceMesh3D> triFaceMesh, D3Q27BoundaryConditionAdapterPtr BC, int
    // type, std::string name="D3Q27TriFaceMeshInteractor");

    ~D3Q27TriFaceMeshInteractor() override;

    void initInteractor(const real &timeStep = 0) override;
    virtual void initInteractor2(const real &timeStep = 0);

    void updateInteractor(const real &timestep = 0) override;

    void updateMovedGeometry(const real &timeStep = 0);
    void setQs(const real &timeStep);
    void refineBlockGridToLevel(int level, real startDistance, real stopDistance);

    bool setDifferencesToGbObject3D(const SPtr<Block3D> block) override;

    void setRegardPointInObjectTest(bool opt) { this->regardPIOTest = opt; }

    ObObject *clone() { throw UbException(UB_EXARGS, "not implemented"); }

    UbTupleDouble3 getForces() override;
    UbTupleDouble3 getForcesTriangle();

    void setStressMode(int stressMode) { this->stressMode = stressMode; }
    void setUseHalfSpaceCheck(bool useHalfSpace) { this->useHalfSpace = useHalfSpace; }
    // void setReinitWithStoredQs(bool reinitWithStoredQsFlag) { this->reinitWithStoredQsFlag = reinitWithStoredQsFlag;
    // }

    void calculateForces();
    void calculateStresses();
    void calculateStressesAlternativ();

    void calcStressesLine(UbTupleDouble6 &stresses, const real &weight, const UbTupleDouble6 &stvW,
                          const UbTupleDouble6 &stvE);
    void calcStressesFace(UbTupleDouble6 &stresses, const real &weightX, const real &weightY,
                          const UbTupleDouble6 &stvSW, const UbTupleDouble6 &stvSE, const UbTupleDouble6 &stvNE,
                          const UbTupleDouble6 &stvNW);
    void calcStressesCube(UbTupleDouble6 &stresses, const real &weightX, const real &weightY, const real &weightZ,
                          const UbTupleDouble6 &stvBSW, const UbTupleDouble6 &stvBSE, const UbTupleDouble6 &stvBNE,
                          const UbTupleDouble6 &stvBNW, const UbTupleDouble6 &stvTSW, const UbTupleDouble6 &stvTSE,
                          const UbTupleDouble6 &stvTNE, const UbTupleDouble6 &stvTNW);

    void calculatePressure();
    void calcPressureLine(real &p, const real &weight, const real &pW, const real &pE);
    void calcPressureFace(real &p, const real &weightX, const real &weightY, const real &pSW, const real &pSE,
                          const real &pNE, const real &pNW);
    void calcPressureCube(real &p, const real &weightX, const real &weightY, const real &weightZ,
                          const real &pBSW, const real &pBSE, const real &pBNE, const real &pBNW,
                          const real &pTSW, const real &pTSE, const real &pTNE, const real &pTNW);

    void setForceShift(real forceshift)
    {
        this->forceshift       = forceshift;
        this->forceshiftpolicy = true;
    }
    void setVelocityShift(real velocityshift)
    {
        this->velocityshift       = velocityshift;
        this->velocityshiftpolicy = true;
    }
    real getForceShift() { return this->forceshift; }
    real getVelocityShift() { return this->velocityshift; }
    bool getForceShiftPolicy() { return forceshiftpolicy; }
    bool getVelocityShiftPolicy() { return velocityshiftpolicy; }

    void clearBcNodeIndicesAndQsMap() { this->bcNodeIndicesAndQsMap.clear(); }

    virtual std::string toString();

protected:
    int stressMode;

    double forceshift{ 0.0 };
    double velocityshift{ 0.0 };
    bool forceshiftpolicy{ false };
    bool velocityshiftpolicy{ false };
    bool useHalfSpace{ true };
    bool regardPIOTest{ true };

    void reinitWithStoredQs(const real &timeStep);
    //   bool reinitWithStoredQsFlag;
    std::map<SPtr<Block3D>, std::map<UbTupleInt3, std::vector<float>>>
        bcNodeIndicesAndQsMap; //!!! es kann sein, dass in diesem interactor
    // an eine rpos eine BC gesetzt wurde, aber derselbe node in
    // in einem anderen in einen anderen Typ (z.B. Solid) geaendert
    // wurde --> es ist keine BC mehr an der stelle!

    enum SolidCheckMethod { ScanLine, PointInObject };

    enum FLAGS { BC_FLAG, UNDEF_FLAG, FLUID_FLAG, SOLID_FLAG, OLDSOLID_FLAG };
    void recursiveGridFill(CbArray3D<FLAGS> &flagfield, const short &xs, const short &ys, const short &zs,
                           const FLAGS &type);
    void iterativeGridFill(CbArray3D<FLAGS> &flagfield, const short &xs, const short &ys, const short &zs,
                           const FLAGS &type);
};

#endif