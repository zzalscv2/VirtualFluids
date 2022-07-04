#ifndef BC_FACTORY
#define BC_FACTORY

#include <functional>
#include <map>
#include <string>

#include "LBM/LB.h"

struct LBMSimulationParameter;

using boundaryCondition = std::function<void(LBMSimulationParameter *, QforBoundaryConditions *)>;

class BoundaryConditionFactory
{
public:
    //! \brief An enumeration for selecting a velocity boundary condition
    enum class VelocityBC {
        //! - VelocitySimpleBounceBackCompressible = plain bounce back velocity boundary condition
        VelocitySimpleBounceBackCompressible,
        //! - VelocityIncompressible = interpolated velocity boundary condition, based on subgrid distances
        VelocityIncompressible,
        //! - VelocityCompressible = interpolated velocity boundary condition, based on subgrid distances
        VelocityCompressible,
        //! - VelocityAndPressureCompressible = interpolated velocity boundary condition, based on subgrid distances.
        //! Also sets the pressure to the bulk pressure.
        VelocityAndPressureCompressible
    };

    //! \brief An enumeration for selecting a no-slip boundary condition
    enum class NoSlipBC {
        //! - NoSlipBounceBack = bounce back no-slip boundary condition
        NoSlipBounceBack,
        //! - NoSlipIncompressible = interpolated no-slip boundary condition, based on subgrid distances
        NoSlipIncompressible,
        //! - NoSlipCompressible = interpolated no-slip boundary condition, based on subgrid distances
        NoSlipCompressible
    };

    //! \brief An enumeration for selecting a slip boundary condition
    enum class SlipBC {
        //! - SlipIncompressible = interpolated slip boundary condition, based on subgrid distances
        SlipIncompressible,
        //! - SlipCompressible = interpolated slip boundary condition, based on subgrid distances
        SlipCompressible,
        //! - SlipCompressible = interpolated slip boundary condition, based on subgrid distances.
        //! With turbulent viscosity -> para->setUseTurbulentViscosity(true) has to be set to true
        SlipCompressibleTurbulentViscosity
    };

    //! \brief An enumeration for selecting a pressure boundary condition
    enum class PressureBC {
        //! - PressureEquilibrium = pressure boundary condition based on equilibrium
        PressureEquilibrium, // incorrect pressure :(
        //! - PressureEquilibrium2 = pressure boundary condition based on equilibrium (potentially better?! than PressureEquilibrium)
        PressureEquilibrium2, // is broken --> nan :(
        //! - PressureNonEquilibriumIncompressible = pressure boundary condition based on non-equilibrium
        PressureNonEquilibriumIncompressible,
        //! - PressureNonEquilibriumCompressible = pressure boundary condition based on non-equilibrium
        PressureNonEquilibriumCompressible,
        //! - OutflowNonReflective = outflow boundary condition
        OutflowNonReflective
    };

    // enum class OutflowBoundaryCondition {};  // TODO:
    // https://git.rz.tu-bs.de/m.schoenherr/VirtualFluids_dev/-/issues/16

    void setVelocityBoundaryCondition(const VelocityBC boundaryConditionType);
    void setNoSlipBoundaryCondition(const NoSlipBC boundaryConditionType);
    void setSlipBoundaryCondition(const SlipBC boundaryConditionType);
    void setPressureBoundaryCondition(const PressureBC boundaryConditionType);
    // void setGeometryBoundaryCondition(const std::variant<VelocityBC, NoSlipBC, SlipBC> boundaryConditionType);

    // void setOutflowBoundaryCondition(...); // TODO:
    // https://git.rz.tu-bs.de/m.schoenherr/VirtualFluids_dev/-/issues/16

    boundaryCondition getVelocityBoundaryConditionPost() const;
    boundaryCondition getNoSlipBoundaryConditionPost() const;
    boundaryCondition getSlipBoundaryConditionPost() const;
    boundaryCondition getPressureBoundaryConditionPre() const;

private:
    VelocityBC velocityBoundaryCondition;
    NoSlipBC noSlipBoundaryCondition;
    SlipBC slipBoundaryCondition;
    PressureBC pressureBoundaryCondition;

    // OutflowBoundaryConditon outflowBC // TODO: https://git.rz.tu-bs.de/m.schoenherr/VirtualFluids_dev/-/issues/16
};

#endif
