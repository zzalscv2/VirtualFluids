#include "BoundaryConditionFactory.h"
#include "GPU/GPU_Interface.h"
#include "Parameter/Parameter.h"
#include "grid/BoundaryConditions/BoundaryCondition.h"
#include <variant>

void BoundaryConditionFactory::setVelocityBoundaryCondition(VelocityBC boundaryConditionType)
{
    this->velocityBoundaryCondition = boundaryConditionType;
}

void BoundaryConditionFactory::setNoSlipBoundaryCondition(const NoSlipBC boundaryConditionType)
{
    this->noSlipBoundaryCondition = boundaryConditionType;
}

void BoundaryConditionFactory::setSlipBoundaryCondition(const SlipBC boundaryConditionType)
{
    this->slipBoundaryCondition = boundaryConditionType;
}

void BoundaryConditionFactory::setPressureBoundaryCondition(const PressureBC boundaryConditionType)
{
    this->pressureBoundaryCondition = boundaryConditionType;
}

void BoundaryConditionFactory::setGeometryBoundaryCondition(
    const std::variant<VelocityBC, NoSlipBC, SlipBC> boundaryConditionType)
{
    this->geometryBoundaryCondition = boundaryConditionType;
}

void BoundaryConditionFactory::setStressBoundaryCondition(const StressBC boundaryConditionType)
{
    this->stressBoundaryCondition = boundaryConditionType;
}

boundaryCondition BoundaryConditionFactory::getVelocityBoundaryConditionPost(bool isGeometryBC) const
{
    const VelocityBC &boundaryCondition =
        isGeometryBC ? std::get<VelocityBC>(this->geometryBoundaryCondition) : this->velocityBoundaryCondition;

    // for descriptions of the boundary conditions refer to the header
    switch (boundaryCondition) {
        case VelocityBC::VelocitySimpleBounceBackCompressible:
            return QVelDevicePlainBB27;
            break;
        case VelocityBC::VelocityIncompressible:
            return QVelDev27;
            break;
        case VelocityBC::VelocityCompressible:
            return QVelDevComp27;
            break;
        case VelocityBC::VelocityAndPressureCompressible:
            return QVelDevCompZeroPress27;
            break;
        default:
            return nullptr;
    }
}

boundaryCondition BoundaryConditionFactory::getNoSlipBoundaryConditionPost(bool isGeometryBC) const
{
    const NoSlipBC &boundaryCondition =
        isGeometryBC ? std::get<NoSlipBC>(this->geometryBoundaryCondition) : this->noSlipBoundaryCondition;

    // for descriptions of the boundary conditions refer to the header
    switch (boundaryCondition) {
        case NoSlipBC::NoSlipImplicitBounceBack:
            return [](LBMSimulationParameter *, QforBoundaryConditions *) {};
            break;
        case NoSlipBC::NoSlipBounceBack:
            return BBDev27;
            break;
        case NoSlipBC::NoSlipIncompressible:
            return QDev27;
            break;
        case NoSlipBC::NoSlipCompressible:
            return QDevComp27;
            break;
        case NoSlipBC::NoSlip3rdMomentsCompressible:
            return QDev3rdMomentsComp27;
            break;
        default:
            return nullptr;
    }
}

boundaryCondition BoundaryConditionFactory::getSlipBoundaryConditionPost(bool isGeometryBC) const
{
    const SlipBC &boundaryCondition =
        isGeometryBC ? std::get<SlipBC>(this->geometryBoundaryCondition) : this->slipBoundaryCondition;

    // for descriptions of the boundary conditions refer to the header
    switch (boundaryCondition) {
        case SlipBC::SlipIncompressible:
            return QSlipDev27;
            break;
        case SlipBC::SlipCompressible:
            return QSlipDevComp27;
            break;
        case SlipBC::SlipCompressibleTurbulentViscosity:
            return QSlipDevCompTurbulentViscosity27;
            break;
        default:
            return nullptr;
    }
}

boundaryCondition BoundaryConditionFactory::getPressureBoundaryConditionPre() const
{
    // for descriptions of the boundary conditions refer to the header
    switch (this->pressureBoundaryCondition) {
        case PressureBC::PressureEquilibrium:
            return QPressDev27;
            break;
        case PressureBC::PressureEquilibrium2:
            return QPressDevEQZ27;
            break;
        case PressureBC::PressureNonEquilibriumIncompressible:
            return QPressDevIncompNEQ27;
            break;
        case PressureBC::PressureNonEquilibriumCompressible:
            return QPressDevNEQ27;
            break;
        case PressureBC::OutflowNonReflective:
            return QPressNoRhoDev27;
            break;
        default:
            return nullptr;
    }
}

boundaryConditionPara BoundaryConditionFactory::getStressBoundaryConditionPost() const
{
    switch (this->stressBoundaryCondition) {
        case StressBC::StressBounceBack:
            return BBStressDev27;
            break;
        case StressBC::StressCompressible:
            return QStressDevComp27;
            break;
        default:
            return nullptr;
    }
}

boundaryCondition BoundaryConditionFactory::getGeometryBoundaryConditionPost() const
{
    if (std::holds_alternative<VelocityBC>(this->geometryBoundaryCondition))
        return this->getVelocityBoundaryConditionPost(true);
    else if (std::holds_alternative<NoSlipBC>(this->geometryBoundaryCondition))
        return this->getNoSlipBoundaryConditionPost(true);
    else if (std::holds_alternative<SlipBC>(this->geometryBoundaryCondition))
        return this->getSlipBoundaryConditionPost(true);
    return nullptr;
}