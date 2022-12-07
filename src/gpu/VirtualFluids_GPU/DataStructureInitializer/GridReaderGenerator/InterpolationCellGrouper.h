//! \file InterpolationCellGrouper.h
//! \ingroup GPU
//! \author Anna Wellmann
//! \details See [master thesis of Anna Wellmann]

#ifndef InterpolationCellGrouper_H
#define InterpolationCellGrouper_H

#include <basics/Core/DataTypes.h>
#include <basics/PointerDefinitions.h>
#include <memory>
#include <vector>

struct LBMSimulationParameter;
class GridBuilder;

using LBMSimulationParameters = std::vector<std::shared_ptr<LBMSimulationParameter>>;

class InterpolationCellGrouper {
public:
    //! \brief Construct InterpolationCellGrouper object
    InterpolationCellGrouper(const LBMSimulationParameters &parHs, const LBMSimulationParameters &parDs,
                             SPtr<GridBuilder> builder);

    //////////////////////////////////////////////////////////////////////////
    // split interpolation cells
    //////////////////////////////////////////////////////////////////////////

    //! \brief Split the interpolation cells from coarse to fine into border an bulk
    //! \details For communication hiding, the interpolation cells from the coarse to the fine grid need to be split
    //! into two groups:
    //!
    //! - cells which are at the border between two gpus --> "border"
    //!
    //! - the other cells which are not directly related to the communication between the two gpus --> "bulk"
    //!
    //! see [master thesis of Anna Wellmann (p. 62-68: "Überdeckung der reduzierten Kommunikation")]
    void splitCoarseToFineIntoBorderAndBulk(uint level) const;

    //! \brief Split the interpolation cells from fine to coarse into border an bulk
    //! \details For communication hiding, the interpolation cells from the fine to the coarse grid need to be split
    //! into two groups:
    //!
    //! - cells which are at the border between two gpus --> "border"
    //!
    //! - the other cells which are not directly related to the communication between the two gpus --> "bulk"
    //!
    //! See [master thesis of Anna Wellmann (p. 62-68: "Überdeckung der reduzierten Kommunikation")]
    void splitFineToCoarseIntoBorderAndBulk(uint level) const;

protected:
    //////////////////////////////////////////////////////////////////////////
    // split interpolation cells
    //////////////////////////////////////////////////////////////////////////

    //! \brief This function reorders the arrays of CFC/CFF indices and sets the pointers and sizes of the new
    //! subarrays: \details The coarse cells for interpolation from coarse to fine (iCellCFC) are divided into two
    //! subgroups: border and bulk. The fine cells (iCellCFF) are reordered accordingly. The offset cells (xOffCF,
    //! yOffCF, zOffCF) must be reordered in the same way.
    void reorderCoarseToFineIntoBorderAndBulk(uint level) const;

    //! \brief This function reorders the arrays of FCC/FCF indices and return pointers and sizes of the new subarrays:
    //! \details The coarse cells for interpolation from fine to coarse (iCellFCC) are divided into two subgroups:
    //! border and bulk. The fine cells (iCellFCF) are reordered accordingly. The offset cells (xOffFC,
    //! yOffFC, zOffFC) must be reordered in the same way.
    void reorderFineToCoarseIntoBorderAndBulk(uint level) const;

private:
    SPtr<GridBuilder> builder;
    const LBMSimulationParameters &parHs;
    const LBMSimulationParameters &parDs;
};

#endif
