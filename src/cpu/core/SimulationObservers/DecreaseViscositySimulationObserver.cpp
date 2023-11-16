/*
 *  DecreaseViscositySimulationObserver
 *
 *  Created on: 10.05.2013
 *  Author: uphoff
 */

#include "DecreaseViscositySimulationObserver.h"

#include <vector>

#include "Block3D.h"
#include <parallel/Communicator.h>
#include "Grid3D.h"
#include "LBMKernel.h"
#include "UbScheduler.h"

DecreaseViscositySimulationObserver::DecreaseViscositySimulationObserver(SPtr<Grid3D> grid, SPtr<UbScheduler> s, mu::Parser *nueFunc,
                                                           std::shared_ptr<vf::parallel::Communicator> comm)

    : SimulationObserver(grid, s), nueFunc(nueFunc), comm(comm)
{
    if (comm->getProcessID() == comm->getRoot()) {
    }
}
//////////////////////////////////////////////////////////////////////////
DecreaseViscositySimulationObserver::~DecreaseViscositySimulationObserver() = default;
//////////////////////////////////////////////////////////////////////////
void DecreaseViscositySimulationObserver::update(real step)
{
    if (scheduler->isDue(step))
        setViscosity(step);
}
//////////////////////////////////////////////////////////////////////////
void DecreaseViscositySimulationObserver::setViscosity(real step)
{

    UBLOG(logDEBUG3, "DecreaseViscositySimulationObserver::update:" << step);
    int gridRank     = grid->getRank();
    int minInitLevel = this->grid->getCoarsestInitializedLevel();
    int maxInitLevel = this->grid->getFinestInitializedLevel();

    if (comm->getProcessID() == comm->getRoot()) {

        for (int level = minInitLevel; level <= maxInitLevel; level++) {
            std::vector<SPtr<Block3D>> blockVector;
            grid->getBlocks(level, gridRank, blockVector);
            for (SPtr<Block3D> block : blockVector) {
                SPtr<ILBMKernel> kernel = block->getKernel();
            }
        }

        int istep      = static_cast<int>(step);
        this->timeStep = istep;
        nueFunc->DefineVar("t", &this->timeStep);
        real nue = nueFunc->Eval();

        for (int level = minInitLevel; level <= maxInitLevel; level++) {
            std::vector<SPtr<Block3D>> blockVector;
            grid->getBlocks(level, gridRank, blockVector);
            for (SPtr<Block3D> block : blockVector) {
                SPtr<ILBMKernel> kernel = block->getKernel();
                if (kernel) {
                    real collFactor = LBMSystem::calcCollisionFactor(nue, block->getLevel());
                    kernel->setCollisionFactor(collFactor);
                }
            }
        }
    }
}