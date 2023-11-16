#ifndef RatioBlockVisitor_H
#define RatioBlockVisitor_H

#include <PointerDefinitions.h>
#include <string>

#include "Block3DVisitor.h"

class Grid3D;
class Block3D;

class RatioBlockVisitor : public Block3DVisitor
{
public:
    RatioBlockVisitor(int levelDepth, bool includeNotActiveBlocks = false);

    ~RatioBlockVisitor() override = default;

    bool expandsByAdaptation() { return this->expandBlocks; }

    void setExpandByAdaptation(bool expandBlocks);

    int getLevelRatio() { return this->maxLevelRatio; }
    bool isIterative() { return true; }

    void setLevelRatio(int ratio);

    int getStartLevel();
    int getStopLevel();

    void setStartLevel(int level);
    void setStopLevel(int level);

    std::string getSpecificDescription();

    void visit(SPtr<Grid3D> grid, SPtr<Block3D> block) override;

protected:
    bool lookForExpand(SPtr<Grid3D> grid, const int &ix1, const int &ix2, const int &ix3, const int &level);
    bool lookForCollapse(SPtr<Grid3D> grid, const int &ix1, const int &ix2, const int &ix3, const int &level);

private:
    int maxLevelRatio;
    bool expandBlocks;
    int levelDepth;
    bool includeNotActiveBlocks;
};

#endif