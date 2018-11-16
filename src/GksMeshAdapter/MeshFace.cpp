#include "MeshFace.h"

MeshFace::MeshFace()
{
    for( uint& node : this->faceToNode ) node = INVALID_INDEX;

    isWall = false;
    posCell       = INVALID_INDEX;
    negCell       = INVALID_INDEX;
    posCellCoarse = INVALID_INDEX;
    negCellCoarse = INVALID_INDEX;
}