#ifndef ArrowTransformator_h
#define ArrowTransformator_h

#include <memory>

#include "global.h"

class Arrow;

class ArrowTransformator
{
public:
    static VIRTUALFLUIDS_GPU_EXPORT std::shared_ptr<ArrowTransformator> makeTransformator(real delta, real dx, real dy, real dz);
	virtual ~ArrowTransformator() {}

protected:
	ArrowTransformator() {}
	
public:
	virtual void transformGridToWorld(std::shared_ptr<Arrow> arrow) const = 0;
};


#endif