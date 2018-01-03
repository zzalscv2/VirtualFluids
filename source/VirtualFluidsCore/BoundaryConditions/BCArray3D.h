#ifndef BCArray_H
#define BCArray_H

#include "BoundaryConditions.h"
#include "basics/container/CbArray3D.h"

#include <typeinfo>

#include <boost/serialization/serialization.hpp>
#include <memory>

class BCArray3D;
typedef std::shared_ptr<BCArray3D> BCArray3DPtr;

class BCArray3D
{
public:
   //////////////////////////////////////////////////////////////////////////
   BCArray3D();
   //////////////////////////////////////////////////////////////////////////
   BCArray3D(std::size_t nx1, std::size_t nx2, std::size_t nx3);
   //////////////////////////////////////////////////////////////////////////
   BCArray3D(std::size_t nx1, std::size_t nx2, std::size_t nx3, int val);
   //////////////////////////////////////////////////////////////////////////
   virtual ~BCArray3D();
   //////////////////////////////////////////////////////////////////////////
   inline std::size_t getNX1() const;
   //////////////////////////////////////////////////////////////////////////
   inline std::size_t getNX2() const;
   //////////////////////////////////////////////////////////////////////////
   inline std::size_t getNX3() const;
   //////////////////////////////////////////////////////////////////////////
   void resize(std::size_t nx1, std::size_t nx2, std::size_t nx3);
   //////////////////////////////////////////////////////////////////////////
   void resize(std::size_t nx1, std::size_t nx2, std::size_t nx3, int val);
   //////////////////////////////////////////////////////////////////////////
   bool validIndices(std::size_t x1, std::size_t x2, std::size_t x3)  const;
   //////////////////////////////////////////////////////////////////////////
   inline bool hasBC(std::size_t x1, std::size_t x2, std::size_t x3)  const;
   //////////////////////////////////////////////////////////////////////////
   void setBC(std::size_t x1, std::size_t x2, std::size_t x3, BoundaryConditionsPtr const& bc);
   //////////////////////////////////////////////////////////////////////////
   inline int getBCVectorIndex(std::size_t x1, std::size_t x2, std::size_t x3) const;
   //////////////////////////////////////////////////////////////////////////
   inline const BoundaryConditionsPtr getBC(std::size_t x1, std::size_t x2, std::size_t x3) const;
   //////////////////////////////////////////////////////////////////////////
   inline BoundaryConditionsPtr getBC(std::size_t x1, std::size_t x2, std::size_t x3);
   //////////////////////////////////////////////////////////////////////////
   void setSolid(std::size_t x1, std::size_t x2, std::size_t x3);
   //////////////////////////////////////////////////////////////////////////
   inline bool isSolid(std::size_t x1, std::size_t x2, std::size_t x3) const;
   //////////////////////////////////////////////////////////////////////////
   void setFluid(std::size_t x1, std::size_t x2, std::size_t x3);
   //////////////////////////////////////////////////////////////////////////
   //true : FLUID or BC
   //false: UNDEFINED or SOLID
   inline bool isFluid(std::size_t x1, std::size_t x2, std::size_t x3) const;
   //////////////////////////////////////////////////////////////////////////
   inline bool isFluidWithoutBC(std::size_t x1, std::size_t x2, std::size_t x3) const;
   //////////////////////////////////////////////////////////////////////////
   inline bool isUndefined(std::size_t x1, std::size_t x2, std::size_t x3) const;
   //////////////////////////////////////////////////////////////////////////
   void setUndefined(std::size_t x1, std::size_t x2, std::size_t x3);
   //////////////////////////////////////////////////////////////////////////
   inline bool isInterfaceCF(std::size_t x1, std::size_t x2, std::size_t x3) const;
   //////////////////////////////////////////////////////////////////////////
   void setInterfaceCF(std::size_t x1, std::size_t x2, std::size_t x3);
   //////////////////////////////////////////////////////////////////////////
   inline bool isInterfaceFC(std::size_t x1, std::size_t x2, std::size_t x3) const;
   //////////////////////////////////////////////////////////////////////////
   void setInterfaceFC(std::size_t x1, std::size_t x2, std::size_t x3);
   //////////////////////////////////////////////////////////////////////////
   std::size_t getNumberOfSolidEntries() const;
   //////////////////////////////////////////////////////////////////////////
   std::size_t getNumberOfFluidEntries() const;
   //////////////////////////////////////////////////////////////////////////
   std::size_t getNumberOfFluidWithoutBCEntries() const;
   //////////////////////////////////////////////////////////////////////////
   std::size_t getNumberOfBCEntries() const;
   //////////////////////////////////////////////////////////////////////////
   std::size_t getNumberOfUndefinedEntries() const;
   //////////////////////////////////////////////////////////////////////////
   std::size_t getBCVectorSize() const;
   //////////////////////////////////////////////////////////////////////////
   std::string toString() const;
   //////////////////////////////////////////////////////////////////////////
   std::vector< int >& getBcindexmatrixDataVector();
   //////////////////////////////////////////////////////////////////////////
   bool isInsideOfDomain(const int &x1, const int &x2, const int &x3, const int& ghostLayerWidth) const;

   static const int SOLID;     
   static const int FLUID;     
   static const int INTERFACECF; 
   static const int INTERFACEFC; 
   static const int UNDEFINED; 

private:
   //////////////////////////////////////////////////////////////////////////
   void deleteBCAndSetType(std::size_t x1, std::size_t x2, std::size_t x3, int type);
   //////////////////////////////////////////////////////////////////////////
   void deleteBC(std::size_t x1, std::size_t x2, std::size_t x3);

   friend class MPIIORestart1CoProcessor;
   friend class MPIIORestart2CoProcessor;
   friend class MPIIORestart11CoProcessor;
   friend class MPIIORestart21CoProcessor;

   friend class boost::serialization::access;
   template<class Archive>
   void serialize(Archive & ar, const unsigned int version)
   {
      ar & bcindexmatrix;
      ar & bcvector;
      ar & indexContainer;
   }
protected:
   //////////////////////////////////////////////////////////////////////////
   //-1 solid // -2 fluid -...
   CbArray3D<int, IndexerX3X2X1> bcindexmatrix;
   std::vector<BoundaryConditionsPtr> bcvector;
   std::vector<int> indexContainer;
};


//////////////////////////////////////////////////////////////////////////
inline std::size_t BCArray3D::getNX1() const { return bcindexmatrix.getNX1(); }
//////////////////////////////////////////////////////////////////////////
inline std::size_t BCArray3D::getNX2() const { return bcindexmatrix.getNX2(); }
//////////////////////////////////////////////////////////////////////////
inline std::size_t BCArray3D::getNX3() const { return bcindexmatrix.getNX3(); }
//////////////////////////////////////////////////////////////////////////
inline bool BCArray3D::hasBC(std::size_t x1, std::size_t x2, std::size_t x3)  const
{
   return bcindexmatrix(x1, x2, x3) >= 0;
}
//////////////////////////////////////////////////////////////////////////
inline int BCArray3D::getBCVectorIndex(std::size_t x1, std::size_t x2, std::size_t x3) const
{
   return bcindexmatrix(x1, x2, x3);
}
//////////////////////////////////////////////////////////////////////////
inline const BoundaryConditionsPtr  BCArray3D::getBC(std::size_t x1, std::size_t x2, std::size_t x3) const
{
   int index = bcindexmatrix(x1, x2, x3);
   if (index < 0) return BoundaryConditionsPtr(); //=> NULL Pointer

   return bcvector[index];
}
//////////////////////////////////////////////////////////////////////////
inline BoundaryConditionsPtr BCArray3D::getBC(std::size_t x1, std::size_t x2, std::size_t x3)
{
   int index = bcindexmatrix(x1, x2, x3);
   if (index < 0) return BoundaryConditionsPtr(); //=> NULL Pointer

   return bcvector[index];
}
//////////////////////////////////////////////////////////////////////////
inline bool BCArray3D::isSolid(std::size_t x1, std::size_t x2, std::size_t x3) const
{
   return bcindexmatrix(x1, x2, x3) == SOLID;
}
//////////////////////////////////////////////////////////////////////////
//true : FLUID or BC
//false: UNDEFINED or SOLID
inline bool BCArray3D::isFluid(std::size_t x1, std::size_t x2, std::size_t x3) const
{
   int tmp = bcindexmatrix(x1, x2, x3);
   return (tmp == FLUID || tmp >= 0);
}
//////////////////////////////////////////////////////////////////////////
inline bool BCArray3D::isFluidWithoutBC(std::size_t x1, std::size_t x2, std::size_t x3) const
{
   return bcindexmatrix(x1, x2, x3) == FLUID;
}
//////////////////////////////////////////////////////////////////////////
inline bool BCArray3D::isUndefined(std::size_t x1, std::size_t x2, std::size_t x3) const
{
   return bcindexmatrix(x1, x2, x3) == UNDEFINED;
}
//////////////////////////////////////////////////////////////////////////
inline bool BCArray3D::isInterfaceCF(std::size_t x1, std::size_t x2, std::size_t x3) const
{
   return bcindexmatrix(x1, x2, x3) == INTERFACECF;
}
//////////////////////////////////////////////////////////////////////////
inline bool BCArray3D::isInterfaceFC(std::size_t x1, std::size_t x2, std::size_t x3) const
{
   return bcindexmatrix(x1, x2, x3) == INTERFACEFC;
}
//////////////////////////////////////////////////////////////////////////
inline bool BCArray3D::isInsideOfDomain(const int& x1, const int& x2, const int& x3, const int& ghostLayerWidth) const
{
    const int minX1 = ghostLayerWidth;
    const int maxX1 = (int)this->getNX1() - 1 - ghostLayerWidth;
    const int minX2 = ghostLayerWidth;
    const int maxX2 = (int)this->getNX2() - 1 - ghostLayerWidth;
    const int minX3 = ghostLayerWidth;
    const int maxX3 = (int)this->getNX3() - 1 - ghostLayerWidth;

    return (!(x1 < minX1 || x1 > maxX1 || x2 < minX2 || x2 > maxX2 || x3 < minX3 || x3 > maxX3));
}

#endif 
