//  _    ___      __              __________      _     __
// | |  / (_)____/ /___  ______ _/ / ____/ /_  __(_)___/ /____
// | | / / / ___/ __/ / / / __ `/ / /_  / / / / / / __  / ___/
// | |/ / / /  / /_/ /_/ / /_/ / / __/ / / /_/ / / /_/ (__  )
// |___/_/_/   \__/\__,_/\__,_/_/_/   /_/\__,_/_/\__,_/____/
//
#ifndef BCAdapter_H
#define BCAdapter_H

#include <PointerDefinitions.h>

#include "BoundaryConditions.h"
#include "basics/objects/ObObject.h"
#include "basics/objects/ObObjectCreator.h"
#include "basics/utilities/UbFileOutput.h"
#include "basics/utilities/UbFileInput.h"
#include "basics/utilities/UbAutoRun.hpp"
#include "BCAlgorithm.h"


/*=========================================================================*/
/*  D3Q27BoundaryConditionAdapter                                          */
/*                                                                         */
/**
<BR><BR>
@author <A HREF="mailto:muffmolch@gmx.de">S. Freudiger</A>
@version 1.0 - 06.09.06
*/ 

/*
usage: ...
*/

class D3Q27Interactor;

class BCAdapter
{
public:
   BCAdapter() 
      :  secondaryBcOption(0)
       , type(0)
       , algorithmType(-1)
   {
   }
   BCAdapter(const short& secondaryBcOption) 
      :  secondaryBcOption(secondaryBcOption) 
       , type(0)
       , algorithmType(-1)
   {
   }
   virtual ~BCAdapter() {}

   //methods
   bool isTimeDependent() { return((this->type & TIMEDEPENDENT) ==  TIMEDEPENDENT); }

   virtual short getSecondaryBcOption() { return this->secondaryBcOption; }
   virtual void  setSecondaryBcOption(const short& val) { this->secondaryBcOption=val; }

   virtual void init(const D3Q27Interactor* const& interactor, const double& time=0) = 0;
   virtual void update(const D3Q27Interactor* const& interactor, const double& time=0) = 0;

   virtual void adaptBC( const D3Q27Interactor& interactor, SPtr<BoundaryConditions> bc, const double& worldX1, const double& worldX2, const double& worldX3, const double& time=0 ) = 0;
   virtual void adaptBCForDirection( const D3Q27Interactor& interactor, SPtr<BoundaryConditions> bc, const double& worldX1, const double& worldX2, const double& worldX3, const double& q, const int& fdirection, const double& time=0 ) = 0;

   void setBcAlgorithm(SPtr<BCAlgorithm> alg) {algorithmType = alg->getType(); algorithm = alg;}
   SPtr<BCAlgorithm> getAlgorithm() {return algorithm;} 
   char getBcAlgorithmType() {return algorithmType;}

protected:
   short secondaryBcOption;

   char  type;

   SPtr<BCAlgorithm> algorithm;
   char algorithmType;

   static const char   TIMEDEPENDENT = 1<<0;//'1';
   static const char   TIMEPERIODIC  = 1<<1;//'2';

private:

};


#endif //D3Q27BOUNDARYCONDITIONADAPTER_H