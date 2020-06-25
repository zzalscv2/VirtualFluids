#include <numerics/geometry3d/GbVector3D.h>

#include <basics/utilities/UbMath.h>
#include <basics/utilities/UbInfinity.h>
#include <numerics/geometry3d/GbPoint3D.h>
                                
using namespace std;


const GbVector3D GbVector3D::ZERO(0.0,0.0,0.0);
const GbVector3D GbVector3D::UNIT_X1(1.0,0.0,0.0);
const GbVector3D GbVector3D::UNIT_X2(0.0,1.0,0.0);
const GbVector3D GbVector3D::UNIT_X3(0.0,0.0,1.0);

//----------------------------------------------------------------------------
GbVector3D::GbVector3D () 
{                                      
   m_afTuple[0] = 0.0;
   m_afTuple[1] = 0.0;
   m_afTuple[2] = 0.0;
}
//----------------------------------------------------------------------------
GbVector3D::GbVector3D (const double& fX, const double& fY, const double& fZ) 
{
   m_afTuple[0] = fX;
   m_afTuple[1] = fY;
   m_afTuple[2] = fZ;
}
//----------------------------------------------------------------------------
GbVector3D::GbVector3D (const GbVector3D& rkV) 
{
   m_afTuple[0] = rkV.m_afTuple[0];
   m_afTuple[1] = rkV.m_afTuple[1];
   m_afTuple[2] = rkV.m_afTuple[2];
}
//----------------------------------------------------------------------------

GbVector3D::GbVector3D (const GbPoint3D& point) 
{
   m_afTuple[0] = point.x1;
   m_afTuple[1] = point.x2;
   m_afTuple[2] = point.x3;
}

//----------------------------------------------------------------------------
string GbVector3D::toString()
{
   std::stringstream ss;
   ss<< "GbVector3D["<<m_afTuple[0]<<","<<m_afTuple[1]<<","<<m_afTuple[2]<<"]";
   ss<<endl;
   return((ss.str()).c_str());
}
//----------------------------------------------------------------------------
GbVector3D::operator const double* () const
{
   return m_afTuple;
}
//----------------------------------------------------------------------------
GbVector3D::operator double* ()
{
   return m_afTuple;
}
//----------------------------------------------------------------------------
double GbVector3D::operator[] (int i) const
{
   assert( 0 <= i && i <= 2 );
   if ( i < 0 )
      i = 0;
   else if ( i > 2 )
      i = 2;

   return m_afTuple[i];
}
//----------------------------------------------------------------------------
double& GbVector3D::operator[] (int i)
{
   assert( 0 <= i && i <= 2 );
   if ( i < 0 )
      i = 0;
   else if ( i > 2 )
      i = 2;

   return m_afTuple[i];
}
//----------------------------------------------------------------------------
double GbVector3D::X1 () const
{
   return m_afTuple[0];
}
//----------------------------------------------------------------------------
double& GbVector3D::X1 ()
{
   return m_afTuple[0];
}
//----------------------------------------------------------------------------
double GbVector3D::X2 () const
{
   return m_afTuple[1];
}
//----------------------------------------------------------------------------
double& GbVector3D::X2 ()
{
   return m_afTuple[1];
}
//----------------------------------------------------------------------------
double GbVector3D::X3 () const
{
   return m_afTuple[2];
}
//----------------------------------------------------------------------------
double& GbVector3D::X3 ()
{
   return m_afTuple[2];
}
//----------------------------------------------------------------------------
GbVector3D& GbVector3D::operator= (const GbVector3D& rkV)
{
   m_afTuple[0] = rkV.m_afTuple[0];
   m_afTuple[1] = rkV.m_afTuple[1];
   m_afTuple[2] = rkV.m_afTuple[2];
   return *this;
}
//----------------------------------------------------------------------------
int GbVector3D::CompareArrays (const GbVector3D& rkV) const
{
   return memcmp(m_afTuple,rkV.m_afTuple,3*sizeof(double));
}
//----------------------------------------------------------------------------
bool GbVector3D::operator== (const GbVector3D& rkV) const
{
   return CompareArrays(rkV) == 0;
}
//----------------------------------------------------------------------------
bool GbVector3D::operator!= (const GbVector3D& rkV) const
{
   return CompareArrays(rkV) != 0;
}
//----------------------------------------------------------------------------
bool GbVector3D::operator< (const GbVector3D& rkV) const
{
   return CompareArrays(rkV) < 0;
}
//----------------------------------------------------------------------------
bool GbVector3D::operator<= (const GbVector3D& rkV) const
{
   return CompareArrays(rkV) <= 0;
}
//----------------------------------------------------------------------------
bool GbVector3D::operator> (const GbVector3D& rkV) const
{
   return CompareArrays(rkV) > 0;
}
//----------------------------------------------------------------------------
bool GbVector3D::operator>= (const GbVector3D& rkV) const
{
   return CompareArrays(rkV) >= 0;
}
//----------------------------------------------------------------------------
GbVector3D GbVector3D::operator+ (const GbVector3D& rkV) const
{
   return GbVector3D(
      m_afTuple[0]+rkV.m_afTuple[0],
      m_afTuple[1]+rkV.m_afTuple[1],
      m_afTuple[2]+rkV.m_afTuple[2]);
}
//----------------------------------------------------------------------------
GbVector3D GbVector3D::Add(GbVector3D& vector)
{
   return GbVector3D(
      m_afTuple[0]+vector.m_afTuple[0],
      m_afTuple[1]+vector.m_afTuple[1],
      m_afTuple[2]+vector.m_afTuple[2]);
}

//----------------------------------------------------------------------------
GbVector3D GbVector3D::operator- (const GbVector3D& rkV) const
{
   return GbVector3D(
      m_afTuple[0]-rkV.m_afTuple[0],
      m_afTuple[1]-rkV.m_afTuple[1],
      m_afTuple[2]-rkV.m_afTuple[2]);
}
//----------------------------------------------------------------------------
GbVector3D GbVector3D::Subtract(GbVector3D& vector)
{
   return GbVector3D(
      m_afTuple[0]-vector.m_afTuple[0],
      m_afTuple[1]-vector.m_afTuple[1],
      m_afTuple[2]-vector.m_afTuple[2]);
}
//----------------------------------------------------------------------------
GbVector3D GbVector3D::operator* (const double& fScalar) const
{
   return GbVector3D( fScalar*m_afTuple[0],
                      fScalar*m_afTuple[1],
                      fScalar*m_afTuple[2]);
}
//----------------------------------------------------------------------------
GbVector3D GbVector3D::operator/ (const double& fScalar) const
{
   GbVector3D kQuot;

   if ( fScalar != 0.0 )
   {
      double fInvScalar = 1.0/fScalar;
      kQuot.m_afTuple[0] = fInvScalar*m_afTuple[0];
      kQuot.m_afTuple[1] = fInvScalar*m_afTuple[1];
      kQuot.m_afTuple[2] = fInvScalar*m_afTuple[2];
   }
   else
   {
      kQuot.m_afTuple[0] = Ub::inf;
      kQuot.m_afTuple[1] = Ub::inf;
      kQuot.m_afTuple[2] = Ub::inf;
   }

   return kQuot;
}
//----------------------------------------------------------------------------
GbVector3D GbVector3D::operator- () const
{
   return GbVector3D(
      -m_afTuple[0],
      -m_afTuple[1],
      -m_afTuple[2]);
}
//----------------------------------------------------------------------------
GbVector3D operator* (const double& fScalar, const GbVector3D& rkV)
{
   return GbVector3D(
      fScalar*rkV[0],
      fScalar*rkV[1],
      fScalar*rkV[2]);
}
//----------------------------------------------------------------------------
GbVector3D& GbVector3D::operator+= (const GbVector3D& rkV)
{
   m_afTuple[0] += rkV.m_afTuple[0];
   m_afTuple[1] += rkV.m_afTuple[1];
   m_afTuple[2] += rkV.m_afTuple[2];
   return *this;
}
//----------------------------------------------------------------------------
GbVector3D& GbVector3D::operator-= (const GbVector3D& rkV)
{
   m_afTuple[0] -= rkV.m_afTuple[0];
   m_afTuple[1] -= rkV.m_afTuple[1];
   m_afTuple[2] -= rkV.m_afTuple[2];
   return *this;
}
//----------------------------------------------------------------------------
GbVector3D& GbVector3D::operator*= (const double& fScalar)
{
   m_afTuple[0] *= fScalar;
   m_afTuple[1] *= fScalar;
   m_afTuple[2] *= fScalar;
   return *this;
}
//----------------------------------------------------------------------------
GbVector3D& GbVector3D::operator/= (const double& fScalar)
{
   if ( fScalar != (double)0.0 )
   {
      double fInvScalar = ((double)1.0)/fScalar;
      m_afTuple[0] *= fInvScalar;
      m_afTuple[1] *= fInvScalar;
      m_afTuple[2] *= fInvScalar;
   }
   else
   {
      m_afTuple[0] = Ub::inf;
      m_afTuple[1] = Ub::inf;
      m_afTuple[2] = Ub::inf;
   }

   return *this;
}
//----------------------------------------------------------------------------
GbVector3D GbVector3D::Scale(const double& x)
{
   GbVector3D PointA(0.0,0.0,0.0);
   PointA.m_afTuple[0] = x * m_afTuple[0];
   PointA.m_afTuple[1] = x * m_afTuple[1];
   PointA.m_afTuple[2] = x * m_afTuple[2];
   return PointA;	
}

//----------------------------------------------------------------------------
double GbVector3D::Length () const
{
   return std::sqrt(
      m_afTuple[0]*m_afTuple[0] +
      m_afTuple[1]*m_afTuple[1] +
      m_afTuple[2]*m_afTuple[2]);
}
//----------------------------------------------------------------------------
double GbVector3D::SquaredLength () const
{
   return
      m_afTuple[0]*m_afTuple[0] +
      m_afTuple[1]*m_afTuple[1] +
      m_afTuple[2]*m_afTuple[2];
}
//----------------------------------------------------------------------------
double GbVector3D::Dot (const GbVector3D& rkV) const
{
   return
      m_afTuple[0]*rkV.m_afTuple[0] +
      m_afTuple[1]*rkV.m_afTuple[1] +
      m_afTuple[2]*rkV.m_afTuple[2];
}
//----------------------------------------------------------------------------
double GbVector3D::Normalize ()
{
   double fLength = Length();

   if ( fLength > UbMath::Epsilon<double>::val() )
   {
      double fInvLength = ((double)1.0)/fLength;
      m_afTuple[0] *= fInvLength;
      m_afTuple[1] *= fInvLength;
      m_afTuple[2] *= fInvLength;
   }
   else
   {
      fLength = 0.0;
      m_afTuple[0] = 0.0;
      m_afTuple[1] = 0.0;
      m_afTuple[2] = 0.0;
   }

   return fLength;
}
//----------------------------------------------------------------------------
GbVector3D GbVector3D::Cross (const GbVector3D& rkV) const
{
   return GbVector3D(
      m_afTuple[1]*rkV.m_afTuple[2] - m_afTuple[2]*rkV.m_afTuple[1],
      m_afTuple[2]*rkV.m_afTuple[0] - m_afTuple[0]*rkV.m_afTuple[2],
      m_afTuple[0]*rkV.m_afTuple[1] - m_afTuple[1]*rkV.m_afTuple[0]);
}

//----------------------------------------------------------------------------

GbVector3D GbVector3D::UnitCross (const GbVector3D& rkV) const
{
   GbVector3D kCross(
      m_afTuple[1]*rkV.m_afTuple[2] - m_afTuple[2]*rkV.m_afTuple[1],
      m_afTuple[2]*rkV.m_afTuple[0] - m_afTuple[0]*rkV.m_afTuple[2],
      m_afTuple[0]*rkV.m_afTuple[1] - m_afTuple[1]*rkV.m_afTuple[0]);
   kCross.Normalize();
   return kCross;
}
//----------------------------------------------------------------------------
void GbVector3D::GetBarycentrics (const GbVector3D& rkV0,
                                  const GbVector3D& rkV1, const GbVector3D& rkV2,
                                  const GbVector3D& rkV3, double afBary[4]) const
{
   // compute the vectors relative to V3 of the tetrahedron
   GbVector3D akDiff[4] =
   {
      rkV0 - rkV3,
         rkV1 - rkV3,
         rkV2 - rkV3,
         *this - rkV3
   };

   // If the vertices have large magnitude, the linear system of
   // equations for computing barycentric coordinates can be
   // ill-conditioned.  To avoid this, uniformly scale the tetrahedron
   // edges to be of order 1.  The scaling of all differences does not
   // change the barycentric coordinates.
   double fMax = (double)0.0;
   int i;
   for (i = 0; i < 3; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         double fValue = std::fabs(akDiff[i][j]);
         if ( fValue > fMax )
            fMax = fValue;
      }
   }

   // scale down only large data
   if ( fMax > (double)1.0 )
   {
      double fInvMax = ((double)1.0)/fMax;
      for (i = 0; i < 4; i++)
         akDiff[i] *= fInvMax;
   }

   double fDet = akDiff[0].Dot(akDiff[1].Cross(akDiff[2]));
   GbVector3D kE1cE2 = akDiff[1].Cross(akDiff[2]);
   GbVector3D kE2cE0 = akDiff[2].Cross(akDiff[0]);
   GbVector3D kE0cE1 = akDiff[0].Cross(akDiff[1]);
   if ( std::fabs(fDet) > UbMath::Epsilon<double>::val() )
   {
      double fInvDet = ((double)1.0)/fDet;
      afBary[0] = akDiff[3].Dot(kE1cE2)*fInvDet;
      afBary[1] = akDiff[3].Dot(kE2cE0)*fInvDet;
      afBary[2] = akDiff[3].Dot(kE0cE1)*fInvDet;
      afBary[3] = (double)1.0 - afBary[0] - afBary[1] - afBary[2];
   }
   else
   {
      // The tetrahedron is potentially flat.  Determine the face of
      // maximum area and compute barycentric coordinates with respect
      // to that face.
      GbVector3D kE02 = rkV0 - rkV2;
      GbVector3D kE12 = rkV1 - rkV2;
      GbVector3D kE02cE12 = kE02.Cross(kE12);
      double fMaxSqrArea = kE02cE12.SquaredLength();
      int iMaxIndex = 3;
      double fSqrArea = kE0cE1.SquaredLength();
      if ( fSqrArea > fMaxSqrArea )
      {
         iMaxIndex = 0;
         fMaxSqrArea = fSqrArea;
      }
      fSqrArea = kE1cE2.SquaredLength();
      if ( fSqrArea > fMaxSqrArea )
      {
         iMaxIndex = 1;
         fMaxSqrArea = fSqrArea;
      }
      fSqrArea = kE2cE0.SquaredLength();
      if ( fSqrArea > fMaxSqrArea )
      {
         iMaxIndex = 2;
         fMaxSqrArea = fSqrArea;
      }

      if ( fMaxSqrArea > UbMath::Epsilon<double>::val() )
      {
         double fInvSqrArea = ((double)1.0)/fMaxSqrArea;
         GbVector3D kTmp;
         if ( iMaxIndex == 0 )
         {
            kTmp = akDiff[3].Cross(akDiff[1]);
            afBary[0] = kE0cE1.Dot(kTmp)*fInvSqrArea;
            kTmp = akDiff[0].Cross(akDiff[3]);
            afBary[1] = kE0cE1.Dot(kTmp)*fInvSqrArea;
            afBary[2] = (double)0.0;
            afBary[3] = (double)1.0 - afBary[0] - afBary[1];
         }
         else if ( iMaxIndex == 1 )
         {
            afBary[0] = (double)0.0;
            kTmp = akDiff[3].Cross(akDiff[2]);
            afBary[1] = kE1cE2.Dot(kTmp)*fInvSqrArea;
            kTmp = akDiff[1].Cross(akDiff[3]);
            afBary[2] = kE1cE2.Dot(kTmp)*fInvSqrArea;
            afBary[3] = (double)1.0 - afBary[1] - afBary[2];
         }
         else if ( iMaxIndex == 2 )
         {
            kTmp = akDiff[2].Cross(akDiff[3]);
            afBary[0] = kE2cE0.Dot(kTmp)*fInvSqrArea;
            afBary[1] = (double)0.0;
            kTmp = akDiff[3].Cross(akDiff[0]);
            afBary[2] = kE2cE0.Dot(kTmp)*fInvSqrArea;
            afBary[3] = (double)1.0 - afBary[0] - afBary[2];
         }
         else
         {
            akDiff[3] = *this - rkV2;
            kTmp = akDiff[3].Cross(kE12);
            afBary[0] = kE02cE12.Dot(kTmp)*fInvSqrArea;
            kTmp = kE02.Cross(akDiff[3]);
            afBary[1] = kE02cE12.Dot(kTmp)*fInvSqrArea;
            afBary[2] = (double)1.0 - afBary[0] - afBary[1];
            afBary[3] = (double)0.0;
         }
      }
      else
      {
         // The tetrahedron is potentially a sliver.  Determine the edge of
         // maximum length and compute barycentric coordinates with respect
         // to that edge.
         double fMaxSqrLength = akDiff[0].SquaredLength();
         iMaxIndex = 0;  // <V0,V3>
         double fSqrLength = akDiff[1].SquaredLength();
         if ( fSqrLength > fMaxSqrLength )
         {
            iMaxIndex = 1;  // <V1,V3>
            fMaxSqrLength = fSqrLength;
         }
         fSqrLength = akDiff[2].SquaredLength();
         if ( fSqrLength > fMaxSqrLength )
         {
            iMaxIndex = 2;  // <V2,V3>
            fMaxSqrLength = fSqrLength;
         }
         fSqrLength = kE02.SquaredLength();
         if ( fSqrLength > fMaxSqrLength )
         {
            iMaxIndex = 3;  // <V0,V2>
            fMaxSqrLength = fSqrLength;
         }
         fSqrLength = kE12.SquaredLength();
         if ( fSqrLength > fMaxSqrLength )
         {
            iMaxIndex = 4;  // <V1,V2>
            fMaxSqrLength = fSqrLength;
         }
         GbVector3D kE01 = rkV0 - rkV1;
         fSqrLength = kE01.SquaredLength();
         if ( fSqrLength > fMaxSqrLength )
         {
            iMaxIndex = 5;  // <V0,V1>
            fMaxSqrLength = fSqrLength;
         }

         if ( fMaxSqrLength > UbMath::Epsilon<double>::val() )
         {
            double fInvSqrLength = ((double)1.0)/fMaxSqrLength;
            if ( iMaxIndex == 0 )
            {
               // P-V3 = t*(V0-V3)
               afBary[0] = akDiff[3].Dot(akDiff[0])*fInvSqrLength;
               afBary[1] = (double)0.0;
               afBary[2] = (double)0.0;
               afBary[3] = (double)1.0 - afBary[0];
            }
            else if ( iMaxIndex == 1 )
            {
               // P-V3 = t*(V1-V3)
               afBary[0] = (double)0.0;
               afBary[1] = akDiff[3].Dot(akDiff[1])*fInvSqrLength;
               afBary[2] = (double)0.0;
               afBary[3] = (double)1.0 - afBary[1];
            }
            else if ( iMaxIndex == 2 )
            {
               // P-V3 = t*(V2-V3)
               afBary[0] = (double)0.0;
               afBary[1] = (double)0.0;
               afBary[2] = akDiff[3].Dot(akDiff[2])*fInvSqrLength;
               afBary[3] = (double)1.0 - afBary[2];
            }
            else if ( iMaxIndex == 3 )
            {
               // P-V2 = t*(V0-V2)
               akDiff[3] = *this - rkV2;
               afBary[0] = akDiff[3].Dot(kE02)*fInvSqrLength;
               afBary[1] = (double)0.0;
               afBary[2] = (double)1.0 - afBary[0];
               afBary[3] = (double)0.0;
            }
            else if ( iMaxIndex == 4 )
            {
               // P-V2 = t*(V1-V2)
               akDiff[3] = *this - rkV2;
               afBary[0] = (double)0.0;
               afBary[1] = akDiff[3].Dot(kE12)*fInvSqrLength;
               afBary[2] = (double)1.0 - afBary[1];
               afBary[3] = (double)0.0;
            }
            else
            {
               // P-V1 = t*(V0-V1)
               akDiff[3] = *this - rkV1;
               afBary[0] = akDiff[3].Dot(kE01)*fInvSqrLength;
               afBary[1] = (double)1.0 - afBary[0];
               afBary[2] = (double)0.0;
               afBary[3] = (double)0.0;
            }
         }
         else
         {
            // tetrahedron is a nearly a point, just return equal weights
            afBary[0] = (double)0.25;
            afBary[1] = afBary[0];
            afBary[2] = afBary[0];
            afBary[3] = afBary[0];
         }
      }
   }
}
//----------------------------------------------------------------------------
void GbVector3D::Orthonormalize (GbVector3D& rkU, GbVector3D& rkV, GbVector3D& rkW)
{
   // If the input vectors are v0, v1, and v2, then the Gram-Schmidt
   // orthonormalization produces vectors u0, u1, and u2 as follows,
   //
   //   u0 = v0/|v0|
   //   u1 = (v1-(u0*v1)u0)/|v1-(u0*v1)u0|
   //   u2 = (v2-(u0*v2)u0-(u1*v2)u1)/|v2-(u0*v2)u0-(u1*v2)u1|
   //
   // where |A| indicates length of vector A and A*B indicates dot
   // product of vectors A and B.

   // compute u0
   rkU.Normalize();

   // compute u1
   double fDot0 = rkU.Dot(rkV); 
   rkV -= fDot0*rkU;
   rkV.Normalize();

   // compute u2
   double fDot1 = rkV.Dot(rkW);
   fDot0 = rkU.Dot(rkW);
   rkW -= fDot0*rkU + fDot1*rkV;
   rkW.Normalize();
}
//----------------------------------------------------------------------------
void GbVector3D::Orthonormalize (GbVector3D* akV)
{
   Orthonormalize(akV[0],akV[1],akV[2]);
}
//----------------------------------------------------------------------------
void GbVector3D::GenerateOrthonormalBasis (GbVector3D& rkU, GbVector3D& rkV,
                                           GbVector3D& rkW, bool bUnitLengthW)
{
   if ( !bUnitLengthW )
      rkW.Normalize();

   double fInvLength;

   if ( std::fabs(rkW.m_afTuple[0]) >=
      std::fabs(rkW.m_afTuple[1]) )
   {
      // W.x or W.z is the largest magnitude component, swap them
      fInvLength = UbMath::invSqrt(rkW.m_afTuple[0]*rkW.m_afTuple[0] + rkW.m_afTuple[2]*rkW.m_afTuple[2]);
      rkU.m_afTuple[0] = -rkW.m_afTuple[2]*fInvLength;
      rkU.m_afTuple[1] = (double)0.0;
      rkU.m_afTuple[2] = +rkW.m_afTuple[0]*fInvLength;
   }
   else
   {
      // W.y or W.z is the largest magnitude component, swap them
      fInvLength = UbMath::invSqrt(rkW.m_afTuple[1]*rkW.m_afTuple[1] + rkW.m_afTuple[2]*rkW.m_afTuple[2]);
      rkU.m_afTuple[0] = (double)0.0;
      rkU.m_afTuple[1] = +rkW.m_afTuple[2]*fInvLength;
      rkU.m_afTuple[2] = -rkW.m_afTuple[1]*fInvLength;
   }

   rkV = rkW.Cross(rkU);
}
//----------------------------------------------------------------------------

