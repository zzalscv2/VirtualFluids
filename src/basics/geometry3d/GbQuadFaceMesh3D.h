//  _    ___      __              __________      _     __
// | |  / (_)____/ /___  ______ _/ / ____/ /_  __(_)___/ /____
// | | / / / ___/ __/ / / / __ `/ / /_  / / / / / / __  / ___/
// | |/ / / /  / /_/ /_/ / /_/ / / __/ / / /_/ / / /_/ (__  )
// |___/_/_/   \__/\__,_/\__,_/_/_/   /_/\__,_/_/\__,_/____/
//
#ifndef GBQUADFACEMESH3D_H
#define GBQUADFACEMESH3D_H

#include <sstream>
#include <iostream>


#include <geometry3d/GbObject3D.h>
#include <basics/utilities/UbException.h>  

#include <PointerDefinitions.h>

class UbFileOutput;
class UbFileInput;
/*=========================================================================*/
/* GbQuadFaceMesh3D                                                                  */
/*                                                                         */
/**
 * This Class provides the triangular meshes.
 * Note, that up to now no methods for checking consistency are included.
 * in this context this class describes facettes from an 3D-object !!!
*/
class GbQuadFaceMesh3D : public GbObject3D 
{   
public:
  // nested class start
   class Vertex
   {
   public:
      Vertex()= default;
      Vertex(float x, float y, float z)
      {
         this->x=x;
         this->y=y;
         this->z=z;
      }
      float x, y, z;
   };

   class QuadFace
   {
   public:
      QuadFace() = default;
      QuadFace(int v1, int v2, int v3, int v4)
      {
         this->vertex1=v1;
         this->vertex2=v2;
         this->vertex3=v3;
         this->vertex4=v4;
      }

      int vertex1, vertex2, vertex3, vertex4;
   };
 // nested class end

public:
   GbQuadFaceMesh3D();
	GbQuadFaceMesh3D(std::string name, std::vector<Vertex> *nodes, std::vector<QuadFace> *quads);
	~GbQuadFaceMesh3D() override;   
   GbQuadFaceMesh3D* clone() override { throw UbException(UB_EXARGS,"clone() - not implemented"); }
   void finalize() override           { throw UbException(UB_EXARGS,"finalize() - not implemented");}

   std::string toString() override;
   std::string getName() override;
   std::vector<Vertex>*  getNodes();
   std::vector<QuadFace>* getQuads();
   double getX1Centroid() override;
   double getX2Centroid() override;
   double getX3Centroid() override;
   double getX1Minimum() override;
   double getX1Maximum() override;
   double getX2Minimum() override;
   double getX2Maximum() override;
   double getX3Minimum() override;
   double getX3Maximum() override;
   void calculateValues();

   bool isPointInGbObject3D(const double& x1, const double& x2, const double& x3) override;
   bool isPointInGbObject3D(const double& x1, const double& x2, const double& x3, bool& pointIsOnBoundary) override;

   bool isPointInObject3DHalfSpace(const double& xp, const double& yp, const double& zp);    //based on Halfspace algorithm
   //bool isPointInObject3DSpherical(const double& xp, const double& yp, const double& zp, int numQuads);    //based on Spherical polygon area method        
   //bool isPointInObject3DRayCrossing(const double& xp, const double& yp, const double& zp, int radius, int numVertices, int numQuads);  //based on Ray tracing algorithm
   
   //char SegPlaneInt(GbQuad3D *quad, GbVector3D  &PointQ, GbVector3D &PointR, GbVector3D &Point, int *m);
   //char SegQuadCross(GbQuad3D *quad, GbVector3D  &PointQ, GbVector3D &PointR);
   //till here !!!

   GbLine3D* createClippedLine3D (GbPoint3D &point1,GbPoint3D &point2) override;
   //virtual std::vector<GbQuad3D*> getSurfaceQuadSet();
	std::vector<GbTriangle3D*> getSurfaceTriangleSet() override;

   virtual void write(UbFileOutput*  /*out*/) { std::cout<<"GbQuadFaceMesh3D::write - sorry not implemented\n"; }
   virtual void read(UbFileInput*  /*in*/)    { std::cout<<"GbQuadFaceMesh3D::read  - sorry not implemented\n"; }

   void writeAVSMesh(UbFileOutput *out, bool normals=false);

   /*======================================================================*/
   using GbObject3D::isPointInGbObject3D; //Grund: dadurch muss man hier  isPointInGbObject3D(GbPoint3D*) nicht ausprogrammieren, welche sonst hier "ueberdeckt" waere
private:
   void init();
   /*======================================================================*/
   std::string name;
   std::vector<Vertex>   *nodes;
   std::vector<QuadFace> *quads;
   double      x1min;
   double      x1max;
   double      x2min;
   double      x2max;
   double      x3min;
   double      x3max;
   bool        consistent;
};
/*=========================================================================*/

#endif
