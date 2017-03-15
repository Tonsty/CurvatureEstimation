#ifndef MESH_INFO_INCLUDED
#define MESH_INFO_INCLUDED

template<class MeshReal>
class MeshInfo
{
public:
	std::vector<Point3D<MeshReal> > vertexNormals;
	std::vector<Point3D<MeshReal> > vertices;
	std::vector<MeshReal> vertexCurvatures;

	template<class Vertex,class Real>
	void set(const std::vector<Vertex>& vertices,const std::vector<std::vector<int> >& polygons,const Real& width,
		Point3D<Real>& translate,Real& scale,const int& noTransform);
};

#include "MeshInfo.inl"

#endif