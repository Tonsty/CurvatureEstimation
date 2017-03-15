template<class MeshReal>
template<class Vertex,class Real>
void MeshInfo<MeshReal>::set(const std::vector<Vertex>& verts,const std::vector<std::vector<int> >& polys,const Real& width,
	Point3D<Real>& translate,Real& scale,const int& noTransform)
{
	if(!noTransform)
	{
		Point3D<Real> min,max;
		for(size_t i=0;i<verts.size();i++)
		{
			for(int j=0;j<3;j++)
			{
				if(!i || Point3D<Real>(verts[i])[j]<min[j])	min[j]=Point3D<Real>(verts[i])[j];
				if(!i || Point3D<Real>(verts[i])[j]>max[j])	max[j]=Point3D<Real>(verts[i])[j];
			}
		}

		scale=max[0]-min[0];
		if( (max[1]-min[1])>scale )		scale=max[1]-min[1];
		if( (max[2]-min[2])>scale )		scale=max[2]-min[2];

		scale*=width;
		scale=Real(1.0)/scale;
		Point3D<Real> ctr;
		ctr[0]=ctr[1]=ctr[2]=Real(0.5);

		translate=ctr/scale-(max+min)/2;
	}
	else
	{
		translate[0]=translate[1]=translate[2]=0;
		scale=1;
	}
	vertices.resize(verts.size());
	for(size_t i=0;i<verts.size();i++)
		for(int j=0;j<3;j++)
			vertices[i][j]=MeshReal((translate[j]+(Point3D<Real>(verts[i]))[j])*scale);

	//vertices have precomputed normals, use point to point (plane) distance metric to compute signed distance
	vertexNormals.resize(vertices.size());
	vertexCurvatures.resize(vertices.size());
	for(size_t i=0;i<verts.size();i++) {
		for(int j=0;j<3;j++)
			vertexNormals[i][j]=verts[i].normal[j]; //MeshReal(*(float*)((char*)&verts[i]+Vertex::Properties[3+j].offset));
		vertexNormals[i]/=Length(vertexNormals[i]);
		vertexCurvatures[i]=verts[i].curvature; //MeshReal(*(float*)((char*)&verts[i]+Vertex::Properties[6].offset));
		vertexCurvatures[i]=(MeshReal)vertexCurvatures[i]/scale;
	}
}