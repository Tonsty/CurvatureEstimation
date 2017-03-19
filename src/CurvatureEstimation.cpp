#include "Ply.h"
#include "PlyFile.h"
#include "Geometry.h"
#include "Kdtree.h"
#include "Time.h"
#include "CmdLineParser.h"
#include "MeshInfo.h"
#include "CurvatureEstimation.h"
#include <stdio.h>
#include <vector>

typedef PlyVertexWithNormal InOutPlyVertex;

template<class Vertex>
void saveVTK(const char*vtkFileName, const std::vector<Vertex> &vertices, const std::vector<std::vector<int> > polygons)
{
	// Open file
	FILE*vtkFile=fopen(vtkFileName,"w");

	int npts=vertices.size();

	// Write the header information
	fprintf(vtkFile,"# vtk DataFile Version 3.0\n");
	fprintf(vtkFile,"vtk output\n");
	fprintf(vtkFile,"ASCII\n");
	fprintf(vtkFile,"DATASET POLYDATA\n");
	fprintf(vtkFile,"POINTS %d float\n",npts);

	// Iterate through the points
	for(int i=0;i<npts;++i) fprintf(vtkFile,"%f %f %f\n",vertices[i].point[0],vertices[i].point[1],vertices[i].point[2]);

	// Write vertices
	fprintf(vtkFile,"\nVERTICES %d %d\n", npts, 2*npts);
	for(int i=0;i<npts;++i) fprintf(vtkFile,"1 %d\n",i);

	// Write RGB values
	fprintf(vtkFile,"\nPOINT_DATA %d\n",npts);
	fprintf(vtkFile,"COLOR_SCALARS RGB 3\n");
	for(int i=0;i<npts;++i) fprintf(vtkFile,"1.0 1.0 1.0\n");

	// Write normal values
	fprintf(vtkFile,"\nNORMALS normal float\n");
	for(int i=0;i<npts;++i) fprintf(vtkFile,"%f %f %f\n",vertices[i].normal[0],vertices[i].normal[1],vertices[i].normal[2]);

	// Write curvature values
	fprintf(vtkFile,"\nSCALARS curvature float\n");
	fprintf(vtkFile,"LOOKUP_TABLE default\n");
	for(int i=0;i<npts;++i) fprintf(vtkFile,"%f\n",vertices[i].curvature);

	// Close file
	fclose(vtkFile);
}

void ShowUsage(char* ex)
{
	printf("Usage: %s\n",ex);
	printf("\t--in  <input data>\n");
	printf("\t\tInput mesh (.ply) used to compute curvature.\n\n");

	printf("\t--out <ouput data>\n");
	printf("\t\tOutput mesh (.ply) with the curvature.\n\n");

	printf("\t--knn <number of neighbors>\n");
	printf("\t\tNumber of nearest neighbors to search for MLS.\n\n");

	printf("\t--gauss <gauss delta>\n");
	printf("\t\tGauss delta for MLS.\n\n");

	printf("\t--adaptive <factor=0.3>\n");
	printf("\t\tIf this flag is set, adaptive MLS is perfomed.\n");
	printf("\t\tThe factor is multiplied with midRadius\n");
	printf("\t\tto generate a gauss for midRadius point,\n");
	printf("\t\tgauss for others are adaptive.\n\n");


	printf("\t--vtk <vtkfile name>\n");
	printf("\t\tSave vtk format file.\n\n");

	printf("\t--normalize\n");
	printf("\t\tIf this flag is set, the bounding box will\n");
	printf("\t\tbe normalized to [0,0,0]~[1,1,1].\n\n");
}

int main(int argc, char**argv)
{
	//int number_of_neighbor=20; //bunny
	//int gaussian_param=0.0015; //bunny

	//int number_of_neighbor=20; //horse
	//int gaussian_param=0.008; //horse
	//
	//int number_of_neighbor=20; //dragon
	//int gaussian_param=0.0003; //dragon

	cmdLineString In,Out,VTK;
	cmdLineInt Knn;
	cmdLineFloat Gauss,Adaptive(0.3),Normalize;
	char* paramNames[]=
	{
		"in","out","knn","gauss","adaptive","vtk","normalize"
	};
	cmdLineReadable* params[]= 
	{
		&In,&Out,&Knn,&Gauss,&Adaptive,&VTK,&Normalize
	};
	int paramNum=sizeof(paramNames)/sizeof(char*);
	cmdLineParse(argc-1,&argv[1],paramNames,paramNum,params,1);

	if(!In.set || !Out.set || !Knn.set || (!Gauss.set && !Adaptive.set))
	{
		ShowUsage(argv[0]);
		return EXIT_FAILURE;
	}
	if(Gauss.set) Adaptive.value=(float)-1.0;

	int ft;
	std::vector<InOutPlyVertex> inVertices;
	std::vector<std::vector<int> > polygons;

	double t=Time();
	printf("Loading data ...\n");
	PlyReadPolygons(In.value,inVertices,polygons,ft);
	printf("Got data in: %f\n", Time()-t);

	t=Time();
	printf("Transforming data ...\n");
	Point3D<float> translate;
	float scale=1.f;
	translate[0]=translate[1]=translate[2]=0;
	MeshInfo<float> mInfo;
	mInfo.set(inVertices,polygons,(float)1.0,translate,scale,0);
	printf("Finished transforming in: %f\n", Time()-t);

	t=Time();
	printf("Building Kdtree ...\n");
	KDTree<float> kdtree;
	kdtree.setInputPoints((float*)mInfo.vertices.data(),mInfo.vertices.size());
	printf("Finished building in: %f\n", Time()-t);

	t=Time();
	printf("Computing MLS curvatures ...\n");
	CurvatureEstimation<float> estimator;
	estimator.setData((float*)mInfo.vertices.data(),(float*)mInfo.vertexNormals.data(),mInfo.vertices.size(),&kdtree);
	estimator.setParameters(Knn.value,Gauss.value,Adaptive.value);
	std::vector<float> gaussian, mean, k1, k2;
	estimator.compute(gaussian,mean,k1,k2);
	if(Adaptive.set) printf("midRadius=%f\n",estimator.midRadius);
	printf("Finished computing in: %f\n", Time()-t);

	for(int i=0;i<inVertices.size();i++)
	{
		float curvature;
		if(abs(k1[i])>abs(k2[i])) curvature=abs(k1[i]);
		else curvature=abs(k2[i]);
		inVertices[i].curvature=curvature*scale;
	}

	if(Normalize.set)
	{
		for(int i=0;i<inVertices.size();i++)
		{
			inVertices[i].point=(inVertices[i].point+translate)*scale;
			inVertices[i].curvature/=scale;
		}
	}

	PlyWritePolygons(Out.value,inVertices,polygons,ft);

	if(VTK.set) saveVTK(VTK.value,inVertices,polygons);

	return EXIT_SUCCESS;
}


