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

void ShowUsage(char* ex)
{
	printf("Usage: %s\n",ex);
	printf("\t--in  <input data>\n");
	printf("\t\tInput mesh (.ply) used to compute curvature.\n");

	printf("\t--out <ouput data>\n");
	printf("\t\tOutput mesh (.ply) with the curvature.\n");

	printf("\t--knn <number of neighbors>\n");
	printf("\t\tNumber of nearest neighbors to search for MLS.\n");

	printf("\t--gauss <gauss delta>\n");
	printf("\t\tGauss delta for MLS.\n");

	printf("\t--adaptive \n");
	printf("\t\tPerform adaptive MLS.\n");
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

	cmdLineString In, Out;
	cmdLineInt Knn;
	cmdLineFloat Gauss;
	cmdLineReadable Adaptive;
	char* paramNames[]=
	{
		"in","out","knn","gauss","adaptive"
	};
	cmdLineReadable* params[]= 
	{
		&In,&Out,&Knn,&Gauss,&Adaptive
	};
	int paramNum=sizeof(paramNames)/sizeof(char*);
	cmdLineParse(argc-1,&argv[1],paramNames,paramNum,params,1);

	if(!In.set || !Out.set || !Knn.set || !Gauss.set)
	{
		ShowUsage(argv[0]);
		return EXIT_FAILURE;
	}

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
	mInfo.set(inVertices,polygons,(float)1.1,translate,scale,1);
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
	estimator.setParameters(Knn.value,Gauss.value,Adaptive.set);
	std::vector<float> gaussian, mean, k1, k2;
	estimator.compute(gaussian,mean,k1,k2);
	printf("Finished computing in: %f\n", Time()-t);

	for(int i=0;i<inVertices.size();i++)
	{
		float curvature;
		if(abs(k1[i])>abs(k2[i])) curvature=abs(k1[i]);
		else curvature=abs(k2[i]);
		inVertices[i].curvature=curvature*scale;
	}
	PlyWritePolygons(Out.value,inVertices,polygons,ft);

	return EXIT_SUCCESS;
}