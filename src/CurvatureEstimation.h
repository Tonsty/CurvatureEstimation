#ifndef CURVATURE_ESTIMATION_INCLUDED
#define CURVATURE_ESTIMATION_INCLUDED

#include <vector>
#include "Kdtree.h"

template<class Real>
class CurvatureEstimation
{
	size_t knn;
	float gauss;
	Real* points;
	Real* normals;
	size_t npts;
	KDTree<Real> *kdtree;
public:
	CurvatureEstimation():knn(20),gauss(-1){}
	inline void setParameters(size_t knn, Real gauss) {this->knn=knn; this->gauss=gauss;}
	inline void setData(Real* points, Real* normals, size_t npts, KDTree<Real> *kdtree) {this->points=points; this->normals=normals; this->npts=npts; this->kdtree=kdtree;}
	void compute(std::vector<Real>& gaussian, std::vector<Real> &mean, std::vector<Real> &k1, std::vector<Real> &k2);
};

#include "CurvatureEstimation.inl"

#endif
