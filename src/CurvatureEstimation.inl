#include <Eigen/Dense>
#include <omp.h>

template<class Real>
void CurvatureEstimation<Real>::compute(std::vector<Real>& gaussian, std::vector<Real> &mean, std::vector<Real> &k1, std::vector<Real> &k2)
{
	Eigen::Matrix<Real,Eigen::Dynamic,1> output_gaussian(npts),output_mean(npts);

	int num_procs=omp_get_num_procs();
	omp_set_num_threads(num_procs);
#pragma omp parallel for
	for(int i=0;i<npts;i++)
	{
		std::vector<std::vector<int> > indices; 
		std::vector<std::vector<Real> > dists;
		kdtree->KnnSearch((Real*)(points+3*i),1,indices,dists,knn);

		Real gauss;
		if(!adaptive) gauss=this->gauss;
		else
		{
			Real radius2=dists[0][knn-1];
			Real density=knn/(3.1415926*radius2);
			gauss=this->gauss/density;
		}

		Eigen::Matrix<Real,3,1> eval_pnt;
		eval_pnt(0)=points[3*i+0];
		eval_pnt(1)=points[3*i+1];
		eval_pnt(2)=points[3*i+2];
		//std::cout<<"eval_pnt=\n"<<eval_pnt<<std::endl;

		Eigen::Matrix<Real,3,Eigen::Dynamic> neighbor_pnts(3,knn);
		for(int ii=0;ii<knn;ii++)
		{
			neighbor_pnts(0,ii)=points[3*indices[0][ii]+0];
			neighbor_pnts(1,ii)=points[3*indices[0][ii]+1];
			neighbor_pnts(2,ii)=points[3*indices[0][ii]+2];
		}
		//std::cout<<"neighbor_pnts=\n"<<neighbor_pnts<<std::endl;

		Eigen::Matrix<Real,3,Eigen::Dynamic> diff_vec(3,knn);
		for(int ii=0;ii<knn;ii++) diff_vec.col(ii)=eval_pnt-neighbor_pnts.col(ii);
		//std::cout<<"diff_vec=\n"<<diff_vec<<std::endl;
		Eigen::Matrix<Real,1,Eigen::Dynamic> dist_squared=diff_vec.cwiseAbs2().colwise().sum();
		//std::cout<<"dist_squared=\n"<<dist_squared<<std::endl;
		Real gauss2=gauss*gauss;
		Eigen::Matrix<Real,1,Eigen::Dynamic> weight=(-dist_squared/gauss2).array().exp().matrix();
		//std::cout<<"weight=\n"<<weight<<std::endl;

		Eigen::Matrix<Real,3,Eigen::Dynamic> neighbor_normals(3,knn);
		for(int ii=0;ii<knn;ii++)
		{
			neighbor_normals(0,ii)=normals[3*indices[0][ii]+0];
			neighbor_normals(1,ii)=normals[3*indices[0][ii]+1];
			neighbor_normals(2,ii)=normals[3*indices[0][ii]+2];
		}
		//std::cout<<"neighbor_normals=\n"<<neighbor_normals<<std::endl;

		Eigen::Matrix<Real,3,1> eval_normal;
		for(int ii=0;ii<3;ii++) eval_normal(ii) = weight.cwiseProduct(neighbor_normals.row(ii)).sum();
		//std::cout<<"eval_normal=\n"<<eval_normal<<std::endl;

		Eigen::Matrix<Real,3,1> normalized_eval_normal=eval_normal.normalized();
		//std::cout<<"normalized_eval_normal=\n"<<normalized_eval_normal<<std::endl;

		Eigen::Matrix<Real,1,Eigen::Dynamic> projected_diff_vec=
			diff_vec.row(0)*normalized_eval_normal(0)
			+diff_vec.row(1)*normalized_eval_normal(1)
			+diff_vec.row(2)*normalized_eval_normal(2);
		//std::cout<<"projected_diff_vec=\n"<<projected_diff_vec<<std::endl;

		Eigen::Matrix<Real,3,1> delta_g=Eigen::Matrix<Real,3,1>::Zero();
		Eigen::Matrix<Real,1,Eigen::Dynamic> term_1st=
			2*weight.cwiseProduct(
			2/gauss2*projected_diff_vec.cwiseProduct(
			projected_diff_vec.cwiseAbs2()/gauss2
			-Eigen::Matrix<Real,1,Eigen::Dynamic>::Ones(knn)));
		//std::cout<<"term_1st=\n"<<term_1st<<std::endl;

		for(int ii=0;ii<3;ii++) delta_g(ii)=(term_1st.cwiseProduct(diff_vec.row(ii))).sum();

		Eigen::Matrix<Real,1,Eigen::Dynamic> term_2nd=
			2*weight.cwiseProduct(
			Eigen::Matrix<Real,1,Eigen::Dynamic>::Ones(knn)
			-3/gauss2*projected_diff_vec.cwiseAbs2());
		//std::cout<<"term_2nd=\n"<<term_2nd<<std::endl;

		Eigen::Matrix<Real,3,3> temp=Eigen::Matrix<Real,3,3>::Zero();
		for(int ii=0;ii<3;ii++)
		{
			for(int jj=0;jj<3;jj++)
			{
				temp(ii,jj)=-2/gauss2*weight.cwiseProduct(neighbor_normals.row(ii)).cwiseProduct(diff_vec.row(jj)).sum();
			}
		}

		Eigen::Matrix<Real,3,Eigen::Dynamic> temp_2=(Eigen::Matrix<Real,3,3>::Identity()-eval_normal*eval_normal.transpose()/eval_normal.squaredNorm())*temp.transpose()/eval_normal.norm()*diff_vec;

		for(int ii=0;ii<3;ii++)
		{
			delta_g(ii)=delta_g(ii)+(term_2nd*normalized_eval_normal(ii)).sum()+term_2nd.cwiseProduct(temp_2.row(ii)).sum();
		}
		//std::cout<<"delta_g=\n"<<delta_g<<std::endl;

		Eigen::Matrix<Real,3,3> delta2_g=Eigen::Matrix<Real,3,3>::Zero();
		Eigen::Matrix<Real,3,3> delta_eval_normal=(Eigen::Matrix<Real,3,3>::Identity()-eval_normal*eval_normal.transpose()/eval_normal.squaredNorm())*temp/eval_normal.norm();
		//std::cout<<"delta_eval_normal=\n"<<delta_eval_normal<<std::endl;

		term_1st=-4/gauss2*weight.cwiseProduct(2/gauss2*projected_diff_vec.cwiseProduct(projected_diff_vec.cwiseAbs2()/gauss2-Eigen::Matrix<Real,1,Eigen::Dynamic>::Ones(knn)));
		//std::cout<<"term_1st=\n"<<term_1st<<std::endl;
		for(int ii=0;ii<knn;ii++)
		{
			delta2_g=delta2_g+term_1st(ii)*(diff_vec.col(ii)*diff_vec.col(ii).transpose());
		}
		//std::cout<<"delta2_g=\n"<<delta2_g<<std::endl;

		term_2nd=-4/gauss2*weight.cwiseProduct(Eigen::Matrix<Real,1,Eigen::Dynamic>::Ones(knn)-3*projected_diff_vec.cwiseAbs2()/gauss2);
		//std::cout<<"term_2nd=\n"<<term_2nd<<std::endl;
		for(int ii=0;ii<knn;ii++)
		{
			delta2_g=delta2_g+term_2nd(ii)*(normalized_eval_normal+delta_eval_normal.transpose()*diff_vec.col(ii))*diff_vec.col(ii).transpose();
		}
		//std::cout<<"delta2_g=\n"<<delta2_g<<std::endl;

		Real gauss4=gauss2*gauss2;
		Eigen::Matrix<Real,1,Eigen::Dynamic> term_3rd=2*weight.cwiseProduct(6*projected_diff_vec.cwiseAbs2()/gauss4-2/gauss2*Eigen::Matrix<Real,1,Eigen::Dynamic>::Ones(knn));
		//std::cout<<"term_3rd=\n"<<term_3rd<<std::endl;

		for(int ii=0;ii<knn;ii++)
		{
			delta2_g=delta2_g+term_3rd(ii)*diff_vec.col(ii)*(diff_vec.col(ii).transpose()*delta_eval_normal+normalized_eval_normal.transpose());
		}
		//std::cout<<"delta2_g=\n"<<delta2_g<<std::endl;

		Eigen::Matrix<Real,1,Eigen::Dynamic> term_4th=4/gauss2*weight.cwiseProduct(projected_diff_vec).cwiseProduct(projected_diff_vec.cwiseAbs2()/gauss2-Eigen::Matrix<Real,1,Eigen::Dynamic>::Ones(knn));
		//std::cout<<"term_4th=\n"<<term_4th<<std::endl;

		for(int ii=0;ii<knn;ii++)
		{
			delta2_g=delta2_g+term_4th(ii)*Eigen::Matrix<Real,3,3>::Identity();
		}

		Eigen::Matrix<Real,1,Eigen::Dynamic> term_5th=-12/gauss2*weight.cwiseProduct(projected_diff_vec);
		//std::cout<<"term_5th=\n"<<term_5th<<std::endl;

		for(int ii=0;ii<knn;ii++)
		{
			delta2_g=delta2_g+term_5th(ii)*(normalized_eval_normal+delta_eval_normal.transpose()*diff_vec.col(ii))*(diff_vec.col(ii).transpose()*delta_eval_normal+normalized_eval_normal.transpose());
		}
		//std::cout<<"delta2_g=\n"<<delta2_g<<std::endl;

		Eigen::Matrix<Real,1,Eigen::Dynamic> term_6th=2*weight.cwiseProduct(Eigen::Matrix<Real,1,Eigen::Dynamic>::Ones(knn)-3/gauss2*projected_diff_vec.cwiseAbs2());
		//std::cout<<"term_6th=\n"<<term_6th<<std::endl;

		for(int ii=0;ii<knn;ii++)
		{
			delta2_g=delta2_g+term_6th(ii)*(delta_eval_normal+delta_eval_normal.transpose());
		}
		//std::cout<<"delta2_g=\n"<<delta2_g<<std::endl;

		Eigen::Matrix<Real,3,3> temp_[3]={Eigen::Matrix<Real,3,3>::Zero(),Eigen::Matrix<Real,3,3>::Zero(),Eigen::Matrix<Real,3,3>::Zero()};
		for(int kk=0;kk<3;kk++)
		{
			for(int ii=0;ii<3;ii++)
			{
				for(int jj=0;jj<3;jj++)
				{
					temp_[kk](ii,jj)=temp_[kk](ii,jj)+(4/gauss4*weight.cwiseProduct(diff_vec.row(ii)).cwiseProduct(diff_vec.row(jj)).cwiseProduct(neighbor_normals.row(kk))).sum();
					if(ii==jj)
					{
						temp_[kk](ii,jj)=temp_[kk](ii,jj)+(-2/gauss2*weight.cwiseProduct(neighbor_normals.row(kk))).sum();
					}
				}
			}
			//std::cout<<"temp_["<<kk<<"]=\n"<<temp_[kk]<<std::endl;
		}

		for(int ii=0;ii<knn;ii++)
		{
			Eigen::Matrix<Real,3,3> temp_2=temp_[0]*diff_vec(0,ii)+temp_[1]*diff_vec(1,ii)+temp_[2]*diff_vec(2,ii);
			delta2_g=delta2_g+term_6th(ii)*(Eigen::Matrix<Real,3,3>::Identity()-eval_normal*eval_normal.transpose()/eval_normal.squaredNorm())*temp_2/eval_normal.norm();
		}
		//std::cout<<"delta2_g=\n"<<delta2_g<<std::endl;

		Eigen::Matrix<Real,4,4> temp_matrix;
		temp_matrix.topLeftCorner(3,3)=delta2_g;
		temp_matrix.topRightCorner(3,1)=delta_g;
		temp_matrix.bottomLeftCorner(1,3)=delta_g.transpose();
		temp_matrix(3,3)=0;

		Real delta_g_norm=delta_g.norm();
		Real delta_g_norm2=delta_g_norm*delta_g_norm;
		Real delta_g_norm4=delta_g_norm2*delta_g_norm2;
		output_gaussian(i)=-temp_matrix.determinant()/delta_g_norm4;
		//std::cout<<"output_gaussian=\n"<<output_gaussian(i)<<std::endl;

		Real delta_g_norm3=delta_g_norm*delta_g_norm*delta_g_norm;
		output_mean(i)=(delta_g.transpose()*delta2_g*delta_g-delta_g.squaredNorm()*delta2_g.trace())/delta_g_norm3/2;
		//std::cout<<"output_mean=\n"<<output_mean(i)<<std::endl;
	}

	Eigen::Matrix<Real,Eigen::Dynamic,1> temp=output_mean.cwiseProduct(output_mean)-output_gaussian;
	for(int i=0;i<temp.size();i++) if(temp(i)<0) temp(i)=0;
	Eigen::Matrix<Real,Eigen::Dynamic,1> output_k1=output_mean+temp.cwiseSqrt();
	Eigen::Matrix<Real,Eigen::Dynamic,1> output_k2=output_mean-temp.cwiseSqrt();

	gaussian.clear();
	mean.clear();
	k1.clear();
	k2.clear();
	for(int i=0;i<npts;i++)
	{
		gaussian.push_back(output_gaussian(i));
		mean.push_back(output_mean(i));
		k1.push_back(output_k1(i));
		k2.push_back(output_k2(i));
	}
}

