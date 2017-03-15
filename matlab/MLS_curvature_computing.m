function [output_gaussian, output_mean, output_k1, output_k2] = MLS_curvature_computing(input_pnts, input_normals, eval_pnts, number_of_neighbor, gaussian_param)
% Direct computing of point-set surface curvatures  
%
% [out_a , out_b , out_c , out_d] = MLS_curvature_computing (in_1 , in_2 , in_3 , in_4 , in_5)
%
% Description :
% Direct computing of surface curvatures for point-set surfaces based on  
% a set of analytical equations derived from MLS.
%
% Parameters [IN ]:
% in_1 : input synthetic/real point data, which will define an MLS surface
% in_2 : corresponding normals of the input synthetic/real point data
% in_3 : positions where to evaluate surface curvatures
% in_4 : number of neighbor points, which will contribute to the curvature
%        computing at each evaluation position
% in_5 : Gaussian scale parameter that determines the width of the Gaussian
%        kernel
%
% Returns [ OUT ]:
% out_a : output gaussian curvatures
% out_b : output mean curvatures
% out_c : output maximum principal curvatures  
% out_d : output minimum principal curvatures
%
% Example :
% see the demo
%
% References :
%  Yang, P. and Qian, X., "Direct computing of surface curvatures for
%  point-set surfaces," Proceedings of 2007 IEEE/Eurographics Symposium 
%  on Point-based Graphics(PBG), Prague, Czech Republic, Sep. 2007.
%
% Authors : Pinghai Yang
% Created : Sep. 25, 2007
% Version : 1.0
% Matlab Version : 7.0 ( Windows )
%
% Copyright (c) 2007 CDM Lab, Illinois Institute of Technology , U.S.A.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Output Gaussian curvature
output_gaussian = zeros(size(eval_pnts,2),size(eval_pnts,3));
% Output Mean curvature
output_mean = zeros(size(eval_pnts,2),size(eval_pnts,3));

for i = 1:size(eval_pnts,2),
    for j = 1:size(eval_pnts,3),

        % Coordinates of the current point
        eval_pnt = eval_pnts(:,i,j);
        neighbor_index = nearestneighbour(eval_pnt, input_pnts, 'NumberOfNeighbours', number_of_neighbor);
        neighbor_pnts = input_pnts(:,neighbor_index);

        % Difference vectors from neighbor points to the evaluation point 
        diff_vec = repmat(eval_pnt, 1, number_of_neighbor) - neighbor_pnts;

        dist_squared = diff_vec(1,:).*diff_vec(1,:) + diff_vec(2,:).*diff_vec(2,:) + diff_vec(3,:).*diff_vec(3,:);
        weight = exp(- dist_squared / gaussian_param^2);

        % Delta_g of the current point
        neighbor_normals = input_normals(:,neighbor_index);
        for ii = 1:3
            eval_normal(ii,1) = sum(weight.*neighbor_normals(ii,:));
        end

        normalized_eval_normal = eval_normal/norm(eval_normal);

        projected_diff_vec = diff_vec(1,:).*normalized_eval_normal(1,1) + diff_vec(2,:).*normalized_eval_normal(2,1)...
            + diff_vec(3,:).*normalized_eval_normal(3,1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %delta_g
        delta_g = [0;0;0];
        term_1st = 2*weight.*(2/gaussian_param^2*projected_diff_vec.*(projected_diff_vec.^2/gaussian_param^2-1));

        for ii = 1:3
            delta_g(ii,1) = sum(term_1st.*diff_vec(ii,:));
        end

        term_2nd = 2*weight.*(1-3/gaussian_param^2*projected_diff_vec.^2);

        temp = zeros(3,3);
        for ii = 1:3,
            for jj = 1:3,
                temp(ii,jj) = sum(-2/gaussian_param^2*weight.*neighbor_normals(ii,:).*diff_vec(jj,:));
            end
        end

        temp_2 = (diag([1 1 1]) - eval_normal*eval_normal'/norm(eval_normal)^2)...
            *temp'/norm(eval_normal)*diff_vec;

        for ii = 1:3
            delta_g(ii,1) = delta_g(ii,1) + sum(term_2nd.*normalized_eval_normal(ii,1)) ...
                + sum(term_2nd.*temp_2(ii,:));
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %Derivative of delta_g
        delta2_g = zeros(3,3);

        delta_eval_normal = (diag([1 1 1]) - eval_normal*eval_normal'/norm(eval_normal)^2)...
            *temp/norm(eval_normal);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        term_1st = -4/gaussian_param^2*weight.*(2/gaussian_param^2*projected_diff_vec.*(projected_diff_vec.^2/gaussian_param^2-1));

        for ii =1:number_of_neighbor
            delta2_g = delta2_g + term_1st(1,ii)*(diff_vec(:,ii)*diff_vec(:,ii)');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        term_2nd = -4/gaussian_param^2*weight.*(1-3*projected_diff_vec.^2/gaussian_param^2);

        for ii =1:number_of_neighbor
            delta2_g = delta2_g + term_2nd(1,ii)*(normalized_eval_normal+delta_eval_normal'*diff_vec(:,ii))*diff_vec(:,ii)';
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        term_3rd = 2*weight.*(6*projected_diff_vec.^2/gaussian_param^4-2/gaussian_param^2);

        for ii =1:number_of_neighbor
            delta2_g = delta2_g + term_3rd(1,ii)*diff_vec(:,ii)*(diff_vec(:,ii)'*delta_eval_normal + normalized_eval_normal');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        term_4th = 4/gaussian_param^2*weight.*projected_diff_vec.*(projected_diff_vec.^2/gaussian_param^2-1);

        for ii =1:number_of_neighbor
            delta2_g = delta2_g + term_4th(1,ii)*(diag([1 1 1]));
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        term_5th = -12/gaussian_param^2*weight.*projected_diff_vec;

        for ii =1:number_of_neighbor
            delta2_g = delta2_g + term_5th(1,ii)*(normalized_eval_normal + delta_eval_normal'*diff_vec(:,ii))*(diff_vec(:,ii)'*delta_eval_normal + normalized_eval_normal');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        term_6th = 2*weight.*(1-3/gaussian_param^2*projected_diff_vec.^2);

        for ii =1:number_of_neighbor
            delta2_g = delta2_g + term_6th(1,ii)*(delta_eval_normal + delta_eval_normal');
        end

        temp = zeros(3,3,3);
        for kk = 1:3
            for ii = 1:3,
                for jj = 1:3,
                    temp(ii,jj,kk) = temp(ii,jj,kk) + sum(4/gaussian_param^4*weight.*diff_vec(ii,:)...
                        .*diff_vec(jj,:).*neighbor_normals(kk,:));
                    if ii==jj,
                        temp(ii,jj,kk) = temp(ii,jj,kk) + sum(-2/gaussian_param^2*weight.*neighbor_normals(kk,:));
                    end
                end
            end
        end
        
        for ii =1:number_of_neighbor,
            temp_2 = temp(:,:,1)*diff_vec(1,ii) + temp(:,:,2)*diff_vec(2,ii) + temp(:,:,3)*diff_vec(3,ii);
            delta2_g = delta2_g + term_6th(1,ii)*(diag([1 1 1])...
                - eval_normal*eval_normal'/norm(eval_normal)^2)*temp_2/norm(eval_normal);
        end

        temp_matrix = delta2_g;
        temp_matrix(:,4) = delta_g;
        temp_matrix(4,:) = [delta_g',0];

        output_gaussian(i,j) = - det(temp_matrix)/norm(delta_g)^4;
        output_mean(i,j) = (delta_g'*delta2_g*delta_g - norm(delta_g)^2*trace(delta2_g))/norm(delta_g)^3/2; 

    end
end


temp = output_mean.*output_mean - output_gaussian;
temp(find(temp < 0)) = 0;

output_k1 = output_mean + sqrt(temp);
output_k2 = output_mean - sqrt(temp);
