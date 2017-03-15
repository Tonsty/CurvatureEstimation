function [output_pnt] = MLS_projection(input_pnts, input_normals, eval_pnt, number_of_neighbor, gaussian_param)
% Project a point onto the MLS surface defined by input points  
%
% [out_a] = MLS_curvature_computing (in_1 , in_2 , in_3 , in_4 , in_5)
%
% Parameters [IN ]:
% in_1 : input synthetic/real point data, which will define an MLS surface
% in_2 : corresponding normals of the input synthetic/real point data
% in_3 : input original point
% in_4 : number of neighbor points, which will contribute to the curvature
%        computing at each evaluation position
% in_5 : Gaussian scale parameter that determines the width of the Gaussian
%        kernel
%
% Returns [ OUT ]:
% out_a : output projected point
%
%
% Authors : Pinghai Yang
% Created : Sep. 25, 2007
% Version : 1.0
% Matlab Version : 7.0 ( Windows )
%
% Copyright (c) 2007 CDM Lab, Illinois Institute of Technology , U.S.A.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ERR = 1e-5;

source = eval_pnt;
old_source = source + 1;

while norm(source-old_source)>ERR,
    neighbor_index = nearestneighbour(source, input_pnts, 'NumberOfNeighbours', number_of_neighbor);
    neighbor_pnts = input_pnts(:,neighbor_index);
    neighbor_normals = input_normals(:,neighbor_index);

    diff_vec = neighbor_pnts - repmat(source, 1, number_of_neighbor);
    dist_squared = diff_vec(1,:).*diff_vec(1,:) + diff_vec(2,:).*diff_vec(2,:) + diff_vec(3,:).*diff_vec(3,:);
    
    weight = exp(- dist_squared / gaussian_param^2);
    bound = max(dist_squared);

    normal = [sum(neighbor_normals(1,:).*weight);sum(neighbor_normals(2,:).*weight);sum(neighbor_normals(3,:).*weight)];

    normal = normal/norm(normal);

    x = fminbnd(@(t) MLS_energy(t, number_of_neighbor, gaussian_param, source, normal, neighbor_pnts, weight),...
        -bound, bound,optimset('Display','off'));
    
    old_source = source;
    source = source + x*normal;
end

output_pnt = source;

