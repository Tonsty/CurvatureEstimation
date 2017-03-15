function f = MLS_energy(t, number_of_neighbor, gaussian_param, source, normal, neighbor_pnts, weight)
% Calculate the energy value of a given point 
%
% [out_a] = MLS_curvature_computing (in_1, in_2  in_3, in_4, in_5, in_6, in_7)
%
% Parameters [IN ]:
% in_1 : optimization variable
% in_2 : number of neighbor points, which will contribute to the energy
%        computing
% in_3 : Gaussian scale parameter that determines the width of the Gaussian
%        kernel
% in_4 : input position where to calculate the energy
% in_5 : corresponding normal of the input point
% in_6 : input neighbor points
% in_7 : corresponding gaussian weights
%
% Returns [ OUT ]:
% out_a : output energy value
%
% Authors : Pinghai Yang
% Created : Sep. 25, 2007
% Version : 1.0
% Matlab Version : 7.0 ( Windows )
%
% Copyright (c) 2007 CDM Lab, Illinois Institute of Technology , U.S.A.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

q = source + t*normal;
dist2plane_squared = (((repmat(q, 1, number_of_neighbor) - neighbor_pnts)'*normal).*...
    ((repmat(q, 1, number_of_neighbor) - neighbor_pnts)'*normal))';

f = sum(weight.*dist2plane_squared)/sum(weight);