function MLS_demo
% Demo for direct computing of point-set surface curvatures
%
% Authors : Pinghai Yang
% Created : Sep. 29, 2007
% Version : 1.0
% Matlab Version : 7.0 ( Windows )
%
% Copyright (c) 2007 CDM Lab, Illinois Institute of Technology , U.S.A.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

number_of_neighbor = 90;
gaussian_param = 0.06;

% read the input synthetic/real point data and normals
file_name_for_input = 'demo_input_data.txt';

fp = fopen(file_name_for_input, 'r');
points = fscanf(fp,'%f');
bluff_data = (reshape(points,6,size(points,1)/6))';
fclose(fp);

x = bluff_data(:,1);
y = bluff_data(:,2);
z = bluff_data(:,3);
normal_x = bluff_data(:,4);
normal_y = bluff_data(:,5);
normal_z = bluff_data(:,6);

input_pnts = [x';y';z'];

input_normals = [normal_x';normal_y';normal_z'];

% read the evaluation point data
file_name_for_evaluation = 'demo_evaluation_data.txt';

fp = fopen(file_name_for_evaluation, 'r');
points = fscanf(fp,'%f');
bluff_data = (reshape(points,3,size(points,1)/3))';
fclose(fp);

x = bluff_data(:,1);
y = bluff_data(:,2);
z = bluff_data(:,3);

eval_pnts = [x';y';z'];

% Project a point onto the MLS surface defined by input points  
for i = size(eval_pnts,2),
    eval_pnts(:,i) = MLS_projection(input_pnts, input_normals, eval_pnts(:,i), number_of_neighbor, gaussian_param);
end

% Direct computing of point-set surface curvatures  
[gaussian, mean, k1, k2] = MLS_curvature_computing(input_pnts, input_normals, eval_pnts, number_of_neighbor, gaussian_param);

% save the resulting curvatures
file_name_for_evaluation = 'demo_resulting_curvatures.txt';

fp = fopen('demo_resulting_curvatures.txt', 'w+');
for i = 1:size(eval_pnts,2)
    fprintf(fp,'%f %f %f %f\n', gaussian(i), mean(i), k1(i), k2(i));
end
fclose(fp);

