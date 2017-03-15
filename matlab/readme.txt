
*********************************************************************************
This directory contains the Matlab source code for computing surface curvatures from a point-set
surface. The input is a) a point cloud data with normal for each point, b) a set of points at 
which the curvatures need to computed. The output is a file containing Gaussian, Mean and principal
curvatures at these locations.

A demonstration file, MLS_demo.m, is provided to illustrate the curvature computing process.

The algorithm for the curvature computing is detailed in the following paper:
Yang, P. and Qian, X., "Direct computing of surface curvatures for point-set surfaces," 
Proceedings of 2007 IEEE/Eurographics Symposium on Point-based Graphics(PBG), Prague, Czech
Republic, Sep. 2007.

		- XQ. Sep 30, 2007.

*********************************************************************************



File Description:
MLS_curvature_computing.m: main function for curvature computing.
MLS_demo.m: a demo for direct computing of point-set surface curvatures.
MLS_energy.m: calculate the energy value of a given point. 
MLS_projection.m: project a point onto the MLS surface defined by input points. 
nearestneighbour.m: find nearest neighbour points for a given point.

Matlab Command: "MLS_demo".


Input:
1. demo_evaluation_data.txt: point location at which the curvature will be computed.
                     file format:
                     x-coordinate y-coordinate z-coordinate
                     ...

2. demo_input_data.txt: input point cloud data (synthetic/real point data and normals).
                     file format:
                     x-coordinate y-coordinate z-coordinate normal-x normal-y normal-z
                     ... ...
Output:
1. demo_resulting_curvatures.txt: curvatures at specified locations
                     file format:
                     GaussianCurvature MeanCurvature PrincipledCurvature1 PrincipledCurvature2
                     ...

