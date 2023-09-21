%% outlier-rejection GNC point cloud registration using FPFH feature and matching
%% Author: Xiangchen Liu
%% Date: Sep 21, 2023
clc; clear; close all; restoredefaultpath
%% paths to dependencies
spotpath    = '../spotless';
stridepath  = '../STRIDE';
manoptpath  = '../manopt';
mosekpath   = '../mosek';
sdpnalpath  = '../SDPNAL+v1.0';
addpath('../utils')
addpath('./solvers')

%% read point cloud, find the keypoints and feature matches
bunnypcd = pcread("data\cloud_bin_0.ply");
SourcePCD = pcdownsample(bunnypcd,"gridAverage",0.05);
% SourcePCD = bunnypcd;
bunnyxyz = bunnypcd.Location';
translationBound = 1.0;
noiseSigma       = 0.05;
% R_gt                = rand_rotation;
% t_gt                = randn(3,1);
% t_gt                = t_gt/norm(t_gt); 
% t_gt                = (translationBound) * rand * t_gt;
% bunnyxyz_moving              = R_gt * bunnyxyz + t_gt;
% target_bunnyxyz = pointCloud(bunnyxyz_moving');
MovingPCD = pcread("data\cloud_bin_4.ply");
TargetPCD = pcdownsample(MovingPCD,"gridAverage",0.05);
% TargetPCD = target_bunnyxyz;
% pcshowpair(bunnypcd,target_bunnyxyz);
[SourceFeature,SourceID] = extractFPFHFeatures(SourcePCD);
[TargetFeature,TargetID] = extractFPFHFeatures(TargetPCD);
fixedValidPts = select(SourcePCD,SourceID);
movingValidPts = select(TargetPCD,TargetID);
[indexPairs,Score] = pcmatchfeatures(SourceFeature,TargetFeature,...
    fixedValidPts,movingValidPts);
SourcematchedPts = select(fixedValidPts,indexPairs(:,1));
TargetmatchedPts = select(movingValidPts,indexPairs(:,2));

figure
pcshowMatchedFeatures(TargetPCD,SourcePCD,TargetmatchedPts,SourcematchedPts, ...
    'Method','montage')
title('Matched Points')

%% essential parameters setup for GNC
problem.N = size(indexPairs,1);
problem.cloudA = double(SourcematchedPts.Location');
problem.cloudB = double(TargetmatchedPts.Location');
% problem.R_gt        = R_gt;
% problem.t_gt        = t_gt;
noiseBoundSq        = noiseSigma^2 * chi2inv(0.99,3);
noiseBoundSq        = max(4e-2,noiseBoundSq); 
problem.noiseBoundSq= noiseBoundSq;
problem.noiseBound  = sqrt(problem.noiseBoundSq);
% 
solution = gnc_point_cloud_registration(problem);
% gnc.R_err = getAngularError(problem.R_gt,solution.R_est);
% gnc.t_err = getTranslationError(problem.t_gt,solution.t_est);
% gnc.time  = solution.time_gnc;
% gnc.f_est = solution.f_est;
% gnc.info  = solution;
% % draw the registration result of the gnc pcr result
aligned_bunnyxyz = solution.R_est * SourcePCD.Location' + solution.t_est;
aligned_bunnyPCD = pointCloud(aligned_bunnyxyz');
[tform,~] = pcregistericp(aligned_bunnyPCD,TargetPCD,"Metric","pointToPlane");
refined_bunnyPCD = pctransform(aligned_bunnyPCD,tform);
figure;
pcshowpair(TargetPCD,refined_bunnyPCD);
title("GNC PCR result");
hold off;