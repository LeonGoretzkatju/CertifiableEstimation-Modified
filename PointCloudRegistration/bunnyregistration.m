%% outlier-rejection point cloud registration using FPFH feature and matching
%% Author: Xiangchen Liu
%% Date: Sep 19, 2023
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
bunnypcd = pcread("data\bun_zipper_res3.ply");
SourcePCD = pcdownsample(bunnypcd,"gridAverage",0.005);
bunnyxyz = bunnypcd.Location';
translationBound = 0.0;
noiseSigma       = 0.0;
R_gt                = rand_rotation;
t_gt                = randn(3,1);
t_gt                = t_gt/norm(t_gt); 
t_gt                = (translationBound) * rand * t_gt;
bunnyxyz_moving              = R_gt * bunnyxyz + t_gt;
target_bunnyxyz = pointCloud(bunnyxyz_moving');
TargetPCD = pcdownsample(target_bunnyxyz,"gridAverage",0.005);
% pcshowpair(bunnypcd,target_bunnyxyz);
[SourceFeature,SourceID] = extractFPFHFeatures(SourcePCD);
[TargetFeature,TargetID] = extractFPFHFeatures(TargetPCD);
[indexPairs,Score] = pcmatchfeatures(SourceFeature,TargetFeature,...
    SourcePCD,TargetPCD);
SourcematchedPts = select(SourcePCD,indexPairs(:,1));
TargetmatchedPts = select(TargetPCD,indexPairs(:,2));
pcshowMatchedFeatures(SourcePCD,TargetPCD,SourcematchedPts,TargetmatchedPts, ...
    "Method","montage")
title("Matched Points")

%% choose if run GNC for STRIDE
rungnc      = true;

%% generate random point cloud registration problem
if size(indexPairs,1) > 10
    problem.N = 10;
else
    problem.N = size(indexPairs,1);
end
problem.type        = 'point cloud registration';
problem.R_gt        = R_gt;
problem.t_gt        = t_gt;
noiseBoundSq        = noiseSigma^2 * chi2inv(0.99,3);
noiseBoundSq        = max(4e-2,noiseBoundSq); 
problem.noiseBoundSq= noiseBoundSq;
problem.noiseBound  = sqrt(problem.noiseBoundSq);
SourcePts = double(SourcematchedPts.Location');
problem.cloudA = SourcePts(:,1:problem.N);
TargetPts = double(TargetmatchedPts.Location');
problem.cloudB = TargetPts(:,1:problem.N);
problem.translationBound = translationBound;
%% generate SDP relaxation
addpath(genpath(spotpath))
SDP       = relax_point_cloud_registration_v4(problem,'checkMonomials',false);
fprintf('\n\n\n\n\n')
rmpath(genpath(spotpath))

%% Solve using STRIDE
% primal initialization using GNC
if rungnc
    solution = gnc_point_cloud_registration(problem);
    v        = lift_pcr_v4(solution.R_est(:),...
                           solution.t_est,...
                           solution.theta_est,...
                           problem.translationBound);
    X0       = rank_one_lift(v);

    gnc.R_err = getAngularError(problem.R_gt,solution.R_est);
    gnc.t_err = getTranslationError(problem.t_gt,solution.t_est);
    gnc.time  = solution.time_gnc;
    gnc.f_est = solution.f_est;
    gnc.info  = solution;
else
    X0       = [];
end
% Dual initialization using chordal SDP
addpath(genpath(spotpath))
chordalSDP       = chordal_relax_point_cloud_registration(problem);
fprintf('\n\n\n\n\n')
rmpath(genpath(spotpath))
prob = convert_sedumi2mosek(chordalSDP.sedumi.At,...
                            chordalSDP.sedumi.b,...
                            chordalSDP.sedumi.c,...
                            chordalSDP.sedumi.K);
addpath(genpath(mosekpath))
time0   = tic;
param.MSK_IPAR_INTPNT_MAX_ITERATIONS = 20;
[~,res] = mosekopt('minimize info',prob,param);
time_dualInit = toc(time0);
[~,~,Schordal,~] = recover_mosek_sol_blk(res,chordalSDP.blk);
S_assm           = pcr_dual_from_chordal_dual(Schordal);

% STRIDE main algorithm
addpath(genpath(stridepath))
addpath(genpath(manoptpath))

pgdopts.pgdStepSize     = 10;
pgdopts.SDPNALpath      = sdpnalpath;
pgdopts.maxiterPGD      = 5;
% ADMM parameters
pgdopts.tolADMM         = 1e-10;
pgdopts.maxiterADMM     = 1e4;
pgdopts.stopoptionADMM  = 0;

pgdopts.rrOpt           = 1:3;
pgdopts.rrFunName       = 'local_search_pcr_v4';
rrPar.blk = SDP.blk; rrPar.translationBound = problem.translationBound;
pgdopts.rrPar           = rrPar;
pgdopts.maxiterLBFGS    = 1000;
pgdopts.maxiterSGS      = 1000;
pgdopts.S0              = S_assm;

[outPGD,Xopt,yopt,Sopt]     = PGDSDP(SDP.blk, SDP.At, SDP.b, SDP.C, X0, pgdopts);
rmpath(genpath(manoptpath))

infostride              = get_performance_pcr(Xopt,yopt,Sopt,SDP,problem,stridepath);
infostride.totaltime    = outPGD.totaltime + time_dualInit;
infostride.time         = [outPGD.totaltime,time_dualInit];
if rungnc
    infostride.gnc = gnc; 
    infostride.totaltime = infostride.totaltime + gnc.time;
    infostride.time = [infostride.time, gnc.time];
end
fprintf('\n\n\n\n\n')
bunnyxyz_moving1              = infostride.R_est * bunnyxyz + infostride.t_est;
target_bunnyxyz1 = pointCloud(bunnyxyz_moving1');
TargetPCD1 = pcdownsample(target_bunnyxyz1,"gridAverage",0.005);
figure;
pcshowpair(TargetPCD1,TargetPCD);