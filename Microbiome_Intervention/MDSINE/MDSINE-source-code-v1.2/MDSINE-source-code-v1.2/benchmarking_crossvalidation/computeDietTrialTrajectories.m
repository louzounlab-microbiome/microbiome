function [corr_lasso,corr_select,corr_L2,rms_lasso,rms_select,rms_L2] = computeDietTrialTrajectories(T,perturbations,trials,densities,intervene_matrix_filtered_merge,experimentBlocks)
% This file is a part of the MDSINE program.
%
% MDSINE: Microbial Dynamical Systems INference Engine
% Infers dynamical systems models from microbiome time-series datasets, and
% predicts biologically relevant behaviors of the ecosystems.
% Copyright (C) 2015 Vanni Bucci, Georg Gerber
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


Y0 = cell(7,1);
for s=1:7,
    Y0{s} = densities{s}(3,:);
end;

numSubjs = length(trials);

corr_lasso = zeros(1,numSubjs);
corr_select = zeros(1,numSubjs);
corr_L2 = zeros(1,numSubjs);

rms_lasso = zeros(1,numSubjs);
rms_select = zeros(1,numSubjs);
rms_L2 = zeros(1,numSubjs);

cve = find(experimentBlocks == 1);

% matlabpool('open',5);
for s=1:length(cve),
%for s=1:length(cve),
    [corr_lasso(s),corr_select(s),corr_L2(s),rms_lasso(s),rms_select(s),rms_L2(s)] = getCorrelations(cve,s,densities,T,perturbations,Y0,trials{s},experimentBlocks,intervene_matrix_filtered_merge);
end;
% matlabpool close;

function [cc_lasso,cc_select,cc_L2,rms_lasso,rms_select,rms_L2] = getCorrelations(cve,sx,concentrations,T,perturbations,Y0,trial,experimentBlocks,intervene_matrix_filtered_merge)
warning off;

s = cve(sx);
ex = find(experimentBlocks == experimentBlocks(s));
e = find(ex == s);

disp(sprintf('Trial %i lasso trajectories',s));
[traj_lasso,traj_high,traj_low,percentSucceed] = numIntTrajectoriesFromSamples(T,perturbations,experimentBlocks,intervene_matrix_filtered_merge,Y0,trial.med_biomass,trial.med_counts,trial.Theta_samples_lasso,[3 3],[65 29],[0.1 0.1],8,1);
%[cc_lasso,rms_lasso]=computeAvgTrajectoryCorrDiet(concentrations{s},traj_lasso{s},T{1},3:0.1:65,T{1}(4:56),[1 7 11 12 13]);
[cc_lasso,rms_lasso]=computeAvgTrajectoryCorrDiet(concentrations{s},traj_lasso{s},T{1},3:0.1:65,T{1}(4:56),1:13);
%cc_lasso = 0;
%rms_lasso = 0;

disp(sprintf('Trial %i variable select trajectories',s));
[traj_select,traj_high,traj_low,percentSucceed] = numIntTrajectoriesFromSamples(T,perturbations,experimentBlocks,intervene_matrix_filtered_merge,Y0,trial.med_biomass,trial.med_counts,trial.Theta_samples_select,[3 3],[65 29],[0.1 0.1],20,1);
%[cc_select,rms_select]=computeAvgTrajectoryCorrDiet(concentrations{s},traj_select{s},T{1},3:0.1:65,T{1}(4:56),[1 7 11 12 13]);
[cc_select,rms_select]=computeAvgTrajectoryCorrDiet(concentrations{s},traj_select{s},T{1},3:0.1:65,T{1}(4:56),1:13);
%cc_select = 0;
%rms_select = 0;

disp(sprintf('Trial %i L2 trajectories',s));
Theta_L2 = cell(1,1);
Theta_L2{1} = trial.Theta_L2;
[traj_L2,traj_high,traj_low,percentSucceed] = numIntTrajectoriesFromSamples(T,perturbations,experimentBlocks,intervene_matrix_filtered_merge,Y0,0,1,Theta_L2,[3 3],[65 29],[0.1 0.1],1,1);
%[cc_L2,rms_L2]=computeAvgTrajectoryCorrDiet(concentrations{s},traj_L2{s},T{1},3:0.1:65,T{1}(4:56),[1 7 11 12 13]);
[cc_L2,rms_L2]=computeAvgTrajectoryCorrDiet(concentrations{s},traj_L2{s},T{1},3:0.1:65,T{1}(4:56),1:13);
%cc_L2 = 0;
%rms_L2 = 0;
