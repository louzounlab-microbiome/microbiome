function [corr_lasso,corr_select,corr_L2,rms_lasso,rms_select,rms_L2] = computeCdiffTrialTrajectories(T,trials,data_counts,BMD,intervene_matrix,intervene_matrix_filtered_merge,keep_species,experimentBlocks)
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


[concentrations] = calcConcentrations(T,data_counts,BMD,intervene_matrix,3,keep_species);
[Y0] = buildInitialConditionVectorsFromData(T,30*ones(5,1),concentrations);

numSubjs = length(trials);

corr_lasso = zeros(1,numSubjs);
corr_select = zeros(1,numSubjs);
corr_L2 = zeros(1,numSubjs);

rms_lasso = zeros(1,numSubjs);
rms_select = zeros(1,numSubjs);
rms_L2 = zeros(1,numSubjs);

% matlabpool('open',5);
for s=1:numSubjs,
%for s=1:numSubjs,
    [corr_lasso(s),corr_select(s),corr_L2(s),rms_lasso(s),rms_select(s),rms_L2(s)] = getCorrelations(s,concentrations,T,Y0,trials{s},experimentBlocks,intervene_matrix_filtered_merge);
end;
% matlabpool close;

function [cc_lasso,cc_select,cc_L2,rms_lasso,rms_select,rms_L2] = getCorrelations(s,concentrations,T,Y0,trial,experimentBlocks,intervene_matrix_filtered_merge)
warning off;

disp(sprintf('Trial %i lasso trajectories',s));
[traj_lasso,traj_high,traj_low,percentSucceed] = numIntTrajectoriesFromSamples(T,[],experimentBlocks,intervene_matrix_filtered_merge,Y0,trial.med_biomass,trial.med_counts,trial.Theta_samples_lasso,30,56,0.1,8,1);
[cc_lasso,rms_lasso]=computeAvgTrajectoryCorrCdiff(concentrations(s,:),traj_lasso{s},T{1},30:0.1:56,T{1}(17:26),2);

disp(sprintf('Trial %i variable select trajectories',s));
[traj_select,traj_high,traj_low,percentSucceed] = numIntTrajectoriesFromSamples(T,[],experimentBlocks,intervene_matrix_filtered_merge,Y0,trial.med_biomass,trial.med_counts,trial.Theta_samples_select,30,56,0.1,20,1);
[cc_select,rms_select]=computeAvgTrajectoryCorrCdiff(concentrations(s,:),traj_select{s},T{1},30:0.1:56,T{1}(17:26),2);

disp(sprintf('Trial %i L2 trajectories',s));
Theta_L2 = cell(1,1);
Theta_L2{1} = trial.Theta_L2;
[traj_L2,traj_high,traj_low,percentSucceed] = numIntTrajectoriesFromSamples(T,[],experimentBlocks,intervene_matrix_filtered_merge,Y0,0,1,Theta_L2,30,56,0.1,1,1);
[cc_L2,rms_L2]=computeAvgTrajectoryCorrCdiff(concentrations(s,:),traj_L2{s},T{1},30:0.1:56,T{1}(17:26),2);
