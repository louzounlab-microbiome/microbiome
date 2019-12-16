function [  ] = calc_simulation( cfg )
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

%INFER_BAYESIAN_SELECTION Summary of this function goes here
%   Detailed explanation goes here
if ~isdeployed()
    addpath('simulation')
end

inference_filename = [cfg.general.output_dir cfg.general.algorithm '.mat'];
P = cfg.simulation;

load([cfg.general.output_dir 'input.mat']);
inference = load(inference_filename);

% calculate concentrations from counts data and raw biomass data
[concentrations, s] = calcConcentrations(inference.T, input.counts, inference.BMD, ...
    inference.intervene_matrix, cfg.preprocessing.numReplicates, ...
    inference.keep_species);

% start_times = P.start_time .* ones(length(out.T), 1);
start_times = P.start_time(input.blocks);

% build initial condition vectors at day 30 from concentrations
[Y0] = buildInitialConditionVectorsFromData(inference.T, start_times, ...
    concentrations);

% cdiff
[traj,traj_high,traj_low,percentSucceed] = ...
    numIntTrajectoriesFromSamples(inference.T, inference.perturbations, ...
    inference.experimentBlocks, inference.intervene_matrix_filtered_merge, Y0, ...
    inference.med_biomass, inference.med_counts, inference.Theta_samples_global, ...
    P.start_time, P.end_time, P.time_step, P.thin_rate, P.assume_stiff);

end_times = P.end_time(input.blocks);
delta_ts = P.time_step(input.blocks);

traj_time = cell(length(start_times), 1);
for i=1:length(traj_time)
    traj_time{i} = start_times(i):delta_ts(i):end_times(i);
end

clearvars -except cfg traj traj_high traj_low traj_time concentrations s percentSucceed

save([cfg.general.output_dir cfg.general.algorithm '_sims.mat'], '-v7.3')

% diet
% [traj_lasso,traj_lasso_high,traj_lasso_low,percentSucceed_lasso] =  ...
%     numIntTrajectoriesFromSamples(T, perturbations, experimentBlocks, ...
%     intervene_matrix_filtered_merge, Y0, 0, 1, A_samples_lasso, ...
%     [3 3],[65 29],[0.1 0.1],1,1);


if ~isdeployed()
    rmpath('simulation')
end

end
