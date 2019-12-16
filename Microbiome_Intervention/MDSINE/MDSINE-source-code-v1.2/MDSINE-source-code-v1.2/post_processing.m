function [ ] = post_processing( cfg )
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

%POST_PROCESSING Summary of this function goes here
%   Detailed explanation goes here
%
if ~isdeployed()
    addpath('post_processing')
end

load([cfg.general.output_dir 'input.mat']); %TODO
inference = load([cfg.general.output_dir cfg.general.algorithm '.mat']);
% input and cfg added to variable space

filename_prefix = [cfg.general.output_dir cfg.general.algorithm '.results'];

if ~isempty(inference.Theta_bayes_factors)
    infs = isinf(inference.Theta_bayes_factors);
    inference.Theta_bayes_factors(infs) = -Inf;
    bayes_max = max(max(inference.Theta_bayes_factors)); % 2 dims, 2 maxes
    inference.Theta_bayes_factors(infs) = 10 * bayes_max;
    bma = 10 * bayes_max * ones(size(inference.Theta_bayes_factors, 1), 1);
    inference.Theta_bayes_factors = [bma inference.Theta_bayes_factors];

    % need todo for perturbations?
end

[~,lixmax]=max(cellfun(@(x) numel(x),inference.keep_species));

shortnames = input.species_names(inference.keep_species{lixmax});
names = ['Growth'; shortnames];
if ~isempty(input.perturbations)
    for i=nonzeros(unique(cell2mat(input.perturbations)))
        names = [names; ['Perturbation' int2str(i)]];
    end
end

if cfg.postProcessing.write_parameters == true
    % output for plotting parameters (to be read by rscript plot_model_parameters.R)
    postprocess_write_file_for_parameters(filename_prefix, ...
        inference.Theta_global, inference.Theta_samples_global, ...
        inference.Theta_bayes_factors, names);
end

if cfg.postProcessing.write_cytoscape == true
    % output for cytoscape
    postprocess_write_file_for_cytoscape(filename_prefix, ...
        inference.Theta_global, inference.Theta_bayes_factors, names);
end

% load trajectories
sims = load([cfg.general.output_dir cfg.general.algorithm '_sims.mat']);

if cfg.postProcessing.write_trajectories == true
    % output for plotting trajectories (to be read by rscript plot_simulated_trajectories.R)
    postprocess_write_file_for_trajectories(filename_prefix, sims.traj, ...
        sims.traj_high, sims.traj_low, sims.traj_time, input.species_names, ...
        1, sims.concentrations, input.T);
end

stability_dir = [cfg.general.output_dir cfg.general.algorithm '_stability/'];
fnames = [stability_dir cfg.general.algorithm '_%s_ls.mat'];
% the %s is to be processed in the next function



Thetas = inference.Theta_samples_global(1:cfg.linearStability.sample_step:end);
num_presence_profiles = (2^(size(Thetas{1},1))-1);

N_perturbations = numel(unique(cell2mat(input.perturbations)));

if cfg.postProcessing.write_stability_analysis == true
    % output for plotting steady states and eigenvalues distributions (to be read by rscript plot_stability_analysis.R)
    postprocess_write_files_for_stability_analysis_output(filename_prefix, ...
        num_presence_profiles, fnames, input.species_names, ...
        cfg.postProcessing.keystone_cutoff, ...
        cfg.postProcessing.perform_keystone_analysis, N_perturbations)
end

if ~isdeployed()
    rmpath('post_processing')
end

end

