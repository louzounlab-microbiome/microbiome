function [ ] = infer_ridge(input, cfg, isConstrained )
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

%INFER_RIDGE_CONSTRAINED Summary of this function goes here
%   Detailed explanation goes here

% Average the biomass for ridge regression
% NOTE: Assumes length(BMD{i})/length(T{i}) = number of samples per time point

biomass_av = cell(length(input.biomass), 1);
for i=1:length(input.biomass)
    per_T = length(input.biomass{i})/length(input.T{i});
    biomass_av{i} = zeros(length(input.T{i}), 1);
    for t=1:length(input.T{i})
        biomass_av{i}(t) = mean(input.biomass{i}((t-1)*per_T+1:t*per_T));
    end
end

params = cfg.ridgeRegression;

if ~isdeployed()
    addpath('ridge_regression')
end

mesh = logspace(params.min, params.max, params.N);

[F,time,ID,U,taxa_data,time_data,ID_data,U_data,F_prime,X, magnitude]=...
    import_data_and_construct_matrices(input.counts, biomass_av, ...
    input.perturbations, input.T, params.normalize_counts, ...
    params.differentiation, params.scaling_factor, []);

% note mix_trajectories is 'notted' because originally coded as
% "keep all trajectories", and mix made more sense to explain in a config
params.cores = cfg.parallel.cores;
if params.cores == 1
    params.cores = [];  % don't need to set up parpool for 1 worker.
end

[regularizer_global,Theta_global,Theta_mean]=...
    inf_regularization_based_inference(cfg.general.seed, F, time, ID, ...
    U, F_prime, X, mesh, params.k, ~(params.mix_trajectories), ...
    isConstrained, 'qp', true, params.cores, params.replicates, magnitude);

% for compatibility with the MCMC sample numerical integration
T = input.T;
perturbations = input.perturbations;
experimentBlocks = input.blocks;
BMD = input.biomass;
intervene_matrix = input.intervene;
intervene_matrix_filtered_merge = input.intervene;
keep_species = {1:length(input.species_names)};
Theta_samples_global = {Theta_global};
med_biomass = 0;
med_counts = 10^magnitude;
Theta_bayes_factors = [];

output_name = [cfg.general.output_dir cfg.general.algorithm '.mat'];
save(output_name, '-v7.3')

if ~isdeployed()
    rmpath('ridge_regression')
end

return
end

