function [] = mdsine(config_filename)
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

% main
% <The authors>
% for info:
% Vanni Bucci, Ph.D. <vbucci@umassd.edu>
% Goerg Gerber, Ph.D. <vbucci@umassd.edu>

%% Step 0: check for crossvalidation. Temporary!

if strcmp(config_filename, 'cdiff_crossvalidation')
    run_cdiff_crossvalidation()
    return
end
if strcmp(config_filename, 'diet_crossvalidation')
    run_diet_crossvalidation()
    return
end
%% --------- Step 1: Read in configuration and input ----------------------
% The configuration file config.txt contains all the information needed to
% run the framework. It specifies location of the data and all the options
% corresponding to each of the implemented inference algorithms
% It also contains run control information for execution of dynamical
% simulations and combinatorial linear stability analysis calculation.
if ~isdeployed()
    addpath('import_data')
end

config = formatconfig(config_filename);
[BMD, counts, intervene, perturbations, species_names, T, experimentBlocks] = ...
    wrapper_ParseData(config.general.metadata_file, config.general.counts_file, config.general.biomass_file, 2);

% set seed
if isnan(config.general.seed)
    rng('shuffle');
    rngState = rng;
    deltaSeed = uint32(feature('getpid'));
    uSeed = rngState.Seed + deltaSeed;
    rng(uSeed);
else
    rng(config.general.seed)
end

% this might need processing?
blocks = experimentBlocks;

% for convenience
input.biomass = BMD;
input.counts = counts;
input.intervene = intervene;
input.perturbations = perturbations;
input.species_names = species_names;
input.T = T;
input.blocks = blocks;

clearvars -except input config

mkdir(config.general.output_dir)
save([config.general.output_dir 'input.mat'], '-v7.3')

if ~isdeployed()
    rmpath('import_data')
end


%% --------- Step 2: Infer model parameters -------------------------------
% The configuration file determines what algorithm is used in the glv
% parameters inference step.

if config.general.run_inference == true
    disp(['Starting ' config.general.algorithm ' inference...'])
    switch config.general.algorithm
        case 'MLRR'
            infer_ridge(input, config, false)

        case 'MLCRR'
            infer_ridge(input, config, true)

        case 'BAL'
            infer_bayesian_lasso(input, config)

        case 'BVS'
            infer_bayesian_selection(input, config)

        otherwise
            error('Algorithm selection does not exist');
    end
end

%% --------- Step 3: Perform numerical simulations ------------------------

if config.general.run_simulations == true
    disp('Simulating trajectories...')
    calc_simulation(config)
end



%% --------- 4: Perform linear stability analysis -------------------------

if config.general.run_linear_stability == true
    disp('Calculating linear stability of all presence profiles...')
    calc_linear_stability(config)
end



%% --------- 5: Perform post processing -----------------------------------

if config.general.run_post_processing == true
    disp('Performing post processing...')
    post_processing(config)
    disp(['Output written to ' config.general.output_dir])
end

end
