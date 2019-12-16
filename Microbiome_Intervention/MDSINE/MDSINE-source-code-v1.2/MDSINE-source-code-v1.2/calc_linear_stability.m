function [ ] = calc_linear_stability( cfg )
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

if ~isdeployed()
    addpath('linear_stability')
end

% setup parallelization. TODO: make function
if cfg.parallel.cores > 1
    poolobj = gcp('nocreate'); % If no pool, do not create new one.
    if isempty(poolobj)
        poolsize = 0;
    else
        poolsize = poolobj.NumWorkers;
    end
    if  poolsize == 0
        % comment this out if you do not have the parallel computing
        % toolbox
        parpool('local', cfg.parallel.cores);
    end
else
    poolobj = gcp('nocreate');
end


inference_filename = [cfg.general.output_dir cfg.general.algorithm '.mat'];
inf = load(inference_filename);

stability_dir = [cfg.general.output_dir cfg.general.algorithm '_stability/'];
if ~exist(stability_dir, 'dir')
    mkdir(stability_dir)
end
stability_prefix = [stability_dir cfg.general.algorithm];

Thetas = inf.Theta_samples_global(1:cfg.linearStability.sample_step:end);
num_presence_profiles = size(Thetas{1},1);
% number of perturbations = columns - 1(growth) - rows(for interactions)
num_perts = size(Thetas{1}, 2) - size(Thetas{1}, 1) - 1;

if cfg.parallel.cores > 1
    parfor pp_count = 1:2^num_presence_profiles-1
        presence_profile=dec2bin(pp_count, num_presence_profiles)-'0';
        perform_linstability_analysis(presence_profile, pp_count, ...
            Thetas, num_perts, stability_prefix);
    end
else
    for pp_count = 1:2^num_presence_profiles-1
        presence_profile=dec2bin(pp_count, num_presence_profiles)-'0';
        perform_linstability_analysis(presence_profile, pp_count, ...
            Thetas, num_perts, stability_prefix);
    end
end

if ~isdeployed()
    rmpath('linear_stability')
end

end

function [] = perform_linstability_analysis(presence_profile, pp_count, Thetas, ...
    num_p, stability_prefix)
N_mcmc = length(Thetas);
alpha = cell(N_mcmc, 1 + num_p);
beta = cell(N_mcmc, 1);
steady_states = cell(N_mcmc, 1 + num_p);
eigenvalues = cell(N_mcmc, 1 + num_p);
stable_samples = nan(size(Thetas{1}, 1), 1 + num_p, N_mcmc); % for later nanmedian
stable_count = zeros(1, 1 + num_p);
for sample=1:N_mcmc
    beta{sample} = Thetas{sample}(:,2:end-num_p);
    for pert = 1:(1+num_p)
        if pert > 1
            alpha{sample, pert} = alpha{sample, 1} ...
                + Thetas{sample}(:, end-(num_p+1) + pert); %perturbation column
        else
            alpha{sample, pert} = Thetas{sample}(:, pert);
        end
        [eigenvalues{sample, pert}, steady_states{sample, pert}, ~, ~]= ...
            linstability_get_steadystates_and_eigenvalues ...
            (presence_profile,beta{sample},alpha{sample, pert});
        if (all(nonzeros(steady_states{sample, pert}) > 0) && ...
                all(real(eigenvalues{sample, pert}) < 0))
            stable_count(pert) = stable_count(pert) + 1;
            stable_samples(:, pert, sample) = steady_states{sample, pert};
        end
    end
end

for i=1:num_p + 1
    median_stable = nanmedian(stable_samples, 3);
    frequency_of_stability = stable_count / N_mcmc;
%     median_stable_states = nanmedian(cell2mat(stable_samples'), 2);
end

fname = [stability_prefix '_' int2str(pp_count) '_ls.mat'];
clearvars -except presence_profile Thetas steady_states ...
     median_stable frequency_of_stability eigenvalues fname
save(fname, '-v7.3');
end
