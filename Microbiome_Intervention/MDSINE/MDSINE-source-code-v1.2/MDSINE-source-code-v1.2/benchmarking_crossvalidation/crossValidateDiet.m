function [trials] = crossValidateDiet(cfg,T,perturbations,intervene_matrix,data,species_names,BMD,experimentBlocks)
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


%paramsDir = 'params_perturb/';

%preProcessDataParmsFileName = [paramsDir 'preProcessDataParams.txt'];

% find first index for each experimental block
uniqueExperiments = unique(experimentBlocks);
blockIdx = zeros(length(uniqueExperiments),1);
for e=1:length(blockIdx),
    f = find(experimentBlocks == uniqueExperiments(e));
    blockIdx(e) = f(1);
end;

cve = find(experimentBlocks == 1);

numSubjs = length(cve);
trials = cell(numSubjs,1);

% filter out OTUs with low counts
[keep_species,intervene_matrix_filtered,intervene_matrix_filtered_merge,data_counts_filtered,species_names_filtered] = filterData(intervene_matrix,data,species_names,experimentBlocks,cfg.preprocessing);

%matlabpool('open',5);
for sx=1:length(cve),
%for s=1:length(cve),
    trials{sx} = doOneHoldout(cfg,sx,cve,T,perturbations,BMD,data,species_names,experimentBlocks,keep_species,intervene_matrix_filtered,intervene_matrix_filtered_merge,data_counts_filtered);
end;
%matlabpool close;

function [trial] = doOneHoldout(cfg,sx,cve,T,perturbations,BMD,data_counts,species_names,experimentBlocks,keep_species,intervene_matrix_filtered,intervene_matrix_filtered_merge,data_counts_filtered)

    s = cve(sx);
    disp(sprintf('Holdout %i',s));
    BMD_train = BMD;
    BMD_train(s) = [];
    data_counts_train = data_counts;
    data_counts_train(s) = [];
    experimentBlocks_train = experimentBlocks;
    experimentBlocks_train(s) = [];

    intervene_matrix_filtered_merge_train = intervene_matrix_filtered_merge;
    intervene_matrix_filtered_merge_train(s) = [];
    perturbations_train = perturbations;
    perturbations_train(s) = [];
    T_train = T;
    T_train(s) = [];

    ex = find(experimentBlocks == experimentBlocks(s));
    e = find(ex == s);
    data_counts_filtered_train = data_counts_filtered;
    data_counts_filtered_train{experimentBlocks(s)}(e,:) = [];

    Theta_samples_lasso = [];
    Theta_samples_select = [];
    med_counts = 0;
    med_biomass = 0;

    try
        [Theta_samples_lasso,Theta_samples_select,med_counts,med_biomass] = benchmarkDietDataBayes(cfg,T_train,perturbations_train,data_counts_train,species_names,BMD_train,experimentBlocks_train,keep_species,intervene_matrix_filtered,intervene_matrix_filtered_merge_train,data_counts_filtered_train);
    catch errM
    end

    [Theta_global_L2] = benchmarkDietDataL2Opt(cfg, T_train,perturbations_train,BMD_train,keep_species,data_counts_train);

    trial = struct('med_counts',med_counts,'med_biomass',med_biomass,'Theta_samples_lasso',{Theta_samples_lasso},'Theta_samples_select',{Theta_samples_select},'Theta_L2',Theta_global_L2);
