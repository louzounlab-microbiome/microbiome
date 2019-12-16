function [ ] = run_cdiff_crossvalidation( )
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
    addpath('import_data')
end

config = formatconfig('data_cdiff/crossvalidation_parameters.cfg');
[BMD, counts, intervene, ~, species_names, T, experimentBlocks] = ...
    wrapper_ParseData(config.general.metadata_file, config.general.counts_file, config.general.biomass_file, 2);

% set seed
if isnan(config.general.seed)
    rng('shuffle')
else
    rng(config.general.seed)
end


if ~isdeployed()
    rmpath('import_data')
end

if ~isdeployed()
    addpath('benchmarking_crossvalidation')
    addpath('bayesian_inference')
    addpath('ridge_regression')
    addpath('simulation')
end

[trials] = crossValidateCdiff(config, T, intervene, counts, ...
    species_names, BMD, experimentBlocks);

[corr_lasso,corr_select,corr_L2,rms_lasso,rms_select,rms_L2] = ...
    computeCdiffTrialTrajectories(T, trials, counts, BMD, intervene, ...
    intervene_matrix_filtered_merge,keep_species,experimentBlocks);

mkdir(config.general.output_dir);
save([config.general.output_dir 'cdiff_crossvalidation.mat'], '-v7.3')

outf = [config.general.output_dir 'cdiff_crossvalidation_rmse.txt'];
fid = fopen(outf, 'w');
fprintf(fid, 'algorithm\tholdout\trmse\ttrajectory_correlation\n');
for i=1:numel(corr_lasso)
    fprintf(fid, 'lasso\t%d\t%6.4f\t%6.4f\n', i, rms_lasso(i), corr_lasso(i));
end
for i=1:numel(corr_select)
    fprintf(fid, 'select\t%d\t%6.4f\t%6.4f\n', i, rms_select(i), corr_select(i));
end
for i=1:numel(corr_L2)
    fprintf(fid, 'L2\t%d\t%6.4f\t%6.4f\n', i, rms_L2(i), corr_L2(i));
end
fclose(fid);

end

