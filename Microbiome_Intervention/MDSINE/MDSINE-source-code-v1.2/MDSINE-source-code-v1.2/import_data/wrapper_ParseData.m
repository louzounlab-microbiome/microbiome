function [BMD_raw, counts_data, intervene_matrix, perturbations, species_names, T, experimentBlocks] = wrapper_ParseData(metadata_file, counts_raw, biomass_raw, counts_dataFormat)
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


[all_data] = ParseData(metadata_file, counts_raw, biomass_raw, counts_dataFormat);

%BMD_raw_CFU = all_data.biomass_data.BMD_raw_CFU;
%BMD_raw_CT = all_data.biomass_data.BMD_raw_CT;
%BMD_raw = all_data.biomass_data.BMD_raw;

fn = fieldnames(all_data.biomass_data);
BMD_raw = cell(length(fn),1);
for i=1:length(fn)
    BMD_raw{i} = getfield(all_data.biomass_data, fn{i});
end

fn = fieldnames(all_data.counts_data);
counts_data = cell(length(fn),1);
tot_dims = size(getfield(all_data.counts_data, fn{1}));
max_num_measurements = tot_dims(1);
numOTUs = tot_dims(2);

for i=1:length(fn)
    counts_data{i} = getfield(all_data.counts_data, fn{i});
end


fn = fieldnames(all_data.metadata.perturbations);
perturbations = cell(length(fn),1);
ptb_sum = 0;
for i=1:length(fn)
    perturbations{i} = getfield(all_data.metadata.perturbations, fn{i});
%    ptbs = perturbations{i}(perturbations{i}~=-1);
    ptb_sum = ptb_sum + sum(perturbations{i});
end
if(ptb_sum==0) %somewhat roundabout...
   perturbations = [];
end

numOTUs = size(all_data.metadata.species_names,1);
species_names = cell(numOTUs,1);
for i=1:numOTUs
    species_names{i} = strtrim(all_data.metadata.species_names(i,:));
end


fn = fieldnames(all_data.metadata.intervene_matrices);
intervene_matrix = cell(length(fn),1);
for e=1:length(fn)
    intervene_matrix{e} = getfield(all_data.metadata.intervene_matrices, fn{e});
end

experimentBlocks = all_data.metadata.subj_blockID;

fn = fieldnames(all_data.metadata.T);
T = cell(length(fn),1);
for e=1:length(fn)
    T{e} = getfield(all_data.metadata.T, fn{e});
end



end
