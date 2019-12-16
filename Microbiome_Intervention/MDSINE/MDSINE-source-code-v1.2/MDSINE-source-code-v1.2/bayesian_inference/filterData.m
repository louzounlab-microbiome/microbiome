function [keep_species,intervene_matrix_filtered,intervene_matrix_filtered_merge,data_counts_filtered,species_names_filtered] = filterData(intervene_matrix,data_counts,species_names,experimentBlocks,params)
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

% filter out species that are low abundance across all subjects and time-points

%params = readParameterFile(paramFileName);

uniqueExperiments = unique(experimentBlocks);
blockIdx = zeros(length(uniqueExperiments),1);
for e=1:length(blockIdx),
    f = find(experimentBlocks == uniqueExperiments(e));
    blockIdx(e) = f(1);
end;

keep_species = cell(length(uniqueExperiments),1);
intervene_matrix_filtered = cell(length(uniqueExperiments),1);
intervene_matrix_filtered_merge = cell(length(uniqueExperiments),1);
data_counts_filtered = cell(length(uniqueExperiments),1);
species_names_filtered = cell(length(uniqueExperiments),1);

for e=1:length(uniqueExperiments),
    ep = find(uniqueExperiments(e) == experimentBlocks);
    % filter out species with low counts (must have median counts [over all time-points and subjects] > threshold)
    [keep_species{e},intervene_matrix_filtered{e},data_counts_filtered{e},species_names_filtered{e}] = filterData_sub(intervene_matrix{blockIdx(e)},data_counts(ep),species_names,params.minMedCount);
end;

keep_species_total = [];
for e=1:length(blockIdx),
    keep_species_total = union(keep_species_total,keep_species{e});
end;
no_keep = setdiff(1:size(data_counts{1},2),keep_species_total);
ivmf = cell(length(experimentBlocks),1);
for e=1:length(blockIdx),
    ep = find(experimentBlocks == uniqueExperiments(e));
    for s=1:length(ep),
        ivt = zeros(size(intervene_matrix_filtered{e},1),size(data_counts{1},2));
        ivt(:,keep_species{e}) = intervene_matrix_filtered{e};
        ivt(:,no_keep) = [];
        ivmf{ep(s)} = ivt;
    end;
end;
intervene_matrix_filtered_merge = ivmf;

function [keep_species,intervene_matrix_filtered,data_counts_filtered,species_names_filtered] = filterData_sub(intervene_matrix,data_counts,species_names,minMedCount)
% filter out species that are low abundance across all subjects and time-points

keep_species = [];

for s=1:size(intervene_matrix,2),
    tc = [];
    for m=1:length(data_counts),
        c = data_counts{m}(find(intervene_matrix(:,s)>0),s);
        tc = [tc ; c];
    end;

    if median(tc) >= minMedCount,
        keep_species = [keep_species ; s];
    end;
end;

intervene_matrix_filtered = intervene_matrix(:,keep_species);
numOTUs = size(intervene_matrix_filtered,2);

data_counts_filtered = cell(length(data_counts),1);
for m=1:length(data_counts),
    mct = data_counts{m}(:,keep_species);
    for o=1:numOTUs,
        f = find(intervene_matrix_filtered(:,o) > 0);
        data_counts_filtered{m,o} = mct(f,o);
    end;
end;

species_names_filtered = species_names(keep_species);

