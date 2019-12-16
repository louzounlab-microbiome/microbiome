function [keep_species_merge,species_names_filtered_merge] = mergeData(experimentBlocks,keep_species,data_counts,species_names,intervene_matrix_filtered)
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

% merge OTU data and associated representations

% find first index for each experimental block
uniqueExperiments = unique(experimentBlocks);
blockIdx = zeros(length(uniqueExperiments),1);
for e=1:length(blockIdx),
    bf = find(experimentBlocks == uniqueExperiments(e));
    blockIdx(e) = bf(1);
end;

keep_species_merge = [];
for e=1:length(blockIdx),
    keep_species_merge = union(keep_species_merge,keep_species{e});
end;
species_names_filtered_merge = species_names(keep_species_merge);

no_keep = setdiff(1:size(data_counts{1},2),keep_species_merge);
ivmf = cell(length(experimentBlocks),1);
for e=1:length(blockIdx),
    ep = find(experimentBlocks == uniqueExperiments(e));
    for s=1:length(ep),
        ivt = zeros(size(data_counts{ep(s)},1),size(data_counts{1},2));
        ivt(:,keep_species{e}) = intervene_matrix_filtered{e};

        ft(:,no_keep) = [];
        ft_deriv(:,no_keep) = [];
        ivt(:,no_keep) = [];

        f{ep(s)} = ft;
        f_deriv{ep(s)} = ft_deriv;
        ivmf{ep(s)} = ivt;
    end;
end;
intervene_matrix_filtered = ivmf;

