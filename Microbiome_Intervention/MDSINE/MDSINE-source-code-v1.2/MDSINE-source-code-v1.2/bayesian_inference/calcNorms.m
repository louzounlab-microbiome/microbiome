function [total_count_norm,biomass_norm,scaleFactor,med_counts,med_biomass] = calcNorms(keep_species,data_counts,intervene_matrix_filtered,biomass,experimentBlocks)
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

% return normalizing constants

uniqueExperiments = unique(experimentBlocks);
blockIdx = zeros(length(uniqueExperiments),1);
for e=1:length(blockIdx),
    f = find(experimentBlocks == uniqueExperiments(e));
    blockIdx(e) = f(1);
end;

total_count_norm = cell(length(uniqueExperiments),1);
biomass_norm = cell(length(uniqueExperiments),1);
scaleFactor = cell(length(uniqueExperiments),1);

med_counts = [];
med_biomass = [];
for e=1:length(uniqueExperiments),
    ep = find(uniqueExperiments(e) == experimentBlocks);

    % calculate normalizing factor for each experiment based on total
    % sequencing counts for each experiment
    [total_count_norm{e},mc] = calcTotalCountNorm(data_counts(ep),keep_species{e});
    med_counts = [med_counts ; mc];

    % calculate normalizing factor for each experiment based on bacterial
    % biomass
    [biomass_norm{e},mb] = calcBiomassNorm(biomass(ep));
    med_biomass = [med_biomass ; mb];
end;

med_counts = median(med_counts);
med_biomass = median(med_biomass);

for e=1:length(uniqueExperiments),
    for s=1:length(total_count_norm{e}),
        total_count_norm{e}{s} = total_count_norm{e}{s} - log(med_counts);
        biomass_norm{e}{s} = biomass_norm{e}{s} - med_biomass;
    end;

    ep = find(uniqueExperiments(e) == experimentBlocks);
    % calculate scaling factor for each OTU (helps to keep all OTUs on the same
    % scale, so fewer numerical issues)
    [scaleFactor{e}] = calcScaleFactor(keep_species{e},data_counts(ep),total_count_norm{e},biomass_norm{e},intervene_matrix_filtered{e});
end;

function [total_count_norm,med_counts] = calcTotalCountNorm(data_counts,keep_species)
% return normalizing factor for each mouse in each time-point

% set up to use partial sum over quantiles, but now just using all quantiles
quantile = 1.0;

numSubjects = length(data_counts);

total_count_norm = cell(numSubjects,1);

med_counts = [];
for m=1:numSubjects,
    nc = data_counts{m}(:,keep_species);
    nc = sort(nc,2);
    pr = round(quantile*size(nc,2));
    nc = sum(nc(:,1:pr),2);

    % calculate total counts at each time-point
    med_counts = [med_counts ; nc];
    total_count_norm{m} = nc;
end;

%med_counts = median(med_counts);

for m=1:numSubjects,
    %total_count_norm{m} = log(total_count_norm{m}) - log(med_counts);
    total_count_norm{m} = log(total_count_norm{m});
end;

function [biomass_norm,med_biomass] = calcBiomassNorm(biomass)
% return normalizing factor for each subject in each time-point

numSubjects = length(biomass);

biomass_norm = cell(numSubjects,1);

med_biomass = [];
for m=1:numSubjects,
    nc = biomass{m};
    % total biomass at each time-point
    med_biomass = [med_biomass ; nc];
    biomass_norm{m} = nc;
end;

%med_biomass = median(med_biomass);

%for m=1:numSubjects,
    %biomass_norm{m} = biomass_norm{m} - med_biomass;
%end;

function [scaleFactor] = calcScaleFactor(keep_species,data_counts,total_count_norm,biomass_norm,intervene_matrix_filtered)
% compute factor to center observations in log space (used as an offset
% in the GLM model)

numOTUs = length(keep_species);
numSubjects = length(data_counts);

scaleFactor = cell(numSubjects,numOTUs);

for s=1:numSubjects,
    for ox=1:numOTUs,
        o = keep_species(ox);
        mc = log(data_counts{s}(:,o)+1);
        f = find(intervene_matrix_filtered(:,ox) > 0);
        mc = mc(f) - (total_count_norm{s}(f) - biomass_norm{s}(f));
        scaleFactor{s,ox} = mean(mc);
    end;
end;
