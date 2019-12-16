function [keep_species,intervene_matrix_filtered,data_counts_filtered,species_names_filtered,total_count_norm,biomass_norm,scaleFactor,med_counts,med_biomass] = preProcessData(intervene_matrix,data_counts,species_names,biomass,experimentBlocks,uniqueExperiments,blockIdx,paramFileName)
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

% preprocess data and return normalizing constants

params = readParameterFile(paramFileName);

keep_species = cell(length(uniqueExperiments),1);
intervene_matrix_filtered = cell(length(uniqueExperiments),1);
data_counts_filtered = cell(length(uniqueExperiments),1);
species_names_filtered = cell(length(uniqueExperiments),1);
total_count_norm = cell(length(uniqueExperiments),1);
biomass_norm = cell(length(uniqueExperiments),1);
scaleFactor = cell(length(uniqueExperiments),1);

med_counts = [];
med_biomass = [];
for e=1:length(uniqueExperiments),
    ep = find(uniqueExperiments(e) == experimentBlocks);
    % filter out species with low counts (must have median counts [over all time-points and subjects] > threshold)
    [keep_species{e},intervene_matrix_filtered{e},data_counts_filtered{e},species_names_filtered{e}] = filterData(intervene_matrix{blockIdx(e)},data_counts(ep),species_names,params.minMedCount);

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
    [scaleFactor{e}] = calcScaleFactor(data_counts_filtered{e},total_count_norm{e},biomass_norm{e},intervene_matrix_filtered{e});
end;

function [keep_species,intervene_matrix_filtered,data_counts_filtered,species_names_filtered] = filterData(intervene_matrix,data_counts,species_names,minMedCount)
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

function [scaleFactor] = calcScaleFactor(data_counts_filtered,total_count_norm,biomass_norm,intervene_matrix_filtered)
% compute factor to center observations in log space (used as an offset
% in the GLM model)

numOTUs = size(data_counts_filtered,2);
numSubjects = size(data_counts_filtered,1);

scaleFactor = cell(numSubjects,numOTUs);

for s=1:numSubjects,
    for o=1:numOTUs,
        mc = log(data_counts_filtered{s,o}+1);
        f = find(intervene_matrix_filtered(:,o) > 0);
        mc = mc - (total_count_norm{s}(f) - biomass_norm{s}(f));
        scaleFactor{s,o} = mean(mc);
    end;
end;
