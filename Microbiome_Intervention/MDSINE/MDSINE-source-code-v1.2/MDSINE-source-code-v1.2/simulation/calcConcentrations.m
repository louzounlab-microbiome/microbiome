function [concentrations, factor] = calcConcentrations(T,data_counts,BMD,intervene_matrix,numReplicates,keep_species)
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

% calculate concentrations, in units of biomass

numSubjects = length(data_counts);
% remove filtered out OTUs from concentration matrices
keep_species_total = [];
for e=1:length(keep_species),
    keep_species_total = union(keep_species_total,keep_species{e});
end;
concentrations = cell(numSubjects,length(keep_species_total));
factor = concentrations;

biomass_avg = avgBiomass(BMD,numReplicates);
total_counts = cell(numSubjects,1);

for s=1:numSubjects,
    numTimepoints = length(T{s});
    tc = zeros(numTimepoints,1);
    for o=1:size(intervene_matrix{s},2),
        tv = zeros(numTimepoints,1);
        f = find(intervene_matrix{s}(:,o) > 0);
        if ~isempty(f),
            tv(f) = data_counts{s}(f,o);
        end;
        tc = tc + tv;
    end;
    total_counts{s} = tc;
end;

for s=1:numSubjects,
    for ox=1:length(keep_species_total),
        o = keep_species_total(ox);
        numTimepoints = length(T{s});
        tv = zeros(numTimepoints,1);
        f = find(intervene_matrix{s}(:,o) > 0);
        if ~isempty(f),
            tv(f) = data_counts{s}(f,o);
        end;
        %concentrations{s,ox} = (tv./total_counts{s}).*biomass_avg{s};
        concentrations{s,ox} = tv;
        factor{s, ox} = biomass_avg{s}./total_counts{s};
    end;
end;

function [biomass_avg] = avgBiomass(BMD,numReplicates)

numSubjects = length(BMD);

biomass_avg = cell(numSubjects,1);

for s=1:numSubjects,
    numTimepoints = size(BMD{s},1)/numReplicates;
    tbm = zeros(numTimepoints,1);
    for t=1:numTimepoints,
        %tbm(t) = exp(mean(BMD{s}( ((t-1)*numReplicates + 1):(t*numReplicates))));
        tbm(t) = exp(mean(BMD{s}(t:numTimepoints:end)));
    end;
    biomass_avg{s} = tbm;
end;



