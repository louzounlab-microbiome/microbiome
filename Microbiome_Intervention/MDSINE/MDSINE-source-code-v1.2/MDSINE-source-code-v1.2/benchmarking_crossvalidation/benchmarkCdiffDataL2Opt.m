function [A_global_L2] = benchmarkCdiffDataL2Opt(cfg, T,BMD,keep_species,data_counts)
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


numSubjects = length(data_counts);
% remove filtered out OTUs from concentration matrices
keep_species_total = [];
for e=1:length(keep_species),
    keep_species_total = union(keep_species_total,keep_species{e});
end;

[BMD_avg] = avgBiomass(BMD,T);

data_counts_filtered = cell(numSubjects,1);
for s=1:numSubjects,
    data_counts_filtered{s} = data_counts{s}(:,keep_species_total);
end;

 disp('running L2 opt');

 perturbations = [];
  total_biomass = cell(numSubjects,1);
  for s=1:numSubjects
    total_biomass{s} = BMD_avg{s};
  end;

  [A_global_L2] = doMLRRInference(cfg, data_counts_filtered, total_biomass, perturbations, T);

  % [A_global_L2,F,Y,lambda_global] = DoBucciInference(data_counts_filtered,total_biomass, perturbations, T, numSubjects,1);

function [biomass_avg] = avgBiomass(BMD,T)


biomass_avg = cell(length(BMD), 1);
for i=1:length(BMD)
    per_T = length(BMD{i})/length(T{i});
    biomass_avg{i} = zeros(length(T{i}), 1);
    for t=1:length(T{i})
        biomass_avg{i}(t) = mean(BMD{i}((t-1)*per_T+1:t*per_T));
    end
end
