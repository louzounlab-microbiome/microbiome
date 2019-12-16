function [traj] = wrap_numIntTrajFromSamples(T_sample,t0,t_end,dt,perturbations,experimentBlocks,pathogen_ID,pathogen_t0,Y0_ex_pathogen,Y0_pathogen,med_biomass,med_counts,glv_params,assumeStiff)
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


%expect perturbations = [] b/c this is infection experiment
%expect all subjects to belong to same experimental block; i.e.,
%experimentBlocks = ones(numSubjects,1);
%T_sample = cell of length #subjects that gives timepoints at which each
%subject was sampled at; should be the same across all subjects if there is
%only 1 experimental block (previous assumption)


numSubjects = length(experimentBlocks);
intervene_matrix_filtered_merge = cell(numSubjects,1);
intervene_matrix = ones(length(T_sample{1}),size(glv_params,1));
if (~isempty(pathogen_ID) && ~isempty(pathogen_t0))
    idx_pathogen_t0 = find(T_sample{1}==pathogen_t0);
    intervene_matrix(1:idx_pathogen_t0-1,pathogen_ID) = 0; %by construction, pathogen not present at t(1) = timepoint at which other OTU's introduced
end
startTime = cell(numSubjects,1);
endTime = cell(numSubjects,1);

Y0 = cell(numSubjects,1);
Y0_vec = Y0_ex_pathogen;
if (~isempty(pathogen_ID) && ~isempty(pathogen_t0))
    Y0_vec(pathogen_ID) = Y0_pathogen;
end

for s=1:numSubjects
    startTime{s} = t0;
    endTime{s} = t_end;
    intervene_matrix_filtered_merge{s} = intervene_matrix;
    Y0{s} = Y0_vec;
end


A_samples = cell(1,1);
A_samples{1} = glv_params;

[traj,~,~,~] = numIntTrajectoriesFromSamples(T_sample,perturbations,experimentBlocks,intervene_matrix_filtered_merge,Y0,med_biomass,med_counts,A_samples,t0,t_end,dt,1,assumeStiff);

end
