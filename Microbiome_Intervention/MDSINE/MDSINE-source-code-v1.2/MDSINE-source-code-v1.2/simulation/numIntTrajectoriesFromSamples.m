function [traj,traj_high,traj_low,percentSucceed] = ...
    numIntTrajectoriesFromSamples(T, perturbations, experimentBlocks, ...
    intervene_matrix_filtered_merge, Y0, med_biomass, med_counts, ...
    Theta_samples, startTime, endTime, timeInc, thinRate, assumeStiff)
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

% numerically integrate trajectories using MCMC samples of growth
% rates/interaction parameters
%
% inputs:
% T = cell array of time-points that data was sampled at
% intervene_matrix_filtered = cell array of filtered intervention matrix
% perturbations = cell array of perturbations
% Y0 = cell (dim # subjects) of vectors of initial conditions
% med_biomas, med_counts = median biomass and counts (used to rescale
% initial conditions for numerical stability)
% Theta_samples = cell array of MCMC samples of growth rates/interaction
% parameter matrice
% startTime = vector of start time for each experimental block
% endTime = vector of end time for each experimental block
% timeInc = vector of time increments for each experimental block
% thinRate = specifies rate to thin MCMC samples (e.g, use every nth
% sample)
% assumeStiff = set to 1 to use Matlab numerical integration function for
% stiff systems
%
% outputs:
% traj, traj_high, traj_low = cell array of median and 95% credible
% interval trajectories for each experimental block
% percentSucceed = vector of percentage of MCMC samples that succeed numerical
% integration for each experimental block

% find first index for each experimental block
uniqueExperiments = unique(experimentBlocks);
blockIdx = zeros(length(uniqueExperiments),1);
for e=1:length(blockIdx),
    f = find(experimentBlocks == uniqueExperiments(e));
    blockIdx(e) = f(1);
end;

traj = cell(length(experimentBlocks),1);
traj_high = cell(length(experimentBlocks),1);
traj_low = cell(length(experimentBlocks),1);
percentSucceed = zeros(length(uniqueExperiments),1);

for e=1:length(uniqueExperiments),
    ep = find(experimentBlocks == uniqueExperiments(e));
    perturbation = [];
    if ~isempty(perturbations),
        perturbation = perturbations{blockIdx(e)};
    end;
    [traj(ep),traj_high(ep),traj_low(ep),percentSucceed(e)] = ...
        sub_numIntTrajectoriesFromSamples(T{blockIdx(e)},perturbation, ...
        intervene_matrix_filtered_merge{blockIdx(e)},Y0(ep),med_biomass,...
        med_counts,Theta_samples,startTime(e),endTime(e),timeInc(e),thinRate,assumeStiff);
end;

function [traj,traj_high,traj_low,percentSucceed] = ...
    sub_numIntTrajectoriesFromSamples(T,perturbation,...
    intervene_matrix_filtered,Y0,med_biomass,med_counts,...
    Theta_samples,startTime,endTime,timeInc,thinRate,assumeStiff)

numSubjects = length(Y0);
numSamples = length(Theta_samples);

traj = cell(numSubjects,1);
traj_high = cell(numSubjects,1);
traj_low = cell(numSubjects,1);

% use very nth sample
useSamples = 1:thinRate:numSamples;
tspan = startTime:timeInc:endTime;

rescaleFactor = exp(med_biomass)/med_counts;

Y0_rescale = Y0;
for s=1:numSubjects,
    Y0_rescale{s} = Y0_rescale{s}/rescaleFactor;
end;

for subj=1:numSubjects,
    traj{subj} = ones(length(tspan),length(Y0{1}),length(useSamples))*NaN;
end;

perturbation_begin_end = [];
if ~isempty(perturbation),
    perturbation_begin_end = zeros(1,2);
    f = find(perturbation == 1);
    if ~isempty(f),
        perturbation_begin_end(1,1) = T(f(1));
        perturbation_begin_end(1,2) = T(f(length(f)));
    else
        perturbation_begin_end = [];
    end;
end;

intervene_from_start = zeros(1,size(intervene_matrix_filtered,2));
for o=1:length(intervene_from_start),
    f = find(intervene_matrix_filtered(:,o) > 0);
    if ~isempty(f),
        intervene_from_start(o) = max(startTime,T(f(1)));
    else
        intervene_from_start(o) = startTime;
    end;
end;

percentSucceed = 0;
for s=1:length(useSamples),
    try
        % [Y,YD,intervene_matrix_sim] 
        [Y,~,~] = numInitGVL(tspan,Theta_samples{useSamples(s)}, ...
            Y0_rescale, intervene_from_start, perturbation_begin_end, ...
            assumeStiff);
        for subj=1:numSubjects,
            traj{subj}(:,:,s) = Y{subj}*rescaleFactor;
        end;
        percentSucceed = percentSucceed + 1;
    catch
    end
end;

for subj=1:numSubjects,
    traj_high{subj} = prctile(traj{subj},97.5,3);
    traj_low{subj} = prctile(traj{subj},2.5,3);
    traj{subj} = nanmedian(traj{subj},3);
end;

percentSucceed = percentSucceed/length(useSamples);

