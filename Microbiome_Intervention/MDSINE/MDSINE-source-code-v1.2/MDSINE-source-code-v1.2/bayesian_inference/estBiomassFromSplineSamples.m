function [biomass,biomass_high,biomass_low] = estBiomassFromSplineSamples(S_sample,T,BO,biomassScaleFactor)
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

% estimate biomass trajectories from MCMC spline samples

% biomass = trajectory estimates (biomass_high = 97.5% CI estimates, biomass_low = 2.5% CI
% estimates)

numSubjects = size(S_sample,1);

biomass = cell(numSubjects,1);
biomass_high = cell(numSubjects,1);
biomass_low = cell(numSubjects,1);

for s=1:numSubjects,
    biomass{s} = zeros(length(T),1);
    biomass_high{s} = zeros(length(T),1);
    biomass_low{s} = zeros(length(T),1);

    biomass{s} = BO*mean(S_sample{s},2)+biomassScaleFactor(s);
    biomass_high{s} = BO*prctile(S_sample{s},97.5,2)+biomassScaleFactor(s);
    biomass_low{s} = BO*prctile(S_sample{s},2.5,2)+biomassScaleFactor(s);
end;


