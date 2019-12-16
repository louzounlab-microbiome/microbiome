function [biomass_avg] = avgBiomass(BMD,numReplicates)
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

% avg biomass replicates
% BMD is a cell w/ # of subjects
% each entry is a vector w/ |T| x numReplicates
% such that the first 1:T entries are biomasses at all time-points for
% replicate 1, next (T+1):2*T entries are entries for replicate 2, etc.

numSubjects = length(BMD);

biomass_avg = cell(numSubjects,1);

for s=1:numSubjects,
    numTimepoints = size(BMD{s},1)/numReplicates;
    tbm = zeros(numTimepoints,1);
    for t=1:numTimepoints,
        tbm(t) = exp(mean(BMD{s}(t:numTimepoints:end)));
    end;
    biomass_avg{s} = tbm;
end;

