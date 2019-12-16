function [Y0] = buildInitialConditionVectorsFromSplines(T,init_times,f,med_biomass,med_counts)
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

% build vectors of initial conditions from spline estimates

numSubjects = length(f);
numOTUs = size(f{1},2);

rescaleFactor = exp(med_biomass)/med_counts;

for s=1:numSubjects,
    tidx = find(T{s} == init_times(s));
    tv = zeros(numOTUs,1);
    for o=1:numOTUs,
        tv(o) = f{s}(tidx,o)*rescaleFactor;
    end;
    Y0{s} = tv;
end;


