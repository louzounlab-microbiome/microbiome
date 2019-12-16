function [BMD,sigma_biomass_est] = GetBiomassStd(BMD_raw,params)
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

% reformat biomass data into cell data structure
%
% format of BMD_raw is:
% 1st col = time of collection (will be repeated for each replicate), subsequent columns are the different subjects

%params = readParameterFile(paramsFileName);
numReplicates = params.numReplicates;
numSubjects = length(BMD_raw);

BMD = cell(numSubjects,1);

%T_BMD = unique(BMD_raw(:,1));

ssd = []; %0;
for m=1:numSubjects,
%    D = zeros(numReplicates*length(T_BMD),1);
%    for r=1:numReplicates,
%        D(((r-1)*length(T_BMD)+1):(r*length(T_BMD))) = BMD_raw(r:numReplicates:end,m+1);
%    end;
    BMD{m} = log(BMD_raw{m});
    % estimate std of data from replicates
    for t=1:length(BMD{m})/numReplicates,
        dt = BMD{m}(((t-1)*numReplicates + 1):t*numReplicates);
        dm = mean(dt);
        ssd = [ssd ; (dt-dm).^2];
    end;
end;

sigma_biomass_est = sqrt(mean(ssd));
