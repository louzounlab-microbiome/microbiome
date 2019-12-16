function [f_fd,f_fd_deriv,intervene_matrix_fd,perturbations_fd] = estForwardDiffDeriv(T,data,keep_species,intervene_matrix,perturbations)
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

% estimate differences based on forward differences (scaled appropriately
% for Bayesian algorithms)

numSubjects = size(data,1);

intervene_matrix_fd = intervene_matrix;
perturbations_fd = perturbations;

intervene_matrix_fd(size(intervene_matrix_fd,1),:) = [];

for s=1:numSubjects,
    if ~isempty(perturbations),
        perturbations_fd{s}(length(perturbations{s})) = [];
    end;
end;

f_fd_deriv = cell(numSubjects,1);
f_fd = cell(numSubjects,1);
for s=1:numSubjects,
    numOTUs = length(keep_species);
    fdt = zeros(length(T)-1,numOTUs);
    fft = zeros(length(T)-1,numOTUs);
    for o=1:numOTUs,
         tidx = find(intervene_matrix(:,o) > 0);
         ff = data{s}(tidx,keep_species(o));
         fftt = ff(1:(length(T(tidx))-1));
         %ffd = (diff(log(ff))./diff(T(tidx))).*fftt;
         ffd = (diff(log(ff))./diff(T(tidx)));
         fdt(tidx(1:(length(tidx)-1)),o) = ffd;
         fft(tidx(1:(length(tidx)-1)),o) = fftt;
    end;
    f_fd_deriv{s} = fdt;
    f_fd{s} = fft;
end;


