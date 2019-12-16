function [r] = sample_inverse_gaussian(mu,lambda)
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

% sample from an inverse Gaussian distribution

sizeOut = size(mu);

c = mu.*chi2rnd(1,sizeOut);
r = (mu./(2.*lambda)) .* (2.*lambda + c - sqrt(4.*lambda.*c + c.^2));
invert = (rand(sizeOut).*(mu+r) > mu);
r(invert) = mu(invert).^2 ./ r(invert);
