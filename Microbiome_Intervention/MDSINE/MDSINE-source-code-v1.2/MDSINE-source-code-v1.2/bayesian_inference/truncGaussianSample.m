function [x] = truncGaussianSample(mu,s,a,isUpper)
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

% sample from truncated Gaussan distribution

% mu = mean
% s = std
% a = truncation limit
% isUpper = 1 will use a as an upper bound (lower bound is -Inf)
% isUpper = 0 will use a as a lower bound (upper bound is +Inf)

z = (a-mu)/s;
z_cdf = normcdf(z);
u = unifrnd(0,1);

if isUpper,
    xi_cdf = u * z_cdf;
else,
    xi_cdf = u + z_cdf*(1-u);
end;
xi = norminv(xi_cdf,0,1);
x = mu + s*xi;
