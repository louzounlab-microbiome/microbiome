function [mu_cond,sigma_cond] = condMVN(mu,sigma,x2,cond_idx)
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

% compute conditional multivariate normal, conditioning on variables
% w/ indices specified in cond_idx

idx = setdiff(1:length(mu),cond_idx);
sigma1 = sigma(idx,idx);
sigma2 = sigma(cond_idx,cond_idx);
sigma12 = sigma(idx,cond_idx);
sigma2_inv = inv_chol(sigma2);
sigma12_m_sigma2_inv = sigma12*sigma2_inv;
mu_cond = mu(idx)+sigma12_m_sigma2_inv*(x2-mu(cond_idx));
sigma_cond = sigma1 - sigma12_m_sigma2_inv*sigma12';

