function [Theta_low,A,Theta_high,Theta_indicator] = estInteractMatrixFromSamples(Theta_samples,ci)
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

% computes a consensus interaction matrix from MCMC samples
%
% inputs:
% Theta_samples = MCMC samples of interaction matrix
% ci = credible interval (e.g., 0.95 for 95% CI)
%
% outputs:
% Theta_low, A, Theta_high = low centile, mean, and high centile interaction
% matrices
% Theta_indicator = thresholded indicator matrix (entry is 1 if the specified
% CI does not contain 0)

numOTUs = size(Theta_samples{1},1);
numSamples = length(Theta_samples);

Theta_low = zeros(size(Theta_samples{1}));
A = zeros(size(Theta_samples{1}));
Theta_high = zeros(size(Theta_samples{1}));
Theta_indicator = zeros(size(Theta_samples{1}));

for o=1:numOTUs,
    TM = zeros(numSamples,size(A,2));
    for s=1:numSamples,
        TM(s,:) = Theta_samples{s}(o,:);
    end;
    Theta_low(o,:) = prctile(TM,100*(1-ci)/2.0,1);
    Theta_high(o,:) = prctile(TM,50+100*ci/2.0,1);
    A(o,:) = mean(TM);

    f = find(Theta_low(o,:) > 0 | Theta_high(o,:) < 0);
    Theta_indicator(o,f) = 1;
end;
