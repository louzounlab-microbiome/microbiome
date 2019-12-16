function [  ] = infer_bayesian_selection( input, params )
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

%INFER_BAYESIAN_SELECTION Summary of this function goes here
%   Detailed explanation goes here

if ~isdeployed()
    addpath('bayesian_inference')
end

doBayesianSelect(params, params.general.output_dir, input.T, ...
    input.intervene, input.perturbations, input.counts, ...
    input.species_names, input.biomass, input.blocks);

if ~isdeployed()
    rmpath('bayesian_inference')
end

end

