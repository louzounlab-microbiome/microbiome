function [ cfg ] = formatconfig( fname )
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

cfg = readconfig(fname);

cfg.general.seed = str2double(cfg.general.seed);

cfg.general.run_inference = str2double(cfg.general.run_inference);
cfg.general.run_simulations = str2double(cfg.general.run_simulations);
cfg.general.run_linear_stability = str2double(cfg.general.run_linear_stability);
cfg.general.run_post_processing = str2double(cfg.general.run_post_processing);

if cfg.general.output_dir(end) ~= '/'
    cfg.general.output_dir(end+1) = '/';
end

% following sections' fields are one number each
sections = {'Ridge Regression', 'Bayesian Lasso', 'Bayesian Select', ...
    'Preprocessing', 'bayesian spline biomass', 'bayesian spline counts', ...
    'parallel', 'post processing', 'linear stability'};

for j = 1:numel(sections)
    section = matlab.lang.makeValidName(lower(sections{j}));
    fields = fieldnames(cfg.(section));
    for i = 1:numel(fields)
      cfg.(section).(fields{i}) = str2double(cfg.(section).(fields{i}));
    end
end

section = matlab.lang.makeValidName('simulation');
fields = fieldnames(cfg.(section));
for i = 1:numel(fields)
    cfg.(section).(fields{i}) = cellfun(@str2num, ...
                                strsplit(cfg.(section).(fields{i}), ' '));
end

end

