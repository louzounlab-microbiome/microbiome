function [params] = readParameterFile(fname)
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

% parameter file is tab delimited
% first col = parameter name
% second col = parameter value
% third col = comment

% returns a struct containing the parameters
% the str2num function is used to convert numbers to numerical values
% (leaves string alone)

fid = fopen(fname,'r');
tp = textscan(fid,'%s %s %s','delimiter','\t');
fclose(fid);

for s=1:length(tp{1}),
    tv = str2num(tp{2}{s});
    if ~isempty(tv),
        tp{2}{s} = tv;
    end;
end;

params = cell2struct(tp{2},tp{1});

