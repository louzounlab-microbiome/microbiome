function [ params ] = readconfig( fname )
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

%READCONFIG Reads ini file format, returning a struct
%   standard format:
%   [Section]
%   field = value
%
%   yeilds:
%   params.Section.field = value
%
% =========================================================================

fid = fopen(fname);
params = [];

nline = fgetl(fid);
linecount = 1;
section = matlab.lang.makeValidName('missing section');
while ischar(nline)
    % remove comments and trailing whitespsace
    % comments are anything after a #
    tline = regexprep(strtrim(nline), '#.*$', '');
    nline = fgetl(fid);
    linecount = linecount + 1;

    % if empty, skip the rest
    if isempty(tline)
        continue
    end

    % check if it's a section
    if ~isempty(regexp(tline, '^\[.*\]$', 'once'))
        section = matlab.lang.makeValidName(lower(tline(2:end-1)));
        params.(section) = [];
        continue
    end

    split = strsplit(tline, '=');
    if length(split) ~= 2
        error('ERROR: line %d improperly formatted', linecount)
    end
    parameter = matlab.lang.makeValidName(strtrim(split{1}));
    value = strtrim(split{2});
    params.(section).(parameter) = value;
end
fclose(fid);
return
end
