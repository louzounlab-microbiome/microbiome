function [otuMap1,otuMap2,otu_names_merged] = generateMergedOTUMap(otu_names1,otu_names2)
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


otu_names_merged = [otu_names1 ; otu_names2];
otu_names_merged = unique(otu_names_merged);

% indices of otus1 mapped into merged otus
otuMap1 = zeros(length(otu_names1),1);

% indices of otus2 mapped into merged otus
otuMap2 = zeros(length(otu_names2),1);

for i=1:length(otuMap1),
    otuMap1(i) = strmatch(otu_names1{i},otu_names_merged,'exact');
end;

for i=1:length(otuMap2),
    otuMap2(i) = strmatch(otu_names2{i},otu_names_merged,'exact');
end;

