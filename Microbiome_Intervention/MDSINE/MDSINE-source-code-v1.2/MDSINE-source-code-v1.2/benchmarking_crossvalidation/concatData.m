function [species_names,counts_data,intervene_matrix] = concatData(species_names1,counts_data1,intervene_matrix1,species_names2,counts_data2,intervene_matrix2)
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


[otuMap1,otuMap2,species_names] = generateMergedOTUMap(species_names1,species_names2);

numSubjects1 = length(counts_data1);
numSubjects2 = length(counts_data2);

intervene_matrix = cell(numSubjects1+numSubjects2,1);
counts_data = cell(numSubjects1+numSubjects2,1);

for s=1:numSubjects1,
    im = zeros(size(intervene_matrix1,1),length(species_names));
    im(:,otuMap1) = intervene_matrix1;
    intervene_matrix{s} = im;

    cd = zeros(size(intervene_matrix1,1),length(species_names));
    cd(:,otuMap1) = counts_data1{s};
    counts_data{s} = cd;
end;

for s=1:numSubjects2,
    im = zeros(size(intervene_matrix2,1),length(species_names));
    im(:,otuMap2) = intervene_matrix2;
    intervene_matrix{s+numSubjects1} = im;

    cd = zeros(size(intervene_matrix2,1),length(species_names));
    cd(:,otuMap2) = counts_data2{s};
    counts_data{s+numSubjects1} = cd;
end;

