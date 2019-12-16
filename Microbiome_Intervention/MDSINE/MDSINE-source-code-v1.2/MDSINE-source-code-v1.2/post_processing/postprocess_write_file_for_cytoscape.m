function postprocess_write_file_for_cytoscape(filename_prefix,Theta,...
    bayes_factors,names)
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

% -------------------------------------------------------------------------------------------------
% This function converts the model parameters and parameter signficance (if any) matrices into flat
% files to be imported into cytoscape for network visualization.
% -------------------------------------------------------------------------------------------------

L=size(Theta,1); % number of species
P=size(Theta,2)-1-L; % number of perturbations
d_interactions=Theta(1:L, 2:L+1);
taxa_names = names(2:L+1);
cytoscape_filename = ...
    sprintf('%s.cytoscape.txt',filename_prefix);

d_growth = Theta(1:L,1);

if ~isempty(bayes_factors)
    significance_growth=bayes_factors(:,1);
    significance_interactions=bayes_factors(1:L, 2:L+1);
else
    significance_growth=[];
    significance_interactions=[];
end

if P > 0
    perturbation_names=names(L+2:L+1+P);
    d_perturbations=Theta(1:L,L+2:L+1+P);
    if ~isempty(bayes_factors)
        significance_perturbations=bayes_factors(1:L, L+2:L+1+P);
    else
        significance_perturbations=[];
    end
else
    perturbation_names=[];
    d_perturbations=[];
    significance_perturbations=[];
end

fid=fopen(cytoscape_filename,'w');
if ~isempty(d_interactions)
    for i=1:length(taxa_names);
        for j=1:length(taxa_names);
            if i~=j % do not print self interaction
                if ~isempty(significance_interactions)
                    if d_interactions(i,j)~=0
                        fprintf(fid,'%s %s %s %s %s',taxa_names{j},...
                            taxa_names{i},num2str(sign(d_interactions(i,j))),...
                            num2str(abs(d_interactions(i,j))),num2str(significance_interactions(i,j)));
                        fprintf(fid,'\n');
                    end
                elseif d_interactions(i,j)~=0
                    if num2str(abs(d_interactions(i,j)))~=0
                        fprintf(fid,'%s %s %s %s',taxa_names{j},...
                            taxa_names{i},num2str(sign(d_interactions(i,j))),...
                            num2str(abs(d_interactions(i,j))));
                        fprintf(fid,'\n');
                    end
                end
            end
        end
    end
end

if P>0
    for l=1:length(perturbation_names);
        for i=1:length(taxa_names);
            if ~isempty(significance_perturbations)
                if d_perturbations(i,l)~=0
                    fprintf(fid,'%s %s %s %s %s',perturbation_names{l},...
                        taxa_names{i},num2str(sign(d_perturbations(i,l))),...
                        num2str(abs(d_perturbations(i,l))),...
                        num2str(significance_perturbations(i,l)));
                    fprintf(fid,'\n');
                end
            else
                if d_perturbations(i,l)~=0
                    fprintf(fid,'%s %s %s %s',perturbation_names{l},...
                        taxa_names{i},num2str(sign(d_perturbations(i,l))),...
                        num2str(abs(d_perturbations(i,l))));
                    fprintf(fid,'\n');
                end
            end
        end
    end
end
fclose(fid);

return
