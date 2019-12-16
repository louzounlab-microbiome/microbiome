function postprocess_write_file_for_parameters(filename_prefix, ...
    Theta, Theta_samples, bayes_factors, names)
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
% This function converts the model parameters and parameter significance (if any) matrices into flat
% files to be imported into any type of spreadsheet/visualization software.
% R {ggplot2} visualization is reccomended, see provided R utility scripts.
% -------------------------------------------------------------------------------------------------

if numel(Theta_samples) > 1
    Theta_samples_array = zeros(size(Theta, 1), size(Theta, 2), numel(Theta_samples));
    for i = 1:numel(Theta_samples)
        Theta_samples_array(:, :, i) = Theta_samples{i};
    end
    Theta_MCMC_std = std(Theta_samples_array, [], 3);
else
    Theta_MCMC_std = Theta;
end


L=size(Theta,1); % number of species
P=size(Theta,2)-1-L; % number of perturbations
d_interactions=Theta(1:L, 2:L+1);
d_interactions_std=Theta_MCMC_std(1:L, 2:L+1);
taxa_names = names(2:L+1);
parameters_filename = ...
    sprintf('%s.parameters.txt',filename_prefix);
d_growth = Theta(1:L,1);
d_growth_std = Theta_MCMC_std(1:L,1);

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
    d_perturbations_std=Theta_MCMC_std(1:L,L+2:L+1+P);
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

fid=fopen(parameters_filename,'w');

fprintf(fid,['parameter_type\tsource_taxon\ttarget_taxon\tvalue' ...
    '\tsignificance\tMCMC_std\n']);

if ~isempty(d_growth)
    for i=1:length(taxa_names)
        if ~isempty(significance_growth)
            fprintf(fid,'growth_rate\tNA\t%s\t%s\t%s\t%s\n',taxa_names{i},...
                num2str(d_growth(i)),num2str(significance_growth(i)), ...
                num2str(d_growth_std(i)));
        else
            fprintf(fid,'growth_rate\tNA\t%s\t%s\tNA\tNA\n',taxa_names{i},...
            num2str(d_growth(i)));
        end
    end
end

if ~isempty(d_interactions)
    for i=1:length(taxa_names);
        for k=1:length(taxa_names);
            if ~isempty(significance_interactions)
                fprintf(fid,'interaction\t%s\t%s\t%s\t%s\t%s\n', ...
                    taxa_names{i}, taxa_names{k}, ...
                    num2str(d_interactions(k,i)), ...
                    num2str(significance_interactions(k,i)), ...
                    num2str(d_interactions_std(k,i)));
            else
                fprintf(fid,'interaction\t%s\t%s\t%s\tNA\tNA\n', ...
                    taxa_names{i}, taxa_names{k},...
                    num2str(d_interactions(k,i)));
            end
        end
    end
end

if P>0
    for l=1:length(perturbation_names);
        for k=1:length(taxa_names);
            if ~isempty(significance_perturbations)
                fprintf(fid,'perturbation\t%s\t%s\t%s\t%s\t%s\n', ...
                    perturbation_names{l},taxa_names{k}, ...
                    num2str(d_perturbations(k,l)), ...
                    num2str(significance_perturbations(k,l)),...
                    num2str(d_perturbations_std(k,l)));
                
            else
                fprintf(fid,'perturbation\t%s\t%s\t%s\tNA\tNA\n', ...
                    perturbation_names{l}, taxa_names{k}, ...
                    num2str(d_perturbations(k,l)));
            end
        end
    end
end
fclose(fid);

return


