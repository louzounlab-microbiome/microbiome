function postprocess_write_files_for_stability_analysis_output(...
    filename_prefix, N, outname_format, species_names, cutoff, doKeystone, ...
    num_perturbations)
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

ss_analysis_filename = [filename_prefix '.stability_analysis.txt'];
ks_analysis_filename = [filename_prefix '.keystone_analysis.txt'];

if doKeystone
    N_species = numel(species_names);
    if N_species > 21
        disp('WARNING: keystone analysis may take a very large amount of memory')
    end

    keystone_root = zeros(2^N_species-1, 2+N_species, num_perturbations);
end

fid=fopen(ss_analysis_filename,'w');
fprintf(fid,'ProfileID\tPerturbationID\tN_species\tfrequency\t');
fprintf(fid, [strjoin(species_names, '\t') '\n']);
for file_num=1:N
    outfname = sprintf(outname_format, int2str(file_num));
    linanalysis = load(outfname);
    total_taxa = size(linanalysis.median_stable, 1);
    num_alpha = size(linanalysis.median_stable, 2);
    num_stable = size(linanalysis.eigenvalues{1, 1}, 1);
    for p=1:num_alpha
        fprintf(fid, '%d\t%d\t%d\t%g', file_num, p-1, num_stable, ...
            linanalysis.frequency_of_stability(p));
        for t=1:total_taxa
            fprintf(fid, '\t%g', linanalysis.median_stable(t, p));
        end
        fprintf(fid, '\n');
        
        if doKeystone
            keystone_root(file_num, :, p) = ...
                [length(nonzeros(linanalysis.median_stable(:, p))) ...
                linanalysis.frequency_of_stability(p) ...
                linanalysis.median_stable(:, p)'];
        end
    end
end
fclose(fid);


if doKeystone
    fid=fopen(ks_analysis_filename,'w');
    fprintf(fid,'StateID\tParentID\tPerturbationID\tN_species\tfrequency\t');
    fprintf(fid, [strjoin(species_names, '\t') '\n']);
    for p=1:num_perturbations
        keystone_found = 0;
        state_count = 0;
        N_species = numel(species_names);
        while ~keystone_found
            % care only about correct N_species and > cutoff frequency
            ksr_slice = keystone_root(keystone_root(:, 1, p) == N_species ...
                & keystone_root(:, 2, p) > cutoff, :, p);
            if size(ksr_slice, 1) >= 1  % something needs to be there
                keystone_found = 1;
                subspecies = N_species - 1;
                for i=1:size(ksr_slice, 1)
                    state_count = state_count + 1;
                    parent = state_count;
                    write_steady_state(fid, ksr_slice(i, :), state_count, parent, p)
                    % adding zeros to ignore the ID and frequency columns
                    absent_species = logical([0, 0, (ksr_slice(i, 3:end) == 0)]);
                    % grab the subset where N_species is 1 less than what we're at
                    subset = keystone_root(keystone_root(:, 1, p) == subspecies & ...
                        keystone_root(:, 2, p) > cutoff, :, p);
                    % find only those that match our steady state (absent species
                    % must still be absent)
                    subset = subset(all(subset(:, absent_species) == 0, 2), :);
                    
                    for j=1:size(subset, 1)
                        state_count = state_count + 1;
                        write_steady_state(fid, subset(j, :), ...
                            state_count, parent, p)
                    end
                    
                end
            end
            N_species = N_species - 1;
        end
    end
    fclose(fid);
end

end

function [] = write_steady_state(fid, state, count, parent, perturbation)
fprintf(fid, '%d\t%d\t%d', count, parent, perturbation);
for st=state
    fprintf(fid, '\t%g', st);
end
fprintf(fid, '\n');
end