function postprocess_write_file_for_trajectories(filename_prefix,traj,traj_high,...
    traj_low,traj_time,taxa_names,o_write_original_data, concentrations, c_times)
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

% Vanni Bucci, Ph.D.
% Assistant Professor
% Department of Biology
% Room: 335A
% University of Massachusetts Dartmouth
% 285 Old Westport Road
% N. Dartmouth, MA 02747-2300
% Phone: (508)999-9219
% Email: vbucci@umassd.edu
% Web: www.vannibucci.org
% -------------------------------------------------------------------------------------------------
% This function converts the data contained in the sims structure
% (numerical simulations and original data, if any) into a tab delimited
% file to be used for plotting and visualization. R {ggplot2} visualization is reccomended,
% see provided R utility scripts. is reccomanded see provided R utility scripts.
% ##################
% This functions has one option o_write_original_data (1: write also the
% data corresponding to the trajecory -- useful if we want to compare
% against some experimentally determined data; 0: leave the original data
% writing aside)
% ##################
% -------------------------------------------------------------------------------------------------
simulations_filename = ...
    sprintf('%s.simulations.txt',filename_prefix);

fid=fopen(simulations_filename,'w');

fprintf(fid,'trajectory_ID\ttaxon\tabundance\ttime\ttype\n');

for subj=1:length(traj)
    trajectory=traj{subj}';
    time=traj_time{subj}';
    for k=1:size(trajectory,1)
        for j=1:size(trajectory,2)
            fprintf(fid,'%s\t%s\t%s\t%s\t%s\n',num2str(subj),taxa_names{k},...
            num2str(trajectory(k,j)),num2str(time(j)),'simulation');
        end
    end
end

if o_write_original_data
    for subj=1:size(concentrations,1)
        time_original=c_times{subj};
        for taxon=1:size(concentrations, 2)
            for t=1:size(concentrations{subj, taxon}, 1)
                fprintf(fid, '%s\t%s\t%s\t%s\t%s\n',num2str(subj), ...
                    taxa_names{taxon}, ...
                    num2str(concentrations{subj, taxon}(t)), ...
                    num2str(time_original(t)),'data');
            end
        end
    end
end
