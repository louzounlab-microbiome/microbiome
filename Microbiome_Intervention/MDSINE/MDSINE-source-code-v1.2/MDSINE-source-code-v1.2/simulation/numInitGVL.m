function [Y,YD,intervene_matrix_sim] = numInitGVL(tspan,C,Y0,interveneTimes,perturbation_begin_end,assumeStiff)
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

% numerically integrate generalized Volterra Lotka system

% inputs:
% tspan = time-span to integrate over
% C = growth parameters are first column, subject entries are interaction
% parameters
% Y0 = cell of initial conditions
% interveneTimes = vector of times for intervention (OTU is assumed to be
% at zero concentration prior to the intervene time)
% assumeStiff = set to 1 to use Matlab numerical integration function for
% stiff systems

% outputs:
% Y = cell of trajectories
% YD = cell of trajectory derivatives
% intervene_matrix_sim = indicator matrix for tspan, indicating if OTU has
% been introduced yet (used for testing/debugging)

activeOTUs = [];
numOTU = size(C,1);
numSim = length(Y0);

% simulated trajectories
Y = cell(numSim,1);
% simulated derivatives
YD = cell(numSim,1);

intervene_matrix_sim = zeros(length(tspan),numOTU);
for o=1:numOTU,
    f = find(tspan >= interveneTimes(o));
    intervene_matrix_sim(f,o) = 1;
end;

intervene_union = sort(unique(interveneTimes));

for i=1:numSim,
    Y{i} = zeros(length(tspan),numOTU);
    YD{i} = zeros(length(tspan),numOTU);

    % do numerical integration taking into account which OTUs are present
    % in the system
    for it=1:length(intervene_union),
        activeOTUs = find(interveneTimes <= intervene_union(it));
        established_OTUs = find(interveneTimes < intervene_union(it));

        % set span for integration
        tspan_start = find(tspan == intervene_union(it));
        if it<length(intervene_union),
            tspan_end = find(tspan == intervene_union(it+1));
        else,
            tspan_end = length(tspan);
        end;
        current_tspan_idx = tspan_start:tspan_end;

        Y_start = Y0{i};
        if ~isempty(established_OTUs),
            ft = find(tspan == intervene_union(it));
            Y_start(established_OTUs) = Y{i}(ft,established_OTUs);
        end;

        % solve gVL system
        if assumeStiff == 1,
            [T,Y{i}(current_tspan_idx,activeOTUs)] = ode15s(@evalVL,tspan(current_tspan_idx),Y_start(activeOTUs));
        else,
            [T,Y{i}(current_tspan_idx,activeOTUs)] = ode45(@evalVL,tspan(current_tspan_idx),Y_start(activeOTUs));
        end;
        %[T,Y{i}(current_tspan_idx,activeOTUs)] = ode45(@evalVL,tspan(current_tspan_idx),Y_start(activeOTUs));

        % calculate derivatives
        for t=1:length(current_tspan_idx),
            YD{i}(current_tspan_idx(t),activeOTUs) = evalVL(tspan(current_tspan_idx(t)),Y{i}(current_tspan_idx(t),activeOTUs)')';
        end;
    end;
end;

function [yp] = evalVL(t,y)
        yp = (C(activeOTUs,activeOTUs+1)*y).*y+C(activeOTUs,1).*y;
        if ~isempty(perturbation_begin_end),
            for ipx=1:size(perturbation_begin_end,1),
                if t >= perturbation_begin_end(ipx,1) && t <= perturbation_begin_end(ipx,2),
                     yp = yp + C(activeOTUs,numOTU+1+ipx).*y;
                end;
            end;
        end;
    end
end
