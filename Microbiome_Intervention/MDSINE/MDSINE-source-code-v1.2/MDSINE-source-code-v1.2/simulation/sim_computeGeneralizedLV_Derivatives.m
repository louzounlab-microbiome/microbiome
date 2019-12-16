function [DF]= sim_computeGeneralizedLV_Derivatives(t,F,struct)
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

% Vanni Bucci
% for info:
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
%-----------------------
Beta=struct.Beta;
alpha=struct.alpha;
Gamma=struct.Gamma;
U=struct.U;
tspan=struct.tspan;
dt_dummy=tspan(end)-tspan(end-1);
tspan2=[tspan tspan(end)+dt_dummy];
idXneg=find(F<0);
if ~isempty(idXneg)
    F(idXneg)=0;
end
for i=1:size(U,1)
     U_tmp=U(i,:);
     U_tmp=[U_tmp U_tmp(end)];
     U2(i)=interp1(tspan2,U_tmp,t);
end

for i=1:length(F)
    if F(i)>0
        T=0;
        T=alpha(i)*F(i);
        for j=1:length(F)
            if F(j)>0;T=T+Beta(i,j)*F(j)*F(i);end
        end
        if ~isempty(U)
            for k=1:size(Gamma,2);T=T+Gamma(i,k)*U2(k)*F(i);end
        end
        DF(i,1)=T;
    else
        DF(i,1)=0;
    end
end

% Below is where we check for stopping conditions
MAXTIME = 6000; %Max time in seconds
MINSTEP = 1e-11; %Minimum step
persistent tprev elapsedtime
if isempty(tprev)
    tprev = -inf;
end
if isempty(elapsedtime)
    elapsedtime = tic;
end

timestep = t - tprev;
tprev = t;

if (t > 0.01) && (timestep > 0) && (timestep < MINSTEP)
    error(['Stopped. Time step is too small: ' num2str(timestep)])
elseif toc(elapsedtime) > MAXTIME
    elapsedtime=[];
    error('Stopped. Taking too long.')
end

warningMessID1='MATLAB:illConditionedMatrix';
[warnmsg, msgid] = lastwarn;

if strcmp(warningMessID1,msgid)==1
    lastwarn('');
    error('Exiting due to: %s',warningMessID1);
end

return
