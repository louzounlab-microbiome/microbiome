function [RES]= sim_computeGeneralizedLV_Derivatives_ImplicitForm(t,F,Fp,struct)
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


Beta=struct.Beta;
alpha=struct.alpha;
Gamma=struct.Gamma;
U=struct.U;
tspan=struct.tspan;

dt_dummy=tspan(end)-tspan(end-1);
tspan2=[tspan tspan(end)+dt_dummy];

% remove negative Xs
idXnegative=find(F<0);
if ~isempty(idXnegative)
    F(idXnegative)=0;
end

for i=1:size(U,1)
     U_tmp=U(i,:);U_tmp=[U_tmp U_tmp(end)];U2(i)=interp1(tspan2,U_tmp,t);
end

RES=zeros(length(Fp),1);
DX=zeros(length(Fp),1);

for i=1:length(F)
    if F(i)>0
        T=0;
        % growth & interactions
        T=alpha(i)*F(i);
        for j=1:length(F)
            if F(j)>0
                T=T+Beta(i,j)*F(j)*F(i);
            end
        end

        if ~isempty(U)
            % perturbations
            for k=1:size(Gamma,2)
                T=T+Gamma(i,k)*U2(k)*F(i);
            end
        end
        % differential
        DX(i,1)=T;
    else
        DX(i,1)=0;
    end
    % for the implicit case we are returning the RESIDUAL (see ihb1dae.m
    % example)
    RES(i,1)=Fp(i)-DX(i,1);
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
