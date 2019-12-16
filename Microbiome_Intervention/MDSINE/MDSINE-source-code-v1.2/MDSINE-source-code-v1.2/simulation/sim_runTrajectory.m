function [tF_OUT,isCompleted]=sim_runTrajectory(Theta,taxa_data,U,time_data,t0,dt,tmax,...
    toIntroduce,density_toIntroduce,time_toIntroduce,variance_toIntroduce)
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
%---------------------
% initialize outputs
tF_OUT=[];
isCompleted=[];

%---------------------
% setting some harcoded constants/options
break_tolerance=1e15;
extinction_threshold=0;
o_use_implicit_solver=0;

%---------------------
% extracting the parameters of the glv model
Beta=Theta(:,2:size(Theta,1)+1);
alpha=Theta(:,1);
if size(Theta,2)>size(Theta,1)+1
    Gamma=Theta(:,size(Theta,1)+2:size(Theta,2));
else
    Gamma=zeros(size(Theta,1),1);
end

%---------------------
% set up the species invading from outside
invaderID=[];
invaderDensity=[];
invaderTime=[];
addedToSystem=[];

if ~isempty(time_data)
    shift_time=(min(min(time_data)));
    shifted_time_data=time_data-shift_time;
    % find the species that are absent at the intial step and will come up in
    % the future (real invaders)
    idx_Invaders=find(taxa_data(:,1)==0);
    % find time at which they pop up and their density
    for is=1:length(idx_Invaders)
        idx_Invaders_not_zero=find(taxa_data(idx_Invaders(is),:)>0);
        if ~isempty(idx_Invaders_not_zero);
            invaderID(end+1)=idx_Invaders(is);
            value=taxa_data(idx_Invaders(is),idx_Invaders_not_zero(1));
            UL=value+variance_toIntroduce*value;
            LL=value-variance_toIntroduce*value;
            invaderDensity(end+1)=value;
            invaderTime(end+1)=shifted_time_data(idx_Invaders_not_zero(1))-dt;
            addedToSystem(end+1)=0;
        end
    end
end
if ~isempty(toIntroduce)
    for is=1:length(toIntroduce)
        invaderID(end+1)=toIntroduce(is);
        invaderTime(end+1)=time_toIntroduce(is);
        invaderDensity(end+1)=density_toIntroduce(is);
        addedToSystem(end+1)=0;
    end
end

%---------------------
% set the initial state for the integration
F=taxa_data(:,1);t=t0;cnt=1;nTaxa=size(taxa_data,1);
tt=t:dt:tmax;tspan=tt(1:end);
F1=F';t1=t;
%---------------------
% set up the perturbations time-series
if ~isempty(time_data)
    U2=zeros(size(U,1),length(tt));
    for iu=1:size(U,1)
        for it=2:length(shifted_time_data)
            idx_time_before_sampling=find(tt>=shifted_time_data(it-1)& tt<shifted_time_data(it));
            U2(:,idx_time_before_sampling)=repmat(U(:,it-1),1,length(idx_time_before_sampling));
        end
    end
else
    U2=U; %this is used when we are not simulating the original trajectory
    % but we are performing a numerical experiment with U provided from
    % outside
end

% setting the integrator structure
struct.Beta=Beta;
struct.alpha=alpha;
struct.Gamma=Gamma;
struct.U=U2;
struct.tspan=tspan;
%---------------------
% integration options
if o_use_implicit_solver
    options = odeset('RelTol',1e-10,'AbsTol',1e-10);
else
    options = odeset('RelTol',1e-8,'AbsTol',1e-8,'NonNegative',1:nTaxa);
end

%---------------------
% looping
while t1(end)<tmax
    % simulating invasion
    if ~isempty(invaderTime)
        idx_Invaders_toAdd=find(invaderTime<t);
        if ~isempty(idx_Invaders_toAdd)
            for jInv=1:length(idx_Invaders_toAdd)
                if addedToSystem(idx_Invaders_toAdd(jInv))~=1
                    F(invaderID(idx_Invaders_toAdd(jInv)))=...
                        F(invaderID(idx_Invaders_toAdd(jInv)))+invaderDensity(idx_Invaders_toAdd(jInv));
                    addedToSystem(idx_Invaders_toAdd(jInv))=1;
                end
            end
        end
    end
    cnt=cnt+1; % increase the counter

    isCatched=0; % processing integration
    try
        % negative species?
        negativeIdx0=find(F<=extinction_threshold);
        if ~isempty(negativeIdx0)
            F(negativeIdx0)=0;
        else
            negativeIdx0=[];
        end

        if o_use_implicit_solver
            % if using fully implicit ode15i
            [F,Fp] = decic(@sim_computeGeneralizedLV_Derivatives_ImplicitForm,...
                t,X,zeros(1,length(F)),zeros(1,length(F)),[],options,struct);

            [t,F]=ode15i(@sim_computeGeneralizedLV_Derivatives_ImplicitForm,...
                [t t+dt],F,Fp,options,struct);
        else
            [t,F]=ode15s(@sim_computeGeneralizedLV_Derivatives,...
                [t t+dt],F,options,struct);
        end
        warningMessID1='MATLAB:illConditionedMatrix';
        [warnmsg, msgid] = lastwarn;
        if strcmp(warningMessID1,msgid)==1
            lastwarn('');
            error('Exiting due to: %s',warningMessID1);
        end
    catch
        isCatched=1; % move to the next section
        disp('Catching ODE unfinished integration....');
        disp(F);
        lastF=F(end,:);
        lastt=t(end);

        F1(cnt,:)=lastF;
        t1(cnt)=lastt;
        tF_OUT=[t1;F1'];
        disp('Exiting at time');
        disp(lastt);
        return
    end
    % updating state variables after integration accomplished
    if isCatched==1
        disp('Debug: keep integrating after catching')
        disp(lastt);
    else
        isCatched=0;
    end

    lastF=F(end,:);
    lastt=t(end);

    % did numerical integration produced negative values? / if so set to 0
    neg_idx=find(lastF<=extinction_threshold);
    if ~isempty(neg_idx)
        lastF(neg_idx)=0;
    else
        neg_idx=[];
    end
    % record last iteration
    F1(cnt,:)=lastF;t1(cnt)=lastt;
    if break_tolerance~=-1
        idx=sum(sum(F1(cnt,:)>break_tolerance));
        if idx > 0
            disp ('Computation Blowing UP-->EXIT');
            isCompleted=0;
            return;
        end
    end
    % overwrite the working variables
    F=lastF;t=lastt;
end

% record everything vefore exiting
tF_OUT=[t1;F1'];
isCompleted=1;
disp('simulation done');
return


function [value,isterminal,direction] = event_function(t,X)
% when value is equal to zero, an event is triggered.
% set isterminal to 1 to stop the solver at the first event, or 0 to
% get all the events.
%  direction=0 if all zeros are to be computed (the default), +1 if
%  only zeros where the event function is increasing, and -1 if only
%  zeros where the event function is decreasing.
for i=1:length(X)
    value = X(i);
    if value >=0
        isterminal = 1; % terminate after the first event
        direction = -1;  % get all the zeros
    end
end
% categories: ODEs
% tags: reaction engineering
return

