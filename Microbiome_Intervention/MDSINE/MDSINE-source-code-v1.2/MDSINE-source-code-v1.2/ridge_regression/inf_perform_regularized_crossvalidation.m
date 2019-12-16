function [res,pars]=inf_perform_regularized_crossvalidation(k,irep,mesh,ID,X,F_prime,L,o_keep_all_trajectory...
    ,o_force_constraint,constraint_algorithm,o_use_initial_guess,nCores)
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

%rng(irep); % initialize the random number seed

K=length(mesh); % total number of grid points for lambda search
if size(X,1)>L+1
    res=zeros(K,K,K); % there are perturbations
else
    res=zeros(K,K); % there are no perturbations
end
pars=[];

if o_keep_all_trajectory
    if k>length(unique(ID))
       error('k larger than number of samples')
    end
    [IDout,group]=...
        inf_determineCVgroups(unique(ID),k); % CV performed usign whole traj
else
    [IDout,group]=...
        inf_determineCVgroups(ID,k); % CV performed chopping sample in groups
end

% disp(irep);
% disp(IDout);
% disp(group);

uGroup=unique(group);

% Inferring along the regularizer mesh
if numel(size(res))==3 % 3D - accounting for perturbations
    for i1=1:K
        for i2=1:K
            for i3=1:K
                for i=1:length(uGroup)
                    % training
                    indGroup=find(group~=uGroup(i));
                    if o_keep_all_trajectory
                        trainID=IDout(indGroup);
                        ind_traiIDg=[];
                        for j=1:length(trainID)
                            indTrainID=find(trainID(j)==ID);
                            ind_traiIDg=[ind_traiIDg indTrainID];
                        end
                    else
                        ind_traiIDg=indGroup;
                    end
                    if o_force_constraint
                        % constrained solution to select for positive growth and negative
                        % self interaction
                        Theta=constrained_solution_MetaInfer([mesh(i1) mesh(i2) mesh(i3)],...
                            X(:,ind_traiIDg),F_prime(:,ind_traiIDg),o_use_initial_guess,constraint_algorithm,nCores);
                    else
                        % classic inference (no prior knowledge of parameters)
                        Theta=tikhonov_MetaInfer([mesh(i1) mesh(i2) mesh(i3)],...
                            X(:,ind_traiIDg),F_prime(:,ind_traiIDg));
                    end
                    ind_valID=setdiff(1:1:length(ID),ind_traiIDg);
                    pars(i1,i2,i3).Theta(uGroup(i)).val=Theta;
                    res(i1,i2,i3)=res(i1,i2,i3)...
                        +norm(Theta*X(:,ind_valID)-F_prime(:,ind_valID),'fro')^2/(size(F_prime(:,ind_valID),2)*L);
                end
                res(i1,i2,i3)=res(i1,i2,i3)/length(uGroup);
            end
        end
    end
elseif numel(size(res))==2 % without perturbations
    for i1=1:K
        for i2=1:K
            for i=1:length(uGroup)
                % training
                indGroup=find(group~=uGroup(i));
                if o_keep_all_trajectory
                    trainID=IDout(indGroup);
                    ind_traiIDg=[];
                    for j=1:length(trainID)
                        indTrainID=find(trainID(j)==ID);
                        ind_traiIDg=[ind_traiIDg indTrainID];
                    end
                else
                    ind_traiIDg=indGroup;
                end
                if o_force_constraint
                    % constrained solution to select for positive growth and negative
                    % self interaction
                    Theta=constrained_solution_MetaInfer([mesh(i1) mesh(i2)],...
                        X(:,ind_traiIDg),F_prime(:,ind_traiIDg),o_use_initial_guess,constraint_algorithm,nCores);
                else
                    % classic inference (no prior knowledge of parameters)
                    Theta=tikhonov_MetaInfer([mesh(i1) mesh(i2)],...
                        X(:,ind_traiIDg),F_prime(:,ind_traiIDg));
                end
                ind_valID=setdiff(1:1:length(ID),ind_traiIDg);
                pars(i1,i2).Theta(uGroup(i)).val=Theta;
                res(i1,i2)=res(i1,i2)...
                    +norm(Theta*X(:,ind_valID)-F_prime(:,ind_valID),'fro')^2/(size(F_prime(:,ind_valID),2)*L);

            end
            res(i1,i2)=res(i1,i2)/length(uGroup);
        end
    end
end

return
