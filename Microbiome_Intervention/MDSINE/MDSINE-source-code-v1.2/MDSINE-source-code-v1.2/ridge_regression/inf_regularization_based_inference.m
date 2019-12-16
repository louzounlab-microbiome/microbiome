function [regularizer_global,Theta_global,Theta_mean]=...
    inf_regularization_based_inference(~,F,t,ID,U,F_prime,X,mesh,...
    k,o_keep_all_trajectory,o_force_constraint,constraint_algorithm,...
    o_use_initial_guess,nCores,rep,magnitude)
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


% Vanni Bucci, Richard Stein
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

% use parallel or not
if ~isempty(nCores)
    poolobj = gcp('nocreate'); % If no pool, do not create new one.
    if isempty(poolobj)
        poolsize = 0;
    else
        poolsize = poolobj.NumWorkers;
    end
    if  poolsize == 0
        % comment this out if you do not have the parallel computing
        % toolbox
        parpool('local',nCores);
    end
else
    poolobj = gcp('nocreate');
end

if ~isempty(U)
    res_total=zeros(length(mesh),length(mesh),length(mesh));
else
    res_total=zeros(length(mesh),length(mesh));
end

% if ~isempty(myseed)
%     rng(myseed); % initialize the random number seed
% else
%     rng('shuffle');
% end

if ~isempty(nCores)
    parfor ir=1:rep
        [res(ir).value,~]=inf_perform_regularized_crossvalidation(k,ir,mesh,ID,X,F_prime,size(F,1),...
            o_keep_all_trajectory,o_force_constraint,constraint_algorithm,o_use_initial_guess,nCores);
    end
else
    for ir=1:rep
        [res(ir).value,~]=inf_perform_regularized_crossvalidation(k,ir,mesh,ID,X,F_prime,size(F,1),...
            o_keep_all_trajectory,o_force_constraint,constraint_algorithm,o_use_initial_guess,nCores);
    end
end

Theta_seed=zeros(size(F,1),size(F,1)+1+size(U,1),rep);

for i=1:rep
    if numel(size(res(i).value))>2
        [I,J,K] = ind2sub(size(res(i).value),find(res(i).value==min(min(min(res(i).value)))));
    else
        [I,J] = ind2sub(size(res(i).value),find(res(i).value==min(min(min(res(i).value)))));
    end
    %
    for l=1:length(I)
        if numel(size(res(i).value))>2
            Minimal_res=res(i).value(I(l),J(l),K(l));
            Optimal_regularizer=[mesh(I(l)),mesh(J(l)),mesh(K(l))];
        else
            Minimal_res=res(i).value(I(l),J(l));
            Optimal_regularizer=[mesh(I(l)),mesh(J(l))];
        end
        fprintf('Repetition %d - Optimal lambda: %d\t %d\t %d\n',i,Optimal_regularizer);
        fprintf('Repetition %d - Minimal residue: %d\n',i, Minimal_res);
    end

    if o_force_constraint
        Theta_seed(:,:,i)=...
            constrained_solution_MetaInfer(Optimal_regularizer,X,F_prime,....
        o_use_initial_guess,constraint_algorithm,nCores);
    else
        Theta_seed(:,:,i)=tikhonov_MetaInfer(Optimal_regularizer,X,F_prime);
    end
    res_total=res_total+res(i).value;
end
res_total=res_total/rep;

if ~isempty(U)
    [I,J,K] = ind2sub(size(res_total),find(res_total==min(min(min(res_total)))));
    regularizer_global=[mesh(I),mesh(J),mesh(K)];
    for l=1:length(I)
        fprintf('\n')
        fprintf('Global Optimal Regularizer: %d\t %d\t %d\n',mesh(I(l)),mesh(J(l)),mesh(K(l)))
        fprintf('Global Minimal Residue: %d\n',res_total(I(l),J(l),K(l)))
    end
else
    [I,J] = ind2sub(size(res_total),find(res_total==min(min(min(res_total)))));
    regularizer_global=[mesh(I),mesh(J)];
    for l=1:length(I)
        fprintf('\n')
        fprintf('Global Optimal Regularizer: %d\t %d\n',mesh(I(l)),mesh(J(l)))
        fprintf('Global Minimal Residue: %d\n',res_total(I(l),J(l)))
    end
end

% get the Global parameter set
if o_force_constraint
    Theta_global=constrained_solution_MetaInfer(regularizer_global,X,F_prime,....
        o_use_initial_guess,constraint_algorithm,nCores);
else
    Theta_global=tikhonov_MetaInfer(regularizer_global,X,F_prime);
end

% calculate the mean parameters (averages across the different seeds)
Theta_mean=mean(Theta_seed,3);

% rescale by the magnitude to get into the original data densities
Theta_global(:,2:size(Theta_global,1)+1)=Theta_global(:,2:size(Theta_global,1)+1)./10^magnitude;
Theta_mean(:,2:size(Theta_mean,1)+1)=Theta_mean(:,2:size(Theta_mean,1)+1)./10^magnitude;
return
