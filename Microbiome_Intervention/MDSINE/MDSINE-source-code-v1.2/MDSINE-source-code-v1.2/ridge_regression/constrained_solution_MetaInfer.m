function [Theta_sol]=...
    constrained_solution_MetaInfer(rp,X,F_prime,o_use_initial_guess,algorithm,nCores)
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

% July 29 2015:
% Vanni Bucci, Richard Stein, Matt Simmons, Shakti Battarai
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
% ------------------------------------------------------------------------------------------------
% This function uses numerical techniques for contrained minimization
% to solve the problem norm(Theta*X-F_prime','fro')^2+norm(Theta*D.^(1/2),'fro')^2
% under parameters constraints (beta_ii<=0, alpha_i>=0)
% given Theta = [alpha beta gamma]
% The functions can perform the constrained inference using two different
% algorithms.
% ############################
% Option algorithm = 'fmincon'
% we minimize f = sum(sum(Theta*X-F_prime).^2)+sum(sum(Theta.^2*D))
% under the derivative d = 2 * (X * X' + D) * Theta'-2 * X * F_prime'
% constraints are:
% ub = zeros(size(Theta))
% is = find(ub==0)
% ub(is) = Inf
% lb = zeros(size(Theta))
% is = find(lb==0)
% lb(is) = -Inf
% This function uses MATLAB fmincon as solver
% ############################
% Option algorithm = 'qp' (quadratic programming)
% we minimize argimin(x) {1/2 a' * H * a + f' * a}
% where a = 2 * vec_theta_t;
% f = -2 * X2'* vec_F_prime;
% and H = 2 * (X2' * X2 + D2_lambda)
% constraints are:
%-----------------------------------------------------------------------------------------------
L=size(F_prime,1); % number of variables
nrx=size(X,1);
ncx=size(X,2);
P=nrx-1-L; % number of perturbations

% building the D matrix
D=zeros(nrx,nrx);
D(1,1)=rp(1);
D(nrx+2:nrx+1:(nrx+1)*L+1)=rp(2);
if nrx>L+1
    D(1+(nrx+1)*(L+1):nrx+1:end)=rp(3);
end

% ridge solution to start with
Theta_ridge=F_prime*X'*pinv(X*X'+D);

Theta_ridge_growth_neg=find(Theta_ridge(:,1)<0,1);
Theta_ridge_interaction_diag_pos=find(diag(Theta_ridge(1:L,2:L+1))>0,1);

if isempty(Theta_ridge_growth_neg) && isempty(Theta_ridge_interaction_diag_pos)
    Theta_sol=Theta_ridge;
    %disp('No optimization needed. Ridge-solution satisfies constraints');
    return
end

switch algorithm
    case 'fmincon'
        % initial point = ridge solution with adapted growth rates and diagonal elements
        if o_use_initial_guess
            Theta0=Theta_ridge;
        else
            Theta0=[];
        end

        % define constraints
        ub=Inf(L,nrx);
        lb=-Inf(L,nrx);
        ub(L+1:L+1:(L+1)*L+1)=0; %beta_ii <=0
        lb(:,1)=0; % alpha_i >=0

        options = optimoptions('fmincon','Algorithm','trust-region-reflective',...
            'GradObj','on','TolCon',1e-10,'TolFun',1e-10,'TolX',1e-10,'FinDiffType','central',...
            'MaxIter',1000);

        if ~isempty(nCores)
            options = optimoptions(options,'UseParallel',true);
        end

        [Theta_sol,~,exitflag] =...
            fmincon(@(Theta)objfun(Theta,X,F_prime,D),Theta0,[],[],[],[],lb,ub,[],options);

        % evaluate the obj function on the ridge solution
        f_ridge = objfun(Theta_ridge,X,F_prime,D);

        % evaluate the obj function on the constrained solution
        f_cons = objfun(Theta_sol,X,F_prime,D);

    case 'qp'
        % prepare the matrices for quadratic programming
        X2=kron(eye(L,L),X');
        D2=kron(eye(L,L),D);

        H = 2 * (X2'*X2 + D2);

        % check for positive define matrix
        [~,p] = chol(H);

        if p ~=0
            disp ('Warning:')
            disp ('Metainfer -- constrained quadratic programming...');
            disp ('H matrix is not positive definite')
            disp ('')
        end

        Theta_ridge_t=Theta_ridge';
        vec_theta_t=Theta_ridge_t(:);
        a=vec_theta_t;
        F_prime_t=F_prime';
        vec_F_prime_t=F_prime_t(:);
        f=-2*X2'*vec_F_prime_t;

        % define constraints
        ub=Inf(L,nrx);
        lb=-Inf(L,nrx);
        ub(L+1:L+1:(L+1)*L+1)=0; %beta_ii <=0
        lb(:,1)=0; % alpha_i >=0

        ub_t=ub';
        lb_t=lb';
        vec_ub_t=ub_t(:);
        vec_lb_t=lb_t(:);

        options0 = optimoptions('quadprog','Display','off','Algorithm',...
            'trust-region-reflective','MaxIter',10000,'TolFun',1e-14);

        if o_use_initial_guess
            [a_sol,~,~,~,~] = quadprog(H, f, [], [], [], [], vec_lb_t, vec_ub_t, a, options0);
        else
            [a_sol,~,~,~,~] = quadprog(H, f, [], [], [], [], vec_ub_t, vec_lb_t, [], options0);
        end

        Theta_sol=(reshape(a_sol,L+P+1,L))';
end


return

function [f,g] = objfun(Theta,X,F_prime,D)

%objective function to minimize

% July 29 2015:
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

% Evaluate objective function
f = sum(sum((Theta*X-F_prime).^2))+sum(sum(Theta.^2*D));

% Evaluate derivative of objective function
 g = 2 * (X * X' + D) * Theta' -2 * X * F_prime'; % Richard derivation
return

