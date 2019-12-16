function [Theta_samples_lasso] = MDSINELassoMCMC(X,F_hat_prime,growth_mean_p,growth_std_p,self_reg_mean_p,self_reg_std_p,perturb_std_p,numPerturb,node,params)
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


%% input:
%X = cell of design matrices (composed of concentration estimates)
%F_hat_prime = cell of derivative estimate vectors
%prior means and variances for parameters

%% output:
%Theta_samples_lasso = MCMC samples of parameter vectors (as matrix)


%params = readParameterFile(paramFileName);

numIters = params.numIters;
burnin = params.numBurnin;

Theta_samples_lasso = cell(numIters-burnin,1);
numOTUs = length(F_hat_prime);

data_std = ones(numOTUs,1)*params.data_std_init;

growth_mean = growth_mean_p;
growth_std = growth_std_p*100;
self_reg_mean = self_reg_mean_p;
self_reg_std = self_reg_std_p*100;
perturb_std = perturb_std_p*100;

% coefficient matrix
% fill with small values to avoid numerical issues
Theta = ones(numOTUs,numOTUs+1+numPerturb)*10^-10;

% set OTU coefficients
for o=1:numOTUs,
    Theta_0 = pinv(X{o})*F_hat_prime{o};
    Theta_0(1) = abs(Theta_0(1));
    Theta_0(o+1) = -abs(Theta_0(o+1));
    Theta(o,:) = Theta_0;
end;

% prior variance on growth, self, and interaction parameters
tau_growth = ones(numOTUs,1)*growth_std^2;
tau_self = ones(numOTUs,1)*self_reg_std^2;
tau_interact = ones(numOTUs,numOTUs-1)*self_reg_std^2;

tau_perturb = [];
if numPerturb > 0,
    tau_perturb = ones(numOTUs,numPerturb)*perturb_std^2;
end;

% hyperparameters for gamma prior on lambda_interact (lasso on interactions)
lambda_interact = ones(size(Theta))/self_reg_mean^2;
gpB_lambda_interact = lambda_interact(1,1)*10^10;
gpA_lambda_interact = 1/10^10;

lambda_perturb = 0;
gpB_lambda_perturb = 0;
gpA_lambda_perturb = 0;
if numPerturb > 0,
    for o=1:numOTUs,
        lambda_perturb = [lambda_perturb Theta(o,(numOTUs+2):size(Theta,2)).^2];
    end;
    lambda_perturb = 1/mean(lambda_perturb);
    gpB_lambda_perturb = lambda_perturb*10^10;
    gpA_lambda_perturb = 1/10^10;
end;

numUseIter = 0;
for i=1:numIters,
    for o=1:numOTUs,
        % sample interaction parameters from posterior conditioned on
        % growth and self-interaction terms

        % construct full posterior
        % indices of interaction terms
        e_idx = setdiff(1:(numOTUs+numPerturb),o);
        e_idx_interact = setdiff(1:numOTUs,o);
        % prior variances
        cv = ones(1,numOTUs);
        cv(o) = tau_self(o);
        cv(e_idx_interact) = tau_interact(o,:);
        if numPerturb > 0,
            cv = [tau_growth(o) cv tau_perturb(o,:)];
        else,
            cv = [tau_growth(o) cv];
        end;
        % prior means
        m0 = zeros(1,numOTUs);
        m0(o) = self_reg_mean;
        if numPerturb > 0,
            m0 = [growth_mean m0 zeros(1,numPerturb)];
        else,
            m0 = [growth_mean m0];
        end;
        
        % 
        %X{o} = X{o} + 1e-5*rand(size(X{o}));
        
        % prior covariance matrix
        Sigma0_inv = diag(1./cv);
        % posterior covariance matrix
        tmp=X{o}'*X{o};
        if ~isempty(find(isnan(tmp)))
            lixo =1;
        end
        C = inv_chol(Sigma0_inv + X{o}'*X{o}/data_std(o)^2);
        % posterior mean
        m = C*(X{o}'*F_hat_prime{o}/data_std(o)^2 + Sigma0_inv*m0');

        % sample interactions conditional on growth and self parameters
        try
            [m_cond,C_cond] = condMVN(m,C,Theta(o,[1 o+1])',[1 o+1]);
            samp = mvnrnd(m_cond,C_cond)';
            Theta(o,e_idx+1) = samp;
        catch errM
            os = sprintf('err=%s',errM.message);
        %    disp(os);
        end
        % sample growth conditional on other parameters
        [m_cond,C_cond] = condMVN(m,C,Theta(o,2:(numOTUs+1+numPerturb))',2:(numOTUs+1+numPerturb));
        
        % this is to avoid negative std (when numerically is very small)
        C_cond=max(C_cond,0);
        
        Theta(o,1) = truncGaussianSample(m_cond,sqrt(C_cond),0,0);
        if isinf(Theta(o,1)),
            Theta(o,1) = growth_mean_p;
        end;

        % sample self conditional on other parameters
        [m_cond,C_cond] = condMVN(m,C,Theta(o,[1 e_idx+1])',[1 e_idx+1]);
        
        % this is to avoid negative std (when numerically is very small)
        C_cond=max(C_cond,0);

        Theta(o,o+1) = truncGaussianSample(m_cond,sqrt(C_cond),0,1);
        if isinf(Theta(o,o+1)),
            Theta(o,o+1) = self_reg_mean_p;
        end;
    end;

    % sample tau interact
    for o=1:numOTUs,
        e_idx = setdiff(1:numOTUs,o)+1;
        ig_mu = sqrt(lambda_interact(o,e_idx))./abs(Theta(o,e_idx));
        ig_lamb = lambda_interact(o,e_idx);
        tau_interact(o,:) = ones(size(tau_interact(o,:)))./sample_inverse_gaussian(ig_mu,ig_lamb);

        % sample lambda_interact
        lambda_interact(o,e_idx) = gamrnd(gpA_lambda_interact + ones(size(tau_interact(o,:))),1.0./(tau_interact(o,:)/2.0 + 1.0/gpB_lambda_interact));
    end;

    % sample tau perturb
    if numPerturb > 0,
        for o=1:numOTUs,
            e_idx = ((numOTUs+1):(numOTUs+numPerturb)) + 1;
            ig_mu = sqrt(lambda_perturb)./abs(Theta(o,e_idx));
            ig_lamb = lambda_perturb;
            tau_perturb(o,:) = 1./sample_inverse_gaussian(ig_mu,ig_lamb);
        end;
        lambda_perturb = gamrnd(gpA_lambda_perturb + numOTUs*numPerturb,1.0./(sum(sum(tau_perturb))/2.0 + 1.0./gpB_lambda_perturb));
    end;

    % sample data std
    for o=1:numOTUs,
        dv = X{o}*Theta(o,:)'-F_hat_prime{o};
        data_std(o) = sqrt(1/gamrnd((length(dv)+1)/2 + 1,2/sum(dv.^2)));
    end;

    if i > burnin,
        numUseIter = numUseIter + 1;
        Theta_samples_lasso{numUseIter} = Theta;
    end;

    if mod(i,100) == 0
        %ooip = sprintf('iter=%i lambda_interact=%f',i,lambda_interact);
        ooip = sprintf('MDSINE lasso node=%i iter=%i',node,i);
        %disp(ooip);
    end;
end;
