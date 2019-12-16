function [Theta_select_probs,Theta_select,Theta_samples_select,Theta_bayes_factors] = MDSINESelectMCMC(X,F_hat_prime,growth_mean_p,growth_std_p,self_reg_mean_p,interact_std_p,perturb_std_p,numPerturb,nodeNum,Theta_init,Theta_indicator_init,params)
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


% inputs:
% X = matrix w/ trajectory estimates
% F_hat_prime = matrix w/ derivative estimates
% growth_mean_p = hyperparameter for prior for growth parameters mean
% growth_std_p = hyperparameter for prior for growth parameters std
% self_reg_mean_p = hyperparameter for prior for self-regulation parameters
% mean
% interact_std_p = hyperparameter for prior for interaction parameters std
% nodeNum = node number (set to 1, used for debugging/parallel applications
% to ID run #)
% Theta_init = matrix (dim # OTUs X # OTUs + 1; first col is growth parameters, subsequent columns are
% interaction parameters) w/ initial guess for growth/interaction parameters
% Theta_indicator_init = matrix (dim # OTUs X # OTUs) w/ initial guess for
% which interactions occur (entry of 1 indicates an interaction, zero
% otherwise)

% outputs:
% Theta_select_probs = posterior probabilities of interactions
% Theta_select = posterior expectations of interaction parameters
% Theta_samples_select = MCMC samples of growth/interaction parameters

%params = readParameterFile(paramsFileName);

numIters = params.numIters;
burnin = params.numBurnin;
data_std_init = params.data_std_init;

Theta_samples_select = cell(numIters-burnin,1);
numOTUs = length(F_hat_prime);

data_std = data_std_init*ones(numOTUs,1);

growth_mean = growth_mean_p;
growth_std_p = growth_std_p*100.0;
growth_std = growth_std_p;
self_reg_mean = self_reg_mean_p;
self_reg_std_p = interact_std_p*100.0;
self_reg_std = self_reg_std_p;

interact_std_p = interact_std_p*100.0;
interact_std = interact_std_p;

perturb_std_p = perturb_std_p*100.0;
perturb_std = perturb_std_p;

% prior on interactions
interact_beta_a = params.interact_beta_a;
interact_beta_b = params.interact_beta_b;

interact_beta = interact_beta_a/(interact_beta_a + interact_beta_b);

perturb_beta_a = 0;
perturb_beta_b = 0;
perturb_beta = 0;

if numPerturb > 0,
    perturb_beta_a = params.perturb_beta_a;
    perturb_beta_b = params.perturb_beta_b;
    perturb_beta = perturb_beta_a/(perturb_beta_a + perturb_beta_b);
end;

% coefficient matrix
Theta = Theta_init;
Theta(:,2:(numOTUs+1+numPerturb)) = Theta(:,2:(numOTUs+1+numPerturb)).*Theta_indicator_init;
Theta_select = zeros(size(Theta));
% sparsity matrix
AS = Theta_indicator_init;
Theta_select_probs = zeros(size(AS));

% needed for marginalizing over parameters
VD = zeros(1,numOTUs);
DVNI = zeros(1,numOTUs);
for o=1:numOTUs,
    VD(o) = F_hat_prime{o}'*F_hat_prime{o};
    DVNI(o) = sum((F_hat_prime{o}-X{o}(:,[1 o+1])*Theta(o,[1 o+1])').^2);
end;

% find any zero covariates so that these are never selected
no_interact = zeros(numOTUs,numOTUs+numPerturb);
for ox=1:numOTUs,
    for o2x=1:(numOTUs+numPerturb),
        if sum(X{ox}(:,o2x+1)) == 0,
            no_interact(ox,o2x) = 1;
            AS(ox,o2x) = 0;
            Theta(ox,o2x+1) = 0;
        end;
    end;
end;

numUseIter = 0;
for i=1:numIters,
    for ox=1:numOTUs,
        o = ox;
        rot = randperm(numOTUs+numPerturb);
        for o2x=1:(numOTUs+numPerturb),
            o2 = rot(o2x);
            %o2 = o2x;
            if o2 ~= o && ~no_interact(o,o2),
                try
                    cssd = ones(1,numOTUs)*interact_std^2;
                    cssd(o) = self_reg_std^2;
                    if numPerturb > 0,
                        cssd = [cssd ones(1,numPerturb)*perturb_std^2];
                    end;

                    % determine marginal likelihood under current setting
                    A_use = find(AS(o,:)>0);
                    Sigma0_inv = diag(1./[growth_std^2 cssd(A_use)]);
                    ca = [growth_mean zeros(1,numOTUs+numPerturb)];
                    ca(o+1) = self_reg_mean;
                    u0 = ca([1 A_use+1]);
                    p2 = marginalLL(A_use,Sigma0_inv,u0,o);

                    % determine marginal likelihood if we flip the OTU bit
                    AST = AS(o,:);
                    AST(o2) = ~AST(o2);
                    A_use = find(AST>0);
                    Sigma0_inv = diag(1./[growth_std^2 cssd(A_use)]);
                    ca = [growth_mean zeros(1,numOTUs+numPerturb)];
                    ca(o+1) = self_reg_mean;
                    u0 = ca([1 A_use+1]);
                    p1 = marginalLL(A_use,Sigma0_inv,u0,o);

                    if AS(o,o2) == 1,
                        if o2 > numOTUs,
                            aprn = perturb_beta;
                            apr = 1-perturb_beta;
                        else,
                            aprn = interact_beta;
                            apr = 1-interact_beta;
                        end;
                    else,
                        if o2 > numOTUs,
                            aprn = 1-perturb_beta;
                            apr = perturb_beta;
                        else,
                            aprn = 1-interact_beta;
                            apr = interact_beta;
                        end;
                    end;

                    r = min(1,exp(p1 + log(aprn) - p2 - log(apr)));
                    u = unifrnd(0,1);

                    % accept the move
                    if u<=r,
                        if AS(o,o2) == 1,
                            AS(o,o2) = 0;
                            Theta(o,o2+1) = 0.0;
                        else,
                            AS(o,o2) = 1;
                        end;
                    end;
                catch errM
                    os = sprintf('node=%i err=%s',nodeNum,errM.message);
                    %disp(os);
                end
            end;
        end;
    end;

    % sample interaction prior
    atn = sum(AS);
    atn = atn(2:length(atn));
    % don't double count self-interaction terms
    atn = sum(atn - ones(size(atn)));
    TPN = (numOTUs-1)*numOTUs;
    interact_beta = betarnd(interact_beta_a + TPN - atn,interact_beta_b + atn);

    % sample interaction prior
    atn = sum(AS);
    atn = atn(2:(length(atn)-numPerturb));
    % don't double count self-interaction terms
    atn = sum(atn - ones(size(atn)));
    TPN = (numOTUs-1)*numOTUs;
    interact_beta = betarnd(interact_beta_a + TPN - atn,interact_beta_b + atn);

    % sample perturb prior
    if numPerturb > 0,
        atn = sum(AS);
        atn = atn(((numOTUs+1):(numOTUs+numPerturb)));
        atn = sum(atn);
        TPN = numOTUs*numPerturb;
        perturb_beta = betarnd(perturb_beta_a + TPN - atn,perturb_beta_b + atn);
    end;

    % update regression coefficients given sparsity patterns
    for ox=1:numOTUs,
        o = ox;
        try
            A_use = find(AS(o,:)>0);
            aex = [1 A_use+1];
            cssd = ones(1,numOTUs)*interact_std^2;
            cssd(o) = self_reg_std^2;
            if numPerturb > 0,
                cssd = [cssd ones(1,numPerturb)*perturb_std^2];
            end;

            if length(A_use) > 1,
                % update interaction parameters
                Bs = X{o}(:,[1 A_use+1]);
                Sigma0_inv = diag(1./[growth_std^2 cssd(A_use)]);
                CI = (Sigma0_inv + Bs'*Bs/data_std(o)^2);
                C = inv_chol(CI);
                ca = [growth_mean zeros(1,numOTUs+numPerturb)];
                ca(o+1) = self_reg_mean;

                m = C*(Sigma0_inv*ca([1 A_use+1])' + Bs'*F_hat_prime{o}/data_std(o)^2);

                f = find(aex == o+1);
                known_idx = [1 f];
                unknown_idx = setdiff(1:size(Bs,2),known_idx);
                CPI = inv_chol(C(known_idx,known_idx));

                m_prime = m(unknown_idx) + C(unknown_idx,known_idx)*CPI*(Theta(o,aex(known_idx))'-m(known_idx));
                C_prime = inv_chol(CI(unknown_idx,unknown_idx));

                Theta(o,aex(unknown_idx)) = mvnrnd(m_prime,C_prime);
            end;
        catch errM
            os = sprintf('node=%i err=%s',nodeNum,errM.message);
        %    disp(os);
        end

        % update growth parameter
        try
            Bs = X{o}(:,[1 A_use+1]);
            Sigma0_inv = diag(1./[growth_std^2 cssd(A_use)]);
            CI = (Sigma0_inv + Bs'*Bs/data_std(o)^2);
            C = inv_chol(CI);
            ca = [growth_mean zeros(1,numOTUs+numPerturb)];
            ca(o+1) = self_reg_mean;

            m = C*(Sigma0_inv*ca([1 A_use+1])' + Bs'*F_hat_prime{o}/data_std(o)^2);

            unknown_idx = 1;
            known_idx = setdiff(1:size(Bs,2),unknown_idx);
            CPI = inv_chol(C(known_idx,known_idx));

            m_prime = m(unknown_idx) + C(unknown_idx,known_idx)*CPI*(Theta(o,aex(known_idx))'-m(known_idx));
            C_prime = inv_chol(CI(unknown_idx,unknown_idx));
            Theta(o,aex(unknown_idx)) = truncGaussianSample(m_prime,sqrt(C_prime),0,0);
            if isinf(Theta(o,aex(unknown_idx))),
                Theta(o,aex(unknown_idx)) = growth_mean_p;
            end;
        catch errM
            os = sprintf('node=%i err=%s',nodeNum,errM.message);
        %    disp(os);
        end

        % update self-regulation parameter
        try
            Bs = X{o}(:,[1 A_use+1]);
            Sigma0_inv = diag(1./[growth_std^2 cssd(A_use)]);
            CI = (Sigma0_inv + Bs'*Bs/data_std(o)^2);
            C = inv_chol(CI);
            ca = [growth_mean zeros(1,numOTUs+numPerturb)];
            ca(o+1) = self_reg_mean;

            m = C*(Sigma0_inv*ca([1 A_use+1])' + Bs'*F_hat_prime{o}/data_std(o)^2);

            unknown_idx = find(aex == o+1);
            known_idx = setdiff(1:size(Bs,2),unknown_idx);
            CPI = inv_chol(C(known_idx,known_idx));

            m_prime = m(unknown_idx) + C(unknown_idx,known_idx)*CPI*(Theta(o,aex(known_idx))'-m(known_idx));
            C_prime = inv_chol(CI(unknown_idx,unknown_idx));
            Theta(o,aex(unknown_idx)) = truncGaussianSample(m_prime,sqrt(C_prime),0,1);
            if isinf(Theta(o,aex(unknown_idx))),
                Theta(o,aex(unknown_idx)) = self_reg_mean_p;
            end;
        catch errM
            os = sprintf('node=%i err=%s',nodeNum,errM.message);
        %    disp(os);
        end

        DVNI(o) = sum((F_hat_prime{o}-X{o}(:,[1 o+1])*Theta(o,[1 o+1])').^2);
    end;

    % sample data std
    for o=1:numOTUs,
        dv = X{o}*Theta(o,:)'-F_hat_prime{o};
        data_std(o) = sqrt(1/gamrnd((length(dv)+1)/2 + 1,2/sum(dv.^2)));
    end;

    % update growth mean and std
    gpp = sampleMeanPosterior(Theta(:,1)',growth_std,growth_mean_p,growth_std_p);
    if gpp > 0,
        growth_mean = gpp;
    end;
    growth_std = sampleSTDPosterior(Theta(:,1),growth_mean,growth_std_p,3);
    % update self-reg mean and std
    gpp = sampleMeanPosterior(diag(Theta(1:numOTUs,2:(numOTUs+1)))',self_reg_std,self_reg_mean_p,self_reg_std_p);
    if gpp < 0,
        self_reg_mean = gpp;
    end;
    self_reg_std = sampleSTDPosterior(diag(Theta(1:numOTUs,2:(numOTUs+1))),self_reg_mean_p,self_reg_std_p,3);
    % update interaction std
    at = [zeros(numOTUs,1) (AS - eye(size(AS))).*(~no_interact)];
    if numPerturb > 0,
        at = [at zeros(numOTUs,numPerturb)];
    end;
    f = find(at > 0);
    cofd = Theta(f);
    cofd = reshape(cofd,size(cofd,1)*size(cofd,2),1);
    interact_std = sampleSTDPosterior(cofd,0,interact_std_p,3);

    % update perturb std
    if numPerturb > 0,
        f = find(AS(1:numOTUs,((numOTUs+1):(numOTUs+numPerturb))) > 0);
        cofd = Theta(1:numOTUs,((numOTUs+1):(numOTUs+numPerturb)) + 1);
        cofd = cofd(f);
        if ~isempty(cofd),
            cofd = reshape(cofd,size(cofd,1)*size(cofd,2),1);
            perturb_std = sampleSTDPosterior(cofd,0,perturb_std_p,3);
        end;
    end;

    if i > burnin,
        numUseIter = numUseIter + 1;
        Theta_select_probs = Theta_select_probs + AS;
        Theta_select = Theta_select + Theta;
        Theta_samples_select{numUseIter} = Theta;
    end;

    if mod(i,100) == 0,
        ooip = sprintf('MDSINE select node=%i iter=%i beta_interact=%f beta_perturb=%f interacts=%i growth_mean=%f growth_std=%f self_reg_mean=%f self_reg_std=%f interact_std=%f',nodeNum,i,interact_beta,perturb_beta,sum(sum(AS))-numOTUs,growth_mean,growth_std,self_reg_mean,self_reg_std,interact_std);
    %    disp(ooip);
    end;
end;
Theta_select_probs = Theta_select_probs/numUseIter;
Theta_select = Theta_select/numUseIter;

Theta_bayes_factors = calcBayesFactors(Theta_select_probs,interact_beta_a,interact_beta_b,perturb_beta_a,perturb_beta_b,numPerturb);

    function [ML] = marginalLL(A_use,Sigma0_inv,u0,o)
        aex = [1 A_use+1];
        Bs = X{o}(:,aex);
        Bss = Bs'*Bs;
        CI = (Sigma0_inv + Bss/data_std(o)^2);
        C = inv_chol(CI);

        m = C*(Sigma0_inv*u0'+ Bs'*F_hat_prime{o}/data_std(o)^2);

        f = find(aex == o+1);
        known_idx = [1 f];
        unknown_idx = setdiff(1:size(Bs,2),known_idx);

        ML = log(sqrt(det(C))) - log(sqrt(det(inv(Sigma0_inv))));
        ML = ML - 0.5*u0*Sigma0_inv*u0' + 0.5*m'*CI*m - 0.5*VD(o)/data_std(o)^2;

        m_prime = m(known_idx);
        C_prime = C(known_idx,known_idx);
        ML = ML + log(mvnpdf(Theta(o,aex(known_idx))',m_prime,C_prime));
    end
end

function [s] = sampleSTDPosterior(X,mu,s_p,dof)
    % sample standard deviation assuming normal likelihood and scaled
    % inverse chi^2 prior
    % X are the values of the 'data' RV
    % mu is the mean of the 'data' RV
    % s_p is the prior mean STD
    % dof is the degrees of freedom for the scaled inverse chi^2

    n = length(X);
    if n > 0,
        ss = sum((X-mu).^2);
    else,
        ss = 0;
    end;

    ss = (ss + dof*s_p^2)/2.0;
    s = gamrnd((dof+n)/2.0+2.0,1/ss);
    s = sqrt(1/s);
end

function [m] = sampleMeanPosterior(X,s,mu_p,s_p)
    % sample mean assuming normal likelihood and normal prior
    % X are the values of the 'data' RV (can be matrix in which each row is
    % an R.V.)
    % s is the STD for the 'data' RV
    % mu_p is the prior mean
    % s_p is the prior STD

    n = size(X,2);
    if n > 0,
        mus = sum(X,2);
    else,
        mus = zeros(size(X,1),1);
    end;
    % posterior variance
    v_pos = 1/(1/s_p^2 + n/s^2);
    % posterior mean
    m_pos = (mu_p/s_p^2 + mus/s^2)*v_pos;
    m = normrnd(m_pos,sqrt(v_pos)*ones(size(mus)));
end

function [Theta_bayes_factors] = calcBayesFactors(Theta_prob,a_interact,b_interact,a_perturb,b_perturb,numPerturb)
% compute Bayes factors for interaction matrix
%
% inputs:
% Theta_prob = probability matrix for interactions
% a = prior probability of no interaction for beta prior
%
% output:
% Theta_bayes_factors = Bayes factors for interactions

Theta_prior = ones(size(Theta_prob,1),size(Theta_prob,2)-numPerturb)*(a_interact+1)/(b_interact+1);
if numPerturb > 0,
    Theta_prior = [Theta_prior ones(size(Theta_prob,1),numPerturb)*(a_perturb+1)/(b_perturb+1)];
end;

Theta_bayes_factors = (Theta_prob./(ones(size(Theta_prob))-Theta_prob)).*Theta_prior;
end
