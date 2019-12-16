function [phi_sample,B,BD] = splineCountsMCMC(T,data_trajectories,total_count_norm,biomass_norm,scaleFactor,intervene_matrix_filtered,params,nodeNum)
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

% infer spline from counts data
%
% inputs:
% T = time-points sampled
% data_trajectories = cell array of counts data (dim # subjects X # OTUs)
% total_count_norm, biomass_norm, scaleFactor = normalization/scaling
% factors calculated by preProcessData.m
% intervene_matrix_filtered = matrix (dim # time-points X # OTUs) with
% entry = 1 if OTU is present, 0 if OTU is not present
% nodeNum = integer identifier (used for debugging)
%
% outputs:
% phi_sample = spline coefficient samples
% B = spline basis matrix
% BD = derivative basis matrix

%params = readParameterFile(parameterFileName);

% number of MCMC iterations
numIters = params.numIters;
% number of burnin samples
numBurnin = params.numBurnin;

numSubjects = size(data_trajectories,1);
numOTUs = size(intervene_matrix_filtered,2);
TO = cell(numOTUs,1);
% spline basis
B = cell(numOTUs,1);
% derivative basis
BD = cell(numOTUs,1);
% constraint matrix; corresponds to Upsilon in supplementary material
BC = cell(numOTUs,1);

% to store offset term computing from normalizing factors
offset = cell(numSubjects,numOTUs);
% to store rescaling term for computing NBD
% parameter
rescale = cell(numSubjects,numOTUs);

splineDegree = 4;
% interval in days for placing spline knots
splineInterval = 2;
% order of constraint (right now just supports 1st or 2nd order)
smoothnessOrder = params.smoothnessOrder;

% find minimum total_count ratio
mbt = Inf;
for o=1:numOTUs,
    f = find(intervene_matrix_filtered(:,o) > 0);
    TO{o} = T(f);
    for s=1:numSubjects,
        offset{s,o} = total_count_norm{s}(f) - biomass_norm{s}(f) + scaleFactor{s,o};
        rescale{s,o} = - biomass_norm{s}(f) + scaleFactor{s,o};
        mbt2 = min(-total_count_norm{s}(f));
        if mbt2 < mbt,
            mbt = mbt2;
        end;
    end;

    % set up spline matrices
    minT = min(TO{o});
    maxT = max(TO{o});

    numSplineSegs = round((maxT-minT)/splineInterval);
    knots = augknt(linspace(minT,maxT,numSplineSegs),splineDegree);
    colmat = spcol(knots,splineDegree,brk2knt(TO{o},2));
    BT = colmat(1:2:end,:);
    BDT = colmat(2:2:end,:);

    B{o} = sparse(BT);
    BD{o} = sparse(BDT);

    % set up constraint matrices
    % create constraint matrix to penalize spline roughness
    % first part is constraint on individual coefficients
    nc = size(B{o},2);
    BC1 = eye(nc);
    % second part is constraint on adjacent coefficients

    np = nc-smoothnessOrder;
    BC2 = zeros(np,nc);

    for i=1:np,
        if smoothnessOrder == 1,
            BC2(i,i) = 1;
            BC2(i,i+1) = -1;
        end;
        if smoothnessOrder == 2,
            BC2(i,i) = 1;
            BC2(i,i+1) = -2;
            BC2(i,i+2) = 1;
        end;
    end;
    BC{o} = [BC1 ; BC2];
end;

phi_sample = cell(numSubjects,numOTUs);
for s=1:numSubjects,
    for o=1:numOTUs,
        phi_sample{s,o} = zeros(size(B{o},2),numIters-numBurnin);
    end;
end;

S = cell(numSubjects,numOTUs);

% variance parameters on spline coefficients
tau = cell(numSubjects,numOTUs);
% variance parameters on adjacent spline coefficients
omega = cell(numSubjects,numOTUs);

% penalty parameters
lambda_omega = cell(numOTUs,1);

% cell arrays to store NBD log mean, exp mean, and loglikelihoods
nus = cell(numSubjects,numOTUs);
mu_hat = cell(numSubjects,numOTUs);
% exp mean on 'relative abundance scale' (e.g., relative abundance of OTU
% in the ecosystem) - used in NBD error model
mu_hat_rel_abn_scale = cell(numSubjects,numOTUs);

% initial estimates for spline coefficients
medxs = [];
medd = [];
medx = [];
for i=1:numSubjects,
    for o=1:numOTUs,
        S{i,o} = pinv(full(B{o}))*(log(data_trajectories{i,o}+1)-offset{i,o});
        % set zeros to small random #'s to avoid numerical issues
        f = find(S{i,o} == 0);
        if ~isempty(f),
            S{i,o}(f) = normrnd(zeros(size(f)),10e-3*ones(size(f)));
        end;
        bt = log(data_trajectories{i,o}+1)-offset{i,o};
        medxs = [medxs std(bt)];
        medd = [medd std(diff(bt,smoothnessOrder))];
        medx = [medx mean(abs(diff(bt,smoothnessOrder)))];

        nus{i,o} = B{o}*S{i,o}+offset{i,o};
        mu_hat{i,o} = exp(nus{i,o});
        mu_hat_rel_abn_scale{i,o} = exp(B{o}*S{i,o}+rescale{i,o});
    end;
end;

medxs = nanmedian(medxs);
medd = nanmedian(medd);
medx = nanmedian(medx);

for i=1:numSubjects,
    for o=1:numOTUs,
    %    tau{i,o} = ones(length(S{i,o}),1)*params.tauScale;
    %    omega{i,o} = ones(length(S{i,o})-smoothnessOrder,1);

    tau{i,o} = 1000*medxs^2*ones(length(S{i,o}),1);
    omega{i,o} = medd^2*ones(length(S{i,o})-smoothnessOrder,1);
    end;
end;

% hyperparameters for gamma prior on lambda_omega (penalization for
% adjacency of spline coefficients)
gpB_omega = cell(numOTUs,1);
gpA_omega = cell(numOTUs,1);
for o=1:numOTUs,
    %lambda_omega{o} = params.lambda_omega_init*ones(size(omega{1,o}));
    %gpB_omega{o} = params.gpB_omega*ones(size(lambda_omega{o})); gpA_omega{o} = params.gpA_omega;

    lambda_omega{o} = ones(size(omega{1,o}))/(10*medx)^2;
    gpB_omega{o} = (lambda_omega{o}(1)*10^6)*ones(size(lambda_omega{o})); gpA_omega{o} = (1/10^6)*ones(size(lambda_omega{o}));
end;

% initialization for NBD variance parameters
% assumed error model is eps = eps_a0/mu + eps_a1
eps_a1 = params.eps_a1_init;
eps_a0 = (1-eps_a1)*exp(mbt)*10;

tune_eps_a1 = abs(log(eps_a1))/params.tune_eps_a1_factor;
tune_eps_a0 = abs(log(eps_a0))/params.tune_eps_a0_factor;

EPS_PRIOR_UB = 1e8; %uniform prior upper bound for eps_a0, eps_a1; LB = 0

% these parameters are used to do initial sampling using a Gaussian to get decent
% starting estimates
v_prop = params.v_prop;
numInitEstimate = params.numInitEstimate;

% loglikelihood of data
LL_Y = zeros(numSubjects,numOTUs);
LL_eps_new = zeros(numSubjects,numOTUs);
% loglikelihood of spline coefficients
LL_B = zeros(numSubjects,numOTUs);

% calculate initial loglikelihoods
for s=1:numSubjects,
    for o=1:numOTUs,
        R_inv = diag([1./tau{s,o}' 1./omega{s,o}']);
        R_inv = BC{o}'*R_inv*BC{o};
        R = inv_chol(R_inv);
        eps = eps_a0./mu_hat_rel_abn_scale{s,o} + eps_a1;
        LL_Y(s,o) = sum(NBlogLikErr2(data_trajectories{s,o},exp(B{o}*S{s,o}+offset{i,o}),eps));
        LL_B(s,o) = log(mvnpdf(S{s,o},zeros(size(S{s,o})),R));
    end;
end;

total_sample_S = 0;
success_sample_phi = 0;
total_sample_eps_a0 = 0;
success_sample_eps_a0 = 0;
total_sample_eps_a1 = 0;
success_sample_eps_a1 = 0;

% do MCMC sampling
for i=1:numIters,
    for s=1:numSubjects,
        for o=1:numOTUs,
            try
                total_sample_S = total_sample_S + 1;
                % compute prior covariance matrix
                R_inv = diag([1./tau{s,o}' 1./omega{s,o}']);
                R_inv = BC{o}'*R_inv*BC{o};
                R = inv_chol(R_inv);

                % compute 'data' estimate and weights given NBD noise model
                if i < numInitEstimate,
                    YN_hat = log(data_trajectories{s,o}+1) - offset{s,o};
                    w = ones(size(mu_hat{s,o}))/(v_prop);
                else,
                    YN_hat = nus{s,o} - offset{s,o} + (data_trajectories{s,o}-mu_hat{s,o})./mu_hat{s,o};
                    eps = eps_a0./mu_hat_rel_abn_scale{s,o} + eps_a1;
                    w = mu_hat{s,o}./(1+eps.*mu_hat{s,o});
                end;

                W = diag(w);

                % sample from approximate posterior
                C = inv_chol(R_inv + B{o}'*W*B{o});
                m = C*(B{o}'*W*YN_hat);
                S_new = mvnrnd(m,C)';

                % update NBD mean and likelihoods
                nus_t = B{o}*S_new;
                nus_new = nus_t+offset{s,o};
                mu_hat_new = exp(nus_new);
                mu_hat_new_unscaled = exp(nus_t+rescale{s,o});
                eps_new = eps_a0./mu_hat_new_unscaled + eps_a1;
                LL_Y_new = sum(NBlogLikErr2(data_trajectories{s,o},mu_hat_new,eps_new));
                LL_B_new = log(mvnpdf(S_new,zeros(size(S_new)),R));

                % now compute reverse move for MCMC (i.e., going backwards from
                % S_new that we've just sampled)
                if i < numInitEstimate,
                    YN_hat_new = log(data_trajectories{s,o}+1) - offset{s,o};
                    w_new = ones(size(mu_hat{s,o}))/(v_prop);
                else,
                    YN_hat_new = nus_new - offset{s,o} + (data_trajectories{s,o}-mu_hat_new)./mu_hat_new;
                    w_new = mu_hat_new./(1+eps_new.*mu_hat_new);
                end;

                W_new = diag(w_new);

                C_new = inv_chol(R_inv + B{o}'*W_new*B{o});
                m_new = C_new*(B{o}'*W_new*YN_hat_new);

                q1 = log(mvnpdf(S_new,m,C));
                q2 = log(mvnpdf(S{s,o},m_new,C_new));

                % compute Metropolis-Hastings ratio for move to S_new
                r = LL_Y_new + LL_B_new - LL_Y(s,o) - LL_B(s,o) + q2 - q1;
                r = min(1,exp(r));
                u = unifrnd(0,1);

                % accept the move
                if u <= r,
                    S{s,o} = S_new;
                    LL_Y(s,o) = LL_Y_new;
                    LL_B(s,o) = LL_B_new;
                    nus{s,o} = nus_new;
                    mu_hat{s,o} = mu_hat_new;
                    mu_hat_rel_abn_scale{s,o} = mu_hat_new_unscaled;
                    success_sample_phi = success_sample_phi + 1;
                end;
            catch errM
                os = sprintf('node=%i err=%s',nodeNum,errM.message);
            %    disp(os);
            end

            end;
        end;

    % sample variables controlling NBD variance
    % assuming uniform prior

    % sample eps_a0
    eps_a0_l = log(eps_a0);
    eps_a0_l_new = normrnd(eps_a0_l,tune_eps_a0);

    LL_t_old = 0;
    LL_t_new = 0;
    for s=1:numSubjects,
        for o=1:numOTUs,
            eps = exp(eps_a0_l_new)./mu_hat_rel_abn_scale{s,o} + eps_a1;
            LL_eps_new(s,o) = sum(NBlogLikErr2(data_trajectories{s,o},mu_hat{s,o},eps));
            LL_t_old = LL_Y(s,o) + LL_t_old;
            LL_t_new = LL_eps_new(s,o) + LL_t_new;
        end;
    end;

    % compute MH ratio
    r = min(1,exp(LL_t_new - LL_t_old));
    u = unifrnd(0,1);
    total_sample_eps_a0= total_sample_eps_a0 + 1;
    % accept the proposal
    if u (u <= r && eps_a0_l_new<=EPS_PRIOR_UB),
        eps_a0 = exp(eps_a0_l_new);
        LL_Y = LL_eps_new;
        success_sample_eps_a0 = success_sample_eps_a0 + 1;
    end;

    % sample eps_a1
    eps_a1_l = log(eps_a1);
    eps_a1_l_new = normrnd(eps_a1_l,tune_eps_a1);

    LL_t_old = 0;
    LL_t_new = 0;
    for s=1:numSubjects,
        for o=1:numOTUs,
            eps = eps_a0./mu_hat_rel_abn_scale{s,o} + exp(eps_a1_l_new);
            LL_eps_new(s,o) = sum(NBlogLikErr2(data_trajectories{s,o},mu_hat{s,o},eps));

            LL_t_old = LL_Y(s,o) + LL_t_old;
            LL_t_new = LL_eps_new(s,o) + LL_t_new;
        end;
    end;

    % compute MH ratio
    r = min(1,exp(LL_t_new - LL_t_old));
    u = unifrnd(0,1);
    total_sample_eps_a1 = total_sample_eps_a1 + 1;
    % accept the proposal
    if (u <= r && eps_a1_l_new<=EPS_PRIOR_UB),
        eps_a1 = exp(eps_a1_l_new);
        LL_Y = LL_eps_new;
        success_sample_eps_a1 = success_sample_eps_a1 + 1;
    end;

    % sample omega
    for s=1:numSubjects,
        for o=1:numOTUs,
            ig_mu = sqrt(lambda_omega{o})./abs(diff(S{s,o},smoothnessOrder));
            ig_lamb = lambda_omega{o};
            omega{s,o} = 1./sample_inverse_gaussian(ig_mu,ig_lamb);
        end;
    end;

    % sample lambda_omega
    for o=1:numOTUs,
        sn = 0;
        ss = zeros(size(omega{1,o}));
        for s=1:numSubjects,
            ss = ss + omega{s,o};
        end;
        lambda_omega{o} = gamrnd(gpA_omega{o} + numSubjects,1.0./(ss/2.0 + 1.0./gpB_omega{o}));
    end;


    for s=1:numSubjects,
        for o=1:numOTUs,
            R_inv = diag([1./tau{s,o}' 1./omega{s,o}']);
            R_inv = BC{o}'*R_inv*BC{o};
            try
                R = inv_chol(R_inv);
             catch errM
                os = sprintf('node=%i err=%s',nodeNum,errM.message);
                disp(os);
            end
            LL_B(s,o) = log(mvnpdf(S{s,o},zeros(size(S{s,o})),R));
        end;
    end;

    if mod(i,100) == 0,
    %    os = sprintf('splineMCMC node=%i iter=%i eps_a0=%f accept_eps_a0=%f eps_a1=%f accept_eps_a1=%f accept_S=%f',nodeNum,i,eps_a0,success_sample_eps_a0/total_sample_eps_a0,eps_a1,success_sample_eps_a1/total_sample_eps_a1,success_sample_phi/total_sample_S);
    %    disp(os);
    end;

    if i > numBurnin,
        for s=1:numSubjects,
            for o=1:numOTUs,
                phi_sample{s,o}(:,i-numBurnin) = S{s,o};
            end;
        end;
    end;
end;
