function [S_sample,BO,BD,biomassScaleFactor] = splineBiomassMCMC(T,BMD,sigma_biomass_est,params,numReplicates)
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

% MCMC samples from a penalized spline model (using 1st or 2nd order penalization)

% inputs:
% T = input times
% BMD = input biomass data - cell array of vectors, w/ each vector a set of
% concatenated time-series for each subject (i.e., first time-series is all
% time-points for replicate 1, second time-series is all time-points
% sigma_data_est = estimate of data std
% paramFileName = parameter file name

% outputs:
% S_sample = cell array of spline coefficient samples of size number of
% samples x number of subjects
% BO = spline coefficient matrix excluding replicates
% B = spline coefficient matrix including replicates (just used for
% inferring parameters)
% BD = spline derivative coefficient matrix

    %params = readParameterFile(paramFileName);

    % numReplicates = number of replicates per subject
    % numReplicates = params.numReplicates;

    % number of MCMC iterations
    numIters = params.numIters;
    % number of burnin samples
    numBurnin = params.numBurnin;

    % set up spline matrices
    minT = min(T);
    maxT = max(T);
    splineDegree = 4;
    % interval in days
    splineInterval = 2;
    % order of constraint (right now just supports 1st or 2nd order)
    smoothnessOrder = params.smoothnessOrder;

    numSplineSegs = round((maxT-minT)/splineInterval);
    %numSplineSegs =  size(T,1) - 2;
    knots = augknt(linspace(minT,maxT,numSplineSegs),splineDegree);
    colmat = spcol(knots,splineDegree,brk2knt(T,2));
    BO = colmat(1:2:end,:);
    % replicate entries in B
    B = repmat(BO,numReplicates,1);
    BD = colmat(2:2:end,:);

    % create constraint matrix to penalize spline roughness
    % first part is constraint on individual coefficients
    nc = size(B,2);
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
    BC = [BC1 ; BC2];

    numSubjects = length(BMD);
    S = cell(numSubjects,1);

    biomassScaleFactor = zeros(numSubjects,1);
    for s=1:numSubjects,
        biomassScaleFactor(s) = mean(BMD{s});
    end;

    % variance parameters on spline coefficients
    tau = cell(numSubjects,1);

    % variance parameters on spline coefficient differences
    omega = cell(numSubjects,1);

    sigma_data = sigma_biomass_est;

    S_sample = cell(numSubjects,1);
    for s=1:numSubjects,
        S_sample{s} = zeros(nc,numIters-numBurnin);
    end;

    % initial estimates for spline coefficients
    medxs = [];
    medd = [];
    medx = [];
    for i=1:numSubjects,
        S{i} = pinv(B)*(BMD{i}-biomassScaleFactor(i));
        bt = (BMD{i}-biomassScaleFactor(i));
        bt = bt(1:numReplicates:length(bt));
        medxs = [medxs std(bt)];
        medd = [medd std(diff(bt,smoothnessOrder))];
        medx = [medx mean(abs(diff(bt,smoothnessOrder)))];
    end;

    medxs = nanmedian(medxs);
    medd = nanmedian(medd);
    medx = nanmedian(medx);

    for i=1:numSubjects,
    %    tau{i} = params.tauScale*ones(size(S{i}));
    %    omega{i} = ones(np,size(S{i},2));
        tau{i} = 1000*medxs^2*ones(size(S{i}));
        omega{i} = medd^2*ones(np,size(S{i},2));
    end;

    %lambda = params.init_lambda*ones(np,1);
    %gpB = params.gpB*ones(np,1); gpA = params.gpA*ones(np,1);
    lambda = ones(np,1)/(10*medx)^2;
    gpB = (lambda(1)*10^6)*ones(np,1); gpA = (1/10^6)*ones(np,1);

    % do MCMC sampling
    for i=1:numIters,
        for s=1:numSubjects,
            try
                % compute prior covariance matrix
                R_inv = diag([1./tau{s}' 1./omega{s}']);
                R_inv = BC'*R_inv*BC;

                % sample from approximate posterior
                C = inv_chol(R_inv + B'*B/sigma_data^2);
                m = C*(B'*(BMD{s}-biomassScaleFactor(s))/sigma_data^2);
                S{s} = mvnrnd(m,C)';
            catch errM
                os = sprintf('err=%s',errM.message);
             %   disp(os);
            end

       end;

       % sample sigma_data
        dv = [];
        for s=1:numSubjects,
            dv = [dv ; B*S{s}-BMD{s}+biomassScaleFactor(s)];
        end;
        sigma_data = sqrt(1/gamrnd((length(dv)+1)/2 + 1,2/sum(dv.^2)));

        % sample omega
        for s=1:numSubjects,
            if smoothnessOrder == 1,
                ig_mu = sqrt(lambda)./abs(diff(S{s},1));
            end;
            if smoothnessOrder == 2,
                ig_mu = sqrt(lambda)./abs(diff(S{s},2));
            end;
            ig_lamb = lambda;
            omega{s} = 1./sample_inverse_gaussian(ig_mu,ig_lamb);
        end;

        % sample lambda
        ss = zeros(size(omega{1}));
        for s=1:numSubjects,
            ss = ss + omega{s};
        end;
        lambda = gamrnd(gpA + numSubjects,1.0./(ss/2.0 + 1.0./gpB));

        if mod(i,100) == 0,
            os = sprintf('iter=%i sigma_data=%f',i,sigma_data);
        %    disp(os);
        end;

        if i > numBurnin,
            for s=1:numSubjects,
                S_sample{s}(:,i-numBurnin) = S{s};
            end;
        end;
    end;

end
