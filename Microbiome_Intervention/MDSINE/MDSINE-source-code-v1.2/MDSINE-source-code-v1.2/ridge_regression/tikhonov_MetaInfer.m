function [Theta] = tikhonov_MetaInfer(rp,X,F_prime)
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

if min(min(rp))<0
  error('Illegal (nagative) regularization parameters')
end

L=size(F_prime,1); % number of variables
nrx=size(X,1);
P=nrx-1-L; % number of perturbations

% building the D matrix
D=zeros(nrx,nrx);
D(1,1)=rp(1);
D(nrx+2:nrx+1:(nrx+1)*L+1)=rp(2);

if P>0
    D(1+(nrx+1)*(L+1):nrx+1:end)=rp(3);
end

% ridge solution to start with
%Theta=F_prime*X'*pinv(X*X'+D*D');
Theta=F_prime*X'*pinv(X*X'+D);


return
