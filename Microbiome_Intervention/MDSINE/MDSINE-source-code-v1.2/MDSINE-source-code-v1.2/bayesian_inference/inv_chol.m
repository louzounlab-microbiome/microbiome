function Y = inv_chol(M)
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

% Matrix Inversion using Cholesky Decomposition
%
% Finds the inverse of the matrix X, given its (lower triangular) Cholesky
% Decomposition; i.e. X = LL', according to the paper 'Matrix Inversion
% Using Cholesky Decomposition', Aravindh Krishnamoorthy, Deepak Menon,
% arXiv:1111.4144.
%

% Version 0.1, 2013-05-25, Aravindh Krishnamoorthy
% e-mail: aravindh.k@ieee.org

L = chol(M,'lower');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = size(L, 1) ;
Y = zeros(N, N) ;
% Work with the upper triangular matrix
R = L' ;
% Construct the auxillary diagonal matrix S = 1/rii
S = inv(diag(diag(R))) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=N:-1:1
    for i=j:-1:1
        Y(i,j) = S(i,j) - R(i,i+1:end)*Y(i+1:end,j) ;
        Y(i,j) = Y(i,j)/R(i,i) ;
        % Write out the symmetric element
        Y(j,i) = conj(Y(i,j)) ;
    end
end
