%% Number Shuffling
% ================
%   The syntax is
%       y = shuffle(X)
%   where
%   X: an m by n matrix or a 1 by n vector
%   y: an m by n matrix or a 1 by n vector containing shuffled numbers of
%   m by n matrix X or 1 by n vector X
function t = shuffle(X)
%% Created by: Aamir Alaud-din
%% Created on: 01-02-2012
%% Input:
%   A vector or a matrix
%% Output:
%   Shuffled vector or matrix of input vector or matrix
%% Program Initialization
[r c] = size(X);
b = reshape(X',r*c,1);
b = b';
t = b(randperm(length(b)));
%y = vec2mat(t,c);
nc=size(X,2);
% c is the original vector, matrix, nc is the number of columns:
y=reshape([X(:) ; zeros(rem(nc - rem(numel(X),nc),nc),1)],nc,[]);
t=t(y);
end
% That's it!