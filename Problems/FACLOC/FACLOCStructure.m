function [minmax, d, m, VarNature, VarBds, FnGradAvail, NumConstraintGradAvail, StartingSol, budget, ObjBd, OptimalSol, NumRngs] = FACLOCStructure(NumStartingSol)

% Inputs:
%	a) NumStartingSol: Number of starting solutions required. Integer, >= 0
% Return structural information on optimization problem
%   a) minmax: -1 to minimize objective , +1 to maximize objective
%   b) d: positive integer giving the dimension d of the domain
%   c) m: nonnegative integer giving the number of constraints. All
%        constraints must be inequality constraints of the form LHS >= 0.
%        If problem is unconstrained (beyond variable bounds) then should be 0.
%   d) VarNature: a d-dimensional column vector indicating the nature of
%        each variable - real (0), integer (1), or categorical (2).
%   e) VarBds: A d-by-2 matrix, the ith row of which gives lower and
%        upper bounds on the ith variable, which can be -inf, +inf or any
%        real number for real or integer variables. Categorical variables
%        are assumed to take integer values including the lower and upper
%        bound endpoints. Thus, for 3 categories, the lower and upper
%        bounds could be 1,3 or 2, 4, etc.
%   f) FnGradAvail: Equals 1 if gradient of function values are
%        available, and 0 otherwise.
%   g) NumConstraintGradAvail: Gives the number of constraints for which
%        gradients of the LHS values are available. If positive, then those
%        constraints come first in the vector of constraints.
%   h) StartingSol: One starting solution in each row, or NaN if NumStartingSol=0.
%        Solutions generated as per problem writeup
%   i) budget: maximum budget, or NaN if none suggested
%   j) ObjBd is a bound (upper bound for maximization problems, lower
%        bound for minimization problems) on the optimal solution value, or
%        NaN if no such bound is known.
%   k) OptimalSol is a d dimensional column vector giving an optimal
%        solution if known, and it equals NaN if no optimal solution is known.
%   l) NumRngs: the number of random number streams needed by the
%        simulation model

%   *************************************************************
%   ***               Written by Danielle Lertola             ***
%   ***           Edited by Jennifer Shi, Bryan Chong         ***
%   ***            and David Eckman     Sept 9, 2018          ***
%   *************************************************************

% Number of warehouses is user selectable, but in this version we use 2
% warehouses

nWarehouses = 2;
minmax = 1; % maximize num orders completed in tau min(-1 for minimize)
d = 2 * nWarehouses; % (x, y) for each warehouse
m = 0; % no constraints
VarNature = zeros(d, 1); % real variables
VarBds = ones(d, 1) * [0, 1]; % x, y are in [0, 1]
FnGradAvail = 0; % No derivatives when number of warehouses is > 1
NumConstraintGradAvail = 0; % No constraints
budget = 10000;
ObjBd = NaN;
OptimalSol = NaN;
NumRngs = 4;


if (NumStartingSol < 0) || (NumStartingSol ~= round(NumStartingSol))
    fprintf('NumStartingSol should be integer >= 0. \n');
    StartingSol = NaN;
else
    if (NumStartingSol == 0)
        StartingSol = NaN;
    else
        % Matlab fills random matrices by column, thus the transpose
        StartingSol = rand(2*nWarehouses, NumStartingSol)';
    end
end

