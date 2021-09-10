function [minmax d m VarNature VarBds FnGradAvail NumConstraintGradAvail StartingSol budget ObjBd OptimalSol NumRngs] = GPEStructure(NumStartingSol)
% Inputs:
%	a) NumStartingSol: Number of starting solutions required. Integer, >= 0
%	
% Return structural information on optimization problem
%     a) minmax: -1 to minimize objective , +1 to maximize objective
%     b) d: positive integer giving the dimension d of the domain
%     c) m: nonnegative integer giving the number of constraints. All
%        constraints must be inequality constraints of the form LHS >= 0.
%        If problem is unconstrained (beyond variable bounds) then should be 0.
%     d) VarNature: a d-dimensional column vector indicating the nature of
%        each variable - real (0), integer (1), or categorical (2).
%     e) VarBds: A d-by-2 matrix, the ith row of which gives lower and
%        upper bounds on the ith variable, which can be -inf, +inf or any
%        real number for real or integer variables. Categorical variables
%        are assumed to take integer values including the lower and upper
%        bound endpoints. Thus, for 3 categories, the lower and upper
%        bounds could be 1,3 or 2, 4, etc.
%     f) FnGradAvail: Equals 1 if gradient of function values are
%        available, and 0 otherwise.
%     g) NumConstraintGradAvail: Gives the number of constraints for which
%        gradients of the LHS values are available. If positive, then those
%        constraints come first in the vector of constraints.
%     h) StartingSol: One starting solution in each row, or NaN if NumStartingSol=0.
%        Solutions generated as per problem writeup
%     i) budget: maximum budget, or NaN if none suggested
%     j) NumRngs: the number of random number streams needed by the
%        simulation model

%   ***************************************
%   *** Written by Jessica Wu           ***
%   *** Edited by Jennifer Shih         ***
%   *** Edited by Bryan Chong           ***
%   *** Edited by David Newton          ***
%   *** Last updated Oct 16, 2018       ***
%   ***************************************

minmax = 1; % Maximization problem (-1 for Minimization)
d = 2; % x has two components
m = 0; % No constraints
VarNature = 0; % x is a vector of real variables
VarBds = [0 20;0 20];
FnGradAvail = 0; % No derivative is computed
NumConstraintGradAvail = 0; % No constraints
budget = 10000;
ObjBd = NaN;
OptimalSol = [2 5]; %the optimal solution is found where x* = [2,5]
NumRngs = 2;

if (NumStartingSol < 0) || (NumStartingSol ~= round(NumStartingSol))
    fprintf('NumStartingSol should be integer >= 0, seed must be a positive integer\n');
    StartingSol = NaN;
else
    if (NumStartingSol == 0)
        StartingSol = NaN;
    else 
        StartingSol = unifrnd(0,10,2,NumStartingSol)';
    end   
end