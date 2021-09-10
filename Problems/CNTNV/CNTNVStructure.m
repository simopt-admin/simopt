function [minmax d m VarNature VarBds FnGradAvail NumConstraintGradAvail, StartingSol, budget, ObjBd, OptimalSol, NumRngs] = CNTNVStructure(NumStartingSol)
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
%     j) ObjBd is a bound (upper bound for maximization problems, lower
%        bound for minimization problems) on the optimal solution value, or
%        NaN if no such bound is known.
%     k) OptimalSol is a d dimensional column vector giving an optimal
%        solution if known, and it equals NaN if no optimal solution is known.
%     l) NumRngs: the number of random number streams needed by the
%        simulation model

%   ****************************************
%   *** Written by Shane Henderson to    ***
%   *** use standard calling and random  ***
%   *** number streams                   ***
%   *** Edited by Bryan Chong            ***
%   *** bhc34@cornell.edu, Sept 21, 2014 ***
%   *** Edited by David Newton           ***
%   *** newton34@purdue.edu, Oct 5, 2018 ***
%   ****************************************
% Last updated Sept 21, 2014

% Recommended parameters
c = 5; % Cost per unit
s = 9; % Selling price per unit
w = 1; % Salvage price
alpha = 2; % alpha and beta are the Burr Type XII cdf parameters
beta = 20;

minmax = 1; % maximize mean profit (-1 for minimize)
d = 1; % one-dimensional decision (quantity to buy)
m = 0; % no constraints
VarNature = 0; % real variables
VarBds = [0, inf]; % must buy a nonnegative quantity
FnGradAvail = 1; % Derivative is computed
NumConstraintGradAvail = 0; % No constraints
budget = 15000;
ObjBd=NaN;
r = (s-c)/(s-w); % For use in OptimalSol
OptimalSol = (1/(1-r)^(1/beta)-1)^(1/alpha); %x* = 0.1878, max obj = 0.4635
NumRngs = 1;

if (NumStartingSol < 0) || (NumStartingSol ~= round(NumStartingSol))
    fprintf('NumStartingSol should be integer >= 0');
    StartingSol = NaN;
else
    if (NumStartingSol == 0)
        StartingSol = NaN;
    else
        StartingSol = exprnd(1, 1, NumStartingSol)'; % Column vector of exponential r.v.s with mean 1
    end %if NumStartingSol
end %if inputs ok
