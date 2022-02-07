function [minmax d m VarNature VarBds FnGradAvail NumConstraintGradAvail, StartingSol, budget, ObjBd, OptimalSol] = IronOreInventoryStructure(NumStartingSol, seed)
% Inputs:
%	a) NumStartingSol: Number of starting solutions required. Integer, >= 0
%	b) seed: Seed for generating random starting solutions. Integer, >= 1
% Return structural information on optimization problem:    
%     a) minmax: -1 to minimize objective , +1 to maximize objective    
%     b) d: positive integer giving the dimension d of the domain    
%     c) m: nonnegative integer giving the number of constraints. All
%        constraints must be inequality constraints of the form LHS >= 0.
%        If problem is unconstrained (beyond variable bounds) then should
%        be 0.
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
%     h) StartingSol: One starting solution in each row, or NaN if
%        NumStartingSol=0.
%        Solutions generated as per problem writeup
%     i) budget: Column vector of suggested budgets, or NaN if none
%        suggested j) ObjBd is a bound (upper bound for maximization problems,
%        lower
%        bound for minimization problems) on the optimal solution value, or
%        NaN if no such bound is known.
%     k) OptimalSol is a d dimensional column vector giving an optimal
%        solution if known, and it equals NaN if no optimal solution is
%        known.

%   *************************************************************
%   ***             Written by Bryan Chong                    ***
%   ***       bhc34@cornell.edu    January 27th, 2015         ***
%   *************************************************************
minPrice = 0;
maxPrice = 200;
capacity = 10000;

minmax = 1; % Maximize expected profit. (-1 for Minimize)
d = 4;
m = 0; % no constraints
VarNature = zeros(1,d);
VarNature(2)= 1;
VarBds = [minPrice, maxPrice; 0, capacity; minPrice, maxPrice; minPrice, maxPrice];
FnGradAvail = 0; % No derivatives
NumConstraintGradAvail = 0; % No constraint gradients
budget = [1000; 10000; 50000]; 
ObjBd=NaN;
OptimalSol=NaN;
if (NumStartingSol < 0) || (NumStartingSol ~= round(NumStartingSol)) || (seed <= 0) || (round(seed) ~= seed),
    fprintf('NumStartingSol must be a nonnegative integer, seed must be a positive integer\n');
    StartingSol = NaN;
else
    if NumStartingSol == 0,
        StartingSol = NaN;
    elseif NumStartingSol == 1,
        StartingSol = [80, 7000, 40, 100];
    else
        MyStream= RandStream.create('mrg32k3a', 'NumStreams', 1);
        MyStream.Substream = seed;
        OldStream = RandStream.setGlobalStream(MyStream);
        d1 = unifrnd(70, 90, NumStartingSol, 1);
        d2 = unidrnd(6001, NumStartingSol, 1) + 1999;
        d3 = unifrnd(30, 50, NumStartingSol, 1);
        d4 = unifrnd(90, 110, NumStartingSol, 1);
        StartingSol = [d1, d2, d3, d4];
        RandStream.setGlobalStream(OldStream); % Restore previous stream
    end
end
