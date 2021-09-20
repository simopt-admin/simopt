function [minmax, d, m, VarNature, VarBds, FnGradAvail, NumConstraintGradAvail, StartingSol, budget, ObjBd, OptimalSol, NumRngs] = EOQStructure(NumStartingSol)

% Inputs:
%	  a) NumStartingSol: Number of starting solutions required. Integer, >= 0
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
%     h) StartingSol: Q- order quantity
%     i) budget: maximum budget, or NaN if none suggested 
%     j) ObjBd is a bound (upper bound for maximization problems, lower 
%        bound for minimization problems) on the optimal solution value, or
%        NaN if no such bound is known.
%     k) OptimalSol is a d dimensional column vector giving an optimal
%        solution if known, and it equals NaN if no optimal solution is
%        known.
%     l) NumRngs: the number of random number streams needed by the
%        simulation model
%
%   ************************************************************* 
%   ***               dWritten by Danielle Lertola             ***
%   ***           dcl96@cornell.edu    July 17, 2012          ***
%   ***                 Edited by Jennifer Shi                ***
%   ***          jls493@cornell.edu    June 20th, 2014        ***
%   ***                Edited by Shane Henderson              ***
%   ***            sgh9@cornell.edu   July 14, 2014           ***
%   ***                 Edited by David Eckman                ***
%   ***           dje88@cornell.edu   Sept 14, 2018           ***
%   *************************************************************
%
% Last updated Sept 14, 2018
%


mu = 1040;
K = 10;
h = 2.5;

minmax = -1; % minimize cost
d = 1; % Q- order quantity
m = 0; % no constraints
VarNature = 0; % real
VarBds = [0 inf]; % bounds set for Q
FnGradAvail = 1; % Derivatives are available
NumConstraintGradAvail = 0; % No constraint gradients
budget = 15000;
ObjBd = NaN;
OptimalSol = sqrt(2*mu*K/h);
NumRngs = 1;

if NumStartingSol<0 || (NumStartingSol ~= round(NumStartingSol))
    fprintf('NumStartingSol should be integer >= 0. \n');
    StartingSol = NaN;
    
else
       
    if NumStartingSol == 0
        StartingSol = NaN;            
    else
        StartingSol = 500*rand(NumStartingSol, 1);   % Generate Q value uniformly in (0, 500)
    end   
end
