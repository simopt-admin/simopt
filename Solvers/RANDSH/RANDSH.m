%==========================================================================
%                The Random Search Algorithm
%==========================================================================
% DATE
%        Sept 2018
%
% AUTHOR
%        David Eckman
%
% REFERENCE	
%        None
%
%==========================================================================
%
% INPUT
%        problem
%              Problem function name
%        probstructHandle
%              Problem structure function name 
%        problemRng
%              Random number generators (streams) for problems
%        solverRng
%              Random number generator (stream) for solver
%        numBudget
%              number of budgets to record, >=3; the spacing between
%              adjacent budget points should be about the same
%
%
% OUTPUT
%        Ancalls
%              An array (size = 'NumSoln' X 1) of budget expended
%        A
%              An array (size = 'NumSoln' X 'dim') of solutions
%              returned by solver
%        Afn
%              An array (size = 'NumSoln' X 1) of estimates of expected
%              objective function value
%        AFnVar
%              An array of variances corresponding to
%              the objective function at A
%              Equals NaN if solution is infeasible
%        AFnGrad
%              An array of gradient estimates at A; not reported
%        AFnGardCov
%              An array of gradient covariance matrices at A; not reported
%        Aconstraint
%              A vector of constraint function estimators; not applicable
%        AConstraintCov
%              An array of covariance matrices corresponding to the
%              constraint function at A; not applicable
%        AConstraintGrad
%              An array of constraint gradient estimators at A; not
%              applicable
%        AConstraintGradCov
%              An array of covariance matrices of constraint gradient
%              estimators at A; not applicable
%
%==========================================================================

%% Random Search
function [Ancalls, A, Afn, AFnVar, AFnGrad, AFnGradCov, ...
    Aconstraint, AConstraintCov, AConstraintGrad, ...
    AConstraintGradCov] = RANDSH(probHandle, probstructHandle, ...
    problemRng, solverRng, numBudget)


%% Unreported
AFnGrad = NaN;
AFnGradCov = NaN;
Aconstraint = NaN;
AConstraintCov = NaN;
AConstraintGrad = NaN;
AConstraintGradCov = NaN;

% Separate the two solver random number streams
solverInitialRng = solverRng{1}; % RNG for finding initial solutions
solverInternalRng = solverRng{2}; % RNG for the solver's internal randomness
% solverInternalRng is not used for Random Search

% Set default values
r = 30;  % Number of replications taken at each solution

% Get parameters of the problem
[minmax, dim, ~, ~, ~, ~, ~, ~, budgetR, ~, ~, ~] = probstructHandle(0);

NumFinSoln = numBudget + 1; % Number of solutions returned by solver (+1 for initial solution)
budget = [0, round(linspace(budgetR(1), budgetR(2), numBudget))];
if min(budget(2:end)) < r*(dim + 2) % Need to evaluate all initial solns in ssolsM
   fprintf('A budget is too small for a good quality run of Random Search.');
   return
end

% Using CRN: for each solution, start at substream 1
problemseed = 1;
% If not using CRN: problemseed needs to be updated throughout the code

% Determine maximum number of pts to evaluate at each budget point
numPointsV = floor(budget/r);

% Get initial solutions
RandStream.setGlobalStream(solverInitialRng)
[~, ~, ~, ~, ~, ~, ~, ssolsM, ~, ~, ~, ~] = probstructHandle(numPointsV(end));

% Initialize
fnV = zeros(dim + 1, 1);
fnVarV = zeros(dim + 1, 1);

% Evaluate all solutions
for i = 1:numPointsV(end)
    [fnV(i), fnVarV(i), ~, ~, ~, ~, ~, ~] = probHandle(ssolsM(i,:), r, problemRng, problemseed);
end

% Initialize
A = zeros(NumFinSoln, dim);
Afn = zeros(NumFinSoln, 1);
AFnVar = zeros(NumFinSoln, 1);
Ancalls = zeros(NumFinSoln, 1);

% Record first solution
A(1,:) = ssolsM(1,:);
Afn(1) = fnV(1);
AFnVar(1) = fnVarV(1);
Ancalls(1) = r;

% Record data from the best solution visited up to each budget point
for Bref = 2:NumFinSoln
    
    % Identify "best" solution visited up to budget point
    [~, bestId] = min(-minmax*fnV(1:numPointsV(Bref),:));
    
    A(Bref,:) = ssolsM(bestId,:);
    Afn(Bref) = fnV(bestId);
    AFnVar(Bref) = fnVarV(bestId);
    Ancalls(Bref) = numPointsV(Bref)*r;
    
end