%==========================================================================
%                The Random Search Algorithm
%==========================================================================
% DATE
%        Dec 2019
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
%
% OUTPUT
%        Ancalls
%              An array (size = 'NumSoln' X 1) of budget expended
%        A
%              An array (size = 'NumSoln' X 'dim') of solutions
%              returned by solver
%        AFnMean
%              An array (size = 'NumSoln' X 1) of estimates of expected
%              objective function value
%        AFnVar
%              An array of variances corresponding to
%              the objective function at A
%              Equals NaN if solution is infeasible
%        AFnGrad
%              An array of gradient estimates at A; not reported
%        AFnGradCov
%              An array of gradient covariance matrices at A; not reported
%        AConstraint
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
function [Ancalls, A, AFnMean, AFnVar, AFnGrad, AFnGradCov, ...
    AConstraint, AConstraintCov, AConstraintGrad, ...
    AConstraintGradCov] = RANDSH(probHandle, probstructHandle, ...
    problemRng, solverRng)


%% Unreported
AFnGrad = NaN;
AFnGradCov = NaN;
AConstraint = NaN;
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
[minmax, dim, ~, ~, ~, ~, ~, ~, budget, ~, ~, ~] = probstructHandle(0);

% Using CRN: for each solution, start at substream 1
problemseed = 1;
% If not using CRN: problemseed needs to be updated throughout the code

% Determine maximum number of solutions that can be sampled within max budget
numPointsV = floor(budget/r);

% Get initial solutions
RandStream.setGlobalStream(solverInitialRng)
[~, ~, ~, ~, ~, ~, ~, ssolsM, ~, ~, ~, ~] = probstructHandle(numPointsV);

% Initialize
FnMeanV = zeros(numPointsV, 1);
FnVarV = zeros(numPointsV, 1);

% Evaluate all solutions
for i = 1:numPointsV
    [FnMeanV(i), FnVarV(i), ~, ~, ~, ~, ~, ~] = probHandle(ssolsM(i,:), r, problemRng, problemseed);
end

% Initialize larger than necessary (extra point for end of budget)
Ancalls = zeros(numPointsV + 1, 1);
A = zeros(numPointsV + 1, dim);
AFnMean = zeros(numPointsV + 1, 1);
AFnVar = zeros(numPointsV + 1, 1);

% Record first solution
Ancalls(1) = 0;
A(1,:) = ssolsM(1,:);
AFnMean(1) = FnMeanV(1);
AFnVar(1) = FnVarV(1);

% Record only when recommended solution changes
record_index = 2;
oldbestID = 1;

% Record data from the best solution visited up to each budget point
for i = 1:numPointsV
    
    % Identify "best" solution visited up to budget point
    [~, newbestID] = min(-minmax*FnMeanV(1:i,:));
    
    if newbestID ~= oldbestID
        
        Ancalls(record_index) = i*r;
        A(record_index,:) = ssolsM(newbestID,:);
        AFnMean(record_index) = FnMeanV(newbestID);
        AFnVar(record_index) = FnVarV(newbestID);

        oldbestID = newbestID;
        record_index = record_index + 1;

    end
    
end

% Record solution at max budget
Ancalls(record_index) = budget;
A(record_index,:) = ssolsM(newbestID,:);
AFnMean(record_index) = FnMeanV(newbestID);
AFnVar(record_index) = FnVarV(newbestID);

% Trim empty rows from data
Ancalls = Ancalls(1:record_index);
A = A(1:record_index,:);
AFnMean = AFnMean(1:record_index);
AFnVar = AFnVar(1:record_index);
