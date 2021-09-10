%==========================================================================
%                       The SPSA Algorithm
%==========================================================================
% DATE
%        Feb 2017 (Updated in Dec 2019 by David Eckman)
%
% AUTHOR
%        Anna Dong
%
% REFERENCE	
%        James C. Spall, (1998). An Overview of the Simultaneous 
%        Perturbation Method for Efficient Optimization.
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

%% Simultaneous Perturbation Stochastic Approximation (SPSA)
function [Ancalls, A, AFnMean, AFnVar, AFnGrad, AFnGradCov, ...
    AConstraint, AConstraintCov, AConstraintGrad, ...
    AConstraintGradCov] = SPSA(probHandle, probstructHandle, problemRng, ...
    solverRng)

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

% Set default values
sensitivity = 10^(-7); % shrinking scale for VarBds
alpha = 0.602;
gamma = 0.101;
step = 0.1; % 'What is the initial desired magnitude of change in the theta elements?'
gavg = 1; % 'How many averaged SP gradients will be used per iteration? '
r = 30; % Number of replications takes at each solution
NL = 2; % 'How many loss function evaluations do you want to use in this gain calculation? '
% NL/(2*gavg) SP gradient estimates

% Get initial information about the problem
[minmax, dim, ~, ~, VarBds, ~, ~, ~, budget, ~, ~, ~] = probstructHandle(0);

% Shrink VarBds to prevent floating point errors
VarBds(:,1) = VarBds(:,1) + sensitivity;
VarBds(:,2) = VarBds(:,2) - sensitivity;

% Determine maximum number of solutions that can be sampled within max budget
MaxNumSoln = floor(budget/r); 

% Initialize larger than necessary (extra point for end of budget)
Ancalls = zeros(MaxNumSoln + 1, 1);
A = zeros(MaxNumSoln + 1, dim);
AFnMean = zeros(MaxNumSoln + 1, 1);
AFnVar = zeros(MaxNumSoln + 1, 1);

% Using CRN: for each solution, start at substream 1
problemseed = 1;
% If not using CRN: problemseed needs to be updated throughout the code

% Get initial solution
RandStream.setGlobalStream(solverInitialRng);
[~, ~, ~, ~, ~, ~, ~, theta0, ~, ~, ~, ~] = probstructHandle(1);

% Evaluate theta0
[ftheta, fthetaVar, ~, ~, ~, ~, ~, ~] = probHandle(theta0, r, problemRng, problemseed);
Bspent = r;
% fthetaVar answers 'What is the standard deviation of the measurement noise at i.c.? '
c = max(fthetaVar/gavg^0.5, .0001);

% Record initial solution data
Ancalls(1) = 0; 
A(1,:) = theta0;
AFnMean(1) = ftheta;
AFnVar(1) = fthetaVar;

% Record only when recommended solution changes
record_index = 2;

% Relabel initial solution and mark its estimate as best so far
theta = theta0;
ftheta_best = ftheta;

% Set other parameters
nEvals = round((budget/r)/3 *2); % 'What is the expected number of loss evaluations per run? '
Aalg = .10*nEvals/(2*gavg);

% Determine initial value for the parameter a (according to Section III.B of Spall (1998)) 
gbar = zeros(1, dim);

for i = 1:NL/(2*gavg)
   ghat = zeros(1, dim);
   for j = 1:gavg
      % Generate a random random direction (delta)
      RandStream.setGlobalStream(solverInternalRng);
      delta = 2*round(rand(1, dim)) - 1;

      % Determine points forward/backward relative to random direction
      thetaplus = theta + c*delta;
      thetaminus = theta - c*delta;
      thetaplus2 = checkCons(VarBds, thetaplus, theta);
      thetaminus2 = checkCons(VarBds, thetaminus, theta);

      % Evaluate two points and update budget spent
      [fn, ~] = probHandle(thetaplus2, r, problemRng, problemseed); 
      yplus = -minmax*fn;
      [fn, ~] = probHandle(thetaminus2, r, problemRng, problemseed); 
      yminus = -minmax*fn; 
      Bspent = Bspent + 2*r;

      % Estimate gradient
      ghat = (yplus - yminus)./(2*c*delta) + ghat;
   end
   gbar = gbar + abs(ghat/gavg);
end

meangbar = mean(gbar);
meangbar = meangbar/(NL/(2*gavg));
a = step*((Aalg + 1)^alpha)/meangbar;

% Run the main algorithm

k = 1; % Iteration counter

while Bspent <= budget
    
    ak = a/(k + Aalg)^alpha;
    ck = c/(k^gamma);
    
    % Generate a random random direction (delta)
    RandStream.setGlobalStream(solverInternalRng);
    delta = 2*round(rand(1, dim)) - 1;

    % Determine points forward/backward relative to random direction
    thetaplus = theta + ck*delta;
    thetaminus = theta - ck*delta;
    thetaplus2 = checkCons(VarBds, thetaplus, theta);
    thetaminus2 = checkCons(VarBds, thetaminus, theta);

    % Evaluate two points and update budget spent
    [fn, ~] = probHandle(thetaplus2, r, problemRng, problemseed); 
    yplus = -minmax*fn;
    [fn, ~] = probHandle(thetaminus2, r, problemRng, problemseed); 
    yminus = -minmax*fn;
    Bspent = Bspent + 2*r;
    
    % Estimate current solution's obj fun value by averaging 
    % estimates at thetaplus2 and thetaminus2
    ftheta = -minmax*(yplus + yminus)/2;
    
    % Check if current solution looks better than the best so far
    if -minmax*ftheta < -minmax*ftheta_best && Bspent <= budget
        
        theta_best = theta;
        ftheta_best = ftheta;
        
        %Record data from the new best solution
        Ancalls(record_index) = Bspent;
        A(record_index,:) = theta_best;
        AFnMean(record_index) = ftheta_best;
        AFnVar(record_index) = NaN; % no estimate of variance
        record_index = record_index + 1;
        
    end

    % Estimate gradient
    ghat = (yplus - yminus)./(2*c*delta);

    % Take step and check feasibility
    theta_next = theta - ak*ghat;
    theta = checkCons(VarBds, theta_next, theta);
    
end

% Record solution at max budget
Ancalls(record_index) = budget;
A(record_index,:) = A(record_index - 1,:);
AFnMean(record_index) = AFnMean(record_index - 1);
AFnVar(record_index) = AFnVar(record_index - 1);

% Trim empty rows from data
Ancalls = Ancalls(1:record_index);
A = A(1:record_index,:);
AFnMean = AFnMean(1:record_index);
AFnVar = AFnVar(1:record_index);

%% Helper Functions
% Helper 1: Check & Modify (if needed) the new point, based on VarBds.
% ssolsV2 original solution. ssolsV expected solution.
    function modiSsolsV = checkCons(VarBds, ssolsV, ssolsV2)
        col = size(ssolsV, 2);
        stepV = ssolsV - ssolsV2;
        % t>0 for the correct direction
        tmaxV = ones(2, col);
        uV = VarBds(stepV > 0, 2); uV = uV';
        lV = VarBds(stepV < 0, 1); lV = lV';
        if isempty(uV) == 0 %length(uV)> 0
            tmaxV(1, stepV > 0) = (uV - ssolsV2(stepV > 0)) ./ stepV(stepV > 0);
        end
        if isempty(lV) == 0 %length(lV)>0
            tmaxV(2,stepV < 0) = (lV - ssolsV2(stepV < 0)) ./ stepV(stepV < 0);
        end
        t = min(min(tmaxV));
        modiSsolsV = ssolsV2 + t*stepV;
        %rounding error, may remove
        for kc = 1:col
            if abs(modiSsolsV(kc)) < 0.00000005
                modiSsolsV(kc) = 0;
            end
        end
    end

end


