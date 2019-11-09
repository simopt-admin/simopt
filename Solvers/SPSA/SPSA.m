%==========================================================================
%                       The SPSA Algorithm
%==========================================================================
% DATE
%        Feb 2017 (Updated in Sept 2018 by David Eckman)
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
%        numBudget
%              number of budgets to record, >=3; the spacing between
%              adjacent budget points should be about the same
%        logOption
%              produce log if =1, not write a log if =0
%        logfilename
%              string, no need for .txt
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
%        AFnGradCov
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

%% Simultaneous Perturbation Stochastic Approximation (SPSA)
function [Ancalls, A, Afn, AFnVar, AFnGrad, AFnGradCov, ...
    Aconstraint, AConstraintCov, AConstraintGrad, ...
    AConstraintGradCov] = SPSA(probHandle, probstructHandle, problemRng, ...
    solverRng, numBudget)

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
[minmax, dim, ~, ~, VarBds, ~, ~, ~, budgetR, ~, ~, ~] = probstructHandle(0);

% Shrink VarBds to prevent floating point errors
VarBds(:,1) = VarBds(:,1) + sensitivity;
VarBds(:,2) = VarBds(:,2) - sensitivity;

% Setup budget
NumFinSoln = numBudget + 1; % Number of solutions returned by solver (+1 for initial solution)
budget = [0, round(linspace(budgetR(1), budgetR(2), numBudget))];
Bref = 1; % The budget currently referred to, = 1, ..., numBudget + 1

% Get initial solution
RandStream.setGlobalStream(solverInitialRng);
[~, ~, ~, ~, ~, ~, ~, theta0, ~, ~, ~, ~] = probstructHandle(1);

% Initialize
A = zeros(NumFinSoln, dim);
Afn = zeros(NumFinSoln, 1);
AFnVar = zeros(NumFinSoln, 1);
Ancalls = zeros(NumFinSoln, 1);

% Using CRN: for each solution, start at substream 1
problemseed = 1;
% If not using CRN: problemseed needs to be updated throughout the code

% Evaluate theta0
[ftheta, fthetaVar, ~, ~, ~, ~, ~, ~] = probHandle(theta0, r, problemRng, problemseed);
Bspent = r;
% fthetaVar answers 'What is the standard deviation of the measurement noise at i.c.? '
c = max(fthetaVar/gavg^0.5, .0001);

% Record initial solution data
A(Bref,:) = theta0;
Afn(Bref) = ftheta;
AFnVar(Bref) = fthetaVar;
Ancalls(Bref) = Bspent;
Bref = Bref + 1;

% Relabel initial solution and mark as the best so far
theta = theta0;
theta_best = theta;
ftheta_best = ftheta;
fthetaVar_best = fthetaVar;

% Set other parameters
nEvals = round((budgetR(2)/r)/3 *2); % 'What is the expected number of loss evaluations per run? '
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

while Bspent <= budgetR(2)
    
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

    % Estimate gradient
    ghat = (yplus - yminus)./(2*c*delta);

    % Take step and check feasibility
    theta_next = theta - ak*ghat;
    theta = checkCons(VarBds, theta_next, theta);

    % Evaluate new solution and update budget spent
    [ftheta, fthetaVar, ~, ~, ~, ~, ~, ~] = probHandle(theta, r, problemRng, problemseed);
    Bspent = Bspent + r;

    % Check if new solution is an improvement
    if -minmax*ftheta < -minmax*ftheta_best
        theta_best = theta;
        ftheta_best = ftheta;
        fthetaVar_best = fthetaVar;
    end        

    % Check if finish referring to current budget
    while Bspent + 2*r > budget(Bref)
        % Record current best soln
        A(Bref,:) = theta_best;
        Afn(Bref) = -minmax*ftheta_best;
        AFnVar(Bref) = fthetaVar_best;
        Ancalls(Bref) = Bspent;
        Bref = Bref + 1; % Now refer to next budget
        if Bref > numBudget + 1 % If exceeds the max budget
            return
        end   
    end
    
end

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


