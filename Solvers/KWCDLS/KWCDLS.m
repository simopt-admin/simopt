%==========================================================================
% Kiefer-Wolfowitz Central Differences with Line Search and Random Restarts
%==========================================================================
% DATE
%        Nov 2016
%
% AUTHOR
%        Xueqi Zhao
%
% REFERENCE
%       Based on the Gradient Search Algorithm by Anna Dong and Nellie Wu
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

%% Gradient Search Constrained
function [Ancalls, A, Afn, AFnVar, AFnGrad, AFnGradCov, ...
    Aconstraint, AConstraintCov, AConstraintGrad, AConstraintGradCov] = ...
    KWCDLS(probHandle, probstructHandle, problemRng, solverRng, ...
    numBudget)

% Unreported
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
thres = 0.000005;
r = 30;
RestartThres = 0.0001;
sensitivity = 10^(-7); % shrinking scale for VarBds

% Generate two initial solutions (will use the first) and get other inputs
RandStream.setGlobalStream(solverInitialRng);
[minmax, dim, ~, ~, VarBds, ~, ~, x0, budgetR, ~, ~, ~] = probstructHandle(2);

% Shrink VarBds to prevent floating errors
VarBds(:,1) = VarBds(:,1) + sensitivity;
VarBds(:,2) = VarBds(:,2) - sensitivity;

% Scale steph
steph = ones(1,dim)*min(abs(x0(2,:)-x0(1,:)))/3;
x0 = x0(1,:); % Use the first solution as the initial solution

NumFinSoln = numBudget + 1; % Number of solutions returned by solver (+1 for initial solution)
budget = [0, round(linspace(budgetR(1), budgetR(2), numBudget))];

% Initialize
A = zeros(NumFinSoln, dim);
Afn = zeros(NumFinSoln, 1);
AFnVar = zeros(NumFinSoln, 1);
Ancalls = zeros(NumFinSoln, 1);

% Using CRN: for each solution, start at substream 1
problemseed = 1;
% If not using CRN: problemseed needs to be updated throughout the code

% Initialize
x0current = x0;
x0best = x0;
graV = zeros(1,dim);

% Evaluate the initial solution
[fn0current, fn0varcurrent, ~, ~, ~, ~, ~, ~] = probHandle(x0current, r, problemRng, problemseed);
fn0current = -minmax*fn0current;
Bref = 1; % The budget currently referred to, = 1,...,numBudget
Bspent = r;
fn0best = fn0current;
fn0varbest = fn0varcurrent;

while Bspent <= budgetR(2)

    % Check if finish referring to current budget
    while Bspent + 2*r*dim > budget(Bref)
        % Record current best solution
        Ancalls(Bref) = Bspent;
        A(Bref,:) = x0best;
        Afn(Bref) = -minmax*fn0best;
        AFnVar(Bref) = fn0varbest;     
        Bref = Bref + 1; % Now refer to next budget
        if Bref > numBudget + 1 % If exceeds the max budget
            return
        end
    end
    
    % Approximate gradient via central finite differences
    for i = 1:dim
        steph1 = steph(i);
        steph2 = steph(i);
        
        x1 = x0current;
        x1(i) = x1(i) + steph1; % step forward
        if x1(i) > VarBds(i,2) % if ==, then same as backward diff
            x1(i) = VarBds(i,2);
            steph1 = abs(x1(i) - x0current(i)); % can remove abs()
        end
        [fn1, ~, ~, ~, ~, ~, ~, ~] = probHandle(x1, r, problemRng, problemseed);
        fn1 = -minmax*fn1;

        x2 = x0current;
        x2(i) = x2(i)-steph2; % step backward
        if x2(i) < VarBds(i,1) % if ==, then same as forward diff
            x2(i) = VarBds(i,1);
            steph2 = abs(x0current(i) - x2(i)); % can remove abs()
        end        
        [fn2, ~, ~, ~, ~, ~, ~, ~] = probHandle(x2, r, problemRng, problemseed);
        fn2 = -minmax*fn2;
        
        graV(i) = (fn1-fn2)/(steph1 + steph2);
    end
    
    if Bspent == r % If this was the first iteration...
        steph = abs(sqrt(fn0varcurrent)./(sqrt(2*r)*graV));
    end
    
    % Take a step opposite the gradient and evaluate the new solution xG
    t = 2; % initialize step size
    xG = x0current - t.*graV;
    xG = checkCons(VarBds, xG, x0current);
    [fnG, fnGvar, ~, ~, ~, ~, ~, ~] = probHandle(xG, r, problemRng, problemseed);
    fnG = -minmax*fnG;
    
    % Update best soln so far
    if fnG < fn0best
        x0best = xG;
        fn0best = fnG;
        fn0varbest = fnGvar;
    end
    
    % Update budget
    Bspent = Bspent + (2*dim + 1)*r;
    
    % Check if finish referring to current budget
    while Bspent + r > budget(Bref)
        % Record current best solution
        Ancalls(Bref) = Bspent;
        A(Bref,:) = x0best;
        Afn(Bref) = -minmax*fn0best;
        AFnVar(Bref) = fn0varbest;     
        Bref = Bref + 1; % Now refer to next budget
        if Bref > numBudget + 1 % If exceeds the max budget
            return
        end
    end
        
    % Backtracking (bisecting) line search (if no improvement)
    while fnG >= fn0current && t >= thres
        t = 0.5*t;
        xG = x0current - t.*graV;
        xG = checkCons(VarBds, xG, x0current);
        
        % Evaluate the new solution xG
        [fnG, fnGvar, ~, ~, ~, ~, ~, ~] = probHandle(xG, r, problemRng, problemseed);
        fnG = -minmax*fnG;
        % Update best soln so far
        if fnG < fn0best
            x0best = xG;
            fn0best = fnG;
            fn0varbest = fnGvar;
        end
        
        % Update budget
        Bspent = Bspent + r;
        
        % Check if finish referring to current budget
        while Bspent + r > budget(Bref)
            % Record current best solution
            Ancalls(Bref) = Bspent;
            A(Bref,:) = x0best;
            Afn(Bref) = -minmax*fn0best;
            AFnVar(Bref) = fn0varbest;     
            Bref = Bref + 1; % Now refer to next budget
            if Bref > numBudget + 1 % If exceeds the max budget
                return
            end
        end
    end
    
    % If no significant improvement, then randomly jump to another point
    if (fn0current - fnG < RestartThres*(1 + abs(fnG))) && (norm(x0current - xG) < sqrt(RestartThres)*(1 + norm(xG))) && (norm(graV) <= nthroot(RestartThres, 3)*(1 + abs(fnG))) && (norm(graV) < fn0varcurrent)
        
        % Restart at a randomly generated solution
        RandStream.setGlobalStream(solverInternalRng);
        [~, ~, ~, ~, ~, ~, ~, x0current, ~, ~, ~, ~] = probstructHandle(1);

        % Check if finish referring to current budget
        while Bspent + r > budget(Bref)
            % Record current best solution
            Ancalls(Bref) = Bspent;
            A(Bref,:) = x0best;
            Afn(Bref) = -minmax*fn0best;
            AFnVar(Bref) = fn0varbest;     
            Bref = Bref + 1; % Now refer to next budget
            if Bref > numBudget + 1 % If exceeds the max budget
                return
            end
        end
        
        % Evaluate (new) current solution
        [fn0current, fn0varcurrent, ~, ~, ~, ~, ~, ~] = probHandle(x0current, r, problemRng, problemseed);
        fn0current = -minmax*fn0current;
             
        % Update best soln so far
        if fn0current < fn0best
            x0best = x0current;
            fn0best = fn0current;
            fn0varbest = fn0varcurrent;
        end
        
        % Update budget
        Bspent = Bspent + r;
        
        % Check if finish referring to current budget
        while Bspent + r > budget(Bref)
            % Record current best solution
            Ancalls(Bref) = Bspent;
            A(Bref,:) = x0best;
            Afn(Bref) = -minmax*fn0best;
            AFnVar(Bref) = fn0varbest;     
            Bref = Bref + 1; % Now refer to next budget
            if Bref > numBudget + 1 % If exceeds the max budget
                return
            end
        end
        
    else
        x0current = xG;
        fn0current = fnG;
    
    end
    
end 

%% check and modify (if needed) the new point, based on VarBds.
    function modiSsolsV = checkCons(VarBds, ssolsV, ssolsV2) 
        col = size(ssolsV,2);
        stepV = ssolsV - ssolsV2;
        %t = 1; % t > 0 for the correct direction
        tmaxV = ones(2,col);
        uV = VarBds(stepV > 0,2); uV = uV';
        lV = VarBds(stepV < 0,1); lV = lV';
        if isempty(uV) == 0
            tmaxV(1,stepV > 0) = (uV - ssolsV2(stepV > 0)) ./ stepV(stepV > 0);
        end
        if isempty(lV) == 0
            tmaxV(2,stepV < 0) = (lV - ssolsV2(stepV < 0)) ./ stepV(stepV < 0);
        end
        t2 = min(min(tmaxV));
        modiSsolsV = ssolsV2 + t2*stepV;
        %rounding error
        for kc = 1:col
            if modiSsolsV(kc) < 0 && modiSsolsV(kc) > -0.00000005
                modiSsolsV(kc) = 0;
            end
        end
    end

end