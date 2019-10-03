%==========================================================================
%                 The Anderson-Ferris Direct Search Algorithm
%==========================================================================
% DATE
%        Dec 2017 (Updated in Jan 2018 by David Eckman)
%
% AUTHOR
%        Jae Won Jung, David Eckman
%
% REFERENCE		
%        Edward J. Anderson, Michael C. Ferris, (2001)
%		 A Direct Search Algorithm for Optimization with 
%        Noisy Function Evaluations.11(3):837-857
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

%% Anderson-Ferris
function [Ancalls, A, Afn, AFnVar, AFnGrad, AFnGradCov, ...
    Aconstraint, AConstraintCov, AConstraintGrad, ...
    AConstraintGradCov] = ANDFER(probHandle, probstructHandle, problemRng, solverRng, ...
    numBudget)

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
r = 30; % Number of replications taken at each solution
sensitivity = 10^(-7); % shrinking scale for VarBds
seq = 10^(-18); % \eta sequence from Anderson-Ferris paper

% Generate 10 random solutions and compute the std dev in each dimension
RandStream.setGlobalStream(solverInternalRng);
[~, ~, ~, ~, ~, ~, ~, ssolsM, ~, ~, ~, ~] = probstructHandle(10);
e2 = std(ssolsM);

% Generate new starting point x0
RandStream.setGlobalStream(solverInitialRng);
[minmax, dim, ~, ~, VarBds, ~, ~, x0, budgetR, ~, ~, ~] = probstructHandle(1); 
numExtPts = 4*dim + 1; % Using (4d+1)-point cross structure

% Shrink VarBds to prevent floating errors
VarBds(:,1) = VarBds(:,1) + sensitivity; 
VarBds(:,2) = VarBds(:,2) - sensitivity;

NumFinSoln = numBudget + 1; % Number of solutions returned by solver (+1 for initial solution)
budget = [0, round(linspace(budgetR(1), budgetR(2), numBudget))];
if budget(end) < r*numExtPts % Need to evaluate all initial solns in ssolsM
    fprintf('The budget is too small for a good quality run of Anderson-Ferris.');
    return
end

% Initialize
A = zeros(NumFinSoln, dim);
Afn = zeros(NumFinSoln, 1);
AFnVar = zeros(NumFinSoln, 1);
Ancalls = zeros(NumFinSoln, 1);

% Create initial (cross) structure of 4d+1 points
ssolsM = repmat(x0, numExtPts, 1);
for d = 1:dim     
    ssolsM(4*d - 2, d) = x0(d) + e2(d);
    ssolsM(4*d - 1, d) = x0(d) + 2*e2(d);
    ssolsM(4*d, d) = x0(d) - e2(d);
    ssolsM(4*d + 1, d) = x0(d) - 2*e2(d);
end

% Ensure initial structure is within bounds
 for d = 1:dim
    % Violated points are moved back in the direction of x0, in each dim
    ssolsM(:,d) = modify(VarBds, ssolsM(:,d), x0(d)*ones(numExtPts, 1), d, numExtPts);
 end

% Using CRN: for each solution, start at substream 1
problemseed = 1;
% If not using CRN: problemseed needs to be updated throughout the code
 
%% Start Solving
%display(['Maximum Budget = ',num2str(budgetR(2)),'.'])
Bref = 1; % The budget currently referred to, = 1, ..., numBudget + 1

% Evaluate points in initial structure and sort
[ssolsMl2h, l2hfnV, l2hfnVarV] = evalExtM(ssolsM, numExtPts,r, probHandle, problemRng, problemseed, minmax);
Bspent = r*numExtPts; % Total budget spent
b = l2hfnV(1); % Best obj function value in initial structure

% For reporting purposes later
bestFnVal = b;
bestPt = ssolsMl2h(1,:);
bestFnValVar = l2hfnVarV(1);

while Bspent <= budgetR(2)

    % Structure S is stored in ssolsM12h
    FS = l2hfnV(1); % Best obj function value in structure S
    vS = ssolsMl2h(1, :); % Best point in structure S

    % Compute reflected structure T, with respect to current best pt vS    
    T = 2*repmat(vS, numExtPts, 1) - ssolsMl2h;

    % Check if reflected structure respects VarBds, if not, modify it   
    for d = 1:dim
        T(:,d) = modify(VarBds, T(:,d), ssolsMl2h(:,d), d, numExtPts);
    end
        
    % Evaluate the new points in the reflected structure
    [TssolsMl2h, Tl2hfnV, Tl2hfnVarV] = evalExtM(T(2:numExtPts,:), numExtPts - 1, r, probHandle, problemRng, problemseed, minmax);
    Bspent = Bspent + r*(numExtPts - 1); % Total budget spent

    % Store points, fn values, and fn value variances for all points in T
    TssolsM = [ssolsMl2h(1, :); TssolsMl2h];
    TfnV = [l2hfnV(1); Tl2hfnV];
    TfnVar = [l2hfnVarV(1); Tl2hfnVarV];
    
    % Find best point in reflected structure via sorting
    [Tl2hfnV, Tl2hfnIndV1] = sort(TfnV);
    TssolsMl2h = TssolsM(Tl2hfnIndV1,:);
    Tl2hfnVarV = TfnVar(Tl2hfnIndV1,:);
    
    FT = Tl2hfnV(1); % Best objective value in T
    vT = TssolsMl2h(1,:); % Best solution in T

    % For reporting purposes
    if FT < bestFnVal
        bestFnVal = FT;
        bestPt = vT;
        bestFnValVar = Tl2hfnVarV(1);        
    end

    % Check if finish referring to current budget
    while Bspent + r > budget(Bref)
        % Record current best soln
        A(Bref,:) = bestPt;
        Afn(Bref) = -minmax*bestFnVal;
        AFnVar(Bref) = bestFnValVar;
        Ancalls(Bref) = Bspent;
        Bref = Bref + 1; % Now refer to next budget
        if Bref > numBudget + 1 % If exceeds the max budget
            return
        end   
    end
        
    if FT < FS % Best val in reflected T is better than best val in old S
        if FT < b
            b = FT;
        end
        
        % Generate expanded structure (this way first point is still vS)
        U = 2*TssolsM - repmat(vS, numExtPts, 1);
        
        % Check if expanded structure respects VarBds, if not, modify it
        for d = 1:dim
            U(:,d) = modify(VarBds, U(:,d), TssolsMl2h(:,d), d, numExtPts);
        end
        
        % Evaluate the new points in the expanded structure U
        [UssolsMl2h, Ul2hfnV, Ul2hfnVarV] = evalExtM(U(2:numExtPts,:), numExtPts - 1, r,  probHandle, problemRng, problemseed, minmax);
        Bspent = Bspent + r*(numExtPts - 1); % Total budget spend

        % Store points, fn values, and fn value variances for all points in T
        UssolsM = [ssolsMl2h(1,:); UssolsMl2h];
        UfnV=[l2hfnV(1); Ul2hfnV];
        UfnVar = [l2hfnVarV(1); Ul2hfnVarV];

        % Find best point in expanded structure
        [Ul2hfnV, Ul2hfnIndV1] = sort(UfnV);
        UssolsMl2h = UssolsM(Ul2hfnIndV1,:);
        Ul2hfnVarV = UfnVar(Ul2hfnIndV1,:);
        FU = Ul2hfnV(1); % Best objective value in U
        vU = UssolsMl2h(1,:); % Best solution in U
          
        % Determine whether to use expanded or reflected structure
        if FU < b - seq % Accept expansion
            b = FU;
            ssolsMl2h = UssolsMl2h; % Set S = U for next iteration
            l2hfnV = Ul2hfnV;
            l2hfnVarV = Ul2hfnVarV;            
        else % Accept reflection
            ssolsMl2h = TssolsMl2h; % Set S = T for next iteration
            l2hfnV = Tl2hfnV;
            l2hfnVarV = Tl2hfnVarV;
        end
        
        % For reporting purposes
        if FU < bestFnVal
            bestFnVal = FU;
            bestPt = vU;
            bestFnValVar = Ul2hfnVarV(1); 
        end
        
        % Check if finish referring to current budget
        while Bspent + r > budget(Bref)
            % Record current best soln
            A(Bref,:) = bestPt;
            Afn(Bref) = -minmax*bestFnVal;
            AFnVar(Bref) = bestFnValVar;
            Ancalls(Bref) = Bspent;
            Bref = Bref + 1; % Now refer to next budget
            if Bref > numBudget + 1 % If exceeds the max budget
                return
            end   
        end
    
    else % Contraction: no improvement
        
        % Generate contracted structure
        C = 0.5*(repmat(vS, numExtPts,1) + ssolsMl2h);
        
        % Evaluate all (including the pivot) point in the contracted structure C
        [CssolsMl2h, Cl2hfnV, Cl2hfnVarV] = evalExtM(C, numExtPts, r, probHandle, problemRng, problemseed, minmax);
        Bspent = Bspent + r*numExtPts;
        % !! If using CRN, the pivot point vS will have the same estimate
        % In which case, there wouldn't be a need to re-evaluate vS
        
        FC = Cl2hfnV(1); % Best objective value in C
        vC = CssolsMl2h(1); % Best solution in C
        
        if FC < b  % Accept contraction
            b = FC; 
        end
        
        % Set S = C for next iteration
        ssolsMl2h = CssolsMl2h;
        l2hfnV = Cl2hfnV;
        l2hfnVarV = Cl2hfnVarV;
        
        % For reporting purposes
        if FC < bestFnVal
            bestFnVal = FC;
            bestPt = vC;
            bestFnValVar = Cl2hfnVarV(1);
        end
        
        % Check if finish referring to current Budget
        while Bspent + r > budget(Bref)
            % Record current best soln
            A(Bref,:) = bestPt;
            Afn(Bref) = -minmax*bestFnVal;
            AFnVar(Bref) = bestFnValVar;
            Ancalls(Bref) = Bspent;
            Bref = Bref + 1; % Now refer to next budget
            if Bref > numBudget + 1 % If exceeds the max budget
                return
            end   
        end
    
    end 
    
end
 

%% Helper Functions
% Helper 1: Evaluate obj fcn values z at all extreme points & Sort low2high
% If called, will spend (r*numExtPts) budget.
% Maximization problem is converted to minimization by -z.
    function [ssolsMl2h, l2hfnV, l2hfnVarV] = evalExtM(ssolsM, numExtPts,...
            r, probHandle, problemRng, problemseed, minmax)
        fnV = zeros(numExtPts, 1); % To track soln
        fnVarV = zeros(numExtPts, 1);
        for i1 = 1:numExtPts
            [fn, FnVar, ~, ~, ~, ~, ~, ~] = probHandle(ssolsM(i1,:), r, problemRng, problemseed);
            fnV(i1) = -minmax*fn; % Minimize fn
            fnVarV(i1) = FnVar;
        end
        [l2hfnV, l2hfnIndV1] = sort(fnV);
        l2hfnVarV = fnVarV(l2hfnIndV1,:);
        ssolsMl2h = ssolsM(l2hfnIndV1,:);
    end


% Helper 2: Check & Modify (if needed) the new matrix, based on VarBds.
    function modi = modify(VarBds, A, B, d, numExtPts)
        stepV = A - B;
        tmaxV = ones(numExtPts, 2);
   
        uV = VarBds(d, 2); 
        lV = VarBds(d, 1); 
        if max(A) > uV
            tmaxV(stepV > 0, 2) = (uV - B(stepV > 0))./stepV(stepV > 0);
        end
        if min(A) < lV
            tmaxV(stepV < 0, 1) = (lV - B(stepV < 0))./stepV(stepV < 0);
        end
        t = min(min(tmaxV));
        modi = B + t*stepV;
            
    end 
end

