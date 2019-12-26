%==========================================================================
%                 The Anderson-Ferris Direct Search Algorithm
%==========================================================================
% DATE
%        Dec 2017 (Updated in Dec 2019 by David Eckman)
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

%% Anderson-Ferris
function [Ancalls, A, AFnMean, AFnVar, AFnGrad, AFnGradCov, ...
    AConstraint, AConstraintCov, AConstraintGrad, ...
    AConstraintGradCov] = ANDFER(probHandle, probstructHandle, problemRng, solverRng)

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
r = 30; % Number of replications taken at each solution
sensitivity = 10^(-7); % shrinking scale for VarBds
seq = 10^(-18); % \eta sequence from Anderson-Ferris paper

% Generate 10 random solutions and compute the std dev in each dimension
RandStream.setGlobalStream(solverInternalRng);
[~, ~, ~, ~, ~, ~, ~, ssolsM, ~, ~, ~, ~] = probstructHandle(10);
e2 = std(ssolsM);

% Generate new starting point x0
RandStream.setGlobalStream(solverInitialRng);
[minmax, dim, ~, ~, VarBds, ~, ~, x0, budget, ~, ~, ~] = probstructHandle(1); 

% Shrink VarBds to prevent floating errors
VarBds(:,1) = VarBds(:,1) + sensitivity; 
VarBds(:,2) = VarBds(:,2) - sensitivity;

% Check for sufficiently large budget
numExtPts = 4*dim + 1; % Using (4d+1)-point cross structure
if budget < r*numExtPts % Need to evaluate all initial solns in ssolsM
    fprintf('The budget is too small for a good quality run of Anderson-Ferris.');
    return
end

% Determine maximum number of solutions that can be sampled within max budget
MaxNumSoln = floor(budget/r);

% Initialize larger than necessary (extra point for end of budget)
Ancalls = zeros(MaxNumSoln, 1);
A = zeros(MaxNumSoln, dim);
AFnMean = zeros(MaxNumSoln, 1);
AFnVar = zeros(MaxNumSoln, 1);

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

% Track overall budget spent
Bspent = 0;

%% Start Solving

% For reporting purposes only, evaluate x0 and record data. Do not count
% the function evaluations towards the budget.
% Record initial solution data
Ancalls(1) = 0;
A(1,:) = x0;
[AFnMean(1), AFnVar(1), ~, ~, ~, ~, ~, ~] = probHandle(x0, r, problemRng, problemseed);

% Record only when recommended solution changes
record_index = 2;

% Evaluate points in initial structure and sort
[ssolsMl2h, l2hfnV, l2hfnVarV] = evalExtM(ssolsM, numExtPts,r, probHandle, problemRng, problemseed, minmax);
Bspent = Bspent + r*numExtPts; % Total budget spent
b = l2hfnV(1); % Best obj function value in initial structure

% Record data for best solution in initial structure
Ancalls(record_index) = Bspent;
A(record_index,:) = ssolsMl2h(1,:);
AFnMean(record_index) = -minmax*l2hfnV(1); % flip sign back
AFnVar(record_index) = l2hfnVarV(1);
record_index = record_index + 1;

while Bspent <= budget

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
    %vT = TssolsMl2h(1,:); % Best solution in T

    % Reporting
    if FT < FS && Bspent <= budget
        
        Ancalls(record_index) = Bspent;
        A(record_index,:) = TssolsMl2h(1,:);
        AFnMean(record_index) = -minmax*Tl2hfnV(1); % flip sign back
        AFnVar(record_index) = Tl2hfnVarV(1);
        record_index = record_index + 1;
    end

    if FT < FS % Best val in reflected T is better than best val in old S
        if FT < b
            b = FT;
        end
        
        % Generate expanded structure (this way first point is still vS)
        U = 2*TssolsMl2h - repmat(vS, numExtPts, 1);
        
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
          
        % Determine whether to use expanded or reflected structure
        if FU < b - seq % Accept expansion
            b = FU;
            ssolsMl2h = UssolsMl2h; % Set S = U for next iteration
            l2hfnV = Ul2hfnV;
            l2hfnVarV = Ul2hfnVarV;
            
            % Reporting
            if Bspent <= budget
                Ancalls(record_index) = Bspent;
                A(record_index,:) = UssolsMl2h(1,:);
                AFnMean(record_index) = -minmax*Ul2hfnV(1); % flip sign back
                AFnVar(record_index) = Ul2hfnVarV(1);
                record_index = record_index + 1;
            end
            
        else % Accept reflection
            ssolsMl2h = TssolsMl2h; % Set S = T for next iteration
            l2hfnV = Tl2hfnV;
            l2hfnVarV = Tl2hfnVarV;
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
        
        if FC < b  % Accept contraction
            b = FC; 
        end
        
        % Set S = C for next iteration
        ssolsMl2h = CssolsMl2h;
        l2hfnV = Cl2hfnV;
        l2hfnVarV = Cl2hfnVarV;
                
        % Reporting
        if Bspent <= budget
            Ancalls(record_index) = Bspent;
            A(record_index,:) = CssolsMl2h(1,:);
            AFnMean(record_index) = -minmax*Cl2hfnV(1); % flip sign back
            AFnVar(record_index) = Cl2hfnVarV(1);
            record_index = record_index + 1;
        end
    
    end 
    
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
        
        % Calculate the step size (and sign) in dimension d
        stepV = A - B;
        
        % Will determine if the original step sizes take the new structure
        % out of bounds.
        
        % For each point in the sturcture, record the ratio of the maximum 
        % allowable step size to the desired the step size. Initialize to 1.
        tmaxV = ones(numExtPts, 2);
   
        uV = VarBds(d, 2); 
        lV = VarBds(d, 1);
        % Consider steps that increase/decrease the d-coordinate separately.
        if max(A) > uV
            tmaxV(stepV > 0, 2) = (uV - B(stepV > 0))./stepV(stepV > 0);
        end
        if min(A) < lV
            tmaxV(stepV < 0, 1) = (lV - B(stepV < 0))./stepV(stepV < 0);
        end
        t = min(min(tmaxV));
        
        % Take steps for each point according to the smallest maximum 
        % allowed ratio 
        modi = B + t*stepV;
            
    end 
end

