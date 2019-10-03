%==========================================================================
%                       The Nelder-Mead Algorithm
%==========================================================================
% DATE
%        Feb 2016 (Updated in Oct 2019 by David Eckman)
%
% AUTHOR
%        Anna Dong, Nellie Wu

% REFERENCE		
%       Russell R. Barton, John S. Ivey, Jr., (1996)
%		Nelder-Mead Simplex Modifications for Simulation 
%		Optimization. Management Science 42(7):954-973.
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

%% Nelder-Mead
function [Ancalls, A, Afn, AFnVar, AFnGrad, AFnGradCov, ...
    Aconstraint, AConstraintCov, AConstraintGrad, ...
    AConstraintGradCov] = NELDMD(probHandle, probstructHandle, problemRng, ...
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
% Nelder-Mead does not use any internal randomness

% Set default values
r = 30; % Number of replications taken at each solution
alpha = 1;
gammap = 2;
betap = 0.5;
delta = 0.5;
sensitivity = 10^(-7); % shrinking scale for VarBds

% Get details of the problem
[minmax, dim, ~, ~, VarBds, ~, ~, ~, budgetR, ~, ~, ~] = probstructHandle(0);

% Shrink VarBds to prevent floating errors
VarBds(:,1) = VarBds(:,1) + sensitivity; 
VarBds(:,2) = VarBds(:,2) - sensitivity;

% Setup budget
numExtPts = dim + 1;
NumFinSoln = numBudget + 1; % Number of solutions returned by solver (+1 for initial solution)
budget = [0, round(linspace(budgetR(1), budgetR(2), numBudget))];
if min(budget(2:end)) < r*numExtPts % Need to evaluate all initial solns in ssolsM
    fprintf('A budget is too small for a good quality run of Anderson-Ferris.');
    return
end

% Generate initial simplex from dim+1 random points
RandStream.setGlobalStream(solverInitialRng);
[~, ~, ~, ~, ~, ~, ~, ssolsM, ~, ~, ~] = probstructHandle(numExtPts);

% Initialize
A = zeros(NumFinSoln, dim);
Afn = zeros(NumFinSoln, 1);
AFnVar = zeros(NumFinSoln, 1);
Ancalls = zeros(NumFinSoln, 1);

% Using CRN: for each solution, start at substream 1
problemseed = 1;
% If not using CRN: problemseed needs to be updated throughout the code

%% Start Solving
%display(['Maximum Budget = ',num2str(budgetR(2)),'.'])
Bref = 1; % The budget currently referred to, = 1, ..., numBudget + 1

% Evaluate points in initial structure and sort low to high
[ssolsMl2h, l2hfnV, l2hfnVarV] = evalExtM(ssolsM, numExtPts, r, probHandle, problemRng, problemseed, minmax);
Bspent = r*numExtPts; % Total budget spent

% Reflect Worst & Update ssolsMl2h
% Maximization problem is converted to minimization by -z.
while Bspent <= budgetR(2)
    % Reflect worst point
    Phigh = ssolsMl2h(end,:); % Current worst pt
    Pcent = mean(ssolsMl2h(1:end-1,:)); % Centroid for other pts
    Prefl2 = Phigh; % Save the original point
    Prefl = (1+alpha)*Pcent - alpha*Phigh; % Reflection
    Prefl = checkCons(VarBds,Prefl,Prefl2); % Check if Prefl respects VarBds (if not, change it)
        
    
    % Check if finish referring to current Budget
    while Bspent + r > budget(Bref)
        % Record current best soln
        A(Bref,:) = ssolsMl2h(1,:);
        Afn(Bref) = -minmax*l2hfnV(1);
        AFnVar(Bref) = l2hfnVarV(1);
        Ancalls(Bref) = Bspent;
        Bref = Bref + 1; % Now refer to next budget
        if Bref > numBudget + 1% If exceeds the current Budget
            return
        end   
    end
    
    [Frefl, FreflVar, ~, ~, ~, ~, ~, ~] = probHandle(Prefl, r, problemRng, problemseed); % Cost r
    Bspent = Bspent + r;
    Frefl = -minmax*Frefl;
    
    %
    Plow = ssolsMl2h(1,:); % Current best pt
    Flow = l2hfnV(1);
    Fsechi = l2hfnV(end-1); % Current 2nd worst z
    Fhigh = l2hfnV(end);
    % Check if accept reflection
    if Flow<=Frefl && Frefl<=Fsechi
        ssolsMl2h(end,:) = Prefl; % Prefl replaces Phigh
        l2hfnV(end) = Frefl;
        l2hfnVarV(end) = FreflVar;
        % Sort & End updating
        [l2hfnV,l2hfnIndV] = sort(l2hfnV);
        l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
        ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
    % Check if accept expansion (of reflection in the same direction)
    elseif Frefl<Flow
        Pexp2 = Prefl;
        Pexp = gammap*Prefl + (1-gammap)*Pcent;
        Pexp = checkCons(VarBds,Pexp,Pexp2);
            % Check if finish referring to current Budget
            if Bspent + r > budget(Bref)
                % Record current best soln
                A(Bref,:) = ssolsMl2h(1,:);
                Afn(Bref) = -minmax*l2hfnV(1);
                AFnVar(Bref) = l2hfnVarV(1);
                Ancalls(Bref) = Bspent;
                Bref = Bref + 1; % Now refer to next budget
                if Bref > numBudget + 1 % If exceeds the current Budget
                    return
                end   
            end
        [Fexp, FexpVar, ~, ~, ~, ~, ~, ~] = probHandle(Pexp, r, problemRng, problemseed); % Cost r
            Bspent = Bspent + r;        
        Fexp = -minmax*Fexp;
        if Fexp<Flow
            ssolsMl2h(end,:) = Pexp; % Pexp replaces Phigh
            l2hfnV(end) = Fexp;
            l2hfnVarV(end) = FexpVar;
            % Sort & End updating
            [l2hfnV,l2hfnIndV] = sort(l2hfnV);
            l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
            ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
        else
            ssolsMl2h(end,:) = Prefl; % Prefl replaces Phigh
            l2hfnV(end) = Frefl;
            l2hfnVarV(end) = FreflVar;
            % Sort & End updating
            [l2hfnV,l2hfnIndV] = sort(l2hfnV);
            l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
            ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
        end
    % Check if accept contraction or shrink
    elseif Frefl>Fsechi % When Frefl is the worst z in nex simplex
        if Frefl<=Fhigh
            Phigh = Prefl; % Prefl replaces Phigh
            Fhigh = Frefl; % Frefl replaces Fhigh
        end
        % Keep attempting contraction or shrinking
        Pcont2 = Phigh;
        Pcont = betap*Phigh + (1-betap)*Pcent;
        Pcont = checkCons(VarBds,Pcont,Pcont2);
            % Check if finish referring to current Budget
            if Bspent + r > budget(Bref)
                % Record current best soln
                A(Bref,:) = ssolsMl2h(1,:);
                Afn(Bref) = -minmax*l2hfnV(1);
                AFnVar(Bref) = l2hfnVarV(1);
                Ancalls(Bref) = Bspent;
                Bref = Bref + 1; % Now refer to next budget
                if Bref > numBudget + 1 % If exceeds the current Budget
                    return
                end   
            end        
        [Fcont, FcontVar, ~, ~, ~, ~, ~, ~] = probHandle(Pcont, r, problemRng, problemseed); % Cost r
        	Bspent = Bspent + r;
        Fcont = -minmax*Fcont;
        % Accept contraction
        if Fcont<=Fhigh
            ssolsMl2h(end,:) = Pcont; % Pcont replaces Phigh
            l2hfnV(end) = Fcont;
            l2hfnVarV(end) = FcontVar;
            % Sort & End updating
            [l2hfnV,l2hfnIndV] = sort(l2hfnV);
            l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
            ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
        % Contraction fails -> Simplex shrinks by delta, Plow retains
        else
            ssolsMl2h(end,:) = Phigh; % Replaced by Prefl
                % Check if finish referring to current Budget
                if Bspent + r*(numExtPts-1) > budget(Bref)
                    % Record current best soln
                    A(Bref,:) = ssolsMl2h(1,:);
                    Afn(Bref) = -minmax*l2hfnV(1);
                    AFnVar(Bref) = l2hfnVarV(1);
                    Ancalls(Bref) = Bspent;
                    Bref = Bref + 1; % Now refer to next budget
                    if Bref > numBudget + 1 % If exceeds the current Budget
                        return
                    end   
                end            
            for i = 2:size(ssolsMl2h,1) % From Pseclo to Phigh
                Pnew2 = Plow;
                Pnew = delta*ssolsMl2h(i,:) + (1-delta)*Plow;
                Pnew = checkCons(VarBds,Pnew,Pnew2);
                [Fnew, FnewVar, ~, ~, ~, ~, ~, ~] = probHandle(Pnew, r, problemRng, problemseed); % Cost r/loop
                Fnew = -minmax*Fnew;
                % Update ssolsM
                ssolsMl2h(i,:) = Pnew; % Pnew replaces Pi
                l2hfnV(i) = Fnew;
                l2hfnVarV(i) = FnewVar;
            end % Total cost = r*(numExtPts-1)
            	Bspent = Bspent + r*(numExtPts-1);
            % Sort & End updating
            [l2hfnV,l2hfnIndV] = sort(l2hfnV);
            l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
            ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
        end
    end
    
end


%% Helper Functions
% Helper 1: Evaluate obj fcn values z at all extreme points & Sort low2high
% If called, will spend (r*numExtPts) budget.
% Maximization problem is converted to minimization by -z.
    function [ssolsMl2h, l2hfnV, l2hfnVarV] = evalExtM(ssolsM, numExtPts,...
            r, probHandle, problemRng, problemseed, minmax)
        fnV = zeros(numExtPts,1); % To track soln
        fnVarV = zeros(numExtPts,1);
        for i1 = 1:numExtPts
            [fn, FnVar, ~, ~, ~, ~, ~, ~] = probHandle(ssolsM(i1,:), r, problemRng, problemseed);
            fnV(i1) = -minmax*fn; % Minimize fn
            fnVarV(i1) = FnVar;
        end
        [l2hfnV,l2hfnIndV1] = sort(fnV);
        l2hfnVarV = fnVarV(l2hfnIndV1,:);
        ssolsMl2h = ssolsM(l2hfnIndV1,:);
    end


% Helper 2: Check & Modify (if needed) the new point, based on VarBds.
    function modiSsolsV = checkCons(VarBds,ssolsV,ssolsV2)
        col = size(ssolsV,2);
        stepV = ssolsV - ssolsV2;
        % t>0 for the correct direction
        tmaxV = ones(2,col);
        uV = VarBds(stepV>0,2); uV = uV';
        lV = VarBds(stepV<0,1); lV = lV';
        if ~isempty(uV) %length(uV)> 0
            tmaxV(1,stepV>0) = (uV - ssolsV2(stepV>0)) ./ stepV(stepV>0);
        end
        if ~isempty(lV) %length(lV)>0
            tmaxV(2,stepV<0) = (lV - ssolsV2(stepV<0)) ./ stepV(stepV<0);
        end
        t = min(min(tmaxV));
        modiSsolsV = ssolsV2 + t*stepV;
        %rounding error, may remove
        for kc=1:col
            if abs(modiSsolsV(kc))<0.00000005
                modiSsolsV(kc) = 0;
            end
        end
    end


end

