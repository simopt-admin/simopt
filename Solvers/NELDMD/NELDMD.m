%==========================================================================
%                       The Nelder-Mead Algorithm
%==========================================================================
% DATE
%        Feb 2016 (Updated in Dec 2019 by David Eckman)
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

%% Nelder-Mead
function [Ancalls, A, AFnMean, AFnVar, AFnGrad, AFnGradCov, ...
    AConstraint, AConstraintCov, AConstraintGrad, ...
    AConstraintGradCov] = NELDMD(probHandle, probstructHandle, problemRng, ...
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
% Nelder-Mead does not use any internal randomness

% Set default values
r = 30; % Number of replications taken at each solution
alpha = 1;
gammap = 2;
betap = 0.5;
delta = 0.5;
sensitivity = 10^(-7); % shrinking scale for VarBds

% Get details of the problem
[minmax, dim, ~, ~, VarBds, ~, ~, ~, budget, ~, ~, ~] = probstructHandle(0);

% Shrink VarBds to prevent floating errors
VarBds(:,1) = VarBds(:,1) + sensitivity; 
VarBds(:,2) = VarBds(:,2) - sensitivity;

% Check for sufficiently large budget
numExtPts = dim + 1;
if budget < r*numExtPts % Need to evaluate all initial solns in ssolsM
    fprintf('A budget is too small for a good quality run of Nelder-Mead.');
    return
end

% Determine maximum number of solutions that can be sampled within max budget
MaxNumSoln = floor(budget/r); 

% Generate initial simplex from dim+1 random points
RandStream.setGlobalStream(solverInitialRng);
[~, ~, ~, ~, ~, ~, ~, ssolsM, ~, ~, ~] = probstructHandle(numExtPts);

% Initialize larger than necessary (extra point for end of budget)
Ancalls = zeros(MaxNumSoln, 1);
A = zeros(MaxNumSoln + 1, dim);
AFnMean = zeros(MaxNumSoln, 1);
AFnVar = zeros(MaxNumSoln, 1);

% Using CRN: for each solution, start at substream 1
problemseed = 1;
% If not using CRN: problemseed needs to be updated throughout the code

% Track overall budget spent
Bspent = 0;

%% Start Solving

% Evaluate solutions in initial structure 
fnV = zeros(numExtPts,1); % To track soln
fnVarV = zeros(numExtPts,1);
for i1 = 1:numExtPts
    [fn, FnVar, ~, ~, ~, ~, ~, ~] = probHandle(ssolsM(i1,:), r, problemRng, problemseed);
    Bspent = Bspent + r;
    fnV(i1) = -minmax*fn; % Minimize fn
    fnVarV(i1) = FnVar;
end

% Record initial solution data
Ancalls(1) = 0;
A(1,:) = ssolsM(1,:);
AFnMean(1) = -minmax*fnV(1); % flip sign back
AFnVar(1) = fnVarV(1);

% Sort solutions by obj function estimate
[l2hfnV,l2hfnIndV1] = sort(fnV);
l2hfnVarV = fnVarV(l2hfnIndV1,:);
ssolsMl2h = ssolsM(l2hfnIndV1,:);

% Record only when recommended solution changes
record_index = 2;

% Reflect Worst & Update ssolsMl2h
% Maximization problem is converted to minimization by -z.
while Bspent <= budget
    % Reflect worst point
    Phigh = ssolsMl2h(end,:); % Current worst pt
    Pcent = mean(ssolsMl2h(1:end-1,:)); % Centroid for other pts
    Prefl2 = Phigh; % Save the original point
    Prefl = (1+alpha)*Pcent - alpha*Phigh; % Reflection
    Prefl = checkCons(VarBds,Prefl,Prefl2); % Check if Prefl respects VarBds (if not, change it)        
    
    % Evaluate reflected point
    [Frefl, FreflVar, ~, ~, ~, ~, ~, ~] = probHandle(Prefl, r, problemRng, problemseed); % Cost r
    Bspent = Bspent + r;
    Frefl = -minmax*Frefl;
    
    % Track best, worst, and second worst points
    Plow = ssolsMl2h(1,:); % Current best pt
    Flow = l2hfnV(1);
    Fsechi = l2hfnV(end-1); % Current 2nd worst z
    Fhigh = l2hfnV(end); % Worst z from unreflected structure
    
    % Check if accept reflection
    if Flow <= Frefl && Frefl <= Fsechi
        
        ssolsMl2h(end,:) = Prefl; % Prefl replaces Phigh
        l2hfnV(end) = Frefl;
        l2hfnVarV(end) = FreflVar;
        
        % Sort & end updating
        [l2hfnV,l2hfnIndV] = sort(l2hfnV);
        l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
        ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
        
        % Best solution remains the same, so no reporting
        
    % Check if accept expansion (of reflection in the same direction)
    elseif Frefl < Flow
        
        Pexp2 = Prefl;
        Pexp = gammap*Prefl + (1-gammap)*Pcent;
        Pexp = checkCons(VarBds,Pexp,Pexp2);
        
        % Evaluate expansion point    
        [Fexp, FexpVar, ~, ~, ~, ~, ~, ~] = probHandle(Pexp, r, problemRng, problemseed); % Cost r
        Bspent = Bspent + r;        
        Fexp = -minmax*Fexp;
        
        % Check if expansion point is an improvement relative to simplex
        if Fexp < Flow
            
            % Pexp replaces Phigh
            ssolsMl2h(end,:) = Pexp; 
            l2hfnV(end) = Fexp;
            l2hfnVarV(end) = FexpVar;
            
            % Sort & end updating
            [l2hfnV,l2hfnIndV] = sort(l2hfnV);
            l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
            ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
            
            % Record data from expansion point (new best)
            if Bspent <= budget
                Ancalls(record_index) = Bspent;
                A(record_index,:) = Pexp;
                AFnMean(record_index) = -minmax*Fexp; % flip sign back
                AFnVar(record_index) = FexpVar;
                record_index = record_index + 1;
            end
            
        else
            
            % Prefl replaces Phigh
            ssolsMl2h(end,:) = Prefl; 
            l2hfnV(end) = Frefl;
            l2hfnVarV(end) = FreflVar;
            
            % Sort & end updating
            [l2hfnV,l2hfnIndV] = sort(l2hfnV);
            l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
            ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
            
            % Record data from reflected point (new best)
            if Bspent <= budget
                Ancalls(record_index) = Bspent;
                A(record_index,:) = Prefl;
                AFnMean(record_index) = -minmax*Frefl; % flip sign back
                AFnVar(record_index) = FreflVar;
                record_index = record_index + 1;
            end
            
        end
        
    % Check if accept contraction or shrink
    elseif Frefl > Fsechi % When Frefl is the worst z in nex simplex
        
        if Frefl <= Fhigh
            Phigh = Prefl; % Prefl replaces Phigh
            Fhigh = Frefl; % Frefl replaces Fhigh
        end
        
        % Attempt contraction or shrinking
        Pcont2 = Phigh;
        Pcont = betap*Phigh + (1-betap)*Pcent;
        Pcont = checkCons(VarBds,Pcont,Pcont2);
        
        % Evaluate contraction point
        [Fcont, FcontVar, ~, ~, ~, ~, ~, ~] = probHandle(Pcont, r, problemRng, problemseed);
        Bspent = Bspent + r;
        Fcont = -minmax*Fcont;
        
        % Accept contraction
        if Fcont <= Fhigh
            
            % Pcont replaces Phigh
            ssolsMl2h(end,:) = Pcont;
            l2hfnV(end) = Fcont;
            l2hfnVarV(end) = FcontVar;
            
            % Sort & end updating
            [l2hfnV,l2hfnIndV] = sort(l2hfnV);
            l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
            ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
            
            % Check if contraction point is new best
            if Fcont <= Flow
                % Record data from contraction point (new best)
                if Bspent <= budget
                    Ancalls(record_index) = Bspent;
                    A(record_index,:) = Pcont;
                    AFnMean(record_index) = -minmax*Fcont; % flip sign back
                    AFnVar(record_index) = FcontVar;
                    record_index = record_index + 1;
                end
            end
                
        else % Contraction fails -> Simplex shrinks by delta with Plow fixed
            
            ssolsMl2h(end,:) = Phigh; % Replaced by Prefl              
            
            % Check for new best
            new_best = 0;
            
            for i = 2:size(ssolsMl2h,1) % From Pseclo to Phigh
                Pnew2 = Plow;
                Pnew = delta*ssolsMl2h(i,:) + (1-delta)*Plow;
                Pnew = checkCons(VarBds,Pnew,Pnew2);
                [Fnew, FnewVar, ~, ~, ~, ~, ~, ~] = probHandle(Pnew, r, problemRng, problemseed); % Cost r/loop
                Bspent = Bspent + r;
                Fnew = -minmax*Fnew;
                
                % Check for new best
                if Fnew <= Flow
                    new_best = 1;
                end
                
                % Update ssolsM
                ssolsMl2h(i,:) = Pnew; % Pnew replaces Pi
                l2hfnV(i) = Fnew;
                l2hfnVarV(i) = FnewVar;
            end
                
            % Sort & end updating
            [l2hfnV,l2hfnIndV] = sort(l2hfnV);
            l2hfnVarV = l2hfnVarV(l2hfnIndV,:);
            ssolsMl2h = ssolsMl2h(l2hfnIndV,:);
            
            % Record data if there is a new best solution in the contraction
            if new_best == 1 && Bspent <= budget
                Ancalls(record_index) = Bspent;
                A(record_index,:) = ssolsMl2h(1,:);
                AFnMean(record_index) = -minmax*l2hfnV(1); % flip sign back
                AFnVar(record_index) = l2hfnVarV(1);
                record_index = record_index + 1;
            end
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

% Helper: Check & Modify (if needed) the new point, based on VarBds.
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

