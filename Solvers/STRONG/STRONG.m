%==========================================================================
%                           The STRONG Algorithm
%==========================================================================
% DATE
%        Feb 2017 (Updated in Oct 2019 by David Eckman)
%
% AUTHOR
%        Xueqi Zhao and Naijia Dong
% 
% REFERENCE
%       Kuo-Hao Change, L. Jeff Hong, and Hong Wan (2013)
%       Stochastic Trust-Region Response-Surface Method (STRONG) - A New
%       Response-Surface Framework for Simulation Optimization
%       INFORMS Journal on Computing. 25(2):230-243
%       * Used finite diff instead of DOE
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

%% STRONG
function [Ancalls, A, Afn, AFnVar, AFnGrad, AFnGradCov, ...
    Aconstraint, AConstraintCov, AConstraintGrad, ...
    AConstraintGradCov] = STRONG(probHandle, probstructHandle, problemRng, ...
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

% Set default values.
r = 30; % Number of replications taken at each solution
sensitivity = 10^(-7); % shrinking scale for VarBds

% Get details of the problem and random initial solution
RandStream.setGlobalStream(solverInitialRng);
[minmax, dim, ~, ~, VarBds, ~, ~, solution, budgetR, ~, ~, ~] = probstructHandle(1);

% Shrink VarBds to prevent floating errors
VarBds(:,1) = VarBds(:,1) + sensitivity; 
VarBds(:,2) = VarBds(:,2) - sensitivity;

NumFinSoln = numBudget + 1; % Number of solutions returned by solver (+1 for initial solution)
budget = [0, round(linspace(budgetR(1), budgetR(2), numBudget))];

% Initialize
A = zeros(NumFinSoln, dim);
Afn = zeros(NumFinSoln, 1);
AFnVar = zeros(NumFinSoln, 1);
Ancalls = zeros(NumFinSoln, 1);

Bspent = 0; % Total budget spent
Bref = 1; % The budget currently referred to, = 1, ..., numBudget
iterCount = 1;

% Using CRN: for each solution, start at substream 1
problemseed = 1;
% If not using CRN: problemseed needs to be updated throughout the code

% Set other parameters
delta_threshold = 1.2; 
delta_T = 2;      %the size of trust region       
eta_0 = 0.01;     %the constant of accepting 
eta_1 = 0.3;  
gamma1 = 0.9;     %the constant of shrinking the trust regionthe new solution    
gamma2 = 1.11;    %the constant of expanding the trust region

[Q_bar_old, fn0varcurrent, ~, ~, ~, ~, ~, ~] = probHandle(solution, r, problemRng, problemseed);
Q_bar_old = -minmax*Q_bar_old;
Bspent = Bspent + r;
x0best = solution;
fn0best = Q_bar_old;
fn0varbest = fn0varcurrent;

 %..........................Main Framework..............................  
 while Bspent <= budgetR(2)
     
     % check variable bounds
     forward = (solution == VarBds(:,1)');
     backward = (solution == VarBds(:,2)');
     BdsCheck = forward - backward; 
     % BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff

     if delta_T > delta_threshold    %stage I
         
         % check budget
         NumOfEval = 2*dim - sum(BdsCheck ~= 0); % num of fn evaluations to compute grad
         while budget(Bref) - Bspent < (NumOfEval + 1)*r % budget required for one update
             Ancalls(Bref) = Bspent;
             A(Bref,:) = x0best;
             Afn(Bref) = -minmax*fn0best;
             AFnVar(Bref) = fn0varbest;
             Bref = Bref + 1;
             if Bref > numBudget
                 
                 return
             end
         end
         
         %step1 Build the linear model
         [Grad, Hessian] = FiniteDiff(solution, Q_bar_old, BdsCheck, 1, r, probHandle, problemRng, problemseed, dim, delta_T, VarBds, minmax);
         Bspent = Bspent + NumOfEval*r;
         
         %step2 Solve the subproblem
         [new_solution]=Cauchy_point(Grad, Hessian, solution, VarBds, delta_T); %generate the new solution
         
         %step3 Compute the ratio
         [Q_bar_new, newVar, ~, ~, ~, ~, ~, ~] = probHandle(new_solution, r, problemRng, problemseed);
         Q_bar_new = -minmax*Q_bar_new;
         Bspent = Bspent + r;
         r_old = Q_bar_old;
         r_new = Q_bar_old + (new_solution - solution)*Grad + (1/2)*(new_solution - solution)*Hessian*(new_solution - solution)';
         rho = (Q_bar_old - Q_bar_new)/(r_old - r_new);
         
         %step4 Update the trust region size and determine to accept or reject the solution
         if rho < eta_0 || (Q_bar_old - Q_bar_new) <= 0 || (r_old - r_new) <= 0
             delta_T = gamma1*delta_T;
             
         elseif (eta_0 <= rho) && (rho < eta_1)
             solution = new_solution; %accept the solution and remains the size of  trust ra=[]egion
             Q_bar_old = Q_bar_new;
             
             % update best soln so far
             if Q_bar_new < fn0best
                 x0best = solution;
                 fn0best = Q_bar_new;
                 fn0varbest = newVar;
             end
         else
             delta_T = gamma2*delta_T;
             solution = new_solution;  %accept the solution and expand the size of trust reigon
             Q_bar_old = Q_bar_new;
             
             % update best soln so far
             if Q_bar_new < fn0best
                 x0best = solution;
                 fn0best = Q_bar_new;
                 fn0varbest = newVar;
             end
         end
         r = ceil(1.01*r);
     else    %stage II
         %When trust region size is very small, use the quadratic design
         
         % check budget
         num = sum(BdsCheck ~= 0); % num of on-boundary variables
         if num <= 1
             NumOfEval = dim^2; % num of fn evaluations to compute grad&Hessian 
         else
            NumOfEval = dim^2 + dim - nchoosek(num,2); 
         end
         while budget(Bref) - Bspent < (NumOfEval+1)*r % budget required for one update
             Ancalls(Bref) = Bspent;
             A(Bref,:) = x0best;
             Afn(Bref) = -minmax*fn0best;
             AFnVar(Bref) = fn0varbest;
             Bref = Bref+1;
             if Bref > numBudget
                 
                 return
             end
         end
         
         %step1 Build the quadratic model
         [Grad, Hessian] = FiniteDiff(solution, Q_bar_old, BdsCheck, 2, r, probHandle, problemRng, problemseed, dim, delta_T, VarBds, minmax);
         Bspent = Bspent + NumOfEval*r;
         
         %step2 Solve the subproblem
         [new_solution] = Cauchy_point(Grad, Hessian, solution, VarBds, delta_T); %generate the new solution
         
         %step3 Compute the ratio         
         [Q_bar_new, newVar, ~, ~, ~, ~, ~, ~] = probHandle(new_solution, r, problemRng, problemseed);
         Q_bar_new = -minmax*Q_bar_new;
         Bspent = Bspent + r;
         r_old = Q_bar_old;
         r_new = Q_bar_old + (new_solution - solution)*Grad + (1/2)*(new_solution - solution)*Hessian*(new_solution - solution)';
         rho = (Q_bar_old - Q_bar_new)/(r_old - r_new);
         
         %step4 Update the trust region size and determine to accept or reject the solution
         if rho < eta_0 || (Q_bar_old - Q_bar_new) <= 0 || (r_old - r_new) <= 0

             [n_solution, Q_bar_old, soln_var] = inner_loop(solution, r_old, BdsCheck, NumOfEval, r,  probHandle, problemRng, problemseed);
             if Bref > numBudget
                 return
             end
             
             if sum(n_solution ~= solution) > 0 && (Q_bar_old < fn0best)
                 x0best = n_solution;
                 fn0best = Q_bar_old;
                 fn0varbest = soln_var;
             end
             solution = n_solution;
        
         elseif (eta_0 <= rho) && (rho < eta_1)

             solution = new_solution;         %accept the solution and remains the size of  trust region
             Q_bar_old = Q_bar_new;
             
             % update best soln so far
             if Q_bar_new < fn0best
                 x0best = solution;
                 fn0best = Q_bar_new;
                 fn0varbest = newVar;
             end
         else
             delta_T = gamma2*delta_T;
             solution = new_solution;            %accept the solution and expand the size of trust reigon
             Q_bar_old = Q_bar_new;

             % update best soln so far
             if Q_bar_new < fn0best
                 x0best = solution;
                 fn0best = Q_bar_new;
                 fn0varbest = newVar;
             end
             
         end
         r = ceil(1.01*r);
     end 
     iterCount = iterCount + 1;
 end
 
 %% Helper Function FiniteDiff
    function [Grad, Hessian] = FiniteDiff(solution, fn, BdsCheck, stage, runlen, probHandle, problemRng, problemseed, dim, delta_T, VarBds, minmax)

        FnPlusMinus = zeros(dim,3); % store values for each dimension:
        % col1: f(x+h,y)
        % col2: f(x-h,y)
        % col3: stepsize h
        Grad = zeros(dim,1);
        Hessian = zeros(dim,dim);
        
        for i = 1:dim
            % initialization
            x1 = solution;
            x2 = solution;
            steph1 = delta_T; %forward stepsize
            steph2 = delta_T; %backward stepsize
            
            % check VarBds
            if x1(i) + steph1 > VarBds(i,2)
                steph1 = abs(VarBds(i,2) - x1(i)); % can remove abs()
            end
            if x2(i) - steph2 < VarBds(i,1)
                steph2 = abs(x2(i) - VarBds(i,1)); % can remove abs()
            end
            
            % decide stepsize
            if BdsCheck(i) == 0    % central diff
                FnPlusMinus(i,3) = min(steph1, steph2);
                x1(i) = x1(i) + FnPlusMinus(i,3);
                x2(i) = x2(i) - FnPlusMinus(i,3);
            elseif BdsCheck(i) == 1    % forward diff
                FnPlusMinus(i,3) = steph1;
                x1(i) = x1(i) + FnPlusMinus(i,3);
            else    % backward diff
                FnPlusMinus(i,3) = steph2;
                x2(i) = x2(i) - FnPlusMinus(i,3);
            end
            
            if BdsCheck(i) ~= -1
                [fn1, ~, ~, ~, ~, ~, ~, ~] =probHandle(x1, runlen, problemRng, problemseed);
                fn1 = -minmax*fn1;
                FnPlusMinus(i,1) = fn1; % first column is f(x+h,y)
            end
            
            if BdsCheck(i) ~= 1
                [fn2, ~, ~, ~, ~, ~, ~, ~] =probHandle(x2,runlen, problemRng, problemseed);
                fn2 = -minmax*fn2;
                FnPlusMinus(i,2) = fn2; % second column is f(x-h,y)
            end
            
            if BdsCheck(i) == 0
                Grad(i) = (fn1 - fn2)/(2*FnPlusMinus(i,3));
            elseif BdsCheck(i) == 1
                Grad(i) = (fn1 - fn)/FnPlusMinus(i,3);
            elseif BdsCheck(i) == -1
                Grad(i) = (fn - fn2)/FnPlusMinus(i,3);
            end
        end
        
        if stage == 2
            % diagonal in Hessian
            for i = 1:dim
                if BdsCheck(i) == 0
                    Hessian(i,i) = (FnPlusMinus(i,1) - 2*fn + FnPlusMinus(i,2))/(FnPlusMinus(i,3)^2);
                elseif BdsCheck(i) == 1
                    x3 = solution;
                    x3(i) = x3(i) + FnPlusMinus(i,3)/2;
                    %check budget
                    [fn3, ~, ~, ~, ~, ~, ~, ~] = probHandle(x3, runlen, problemRng, problemseed);
                    fn3 = -minmax*fn3;
                    Hessian(i,i) = 4*(FnPlusMinus(i,1) - 2*fn3 + fn)/(FnPlusMinus(i,3)^2);
                elseif BdsCheck(i) == -1
                    x4 = solution;
                    x4(i) = x4(i) - FnPlusMinus(i,3)/2;
                    %check budget
                    [fn4, ~, ~, ~, ~, ~, ~, ~] = probHandle(x4, runlen, problemRng, problemseed);
                    fn4 = -minmax*fn4;
                    Hessian(i,i) = 4*(fn-2*fn4 + FnPlusMinus(i,2))/(FnPlusMinus(i,3)^2);
                end
                
                % upper triangle in Hessian
                for j = i+1:dim
                    if BdsCheck(i)^2 + BdsCheck(j)^2 == 0 % neither x nor y on boundary
                        % f(x+h,y+k)
                        x5 = solution;
                        x5(i) = x5(i) + FnPlusMinus(i,3);
                        x5(j) = x5(j) + FnPlusMinus(j,3);
                        %check budget
                        [fn5, ~, ~, ~, ~, ~, ~, ~] = probHandle(x5, runlen, problemRng, problemseed);
                        fn5 = -minmax*fn5;
                        % f(x-h,y-k)
                        x6 = solution;
                        x6(i) = x6(i) - FnPlusMinus(i,3);
                        x6(j) = x6(j) - FnPlusMinus(j,3);
                        %check budget
                        [fn6, ~, ~, ~, ~, ~, ~, ~] = probHandle(x6, runlen, problemRng, problemseed);
                        fn6 = -minmax*fn6;
                        % compute second order gradient
                        Hessian(i,j) = (fn5 - FnPlusMinus(i,1) - FnPlusMinus(j,1) + 2*fn - FnPlusMinus(i,2) - FnPlusMinus(j,2) + fn6)/(2*FnPlusMinus(i,3)*FnPlusMinus(j,3));
                        Hessian(j,i) = Hessian(i,j);
                    elseif BdsCheck(j) == 0 % x on boundary, y not
                        % f(x+/-h,y+k)
                        x5 = solution;
                        x5(i) = x5(i) + BdsCheck(i)*FnPlusMinus(i,3);
                        x5(j) = x5(j) + FnPlusMinus(j,3);
                        %check budget
                        [fn5, ~, ~, ~, ~, ~, ~, ~] = probHandle(x5, runlen, problemRng, problemseed);
                        fn5 = -minmax*fn5;
                        % f(x+/-h,y-k)
                        x6 = solution;
                        x6(i) = x6(i) + BdsCheck(i)*FnPlusMinus(i,3);
                        x6(j) = x6(j) - FnPlusMinus(j,3);
                        %check budget
                        [fn6, ~, ~, ~, ~, ~, ~, ~] = probHandle(x6, runlen, problemRng, problemseed);
                        fn6 = -minmax*fn6;
                        % compute second order gradient
                        Hessian(i,j) = (fn5 - FnPlusMinus(j,1) - fn6 + FnPlusMinus(j,2))/(2*FnPlusMinus(i,3)*FnPlusMinus(j,3)*BdsCheck(i));
                        Hessian(j,i) = Hessian(i,j);
                    elseif BdsCheck(i) == 0 % y on boundary, x not
                        % f(x+h,y+/-k)
                        x5 = solution;
                        x5(i) = x5(i) + FnPlusMinus(i,3);
                        x5(j) = x5(j) + BdsCheck(j)*FnPlusMinus(j,3);
                        %check budget
                        [fn5, ~, ~, ~, ~, ~, ~, ~] = probHandle(x5, runlen, problemRng, problemseed);
                        fn5 = -minmax*fn5;
                        % f(x-h,y+/-k)
                        x6 = solution;
                        x6(i) = x6(i) - FnPlusMinus(i,3);
                        x6(j) = x6(j) + BdsCheck(j)*FnPlusMinus(j,3);
                        %check budget
                        [fn6, ~, ~, ~, ~, ~, ~, ~] = probHandle(x6, runlen, problemRng, problemseed);
                        fn6 = -minmax*fn6;
                        % compute second order gradient
                        Hessian(i,j) = (fn5 - FnPlusMinus(i,1) - fn6 + FnPlusMinus(i,2))/(2*FnPlusMinus(i,3)*FnPlusMinus(j,3)*BdsCheck(j));
                        Hessian(j,i) = Hessian(i,j);
                    elseif BdsCheck(i) == 1
                        if BdsCheck(j) == 1
                            % f(x+h,y+k)
                            x5 = solution;
                            x5(i) = x5(i) + FnPlusMinus(i,3);
                            x5(j) = x5(j) + FnPlusMinus(j,3);
                            %check budget
                            [fn5, ~, ~, ~, ~, ~, ~, ~] = probHandle(x5,runlen, problemRng, problemseed);
                            fn5 = -minmax*fn5;
                            % compute second order gradient
                            Hessian(i,j) = (fn5 - FnPlusMinus(i,1) - FnPlusMinus(j,1) + fn)/(FnPlusMinus(i,3)*FnPlusMinus(j,3));
                            Hessian(j,i) = Hessian(i,j);
                        else
                            % f(x+h,y+k)
                            x5 = solution;
                            x5(i) = x5(i) + FnPlusMinus(i,3);
                            x5(j) = x5(j) - FnPlusMinus(j,3);
                            %check budget
                            [fn5, ~, ~, ~, ~, ~, ~, ~] = probHandle(x5, runlen, problemRng, problemseed);
                            fn5 = -minmax*fn5;
                            % compute second order gradient
                            Hessian(i,j) = (FnPlusMinus(i,1) - fn5 - fn + FnPlusMinus(j,2))/(FnPlusMinus(i,3)*FnPlusMinus(j,3));
                            Hessian(j,i) = Hessian(i,j);
                        end
                    elseif BdsCheck(i) == -1
                        if BdsCheck(j) == 1
                            % f(x+h,y+k)
                            x5 = solution;
                            x5(i) = x5(i) - FnPlusMinus(i,3);
                            x5(j) = x5(j) + FnPlusMinus(j,3);
                            %check budget
                            [fn5, ~, ~, ~, ~, ~, ~, ~] = probHandle(x5, runlen, problemRng, problemseed);
                            fn5 = -minmax*fn5;
                            % compute second order gradient
                            Hessian(i,j) = (FnPlusMinus(j,1) - fn - fn5 + FnPlusMinus(i,2))/(FnPlusMinus(i,3)*FnPlusMinus(j,3));
                            Hessian(j,i) = Hessian(i,j);
                        else
                            % f(x+h,y+k)
                            x5 = solution;
                            x5(i) = x5(i) - FnPlusMinus(i,3);
                            x5(j) = x5(j) - FnPlusMinus(j,3);
                            %check budget
                            [fn5, ~, ~, ~, ~, ~, ~, ~] = probHandle(x5, runlen, problemRng, problemseed);
                            fn5 = -minmax*fn5;
                            % compute second order gradient
                            Hessian(i,j) = (fn - FnPlusMinus(j,2) - FnPlusMinus(i,2) + fn5)/(FnPlusMinus(i,3)*FnPlusMinus(j,3));
                            Hessian(j,i) = Hessian(i,j);
                        end
                    end
                end
            end
        end
    end
%% Helper Function CauchyPoint
    function [Cauchy_point] = Cauchy_point(G, B, solution, VarBds, delta_T) %%Finding the Cauchy Point
        Q = G'*B*G;
        b = (-1)*delta_T/norm(G)*G';
        if Q <= 0
            tau = 1;
        else
            if (norm(G))^3/(delta_T*Q)<1
                tau = (norm(G))^3/(delta_T*Q);
            else
                tau=1;
            end
        end
        new_point = solution+tau*b;
        Cauchy_point = checkCons(new_point ,solution, VarBds);
    end

%% Helper Function CheckCons
    function modiSsolsV = checkCons(ssolsV, ssolsV2, VarBds)
        col = size(ssolsV, 2);
        stepV = ssolsV - ssolsV2;
        %t = 1; % t>0 for the correct direction
        tmaxV = ones(2,col);
        uV = VarBds(stepV>0,2); uV = uV';
        lV = VarBds(stepV<0,1); lV = lV';
        if isempty(uV) == 0 % length(uV) > 0
            tmaxV(1, stepV>0) = (uV - ssolsV2(stepV>0)) ./ stepV(stepV>0);
        end
        if isempty(lV) == 0 % length(lV) > 0
            tmaxV(2,stepV<0) = (lV - ssolsV2(stepV<0)) ./ stepV(stepV<0);
        end
        t2 = min(min(tmaxV));
        modiSsolsV = ssolsV2 + t2*stepV;
        %rounding error
        for kc=1:col
            if modiSsolsV(kc) < 0 && modiSsolsV(kc) > -0.00000005
                modiSsolsV(kc) = 0;
            end
        end
    end

%% Helper Function InnerLoop
    function [result_solution, value, var] = inner_loop(solution, rr_old, BdsCheck, NumOfEval, runlength, probHandle, problemRng, problemseed)        
        Q_b_old = rr_old;
        sub_counter = 1;
        result_solution = solution;
        value = 0;
        var = 0;
        while sum(result_solution ~= solution) == 0  %was while result_solution==solution
            numEval = NumOfEval*(sub_counter + 1)*runlength + runlength + ceil(sub_counter^1.01) + ceil(sub_counter^1.01) - ceil((sub_counter - 1)^1.01);
            while budget(Bref) - Bspent < numEval % budget required for one update
                Ancalls(Bref) = Bspent;
                A(Bref,:) = x0best;
                Afn(Bref) = -minmax*fn0best;
                AFnVar(Bref) = fn0varbest;
                Bref = Bref+1;
                if Bref>numBudget
                    return
                end
            end
            [G, H] = FiniteDiff(solution, rr_old, BdsCheck, 2, (sub_counter + 1)*runlength, probHandle, problemRng, problemseed, dim, delta_T, VarBds, minmax);
            Bspent = Bspent + NumOfEval*(sub_counter + 1)*runlength;
            %step2 determine the new inner solution based on the accumulated design matrix X
            [try_solution] = Cauchy_point(G, H, solution, VarBds, delta_T);
            [Q_b_new, var, ~, ~, ~, ~, ~, ~] = probHandle(try_solution, runlength + ceil(sub_counter^1.01), problemRng, problemseed);
            Q_b_new = -minmax*Q_b_new;
            Bspent = Bspent + runlength + ceil(sub_counter^1.01); %%%
            [dummy, ~, ~, ~, ~, ~, ~, ~] = probHandle(solution, ceil(sub_counter^1.01) - ceil((sub_counter - 1)^1.01), problemRng, problemseed); %%%
            dummy = -minmax*dummy;
            Bspent = Bspent + ceil(sub_counter^1.01) - ceil((sub_counter - 1)^1.01); %%%
            Q_b_old = (Q_b_old*(runlength + ceil((sub_counter - 1)^1.01)) + (ceil(sub_counter^1.01) - ceil((sub_counter - 1)^1.01))*dummy)/(runlength + ceil(sub_counter^1.01)); %update the Q_bar_old
            rr_new = Q_b_old + (try_solution - solution)*G + (1/2)*(try_solution - solution)*H*(try_solution - solution)';          
            rr_old = Q_b_old;
            rrho = (Q_b_old - Q_b_new)/(rr_old - rr_new);
            if rrho < eta_0 || (Q_b_old - Q_b_new) <= 0 || (rr_old - rr_new) <= 0
                delta_T = gamma1*delta_T;
                result_solution = solution;
                
            elseif (eta_0 <= rrho) && (rrho < eta_1)

                result_solution = try_solution;         %accept the solution and remains the size of  trust ra=[]egion
                rr_old = Q_b_new;
                value = Q_b_new;
            else
                delta_T = gamma2*delta_T;
                result_solution = try_solution;         %accept the solution and expand the size of trust reigon
                rr_old = Q_b_new;
                value = Q_b_new;
            end
            sub_counter = sub_counter + 1;

        end
    end

end
