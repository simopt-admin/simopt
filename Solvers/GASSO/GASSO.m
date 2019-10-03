%==========================================================================
%                        The GASSO Algorithm 
%==========================================================================
% DATE
%        February 2018
%
% AUTHOR
%        Sait Cakmak
%        scakmak3 AT gatech DOT edu
%        Enlu Zhou
%        enlu.zhou AT isye DOT gatech DOT edu
%
% REFERENCE
%        Enlu Zhou, Shalabh Bhatnagar (2018) 
%        Gradient-Based Adaptive Stochastic Search 
%               for Simulation Optimization Over Continuous Space. 
%        INFORMS Journal on Computing 30(1):154-167. 
%        https://doi.org/10.1287/ijoc.2017.0771
%==========================================================================
% 
% INPUT 
%        probHandle
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

%% GASSO
function [Ancalls, A, Afn, AFnVar, AFnGrad, AFnGradCov, ...
     Aconstraint, AConstraintCov, AConstraintGrad, ...
     AConstraintGradCov] = GASSO(probHandle, probstructHandle, ...
    problemRng, solverRng, numBudget)

%% unreported
AFnGrad = NaN;
AFnGradCov = NaN;
Aconstraint = NaN;
AConstraintCov = NaN;
AConstraintGrad = NaN;
AConstraintGradCov = NaN;

% Separate the two solver random number streams
solverInitialRng = solverRng{1}; % RNG for finding initial solutions
solverInternalRng = solverRng{2}; % RNG for the solver's internal randomness

%% Gather parameters
NumStartingSols = 1;

% Get problem dimension and bounds and initial solution(s)
RandStream.setGlobalStream(solverInitialRng)
[minmax, dim, ~, ~, VarBds, ~, ~, x0, budgetR, ~, ~, ~] = probstructHandle(NumStartingSols);

% x0 
%     Matrix (size = 'dim' X 'NumStartingSol') of 'NumStartingSol' 
%     initial solutions to the solver
%     For GASSO, NumStartingSol = 0 or 1 (set in Structure)

% Set parameters
N = round(50 * sqrt(dim)); % number of samples for each iteration
M = 10; % each candidate solution is simulated M times
rho = 0.1; % quantile parameter

% step size - alpha_k = alpha_0 / (k + alpha_c)^alpha_p
alpha_0 = 50; % step size numerator default = 50
alpha_c = 1500; % step size denominator constant default = 1500
alpha_p = 0.6; % step size denominator exponent default = 0.6

NumFinSoln = numBudget + 1; % Number of solutions returned by solver (+1 for initial solution)
budget = round(linspace(budgetR(1), budgetR(2), numBudget));

% Set parameters
K_v = floor(budget./(N*M)); % number of iterations - vector

% Initialize
A = zeros(NumFinSoln, dim);
Afn = zeros(NumFinSoln, 1);
AFnVar = zeros(NumFinSoln, 1);
Ancalls = zeros(NumFinSoln, 1);

% Using CRN: for each solution, start at substream 1
problemseed = 1;

% Record initial solution and stats based on M samples
% This is for reporting only: GASSO does not use these values
A(1,:) = x0;
[Afn(1), AFnVar(1), ~, ~, ~, ~, ~, ~] = probHandle(x0, M, problemRng, problemseed);
Ancalls(1) = M;

% If a budget is too small, the initial solution is returned.
skip = 2;
while K_v(1) < 1 % Need to evaluate all initial solns in first iteration
    fprintf('A budget is too small for a good quality run of GASSO. \n');
    fprintf('The initial solution will be returned. \n');
    A(skip,:) = x0;
    Afn(skip) = Afn(1);
    AFnVar(skip) = AFnVar(1);
    Ancalls(skip) = M;
    K_v = K_v(2:end);
    skip = skip+1;
end

% run the algorithm
%[samples, Hbar, xbar, hvar] = GASSO_elite(x0, NumStartingSols, VarBds, N, K_v, M, problem, problemseed, solverseed, dim, rho, minmax, alpha_0, alpha_c, alpha_p);
[samples, Hbar, xbar, hvar] = GASSO_elite(x0, NumStartingSols, VarBds, N, K_v, M, probHandle, problemRng, problemseed, solverInternalRng, dim, rho, minmax, alpha_0, alpha_c, alpha_p);

% return solutions
A(skip:end,:) = xbar(K_v,:);
Afn(skip:end) = minmax*Hbar(K_v);
Ancalls(skip:end) = samples(K_v);
AFnVar(skip:end) = hvar(K_v);

end

%% The actual algorithm
function [samples, Hbar, xbar, hvar] = GASSO_elite(x0, NumStartingSols, VarBds, N, K_v, M, probHandle, problemRng, problemseed, solverInternalRng, dim, rho, minmax, alpha_0, alpha_c, alpha_p)
% GASSO: sampling distribution is normal (for unbounded solution set) or truncated normal (for bounded solution set)
% 
% N: # of samples
% M: # of evaluations per sample
% K: # of iterations
% dim: dimension of the problem
% alpha: step size
% rho: quantile parameter
% samples: number of samples
% Hbar: optimal value estimate
% xbar: candidate solution
% hvar: variance of optimal value estimate
% x0: starting solution if one is asked for, o.w. NaN
% mu: mean of the sampling distribution
% var: variance of the sampling distribution


%% Initialization

% initialize mu_k based on starting solution or solution set bounds
mu_k = zeros(1,dim);
if NumStartingSols == 1
    mu_k = x0;
else
    for i=1:dim
        if VarBds(i,1) ~= -inf && VarBds(i,2) == inf
            mu_k(i) = VarBds(i,1) + 50;
        elseif VarBds(i,1) == -inf && VarBds(i,2) ~= inf 
            mu_k(i) = VarBds(i,2) - 50;
        elseif VarBds(i,1) ~= -inf && VarBds(i,2) ~= inf 
            mu_k(i) = (VarBds(i,1) + VarBds(i,2)) / 2;
        end 
    end
end

% set the initial variance
if all(VarBds(:,1) ~= -inf) && all(VarBds(:,2) ~= inf)
    % if bounded - gap = 6sigma
    gap = max(abs(VarBds(:,1) - VarBds(:,2)))/6;
else
    % if unbounded
    gap = sqrt(1000);
end
var_k = gap^2*ones(1,dim);

theta1_k=mu_k./var_k;
theta2_k=-0.5*ones(1,dim)./var_k;
theta_k=[theta1_k, theta2_k];

K = max(K_v);

% Set random number generator for generating random candidate solutions
RandStream.setGlobalStream(solverInternalRng)

% generate candidate solutions from (truncated) normal distribution
x = zeros(N,dim);
kk = 1;
while kk<=N
    X_k=randn(N,dim)*diag(sqrt(var_k))+ones(N,1)*mu_k;
    for i=1:N
        if all(X_k(i,:)' >= VarBds(:,1)) && all(X_k(i,:)' <= VarBds(:,2)) && kk<=N
            x(kk,:) = X_k(i,:);
            kk = kk+1;
        end % if
    end % for
end % while
X_k = x;

k=1;
samples=zeros(K,1);
Hbar = zeros(K,1);
xbar = zeros(K,dim);
hvar = zeros(K,1);
Ntotal=0;

%% iteration
while k<=K
    alpha_k = alpha_0/(k+alpha_c)^alpha_p; % set step size
   
    H = zeros(N,1); % obj function values
    H_var = zeros(N,1); % obj function variances
    for i=1:N
        % simulate each solution X_k(i) for M times - uses CRN
        [fn, H_var(i), ~, ~, ~, ~, ~, ~] = probHandle(X_k(i,:), M, problemRng, problemseed);
        result = minmax*fn;
        H(i) = result;
    end
    problemseed = problemseed + 1;
    
    if all(H == -inf)
        fprintf('The problem returns -inf for valid x values!!!');
        return
    end %if

    Ntotal = Ntotal+N*M;
    samples(k) = Ntotal; % record the budget
        
    % keep the best candidate
    [Hbar(k), I] = max(H); 
    hvar(k) = H_var(I);
    xbar(k,:) = X_k(I,:);
           
    if k>2
        if Hbar(k)<Hbar(k-1)|| isnan(Hbar(k))==1      
            Hbar(k) = Hbar(k-1);
            xbar(k,:) = xbar(k-1,:);
            hvar(k) = hvar(k-1);
        end 
    end
    
    % Shape function - take elite samples
    [G_sort, ~] = sort(H,'descend');   % sorted performances
    gm = G_sort(ceil(N*rho));
    S_theta=(H>gm);
    
    % estimate gradient and hessian
    w_k=S_theta/sum(S_theta);
    CX_k=[X_k,X_k.*X_k]; 
    grad_k=w_k'*CX_k-[mu_k, var_k + mu_k.*mu_k];  
    Hes_k=-cov(CX_k);        
    Hes_inv_k=(Hes_k + 1e-8.*eye(2*dim))\diag(ones(1,2*dim)); 
    
    % update the parameter using an SA iteration
    theta_k=theta_k-alpha_k*(Hes_inv_k*grad_k')';
    theta1_k=theta_k(1:dim);
    theta2_k=theta_k(dim+1:2*dim);
    var_k=-0.5./theta2_k;
    mu_k=theta1_k.*var_k;
    
    % project mu_k and var_k to feasible parameter space
    for i=1:dim
        if mu_k(i) < VarBds(i,1)
            mu_k(i) = VarBds(i,1);
        elseif mu_k(i) > VarBds(i,2)
            mu_k(i) = VarBds(i,2);
        end %if
    end %for
    var_k = abs(var_k);

    % Set random number generator for generating random candidate solutions
    RandStream.setGlobalStream(solverInternalRng)
    
    % generate candidate solutions from (truncated) normal distribution
    x = zeros(N,dim);
    kk = 1;
    while kk<=N
        X_k=randn(N,dim)*diag(sqrt(abs(var_k)))+ones(N,1)*mu_k;
        for i=1:N
            if all(X_k(i,:)' >= VarBds(:,1)) && all(X_k(i,:)' <= VarBds(:,2)) && kk<=N
                x(kk,:) = X_k(i,:);
                kk = kk+1;
            end % if
        end % for
    end % while
    X_k = x;

    k=k+1;
    clear H_var H S_theta w_k theta1_k theta2_k CX_k grad_k Hes_k Hes_inv_k;
end %end while

end % GASSO_elite