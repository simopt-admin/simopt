%==========================================================================
% ASGD (Adaptive Stratified - Stochastic Gradient Descent)
%==========================================================================
% DATE
%        Spring 2020
%
% AUTHOR
%        Pranav Jain, Sara Shashaani
%
% REFERENCE		
%        Bollapragada, R., Byrd, R., & Nocedal, J. (2018). 
%        Adaptive sampling strategies for stochastic optimization. 
%        SIAM Journal on Optimization, 28(4), 3312-3343.
%
% STOPPING TIME VARIANT OF THE ADAPTIVE SAMPLE SIZE
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
%        Asoln
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



%% SASGD
function [Ancalls, Asoln, Afn, AFnVar, AFnGrad, AFnGradCov, ...
    Aconstraint, AConstraintCov, AConstraintGrad, AConstraintGradCov] ...
    = ASGD(probHandle, probstructHandle, problemRng, solverRng)

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

RandStream.setGlobalStream(solverInitialRng);
[minmax, ~, ~, ~, ~, ~, ~, solution, budget, ~, ~, ~] = probstructHandle(1);


% Set parameters' default values
N0 = 15; % Initial number of replications to be taken at each solution
% N0 can also be the total sample size for data-driven problems
K = 1; % Number of strata for stratified sampling
n0 = 1;  
count = 1;  
wi = 1; % Weight of each stratum for stratified sampling
bin = -1; % size of each stratum for stratified sampling
p = 1; % Probability of picking a point from each stratum for stratified sampling
k = 1:K; 
N = zeros(1,K);
calls = [];
theta = [];
budgetmax = budget;
kappa = 10; % constant for adaptive sampling
calls(1) = 0;
mean_Fj = [];
var_Fj = [];
theta(1,:) = solution; % Initial solution
Lk = 1000; % Lipschitz constant of the gradient for backtracking algorithm
N(k) = floor(wi(k).*(N0 - k*n0) + n0); % Size of each stratum
N(N==0 | N==1) = 2; % Minimum size of each stratum = 2
h = 1e-3; % Step size to eavluate gradient using finite difference method
del_Fsk = zeros(1, K); % Gradient of each stratum
eta = 1.05;
%% Start solving

while 1
    del_Fs = 0; % Gradient at each iteration 
    [Fj, calls] = OracleCalls(probHandle, theta(count,:), 1, problemRng, ...
        N, calls, count, [], [], K, minmax, 1);
    
    del_Fj = {};
    [Fjhp, calls]= OracleCalls(probHandle, theta(count,:)+h, 1, ...
        problemRng, N, calls, count, [], [], K, minmax, 1);
    
    [Fjhn, calls]= OracleCalls(probHandle, theta(count,:)-h, 1, ...
        problemRng, N, calls, count, [], [], K, minmax, 1);
    
    for i=1:K
        % Evaluate gradient using central difference
        del_Fj{i} = (Fjhp{i} - Fjhn{i})/(2*h); 
        del_Fsk(i) = 1/N(i)*sum(del_Fj{i});
        del_Fs = del_Fs + 1/sum(N)*sum(del_Fj{i});
    end
    
    % Check for optimal sample size
    [Na, Fj, del_Fj, del_Fsk, del_Fs, calls] = AdaptiveSampling (probHandle, ...
        theta(count,:), problemRng, N, calls, count, Fj, del_Fj, del_Fsk, ...
        del_Fs, p, h, kappa, Fjhp, Fjhn, K, bin, minmax);
    
    di = -del_Fs; % Descent direction 
    
    % Find optimal sample size using backtracking algorithm
    [alphai, calls] = StepSize(theta(count,:), Fj, del_Fj, del_Fs, Na, Lk,...
        problemRng, count, probHandle, calls, budgetmax, K, minmax);
    
    Lk = 1/alphai;
    Fj_mat = ConvertCelltoMatrix(Fj,Na, K); 
    mean_Fj(count) = mean(Fj_mat);
    var_Fj(count) = var(Fj_mat);
    count = count + 1;
    if max(calls) >= budgetmax % Check the budeget
        break
    end
    theta(count,:) = theta(count - 1,:) + alphai*di; % update theta
    calls(count) = calls(count - 1);
    N = Na; % Update sample size for next iteration
end


Ancalls = [0; calls'];
Asoln = [solution; theta];
Afn = -minmax*[0; mean_Fj']; % flip the sign
AFnVar = [0; var_Fj'];
end

function [alphak, calls] = StepSize(theta, Fj, del_Fj, del_Fs, N, Lk,...
    problemRng, seed, probHandle, calls, budgetmax, K, minmax)

% Backtracking algorithm to determin optimal step size
del_Fj = ConvertCelltoMatrix (del_Fj, N, K); % Convert cell array to matrix
Fj = ConvertCelltoMatrix (Fj, N, K); % Convert cell array to matrix
a_k = (var(del_Fj)/(sum(N)*(norm(del_Fs))^2)) + 1;
zeta_k = max(1,2/a_k);
Li = Lk/zeta_k;

[F_new, calls] = OracleCalls(probHandle, theta-1/Li*del_Fs, 1, problemRng, ...
    N, calls, seed, [], [], K, minmax, 2);

while mean(F_new) > (mean(Fj) - 1/(2*Li)*norm(del_Fs)^2)
    Li = eta*Lk;
    [F_new, calls] = OracleCalls(probHandle, theta-1/Li*del_Fs, 1, ...
        problemRng, N, calls, seed, [], [], K, minmax, 2);   
    if calls(seed) >= budgetmax
        break
    end
    Lk = Li;
end
alphak = 1/Li;
end

function [f, calls] = OracleCalls(probHandle, theta, runlength, problemRng, ...
    N, calls, count, Na, Fj, K, minmax, cell_or_mat)

% cell_or mat = 1, return f as a cell array
%             = 2, return f as a vector/matrix
% f_cell stores a separate of different size for each stratum
f_cell = {};
m = 0;
if isempty(Na)
    for i = 1:K
        for j = 1:N(i)
            m = m + 1 ;
            if K == 1
                f_cell{i}(1,j) = -minmax*probHandle(theta, runlength,...
                    problemRng, m);
            else
                f_cell{i}(1,j) = -minmax*probHandle([theta i sum(N) count], ...
                    runlength, problemRng, m);
            end
            calls(count) = calls(count) + 1;
        end
    end
else
    m = 0;
    f_cell = Fj;
    for i = 1:K
        m = m + N(i);
        if N(i) < Na(i)
            for j = N(i)+1:Na(i)
                m = m+1;
                if K == 1
                    f_cell{i}(1,j) = -minmax*probHandle(theta, runlength, ...
                        problemRng, m);
                else
                    f_cell{i}(1,j) = -minmax*probHandle([theta i sum(Na) count],...
                        runlength, problemRng, m);
                end
                calls(count) = calls(count) + 1;
            end
        end
    end
end

if cell_or_mat == 1
    f = f_cell;
else
    if isempty(Na)
        f = ConvertCelltoMatrix(f_cell, N, K);
    else
        f = ConvertCelltoMatrix(f_cell, Na, K);
    end
end
end

function f_mat = ConvertCelltoMatrix (f_cell, N, K)
% Convert cell array to vector/matrix
m = 0;
f_mat = zeros(1,sum(N));
for i = 1:K
    for j=1:N(i)
      m = m+1;
      f_mat(m) = f_cell{i}(1,j);
    end
end
end

function [Na, Fja, del_Fja, del_Fska, del_Fsa, calls] = AdaptiveSampling ...
    (probHandle, theta, problemRng, N, calls, count, ...
    Fj, del_Fj, del_Fsk, del_Fs, p, h, kappa, Fjhp, Fjhn, K, bin, minmax)

Na = N;
for i = 1:K
    % Determine the optimal sample size for each stratum
    while 1
        if var(del_Fj{i}.*del_Fsk(i))/N(i) <=(kappa*norm(del_Fsk(i))^4/p(i))
           break
        elseif N(i) == bin(i) || Na(i) == bin(i)
           break
        else
           N_temp = Na;  
           Na(i) = Na(i) + 1;
           [Fj, calls] = OracleCalls(probHandle, theta, 1, problemRng, ...
               N_temp, calls, count, Na, Fj, K, minmax, 1);
           [Fjhp, calls]= OracleCalls(probHandle, theta+h, 1, problemRng, ...
               N_temp, calls, count, Na, Fjhp, K, minmax, 1);
           [Fjhn, calls]= OracleCalls(probHandle, theta-h, 1, problemRng, ...
               N_temp, calls, count, Na, Fjhn, K, minmax, 1);
           for m=1:K
               % Gradient calculation
                del_Fj{m} = (Fjhp{m} - Fjhn{m})/(2*h);
                del_Fsk(m) = 1/N(m)*sum(del_Fj{m});
                del_Fs = del_Fs + 1/sum(N)*sum(del_Fj{m});
           end
           if Na(i) == bin(i)
               break
           end
        end
    end
end
del_Fja = del_Fj; 
Fja = Fj; 
del_Fska = del_Fsk;
del_Fsa = del_Fs; % Update gradient after adaptive sampling
end
