function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = MM1(beta, runlength, problemRng, seed)

% beta: a vector containing the coefficients used to estimate average waiting time in M/M/1 queue given rho is the utilization rates 
% runlength: the number of replications, each consisting of m customers, to simulate
% problemRng: a cell array of RNG streams for the simulation model 
% seed: the index of the substream to use (integer >= 1)

% Returns error of approx, no var or gradient estimates.

% Approximating expected waiting times per call in an M/M/1 queue through a
% polynomial of the form (beta_0 + beta_1 x + beta_2 x^2)/(1-x)

FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

if (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('runlength should be positive integer, seed must be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
    
else % main simulation
    rho = [0.5, 0.564, 0.706, 0.859, 0.950];  % sampling points (utilization rates)
    nrhos = length(rho);
    nCalls = 1000; % Specified in problem description as number of customers in a sample path
    arrivalRate = 1;
    ServiceTimeMeans = rho / arrivalRate; % Row vector of service time means    
    
    % Get random number streams from the inputs
    ArrivalStream = problemRng{1};
    ServiceStream = problemRng{2};

    %Generate estimates at each point through simulation:
    avgWait = zeros(runlength, nrhos); % Contains the average wait for each replication for each value of the traffic intensity
    varWait = zeros(runlength, nrhos); % Contains the std dev of the waits for each replication for each value of the traffic intensity
    
    % We simulate using the Lindley recursion that gives the next customer wait
    % time in terms of the previous one. W_{j+1} = max{0, W_j + service time -
    % interarrival time}
    for k = 1:runlength      
        
        % Set the substream to the "seed"
        ArrivalStream.Substream = seed + k - 1;
        ServiceStream.Substream = seed + k - 1;
    
        % Generate arrival times for this run
        RandStream.setGlobalStream(ArrivalStream);
        Interarrival = exprnd(1/arrivalRate, 1, nCalls); % now a vector
        
        % Generate service times for this run
        RandStream.setGlobalStream(ServiceStream);
        UnscaledService = exprnd(1, 1, nCalls);
        
        waitTimes = zeros(nCalls, nrhos); % Wait time for each call. The first wait times are always zero.
        
        for j = 2:nCalls
            %waitTimes(j, :) = max(0, waitTimes(j-1, :) + UnscaledService(k, j)*ServiceTimeMeans - Interarrival(k, j)); % Generates all 5 wait times at once
            waitTimes(j, :) = max(0, waitTimes(j-1, :) + UnscaledService(j)*ServiceTimeMeans - Interarrival(j)); % Generates all 5 wait times at once
        end
        
        avgWait(k, :) = mean(waitTimes); % row vector giving average wait on this sample path for each utilization
        varWait(k, :) = var(waitTimes); % row vector giving std deviation of wait on this sample path for each utilization
    end

    % Compute overall objective function estimator and variance
    approx = ((beta(1) + beta(2)*rho + beta(3)*(rho.^2))./(1-rho)); % Row vector giving metamodel approximation for each utilization
    OverallMeanEstimate = mean(avgWait, 1); % Row vector containing an estimator of f bar, indexed by rho
    OverallVarEstimate = mean(varWait, 1); % Row vector containing an estimator of sigma^2, indexed by rho
    fn = sum((OverallMeanEstimate - approx).^2 ./ OverallVarEstimate); % Objective function - temporary because we will average pseudovalues instead

    if runlength == 1
        
        % No variance estimate to report;
        FnVar = NaN;
        
    else

        % To get an estimate of the variability of the function estimate we use
        % Jackknifing
        Pseudovalues = zeros(runlength, 1); % This vector will contain the pseudovalues
        SumOfAvgs = sum(avgWait);
        SumOfVars = sum(varWait);

        for k = 1:runlength % Compute the pseudovalues
            SmallAvg = (SumOfAvgs - avgWait(k, :)) / (runlength - 1);
            SmallVar = (SumOfVars - varWait(k, :)) / (runlength - 1);
            SmallEst = sum((SmallAvg - approx).^2 ./ SmallVar);
            Pseudovalues(k) = runlength * fn - (runlength - 1) * SmallEst;
        end

        fn = mean(Pseudovalues);
        FnVar = var(Pseudovalues) / runlength;

    end
end
