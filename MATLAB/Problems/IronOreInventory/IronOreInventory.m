function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = IronOreInventory(x, runlength, seed, ~)
% x is a row vector that indicates: [Price at which to begin production, 
% Maximum current stock at which to produce, Price at which to stop production, 
% Price at which to sell]
% runlength is number of replications
% seed is the index of the substreams to use (integer >= 1)
% other is not used
% Returns expected maximal operating profit

%   *************************************************************
%   ***                Written by Bryan Chong                 ***
%   ***       bhc34@cornell.edu    January 25th, 2015         ***
%   *************************************************************
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;
FnGrad=NaN;
FnGradCov=NaN;

if (sum(x < 0)>0) || round(x(2)) ~= x(2) > 0 || (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed),
    fprintf('x components must be nonnegative, x(2) must be integral, runlength and seed must be a positive integers\n');
    fn = NaN;
    FnVar = NaN;
    FnGrad = NaN;
    FnGradCov = NaN;
else
    %% Parameters
    MeanPrice = 100; % Mean ore price per unit
    MinPrice = 0; % Minimum ore price per unit
    MaxPrice = 200; % Maximum ore price per unit
    StDev = 7.5; % Standard deviation of random walk steps
    Capacity = 10000; % Stock capacity K
    ProdCost = 100; % Production cost per unit c
    HoldCost = 1; % Holding cost per unit per day h
    ProdPerDay = 100; % Maximum units produced per day m
    T = 1000; % Number of days T
    
    %% From Inputs/Parameters
    ProduceAt = x(1); % Start production when price reaches this level
    MaxStock = x(2); % If we have this much stock, do not produce until stock is sold
    StopAt = x(3); % Stop production when price reaches this level
    SellAt = x(4); % Sell at this level
    nReps = runlength; % Number of replications
    
    %% Generate Random Variables
    [NormStream] = RandStream.create('mrg32k3a', 'NumStreams', 1);
    % Set the substream to the "seed"
    NormStream.Substream = seed;
    
    % Generate standard normal random variables
    OldStream = RandStream.setGlobalStream(NormStream); % Temporarily store old stream
    Norms = normrnd(0,1,nReps,T);
    
    % Restore old random number stream
    RandStream.setGlobalStream(OldStream);
    
    %% Main Simulation
    AvgRevenue = zeros(nReps, 1);
    for rep = 1:nReps
        price = MeanPrice;
        stock = 0;
        producing = 0; % Whether we are currently producing or not
        revenue = 0;
        for t = 1:T
            % Determine new price.
            meanVal = sqrt(sqrt(abs(MeanPrice - price)));
            meanDir = sign(MeanPrice - price);
            meanMove = meanDir * meanVal;
            move = Norms(rep, t)*StDev + meanMove;
            price = max(min(price + move, MaxPrice), MinPrice); % Keep price within bounds
            
            if producing == 0 % Begin production if price is high enough
                if (price > ProduceAt) && (stock < MaxStock)
                    producing = 1;
                    prod = min(Capacity - stock, ProdPerDay);
                    stock = stock + prod;
                    revenue = revenue - prod*ProdCost;
                end
            else % Halt production if price dips too low
                if price < StopAt
                    producing = 0;
                else
                    prod = min(Capacity - stock, ProdPerDay);
                    stock = stock + prod;
                    revenue = revenue - prod*ProdCost;
                end
            end
            
            % Sell if price is high enough
            if price > SellAt
                revenue = revenue + stock*price;
                stock = 0;
            end
            
            % Charge holding cost
            revenue = revenue - stock*HoldCost;
        end
        AvgRevenue(rep) = revenue;
    end
    fn = mean(AvgRevenue);
    FnVar = var(AvgRevenue);
end
end

