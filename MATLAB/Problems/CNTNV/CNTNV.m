%   ***************************************
%   *** Code written by German Gutierrez***
%   ***         gg92@cornell.edu        ***
%   ***                                 ***
%   *** Updated by Shane Henderson to   ***
%   *** use standard calling and random ***
%   *** number streams                  ***
%   ***                                 ***
%   *** Edited by David Newton 10/5/18  ***
%   ***                                 ***
%   *** Edited by Kurtis Konrad         ***
%   *** 2/13/2020 Improve performance   ***
%   *** when runlength==1               ***
%   ***************************************


%Returns Mean of Profit and derivative. Arbitrarily return RH derivative

function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = CNTNV(x, runlength, problemRng, seed)
% x is the quantity to buy
% runlength is the number of days of demand to simulate
% problemRng: a cell array of RNG streams for the simulation model 
% seed is the index of the substreams to use (integer >= 1)


constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;


%if (x < 0) || (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed)
if (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('x should be >= 0, runlength should be positive integer, seed must be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
    FnGrad = NaN;
    FnGradCov = NaN;
else
        
    cost = 5;
    sellPrice = 9;
    salvage = 1;
    alpha = 2;
    beta = 20; %parameters for demand distribution (Burr Type XII)
    
    % Generate a new stream for random numbers and set as global stream
    DemandStream = problemRng{1};
    RandStream.setGlobalStream(DemandStream);
    
    % Initialize for storage
    demand = zeros(runlength, 1);
    
    % Run simulation
    for i = 1:runlength
        
        % Start on a new substream
        DemandStream.Substream = seed + i - 1;
    
        % Generate demands
        demand(i) = ((1-rand(1)).^(-1/beta)-1).^(1/alpha);    
    end
    
    % Compute daily profit
    PerPeriodCost = cost * x;
    sales = min(demand, x);
    revenueSales = sellPrice * sales;
    revenueSalvage = salvage * (x - sales);
    profit = revenueSales + revenueSalvage - PerPeriodCost;
    RHDerivProfit = sellPrice * (demand > x) + salvage * (demand < x) - cost;   %partial derivative wrt x? not differentialble when demand=x
    
    % Calculate summary measures
    if runlength==1
        fn=profit;
        FnVar=0;
        FnGrad=RHDerivProfit;
        FnGradCov=0;
    else
        fn = mean(profit);  %mean of sample mean
        FnVar = var(profit) / runlength;	%variance of sample mean = var/n
        FnGrad = mean(RHDerivProfit);
        FnGradCov = var(RHDerivProfit) / runlength;
    end
    
end
