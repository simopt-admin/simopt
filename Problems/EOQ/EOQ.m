function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = EOQ(x, runlength, problemRng, seed)

%% -- INPUTS:
% x is Q-order quantity
% runlength is the number of replications to simulate
% seed is the index of the substreams to use (integer >= 1)
% other is not used

%% -- OUTPUTS:
% RETURNS fn, FnVar --mean and variance of total cost

%   *************************************************************
%   ***             Written by Danielle Lertola               ***
%   ***         dcl96@cornell.edu    July 23, 2012            ***
%   ***   Parameter Units don't match, needs adjustment 7/23/12**
%   ***            Edited by Jennifer Shih                    ***
%   ***          jls493@cornell.edu    June 20th, 2014        ***
%   *** Edited by Shane Henderson sgh9@cornell.edu, 7/14/14   ***
%   ***                Edited by David Eckman                 ***
%   ***          dje88@cornell.edu    Sept 14th, 2018         ***
%   ***                Edited by Kurtis Konrad                ***
%   ***            kekonrad@ncsu.edu    Feb 13, 2020          ***
%   *************************************************************
%
% Last updated Feb 13, 2020

%% -- SET known outputs 
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

%% -- CHECK FOR ERRORS; 

if (x < 0) || (runlength <= 0) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('x should be nonnegative, runlength should be positive and real, seed should be a positive integer\n');
    fn = NaN;
    FnVar = NaN; 
    FnGrad = NaN;
    FnGradCov = NaN;
else % main simulation
    
    %% *********************PARAMETERS*********************
    
    mu = 1040;    % mean of gamma demand, nominal demand value
    delta = 104;  % std dev of gamma demand 
    k = 10;       % fixed cost to place order
    c = 15;       % variable cost to place order (dollars/ unit-quantity)
    h = 2.5;      % holding cost for one unit of inventory/unit of time

    %% Get random number stream from input and set as global stream
    GammaStream = problemRng{1};
    RandStream.setGlobalStream(GammaStream);
    
    
    %% Generate random data

    shape = mu^2/delta^2;
    scale = delta^2/mu;

    rate = zeros(runlength, 1);
    for i = 1:runlength
        GammaStream.Substream = seed + i - 1;
        rate(i) = gamrnd(shape, scale);    % constant demand rate
    end    
    
    %% Main
    if runlength==1
        MeanRate=rate;
        VarRate=0;
    else
        MeanRate = mean(rate);
        VarRate = var(rate);
    end
    fn = (c + k / x) * MeanRate + h * x / 2;    %value of objective
    FnVar = (c + k / x)^2 * VarRate / runlength;
    FnGrad = (-k / x^2) * MeanRate + h / 2;
    FnGradCov = (k^2 / x^4) * VarRate / runlength;
end
    
    
    