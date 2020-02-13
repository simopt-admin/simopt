function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = GPE(x, runlength, problemRng, seed)
%function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = ParameterEstimation(x, runlength, seed, other)
% x is vector of parameters
% runlength is the number of trials to simulate
% problemRng: a cell array of RNG streams for the simulation model
% seed is the index of the substreams to use (integer >= 1)

%   ****************************************
%   ***   Code written by Jessica WU     ***
%   ***        lw353@cornell.edu         ***
%   ***   Edited by Shane Henderson      ***
%   ***      and Jennifer Shih           ***
%   ***       jls493@cornell.edu         ***
%   ***      and Bryan Chong             ***
%   ***        bhc34@cornell.edu         ***
%   ***      and David Newton            ***
%   ***       newton34@purdue.edu        ***
%   ****************************************
%

% Last updated Feb 13, 2020 by Kurtis Konrad

% RETURNS:  the value G_m giving the sample average of the log likelihood
% m is the "runlength" input

constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

if (x(1)<0)||(x(2)<0)|| (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('x should be a vector with 2 positive entries, runlength should be positive integer, seed must be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
    FnGrad = NaN;
    FnGradCov = NaN;
    
else
    xstar = [2,5]; % point used to generate data
    
    % Generate a new stream for random numbers
    OurStream1 = problemRng{1};
    OurStream2 = problemRng{2};

    % Initialize for storage
    y1 = zeros(runlength, 1);
    y2 = zeros(runlength, 1);
    
    % Run simulation
    for i = 1:runlength
    
        % Set the substream to the "seed"
        OurStream1.Substream = seed + i - 1;
        OurStream2.Substream = seed + i - 1;

        % Generate one set of y's
        RandStream.setGlobalStream(OurStream1);
        y2(i) = gamrnd(xstar(2), 1);

        % Generate other set of y's
        RandStream.setGlobalStream(OurStream2);
        y1(i) = gamrnd(xstar(1) * y2(i), 1);
    end    
        
    ReplicationVector = -y1 - y2 + (x(1) * y2 - 1).* log(y1) + (x(2)-1)* log(y2)-log(gamma(x(1)*y2))-log(gamma(x(2)));
  
    % Calculate summary measures
    if runlength==1
        fn = ReplicationVector;
        FnVar = 0;
    else
        fn = mean(ReplicationVector);
        FnVar = var(ReplicationVector)/runlength;
    end
    FnGrad = NaN;
    FnGradCov = NaN;
    
end

