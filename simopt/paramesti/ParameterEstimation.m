function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = ParameterEstimation(x, runlength, seed, ~)
%function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = ParameterEstimation(x, runlength, seed, other)
% x is vector of parameters
% runlength is the number of trials to simulate
% seed is the index of the substreams to use (integer >= 1)

%   ****************************************
%   ***   Code written by Jessica WU     ***
%   ***        lw353@cornell.edu         ***
%   ***   Edited by Shane Henderson      ***
%   ***      and Jennifer Shih           ***
%   ***       jls493@cornell.edu         ***
%   ***      and Bryan Chong             ***
%   ***        bhc34@cornell.edu         ***
%   ****************************************
%

% Last updated Sept 15, 2014 by Bryan Chong

% RETURNS:  the value G_m giving the sample average of the log likelihood
% m is the "runlength" input

constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

xdimen = size(x);

if (x(1)<0)||(x(2)<0)|| (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed),
    fprintf('x should be a vector with 2 positive entries, runlength should be positive integer, seed must be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
    FnGrad = NaN;
    FnGradCov = NaN;
    
else
    xstar = [2,5]; % point used to generate data
    
    % Generate a new stream for random numbers
    [OurStream1, OurStream2] = RandStream.create('mrg32k3a', 'NumStreams', 2);

    % Set the substream to the "seed"
    OurStream1.Substream = seed;
    OurStream2.Substream = seed;
    
    % Generate one set of y's
    OldStream = RandStream.setGlobalStream(OurStream1);
    y2 = gamrnd(xstar(2), 1, runlength, 1);
    
    % Generate other set of y's
    RandStream.setGlobalStream(OurStream2);
    y1 = gamrnd(xstar(1) * y2, 1, runlength, 1);
    
    RandStream.setGlobalStream(OldStream); %Restore old stream

    ReplicationVector = -y1 - y2 + (x(1) * y2 - 1).* log(y1) + (x(2)-1)* log(y2)-log(gamma(x(1)*y2))-log(gamma(x(2)));
  
    fn= mean(ReplicationVector);
    FnVar = var(ReplicationVector)/runlength;
    FnGrad = NaN;
    FnGradCov = NaN;
    
end

