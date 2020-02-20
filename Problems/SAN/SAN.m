function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = SAN(x, runlength, problemRng, seed)

% INPUTS
% x: a column vector equaling the decision variables theta
% runlength: the number of longest paths to simulate
% problemRng: a cell array of RNG streams for the simulation model 
% seed: the index of the first substream to use (integer >= 1)

% RETURNS
% Estimated fn value
% Estimate of fn variance
% Estimated gradient. This is an IPA estimate so is the TRUE gradient of
% the estimated function value
% Estimated gradient covariance matrix

%   *************************************************************
%   ***                Written by Shane Henderson             ***
%   ***            sgh9@cornell.edu    April 13, 2013         ***
%   ***                Edited by Bryan Chong                  ***
%   ***            bhc34@cornell.edu    Sept 10, 2014         ***
%   ***                Edited by David Eckman                 ***
%   ***            dje88@cornell.edu    Sept 4, 2018          ***
%   ***                Edited by Kurtis Konrad                ***
%   ***            kekonrad@ncsu.edu    Feb 13, 2020          ***
%   *************************************************************

constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

if (runlength <= 0) || (round(runlength) ~= runlength) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('runlength should be a positive integer,\nseed should be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
    FnGrad = NaN;
    FnGradCov = NaN;
    
else % main simulation
    numarcs = 13;
    numnodes = 9;
    [a, b] = size(x);
    if (a == 1) && (b == numarcs)
        theta = x'; %theta is a column vector
    elseif (a == numarcs) && (b == 1)
        theta = x;
    else
        fprintf('x should be a column vector with %d rows\n', numarcs);
        fn = NaN; FnVar = NaN; FnGrad = NaN; FnGradCov = NaN;
        return;
    end
    rowtheta = theta'; % Convert to row vector
    
    % Get random number stream from input and set as global stream
    DurationStream = problemRng{1};
    RandStream.setGlobalStream(DurationStream);
    
    % Initialize for storage
    cost = zeros(runlength, 1);
    CostGrad = zeros(runlength, numarcs);
    
    % Run simulation
    for i = 1:runlength
            
        % Start on a new substream
        DurationStream.Substream = seed + i - 1;
        
        % Generate random duration data
        RawExponentials = exprnd(1, 1, numarcs);
        arcs = RawExponentials .* rowtheta;
        
        T = zeros(numnodes, 1);
        Tderiv = zeros(numnodes, numarcs);
        
        % Brute force calculation since the network is tiny. Easily coded more
        % algorithmically, but not worth it for such a small network.
        
        T(2) = T(1) + arcs(1);
        Tderiv(2, :) = Tderiv(1, :);
        Tderiv(2, 1) = Tderiv(2, 1) + arcs(1) / theta(1);
        
        T(3) = max(T(1) + arcs(2), T(2)+arcs(3));
        if T(1) + arcs(2) > T(2) + arcs(3)
            T(3) = T(1) + arcs(2);
            Tderiv(3, :) = Tderiv(1, :);
            Tderiv(3, 2) = Tderiv(3, 2) + arcs(2) / theta(2);
        else
            T(3) = T(2) + arcs(3);
            Tderiv(3, :) = Tderiv(2, :);
            Tderiv(3, 3) = Tderiv(3, 3) + arcs(3) / theta(3);
        end
        
        T(4) = T(2) + arcs(4);
        Tderiv(4, :) = Tderiv(2, :);
        Tderiv(4, 4) = Tderiv(4, 4) + arcs(4) / theta(4);
        
        T(5) = T(4) + arcs(7);
        Tderiv(5, :) = Tderiv(4, :);
        Tderiv(5, 7) = Tderiv(5, 7) + arcs(7) / theta(7);
        
        [T(6), ind] = max([T(2) + arcs(5), T(3) + arcs(6), T(5) + arcs(9)]);
        if ind==1
            Tderiv(6, :) = Tderiv(2, :);
            Tderiv(6, 5) = Tderiv(6, 5) + arcs(5) / theta(5);
        elseif ind==2
            Tderiv(6, :) = Tderiv(3, :);
            Tderiv(6, 6) = Tderiv(6, 6) + arcs(6) / theta(6);
        else
            Tderiv(6, :) = Tderiv(5, :);
            Tderiv(6, 9) = Tderiv(6, 9) + arcs(9) / theta(9);
        end
        
        T(7) = T(4) + arcs(8);
        Tderiv(7, :) = Tderiv(4, :);
        Tderiv(7, 8) = Tderiv(7, 8) + arcs(8) / theta(8);
        
        if T(7) + arcs(12) > T(5) + arcs(10)
            T(8) = T(7) + arcs(12);
            Tderiv(8, :) = Tderiv(7, :);
            Tderiv(8, 12) = Tderiv(8, 12) + arcs(12) / theta(12);
        else
            T(8) = T(5) + arcs(10);
            Tderiv(8, :) = Tderiv(5, :);
            Tderiv(8, 10) = Tderiv(8, 10) + arcs(10) / theta(10);
        end
        
        if T(6) + arcs(11) > T(8) + arcs(13)
            T(9) = T(6) + arcs(11);
            Tderiv(9, :) = Tderiv(6, :);
            Tderiv(9, 11) = Tderiv(9, 11) + arcs(11) / theta(11);
        else
            T(9) = T(8) + arcs(13);
            Tderiv(9, :) = Tderiv(8, :);
            Tderiv(9, 13) = Tderiv(9, 13) + arcs(13) / theta(13);
        end
        
        cost(i) = T(9) + sum(1 ./ rowtheta);
        CostGrad(i, :) = Tderiv(9, :) - 1 ./ rowtheta.^2;
    end
    
    % Calculate summary measures
    if runlength==1
        fn=cost;
        FnVar=0;
        FnGrad=CostGrad;
        FnGradCov=zeros(length(CostGrad));
    else
        fn = mean(cost);
        FnVar = var(cost)/runlength;
        FnGrad = mean(CostGrad, 1); % Calculates the mean of each column as desired
        FnGradCov = cov(CostGrad); %FnGradCov = cov(CostGrad, 2);
    end
end
