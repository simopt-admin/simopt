function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = QUEGG1(x, runlength, problemRng, seed)

%% -- INPUTS:
% x: a theta value chosen, within limits, represent mean #arrivals
% runlength:the number of replications to simulate
% problemRng: a cell array of RNG streams for the simulation model 
% seed: the index of the first substream to use (integer >= 1)

%% -- OUTPUTS:
% RETURNS fn and FnVar, throughput are variance in customers/minute


%   *************************************************************
%   ***             Written by Danielle Lertola               ***
%   ***         dcl96@cornell.edu    July 16, 2012            ***
%   ***            Edited by Jennifer Shih                    ***
%   ***          jls493@cornell.edu    June 16th, 2014        ***
%   ***            Edited by Bryan Chong                      ***
%   ***          bhc34@cornell.edu    September 13th, 2014    ***
%   *************************************************************
%
% Last updated Februrary 13, 2020 by Kurtis Konrad

%% -- SET known outputs (note that function only returns fn and FnVar)
FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

%% -- CHECK FOR ERRORS; 

thLow = 1;    %theta low limit
thHigh = 2;   %theta high limit

if (runlength <= 0) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('runlength should be positive and real, seed should be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
    
elseif x>thHigh || x<thLow
    fprintf('Theta must be between set limits.');
    fn = NaN;
    FnVar = NaN;
    
else % main simulation
    
    %% *********************PARAMETERS*********************
    
    lambda = 2;                 % parameter for inter-arrival times
    warmup = 20;                % people warmup before start count
    people = 50;                % measure avg sojourn time over next fifty people.  
    theta = x;                  % parameter for service time
    total = warmup + people;         
    nRuns = runlength;          % number of repetitions
    
    %% GENERATE RANDOM NUMBER STREAMS
    
    % Get random number streams from the inputs
    ArrivalStream = problemRng{1};
    ServiceStream = problemRng{2};
    
    %% RUN SIMULATIONS
    
    % Tracks performance times for each run
    MeanCustTime=zeros(nRuns,1);
    for k = 1:nRuns                           
        
        % Set the substreams for the arrival process and service process
        ArrivalStream.Substream = seed + k - 1;
        ServiceStream.Substream = seed + k - 1;
        
        % Generate interarrival times
        RandStream.setGlobalStream(ArrivalStream); 
        arrival = exprnd(1/lambda,total,1);   
        
        %Generate service times
        RandStream.setGlobalStream(ServiceStream);
        service = exprnd(1/theta,total,1);
        
                                            % Customers:: 
        Customer = zeros(total,4);            % Column 1-time of arrival to queue, Column 2-service time, Column
        Customer(:,1) = cumsum(arrival); % 3-time service complete, Column 4-sojourn time

        Customer(:,2) = service;                     % Insert service times into matrix
       
        Customer(1,3) = Customer(1,1) + Customer(1,2);      % Put first customer through empty system
        Customer(1,4) = Customer(1,2);
        
        for i = 2:total
            Customer(i,3) = max([Customer(i,1), Customer(i-1,3)]) + Customer(i,2);
            Customer(i,4) = Customer(i,3) - Customer(i,1);  % remaining customers through system
        end
        MeanCustTime(k) = mean(Customer(21:total,4));     % Calculate mean sojourn time for last 50 customers
    end
    if runlength==1
        fn=MeanCustTime + theta^2;
        FnVar=0;
    else
        fn = mean(MeanCustTime)+ theta^2;                      % Mean and Variance of Alpha Function=Mean sojourn time + theta^2
        FnVar = var(MeanCustTime)/nRuns;
    end
    
end
end
