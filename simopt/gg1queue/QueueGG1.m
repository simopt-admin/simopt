function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = QueueGG1(x, runlength, seed, ~)
% function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = QueueGG1(x, runlength, seed, other);

%% -- INPUTS:
% x is a theta value chosen, within limits, represent mean #arrivals
% runlength is the number of replications to simulate
% seed is the index of the substreams to use (integer >= 1)
% other is not used
%Note: RandStream.setGlobalStream(stream) can only be used for Matlab
%versions 2011 and later
%For earlier versions, use the method RandStream.setDefaultStream(stream)
%

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
% Last updated September 13th, 2014

%% -- SET known outputs (note that function only returns fn and FnVar)
FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

%% -- CHECK FOR ERRORS; 

thLow=1;    %theta low limit
thHigh=2;   %theta high limit

if (runlength <= 0) || (seed <= 0) || (round(seed) ~= seed),
    fprintf('runlength should be positive and real, seed should be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
    
elseif x>thHigh || x<thLow,
    fprintf('Theta must be between set limits.');
    fn = NaN;
    FnVar = NaN;
    
else % main simulation
    
    %% *********************PARAMETERS*********************
    
    lambda=2;       %parameter for inter-arrival times
    warmup=20;      %people warmup before start count
    people=50;      %measure avg sojourn time over next fifty people.  
    theta=x;        %parameter for service time
    total=warmup+people;         
    nRuns=runlength;%number of repetitions
    %% GENERATE RANDOM NUMBER STREAMS
    % Generate new streams for interarrival and service times
    [ArrivalStream, ServiceStream] = RandStream.create('mrg32k3a', 'NumStreams', 2);
    % Set the substream to the "seed"
    ArrivalStream.Substream = seed;
    ServiceStream.Substream = seed;
    
    %% Generate random data
    OldStream = RandStream.setGlobalStream(ArrivalStream);     % Temporarily store old stream
    %OldStream = RandStream.setDefaultStream(ArrivalStream);     %for versions 2010 and earlier
    arrival=exprnd(1/lambda,total,nRuns);                      % Generate Interarrival time
    RandStream.setGlobalStream(ServiceStream);              
    %RandStream.setDefaultStream(ServiceStream);     %for versions 2010 and earlier
    service=exprnd(1/theta,total,nRuns);                       % Generate Service time
    RandStream.setGlobalStream(OldStream);                     % Restore old random number stream
    %RandStream.setDefaultStream(OldStream);     %for versions 2010 and earlier
    
    
    %% RUN SIMULATIONS
    
    %Tracks performance times for each run
    MeanCustTime=zeros(nRuns,1);
    parfor k=1:nRuns                           
                                            % Customers:: 
        Customer=zeros(total,4);            % Column 1-time of arrival to queue, Column 2-service time, Column
        Customer(:,1)=cumsum(arrival(:,k)); % 3-time service complete, Column 4-sojourn time

        Customer(:,2)=service(:,k);                     % Insert service times into matrix
       
        Customer(1,3)=Customer(1,1)+Customer(1,2);      % Put first customer through empty system
        Customer(1,4)=Customer(1,2);
        
        for i=2:total
            Customer(i,3)=max([Customer(i,1),Customer(i-1,3)])+Customer(i,2);
            Customer(i,4)=Customer(i,3)-Customer(i,1);  % remaining customers through system
        end
        MeanCustTime(k)=mean(Customer(21:total,4));     % Calculate mean sojourn time for last 50 customers
    end
    fn=mean(MeanCustTime)+ theta^2;                      % Mean and Variance of Alpha Function=Mean sojourn time + theta^2
    FnVar=var(MeanCustTime)/nRuns;
end
end
