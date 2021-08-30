function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = FACLOC(x, runlength, problemRng, seed)

% x: a column vector containing the coordinates of the warehouses, (x1, y1, x2, y2)
% runlength: the number of longest paths to simulate
% problemRng: a cell array of RNG streams for the simulation model 
% seed: the index of the first substream to use (integer >= 1)

% RETURNS: fn (proportion of deliveries made in less than Tau), FnVar
% Uses discrete event simulation with only 2 event types:
% 1. Order arrives
% 2. Truck arrives back to warehouse.

%   ***********************************************
%   ***    Code written by German Gutierrez     ***
%   ***            gg92@cornell.edu             ***
%   ***    Updated by Danielle Lertola to       ***
%   ***    use standard calling and random      ***
%   ***    number streams - 6/7/2012            ***
%   ***    Updated by Jennifer Shih 7/1/2014    ***
%   ***    Updated by Bryan Chong 9/17/2014     ***
%   ***    Updated by Anna Dong 11/17/2016      ***
%   ***    Updated by David Eckman 9/7/2018     ***
%   ***********************************************

nWarehouses = 2; % # of warehouses

FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

if (sum(x < 0) > 0) || (sum(x>1)>0) || (sum(size(x) ~= [1, 2 * nWarehouses])>0) || (runlength <= 0) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('x (row vector with %d components)\nx components should be between 0 and 1\nrunlength should be positive and real,\nseed should be a positive integer\n', nWarehouses*2);
    fn = NaN;
    FnVar = NaN;
else % main simulation
    
    % Initialize parameters
    arrivalRate = 1/3;      % Arrival rate, per minute
    pickupRate = 1/5;       % Pick up time rate, per minute
    deliveryRate = 1/10;    % Delivery time rate, per minute
    vInKm = 30;
    v = vInKm/30;           % Velocity in terms of unit square
    nTrucks = [10, 10];      % # of trucks per warehouse
    tau = 60;               % Service level time required
    
    bases = zeros(nWarehouses, 2);  % Base locations
    for i = 1:nWarehouses
        bases(i,:) = x((2*i - 1):(2*i));
    end
    
    nDays = runlength;
    nOrders = zeros(nDays, 1);
    nLessThanTau = zeros(nDays, 1);
    
    % Get random number streams from the inputs
    ArrivalTimeStream = problemRng{1};
    PickupTimeStream = problemRng{2};
    DeliveryTimeStream = problemRng{3};
    LocationStream = problemRng{4};

    for i = 1:nDays
                
        % Set the substreams for the arrival process, pickup/delivery times
        % and locations
        ArrivalTimeStream.Substream = seed + i - 1;
        PickupTimeStream.Substream = seed + i - 1;
        DeliveryTimeStream.Substream = seed + i - 1;
        LocationStream.Substream = seed + i - 1;
        
        % Events will track all future events:
        % order arrival (type 1, time, x-location of order, y-location)
        % truck returns to WH (type 2, time, WH number code)
        Events = zeros(4,1);
        %orderQueue tracks all orders not yet fulfilled with the following
        % information: (time of arrival, x-location, y-location)
        orderQueue = zeros(3,1);
        
        % Make t_i trucks available at location i:
        nAvailable = nTrucks;
        
        %Generate first order and its location
        while Events(1,1) == 0
            % Generate location using acceptance-rejection
            RandStream.setGlobalStream(LocationStream);
            u = rand(3,1);

            if 1.6*u(3) <= 1.6 - (abs(u(1) - 0.8) + abs(u(2) - 0.8))
                loc = u(1:2);
                % Generate arrival time
                RandStream.setGlobalStream(ArrivalTimeStream);
                Events(:,1) = [1; exprnd(1/arrivalRate); loc(1); loc(2)];
            end
        end
        
        [~, nextEvent] = min(Events(2,:));
        time = Events(2, nextEvent);
        
        while ( time <= 540 || length(orderQueue(1,:)) > 1 )
                        
            if ( Events(1, nextEvent) == 1 )
                %ORDER ARRIVES:
                %Receive orders until 5PM
                if ( time <= 540 )
                    %If no orders in queue and truck available, delivered by closest available truck
                    if ( sum(nAvailable) > 0 )
                        minDist = 100;
                        closest = nWarehouses + 1;
                        for j = 1:nWarehouses
                            tempDist = abs(Events(3,nextEvent) - bases(j,1)) + abs(Events(4,nextEvent) - bases(j,2));
                            if ( nAvailable(j) > 0 && tempDist < minDist )
                                minDist = tempDist;
                                closest = j;
                            end
                        end
                        
                        nAvailable(closest) = nAvailable(closest) - 1;
                        
                        %Generate pick up time
                        RandStream.setGlobalStream(PickupTimeStream);
                        pTime = exprnd(1/pickupRate);
                        
                        %Generate delivery time
                        RandStream.setGlobalStream(DeliveryTimeStream);
                        dTime = exprnd(1/deliveryRate);
                        
                        oneWayTravelTime = 60*(minDist/v);
                        completionTime = time + pTime + oneWayTravelTime + dTime;
                        returnTime = completionTime + oneWayTravelTime;
                        
                        Events = [Events, [2; returnTime; closest; 0]];
                        
                        % Check if delivery is completed (including "delivery
                        % time") within tau minutes
                        if (completionTime - time < tau)
                            nLessThanTau(i) = nLessThanTau(i) + 1;
                        end
                    else
                        % Order enters queue.
                        orderQueue = [orderQueue, [time; Events(3,nextEvent); Events(4,nextEvent)]];
                    end
                    nOrders(i) = nOrders(i) + 1;
                    
                    
                    %Generate next order
                    success = 0;
                    while success == 0
                        RandStream.setGlobalStream(LocationStream);
                        u = rand(3,1);
                        if 1.6*u(3) <= 1.6-(abs(u(1)-0.8)+abs(u(2)-0.8))
                            RandStream.setGlobalStream(ArrivalTimeStream);
                            Events = [Events, [1; exprnd(1/arrivalRate)+time; u(1); u(2)]];
                            success = 1;
                        end
                    end
                end
                
                Events(:,nextEvent) = [];
                [~, nextEvent] = min(Events(2,:));
                time = Events(2, nextEvent);
                
            elseif ( Events(1, nextEvent) == 2 )
                %TRUCK BECOMES AVAILABLE:
                if ( length(orderQueue(1,:)) > 1 )
                    %Truck takes the first call in queue
                    
                    %Generate pick up time
                    RandStream.setGlobalStream(PickupTimeStream);
                    pTime = exprnd(1/pickupRate);
                    
                    %Generate delivery time
                    RandStream.setGlobalStream(DeliveryTimeStream);
                    dTime = exprnd(1/deliveryRate);
                    
                    minDist = abs(orderQueue(2,2) - bases(Events(3,nextEvent),1)) + abs(orderQueue(3,2) - bases(Events(3,nextEvent),2));
                    oneWayTravelTime = 60*(minDist/v);
                    completionTime = time + pTime + oneWayTravelTime + dTime;
                    returnTime = completionTime + oneWayTravelTime;
                    
                    Events = [Events, [2; returnTime; Events(3,nextEvent); 0]];
                    
                    % Check if delivery is completed (including "delivery
                    % time") within tau minutes
                    if (completionTime - orderQueue(1,2) < tau )
                        nLessThanTau(i) = nLessThanTau(i) + 1;
                    end
                    
                    orderQueue(:,2) = [];
                else
                    %No calls in queue, truck remains idle.
                    nAvailable(Events(3,nextEvent)) = nAvailable(Events(3,nextEvent)) + 1;
                end
                
                Events(:,nextEvent) = [];
                [~, nextEvent] = min(Events(2,:));
                time = Events(2,nextEvent);
            end
        end

    end
    
    % Use ratio estimation
    fn = sum(nLessThanTau)/sum(nOrders); 
    m_x = mean(nLessThanTau);
    numerator = sum(nOrders - fn * nLessThanTau);
    FnVar = (1/m_x^2)*(numerator/(nDays-1)); % Variance estimate for ratio estimaor
    
end
