function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = TOLLNW(x, runlength, problemRng, seed)

% INPUTS
% x: a column vector equaling the decision variables theta
% runlength: the number of longest paths to simulate
% problemRng: a cell array of RNG streams for the simulation model 
% seed: the index of the first substream to use (integer >= 1)

% RETURNS: fn (total profit from making road improvements), FnVar

%   *************************************************************
%   ***                 Written by David Eckman               ***
%   ***            dje88@cornell.edu    Sept 7, 2018          ***
%   *************************************************************

FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

% Restrict p to lie in the nonnegative orthant
if ((sum(x < 0) >= 1) || (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed))
    fprintf('investments must be nonnegative, runlength should be positive integer, seed must be a positive integer\n');
    fn = NaN;
    FnVar = NaN;

else % main simulation

    % Set up the simulation parameters
    
    % Parameters A, R, lambda, TriA, TriB, and TriC are problem-specific

	% Adjacency matrix for network
    A = [0, 1, 1, 1; 1, 0, 1, 1; 1, 1, 0, 1; 1, 1, 1, 0]; % A = ones(nStations, nStations) - eye(nStations) for fully connected network
    [destination, origin] = find(A'); % two vectors containing the pairs of endpoints for routes
    nStations = size(A,1); % Number of stations
    nRoutes = nnz(A); % Number of routes

    % Routing matrix (including a column for leaving the network)
    R = [0, 0.1, 0.05, 0.05, 0.8; 0.1, 0, 0.05, 0.05, 0.8; 0.05, 0.05, 0, 0.1, 0.8; 0.05, 0.05, 0.1, 0, 0.8];
    
    % Raw arrival rates (by route)
    lambda_mat = [0, 3, 4, 5; 3, 0, 5, 6; 4, 5, 0, 7; 5, 6, 7, 0]./4;
    lambda_vec = lambda_mat(lambda_mat > 0)';
    
    % Toll fares for each route
    tolls = [0, 1, 1, 1; 1, 0, 1, 1; 1, 1, 0, 1; 1, 1, 1, 0];

    % Investments (as a matrix)
    route_invest = zeros(nStations, nStations);
    route_invest(A == 1) = x;
    route_invest = route_invest';

    % Make travel times triangular (a=0, b=1, c determined by investments x);
    TriA = zeros(nStations,nStations);
    TriB = ones(nStations,nStations);
    TriC = TriA + (TriB - TriA).*exp(-route_invest);
    
    T = 96; % Length of time horizon for sample path
    
    % Total investment costs
    costs = sum(x);

	% Toll revenues for each simulation of the network
	revenue = zeros(runlength,1);

    % Get random number streams from the inputs
    PoissonProcessStream = problemRng{1};
    TriangleStream = problemRng{2};
    RoutingStream = problemRng{3};
    
    for r = 1:runlength
        
        % Set the substreams for the arrival process, travel times, and routing
        PoissonProcessStream.Substream = seed + r - 1;
        TriangleStream.Substream = seed + r - 1;
        RoutingStream.Substream = seed + r - 1;
        
        % Initialize system state (empty)
        NextArrivalAtDest = Inf*ones(1, nRoutes); % Time a vehicle traveling on route (i,j) reaches j
        QueueLength = zeros(1, nRoutes); % Number of vehicles on or waiting to travel on route (i,j)

        % Set stream for generating passenger arrivals
        RandStream.setGlobalStream(PoissonProcessStream);
        NextArrivalToOrigExo = exprnd(1./lambda_vec); % Next exogenous arrival to route (i,j)
        
        % Set system clock time
        t = 0;

        while t <= T
    
            % Identify the next internal and external arrival events and corresponding routes
            [tInternalArrival, RouteInternalArrival] = min(NextArrivalAtDest);
            [tExternalArrival, RouteExternalArrival] = min(NextArrivalToOrigExo);

            % Advance system clock 
            [t, NextEventType] = min([tInternalArrival, tExternalArrival]);

            % Branch on whether the next arrival is internal or external
            if NextEventType == 1 % If the event is an internal arrival...

                O = origin(RouteInternalArrival); % origin
                D = destination(RouteInternalArrival); % destination

                revenue(r) = revenue(r) + tolls(O,D); % Collect toll from completed trip

                % Remove vehicle from queue
                QueueLength(RouteInternalArrival) = QueueLength(RouteInternalArrival) - 1;

                % Check if the vehicle will make another trip
                RandStream.setGlobalStream(RoutingStream);
                U = rand();
                dest_cumul_dist = cumsum(R(D,:)); % Routing from current destination to new destination
                NewDestination = find(dest_cumul_dist >= U, 1);

                if NewDestination == nStations + 1 % Car leaves network
                    % Do nothing
                elseif NewDestination <= nStations % Car moves elsewhere
                    % Determine index of new route
                    [~, NewRouteID] = max((origin == D).*(destination == NewDestination));
               
                    % Check if there is another vehicle on the new route
                    if QueueLength(NewRouteID) == 0 % If no vehicles, send the vehicle

                        % Generate new travel time
                        RandStream.setGlobalStream(TriangleStream);
                        NewD = NewDestination;
                        a = TriA(D,NewD);
                        b = TriB(D,NewD);
                        c = TriC(D,NewD);
                        TravelTime = GenerateTravelTime(a, b, c);
                        
                        % Add arrival at destination to the event list
                        NextArrivalAtDest(NewRouteID) = t + TravelTime; % Update next event time

                    else % Add car to queue
                        % Do nothing
                    end

                    % Add car to queue at new route
                    QueueLength(NewRouteID) = QueueLength(NewRouteID) + 1;

                end

                % Check if there is another vehicle to send on the original route
                if QueueLength(RouteInternalArrival) == 0 % If no vehicles waiting...
                    NextArrivalAtDest(RouteInternalArrival) = Inf;

                elseif QueueLength(RouteInternalArrival) > 0 % If at least one vehicle waiting

                    % Generate new travel time
                    RandStream.setGlobalStream(TriangleStream);
                    a = TriA(O,D);
                    b = TriB(O,D);
                    c = TriC(O,D);
                    TravelTime = GenerateTravelTime(a, b, c);

                    % Add arrival at destination to the event list
                    NextArrivalAtDest(RouteInternalArrival) = t + TravelTime; % Update next event time
                end

            elseif NextEventType == 2 % If the event is an external arrival...
                
                % Check if the route is not in use.
                if QueueLength(RouteExternalArrival) == 0 % If route is unused, send vehicle on route

                    % Generate new travel time
                    RandStream.setGlobalStream(TriangleStream);
                    O = origin(RouteExternalArrival); % origin
                    D = destination(RouteExternalArrival); % destination
                    a = TriA(O,D);
                    b = TriB(O,D);
                    c = TriC(O,D);
                    TravelTime = GenerateTravelTime(a, b, c);
                    
                    % Add arrival at destination to the event list
                    NextArrivalAtDest(RouteExternalArrival) = t + TravelTime; % Update next event time
                end

                % Add vehicle to the route
                QueueLength(RouteExternalArrival) = QueueLength(RouteExternalArrival) + 1;

                % Generate next external arrival to the route
                RandStream.setGlobalStream(PoissonProcessStream);
                NextArrivalToOrigExo(RouteExternalArrival) = t + exprnd(1./lambda_vec(RouteExternalArrival));

            end

        end
        
    end

    % Record final mean and variance of cumulative costs
	fn = mean(revenue - costs);
	FnVar = var(revenue - costs)/runlength;

end

function TravelTime =  GenerateTravelTime(a, b, c)
U = rand();
if U < (c - a)/(b - a)
    TravelTime = a + sqrt(U*(b - a)*(c - a));
else
    TravelTime = b - sqrt((1-U)*(b - a)*(b - c));
end