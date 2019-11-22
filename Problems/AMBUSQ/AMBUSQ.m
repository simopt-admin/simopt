function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = AMBUSQ(x, runlength, problemRng, seed)

% INPUTS
% x: a column vector containing the coordinates of the ambulances, (x1, y1, x2, y2, x3, y3)
% runlength: the number of hours of simulated time to simulate
% problemRng: a cell array of RNG streams for the simulation model 
% seed: the index of the first substream to use (integer >= 1)

% RETURNS: Mean response time, no var or gradient estimates.

%   ****************************************
%   *** Code written by German Gutierrez ***
%   ***         gg92@cornell.edu         ***
%   ***                                  ***
%   *** Updated by Shane Henderson to    ***
%   *** use standard calling and random  ***
%   *** number streams                   ***
%   *** Edited by Jennifer Shih          ***
%   ***       jls493@cornell.edu         ***
%   *** Edited by Bryan Chong            ***
%   ***       bhc34 @cornell.edu         ***
%   ****************************************
% Last updated November 22, 2019

nAmbulances = 3;                  % # of ambulances

FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

if (sum(x < 0) > 0) || (sum(x > 1) > 0) || (sum(size(x) ~= [1, 2 * nAmbulances])>0) || (runlength <= 0) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('x (row vector with %d components)\nx components should be between 0 and 1\nrunlength should be positive and real,\nseed should be a positive integer\n', nAmbulances*2);
    fn = NaN;
    FnVar = NaN;
    
else % main simulation
    
    % Set up simulation parameters
    lambda = 1/60;                       % rate of call arrival
    velfk = 60; velsk = 40;             % Velocities in kilometers
    vf = velfk/30; vs = velsk/30;       % Velocities in terms of the unit square
    bases = zeros(nAmbulances, 2);  % Base locations
    
    for i = 1:nAmbulances
        bases(i, :) = x(2*i-1:2*i);
    end
    mus = 45/60; sigmas = 15/60;              % mean and StdDev. of  service times ~ Gamma.
    
    % Get random number streams from the inputs
    ArrivalTimeStream = problemRng{1};
    LocStream = problemRng{2};
    SceneTimeStream = problemRng{3};

    % Set the substreams for the arrival time, locations, and scene times
    ArrivalTimeStream.Substream = seed;
    LocStream.Substream = seed;
    SceneTimeStream.Substream = seed;
    
    % INSTEAD OF INDIVIDUAL REPLICATIONS, THIS SIMULATION MODEL RUNS FOR A
    % LONGER TIME --> ONLY ONE SEED IS NEEDED.
    
    % Generate vector of call arrival times. (used to calculate service time per call).
    RandStream.setGlobalStream(ArrivalTimeStream);
    InterarrivalTimes = exprnd(1/lambda, 1, 30*runlength);
    CallTimes = cumsum(InterarrivalTimes);
    Tmax = CallTimes(30*runlength); % German's code uses the time to finish, in addition to the number of calls
    
    NumCalls = 30*runlength;
    
    % Generate scene times
    RandStream.setGlobalStream(SceneTimeStream);
    Serv = gamrnd((mus/sigmas)^2,mus/(mus/sigmas)^2, 1, NumCalls);
    
    % Generate Call Locations - use acceptance rejection
    RandStream.setGlobalStream(LocStream);
    CallLocations = zeros(2, NumCalls);
    i = 1;
    while i~=NumCalls + 1
        u = rand(3,1);
        if 1.6 * u(3) <= 1.6-(abs(u(1)-.8)+abs(u(2)-.8))
            CallLocations(:,i) = u(1:2);
            i = i + 1;
        end
    end
    
    ExitTimes=zeros(1, NumCalls);                 % keeps track of time at which service for each call is completed.
    AmbulanceArrivalTimes=zeros(1, NumCalls);     % keeps track of time at which ambulance arrived to each call.
    
    % The following matrix will contain updated information of the location of
    % last call responded by each ambulance, i.e. col.1=Time at which last
    % call was finished (travel+service) for ambulance 1, col. 2= X location of
    % last call, col.3= Y location of last call, col. 4= X location of its base
    % and col. 5= Y location of its base.
    
    Ambulances = zeros(nAmbulances,5);
    Ambulances(:,4:5) = bases;
    Ambulances(:,2:3) = bases;
    
    % Loop through all calls, assign the ambulance that will respond to it and
    % update the finish time for the given ambulance. To do this, we must look
    % at available ambulances (if any) and assign the closest one. Else, the
    % next available ambulance will respond to the call.
    
    % DistanceToCall;       %keeps track of distance between ambulance and present call
    % xcurrent;             %current x location of an Ambulance (at time of present call)
    % ycurrent;             %current y location of an Ambulance (at time of present call)
    % minTime;              %keeps track of time at which next ambulance will be available(if all currently busy)
    % closestA;             %keeps track of closest Ambulance (out of the ones available)
    % minDistance;          %minimum distance to call out of available ambulances
    % xcall; ycall;         %keeps location of call
    % xlc; ylc;             %keeps location of last call serviced
    % xb; yb;               %keeps location of an ambulance's base
    for i = 1:NumCalls
        closestA = 0;
        minDistance = 1000000;
        xcall = CallLocations(1,i);
        ycall = CallLocations(2,i);
        for j = 1:nAmbulances
            % Check if ambulance j is available, if so calculate how far it is
            % from the call. Keep track of closest available ambulance
            if(Ambulances(j,1)<CallTimes(i))
                xlc = Ambulances(j,2);
                ylc = Ambulances(j,3);
                xb = Ambulances(j,4);
                yb = Ambulances(j,5);
                % Enough time between current and last call for ambulance to be
                % at base
                if(CallTimes(i)-Ambulances(j,1)>(abs(xlc-xb)+abs(ylc-yb))/vs)
                    DistanceToCall = abs(xcall-xb) + abs(ycall-yb);
                    % Ambulances at y-location of base, still traveling towards
                    % x-location of base.
                elseif (CallTimes(i) - Ambulances(j,1) > abs(ylc-yb)/vs)
                    % To discover the horizontal direction of movement
                    % (right/left).
                    if(xb - xlc > 0)
                        xcurrent = xlc + vs*(CallTimes(i) - Ambulances(j,1) - abs(ylc - yb)/vs);
                        DistanceToCall = abs(ycall - yb) + abs(xcall - xcurrent);
                    else
                        xcurrent = xlc - vs*(CallTimes(i) - Ambulances(j,1) - abs(ylc - yb)/vs);
                        DistanceToCall = abs(ycall - yb) + abs(xcall - xcurrent);
                    end
                    % Ambulance still trying to get to y-location of base.
                else
                    % To discover the vertical direction of movement (up/down)
                    if(yb - ylc > 0)
                        ycurrent = ylc + vs*(CallTimes(i) - Ambulances(j,1));
                        DistanceToCall = abs(ycall - ycurrent) + abs(xcall - xlc);
                    else
                        ycurrent = ylc - vs*(CallTimes(i) - Ambulances(j,1));
                        DistanceToCall = abs(ycall - ycurrent)+abs(xcall - xlc);
                    end
                end
                % If ambulance closer than the closest available ambulance so
                % far, keep track of it.
                if(DistanceToCall < minDistance)
                    closestA = j;
                    minDistance = DistanceToCall;
                end
            end
        end
        % If there is an available ambulance, dispatch it. I.e. set its last
        % call to be this one, update its following finish time, x and y
        % locations of last call serviced.
        if minDistance ~= 1000000
            ExitTimes(i) = CallTimes(i) + minDistance/vf+Serv(i);
            AmbulanceArrivalTimes(i) = CallTimes(i) + minDistance/vf;
            Ambulances(closestA,1) = CallTimes(i) + minDistance/vf + Serv(i);
            Ambulances(closestA,2) = xcall;
            Ambulances(closestA,3) = ycall;
        else
            % No available ambulances, therefore the next available ambulance
            % will service the call.
            minTime = Tmax + 10000;
            for j = 1:nAmbulances
                % Find next available ambulance
                if(Ambulances(j,1) < minTime)
                    minTime = Ambulances(j,1);
                    ClosestA = j;
                end
            end
            % Update the next finish time, x and y locations of last call
            % serviced for ambulance that will respond to the call.
            ExitTimes(i) = minTime+((abs(Ambulances(ClosestA,2) - xcall) + abs(Ambulances(ClosestA,3) - ycall))/vf) + Serv(i);
            AmbulanceArrivalTimes(i) = minTime + ((abs(Ambulances(ClosestA,2) - xcall) + abs(Ambulances(ClosestA,3) - ycall))/vf);
            Ambulances(ClosestA,1) = minTime + ((abs(Ambulances(ClosestA,2) - xcall) + abs(Ambulances(ClosestA,3) - ycall))/vf) + Serv(i);
            Ambulances(ClosestA,2) = xcall;
            Ambulances(ClosestA,3) = ycall;
        end
    end
    
    % Calculate AvgResponseTime and Standard deviation. Use Ambulance arrival
    % times to calls - time of call.
    fn = mean(AmbulanceArrivalTimes - CallTimes);
    FnVar = var(AmbulanceArrivalTimes - CallTimes);
end % if input parameters are valid