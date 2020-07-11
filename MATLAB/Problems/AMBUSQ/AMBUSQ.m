function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = AMBUSQ(x, runlength, problemRng, seed)

% INPUTS
% x: a column vector containing the coordinates of the ambulances, (x1, y1, x2, y2, x3, y3)
% runlength: the number of days (each day is 24 hours) of simulated time to simulate
% problemRng: a cell array of RNG streams for the simulation model 
% seed: the index of the first substream to use (integer >= 1)

% RETURNS: Sample mean and variance of the daily mean response time, no gradient estimates.

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
%   *** Edited by Shane Henderson        ***
%   ***       sgh9@cornell.edu           ***
%   *** to                               ***
%   *** run as a terminating simulation  ***
%   *** where each day is one rep.       ***
%   *** Ambs start out avail at base.    ***
%   ****************************************

%   ****************************************
% Last updated February 3, 2020

nAmbulances = 3;                  % # of ambulances

FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;
% Square is normalized to lie within interval [0, 1]^2 so amb coords should
% be in [0, 1]^2.
if (sum(x < 0) > 0) || (sum(x > 1) > 0) || (sum(size(x) ~= [1, 2 * nAmbulances])>0) || (runlength <= 0) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('x (row vector with %d components)\nx components should be between 0 and 1\nrunlength should be positive and real,\nseed should be a positive integer\n', nAmbulances*2);
    fn = NaN;
    FnVar = NaN;
    
else % main simulation
    
    % Set up simulation parameters
    lambda = 1/60;                       % rate of call arrival. 1 call per hour
    velfk = 60; velsk = 40;             % Velocities in kilometers per hr
    vf = velfk/30; vs = velsk/30;       % Velocities scaled to relate to the unit square
    bases = zeros(nAmbulances, 2);  % Base locations
    
    for i = 1:nAmbulances
        bases(i, :) = x(2*i-1:2*i);
    end
    mus = 45/60; sigmas = 15/60;              % mean and StdDev. of  service times ~ Gamma.
    
    % Get random number streams from the inputs
    ArrivalTimeStream = problemRng{1};
    LocStream = problemRng{2};
    SceneTimeStream = problemRng{3};

    % Set up array to contain results from each day/replication
    nDays = runlength;
    Daily_Mean_Response_Time = zeros(1,nDays);
    
    for replication = 1:nDays
                
        % Set the substreams for the arrival time, locations, and scene times
        % Replication i uses substream seed + replication - 1
        ArrivalTimeStream.Substream = seed + replication - 1;
        LocStream.Substream = seed + replication - 1;
        SceneTimeStream.Substream = seed + replication - 1;
     
        % Generate vector of call arrival times. (used to calculate service time per call).
        % Units of interarrival times are minutes
        % We generate twice as many calls as are typically needed for one
        % set of calls during the day (to be safe), then throw away the extras so that
        % we just track those calls that arrive within 24 hours
        RandStream.setGlobalStream(ArrivalTimeStream);
        InterarrivalTimes = exprnd(1/lambda, 1, ceil(lambda * 60 * 24 * 2));
        CallTimes = cumsum(InterarrivalTimes);
        Tmax = 60 * 24; % Last call is before time 1440 minutes = 24 hours
        CallTimes = CallTimes(CallTimes <= Tmax); % Drop any calls that arrive after end of day
    
        NumCalls = length(CallTimes);
    
        % Generate scene times
        RandStream.setGlobalStream(SceneTimeStream);
        Serv = gamrnd((mus/sigmas)^2,mus/(mus/sigmas)^2, 1, NumCalls);
    
        % Generate Call Locations - use acceptance rejection
        RandStream.setGlobalStream(LocStream);
        CallLocations = zeros(2, NumCalls);
        j = 1;
        while j~=NumCalls + 1
            u = rand(3,1);
            if 1.6 * u(3) <= 1.6-(abs(u(1)-.8)+abs(u(2)-.8))
                CallLocations(:,j) = u(1:2);
                j = j + 1;
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
                    if(CallTimes(i)-Ambulances(j,1)>(abs(xlc-xb)+abs(ylc-yb))/vs)
                    % Enough time between current and last call for ambulance to be
                    % at base
                        DistanceToCall = abs(xcall-xb) + abs(ycall-yb);
                    elseif (CallTimes(i) - Ambulances(j,1) > abs(ylc-yb)/vs)
                        % Ambulance at y-location of base, still traveling towards
                        % x-location of base. We assume y travel comes
                        % first
                        % To discover the horizontal direction of movement
                        % (right/left).
                        if(xb - xlc > 0)
                            xcurrent = xlc + vs*(CallTimes(i) - Ambulances(j,1) - abs(ylc - yb)/vs);
                            DistanceToCall = abs(ycall - yb) + abs(xcall - xcurrent);
                        else
                            xcurrent = xlc - vs*(CallTimes(i) - Ambulances(j,1) - abs(ylc - yb)/vs);
                            DistanceToCall = abs(ycall - yb) + abs(xcall - xcurrent);
                        end
                    else
                        % Ambulance still trying to get to y-location of base.
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
                end % If this ambulance is available
            end %for j looping over all ambulances to find closest
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
                minTime = Tmax + 1000000;
                for j = 1:nAmbulances
                    % Find next available ambulance
                    if(Ambulances(j,1) < minTime)
                        minTime = Ambulances(j,1);
                        ClosestA = j;
                    end
                end
                % Update the next finish time, x and y locations of last call
                % serviced for ambulance that will respond to the call.
                % Ambulance responds from its previous call
                AmbulanceArrivalTimes(i) = minTime + ((abs(Ambulances(ClosestA,2) - xcall) + abs(Ambulances(ClosestA,3) - ycall))/vf);
                ExitTimes(i) = AmbulanceArrivalTimes(i) + Serv(i);
                Ambulances(ClosestA,1) = ExitTimes(i);
                Ambulances(ClosestA,2) = xcall;
                Ambulances(ClosestA,3) = ycall;
            end % No available ambulances
        end % for i (one day of simulation)
        Daily_Mean_Response_Time(replication) = mean(AmbulanceArrivalTimes - CallTimes);
    end % for replication
    
    % Calculate statistics
    fn = mean(Daily_Mean_Response_Time);
    FnVar = var(Daily_Mean_Response_Time)/nDays;
end % if input parameters are valid