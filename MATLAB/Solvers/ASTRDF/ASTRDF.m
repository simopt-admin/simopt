%==========================================================================
% ASTRO-DF (Adaptive Sampling Trust Region Optimization - Derivative Free)
%==========================================================================
% DATE
%        Spring 2020
%
% AUTHOR
%        Pranav Jain, Yunsoo Ha, Sara Shashaani, Kurtis Konrad.
%
% REFERENCE		
%        Sara Shashaani, Fatemeh S. Hashemi and Raghu Pasupathy (2018)
%		 ASTRO-DF: A Class of Adaptive Sampling Trust Region Algorithms
%        for Derivative-Free Stochastic Optimization 28(4):3145-3176
%
%==========================================================================
%
% INPUT
%        problem
%              Problem function name
%        probstructHandle
%              Problem structure function name 
%        problemRng
%              Random number generators (streams) for problems
%        solverRng
%              Random number generator (stream) for solver
%        numBudget
%              number of budgets to record, >=3; the spacing between
%              adjacent budget points should be about the same
%
%
% OUTPUT
%        Ancalls
%              An array (size = 'NumSoln' X 1) of budget expended
%        Asoln
%              An array (size = 'NumSoln' X 'dim') of solutions
%              returned by solver
%        Afn
%              An array (size = 'NumSoln' X 1) of estimates of expected
%              objective function value
%        AFnVar
%              An array of variances corresponding to
%              the objective function at A
%              Equals NaN if solution is infeasible
%        AFnGrad
%              An array of gradient estimates at A; not reported
%        AFnGardCov
%              An array of gradient covariance matrices at A; not reported
%        Aconstraint
%              A vector of constraint function estimators; not applicable
%        AConstraintCov
%              An array of covariance matrices corresponding to the
%              constraint function at A; not applicable
%        AConstraintGrad
%              An array of constraint gradient estimators at A; not
%              applicable
%        AConstraintGradCov
%              An array of covariance matrices of constraint gradient
%              estimators at A; not applicable
%
%==========================================================================

%% ASTRO-DF
function [Ancalls, Asoln, Afn, AFnVar, AFnGrad, AFnGradCov, ...
    Aconstraint, AConstraintCov, AConstraintGrad, AConstraintGradCov] ...
    = ASTRDF(probHandle, probstructHandle, problemRng, solverRng)

    %% Unreported
    AFnGrad = NaN;
    AFnGradCov = NaN;
    Aconstraint = NaN;
    AConstraintCov = NaN;
    AConstraintGrad = NaN;
    AConstraintGradCov = NaN;

    % Separate the two solver random number streams
    solverInitialRng = solverRng{1}; % RNG for finding initial solutions
    
    % Get details of the problem and random initial solution
    RandStream.setGlobalStream(solverInitialRng);
    
    [minmax, dim, ~, ~, VarBds, ~, ~, x0, budgetmax, ~, ~, ~] = probstructHandle(1);
    %get an initial solution
    budgetini=min(30,max(floor(0.001*budgetmax)+1,5));
    [xini, varini, ~, ~, ~, ~, ~, ~] = probHandle(x0, budgetini, problemRng, 1);
    budgetmaxPT = budgetmax;

    % Set the sampling rule
    type = 3; % an integer value from 1 to 4
    % See the subfunction 'Adaptive_Sampling' for more details

    Delta_max = norm(VarBds(:,2)-VarBds(:,1),inf); % maximum acceptable trust region radius
    if Delta_max==inf
        Delta_max=100;
    end

    Delta0 = .08*Delta_max; % initial trust region radius 
    shrink=.5^log(dim+1);
    expand=1/shrink;
    radiustry=[1 shrink expand]; %fraction of initial trust region to use for parameter tuning  
    Delta1 = [Delta0 Delta0*shrink Delta0*expand];
    setupprct=.03; %the initial percentage of the budget to use for finding a good initial radius

    ptdict=[];
    ptdict_par = [];
    x_points=[];
    callcount=[];
    func_points =[];
    var_points = [];
    calls=0;
    funcbest=zeros(size(radiustry));
    xbest=zeros(length(radiustry),dim);    
    point_precision = 7; %number of decimal places to keep for any points in ptdict

    ptdict = struct('pts',[x0],'counts',[0],'means',[0],'variances',[0],...
        'rands',[1],'decimal',point_precision, 'iterationNumber', [0 0 0]);

    for i=1:length(radiustry)
        %try to run the algorithm on setupprct of the budget at different
        %fractions of the initial suggested radius
         [callcounti, x_pointsi, func_pointsi, var_pointsi, ptdict]= ASTRDF_Internal(...
            probHandle, problemRng, solverRng, minmax, dim, VarBds, x0, ...
            floor(setupprct*budgetmaxPT), type, Delta1(i), Delta_max, ptdict, 1, 0);        
        
        
        infoi=ptdict.info;
        ptdict=rmfield(ptdict,'info'); %only save the points
        x_pointsi_{i} = x_pointsi;
        func_pointsi_{i} = func_pointsi;
        callcounti_{i} = callcounti;
        var_pointsi_{i} = var_pointsi;
 
        calls=calls+infoi.calls; %total number of calls
                
        if ~isempty(func_pointsi) %if the attempt had a successful iteration
            %use the last point
            funcbest(i)=func_pointsi(end);
            xbest(i,:)=x_pointsi(end,:);            
            Delta_par(i) = ptdict.PTinfo(i).Delta;
        else
            if minmax==-1 %minimzation
                funcbest(i)=Inf; %no success means value was Inf
            elseif minmax==1
                funcbest(i)=0;
            end
            xbest(i,:)=x0;
            Delta_par(i) = Delta1(i);
        end
    end

    %pick the best value from the trials
    funcbest=-1*minmax*funcbest;
    if minmax==-1
        [bestval,best]=min(funcbest);
    elseif minmax==1
        [bestval,best]=max(funcbest);
    end
    
    BestS = 0;
    
    for i = 1:3
        if best == i
            BestS = i;    
            x_aft_tune = xbest(i,:);
            Delta = Delta_par(i);
            x_points_par = x_pointsi_{i};
            func_points_par = cell2mat(func_pointsi_(i));
            callcount_par = cell2mat(callcounti_(i)) + budgetini + (2*floor(setupprct*budgetmaxPT));
            var_points_par = cell2mat(var_pointsi_(i));
            break
        end
    end
    %budgetmax = budgetmax - budgetini - calls;
    
    %run the main algorithm
    [callcount, x_points, func_points, var_points]= ASTRDF_Internal(probHandle,...
        problemRng, solverRng, minmax, dim, VarBds, x_aft_tune, budgetmaxPT-3*floor(setupprct*budgetmaxPT), type, Delta, Delta_max, ptdict, 0, BestS);
    
    callcount = callcount+3*floor(setupprct*budgetmaxPT);
    
    %record points for new SIMOPT format Jan 2020
     Asoln = [x0; x_points_par; x_points];
     Afn = [xini; func_points_par; func_points];
     AFnVar = [varini; var_points_par; var_points];
     Ancalls = [budgetini; callcount_par; callcount];

end


function [callcount, x_points, func_points, var_points, ptdict] ...
    = ASTRDF_Internal(probHandle, problemRng, solverRng, ...
        minmax, dim, VarBds, xk, budgetmax, type, Delta0, Delta_max, ptdict, PT, BestS)
%ASTRDF_Internal runs the main portion of the ASTRO-DF Algorithm
%
%   INPUTS
%
%   probHandle = the problem handle
%   problemRng = Random number generators (streams) for problems
%   solverRng = Random number generator (stream) for solver
%   minmax = +/- 1 whether the goal is minimization or maximization
%          = 1 if problem is being maximized
%          =-1 if objective is being minimized
%   dim = the dimension of the problem
%   VarBds = dim x 2 array of the lower and upper bounds for each input
%               variable dimension
%   x0 = the initial point that the algorithm uses
%   budgetmax = the maximum number of function calls allowed
%   type = an integer used to determine the adaptive sample size rule
%               see Adaptive_Sampling function for details
%   Delta0 = the initial trust region radius size
%   Delta_max = the maximum allowed trust region radius
%   ptdict = (optional) the data dictionary structure that keeps track of visited points
%           .pts = the data points that have been visited
%           .counts = the number of times the function has been called at
%                           that point
%           .means = the mean values that the function has been called at a
%                           point
%           .variances = the variance values of the function at a point
%           .rands = the next random seed to use at a point
%           .decimal = the number of places after the decimal point to
%                           round values off
%           (optional inclusions for warmstarting)
%           .x_points = the visited incumbent solutions
%           .callcount = the number of calls at the incumbent solutions
%           .func_points = the mean function values at the incumbent solutions
%           .var_points = the variances at the incumbent solutions
%           .info = information structure with fields:
%                   .calls = the number of calls already made
%                   .iteration_number = the number of iterations already made
%   PT = the binary variable 
%        if PT = 0, main run
%        if PT = 1, run for parameter tuning
%
%
%   OUTPUTS
%
%   callcount = an array of the number of calls needed to reach the
%                   incumbent solutions
%   x_points = an array of the incumbent solutions
%   func_points = the estimated function value at the incumbent solutions
%   var_points = the estimated variance at the incumbent solutions
%   ptdict = a data dictionary structure.  It contains the same information
%                   as before with these additional fields:
%           .info.delta = the current radius
%           .info.gradnorm = the norm of the current gradient
%           .sensitivity = the sensitivity of the variable bounds

    x_init = xk;
    % Separate the two solver random number streams
    solverInitialRng = solverRng{1}; % RNG for finding initial solutions
    solverInternalRng = solverRng{2}; % RNG for the solver's internal randomness

    % Generate new starting point x0 (it must be a row vector)
    RandStream.setGlobalStream(solverInitialRng);

    % More default values
    eta_1 = 0.10;           %threshhold for decent success
    eta_2 = 0.50;           %threshhold for good success
    w = 0.99; 
    mu = 1.05; 
    beta = 1/mu;
    gamma_1 = (1.25)^(2/dim);  %successful step radius increase
    gamma_2 = 1/gamma_1;    %unsuccessful step radius decrease
  
    %create the output variables or load them if available
    %if nargin< 12  || ~isfield(ptdict,'info')

    x_points=[];
    callcount=[];
    func_points =[];
    var_points = [];
    %Initializations
    calls = 0;
    
    if PT == 1
        iteration_number = 1;
    else
        iteration_number = ptdict.iterationNumber(BestS);
    end

    % Shrink VarBds to prevent floating errors
    %following STRONG code
    sensitivity = 10^(-5); % shrinking scale for VarBds
    VarBds(:,1) = VarBds(:,1) + sensitivity; 
    VarBds(:,2) = VarBds(:,2) - sensitivity;
    ptdict.sensitivity = sensitivity;
    
    Delta = Delta0;
    while calls <= budgetmax
        o = 100;
        if Delta > Delta_max/o       %if Delta > 1.2
            lin_quad = 1; %run a linear model
        else
            lin_quad = 2; %run a quadratic model
        end
        
        %run the adaptive sampling part of the algorithm
        [q, Fbar, Fvar, Deltak, calls, ptdict, budgetmax] = Model_Construction(probHandle, xk, Delta, iteration_number, ...
            type, w, mu, beta, calls, solverInternalRng, problemRng, minmax, ptdict, VarBds,lin_quad,budgetmax, PT, BestS);
        
        % Record Fbar
        x_incumbent = xk;
        Fbar_incumbent = Fbar(1);
        Fvar_incumbent = Fvar(1);

        % Step 3
        % Minimize the constrained model
        fun = @(x)Model_Approximation(x-xk, lin_quad, q);
        nonlcon = @(x)disk(x, xk, Deltak);
        [~,~,H,~,~] = fun(xk);
        
        hessint = @(x,lambda) hessinterior(x,lambda,H);
        options.HessianFcn = hessint;
        options.OptimalityTolerance = Fvar_incumbent*0.1;
        
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = max(xk-Deltak,VarBds(:,1)');
        ub = min(xk+Deltak,VarBds(:,2)');
        maxfuncevals_fmincon = 1000;
        C_Tol = 1e-12;
        S_Tol = 1e-20;
        
        options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', ...
                    'SpecifyObjectiveGradient', true, 'SpecifyConstraintGradient',true,...
                    'MaxFunctionEvaluations', maxfuncevals_fmincon, 'ConstraintTolerance', C_Tol, ...
                    'StepTolerance', S_Tol);
                
        [x_tilde, ~, exitflag, ~] = fmincon(fun, xk, A, b, Aeq, beq, lb, ub, nonlcon, options); 
        
        %Step 4
        %load in the current point's information if it exists already in the
        %point dictionary
        [Result,LocResult] = ismember(round(x_tilde,ptdict.decimal),ptdict.pts,'rows');
        if Result == true
            Fb = ptdict.means(LocResult);
            sig2 = ptdict.variances(LocResult);
            samplecounts = ptdict.counts(LocResult);
            problemseed = ptdict.rands(LocResult); %set to global random seed for use in the function
            if PT == 1
                budgetmax = budgetmax - samplecounts;
            end
        else
            Fb = 0;
            sig2 = 0;
            samplecounts = 0; 
            %set to global random seed for use in the function
            % Using CRN: for each solution, start at substream 1 for problemRng
            problemseed = 1; % Reset seed to get CRN across solutions
        end

        %sample the point enough times
        while 1
            if samplecounts >= Adaptive_Sampling(iteration_number, sig2, Deltak, type)
                break
            else
                if samplecounts > 2 && calls > budgetmax
                    break
                end
                samplecounts = samplecounts + 1;
            end
            [xi, ~, ~, ~, ~, ~, ~, ~] = probHandle(x_tilde, 1, problemRng, problemseed);
            xi = -minmax*xi; % Account for minimization/maximization 
            problemseed = problemseed + 1; % iid observations
            calls = calls + 1;
            F_old = Fb;
            Fb = (samplecounts-1)/samplecounts*Fb + 1/samplecounts*xi; %update mean
            sig2 = (samplecounts-2)/(samplecounts-1)*sig2 + samplecounts*(Fb-F_old)^2; %update variance
            if samplecounts == 1
                sig2 = 0;
            end
        end
        
        if calls > budgetmax 
            if PT == 1
                if Delta0 == .08*Delta_max
                    ptdict.iterationNumber(1) = iteration_number;
                    ptdict.PTinfo(1).Delta = Delta;
                elseif Delta0 < .08*Delta_max
                    ptdict.iterationNumber(2) = iteration_number;
                    ptdict.PTinfo(2).Delta = Delta;
                else
                    ptdict.iterationNumber(3) = iteration_number;
                    ptdict.PTinfo(3).Delta = Delta;
               end                          
            end
        end
        
        Fbar_tilde = Fb;
        Fvar_tilde = sig2;

        %save the information to the point dictionary
        if Result == false
            ptdict.pts = [ptdict.pts; round(x_tilde,ptdict.decimal)];
            ptdict.means = [ptdict.means; Fb];
            ptdict.counts = [ptdict.counts; samplecounts];
            ptdict.variances = [ptdict.variances; sig2];
            ptdict.rands = [ptdict.rands; problemseed]; 
        else
            ptdict.means(LocResult) = Fb;
            ptdict.variances(LocResult) = sig2;
            ptdict.counts(LocResult) = samplecounts;
            ptdict.rands(LocResult) = problemseed;
        end
        
        if Fbar_tilde > min(Fbar)
            Fbar_tilde = min(Fbar);
            x_tilde = ptdict.pts(Fbar_tilde == ptdict.means,:);
        end 
        
        % Step 5 - Model Accuracy
        rho = (Fbar(1) - Fbar_tilde)/ (Model_Approximation(xk-xk,lin_quad,q) - Model_Approximation(x_tilde-xk,lin_quad,q));

        % Step 6 - Trust Region Update step
        if rho >= eta_2 %really good accuracy
            xk = x_tilde;
            Delta = min(gamma_1*Deltak, Delta_max); %expand trust region
            x_points = [x_points;x_tilde];
            callcount = [callcount; calls];
            func_points =[func_points; Fbar_tilde];
            var_points = [var_points; Fvar_tilde];
        elseif rho >= eta_1 %good accuracy
            xk = x_tilde;
            Delta = min(Deltak, Delta_max); %maintain same trust region size 
            x_points = [x_points; x_tilde];
            callcount = [callcount; calls];
            func_points = [func_points; Fbar_tilde];
            var_points = [var_points; Fvar_tilde];
        else %poor accuracy
            Delta = min(gamma_2*Deltak, Delta_max); %shrink trust region
        end 
       
        [~,currentgrad] = Model_Approximation(xk-xk,lin_quad,q);
        iteration_number = iteration_number + 1;        
    end

    %save final information before exiting
    if isempty(x_points) == false
        x_points = [x_points;x_points(end,:)];
        callcount = [callcount; calls];
        func_points = [func_points; func_points(end)];
        var_points = [var_points; var_points(end)];
    end
    info.iteration_number = iteration_number;
    info.delta = Delta;
    info.calls = calls;
    info.gradnorm = norm(currentgrad);
    ptdict.info = info;
end

function [q, Fbar, Fvar, Deltak, calls, ptdict, budgetmax] = Model_Construction(probHandle, x, Delta, k, type, ...
    w, mu, beta, calls, solverInternalRng, problemRng, minmax, ptdict,VarBds,lin_quad, budgetmax, PT, BestS)
%Model_Construction creates a model in the trust region
%
%   INPUTS
%   
%   probHandle = the problem handle
%   x = the center point that the algorithm builds the model around
%   Delta = the trust region radius size
%   k = the iteration number
%   type = an integer used to determine the adaptive sample size rule
%               see Adaptive_Sampling function for details
%   w = trust region contraction coefficient
%   mu = determine if trust region is sufficiently small
%   beta = trust region contraction coefficient (a different one)
%   calls = the number of calls already made
%   solverInternalRng = Random number generator (stream) for internal solver
%   problemRng = Random number generators (streams) for problems
%   minmax = +/- 1 whether the goal is minimization or maximization
%          = 1 if problem is being maximized
%          =-1 if objective is being minimized
%   ptdict = (optional) the data dictionary structure that keeps track of visited points
%           .pts = the data points that have been visited
%           .counts = the number of times the function has been called at
%                           that point
%           .means = the mean values that the function has been called at a
%                           point
%           .variances = the variance values of the function at a point
%           .rands = the next random seed to use at a point
%           .decimal = the number of places after the decimal point to
%                           round values off
%   VarBds = dim x 2 array of the lower and upper bounds for each input
%               variable dimension
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%   budgetmax = the maximum number of function calls allowed
%   PT = the binary variable 
%        if PT = 0, main run
%        if PT = 1, run for parameter tuning
%
%   OUTPUTS
%
%   q = the coefficient vector for the model
%   Fbar = the matrix of function values
%   Fvar = the matrix of function variances
%   Deltak = the expanded or contracted radius size
%   calls = the number of calls already made
%   ptdict = a data dictionary structure.  It contains the same information as before.


    j = 1;
    while 1
        Deltak = Delta*w^(j-1);
        
        %get the set of points to use
        Y = Interpolation_Points(x, Deltak, solverInternalRng, ptdict, k+j/2,VarBds,lin_quad);

        %build the model
        [~,~,~,A] = Model_Approximation(Y-Y(1,:),lin_quad);
        if rcond(A) < eps^.8 %if the model is not well poised, try again
            %on occassion, the algorithm has a problem building a set.  When it
            %trys again, it seems to have no problem
            Y = Make_Lambda_Poised(Y,Delta,50,lin_quad,VarBds,ptdict.decimal);
            [~,~,~,A] = Model_Approximation(Y-Y(1,:),lin_quad);
            if rcond(A) < eps^.95
                warning("The Poisedness Improvement Algorithm generated a poorly conditioned Vandermonde Matrix.")
            end
        end
                 
        if PT == 0 
            if k == ptdict.iterationNumber(BestS)
                Y = ptdict.PTinfo(BestS).pts;
                [~,~,~,A] = Model_Approximation(Y-Y(1,:),lin_quad);
            end
        else           
            if ptdict.iterationNumber(1) == 0
                ptdict.PTinfo(1).pts = Y;
            elseif ptdict.iterationNumber(2) == 0
                ptdict.PTinfo(2).pts = Y;
            elseif ptdict.iterationNumber(3) == 0
                ptdict.PTinfo(3).pts = Y;
            end
        end
        
        p = size(Y, 1);
        Fbar = zeros(p, 1);
        Fvar = zeros(p, 1);
        ks_counts = zeros(p,1);
        stddev2 = zeros(p,1);
        randseeds = zeros(p,1);
        pts_exist = zeros(p,1);

        for i = 1:p %for each point in Y
            %if it already exists, recall the information
            [Result,LocResult] = ismember(round(Y(i,:),ptdict.decimal),ptdict.pts,'rows');
            if Result == true
                Fb = ptdict.means(LocResult);
                sig2 = ptdict.variances(LocResult);
                ks = ptdict.counts(LocResult);
                problemseed = ptdict.rands(LocResult); %set to global random seed for use in the function
                pts_exist(i) = 1;
                if PT == 1
                    budgetmax = budgetmax - ks;
                end   
            else
                Fb = 0;
                sig2 = 0;
                ks = 0;
                % Reset seed to get CRN across solutions
                problemseed = 1;  %set to global random seed for use in the function
            end

            while 1
                %continue until a good sample size is found
                if ks >= Adaptive_Sampling(k, sig2, Deltak, type)
                    break
                else
                    if ks>2 && calls>budgetmax
                        break
                    end
                    ks = ks + 1;
                end

                [xi, ~, ~, ~, ~, ~, ~, ~] = probHandle(Y(i,:), 1, problemRng, problemseed);
                xi = -minmax*xi; % Account for minimization/maximization
                problemseed = problemseed + 1; % iid observations  
                calls = calls + 1;
                F_old = Fb;
                Fb = (ks - 1)/ks*F_old + 1/ks*xi;
                sig2 = (ks - 2)/(ks - 1)*sig2 + ks*(Fb - F_old)^2;
                if ks == 1
                    sig2 = 0;
                end

            end
            Fbar(i) = Fb;
            Fvar(i) = sig2;

            % save the point information
            if pts_exist(i) == 0
                ks_counts(i) = ks;
                stddev2(i) = sig2;
                randseeds(i) = problemseed;
            else
                ptdict.means(LocResult) = Fb;
                ptdict.variances(LocResult) = sig2;
                ptdict.counts(LocResult) = ks;
                ptdict.rands(LocResult) = problemseed;
            end
        end

        %add the new points to the dictionary
        ptdict.pts = [ptdict.pts; round(Y(pts_exist==0,:),ptdict.decimal)];
        ptdict.means = [ptdict.means; Fbar(pts_exist==0)];
        ptdict.counts = [ptdict.counts; ks_counts(pts_exist==0)];
        ptdict.variances = [ptdict.variances; stddev2(pts_exist==0)];
        ptdict.rands = [ptdict.rands; randseeds(pts_exist==0)];

        %get model parameters
        q = A \ Fbar;

        if any(isnan(q))
            warnmsg = ['NAN values occurred for building model parameters.\n',...
                       'Please confirm that the problem correctly works for only one replication at a time.\n',...
                       'Hint: investigate the runlength variable in the problem code as ASTRDF sets runlength=1.'];   
            warning(sprintf(warnmsg))
        end
        
        j = j + 1;
        [~, grad0] = Model_Approximation(Y(1,:)-x,lin_quad, q);

        %check size of adaptive sample
        if Deltak <= mu*norm(grad0)
            break
        elseif calls > budgetmax            
            break
        end
    end
    %contract radius
    Deltak = min(max(beta*norm(grad0), Deltak), Delta);
end

function [Y] = Interpolation_Points(center, radius, solverInternalRng, ptdict, iterations,VarBds,lin_quad)
%   Interpolation_Points selects the initial points to be used for the adaptive sampling
%
%   INPUTS
%
%   center = the center point for the model
%   radius = the maximum allowed radius
%   solverInternalRng = Random number generator (stream) for internal solver
%   ptdict = (optional) the data dictionary structure that keeps track of visited points
%           .pts = the data points that have been visited
%           .counts = the number of times the function has been called at
%                           that point
%           .means = the mean values that the function has been called at a
%                           point
%           .variances = the variance values of the function at a point
%           .rands = the next random seed to use at a point
%           .decimal = the number of places after the decimal point to
%                           round values off
%   iterations = the number of iterations that have occurred
%   VarBds = dim x 2 array of the lower and upper bounds for each input
%               variable dimension
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%
%   OUTPUTS
%
%   Y = the set of points to use for adaptive sampling
%

    [~, dim] = size(center);
    %Next, for reusing points, we determine which points are acceptable and
    %which we want to use.  Then we generate any new points needed or
    %select from within the possible options.
    
    %find out which currently existing points are within the radius
    inside = vecnorm(ptdict.pts-center,2,2);
    inside(inside==0) = radius;
    inside = inside < radius;
    eligible = ptdict.pts(inside,:);
    RandStream.setGlobalStream(solverInternalRng);

    lambda = .05; %Adjustable >0
    Bi_prob = .85*(1-exp(-lambda*iterations))^dim;
    myrnd = rand(size(eligible,1),1);
    chosen = myrnd<Bi_prob;

    %get the remaining points needed and ensure lambda poisedness
    lambda_Poisedness = 100;
    Y = [center; eligible(chosen,:)];  
    while 1
        Y = Make_Lambda_Poised(Y,radius,lambda_Poisedness,lin_quad,VarBds,ptdict.decimal);
        belowbds=Y(Y<VarBds(:,1)'-ptdict.sensitivity/2);
        abovebds=Y(Y>VarBds(:,2)'+ptdict.sensitivity/2);
        if isempty(belowbds) == true && isempty(abovebds) == true
            break
        else
            Y(any(Y<VarBds(:,1)',2),:)=[];
            Y(any(Y>VarBds(:,2)',2),:)=[];
        end
    end
end

function numsamples = Adaptive_Sampling(k, sig2, Delta, type)
%   Adaptive_Sampling determines the adaptive sample size
%
%   INPUTS
%
%   k = the number of times the function has been sampled
%   sig2 = the sample variance at the point
%   Delta = the trust region radius
%   type = a parameter value used for selecting different methods of
%               adaptive sampling
%
%   OUTPUTS
%
%   numsamples = the minimum required number of samples to meet the criteria
%
%

    alphak = 1;
    
    lambdak = 10*log10(k)^1.5; 
    kappa = 10^3;
    numsamples=floor(max([2,lambdak,(lambdak*sig2)/((kappa^2)*Delta^(2*(1+1/alphak)))]));

    if type==1
        numsamples=20;
    elseif type==2
        numsamples=floor(max([2,k^1.2]));
    end
    %cap the number of samples
    if numsamples > 2500000
        numsamples = 2500000;
    end
end

function Y=Make_Lambda_Poised(Y,Delta,lambda,lin_quad,VarBds,digits)
%   Make_Lambda_Poised takes a set of points and creates a poised set
%
%   INPUTS
%
%   Y = the set of possible points to use.  The first point should be the
%           center point
%   Delta = the trust region radius
%   lambda = the poisedness requirements
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%   VarBds = dim x 2 array of the lower and upper bounds for each input
%               variable dimension
%   digits = the number of digits to round the points
%
%   OUTPUTS
%
%   Y = the poised set of points to use
%
%

    alpha=.01; %percentage of radius that must be a 'dead' zone

    x_k = Y(1,:)';
    %scale the points to be centered at 0
    [Y,scalefactor,vb] = shift_and_scale(Y,Delta,alpha,VarBds);
    Delta_tilde = Delta/scalefactor;  

    if Delta_tilde>1e6
        warning("The scaled point selection radius is excessively large.  This may or may not produce a problem.")
    end

    %now everything stays in the ball of r=Delta_tilde centered at the origin
    %create the set of points and the vandermonde matrix for the points
    [M,Y]=Algorithm_6_2(Y,Delta_tilde,alpha,lin_quad,vb);

    %improve the poisedness of the created set
    Y=Algorithm_6_3(Y,M,Delta_tilde,lambda,alpha,lin_quad,vb);

    %ensure uniqueness of points
    HELPME=unique(Y,'rows');
    if any(size(HELPME)~=size(Y))==1
        warning("The point selection Algorithm 6.3 produced a set of non-unique points.")
    end
    %correct the points back to the old location
    Y = undo_shift_and_scale(Y, x_k, scalefactor);
    Y=round(Y,digits);

end


function [Ytilde, scalefactor,vb] = shift_and_scale(Y,Delta,alpha,VarBds)
%   shift_and_scale takes a set of points and centers them at zero with a
%           radius of 1
%
%   INPUTS
%
%   Y = the set of points
%   Delta = the trust region radius
%   alpha = the minimum separation distance
%   VarBds = dim x 2 array of the lower and upper bounds for each input
%               variable dimension 
%
%   OUTPUTS
%
%   Ytilde = the shifted and scaled set of points
%   scalefactor = the amount the points were scaled down
%   vb = a scaled bounds structure
%       .lb = the lower bound for each variable
%       .ub = the upper bound for each variable

    x_0 = Y(1,:);
    vb.lb = max(x_0-Delta,VarBds(:,1)');
    vb.ub = min(x_0+Delta,VarBds(:,2)');
    Ytilde = Y - x_0;
    vb.lb=vb.lb-x_0;
    vb.ub=vb.ub-x_0;
    scalefactor = max(vecnorm(Ytilde'));
    %correct issues with scalefactor
    if scalefactor==0
        scalefactor=1;
    elseif scalefactor<alpha*Delta
        scalefactor=alpha*Delta;
    end
    Ytilde = Ytilde/scalefactor;
    vb.lb= vb.lb/scalefactor;
    vb.ub= vb.ub/scalefactor;
    
end

function Y = undo_shift_and_scale(Y, x_0, scalefactor)
% undo_shift_and_scale corrects the scaled set of points
%
%   INPUTS
%
%   Y = the set of points
%   x_0 = the center point
%   scalefactor = the amount to scale the points
%
%   OUTPUTS
%
%   Y = the corrected scaled set of points
%
%

    if size(x_0, 1) > size(x_0, 2)
        x_0 = x_0';
    end
    Y = Y*scalefactor;
    Y = Y + x_0;
    
end

function [M,Y] = Algorithm_6_2(Y, Delta,alpha,lin_quad,vb)
%   Algorithm_6_2 creates a poised set from a given set of points
%   This is Algorithm 6.2 in "Introduction to Derivative Free Optimization"
%   by Andrew Conn, Katya Scheinberg, and Luis Vicente
%
%   INPUTS
%
%   Y = the set of points
%   Delta = the trust region radius
%   alpha = the minimum allowable separation
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%   vb = a scaled bounds structure
%       .lb = the lower bound for each variable
%       .ub = the upper bound for each variable
%
%   OUTPUTS
%
%   M = the basis matrix for the set of points
%   Y = the selected set of points
%
%
%the assumption is that the points have already been centered at the origin
%   with a radius of Delta


    [p_ini,dim] = size(Y);
    if lin_quad==2
        q = (dim+1)*(dim+2)/2;
    elseif lin_quad==1
        q=dim+1;
    end
    M = eye(max(p_ini,q),q); % Initialize with the monomial basis

    digits = -floor(log10(alpha^3*Delta));
    digits = max(digits,5); %if the radius is very large, values will be round to 5 decimal places

    for i = 1:q 
        %ab_values= abs(M*U(i,:)'); %pick row M(i) of the pivot basis
        [~,~,~,X]=Model_Approximation(Y,lin_quad,zeros(q,1),false,false);
        mod_ab_values = abs(X*M(i,:)');
        mod_ab_values(1:i-1) = -1*ones(i-1,1);

        [value, j_i] = max(mod_ab_values); % Select the point in Y which maximizes the absolute value of the ith pivot polynomial
        value=round(value,digits);
        if j_i < i && i<=p_ini
            warning('Point selection error in Algorithm_6_2.m')
            j_i = i;
        end

        if value>0 && i<=p_ini
            % Swap the i and j_i entry of Y
            holder = Y(j_i,:);
            Y(j_i,:) = [];
            Y = [Y(1:i-1,:); holder; Y(i:end,:)];
        else
            % compute yi
            Y(i,:)=Maximize_Lagrange_poly(M(i,:), zeros(1,dim), Delta,100,[],Y(1:i-1,:),alpha,lin_quad,vb);
        end

        [~,~,~,lyi]=Model_Approximation(Y(i,:),lin_quad,zeros(q,1),false,false);
        div=M(i,:)*lyi';
        if div==0
            %sometimes the approximation produces numerical errors, in that
            %case, use an exact calculation
            Y(i,:)=Max_Lagrange_exactly(M(i,:), zeros(1,dim), Delta,100,[],Y(1:i-1,:),alpha,lin_quad,vb);
            [~,~,~,lyi]=Model_Approximation(Y(i,:),lin_quad,zeros(q,1),false,false);
            div=M(i,:)*lyi';
            if div==0
                error('The Lagrange polynomial divisor is zero in Algorithm_6_2.')
            end
        end
        M(i,:)=M(i,:)/(div); %normalization
        %orthogonalization
        for j=1:q
            if j~=i
                M(j,:)=M(j,:)-(M(j,:)*lyi')*M(i,:);
            end
        end

        if any(isnan(M))
            error("There is a situation of nan in Algorithm_6_2.")
        end
    end
    %only pick the necessary points
    M=M(1:q,1:q);
    Y=Y(1:q,:);

end

function [Y] = Algorithm_6_3(Y, M, Delta,lambda,alpha,lin_quad,vb)
%   Algorithm_6_3 takes a poised set and improves the poisedness
%   This is approximately Algorithm 6.3 in "Introduction to Derivative Free 
%   Optimization" by Andrew Conn, Katya Scheinberg, and Luis Vicente
%
%   INPUTS
%
%   Y = the set of poised points
%   M = the basis matrix for the points
%   Delta = the trust region radius
%   lambda = the desired poisedness
%   alpha = the minimum separation distance
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%   vb = a scaled bounds structure
%       .lb = the lower bound for each variable
%       .ub = the upper bound for each variable
%
%   OUTPUTS
%
%   Y = the selected set of points
%
%

    [lambdak,maxpoints,maxlagrange] = Lambda_Poised(Y,M,Delta,lin_quad,vb,[],alpha);
    %if the set has a problem, try again
    if isnan(lambdak) || lambdak> 10^10
        [M,Y] = Algorithm_6_2(Y,Delta,alpha,lin_quad,vb);
        [lambdak,maxpoints,maxlagrange] = Lambda_Poised(Y,M,Delta,lin_quad,vb,[],alpha);
    end

    dim = size(Y,2);
    if lin_quad == 2
        q = (dim+1)*(dim+2)/2;
    elseif lin_quad == 1
        q = (dim+1);
    end

    while lambdak > lambda
        [~,i_k] = max(maxlagrange);
        if i_k==1 || isempty(i_k) %if it wants to replace the center point
            if lambdak < 10*lambda
                break
            end
            [lambdak,maxpoints,maxlagrange] = Lambda_Poised(Y,M,Delta,lin_quad,vb,[],alpha,true);
            if isnan(lambdak)
                [M,Y] = Algorithm_6_2(Y,Delta,alpha,lin_quad,vb);
                [lambdak,maxpoints,maxlagrange] = Lambda_Poised(Y,M,Delta,lin_quad,vb,[],alpha,true);
            end
            i_k = find(maxlagrange>lambda,1,'last');
            if i_k==1 
                %if lambdak>10*lambda
                    %warning("The poisedness improvement algorithm 6.3 is trying to replace the center point.")
                %end
                break
            elseif isempty(i_k)
                %warning("The poisedness improvement algorithm 6.3 did not find any points greater than lambda.")
                break
            end
            %I do not believe this will ever happen, but I also do not know how
            %I can be sure that it will not happen.
        end

        yik_star = maxpoints(i_k,:);
        Y(i_k,:) = yik_star;
        [~,~,~,lyi] = Model_Approximation(Y(i_k,:),lin_quad,zeros(q,1),false,false);
        div = M(i_k,:)*lyi';
        if div==0
            error('The Lagrange polynomial divisor is zero.')
        end
        M(i_k,:) = M(i_k,:)/(div); %normalization
        %orthogonalization
        for j = 1:q
            if j~=i_k
                M(j,:) = M(j,:)-(M(j,:)*lyi')*M(i_k,:);

            end
        end

        %update lambdak
        [lambdak,maxpoints,maxlagrange] = Lambda_Poised(Y,M,Delta,lin_quad,vb,maxpoints,alpha);
        if isnan(lambdak)
            [M,Y] = Algorithm_6_2(Y,Delta,alpha,lin_quad,vb);
            [lambdak,maxpoints,maxlagrange] = Lambda_Poised(Y,M,Delta,lin_quad,vb,maxpoints,alpha);
        end
    end

end

function [argmax, improved_pivot_value] = Max_Lagrange_exactly(basis, center, radius, func_eval,x0,Y,alpha,lin_quad,vb,internal)
%   Max_Lagrange_exactly picks the point that maximizes the lagrange
%   polynomials using exact methods
%
%   INPUTS
%
%   basis = the basis vector
%   center = the center point (usually all zeros)
%   radius = the trust region radius
%   func_eval = the number of function evaluations to run in the solver
%   x0 = the initial potential starting solution, set to [] if no initial
%   Y = the set of points
%   alpha = the minimum percent separation distance
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%   vb = a scaled bounds structure
%       .lb = the lower bound for each variable
%       .ub = the upper bound for each variable
%   internal = boolean if the function is being called from itself
%                   (optional)
%
%   OUTPUTS
%   
%   argmax = the points that maximize the lagrange function
%   improved_pivot_value = the lagrange function values for the best point
%
%the assumption is that the points have already been centered at the origin
%with a radius of Delta
%


    if nargin<10 || internal ~= true
        %maximize the function, then minimize the function.  Pick the
        %larger in absolute value, and this maximizes abs(function())
        [argmax, maxval] = Max_Lagrange_exactly(basis, center, radius, func_eval,x0,Y,alpha,lin_quad,vb,true);
        [argmin, minval] = Max_Lagrange_exactly(-1*basis, center, radius, func_eval,x0,Y,alpha,lin_quad,vb,true);
        if isnan(minval) || isnan(maxval)
            improved_pivot_value=nan;
        elseif abs(maxval)>abs(minval)
            argmax=argmax;
            improved_pivot_value=maxval;
        else
            argmax=argmin;
            improved_pivot_value=minval;
        end
        return
    end
    
    digits=-floor(log10(alpha^3*radius)); %alpha cubed is used for rounding
    
    %center must be a row vector
    if lin_quad==2
        [~,~,H,~,xstar]=Model_Approximation(center-center,lin_quad,-1*basis',true,true);
    elseif lin_quad==1
        [~,g,H,~,~]=Model_Approximation(center-center,lin_quad,-1*basis',false,true);
        xstar=center+radius*g/norm(g);
    end
    
    %set up constrained minimization
    li_x=@(x) Model_Approximation(x-center,lin_quad,-1*basis',false,true);
    hessint=@(x,lambda) hessinterior(x,lambda,H);
    nonlcon = @(x)diskpoints(x, center, radius,Y,alpha);
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = vb.lb;
    ub = vb.ub;
    if isempty(x0)==true 
        if isempty(xstar)==false & all(nonlcon(xstar))<=0
                x0=xstar;
        else
            x0=center+(alpha^2+alpha)*radius;
        end
    end
    options = optimoptions('fmincon', 'Display', 'none', ...
        'Algorithm', 'interior-point', 'MaxFunctionEvaluations', func_eval,...
        'SpecifyConstraintGradient',true,...
        'SpecifyObjectiveGradient',true,...
        'StepTolerance',1e-4,...
        'HessianFcn',hessint);
    ef=0;
    tries=0;
    x_tilde=x0;
    
    %try up to 10 times 
    while ef<1 & tries<10
        [x_tilde, fval, ef, ~] = fmincon(li_x, x_tilde, A, b, Aeq, beq, lb, ub, nonlcon, options);
        tries=tries+1;
        if ef==0 %no initial solution found
            options.MaxFunctionEvaluations=10^tries*func_eval;
        elseif ef== -2 %no feasible solution found
            tryme=tries;
            while any(nonlcon(x_tilde)>0)==true && tryme<100
                %try to find a feasible starting solution
                x_tilde=center+(-1)^tryme*tryme*(alpha^2+alpha)*radius;
                tryme=tryme+1;
            end
        elseif ef==-3 %infinite objective function
            argmax = x_tilde';
            improved_pivot_value=nan;
            return
        elseif ef<=0
            warning('In Max_Lagrange_exactly, solver encountered an error.')
        end
    end
    x_tilde=round(x_tilde,digits);
    argmax = x_tilde';
    
    improved_pivot_value = -1*li_x(x_tilde);
    
    %if the argmax is too far outside the trust region
    if norm(argmax) > 1.01^(func_eval/100)*radius
        [argmax, improved_pivot_value] = Max_Lagrange_exactly(basis, center, radius, 10*func_eval,x0,Y,alpha,lin_quad,true);
    end
    
        
end  

    
function [lambda,maxpoints,maxlagrange] = Lambda_Poised(Y,M,Delta,lin_quad,vb,oldmax,alpha,exactprecision)
%   Lambda_Poised determines the Lambda Poised value of the set
%
%   INPUTS
%
%   Y = the set of points
%   M = the basis matrix
%   Delta = the trust region radius
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%   vb = a scaled bounds structure
%       .lb = the lower bound for each variable
%       .ub = the upper bound for each variable
%   oldmax = set of old maximum points (optional)
%                   default is []
%   alpha = the minimum percent separation distance (optional)
%                   default is 0.01
%   exactprecision = boolean value if the exact calculations should be used
%                       (optional) default is false
%
%   OUTPUTS
%   
%   lambda = the lambda poised value of the current set
%               lambda = max(maxlagrage)
%   maxpoints = the points that maximize each lagrange function
%   maxlagrange = the lagrange function values for each Y point
%
%the assumption is that the points have already been centered at the origin
%with a radius of Delta
%

    if nargin<8
        exactprecision=false; %whether or not to used exacty lagrange maximization
    end
    if nargin< 7
        alpha=.01; %percentage of radius that must be a 'dead' zone
    end
    if nargin<6
        oldmax=[];
    end
    [count,dim]=size(Y);
    maxlagrange=zeros(1,count);
    maxpoints=zeros(count,dim);
    
    %get values for each lagrange polynomial
    if exactprecision==false
        if isempty(oldmax)==true
            for i=1:count
                [maxpoints(i,:),maxlagrange(i)]=Maximize_Lagrange_poly(M(i,:), zeros(1,dim), Delta,100,[],Y,alpha,lin_quad,vb);
            end
        else
            for i=1:count
                [maxpoints(i,:),maxlagrange(i)]=Maximize_Lagrange_poly(M(i,:), zeros(1,dim), Delta,100,oldmax(i,:),Y,alpha,lin_quad,vb);
            end
        end
    else
        if isempty(oldmax)==true
            for i=1:count
                [maxpoints(i,:),maxlagrange(i)]=Max_Lagrange_exactly(M(i,:), zeros(1,dim), Delta,100,[],Y,alpha,lin_quad,vb);
            end
        else
            for i=1:count
                [maxpoints(i,:),maxlagrange(i)]=Max_Lagrange_exactly(M(i,:), zeros(1,dim), Delta,100,oldmax(i,:),Y,alpha,lin_quad,vb);
            end
        end
    end
    %get lambda value
    if any(isnan(maxlagrange))
        lambda=nan;
    else
        lambda=max(maxlagrange);
    end
    
end

function [c, ceq,c_grad,ceq_grad] = disk(x, center, radius)
%   dispoints does constrained optimization within a disk
%
%   INPUTS
%
%   x = the point
%   center = the center point
%   radius = the trust region radius
%
%   OUTPUTS
%
%   these are the outputs required by the nonlincon function of fmincon
%   refer to MATLAB's help documentation for details
%
%
    dist = x - center;
    c = dot(dist,dist) - radius^2;
    c_grad=2*dist;
    if iscolumn(c_grad) == false
        c_grad = c_grad';
    end
    ceq = [];
    ceq_grad=[];
    
end

function [c, ceq,c_grad,ceq_grad] = diskpoints(x, center, radius,Y,alpha)
%   diskpoints does constrained optimization within a disk, keep points separated
%
%   INPUTS
%
%   x = the point
%   center = the center point
%   radius = the trust region radius
%   Y = the set of all other points
%   alpha = the minimum separation percentage
%
%   OUTPUTS
%
%   these are the outputs required by the nonlincon function of fmincon
%   refer to MATLAB's help documentation for details
%
%
    dist = x - [center;Y];
    dist=dist';
    constraint=[1, alpha^2*ones(1,size(Y,1))]*radius^2;
    c = constraint- sum(dist.^2,1);
    c=c';
    c(1,:)=-1*c(1,:);
    c_grad=-2*dist;
    c_grad(:,1)=-1*c_grad(:,1);
    ceq = [];
    ceq_grad=[];
    
end

function hess = hessinterior(x,lambda,H) 
%   hessinterior is a function that creates a hessian that meets constraints
%       refer to MATLAB nonlinear constrained optimization documentation for help
%
%   INPUTS
%
%   x = the point
%   lambda = a hessian lambda boundary structure
%   H = standard hessian at a point
%
%   OUTPUTS
%
%   hess = the hessian function
%
%
    dim=size(x,1);
    myeye=2*eye(dim);
    lamb=lambda.ineqnonlin;
    hess=H+lamb(1)*myeye;
    for i=2:size(lamb,1)
        hess=hess-lamb(i)*myeye;
    end
    
end


function [argmax, improved_pivot_value] = Maximize_Lagrange_poly(basis, center, radius, func_eval,x0,Y,alpha,lin_quad,vb,internal)
%   Maximize_Lagrange_poly picks the point that maximizes the lagrange
%   polynomials using Cauchy steps if possible
%
%   INPUTS
%
%   basis = the basis vector
%   center = the center point (usually all zeros)
%   radius = the trust region radius
%   func_eval = the number of function evaluations to run in the solver
%   x0 = the initial potential starting solution, set to [] if no initial
%   Y = the set of points
%   alpha = the minimum percent separation distance
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%   vb = a scaled bounds structure
%       .lb = the lower bound for each variable
%       .ub = the upper bound for each variable
%   internal = boolean if the function is being called from itself
%                   (optional)
%
%   OUTPUTS
%   
%   argmax = the points that maximize the lagrange function
%   improved_pivot_value = the lagrange function values for the best point
%
%the assumption is that the points have already been centered at the origin
%with a radius of Delta
%

    if nargin<10 || internal ~= true
        %maximize the function, then minimize the function.  Pick the
        %larger in absolute value, and this maximizes abs(function())
        [argmax, maxval] = Maximize_Lagrange_poly(basis, center, radius, func_eval,x0,Y,alpha,lin_quad,vb,true);
        [argmin, minval] = Maximize_Lagrange_poly(-1*basis, center, radius, func_eval,x0,Y,alpha,lin_quad,vb,true);
        if abs(maxval)>abs(minval)
            argmax = argmax;
            improved_pivot_value = maxval;
        else
            argmax = argmin;
            improved_pivot_value = minval;
        end
        return
    end
    
    digits = -floor(log10(alpha^3*radius)); %alpha cubed is used for rounding

    %center must be a row vector
    if lin_quad==2
        [~,g,H,~,xstar] = Model_Approximation(center-center,lin_quad,-1*basis',true,true);
    elseif lin_quad==1
        [~,g,H,~,~] = Model_Approximation(center-center,lin_quad,-1*basis',false,true);
        xstar = center+radius*g/norm(g);
    end
    
    li_x=@(x) Model_Approximation(x-center,lin_quad,-1*basis',false,false);
    nonlcon = @(x)diskpoints(x, center, radius,Y,alpha);
    if isempty(xstar)==true
        cauchy=center+radius*2/3;     
    else
        cauchy=xstar;
        VarBds=[vb.lb',vb.ub'];
        cauchy=Check_Cons(cauchy ,center, VarBds); 
    end
    c_pivot=-1*li_x(cauchy);    
    size_factor=1;
    epsilon=alpha^4*radius^2; %a condition that allows for points to be slightly closer together
    tries=0;
    %try to find the best point that is a minimum distance away from other points
    while any(round(nonlcon(cauchy),digits)>epsilon^(.8/(2-size_factor)))==true
        cauchy=Cauchy_point(g',H, center, size_factor*radius,xstar,vb);
        cauchy=floor(cauchy*10^digits)/10^digits;
        c_pivot=-1*li_x(cauchy);
        size_factor=size_factor-alpha; 
        if size_factor<=.1-2*alpha
            size_factor=1;
        elseif size_factor<=.1
            %warning("The Cauchy_point algorithm failed to find a satisfactory point.")
            [cauchy, c_pivot] = Max_Lagrange_exactly(basis, center, radius, func_eval,x0,Y,alpha,lin_quad,vb,true);
            cauchy = cauchy';
            cauchy = floor(cauchy*10^digits)/10^digits;
            tries = tries+1;
            c_norm = vecnorm(cauchy);
            if c_norm > radius
                cauchy = radius/c_norm*cauchy;
                cauchy = floor(cauchy*10^digits)/10^digits;
                c_pivot = -1*li_x(cauchy);
            end
            if tries>=5
                %warning("The point selected by the solver may be ever so slightly outside the acceptable range.")
                break
            end
        end
    end
        argmax=cauchy;
        improved_pivot_value=c_pivot;
   
end  

function [Cauchy_point] = Cauchy_point(G, B, solution, delta_T,xstar,vb) %%Finding the Cauchy Point
%initially coded by Xueqi Zhao and Naijia Dong
%modified by Kurtis Konrad

    VarBds=[vb.lb',vb.ub'];
    Q = G'*B*G;
    Gnorm=norm(G);
    b = (-1)*delta_T/Gnorm*G';
    if Q <= 0
        tau = 1;
    else
        tau = min([(Gnorm)^3/(delta_T*Q), 1]);
    end
    Cauchy_point = solution+tau*b;
    %Try DogLeg
    [~,flag]=chol(B);
    if flag==0 %if B is positive definite
        p_u=-1*(G'*G/(Q))*G';
        if tau>=0 && tau<=2
            if tau<=1
                ptau=tau*p_u;
            else
                ptau=tau*p_u+(tau-1)*(xstar-p_u);
            end
            Cauchy_point = solution+ptau;
        end


    %eigenvalue
    else
        [V,D]=eig(B);
        D=diag(D);
        [lamb,ind]=min(D);
        if lamb<0
            Sk=V(:,ind);
            if Sk'*G >0
                Sk=-Sk;
            end
            Cauchy_point = solution+delta_T*Sk'/norm(Sk);
        end

    end
    Cauchy_point = Check_Cons(Cauchy_point ,solution, VarBds);
end

function modiSsolsV = Check_Cons(ssolsV, ssolsV2, VarBds)
%initially coded by Xueqi Zhao and Naijia Dong
%modified by Kurtis Konrad
    col = size(ssolsV, 2);
    stepV = ssolsV - ssolsV2;
    %t = 1; % t>0 for the correct direction
    tmaxV = ones(2,col);
    uV = VarBds(stepV>0,2); uV = uV';
    lV = VarBds(stepV<0,1); lV = lV';
    if isempty(uV) == 0 % length(uV) > 0
        tmaxV(1, stepV>0) = (uV - ssolsV2(stepV>0)) ./ stepV(stepV>0);
    end
    if isempty(lV) == 0 % length(lV) > 0
        tmaxV(2,stepV<0) = (lV - ssolsV2(stepV<0)) ./ stepV(stepV<0);
    end
    t2 = min(min(tmaxV));
    modiSsolsV = ssolsV2 + t2*stepV;
    %rounding error
    for kc=1:col
        if modiSsolsV(kc) < 0 && modiSsolsV(kc) > -0.00000005
            modiSsolsV(kc) = 0;
        end
    end
end
  
function [y, grad,hess,Xvec,xstar] = Model_Approximation(x, lin_quad, qcoef, return_optimal,return_gradhess)
    %build the gradient and the function value of the quadratic model
    %Also build a Vandermonde 'matrix' Xvec
%
% INPUTS
% x = a m x n matrix with m points to evaluate and n total dimensions
%
% lin_quad = a numeric parameter that indicates if a linear or a quadratic
%       model is in use
%           Default is 2. aka quadratic
%
% qcoef = an optional parameter that is a vector of size p=(n+1)*(n+2)/2
%       for the quadratic case and p=n+1 for the linear case
%           Default is all zeros
%
% return_optimal = an optional parameter that indicates if the function
%       should return the optimal solution xstar
%           Default is false
%
% return_gradhess = an optional parameter that indicates if the function
%       should return the gradient and hessian
%           Default is true
%
% OUTPUTS
% y = a m x 1 column vector of function values at each point in x
%
% grad = a m x n matrix that represents the gradient at each of m points
%
% hess = a n x n matrix that represents the hessian of the function
%       since we only allow quadratic and linear functions, hess is the
%       same for all m points
%
% Xvec = the m x p Vandermonde matrix row for each point.
%
% xstar = the global optimal point of the quadratic function assuming it
%       is strictly convex or concave.  Otherwise, it returns a point where
%       the function gradient is 0.

    
    [count,dim] = size(x);
    if nargin<2
        lin_quad=2;
    end
        
    if lin_quad==2
        p = (dim + 1)*(dim + 2)/2;
    elseif lin_quad==1
        p = (dim + 1);
    else
        error("Variable 'lin_quad' must be either 1 or 2.")
    end
    
    if nargin<5
        return_gradhess=true;
    end
    if nargin<4
        return_optimal=false;
    end
    if nargin<3
        qcoef=zeros(p,1);
    elseif iscolumn(qcoef) == false
        qcoef = qcoef';
    end
    
    if return_optimal==true && lin_quad==1
        error("You cannot find an optimal solution to a linear model.")
    end
    
    if size(qcoef,1) ~= p
        error("The size of the coefficient vector is not correct for the model being built.")
    end

    % build vandermonde matrix
    Xvec = zeros(count, p);
    Xvec(:,1) = 1;
    Xvec(:,2:(2 + dim - 1)) = x; 

    if lin_quad==2
        k = 1 + dim + 1;
        for i = 1:dim
            for j = i:dim
                Xvec(:,k) = x(:,i).*x(:,j);
                if i == j
                    Xvec(:,k) = Xvec(:,k)/2;
                end
                k = k+1;
            end
        end
    end
    
    %get function values
    y = Xvec*qcoef;
    
    if nargout==1  %if you only need the y value
        return
    end
    if return_gradhess==false && return_optimal==false
        grad=[];
        hess=[];
        return
    end
    
    beta=qcoef(2:dim+1);
    if lin_quad==2
        gamma=qcoef(dim+2:end);
        A=tril(ones(dim));
        A(A==1)=gamma;
        A=(A'+A)/2;
        A=A-diag(diag(A))/2; %correct the diagonals because the 1/2
        hess=A+A';
    elseif lin_quad==1
        hess=zeros(dim);
    end
    grad=beta+(hess)*x';
    grad=grad';
    
    if return_optimal==true
        if rcond(hess)>.001
            xstar=hess\(-1*beta); %the unconstrained optimal (set grad==0)
            xstar=xstar';
        else
            xstar=[];
        end
    else
        xstar=[];
    end

    
end