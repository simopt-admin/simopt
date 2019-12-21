function PlotWrapper(problemnameArray, solvernameArray, repsSoln)
% Make plots for each problem comparing the performance of different algorithms

% Inputs:
% problemnameArray: structure listing the problem names
% solvernameArray: structure listing the solver names
% repsSoln: number of replications for post-evaluation of solutions

%   *************************************************************
%   ***                 Written by David Eckman               ***
%   ***            dje88@cornell.edu     Dec 20, 2018         ***
%   *************************************************************

% Other default parameters
numBudget = 20; % Number of budget points recorded between lower and upper budget
% If numBudget is changed --> Need to change in RunWrapper.m too
CILevel = 0.95; % Confidence interval level

numAlgs = length(solvernameArray);

for k1 = 1:length(problemnameArray)
        
    % Create function handles for problem and problem structure
    problemname = problemnameArray{k1};
    problempath = strcat(pwd,'/../Problems/',problemname);
    if exist(problempath, 'dir') ~= 7
        disp(strcat('The problem folder ', problemname, ' does not exist.'))
        continue
    end
    addpath(problempath)
    probHandle = str2func(problemname);
    probstructHandle = str2func(strcat(problemname, 'Structure'));
    % Share problem file PROBLEMNAME.m to all processors (in parallel)
    addAttachedFiles(gcp, strcat(problemname,'.m'))
    rmpath(problempath)
            
    % Get the problem's dimension, min/max, budget, and # of streams 
    [minmax, dim, ~, ~, ~, ~, ~, ~, budgetR, ~, ~, NumRngs] = probstructHandle(0);
    
    % Initialize vectors for storing data for this problem
    FMeanVector = zeros(numAlgs, numBudget+1);
    FVarVector = zeros(numAlgs, numBudget+1);
    FnSEM = zeros(numAlgs, numBudget+1);
    EWidth = zeros(numAlgs, numBudget+1);
    FMedianVector = zeros(numAlgs, numBudget+1);
    Fquant25 = zeros(numAlgs, numBudget+1);
    Fquant75 = zeros(numAlgs, numBudget+1);
    
    for k2 = 1:numAlgs       
        
        solvername = solvernameArray{k2};
        
        % Read in output for the solver-problem pairing as "SMatrix"
        load(strcat('RawData/RawData_',solvername,'_on_',problemname,'.mat'),'SMatrix');
        [repsAlg, ~, ~] = size(SMatrix); % Number of times the solver was run on the problem 
        
        % Initialize matrix of function values
        FMatrix = zeros(repsAlg, numBudget+1);
        
        % Create a common set of new random number streams (#s = NumRngs*(j-1)+1, ... NumRngs*j)
        % to use for each macrorep solution.
        % I.e., Streams 1, ..., NumRngs, will be used for ALL solutions recorded at ALL time
        % points across ALL macroreplications.
        problemRng = cell(1, NumRngs);
        for i = 1:NumRngs
            problemRng{i} = RandStream.create('mrg32k3a', 'NumStreams', NumRngs, 'StreamIndices', i);
        end
   
        % Post-evaluate the function at the initial and returned solutions
        fprintf('Post-evaluating solutions from solver %s on problem %s: \n', solvername, problemname)
        for j = 1:repsAlg        
            
            fprintf('\t Macroreplication %d of %d ... \n', j, repsAlg)

            parfor k = 1:numBudget+1
                % Obtain repsSoln replications of the obj fn (using CRN via substreams)
                [FMatrix(j,k), ~, ~, ~, ~, ~, ~, ~] = probHandle(reshape(SMatrix(j,k,:),1,dim), repsSoln, problemRng, 1);
            end          
        end
        
        % Calculate descriptive statistics (mean, median, quantiles)
        FMeanVector(k2,:) = mean(FMatrix);
        if repsAlg==1
            FVarVector(k2,:) = zeros(1,numBudget+1);
        else
            FVarVector(k2,:) = var(FMatrix);
        end
        FnSEM(k2,:) = sqrt(FVarVector(k2,:)/repsAlg); % Std error
        EWidth(k2,:) = norminv(1-(1-CILevel)/2,0,1)*FnSEM(k2,:);
        FMedianVector(k2,:) = median(FMatrix);
        Fquant25(k2,:) = quantile(FMatrix, 0.25); % 0.25 quantile
        Fquant75(k2,:) = quantile(FMatrix, 0.75); % 0.75 quantile

    end

    % Offset the x-coordinates of the plotted points to avoid overlap
    budget = round(linspace(budgetR(1),budgetR(2),numBudget));
    increm = (budget(2)-budget(1))/(2*numAlgs);
    offsetbudget = repmat([0,budget]',1,numAlgs) + repmat(increm*(0:(numAlgs-1)),numBudget+1,1);
     
    % PLOT 1: Mean + Confidence Intervals   
    figure;
    errorbar(offsetbudget,FMeanVector',EWidth','-o', 'LineWidth',1.5);
    
    % Label and format the plot
    xlabel('Budget','FontSize',14); 
    ylabel('Objective Function Value','FontSize',14);
    minmaxList = {'minimize','nah','maximize'};
    title(['Problem: ', problemname, ' (',minmaxList{minmax+2},') -- Mean + CI'],'FontSize',15);
    AlgNamesLegend = solvernameArray;
    legend(AlgNamesLegend,'Location','best');
    miny = min(min(FMeanVector - EWidth));
    maxy = max(max(FMeanVector + EWidth));
    axis([0,max(budget)*1.01, miny*0.99, maxy*1.01]);
    set(gca,'FontSize',12);
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);

    % Save as a .fig file
    plot1filename = strcat('Plots/',problemname,'_MeanCI.fig');
    saveas(gcf,plot1filename);
    fprintf('\t Saved plot of Mean + CI to file "%s" \n', plot1filename)

  
    % PLOT 2: Median + Quantiles
    figure;
    errorbar(offsetbudget,FMedianVector',FMedianVector'-Fquant25',Fquant75'-FMedianVector','-o', 'LineWidth',1.5);

    % Label and format the plot
    xlabel('Budget','FontSize',14); 
    ylabel('Objective Function Value','FontSize',14);
    title(['Problem: ', problemname, ' (',minmaxList{minmax+2},') -- Quantile'],'FontSize',15);
    legend(AlgNamesLegend,'Location','best');
    miny = min(min(Fquant25));
    maxy = max(max(Fquant75));
    axis([0,max(budget)*1.01, miny*0.99, maxy*1.01]);
    set(gca,'FontSize',12);
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);
    
    % Save as a .fig file
    plot2filename = strcat('Plots/',problemname,'_Quantile.fig');
    saveas(gcf, plot2filename);
    fprintf('\t Saved plot of Median + Quantiles to file "%s" \n', plot2filename)
    
end

end
