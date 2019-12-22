function PlotWrapper(problemnameArray, solvernameArray)
% Make plots for each problem comparing the performance of different algorithms

% Inputs:
% problemnameArray: structure listing the problem names
% solvernameArray: structure listing the solver names

%   *************************************************************
%   ***                 Updated by David Eckman               ***
%   ***     david.eckman@northwestern.edu   Dec 22, 2019      ***
%   *************************************************************

% Other default parameters
CILevel = 0.95; % Confidence interval level
low_quantile = 0.25; % Low quantile
high_quantile = 0.75; % High quantile

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
    probstructHandle = str2func(strcat(problemname, 'Structure'));
    rmpath(problempath)
            
    % Get the problem's dimension, min/max, budget, and # of streams 
    [minmax, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = probstructHandle(0);
    
    for k2 = 1:numAlgs       
        
        solvername = solvernameArray{k2};
        
        % Read in output for the solver-problem pairing as "SMatrix"
        load(strcat('PostData/PostData_',solvername,'_on_',problemname,'.mat'),'BudgetMatrix','FMatrix');
        repsAlg = max(BudgetMatrix(:,1)); %%%size(FMatrix, 1);
        
        % Extract budget points
        budget_pts = unique(BudgetMatrix(:,2));
        num_budget_pts = length(budget_pts);
        
        % Initialize and fill in function values at budget points
        FBudget = zeros(repsAlg, num_budget_pts);
        for i = 1:repsAlg            
            for j = 1:num_budget_pts
                index_lookup = max(intersect(find(BudgetMatrix(:,1) == i), find(BudgetMatrix(:,2) <= budget_pts(j))));
                FBudget(i,j) = FMatrix(index_lookup);
            end
        end
        
        % Compute descriptive statistics (mean, median, quantiles)       
        FMeanVector = mean(FBudget,1);
        if repsAlg == 1
            FVarVector = zeros(1, num_budget_pts);
        else
            FVarVector = var(FBudget,[],1);
        end
        FnSEM = sqrt(FVarVector/repsAlg); % Std error
        EWidth = norminv(1-(1-CILevel)/2,0,1)*FnSEM(k2,:);
        FMedianVector = median(FBudget);
        Flowquant = quantile(FBudget, low_quantile);
        Fhighquant = quantile(FBudget, high_quantile);

    end
    
    % PLOT 1: Mean + Confidence Intervals   
    figure;
    hold on;
    stairs(budget_pts, FMeanVector, 'b-', 'LineWidth', 1.5);
    stairs(budget_pts, FMeanVector - FnSEM, 'b:', 'LineWidth', 1.5);
    stairs(budget_pts, FMeanVector + FnSEM, 'b:', 'LineWidth', 1.5);
    hold off;
    
    % Label and format the plot
    xlabel('Budget','FontSize',14); 
    ylabel('Objective Function Value','FontSize',14);
    minmaxList = {'minimize','nah','maximize'};
    title(['Problem: ', problemname, ' (',minmaxList{minmax+2},') -- Mean + CI'],'FontSize',15);
    AlgNamesLegend = solvernameArray;
    legend(AlgNamesLegend,'Location','best');
    miny = min(min(FMeanVector - EWidth));
    maxy = max(max(FMeanVector + EWidth));
    axis([0,max(budget_pts), miny*0.99, maxy*1.01]);
    set(gca,'FontSize',12);
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);

    % Save as a .fig file
    plot1filename = strcat('Plots/',problemname,'_MeanCI.fig');
    saveas(gcf, plot1filename);
    fprintf('\t Saved plot of Mean + CI to file "%s" \n', plot1filename)

  
    % PLOT 2: Median + Quantiles
    figure;
    hold on;
    stairs(budget_pts, FMedianVector, 'b-', 'LineWidth', 1.5);
    stairs(budget_pts, Flowquant, 'b:', 'LineWidth', 1.5);
    stairs(budget_pts, Fhighquant, 'b:', 'LineWidth', 1.5);
    hold off;

    % Label and format the plot
    xlabel('Budget','FontSize',14); 
    ylabel('Objective Function Value','FontSize',14);
    title(['Problem: ', problemname, ' (',minmaxList{minmax+2},') -- Quantile'],'FontSize',15);
    legend(AlgNamesLegend,'Location','best');
    miny = min(min(Flowquant));
    maxy = max(max(Fhighquant));
    axis([0,max(budget_pts), miny*0.99, maxy*1.01]);
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
