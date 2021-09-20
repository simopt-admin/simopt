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

% MATLAB default colors
color_matrix = [
    [0, 0.4470, 0.7410]; ... % blue
    [0.8500, 0.3250, 0.0980]; ... % orange 	          	
    [0.9290, 0.6940, 0.1250]; ... % yellow
    [0.4940, 0.1840, 0.5560]; ... % purple
    [0.4660, 0.6740, 0.1880]; ... % green
    [0.3010, 0.7450, 0.9330]; ... % light blue,
    [0.6350, 0.0780, 0.1840]];    % burgundy
            
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
            
    % Get the problem's min/max and budget 
    [minmax, ~, ~, ~, ~, ~, ~, ~, budget, ~, ~, ~] = probstructHandle(0);
    
    % Initialize cell for storing summary statistics for each algorithm
    plot_data_cell = cell(numAlgs, 7);
    % (budget pts, mean, lowerCI, upperCI, median, lowerquant, upperquant)
    
    for k2 = 1:numAlgs       
        
        solvername = solvernameArray{k2};
        
        % Read in output for the solver-problem pairing as "SMatrix"
        load(strcat('PostData/PostData_',solvername,'_on_',problemname,'.mat'),'BudgetMatrix','FMatrix');
        repsAlg = max(BudgetMatrix(:,1)); %%%size(FMatrix, 1);
        
        % Extract budget points
        budget_pts = unique(BudgetMatrix(:,2))';
        num_budget_pts = length(budget_pts);
        
        % Initialize and fill in function values at budget points
        FBudget = zeros(repsAlg, num_budget_pts);
        for i = 1:repsAlg            
            for j = 1:num_budget_pts
                index_lookup = max(intersect(find(BudgetMatrix(:,1) == i), find(BudgetMatrix(:,2) <= budget_pts(j))));
                FBudget(i,j) = FMatrix(index_lookup);
            end
        end
        
        % Plot convergence curves for all macroreplications of the solver
        % Don't display to the screen
        figure('visible','off');
        set(gcf,'Visible','off','CreateFcn','set(gcf,''Visible'',''on'')')
        stairs(budget_pts, FBudget', 'LineStyle', '-', 'LineWidth', 1.5);
        
        % Labeling
        xlabel('Budget','FontSize',14); 
        ylabel('Objective Function Value','FontSize',14);
        minmaxList = {'min','-','max'};
        title([num2str(repsAlg), ' Macroreps of ', solvername, ' Solver on ', problemname, ' Problem (',minmaxList{minmax+2},')'],'FontSize',15);
        
        % Formatting
        axis([0, budget, min(min(FBudget)), max(max(FBudget))]);
        set(gca,'FontSize',12);
        set(gcf,'Units','Inches');
        pos = get(gcf,'Position');
        set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);
        
        % Save as a .fig file
        indplotfilename = strcat('Plots/',problemname,'_',solvername,'.fig');
        saveas(gcf, indplotfilename);
        fprintf('\t Saved plot of all macroreplications to file "%s" \n', indplotfilename)
 
        % Compute descriptive statistics (mean, median, quantiles)       
        FMeanVector = mean(FBudget,1);
        if repsAlg == 1
            FVarVector = zeros(1, num_budget_pts);
        else
            FVarVector = var(FBudget,[],1);
        end
        FnLowerCI = FMeanVector - norminv(1-(1-CILevel)/2,0,1)*sqrt(FVarVector/repsAlg);
        FnUpperCI = FMeanVector + norminv(1-(1-CILevel)/2,0,1)*sqrt(FVarVector/repsAlg);
        FMedianVector = median(FBudget);
        Flowquant = quantile(FBudget, low_quantile);
        Fhighquant = quantile(FBudget, high_quantile);

        plot_data_cell(k2, :) = {budget_pts, FMeanVector, FnLowerCI, FnUpperCI, FMedianVector, Flowquant, Fhighquant};
        
    end
    
    % PLOT 1: Mean + Confidence Intervals    
    figure;
    hold on;
    
    % Initialize handles for lines to label with legend
    line_handles = zeros(1, numAlgs);
    
    % Add mean + CI curves for each solver
    for k2 = 1:numAlgs
        line_handles(k2) = stairs(plot_data_cell{k2,1}, plot_data_cell{k2,2}, 'color', color_matrix(mod(k2-1,7)+1,:), 'LineStyle', '-', 'LineWidth', 1.5);
        stairs(plot_data_cell{k2,1}, plot_data_cell{k2,3}, 'color', color_matrix(mod(k2-1,7)+1,:), 'LineStyle', ':', 'LineWidth', 1.5);
        stairs(plot_data_cell{k2,1}, plot_data_cell{k2,4}, 'color', color_matrix(mod(k2-1,7)+1,:), 'LineStyle', ':', 'LineWidth', 1.5);
    end
    
    % Labeling
    xlabel('Budget','FontSize',14); 
    ylabel('Objective Function Value','FontSize',14);
    minmaxList = {'min','-','max'};
    title(['Mean-CI Plot for ', problemname, ' Problem (',minmaxList{minmax+2},')'],'FontSize',15);
    
    % Add a legend
    AlgNamesLegend = solvernameArray;
    legend(line_handles, AlgNamesLegend,'Location','best');

    % Formatting
    axis([0, budget, min([plot_data_cell{:,3}]), max([plot_data_cell{:,4}])]);
    set(gca,'FontSize',12);
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);

    hold off;

    % Save as a .fig file
    plot1filename = strcat('Plots/',problemname,'_MeanCI.fig');
    saveas(gcf, plot1filename);
    fprintf('\t Saved plot of Mean + CI to file "%s" \n', plot1filename)

  
    % PLOT 2: Median + Quantiles
    figure;
    hold on;
    
    % Initialize handles for lines to label with legend
    line_handles = zeros(1, numAlgs);
    
    % Add median + quantile curves for each solver
    for k2 = 1:numAlgs
        line_handles(k2) = stairs(plot_data_cell{k2,1}, plot_data_cell{k2,5}, 'color', color_matrix(mod(k2-1,7)+1,:), 'LineStyle', '-', 'LineWidth', 1.5);
        stairs(plot_data_cell{k2,1}, plot_data_cell{k2,6}, 'color', color_matrix(mod(k2-1,7)+1,:), 'LineStyle', ':', 'LineWidth', 1.5);
        stairs(plot_data_cell{k2,1}, plot_data_cell{k2,7}, 'color', color_matrix(mod(k2-1,7)+1,:), 'LineStyle', ':', 'LineWidth', 1.5);
    end
    
    % Labeling
    xlabel('Budget','FontSize',14); 
    ylabel('Objective Function Value','FontSize',14);
    minmaxList = {'min','-','max'};
    title(['Median-Quantile Plot for ', problemname, ' Problem (',minmaxList{minmax+2},')'],'FontSize',15);
    
    % Add a legend
    AlgNamesLegend = solvernameArray;
    legend(line_handles, AlgNamesLegend,'Location','best');

    % Formatting
    axis([0, budget, min([plot_data_cell{:,6}]), max([plot_data_cell{:,7}])]);
    set(gca,'FontSize',12);
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);
    
    hold off;

    % Save as a .fig file
    plot2filename = strcat('Plots/',problemname,'_Quantile.fig');
    saveas(gcf, plot2filename);
    fprintf('\t Saved plot of Median + Quantiles to file "%s" \n', plot2filename)
    
end

end
