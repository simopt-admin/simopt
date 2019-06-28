function RunWrapper(problemnameArray, solvernameArray, repsAlg)
% Run multiple algorithms on multiple problems and write the solutions
% visited, objective function means and variances to .mat files for 
% each algorithm-problem pair.

% Inputs:
% problemnameArray: structure listing the problem names
% solvernameArray: structure listing the solver names
% repsAlg: number of macroreplications of each solver on each problem

%   *************************************************************
%   ***                 Written by David Eckman               ***
%   ***            dje88@cornell.edu     Sept 4, 2018         ***
%   *************************************************************


% Check if number of macroreplications is an integer
if (repsAlg <= 0) || (mod(repsAlg,1) ~= 0)
    disp('The number of macroreplications (repsAlg) must be a positive integer.')
    return
end

% Number of budget points to record between lower and upper budgets
numBudget = 20; % If changed --> Need to change in PlotWrapper.m too

for k1 = 1:length(problemnameArray)
    
    % Create function handles for problem and problem structure
    problemname = problemnameArray{k1};
    problempath = strcat(pwd,'\..\Problems\',problemname);
    if exist(problempath, 'dir') ~= 7
        disp(strcat('The problem folder ', problemname, ' does not exist.'))
        continue
    end
    addpath(problempath)
    probHandle = str2func(problemname);
    probstructHandle = str2func(strcat(problemname, 'Structure'));
    rmpath(problempath)
    
    % Get the dimension of the problem and number of streams needed
    [~, dim, ~, ~, ~, ~, ~, ~, ~, ~, ~, NumRngs] = probstructHandle(0);
        
    for k2 = 1:length(solvernameArray)   
        
        % Create function handle for solver
        solvername = solvernameArray{k2};        
        solverpath = strcat(pwd,'\..\Solvers\',solvername);
        if exist(solverpath, 'dir') ~= 7
            disp(strcat('The solver folder ', solvername, ' does not exist.'))
            continue
        end
        addpath(solverpath)
        solverHandle = str2func(solvername);
        rmpath(solverpath)
        
        % Initialize matrices for solutions and objective function mean and
        % variance
        SMatrix = zeros(repsAlg, numBudget+1, dim);
        FnMeanMatrix = zeros(repsAlg, numBudget+1);
        FnVarMatrix = zeros(repsAlg, numBudget+1);
        
        % Do repsAlg macroreplications of the algorithm on the problem
        fprintf('Solver %s on problem %s: \n', solvername, problemname)
        
        for j = 1:repsAlg
            
            fprintf('\t Macroreplication %d of %d ... \n', j, repsAlg)
            
            % Create (1 + NumRngs) new random number streams to use for each macrorep solution
            % (#s = {(1 + NumRngs)*(j - 1) + 1, ... (1 + NumRngs)*j}) 
            % I.e., for the first macrorep, Stream 1 will be used for the solver
            % and Streams 2, ..., 1 + NumRngs will be used for the problem
            solverRng = cell(1, 2);
            [solverRng{1}, solverRng{2}] = RandStream.create('mrg32k3a', 'NumStreams', (2 + NumRngs)*repsAlg, ...
                'StreamIndices', [(2 + NumRngs)*(j - 1) + 1, (2 + NumRngs)*(j - 1) + 2]);
            %solverRng = RandStream.create('mrg32k3a', 'NumStreams', (1 + NumRngs)*repsAlg, 'StreamIndices', (1 + NumRngs)*(j - 1) + 1);
            
            problemRng = cell(1, NumRngs);
            for i = 1:NumRngs
                problemRng{i} = RandStream.create('mrg32k3a', 'NumStreams', (2 + NumRngs)*repsAlg, 'StreamIndices', (2 + NumRngs)*(j - 1) + 2 + i);
            end
                        
            % Run the solver on the problem and return the solutions (and
            % obj fn mean and variance) at the budget points (including the initial solution)
            [~, SMatrix(j, :, :), FnMeanMatrix(j,:), FnVarMatrix(j,:), ~, ~, ~, ~, ~, ~] = solverHandle(probHandle, probstructHandle, problemRng, solverRng, numBudget);
            
        end
        
        % Store data in .mat file as a matrix with dimensions: repsAlg x numBudget x dim
        solnsfilename = strcat('RawData_',solvername,'_on_',problemname,'.mat');
        if exist(strcat('RawData\',solnsfilename), 'file') == 2
            fprintf('\t Overwriting \t --> ')
        end
        save(strcat(pwd,'\RawData\RawData_',solvername,'_on_',problemname,'.mat'), 'SMatrix', 'FnMeanMatrix', 'FnVarMatrix');
        fprintf('\t Saved output to file "%s" \n', solnsfilename)
    end
end

end