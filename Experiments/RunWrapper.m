function RunWrapper(problemnameArray, solvernameArray, repsAlg)
% Run multiple algorithms on multiple problems and write the solutions
% visited, objective function means and variances to .mat files for 
% each algorithm-problem pair.

% Inputs:
% problemnameArray: structure listing the problem names
% solvernameArray: structure listing the solver names
% repsAlg: number of macroreplications of each solver on each problem

%   *************************************************************
%   ***                 Updated by David Eckman               ***
%   ***     david.eckman@northwestern.edu   Dec 22, 2019      ***
%   *************************************************************


% Check if number of macroreplications is an integer
if (repsAlg <= 0) || (mod(repsAlg,1) ~= 0)
    disp('The number of macroreplications (repsAlg) must be a positive integer.')
    return
end

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
    
    % If Parallel Computing Toolbox installed...
    if exist('gcp', 'file') == 2
        % Share problem file PROBLEMNAME.m to all processors
        addAttachedFiles(gcp, strcat(problemname,'.m'))
        addAttachedFiles(gcp, strcat(problemname,'Structure.m'))
    end
    
    rmpath(problempath)
    
    % Get the number of streams needed for the problem
    [~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, NumRngs] = probstructHandle(0);
        
    for k2 = 1:length(solvernameArray)   
        
        % Create function handle for solver
        solvername = solvernameArray{k2};        
        solverpath = strcat(pwd,'/../Solvers/',solvername);
        if exist(solverpath, 'dir') ~= 7
            disp(strcat('The solver folder ', solvername, ' does not exist.'))
            continue
        end
        addpath(solverpath)
        solverHandle = str2func(solvername);
        
        % If Parallel Computing Toolbox installed...
        if exist('gcp', 'file') == 2
            % Share problem file SOLVERNAME.m to all processors
            addAttachedFiles(gcp, strcat(solvername,'.m'))
        end
    
        rmpath(solverpath)
        
        % Initialize cells for reporting recommended solutions, etc.
        Ancalls_cell = cell(1, repsAlg);
        A_cell = cell(1, repsAlg);
        AFnMean_cell = cell(1, repsAlg);
        AFnVar_cell = cell(1, repsAlg);
                
        % Do repsAlg macroreplications of the algorithm on the problem
        fprintf('Solver %s on problem %s: \n', solvername, problemname)
        
        parfor j = 1:repsAlg
            
            fprintf('\t Macroreplication %d of %d ... \n', j, repsAlg)
            
            % Create (1 + NumRngs) new random number streams to use for each macrorep solution
            % (#s = {(1 + NumRngs)*(j - 1) + 1, ... (1 + NumRngs)*j}) 
            % I.e., for the first macrorep, Stream 1 will be used for the solver
            % and Streams 2, ..., 1 + NumRngs will be used for the problem
            solverRng = cell(1, 2);
            [solverRng{1}, solverRng{2}] = RandStream.create('mrg32k3a', 'NumStreams', (2 + NumRngs)*repsAlg, ...
                'StreamIndices', [(2 + NumRngs)*(j - 1) + 1, (2 + NumRngs)*(j - 1) + 2]);
            
            problemRng = cell(1, NumRngs);
            for i = 1:NumRngs
                problemRng{i} = RandStream.create('mrg32k3a', 'NumStreams', (2 + NumRngs)*repsAlg, 'StreamIndices', (2 + NumRngs)*(j - 1) + 2 + i);
            end
                        
            % Run the solver on the problem and return the solutions (and
            % obj fn mean and variance) whenever the recommended solution changes
            [Ancalls_cell{j}, A_cell{j}, AFnMean_cell{j}, AFnVar_cell{j}, ~, ~, ~, ~, ~, ~] = solverHandle(probHandle, probstructHandle, problemRng, solverRng);
            
            % Append macroreplication number to reporting of budget points
            Ancalls_cell{j} = [j*ones(length(Ancalls_cell{j}),1), Ancalls_cell{j}];
          
        end
        
        % Concatenate cell data across macroreplications
        BudgetMatrix = cat(1, Ancalls_cell{:});
        SolnMatrix = cat(1, A_cell{:});
        FnMeanMatrix = cat(1, AFnMean_cell{:});
        FnVarMatrix = cat(1, AFnVar_cell{:});
        
        % Store data in .mat file in RawData folder
        solnsfilename = strcat('RawData_',solvername,'_on_',problemname,'.mat');
        if exist(strcat('RawData/',solnsfilename), 'file') == 2
            fprintf('\t Overwriting \t --> ')
        end
        save(strcat(pwd,'/RawData/RawData_',solvername,'_on_',problemname,'.mat'), 'BudgetMatrix', 'SolnMatrix', 'FnMeanMatrix', 'FnVarMatrix');
        fprintf('\t Saved output to file "%s" \n', solnsfilename)
    end
end

end