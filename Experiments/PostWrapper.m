function PostWrapper(problemnameArray, solvernameArray, repsSoln)
% Take post-replications at solutions recorded during previous
% runs of solvers on problems.

% Inputs:
% problemnameArray: structure listing the problem names
% solvernameArray: structure listing the solver names
% repsSoln: number of replications for post-evaluation of solutions

%   *************************************************************
%   ***                 Updated by David Eckman               ***
%   ***     david.eckman@northwestern.edu   Dec 22, 2019      ***
%   *************************************************************

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
    
    % If Parallel Computing Toolbox installed...
    if exist('gcp', 'file') == 2
        % Share problem file PROBLEMNAME.m to all processors
        addAttachedFiles(gcp, strcat(problemname,'.m'))
    end
    
    rmpath(problempath)
            
    % Get the number of streams needed for the problem
    [~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, NumRngs] = probstructHandle(0);
    
    for k2 = 1:numAlgs       
        
        solvername = solvernameArray{k2};
        
        % Read in output for the solver-problem pairing 
        load(strcat('RawData/RawData_',solvername,'_on_',problemname,'.mat'), 'BudgetMatrix', 'SolnMatrix');
        
        % Initialize matrix of function values
        FMatrix = zeros(size(SolnMatrix, 1), 1);
        
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
        
        parfor j = 1:size(SolnMatrix, 1)
            % Obtain repsSoln replications of the obj fn (using CRN via substreams)
            [FMatrix(j), ~, ~, ~, ~, ~, ~, ~] = probHandle(SolnMatrix(j,:), repsSoln, problemRng, 1);
        end
       
        % Store data in .mat file in the PostData folder
        solnsfilename = strcat('PostData_',solvername,'_on_',problemname,'.mat');
        if exist(strcat('PostData/',solnsfilename), 'file') == 2
            fprintf('\t Overwriting \t --> ')
        end
        save(strcat(pwd,'/PostData/PostData_',solvername,'_on_',problemname,'.mat'), 'BudgetMatrix', 'FMatrix');
        fprintf('\t Saved output to file "%s" \n', solnsfilename)

    end
    
end

end
