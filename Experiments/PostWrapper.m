function PostWrapper(problemnameArray, solvernameArray, repsSoln)
% Take post-replications at solutions recorded during previous
% runs of solvers on problems.

% Inputs:
% problemnameArray: structure listing the problem names
% solvernameArray: structure listing the solver names
% repsSoln: number of replications for post-evaluation of solutions

%   *************************************************************
%   ***                 Written by David Eckman               ***
%   ***            dje88@cornell.edu     Dec 21, 2018         ***
%   *************************************************************

% Other default parameters
numBudget = 20; % Number of budget points recorded between lower and upper budget
% If numBudget is changed --> Need to change in RunWrapper.m too

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
            
    % Get the problem's dimension, min/max, budget, and # of streams 
    [~, dim, ~, ~, ~, ~, ~, ~, ~, ~, ~, NumRngs] = probstructHandle(0);
    
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
        
        % Store data in .mat file as a matrix with dimensions: repsAlg x numBudge
        solnsfilename = strcat('PostData_',solvername,'_on_',problemname,'.mat');
        if exist(strcat('PostData/',solnsfilename), 'file') == 2
            fprintf('\t Overwriting \t --> ')
        end
        save(strcat(pwd,'/PostData/PostData_',solvername,'_on_',problemname,'.mat'), 'FMatrix');
        fprintf('\t Saved output to file "%s" \n', solnsfilename)

    end
    
end

end
