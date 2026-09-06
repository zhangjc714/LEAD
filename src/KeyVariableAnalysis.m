function [KeyVarIndex,Dataset] = KeyVariableAnalysis(Algorithm,Problem,Dataset,params)
    KeyVarNum = ceil(params.ratio*Problem.D);
    numTree = params.numTree;

    Algorithm.info = [Algorithm.info; Problem.FE, sum(Dataset.labels>0), sum(Dataset.labels<0)];

    if numel(Dataset.labels) == 0
        KeyVarIndex = randperm(Problem.D, KeyVarNum);
        Dataset = struct('EDs', [], 'labels', []);

    else
        RFmodel = TreeBagger(numTree, Dataset.EDs, Dataset.labels, ...
            'Method', 'regression', ...
            'OOBPrediction', 'on', ...
            'OOBPredictorImportance', 'on');
        
        importance = RFmodel.OOBPermutedPredictorDeltaError;
    
        [~, sortedIdx] = sort(importance, 'descend');
        KeyVarIndex = sortedIdx(1:KeyVarNum);
    
        Dataset = struct('EDs', [], 'labels', []);
    end

    Algorithm.KVIs = [Algorithm.KVIs; Problem.FE, KeyVarIndex];
end