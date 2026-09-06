function [ED,Archive] = EvoDirectionModeling(Problem,Archive,KeyVarIndex,params)
    
    N = Problem.N;
    d = size(KeyVarIndex,2);
    
    for i = 1:length(Archive)
        if isempty(Archive{i})
            Archive{i} = zeros(1, d);
        else
            Archive{i} = Archive{i}(:,KeyVarIndex);
        end
    end

    D = vertcat(Archive{:});
 
    if size(D,1) <= N
        learnedED = zeros(N, d);
    else
        if size(D,1) < d
            D = [D; zeros(d-size(D,1)+1, d)];
        end

        mean_init = cellfun(@(x) mean(x, 1), Archive, 'UniformOutput', false);
        mean_init = vertcat(mean_init{:});
        
        cov_init = cellfun(@(x) cov(x), Archive, 'UniformOutput', false);
        
        for i = 1:length(cov_init)
            if ~(isa(cov_init{i}, 'double') && all(size(cov_init{i}) == [d, d]))
                cov_init{i} = eye(d);
            end
        end
        
        cov_init = cat(3, cov_init{:});

        for i = 1:N
            C = cov_init(:,:,i);
            if any(eig(C) <= 0)
                cov_init(:,:,i) = C + 1e-3 * eye(size(C));
            end
        end

        InitPara = struct('mu', mean_init, 'Sigma', cov_init, 'ComponentProportion', ones(1,N)/N);
        options = statset('MaxIter', params.MaxIter, 'TolFun', params.TolFun);

        try
            gmm = fitgmdist(D, N, 'Start', InitPara, 'Options', options, 'Regularize', params.reg);
        catch
            [~, mean_init] = kmeans(D, N);
            cov_init = repmat(eye(size(D, 2)), [1, 1, N]);
            InitPara = struct('mu', mean_init, 'Sigma', cov_init, 'ComponentProportion', ones(1, N) / N);
            gmm = fitgmdist(D, N, 'Start', InitPara, 'Options', options, 'Regularize', params.reg);
        end
        
        costMatrix = zeros(N, N);
        for i = 1:N
            for j = 1:N
                costMatrix(i, j) = norm(mean_init(i,:) - gmm.mu(j,:), 2);
            end
        end

        [matchIdx, ~] = matchpairs(costMatrix, 1e6);
        matchIdx = sortrows(matchIdx, 1);

        learnedED = zeros(N, d);

        for i = 1:N
            if rand() < 0.8
                component_idx = matchIdx(i, 2);
            else
                other_indices = setdiff(1:N, matchIdx(i,2));
                component_idx = other_indices(randi(length(other_indices)));
            end
        
            mu = gmm.mu(component_idx, :);
            Sigma = gmm.Sigma(:, :, component_idx);
            
            learnedED(i, :) = mvnrnd(mu, Sigma);
        end
    end

    ED = zeros(N, Problem.D);
    ED(:, KeyVarIndex) = learnedED;

    for i = 1:N
        Archive{i} = [];
    end
end

