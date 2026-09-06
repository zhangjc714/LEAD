classdef LEAD < ALGORITHM
% LEAD: Learning from Evolutionary directions for convergence Acceleration and Dimensionality reduction

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        info = [];
        KVIs = [];
    end

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [t1,t2,scale,prob,ratio,impLB,impUB,wrsLB,wrsUB,maxNo] = ...
                Algorithm.ParameterSet(10,10,1.0,0.3,0.1,0.0,1.0,0.1,1.0,300);
                
            params = struct('t1', t1, 't2', t2, 'prob', prob, 'ratio', ratio, 'maxNo', maxNo, ...
                    'reg', 1e-3, 'MaxIter', 1000, 'TolFun', 1e-4, 'scale', scale,  'numTree', 100, ...
                    'impLB', impLB, 'impUB', impUB, 'wrsLB', wrsLB, 'wrsUB', wrsUB);
            
            %% Initialization
            [Archive,Dataset,Population,W,B,Z] = Initialization(Problem);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Optimization in the original high-dimension decision space
                gen = 1;
                while gen <= params.t1 && Algorithm.NotTerminated(Population)
                    [Population,Archive,Dataset,Z] = OptimizationInOriginalSpace(Problem,Population,Archive,Dataset,B,Z,W,params);
                    gen = gen + 1;
                end

                % Key variable analysis & Evolutionary direction modeling
                [KeyVarIndex,Dataset] = KeyVariableAnalysis(Algorithm,Problem,Dataset,params);
                [ED,Archive] = EvoDirectionModeling(Problem,Archive,KeyVarIndex,params);
                
                % Optimization in the reduced low-dimension decision space
                gen = 1;
                while gen <= params.t2 && Algorithm.NotTerminated(Population)
                    [Population,Z] = OptimizationInReducedSpace(Problem,KeyVarIndex,Population,ED,B,Z,W,params);
                    gen = gen + 1;
                end
            end
        end
    end
end


function [Archive,Dataset,Population,W,B,Z] = Initialization(Problem)
    Archive = cell(1,Problem.N);
    for i = 1:Problem.N
        Archive{i} = [];
    end

    Dataset = struct('EDs', [], 'labels', []);

    % Generate the weight vectors
    [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
    T = ceil(Problem.N/10);

    % Detect the neighbours of each solution
    B = pdist2(W,W);
    [~,B] = sort(B,2);
    B = B(:,1:T);

    % Generate random population
    Population = Problem.Initialization();
    Z = min(Population.objs,[],1);
end


function [Population,Archive,Dataset,Z] = OptimizationInOriginalSpace(Problem,Population,Archive,Dataset,B,Z,W,params)
    for i = 1 : Problem.N
        % Choose the parents
        if rand() < 0.9
            P = B(i,randperm(size(B,2)));
        else
            P = randperm(Problem.N);
        end

        % Generate an offspring
        Offspring = OperatorDE(Problem,Population(i),Population(P(1)),Population(P(2)));

        % Update the ideal point
        Z = min(Z,Offspring.obj);

        % Update the solutions in P by modified Tchebycheff approach
        g_old = max(abs(Population(P).objs-repmat(Z,length(P),1))./W(P,:),[],2);
        g_new = max(repmat(abs(Offspring.obj-Z),length(P),1)./W(P,:),[],2);
        
        % Collect data
        [Archive,Dataset] = CollectData(Archive,Dataset,Population,Offspring,P,g_old,g_new,i,params);
  
        % Update neighborhood
        Population(P(g_old>=g_new)) = Offspring;
    end
end


function [Archive,Dataset] = CollectData(Archive,Dataset,Population,Offspring,P,g_old,g_new,i,params)
    DeltaG = (g_old-g_new)./max(g_old,g_new);

    idx1 = (DeltaG > 0);
    if any(idx1) && sum(Dataset.labels>0)<params.maxNo
        Archive{i} = [Archive{i}; Offspring.dec - Population(P(idx1)).decs];
        Dataset.EDs = [Dataset.EDs; Offspring.dec - Population(P(idx1)).decs];
        Dataset.labels = [Dataset.labels; DeltaG(idx1)];
    end

    idx2 = (DeltaG < -0.1);
    if any(idx2) && sum(Dataset.labels<0)<params.maxNo
        Dataset.EDs = [Dataset.EDs; Offspring.dec - Population(P(idx2)).decs];
        Dataset.labels = [Dataset.labels; DeltaG(idx2)];
    end
end


function [Population,Z] = OptimizationInReducedSpace(Problem,KeyVarIndex,Population,ED,B,Z,W,params)
    for i = 1 : Problem.N
        % Choose the parents
        if rand() < 0.9
            P = B(i,randperm(size(B,2)));
        else
            P = randperm(Problem.N);
        end
        
        if rand() < params.prob
            % ED-guided Reproduction
            OffDec = Population(i).decs + params.scale * ED(i,:);
            Offspring = Problem.Evaluation(OffDec);
        else
            % MOEA/D Reproduction
            OffDec = Population(i).dec;
            NewDec = OperatorDE(Problem,Population(i).decs,Population(P(1)).decs,Population(P(2)).decs);
            OffDec(KeyVarIndex) = NewDec(KeyVarIndex);
            Offspring = Problem.Evaluation(OffDec);
        end

        % Update the ideal point
        Z = min(Z,Offspring.obj);

        % Update the solutions in P by modified Tchebycheff approach
        g_old = max(abs(Population(P).objs-repmat(Z,length(P),1))./W(P,:),[],2);
        g_new = max(repmat(abs(Offspring.obj-Z),length(P),1)./W(P,:),[],2);
        Population(P(g_old>=g_new)) = Offspring;
    end
end

