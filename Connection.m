classdef Connection < handle
    properties(Constant)
        CONNECT_DISABLE         = 0;
        CONNECT_FULL            = Connection.CONNECT_DISABLE + 1;
        CONNECT_DECISION_FOREST = Connection.CONNECT_DISABLE + 2;

        CONNECT_DECISION_FOREST_OPT      = Connection.CONNECT_DECISION_FOREST + 1;
        CONNECT_DECISION_FOREST_TREE     = Connection.CONNECT_DECISION_FOREST_OPT + 1;
        CONNECT_DECISION_FOREST_NUM      = Connection.CONNECT_DECISION_FOREST_OPT + 2;
        CONNECT_DECISION_FOREST_PRUNEOUT = Connection.CONNECT_DECISION_FOREST_OPT + 3;

        GRADIENT_DISABLE = 10;
        GRADIENT_SGD     = Connection.GRADIENT_DISABLE + 1;
        GRADIENT_ADAGRAD = Connection.GRADIENT_DISABLE + 2;

        GRADIENT_OPT      = 20;
        GRADIENT_LEARN    = Connection.GRADIENT_OPT + 1;
        GRADIENT_MOMENTUM = Connection.GRADIENT_OPT + 2;

        REGULATOR_DISABLE = 30;
        REGULATOR_SQUARED = Connection.REGULATOR_DISABLE + 1;

        REGULATOR_OPT    = 40;
        REGULATOR_LAMBDA = Connection.REGULATOR_OPT + 1;
    end

    properties
        isInit;
        type;

        learn;
        momentum;
        lambda;

        attr;
        aux;
    end

    methods
        function connection = Connection(varargin)
            connection.isInit = false;

            switch(nargin)
                case {0, 1}
                    connection.type.connect = Connection.CONNECT_DISABLE;
                    connection.type.gradient = Connection.GRADIENT_DISABLE;
                    connection.type.regulator = Connection.REGULATOR_DISABLE;
                    connection.learn = 0;
                    connection.momentum = 0;
                    connection.lambda = 0;
                case {2}
                    c = varargin{2};
                    connection.type = c.type;
                    connection.learn = c.learn;
                    connection.momentum = c.momentum;
                    connection.lambda = c.lambda;
                otherwise
                    error('Wrong argument number!');
            end

            switch(nargin)
                case {1, 2}
                    type = varargin{1};
                    i = 1;
                    while(i <= length(type))
                        switch(type(i))
                            case {Connection.CONNECT_FULL}
                                connection.type.connect = type(i);
                            case {Connection.CONNECT_DECISION_FOREST}
                                connection.type.connect = type(i);
                                connection.attr.tree = 1;
                                connection.attr.num = 0;
                                connection.attr.pruneout = 0;
                            case {Connection.CONNECT_DECISION_FOREST_TREE}
                                i = i + 1;
                                connection.attr.tree = type(i);
                            case {Connection.CONNECT_DECISION_FOREST_NUM}
                                i = i + 1;
                                connection.attr.num = type(i);
                            case {Connection.CONNECT_DECISION_FOREST_PRUNEOUT}
                                i = i + 1;
                                connection.attr.pruneout = type(i);
                            case {Connection.GRADIENT_SGD, Connection.GRADIENT_ADAGRAD}
                                connection.type.gradient = type(i);
                            case {Connection.GRADIENT_LEARN}
                                i = i + 1;
                                connection.learn = type(i);
                            case {Connection.GRADIENT_MOMENTUM}
                                i = i + 1;
                                connection.momentum = type(i);
                            case {Connection.REGULATOR_SQUARED}
                                connection.type.regulator = type(i);
                            case {Connection.REGULATOR_LAMBDA}
                                i = i + 1;
                                connection.lambda = type(i);
                        end
                        i = i + 1;
                    end
                otherwise
                    %
            end
        end

        function init(connection, dimFrom, dimTo, gpu, real)
            connection.isInit = true;
            
            switch(connection.type.connect)
                case Connection.CONNECT_FULL
                    connection.attr.weight = ...
                        [Util.zeros(dimTo, 1), ...
                         (Util.rand(dimTo, dimFrom) - 0.5) * 2 / sqrt(dimFrom)];
                    connection.attr.gradient = Util.epsilon * Util.ones(dimTo, dimFrom + 1);
                case Connection.CONNECT_DECISION_FOREST
                    if(connection.attr.tree > 1)
                        connection.attr.logDim = log2(connection.attr.num + 1);
                    else
                        connection.attr.logDim = log2(dimFrom + 1);
                        connection.attr.num = dimFrom;
                    end
                    sample = (connection.attr.num * connection.attr.tree > dimFrom);

                    if(connection.attr.num > dimFrom)
                        error('Sample dimension cannot be bigger than original dimension!');
                    end
                    if(floor(connection.attr.logDim) ~= connection.attr.logDim)
                        error('Dimension must be in the form of 2^n - 1!');
                    end

                    connection.attr.u = Util.cast([]);
                    connection.attr.v = Util.cast([]);
                    for i = 0:(connection.attr.logDim - 1)
                        dim = pow2(i);
                        connection.attr.u = ...
                            [ Util.ones(dim, 1),        connection.attr.u, Util.zeros(dim, dim - 1); ...
                             Util.zeros(dim, 1), Util.zeros(dim, dim - 1),       connection.attr.u];
                        connection.attr.v = ...
                            [Util.zeros(dim, 1),        connection.attr.v, Util.zeros(dim, dim - 1); ...
                              Util.ones(dim, 1), Util.zeros(dim, dim - 1),       connection.attr.v];
                    end

                    I = Util.eye(connection.attr.num + 1);
                    connection.attr.s(((connection.attr.num + 1) / 2):connection.attr.num, (0:connection.attr.num) + 1) = I((1:2:connection.attr.num) + 0, :);
                    connection.attr.t(((connection.attr.num + 1) / 2):connection.attr.num, (0:connection.attr.num) + 1) = I((1:2:connection.attr.num) + 1, :);
                    for i = ((connection.attr.num - 1) / 2):-1:1
                        connection.attr.s(i, :) = connection.attr.s(2 * i + 0, :) + connection.attr.t(2 * i + 0, :);
                        connection.attr.t(i, :) = connection.attr.s(2 * i + 1, :) + connection.attr.t(2 * i + 1, :);
                    end

                    connection.attr.weight = cell(connection.attr.tree, 1);
                    connection.attr.gradient = cell(connection.attr.tree, 1);
                    connection.attr.sample = cell(connection.attr.tree, 1);
                    for i = 1:connection.attr.tree
                        connection.attr.weight{i} = Util.normalize(exp(Util.rand(dimTo, connection.attr.num + 1)));
                        connection.attr.gradient{i} = Util.epsilon * Util.ones(dimTo, connection.attr.num + 1);
                        if(sample)
                            connection.attr.sample{i} = Util.cast(bsxfun(@eq, randsample(dimFrom, connection.attr.num), 1:dimFrom));
                        else
                            connection.attr.sample{i} = Util.cast(bsxfun(@eq, ((1:connection.attr.num) + (i - 1) * connection.attr.num)', 1:dimFrom));
                        end
                    end
                    
                    if(~Util.gpu)
                        connection.attr.u = sparse(connection.attr.u);
                        connection.attr.v = sparse(connection.attr.v);
                        connection.attr.s = sparse(connection.attr.s);
                        connection.attr.t = sparse(connection.attr.t);
                        for i = 1:connection.attr.tree
                            connection.attr.sample{i} = sparse(connection.attr.sample{i});
                        end
                    end

                    connection.aux.sigmoid = cell(connection.attr.tree, 1);
                    connection.aux.mu = cell(connection.attr.tree, 1);
                    connection.aux.value = cell(connection.attr.tree, 1);
                    connection.aux.error = cell(connection.attr.tree, 1);
            end
        end

        function forward(connection, layerFrom, layerTo, batchSize, opt)
            switch(connection.type.connect)
                case Connection.CONNECT_DISABLE
                    error('No connection!');
                case Connection.CONNECT_FULL
                    layerTo.value = layerTo.value + connection.attr.weight * Util.pad(layerFrom.value, batchSize);
                case Connection.CONNECT_DECISION_FOREST
                    for i = 1:connection.attr.tree
                        connection.aux.sigmoid{i} = 1 ./ (1 + exp(-connection.attr.sample{i} * layerFrom.value));
                        if(connection.attr.pruneout > 0)
                            switch(opt.flag)
                                case Opt.TRAIN
                                    pruneout = (Util.rand(connection.attr.num, batchSize) > connection.attr.pruneout);
                                    leftRight = (Util.rand(connection.attr.num, batchSize) > 0.5);
                                    connection.aux.sigmoid{i} = connection.aux.sigmoid{i} .* pruneout + leftRight .* (1 - pruneout);
                                case Opt.TEST
                                    connection.aux.sigmoid{i} = connection.attr.pruneout * connection.aux.sigmoid{i} + (1 - connection.attr.pruneout) / 2;
                            end
                        end
                        connection.aux.mu{i} = exp(connection.attr.u * log(connection.aux.sigmoid{i}) ...
                                                   + connection.attr.v * log(1 - connection.aux.sigmoid{i}));
                        connection.aux.value{i} = connection.attr.weight{i} * connection.aux.mu{i}; 
                        layerTo.value = layerTo.value + connection.aux.value{i} / connection.attr.tree;
                    end
            end
        end

        function backward(connection, layerFrom, layerTo, batchSize, opt)
            switch(connection.type.connect)
                case Connection.CONNECT_DISABLE
                    error('No connection!');
                case Connection.CONNECT_FULL
                    layerFrom.error = layerFrom.error + connection.attr.weight(:, 2:end)' * layerTo.error;
                case Connection.CONNECT_DECISION_FOREST
                    for i = 1:connection.attr.tree
                        A = Util.normalize((connection.attr.weight{i}' * layerTo.error) .* connection.aux.mu{i});
                        connection.aux.error{i} = connection.aux.sigmoid{i} .* (connection.attr.t * A) ...
                                                  - (1 - connection.aux.sigmoid{i}) .* (connection.attr.s * A);
                        layerFrom.error = layerFrom.error + connection.attr.sample{i}' * connection.aux.error{i} / connection.attr.tree;
                    end
            end
        end

        function update(connection, layerFrom, layerTo, batchSize, opt)
            if(connection.type.gradient == Connection.GRADIENT_DISABLE)
                return;
            end
            switch(connection.type.connect)
                case Connection.CONNECT_DISABLE
                    error('No connection!');
                case Connection.CONNECT_FULL
                    gradient = layerTo.error * (Util.pad(layerFrom.value, batchSize))' / batchSize;
                    [gradient, connection.attr.gradient] = connection.accelerate(gradient, connection.attr.gradient);
                    [connection.attr.weight] = connection.regulate(connection.attr.weight);
                    connection.attr.weight = connection.attr.weight - gradient;
                case Connection.CONNECT_DECISION_FOREST
                    for i = 1:connection.attr.tree
                        P = Util.normalize(connection.attr.weight{i} .* ((layerTo.error .* connection.aux.value{i}) * connection.aux.mu{i}'));
                        gradient = connection.attr.weight{i} - P;
                        [gradient, connection.attr.gradient{i}] = connection.accelerate(gradient, connection.attr.gradient{i});
                        [connection.attr.weight{i}] = connection.regulate(connection.attr.weight{i});
                        connection.attr.weight{i} = connection.attr.weight{i} - gradient;
                    end
            end
        end

        function [gradient, gradientHist] = accelerate(connection, gradient, gradientHist)
            switch(connection.type.gradient)
                case Connection.GRADIENT_SGD
                    % V+ = momentum * V + learn * gradient
                    % W+ = W - V+
                    gradient = connection.learn * gradient;
                    if(connection.momentum)
                        gradient = gradient + connection.momentum * gradientHist;
                    end
                    gradientHist = gradient;
                case Connection.GRADIENT_ADAGRAD
                    % V+ = V + gradient .* gradient
                    % W+ = W - learn * gradient ./ sqrt(V+)
                    gradientHist = gradientHist + gradient .* gradient;
                    gradient = connection.learn * gradient ./ sqrt(gradientHist);
            end
        end

        function [weight] = regulate(connection, weight)
            switch(connection.type.regulator)
                case Connection.REGULATOR_SQUARED
                    % W+ = (1 - lambda) * W - gradient
                    if(connection.lambda)
                        weight = (1 - connection.lambda) * weight;
                    end
            end
        end
    end
end
