classdef Layer < handle
    properties(Constant)
        IO_DISABLE = 0;
        IO_INPUT   = Layer.IO_DISABLE + 1;
        IO_OUTPUT  = Layer.IO_DISABLE + 2;

        OP_DISABLE      = 20;
        OP_RELU         = Layer.OP_DISABLE + 1;
        OP_FAST_SIGMOID = Layer.OP_DISABLE + 2;

        DROPOUT_DISABLE = 30;
        DROPOUT         = Layer.DROPOUT_DISABLE + 1;

        LOSS_DISABLE       = 40;
        LOSS_SQUARED         = Layer.LOSS_DISABLE + 1;
        LOSS_SQUARED_RATIO   = Layer.LOSS_DISABLE + 2;
        LOSS_CROSS_ENTROPY   = Layer.LOSS_DISABLE + 3;
        LOSS_DECISION_FOREST = Layer.LOSS_DISABLE + 4;
        
        LOSS_OPT    = 50;
        LOSS_WEIGHT = Layer.LOSS_OPT + 1;
    end

    properties
        id;
        dimension;
        type;
        weight;

        value;
        error;
        dropout;
        aux;
        
        next;
        prev;
        connections;
    end

    methods
        function layer = Layer(dimension, type)
            layer.dimension = dimension;
           
            layer.type.io = Layer.IO_DISABLE;
            layer.type.op = Layer.OP_DISABLE;
            layer.type.dropout = Layer.DROPOUT_DISABLE;
            layer.type.loss = Layer.LOSS_DISABLE;
            for i = 1:length(type)
                switch(type(i))
                    case {Layer.IO_DISABLE, Layer.IO_INPUT, Layer.IO_OUTPUT}
                        layer.type.io = type(i);
                    case {Layer.OP_DISABLE, Layer.OP_RELU, Layer.OP_FAST_SIGMOID}
                        layer.type.op = type(i);
                    case {Layer.DROPOUT_DISABLE, Layer.DROPOUT}
                        layer.type.dropout = type(i);
                    case {Layer.LOSS_DISABLE, Layer.LOSS_SQUARED, Layer.LOSS_CROSS_ENTROPY, Layer.LOSS_DECISION_FOREST}
                        layer.type.loss = type(i);
                        layer.weight = 1;
                    case {Layer.LOSS_WEIGHT}
                        i = i + 1;
                        layer.weight = type(i);
                end
            end

            layer.next = [];
            layer.prev = [];
            layer.connections = Connection.empty(0);
        end

        function next = addNext(layer, next, connection)
            layer.next = [layer.next, next];
            next.prev = [next.prev, layer];
            layer.connections = [layer.connections, connection];
        end

        function prev = addPrev(layer, prev, connection)
            layer.prev = [layer.prev, prev];
            prev.next = [prev.next, layer];
            layer.connections = [layer.connections, connection];
        end

        function clean(layer, batchSize, opt)
            layer.value = Util.zeros(layer.dimension, batchSize);
            layer.error = Util.zeros(layer.dimension, batchSize);
            if(layer.type.dropout ~= Layer.DROPOUT_DISABLE)
                layer.dropout = Util.zeros(layer.dimension, batchSize);
            end
            if(layer.type.loss ~= Layer.LOSS_DISABLE)
                layer.aux = Util.zeros(layer.dimension, batchSize);
            end
        end

        function feed(layer, batch)
            layer.value = batch;
        end

        function forward(layer, batchSize, opt)
            switch(layer.type.op)
                case Layer.OP_DISABLE
                case Layer.OP_RELU
                    layer.value = layer.value .* (layer.value > 0);
                case Layer.OP_FAST_SIGMOID
                    layer.value = layer.value ./ (1 + abs(layer.value));
            end

            switch(layer.type.dropout)
                case Layer.DROPOUT_DISABLE
                case Layer.DROPOUT
                    switch(opt.flag)
                        case Opt.TRAIN
                            if(isempty(layer.dropout))
                                layer.dropout = (Util.rand(layer.dimension, batchSize) > opt.dropout);
                            end
                            layer.value = layer.value .* layer.dropout;
                        case Opt.TEST
                            layer.value = layer.value * (1 - opt.dropout);
                    end
            end

            if(layer.type.io == Layer.IO_OUTPUT)
                switch(layer.type.loss)
                    case Layer.LOSS_DISABLE
                        error('IO_OUTPUT without LOSS set!')
                    case Layer.LOSS_SQUARED
                    case Layer.LOSS_SQUARED_RATIO
                    case Layer.LOSS_CROSS_ENTROPY
                        layer.aux = exp(bsxfun(@minus, layer.value, max(layer.value)));
                        layer.aux = bsxfun(@rdivide, layer.aux, sum(layer.aux));
                    case Layer.LOSS_DECISION_FOREST
                end
            end
        end

        function error = collect(layer, batch)
            switch(layer.type.loss)
                case Layer.LOSS_DISABLE
                case Layer.LOSS_SQUARED
                    layer.error = layer.value - batch;
                    error = gather(1 / 2 * sum(sum(layer.error .* layer.error)));
                case Layer.LOSS_SQUARED_RATIO
                    temp = batch(1, :) - 1 / 2;
                    layer.error = bsxfun(@times, layer.value - batch(2:end, :), temp); 
                    error = gather(sum(log(sum(layer.error .* layer.error)) .* temp));
                case Layer.LOSS_CROSS_ENTROPY
                    layer.error = layer.aux - batch;
                    error = - gather(sum(sum(batch .* log(layer.aux) + (1 - batch) .* log(1 - layer.aux))));
                case Layer.LOSS_DECISION_FOREST
                    layer.error = batch;
                    error = - gather(sum(log(sum(layer.value .* layer.error))));
            end
            layer.error = layer.weight * layer.error;
        end

        function backward(layer, batchSize, opt)
            switch(layer.type.op)
                case Layer.OP_DISABLE
                case Layer.OP_RELU
                    layer.error = layer.error .* (layer.value > 0);
                case Layer.OP_FAST_SIGMOID
                    temp = 1 - abs(layer.value);
                    layer.error = layer.error .* temp .* temp;
            end

            switch(layer.type.dropout)
                case Layer.DROPOUT_DISABLE
                case Layer.DROPOUT
                    layer.error = layer.error .* layer.dropout;
            end
        end
    end
end
