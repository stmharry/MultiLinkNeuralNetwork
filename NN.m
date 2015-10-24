classdef NN < handle
    properties(Constant)
        CONTENT  = 0;
        TITLE    = NN.CONTENT + 1;
        PROGRESS = NN.CONTENT + 2;
        HEADER   = NN.CONTENT + 3;
        BODY     = NN.CONTENT + 4;
        FOOTER   = NN.CONTENT + 5;
    end

    properties
        gpu;
        real = 'single';
        epsilon = 1e-9;

        dataset;
        layers;
        connections;
        opt;
        error;

        input;
        output;
        link;

        batchSize;
        totalSize;
        accumSize;
        progress;
    end

    methods
        function nn = NN(arg)
            if(isa(arg, 'char'))
                file = arg;
                % TODO
            elseif(isa(arg, 'Dataset'))
                dataset = arg;

                nn.gpu = (gpuDeviceCount > 0);
                if(nn.gpu)
                    nn.real = 'single';
                end

                nn.dataset = dataset;
                net = dataset.getNet();
                nn.opt = dataset.getOpt();

                nn.layers = net.layers;
                layerNum = length(nn.layers);
                nn.input = sparse(layerNum, 1);
                nn.output = sparse(layerNum, 1);
                for i = 1:layerNum
                    nn.layers(i).id = i;
                end
                nn.cache();

                nn.totalSize = 0;
                nn.accumSize = 0;
            end
        end

        function cache(nn)
            layerNum = length(nn.layers);

            nn.connections = Connection.empty(0);
            nn.connections(layerNum, layerNum) = Connection();
            for i = 1:layerNum
                layerI = nn.layers(i);
                nn.input(i) = (layerI.type.io == Layer.IO_INPUT); 
                nn.output(i) = (layerI.type.io == Layer.IO_OUTPUT);
                for j = 1:length(layerI.next)
                    layerJ = layerI.next(j);
                    nn.connections(i, layerJ.id) = layerI.connections(j);
                end
            end

            nn.link = struct([]);
            out = find(nn.output);
            if(length(out))
                for j = out'
                    layerJ = nn.layers(j);
                    for layerI = layerJ.prev
                        nn.link = [nn.link, struct('from', layerI.id, 'to', j)];
                    end
                end
            end

            pos = 1;
            check = sparse(layerNum, layerNum);
            while(pos <= length(nn.link))
                j = nn.link(pos).from;
                layerJ = nn.layers(j);
                if(layerJ.type.io ~= Layer.IO_INPUT)
                    for layerI = layerJ.prev
                        if(~check(layerI.id, j))
                            nn.link = [nn.link, struct('from', layerI.id, 'to', j)];
                            check(layerI.id, j) = true;
                        end
                    end
                end
                pos = pos + 1;
            end
            nn.link = fliplr(nn.link);

            for i = 1:length(nn.link)
                f = [nn.link(i).from];
                t = [nn.link(i).to];
                dimFrom = nn.layers(f).dimension;
                dimTo = nn.layers(t).dimension;
  
                connection = nn.connections(f, t);
                if(~connection.init)
                    connection.init = true;
                    connection.weight = ...
                        [NN.zeros(dimTo, 1, nn.gpu, nn.real), ...
                         (NN.rand(dimTo, dimFrom, nn.gpu, nn.real) - 0.5) * 2 / sqrt(dimFrom)];
                    connection.gradient = nn.epsilon * NN.ones(dimTo, dimFrom + 1, nn.gpu, nn.real);
                    connection.type = nn.opt.connectionType;
                    connection.learn = nn.opt.learn;
                    connection.momentum = nn.opt.momentum;
                    connection.lambda = nn.opt.lambda;
                end
            end
            nn.error = zeros(sum(nn.output), 1);
        end

        function trainFor(nn, extraSampleNum)
            nn.opt.sampleNum = nn.opt.sampleNum + extraSampleNum;
            nn.train();
        end

        function train(nn)
            dataset = nn.dataset;
            opt = nn.opt;

            opt.flag = Opt.TRAIN;
            dataset.configure(opt);

            nn.info(Opt.TRAIN, NN.TITLE);
            nn.info(Opt.TRAIN, NN.HEADER);

            tic;
            while(true)
                nn.batchSize = dataset.getDataBatch(opt);
                if(nn.batchSize == 0)
                    break;
                end

                nn.clean();
                nn.feed(dataset.inBatch);
                nn.forward();
                nn.collect(dataset.outBatch);
                nn.backward();
                nn.update();

                nn.accumSize = nn.accumSize + nn.batchSize;
                while(dataset.totalSize >= nn.totalSize + opt.reportInterval)
                    nn.totalSize = nn.totalSize + opt.reportInterval;
                    nn.info(Opt.TRAIN, NN.BODY);
                end
            end
            nn.info(Opt.TRAIN, NN.FOOTER);
        end
        
        function test(nn)
            dataset = nn.dataset;
            opt = nn.opt;

            opt.flag = Opt.TEST;
            dataset.configure(opt);
            dataset.preTest();

            nn.info(Opt.TEST, NN.TITLE);
            out = find(nn.output);
            nn.progress = 0;

            tic;
            while(true)
                nn.batchSize = dataset.getDataBatch(opt); 
                if(nn.batchSize == 0)
                    break;
                end
                
                nn.clean();
                nn.feed(dataset.inBatch);
                nn.forward();
                if(opt.collect)
                    nn.collect(dataset.outBatch);
                end

                nn.info(Opt.TEST, NN.PROGRESS);
                dataset.processTestBatch(nn.layers(out));
            end

            nn.info(Opt.TEST, NN.HEADER);
            nn.info(Opt.TEST, NN.BODY);
            nn.info(Opt.TEST, NN.FOOTER);
            dataset.showTestInfo();
        end

        function clean(nn)
            for i = 1:length(nn.layers)
                layer = nn.layers(i);
                layer.value = NN.zeros(layer.dimension, nn.batchSize, nn.gpu, nn.real);
                layer.error = NN.zeros(layer.dimension, nn.batchSize, nn.gpu, nn.real);
                if(layer.type.dropout ~= Layer.DROPOUT_DISABLE)
                    layer.dropout = NN.zeros(layer.dimension, nn.batchSize, nn.gpu, nn.real);
                end
                if(layer.type.loss ~= Layer.LOSS_DISABLE)
                    layer.aux = NN.zeros(layer.dimension, nn.batchSize, nn.gpu, nn.real);
                end
            end
        end

        function feed(nn, inBatch)
            in = find(nn.input);
            for i = 1:length(in)
                nn.layers(in(i)).value = inBatch{i};
            end
        end

        function forward(nn)
            opt = nn.opt;

            for i = 1:length(nn.link)
                f = [nn.link(i).from];
                t = [nn.link(i).to];
                layerFrom = nn.layers(f);
                layerTo = nn.layers(t);
                
                switch(layerTo.type.lu)
                    case Layer.LU_DISABLE
                        error('No connection!');
                    case Layer.LU
                        layerTo.value = layerTo.value + nn.connections(f, t).weight * NN.pad(layerFrom.value, nn.batchSize, nn.gpu, nn.real);
                end
    
                switch(layerTo.type.op)
                    case Layer.OP_DISABLE
                    case Layer.OP_RELU
                        layerTo.value = layerTo.value .* (layerTo.value > 0);
                    case Layer.OP_FAST_SIGMOID
                        layerTo.value = layerTo.value ./ (1 + abs(layerTo.value));
                end

                switch(layerTo.type.dropout)
                    case Layer.DROPOUT_DISABLE
                    case Layer.DROPOUT
                        if(opt.flag == Opt.TRAIN)
                            if(isempty(layerTo.dropout))
                                layerTo.dropout = (NN.rand(layerTo.dimension, nn.batchSize, nn.gpu, nn.real) > opt.dropout);
                            end
                            layerTo.value = layerTo.value .* layerTo.dropout;
                        elseif(opt.flag == Opt.TEST)
                            layerTo.value = layerTo.value * (1 - opt.dropout);
                        end
                end

                if(layerTo.type.io == Layer.IO_OUTPUT)
                    switch(layerTo.type.loss)
                        case Layer.LOSS_DISABLE
                            error('IO_OUTPUT without LOSS set!')
                        case Layer.LOSS_SQUARED
                        case Layer.LOSS_SQUARED_RATIO
                        case Layer.LOSS_CROSS_ENTROPY
                            layerTo.aux = exp(bsxfun(@minus, layerTo.value, max(layerTo.value)));
                            layerTo.aux = bsxfun(@rdivide, layerTo.aux, sum(layerTo.aux));
                    end
                end
            end
        end

        function collect(nn, outBatch)
            out = find(nn.output);
            for i = 1:length(out)
                layer = nn.layers(out(i));

                switch(layer.type.loss)
                    case Layer.LOSS_DISABLE
                    case Layer.LOSS_SQUARED
                        layer.error = layer.value - outBatch{i};
                        nn.error(i) = nn.error(i) + gather(1 / 2 * sum(sum(layer.error .* layer.error)));
                    case Layer.LOSS_SQUARED_RATIO
                        temp = outBatch{i}(1, :) - 1 / 2;
                        layer.error = bsxfun(@times, layer.value - outBatch{i}(2:end, :), temp); 
                        nn.error(i) = nn.error(i) + gather(sum(log(sum(layer.error .* layer.error)) .* temp));
                    case Layer.LOSS_CROSS_ENTROPY
                        layer.error = layer.aux - outBatch{i};
                        nn.error(i) = nn.error(i) - gather(sum(sum(outBatch{i} .* log(layer.aux))));
                end

                layer.error = layer.weight * layer.error;
            end
        end

        function backward(nn)
            for i = 1:length(nn.link)
                f = [nn.link(i).from];
                t = [nn.link(i).to];
                layerFrom = nn.layers(f);
                layerTo = nn.layers(t);

                switch(layerTo.type.op)
                    case Layer.OP_DISABLE
                    case Layer.OP_RELU
                        layerTo.error = layerTo.error .* (layerTo.value > 0);
                    case Layer.OP_FAST_SIGMOID
                        temp = 1 - abs(layerTo.value);
                        layerTo.error = layerTo.error .* temp .* temp;
                end

                switch(layerTo.type.dropout)
                    case Layer.DROPOUT_DISABLE
                    case Layer.DROPOUT
                        layerTo.error = layerTo.error .* layerTo.dropout;
                end

                switch(layerTo.type.lu)
                    case Layer.LU_DISABLE
                    case Layer.LU
                        layerFrom.error = layerFrom.error + nn.connections(f, t).weight(:, 2:end)' * layerTo.error;
                end
            end
        end

        function update(nn)
            opt = nn.opt;

            for i = 1:length(nn.link)
                f = [nn.link(i).from];
                t = [nn.link(i).to];
                layerFrom = nn.layers(f);
                layerTo = nn.layers(t);
                connection = nn.connections(f, t);

                gradient = layerTo.error * (NN.pad(layerFrom.value, nn.batchSize, nn.gpu, nn.real))' / nn.batchSize;
                switch(connection.type.gradient)
                    case Connection.GRADIENT_DISABLE
                        error('No update method set!');
                    case Connection.GRADIENT_SGD
                        % V+ = momentum * V + learn * gradient
                        % W+ = W - V+
                        gradient = connection.learn * gradient;
                        if(connection.momentum)
                            gradient = gradient + connection.momentum * connection.gradient;
                        end
                        connection.gradient = gradient;
                    case Connection.GRADIENT_ADAGRAD
                        % V+ = V + gradient .* gradient
                        % W+ = W - learn * gradient ./ sqrt(V+)
                        connection.gradient = connection.gradient + gradient .* gradient;
                        gradient = connection.learn * gradient ./ sqrt(connection.gradient);
                end

                switch(connection.type.regulator)
                    case Connection.REGULATOR_SQUARED
                        % W+ = (1 - lambda) * W - gradient
                        if(connection.lambda)
                            connection.weight = (1 - connection.lambda) * connection.weight;
                        end
                    case Connection.REGULATOR_LOGARITHM
                        % W+ = (1 + lambda / |W| ^ 2 ) * W - gradient
                        if(connection.lambda)
                            temp = sum(sum(connection.weight .* connection.weight)) / layerTo.dimension;
                            connection.weight = (1 + connection.lambda / temp) * connection.weight;
                        end
                end
                connection.weight = connection.weight - gradient;
            end
        end

        function info(nn, flag, content)
            switch(content)
                case NN.TITLE
                    switch(flag)
                        case Opt.TRAIN
                            tcprintf('red', '[ DNN Training ]\n');
                        case Opt.TEST
                            space = 59;
                            tcprintf('red', '[ DNN Testing ]\n');
                            tcprintf('green', ['0 |', repmat(char(32), 1, space), '| 100', repmat(char(8), 1, space + 5)]);
                    end
                case NN.PROGRESS
                    space = 59;
                    while(nn.dataset.totalSize > nn.dataset.sampleNum / space * nn.progress)
                        tcprintf('green', '-');
                        nn.progress = nn.progress + 1;
                    end
                    if(nn.dataset.totalSize == nn.dataset.sampleNum)
                        fprintf('\n');
                    end
                case NN.HEADER
                    tcprintf('blue', '+--------------+---------------+----------------------------------+ \n');
                    tcprintf('blue', '| Sample Count | Time Used (s) | Error                            | \n');
                    tcprintf('blue', '+--------------+---------------+----------------------------------+ \n');
                case NN.BODY
                    time = toc;
                    switch(flag)
                        case Opt.TRAIN
                            totalSize = nn.totalSize;
                            str = nn.errorToStr(Opt.TRAIN);
                        case Opt.TEST
                            totalSize = nn.dataset.totalSize;
                            if(nn.opt.collect)
                                str = nn.errorToStr(Opt.TEST);
                            else
                                str = 'N/A';
                            end
                    end
                    tcprintf('blue', '| %12d | %13.3f | %-32s | \n', totalSize, time, str);
                    
                    nn.error = zeros(sum(nn.output), 1);
                    nn.accumSize = 0;
                    tic;
                case NN.FOOTER
                    tcprintf('blue', '+--------------+---------------+----------------------------------+ \n');
                    switch(flag)
                        case Opt.TRAIN
                        case Opt.TEST
                            tcprintf('red', '[ DNN Testing Extra ]\n');
                    end
            end
        end
        
        function str = errorToStr(nn, flag)
            str = '';
            switch(flag)
                case Opt.TRAIN
                    e = nn.error / nn.accumSize;
                case Opt.TEST
                    e = nn.error / dataset.totalSize;
            end

            out = find(nn.output);
            for o = 1:length(out)
                switch(nn.layers(out(o)).type.loss)
                    case Layer.LOSS_SQUARED
                        str = [str, num2str([e(o), log(e(o))], '%.2f (%.2f) ')];
                    case Layer.LOSS_SQUARED_RATIO
                        str = [str, num2str(e(o), '%.2f ')];
                    case Layer.LOSS_CROSS_ENTROPY
                        str = [str, num2str(e(o), '%.2f ')];
                end
            end
        end
    end
    
    methods(Static)
        function out = zeros(x, y, gpu, real)
            if(gpu)
                out = gpuArray.zeros(x, y, real);
            else
                out = zeros(x, y, real);
            end
        end

        function out = ones(x, y, gpu, real)
            if(gpu)
                out = gpuArray.ones(x, y, real);
            else
                out = ones(x, y, real);
            end
        end

        function out = rand(x, y, gpu, real)
            if(gpu)
                out = gpuArray.rand(x, y, real);
            else
                out = rand(x, y, real);
            end
        end
        
        function out = pad(in, size, gpu, real)
            out = [NN.ones(1, size, gpu, real); in];
        end
    end
end
