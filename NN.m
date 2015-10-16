classdef NN < handle
    properties
        gpu;
        real = 'double';

        dataset;
        blobs;
        opt;

        input;
        output;
        link;

        batchSize;
        totalSize;

        weight;
        gradient;
        error;
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
                    nn.real = 'double';
                end
                nn.dataset = dataset;
                nn.blobs = dataset.getBlobs();
                nn.opt = dataset.getOpt();
                blobNum = length(nn.blobs);
                
                nn.input = sparse(blobNum, 1);
                nn.output = sparse(blobNum, 1);
                nn.link = sparse(blobNum, blobNum);
                for i = blobNum:-1:1
                    blob = nn.blobs(i);
                    blob.id = i;
                    nn.input(i) = (bitand(blob.type, Blob.IO_INPUT) ~= 0);
                    nn.output(i) = (bitand(blob.type, Blob.IO_OUTPUT) ~= 0);
                    for j = blob.next
                        nn.link(i, j.id) = true;
                    end
                end

                nn.totalSize = 0;

                nn.weight = cell(blobNum);
                nn.gradient = cell(blobNum);
          
                if(nn.opt.init)
                    [from, to] = find(nn.link);
                    for i = 1:length(from)
                        f = from(i);
                        t = to(i);
             
                        dimFrom = nn.blobs(f).dimension;
                        dimTo = nn.blobs(t).dimension;
             
                        nn.weight{f, t} = ...
                            [NN.zeros(dimTo, 1, nn.gpu, nn.real), ...
                             (NN.rand(dimTo, dimFrom, nn.gpu, nn.real) - 0.5) * 2 / sqrt(dimFrom)];
                        nn.gradient{f, t} = NN.zeros(dimTo, dimFrom + 1, nn.gpu, nn.real);
                    end
                end
            end
        end

        function train(nn)
            dataset = nn.dataset;
            opt = nn.opt;
            if(dataset.flag ~= Dataset.TRAIN)
                error('Training dataset not loaded!');
            end

            opt.flag = Opt.TRAIN;
            out = find(nn.output);
            nn.error = zeros(length(out), 1);
            accumSize = 0;
     
            tic;
            while(true)
                [inBatch, outBatch, nn.batchSize] = dataset.getBatch(opt);
                if(nn.batchSize == 0)
                    break;
                end

                nn.clean();
                nn.feed(inBatch);
                nn.forward();
                nn.collect(outBatch);
                nn.backward();
                nn.update();

                accumSize = accumSize + nn.batchSize;
                while(dataset.totalSize >= nn.totalSize + opt.reportInterval)
                    time = toc;
                    
                    e = [(nn.error / accumSize)'; log(nn.error / accumSize)'];
                    fprintf('[DNN Training] Sample = %d, Time = %.3f s, Error = %s\n', ...
                        nn.totalSize, time, num2str(e(:)', '%.3f (%.3f) '));

                    nn.error = zeros(sum(nn.output), 1);
                    nn.totalSize = nn.totalSize + opt.reportInterval;
                    accumSize = 0;
                    tic;
                end
            end
        end
        
        function test(nn)
            dataset = nn.dataset;
            opt = nn.opt;
            if(dataset.flag ~= Dataset.TEST)
                error('Testing dataset not loaded!');
            end

            opt.flag = Opt.TEST;
            out = find(nn.output);
            nn.error = zeros(length(out), 1);

            tic;
            while(true)
                [inBatch, outBatch, nn.batchSize] = dataset.getBatch(opt); 
                if(nn.batchSize == 0)
                    break;
                end
                
                nn.clean();
                nn.feed(inBatch);
                nn.forward();
                if(opt.collect)
                    nn.collect(outBatch);
                end

                dataset.postTest(nn.blobs(out));
            end
            time = toc;

            e = [(nn.error / dataset.sampleNum)'; log(nn.error / dataset.sampleNum)'];
            fprintf('[DNN Testing] Time = %.3f s, Error = %s\n', time, num2str(e(:)', '%.3f (%.3f) '));
            dataset.showTestInfo();
        end

        function clean(nn)
            for i = 1:length(nn.blobs)
                blob = nn.blobs(i);
                blob.value = NN.zeros(blob.dimension, nn.batchSize, nn.gpu, nn.real);
                blob.error = NN.zeros(blob.dimension, nn.batchSize, nn.gpu, nn.real);
                if(bitand(blob.type, Blob.DROPOUT))
                    blob.dropout = NN.zeros(blob.dimension, nn.batchSize, nn.gpu, nn.real);
                end
                if(bitand(blob.type, Blob.LOSS_SQUARED) || bitand(blob.type, Blob.LOSS_SOFTMAX))
                    blob.aux = NN.zeros(blob.dimension, nn.batchSize, nn.gpu, nn.real);
                end
            end
        end

        function feed(nn, inBatch)
            in = find(nn.input);
            for i = 1:length(in)
                nn.blobs(in(i)).value = inBatch{i};
            end
        end

        function forward(nn)
            opt = nn.opt;

            [from, to] = find(nn.link);
            for i = 1:length(from)
                f = from(i);
                t = to(i);
                blobFrom = nn.blobs(f);
                blobTo = nn.blobs(t);
                
                if(bitand(blobTo.type, Blob.LU))
                    blobTo.value = blobTo.value + nn.weight{f, t} * NN.pad(blobFrom.value, nn.batchSize, nn.gpu, nn.real);
                end

                if(bitand(blobTo.type, Blob.OP_RELU))
                    blobTo.value = blobTo.value .* (blobTo.value > 0);
                end

                if(bitand(blobTo.type, Blob.OP_FAST_SIGMOID))
                    blobTo.value = blobTo.value ./ (1 + abs(blobTo.value));
                end
                
                if(bitand(blobTo.type, Blob.DROPOUT))
                    if(opt.flag == Opt.TRAIN)
                        if(isempty(blobTo.dropout))
                            blobTo.dropout = (NN.rand(blobTo.dimension, nn.batchSize, nn.gpu, nn.real) > opt.dropout);
                        end
                        blobTo.value = blobTo.value .* blobTo.dropout;
                    elseif(opt.flag == Opt.TEST)
                        blobTo.value = blobTo.value * (1 - opt.dropout);
                    end
                end

                if(bitand(blobTo.type, Blob.LOSS_SQUARED))
                
                end

                if(bitand(blobTo.type, Blob.LOSS_SOFTMAX))
                    blobTo.aux = exp(bsxfun(@minus, blobTo.value, max(blobTo.value)));
                    blobTo.aux = bsxfun(@rdivide, blobTo.aux, sum(blobTo.aux));
                end
            end
        end

        function collect(nn, outBatch)
            out = find(nn.output);
            for i = 1:length(out)
                blob = nn.blobs(out(i));
                if(bitand(blob.type, Blob.LOSS_SQUARED))
                    blob.error = blob.value - outBatch{i};
                    nn.error(i) = nn.error(i) + 1 / 2 * sum(sum(blob.error .* blob.error));
                end

                if(bitand(blob.type, Blob.LOSS_SOFTMAX))
                    blob.error = blob.aux - outBatch{i};
                    nn.error(i) = nn.error(i) - sum(sum(outBatch{i} .* log(blob.aux)));
                end
            end
        end

        function backward(nn)
            [from, to] = find(nn.link);
            for i = length(from):-1:1
                f = from(i);
                t = to(i);
                blobFrom = nn.blobs(f);
                blobTo = nn.blobs(t);

                if(bitand(blobTo.type, Blob.OP_RELU))
                    blobTo.error = blobTo.error .* (blobTo.value > 0);
                end

                if(bitand(blobTo.type, Blob.OP_FAST_SIGMOID))
                    temp = 1 - abs(blobTo.value);
                    blobTo.error = blobTo.error .* temp .* temp;
                end
                
                if(bitand(blobTo.type, Blob.DROPOUT))
                    blobTo.error = blobTo.error .* blobTo.dropout;
                end

                if(bitand(blobTo.type, Blob.LU))
                    blobFrom.error = blobFrom.error + nn.weight{f, t}(:, 2:end)' * blobTo.error;
                end
            end
        end

        function update(nn)
            opt = nn.opt;

            [from, to] = find(nn.link);
            for i = 1:length(from)
                f = from(i);
                t = to(i);
                blobFrom = nn.blobs(f);
                blobTo = nn.blobs(t);

                gradient = (opt.learn / nn.batchSize) * blobTo.error * (NN.pad(blobFrom.value, nn.batchSize, nn.gpu, nn.real))';
                if(opt.lambda)
                    gradient = gradient + opt.lambda * nn.weight{f, t};
                end
                if(opt.momentum)
                    gradient = gradient + opt.momentum * nn.gradient{f, t};
                    nn.gradient{f, t} = gradient;
                end
                nn.weight{f, t} = nn.weight{f, t} - gradient;
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
        
        function in = cast(in, real)
            in = cellfun(@(x) cast(x, real), in, 'UniformOutput', false);
        end

        function out = pad(in, size, gpu, real)
            out = [NN.ones(1, size, gpu, real); in];
        end

        function out = slice(in, sel)
            out = cellfun(@(x) x(:, sel), in, 'UniformOutput', false);
        end
    end
end
