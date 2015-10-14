classdef NN < handle
    properties(Constant)
        %SAVE_FIELD = {'gpu', 'real', 'blobs'};

        FLAG  = 0;
        TRAIN = NN.FLAG + 1;
        TEST  = NN.FLAG + 2;
    end

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
                nn.dataset = arg;
                nn.blobs = arg.blobs;
                nn.opt = arg.opt;

                nn.gpu = (gpuDeviceCount > 0);
                if(nn.gpu)
                    nn.real = 'double';
                end
                blobNum = length(nn.blobs);
                
                nn.input = sparse(blobNum, 1);
                nn.output = sparse(blobNum, 1);
                nn.link = sparse(blobNum, blobNum);
                for i = blobNum:-1:1
                    blob = nn.blobs(i);
                    blob.id = i;
                    nn.input(i) = blob.input;
                    nn.output(i) = blob.output;
                    for j = blob.next
                        nn.link(i, j.id) = true;
                    end
                end

                nn.weight = cell(blobNum);
                nn.gradient = cell(blobNum);
           
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

        function train(nn)
            dataset = nn.dataset;
            opt = nn.opt;

            dataset.getTrainData();
            opt.flag = NN.TRAIN;
            batchNum = ceil(dataset.sampleNum / opt.batchSize);
     
            for e = 1:opt.epochNum
                tic;
                permutation = randperm(dataset.sampleNum);
                for b = 1:batchNum
                    sel = permutation((b - 1) * opt.batchSize + 1:min(b * opt.batchSize, dataset.sampleNum));
                    nn.batchSize = length(sel);

                    inBatch = NN.slice(dataset.in, sel);
                    outBatch = NN.slice(dataset.out, sel);

                    nn.clean(opt);
                    nn.feed(inBatch, opt);
                    nn.forward(opt);
                    nn.collect(outBatch, opt);
                    nn.backward(opt);
                    nn.update(opt);
                end
                time = toc;

                fprintf('[DNN Training] Epoch = %d, Time = %.3f s, Error = ', e, time);
                fprintf('%.3f ', nn.error / batchNum);
                fprintf('\n');
            end
        end
        
        function test(nn)
            dataset = nn.dataset;
            opt = nn.opt;

            dataset.getTestData();
            opt.flag = NN.TEST;
            batchNum = ceil(dataset.sampleNum / opt.batchSize);

            out = find(nn.output);
            dataset.predict = cell(1, length(out));

            tic;
            for b = 1:batchNum
                sel = (b - 1) * opt.batchSize + 1:min(b * opt.batchSize, dataset.sampleNum);
                nn.batchSize = length(sel);

                inBatch = NN.slice(dataset.in, sel);
                outBatch = NN.slice(dataset.out, sel);
                
                nn.clean(opt);
                nn.feed(inBatch, opt);
                nn.forward(opt);
                nn.collect(outBatch, opt);

                dataset.postTest(nn.blobs(out));
            end
            time = toc;

            fprintf('[DNN Testing] Time = %.3f s, Error = ', time);
            fprintf('%.3f ', nn.error / batchNum);
            fprintf('\n');
            dataset.showTestInfo();
        end

        function clean(nn, opt)
            nn.error = zeros(sum(nn.output), 1);
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

        function feed(nn, inBatch, opt)
            in = find(nn.input);
            for i = 1:length(in)
                nn.blobs(in(i)).value = inBatch{i};
            end
        end

        function forward(nn, opt)
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
                    if(opt.flag == NN.TRAIN)
                        if(isempty(blobTo.dropout))
                            blobTo.dropout = (NN.rand(blobTo.dimension, nn.batchSize, nn.gpu, nn.real) > opt.dropout);
                        end
                        blobTo.value = blobTo.value .* blobTo.dropout;
                    elseif(opt.flag == NN.TEST)
                        blobTo.value = blobTo.value * (1 - opt.dropout);
                    end
                end

                if(bitand(blobTo.type, Blob.LOSS_SQUARED))
                    % No aux needed
                end

                if(bitand(blobTo.type, Blob.LOSS_SOFTMAX))
                    blobTo.aux = exp(bsxfun(@minus, blobTo.value, max(blobTo.value)));
                    blobTo.aux = bsxfun(@rdivide, blobTo.aux, sum(blobTo.aux));
                end
            end
        end

        function collect(nn, outBatch, opt)
            out = find(nn.output);
            for i = 1:length(out)
                blob = nn.blobs(out(i));
                if(bitand(blob.type, Blob.LOSS_SQUARED))
                    blob.error = blob.value - outBatch{i};
                    nn.error(i) = nn.error(i) + 1 / 2 * sum(sum(blob.error .* blob.error)) / nn.batchSize;
                end

                if(bitand(blob.type, Blob.LOSS_SOFTMAX))
                    blob.error = blob.aux - outBatch{i};
                    nn.error(i) = nn.error(i) - sum(sum(outBatch{i} .* log(blob.aux))) / nn.batchSize;
                end
            end
        end

        function backward(nn, opt)
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

        function update(nn, opt)
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
