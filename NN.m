classdef NN < handle
    properties
        gpu;
        real = 'double';
        epsilon = 1e-9;

        dataset;
        blobs;
        opt;

        input;
        output;
        link;
        linkQueue;

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
                    nn.real = 'single';
                end
                nn.dataset = dataset;
                nn.blobs = dataset.getBlobs();
                nn.opt = dataset.getOpt();
                
                blobNum = length(nn.blobs);
                nn.input = sparse(blobNum, 1);
                nn.output = sparse(blobNum, 1);
                nn.link = sparse(blobNum, blobNum);
                for i = 1:blobNum
                    nn.blobs(i).id = i;
                end
                nn.cache();

                nn.totalSize = 0;
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
                    nn.gradient{f, t} = nn.epsilon * NN.ones(dimTo, dimFrom + 1, nn.gpu, nn.real);
                end
            end
        end

        function cache(nn)
            blobNum = length(nn.blobs);

            for i = 1:blobNum
                blob = nn.blobs(i);
                nn.input(i) = (blob.type.IO == Blob.IO_INPUT); 
                nn.output(i) = (blob.type.IO == Blob.IO_OUTPUT);
                for blobJ = blob.next
                    nn.link(i, blobJ.id) = true;
                end
            end

            nn.linkQueue = struct([]);
            out = find(nn.output);
            if(length(out))
                for j = out'
                    blobJ = nn.blobs(j);
                    for blobI = blobJ.prev
                        nn.linkQueue = [nn.linkQueue, struct('from', blobI.id, 'to', j)];
                    end
                end
            end

            pos = 1;
            isLinkQueued = sparse(blobNum, blobNum);
            while(pos <= length(nn.linkQueue))
                j = nn.linkQueue(pos).from;
                blobJ = nn.blobs(j);
                if(blobJ.type.IO ~= Blob.IO_INPUT)
                    for blobI = blobJ.prev
                        if(~isLinkQueued(blobI.id, j))
                            nn.linkQueue = [nn.linkQueue, struct('from', blobI.id, 'to', j)];
                            isLinkQueued(blobI.id, j) = true;
                        end
                    end
                end
                pos = pos + 1;
            end
            nn.linkQueue = fliplr(nn.linkQueue);
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

            nn.error = zeros(sum(nn.output), 1);
            accumSize = 0;
    
            tcprintf('red', '[ DNN Training ]\n');
            tcprintf('blue', '+--------------+---------------+----------------------------------+ \n');
            tcprintf('blue', '| Sample Count | Time Used (s) | Error                            | \n');
            tcprintf('blue', '+--------------+---------------+----------------------------------+ \n');

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

                accumSize = accumSize + nn.batchSize;
                while(dataset.totalSize >= nn.totalSize + opt.reportInterval)
                    time = toc;
                    nn.totalSize = nn.totalSize + opt.reportInterval;
                    
                    tcprintf('blue', '| %12d | %13.3f | %-32s | \n', ...
                        nn.totalSize, time, num2str(nn.error / accumSize, '%.2E '));

                    accumSize = 0;
                    nn.error = zeros(sum(nn.output), 1);
                    tic;
                end
            end
            tcprintf('blue', '+--------------+---------------+----------------------------------+ \n');
        end
        
        function test(nn)
            dataset = nn.dataset;
            opt = nn.opt;

            opt.flag = Opt.TEST;
            dataset.configure(opt);
            dataset.preTest();

            out = find(nn.output);
            nn.error = zeros(sum(nn.output), 1);
            
            tcprintf('red', '[ DNN Testing ]\n');
            tcprintf('blue', '+--------------+---------------+----------------------------------+ \n');
            tcprintf('blue', '| Sample Count | Time Used (s) | Error                            | \n');
            tcprintf('blue', '+--------------+---------------+----------------------------------+ \n');

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

                dataset.processTestBatch(nn.blobs(out));
            end
            time = toc;

            if(opt.collect)
                errorStr = num2str(nn.error / dataset.totalSize, '%.2E ');
            else
                errorStr = 'N/A';
            end
            tcprintf('blue', '| %12d | %13.3f | %-32s | \n', ...
                dataset.totalSize, time, errorStr);
            tcprintf('blue', '+--------------+---------------+----------------------------------+ \n');
            tcprintf('red', '[ DNN Testing Extra ]\n');
            dataset.showTestInfo();
        end

        function clean(nn)
            for i = 1:length(nn.blobs)
                blob = nn.blobs(i);
                blob.value = NN.zeros(blob.dimension, nn.batchSize, nn.gpu, nn.real);
                blob.error = NN.zeros(blob.dimension, nn.batchSize, nn.gpu, nn.real);
                if(blob.type.DROPOUT ~= Blob.DROPOUT_DISABLE)
                    blob.dropout = NN.zeros(blob.dimension, nn.batchSize, nn.gpu, nn.real);
                end
                if(blob.type.LOSS ~= Blob.LOSS_DISABLE)
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
                
                switch(blobTo.type.LU)
                    case Blob.LU_DISABLE
                        error('No connection!');
                    case Blob.LU
                        blobTo.value = blobTo.value + nn.weight{f, t} * NN.pad(blobFrom.value, nn.batchSize, nn.gpu, nn.real);
                end
    
                switch(blobTo.type.OP)
                    case Blob.OP_DISABLE
                    case Blob.OP_RELU
                        blobTo.value = blobTo.value .* (blobTo.value > 0);
                    case Blob.OP_FAST_SIGMOID
                        blobTo.value = blobTo.value ./ (1 + abs(blobTo.value));
                end

                switch(blobTo.type.DROPOUT)
                    case Blob.DROPOUT_DISABLE
                    case Blob.DROPOUT
                        if(opt.flag == Opt.TRAIN)
                            if(isempty(blobTo.dropout))
                                blobTo.dropout = (NN.rand(blobTo.dimension, nn.batchSize, nn.gpu, nn.real) > opt.dropout);
                            end
                            blobTo.value = blobTo.value .* blobTo.dropout;
                        elseif(opt.flag == Opt.TEST)
                            blobTo.value = blobTo.value * (1 - opt.dropout);
                        end
                end

                if(blobTo.type.IO == Blob.IO_OUTPUT)
                    switch(blobTo.type.LOSS)
                        case Blob.LOSS_DISABLE
                            error('IO_OUTPUT without LOSS set!')
                        case Blob.LOSS_SQUARED
                        case Blob.LOSS_SQUARED_RATIO
                        case Blob.LOSS_CROSS_ENTROPY
                            blobTo.aux = exp(bsxfun(@minus, blobTo.value, max(blobTo.value)));
                            blobTo.aux = bsxfun(@rdivide, blobTo.aux, sum(blobTo.aux));
                    end
                end
            end
        end

        function collect(nn, outBatch)
            out = find(nn.output);
            for i = 1:length(out)
                blob = nn.blobs(out(i));

                switch(blob.type.LOSS)
                    case Blob.LOSS_DISABLE
                    case Blob.LOSS_SQUARED
                        blob.error = blob.value - outBatch{i};
                        nn.error(i) = nn.error(i) + gather(1 / 2 * sum(sum(blob.error .* blob.error)));
                    case Blob.LOSS_SQUARED_RATIO
                        temp = outBatch{i}(1, :) - 1 / 2;
                        blob.error = bsxfun(@times, blob.value - outBatch{i}(2:end, :), temp); 
                        nn.error(i) = nn.error(i) + gather(sum(log(sum(blob.error .* blob.error)) .* temp));
                    case Blob.LOSS_CROSS_ENTROPY
                        blob.error = blob.aux - outBatch{i};
                        nn.error(i) = nn.error(i) - gather(sum(sum(outBatch{i} .* log(blob.aux))));
                end

                blob.error = blob.weight * blob.error;
            end
        end

        function backward(nn)
            [from, to] = find(nn.link);
            for i = length(from):-1:1
                f = from(i);
                t = to(i);
                blobFrom = nn.blobs(f);
                blobTo = nn.blobs(t);

                switch(blobTo.type.OP)
                    case Blob.OP_DISABLE
                    case Blob.OP_RELU
                        blobTo.error = blobTo.error .* (blobTo.value > 0);
                    case Blob.OP_FAST_SIGMOID
                        temp = 1 - abs(blobTo.value);
                        blobTo.error = blobTo.error .* temp .* temp;
                end

                switch(blobTo.type.DROPOUT)
                    case Blob.DROPOUT_DISABLE
                    case Blob.DROPOUT
                        blobTo.error = blobTo.error .* blobTo.dropout;
                end

                switch(blobTo.type.LU)
                    case Blob.LU_DISABLE
                    case Blob.LU
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

                gradient = blobTo.error * (NN.pad(blobFrom.value, nn.batchSize, nn.gpu, nn.real))' / nn.batchSize;
                switch(opt.gradient)
                    case Opt.SGD
                        % V+ = momentum * V + learn * gradient
                        % W+ = W - V+
                        gradient = opt.learn * gradient;
                        if(opt.momentum)
                            gradient = gradient + opt.momentum * nn.gradient{f, t};
                        end
                        nn.gradient{f, t} = gradient;
                    case Opt.ADAGRAD
                        % V+ = V + gradient .* gradient
                        % W+ = W - learn * gradient ./ sqrt(V+)
                        nn.gradient{f, t} = nn.gradient{f, t} + gradient .* gradient;
                        gradient = opt.learn * gradient ./ sqrt(nn.gradient{f, t});
                end
                % W+ = W - (gradient + lambda * W)
                if(opt.lambda)
                    nn.weight{f, t} = (1 - opt.lambda) * nn.weight{f, t};
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
        
        function out = pad(in, size, gpu, real)
            out = [NN.ones(1, size, gpu, real); in];
        end
    end
end
