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
                nn.dataset = arg;
                nn.init();
                nn.cache();

                nn.totalSize = 0;
                nn.accumSize = 0;
            end
        end

        function init(nn)
            net = nn.dataset.getNet();
            nn.opt = net.opt;
            nn.layers = net.layers;

            layerNum = length(nn.layers);
            for i = 1:layerNum
                nn.layers(i).id = i;
            end
            
            nn.connections = Connection.empty(0);
            for i = 1:layerNum
                for j = 1:layerNum
                    nn.connections(i, j) = Connection();
                end
            end
            for i = 1:layerNum
                layerI = nn.layers(i);
                for j = 1:length(layerI.next)
                    layerJ = layerI.next(j);
                    nn.connections(i, layerJ.id) = layerI.connections(j);
                end
            end
        end

        function cache(nn)
            layerNum = length(nn.layers);
            
            nn.input = sparse(layerNum, 1);
            nn.output = sparse(layerNum, 1);
            for i = 1:layerNum
                layerI = nn.layers(i);
                nn.input(i) = (layerI.type.io == Layer.IO_INPUT); 
                nn.output(i) = (layerI.type.io == Layer.IO_OUTPUT);
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
                f = nn.link(i).from;
                t = nn.link(i).to;

                dimFrom = nn.layers(f).dimension;
                dimTo = nn.layers(t).dimension;
  
                connection = nn.connections(f, t);
                if(~connection.isInit)
                    connection.init(dimFrom, dimTo);
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
            dataset.configure(nn);

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
            dataset.configure(nn);
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
                nn.layers(i).clean(nn.batchSize, nn.opt);
            end
        end

        function feed(nn, inBatch)
            in = find(nn.input);
            for i = 1:length(in)
                if(nn.layers(in(i)).dimension ~= size(inBatch{i}))
                    error('Input dimension does not match layer dimension!');
                end
                nn.layers(in(i)).feed(inBatch{i});
            end
        end

        function forward(nn)
            for l = nn.link
                nn.connections(l.from, l.to).forward(nn.layers(l.from), nn.layers(l.to), nn.batchSize, nn.opt);
                nn.layers(l.to).forward(nn.batchSize, nn.opt);
            end
        end

        function collect(nn, outBatch)
            out = find(nn.output);
            for i = 1:length(out)
                if(nn.layers(out(i)).dimension ~= size(outBatch{i}))
                    error('Output dimension does not match layer dimension!');
                end
                nn.error(i) = nn.error(i) + nn.layers(out(i)).collect(outBatch{i});
            end
        end

        function backward(nn)
            for l = nn.link
                nn.layers(l.to).backward(nn.batchSize, nn.opt);
                nn.connections(l.from, l.to).backward(nn.layers(l.from), nn.layers(l.to), nn.batchSize, nn.opt);
            end
        end

        function update(nn)
            for l = nn.link
                nn.connections(l.from, l.to).update(nn.layers(l.from), nn.layers(l.to), nn.batchSize, nn.opt);
            end
        end

        function info(nn, flag, content)
            switch(content)
                case NN.TITLE
                    switch(flag)
                        case Opt.TRAIN
                            tcprintf('red', '[ NN Training ]\n');
                        case Opt.TEST
                            space = 59;
                            tcprintf('red', '[ NN Testing ]\n');
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
                            tcprintf('red', '[ NN Testing Extra ]\n');
                    end
            end
        end
        
        function str = errorToStr(nn, flag)
            str = '';
            switch(flag)
                case Opt.TRAIN
                    e = nn.error / nn.accumSize;
                case Opt.TEST
                    e = nn.error / nn.dataset.totalSize;
            end

            out = find(nn.output);
            for o = 1:length(out)
                switch(nn.layers(out(o)).type.loss)
                    case Layer.LOSS_SQUARED
                        str = [str, num2str([e(o), log10(e(o))], '[SQ] %.2f (%.2f) ')];
                    case Layer.LOSS_SQUARED_RATIO
                        str = [str, num2str(e(o), '[SQR] %.2f ')];
                    case Layer.LOSS_CROSS_ENTROPY
                        str = [str, num2str([e(o), log10(e(o))], '[CE] %.2f (%.2f) ')];
                    case Layer.LOSS_DECISION_FOREST
                        str = [str, num2str([e(o), log10(e(o))], '[DF] %.2f (%.2f) ')];
                end
            end
        end
    end
end 
