classdef Dataset < handle
    properties
        in;
        out;
        inBatch;
        outBatch;
        predict;
        
        sampleNum;
        totalSize;

        flag;
    end
    
    methods(Abstract, Static)
        net = getNet();
    end

    methods
        function dataset = Dataset()
            dataset.sampleNum = 0;
            dataset.totalSize = 0;
            dataset.flag = 0;
        end
        function configure(dataset, nn)
            opt = nn.opt;
            if(opt.flag ~= dataset.flag)
                if(opt.provide == Opt.WHOLE)
                    dataset.getDataWhole(opt);
                end
                dataset.flag = opt.flag;
            end

            if(opt.flag == Opt.TRAIN)
                dataset.totalSize = nn.totalSize;
            elseif(opt.flag == Opt.TEST)
                dataset.totalSize = 0;
            end
        end
        function batchSize = getBatchSize(dataset, opt, num)
            batchSize = min([num - dataset.totalSize, opt.batchSize]);
            dataset.totalSize = dataset.totalSize + batchSize;
        end
        function getDataWhole(dataset, opt)
            % 
        end
        function batchSize = getDataBatch(dataset, opt) 
            switch(opt.flag)
                case Opt.TRAIN
                    batchSize = dataset.getBatchSize(opt, opt.sampleNum);
                    sel = randi(dataset.sampleNum, batchSize, 1);
                case Opt.TEST
                    batchSize = dataset.getBatchSize(opt, dataset.sampleNum);
                    sel = dataset.totalSize - batchSize + (1:batchSize);
            end
            dataset.inBatch = cellfun(@(x) Util.slice(x, sel), dataset.in, 'UniformOutput', false);
            dataset.outBatch = cellfun(@(x) Util.slice(x, sel), dataset.out, 'UniformOutput', false);
        end
        function preTest(dataset)
            %
        end
        function processTestBatch(dataset, layers)
            index = cellfun(@Util.maxIndex, {layers.aux}, 'UniformOutput', false);
            dataset.predict = cellfun(@(x, y) [x, y], dataset.predict, index, 'UniformOutput', false);
        end
        function showTestInfo(dataset)
            %
        end
    end
end
