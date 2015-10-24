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
        %layers = getLayers();
        net = getNet();
        opt = getOpt();
    end

    methods(Static)
        function out = slice(in, sel)
            out = in(:, sel);
        end

        function out = maxIndex(in)
            [~, out] = max(in);
        end
    end

    methods
        function dataset = Dataset()
            dataset.sampleNum = 0;
            dataset.totalSize = 0;
            dataset.flag = 0;
        end
        function configure(dataset, opt)
            if(dataset.flag ~= opt.flag)
                if(opt.provide == Opt.WHOLE)
                    dataset.getDataWhole(opt);
                end
                dataset.totalSize = 0;
                dataset.flag = opt.flag;
            end
        end
        function getDataWhole(dataset, opt)
            % 
        end
        function batchSize = getDataBatch(dataset, opt) 
            switch(opt.flag)
                case Opt.TRAIN
                    batchSize = min([opt.sampleNum - dataset.totalSize, opt.batchSize]);
                    sel = randi(dataset.sampleNum, batchSize, 1);
                case Opt.TEST
                    batchSize = min([dataset.sampleNum - dataset.totalSize, opt.batchSize]);
                    sel = dataset.totalSize + (1:batchSize);
            end
            dataset.totalSize = dataset.totalSize + batchSize;
            dataset.inBatch = cellfun(@(x) Dataset.slice(x, sel), dataset.in, 'UniformOutput', false);
            dataset.outBatch = cellfun(@(x) Dataset.slice(x, sel), dataset.out, 'UniformOutput', false);
        end
        function preTest(dataset)
            dataset.totalSize = 0;
        end
        function processTestBatch(dataset, layers)
            index = cellfun(@Dataset.maxIndex, {layers.aux}, 'UniformOutput', false);
            dataset.predict = cellfun(@(x, y) [x, y], dataset.predict, index, 'UniformOutput', false);
        end
        function showTestInfo(dataset)
            %
        end
    end
end
