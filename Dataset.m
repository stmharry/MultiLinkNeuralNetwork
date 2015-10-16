classdef Dataset < handle
    properties(Constant)
        FLAG = 0;
        TRAIN = Dataset.FLAG + 1;
        TEST  = Dataset.FLAG + 2;
    end

    properties
        in;
        out;
        predict;
        
        sampleNum;
        totalSize;

        flag;
    end
    
    methods(Abstract, Static)
        blobs = getBlobs();
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
        function getTrainData(dataset)
            dataset.flag = Dataset.TRAIN;
            dataset.totalSize = 0;
        end
        function getTestData(dataset)
            dataset.flag = Dataset.TEST;
            dataset.totalSize = 0;
            dataset.predict = cell(1, length(dataset.out));
        end
        function [inBatch, outBatch, batchSize] = getBatch(dataset, opt) 
            switch(opt.flag)
                case Opt.TRAIN
                    batchSize = min([opt.sampleNum - dataset.totalSize, opt.batchSize]);
                    sel = randi(dataset.sampleNum, batchSize, 1);
                case Opt.TEST
                    batchSize = min([dataset.sampleNum - dataset.totalSize, opt.batchSize]);
                    sel = dataset.totalSize + (1:batchSize);
            end
            dataset.totalSize = dataset.totalSize + batchSize;
            inBatch = cellfun(@(x) Dataset.slice(x, sel), dataset.in, 'UniformOutput', false);
            outBatch = cellfun(@(x) Dataset.slice(x, sel), dataset.out, 'UniformOutput', false);
        end
        function postTest(dataset, blobs)
            index = cellfun(@Dataset.maxIndex, {blobs.aux}, 'UniformOutput', false);
            dataset.predict = cellfun(@(x, y) [x, y], dataset.predict, index, 'UniformOutput', false);
        end
        function showTestInfo(dataset)
        end
    end
end
