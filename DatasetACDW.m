classdef DatasetACDW < Dataset
    methods(Static)
        function datum = getTrainDatum(varargin)
            if(nargin == 0)
                datum = Datum();
            elseif(nargin == 1)
                datum = varargin{1};
            end

            amazon = load('data_acdw/data_amazon.mat');
            dslr = load('data_acdw/data_dslr_600.mat');
            datum.in = {amazon.features', dslr.features'};
            datum.out = {amazon.labels', dslr.labels'};
            datum.normalize(@DatasetACDW.normalizer);
            datum.normalize(Datum.FLAG_ACTIVE);
            datum.generate = @DatasetACDW.generate;
        end
        
        function datum = getTestDatum(varargin)
            if(nargin == 0)
                datum = Datum();
            elseif(nargin == 1)
                datum = varargin{1};
            end
            %
        end

        function blobs = getBlobs()
            blobs = [
                Blob( 800, Blob.IO_INPUT), ...
                Blob( 600, Blob.IO_INPUT), ...
                Blob(1024, Blob.LU + Blob.OP_RELU), ...
                Blob(1024, Blob.LU + Blob.OP_RELU), ...
                Blob(1024, Blob.LU + Blob.OP_RELU), ...
                Blob(   2, Blob.LU + Blob.LOSS_SOFTMAX + Blob.IO_OUTPUT) ...
            ];
            blobs(1).setInput().setNext(blobs(3)).setNext(blobs(5));
            blobs(2).setInput().setNext(blobs(4)).setNext(blobs(5)).setNext(blobs(6)).setOutput();
        end

        function opt = getOpt()
            opt = Opt();
            opt.batchSize = 256;
            opt.epochNum = 10;
            opt.dropout = 0;
            opt.learn = 0.1;
            opt.train = Opt.TRAIN_GENERATE;
            opt.test = Opt.TEST_MAX;
        end
        
        function in = normalizer(in)
            in = bsxfun(@rdivide, in, sum(in));
        end

        function generate(datum, opt)
            sel = cellfun(@(x) randi(length(x), opt.batchSize, 1), datum.out, 'UniformOutput', false);
            match = (datum.out{1}(sel{1}) == datum.out{2}(sel{2}));
            
            datum.inBatch = cellfun(@(x, y) x(:, y), datum.in, sel, 'UniformOutput', false);
            datum.outBatch = {[match; ~match]};
        end
    end
end
