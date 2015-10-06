classdef DatasetACDW < Dataset
    methods(Static)
        function datum = getTrainDatum(datum)
            dataset = load('data_acdw/data_amazon.mat');
            datum.in = {dataset.features'};
            datum.out = {dataset.labels'};
            datum.sampleNum = length(dataset.labels);
            datum.normalize(@DatasetACDW.normalizer);
            datum.normalize(Datum.FLAG_ACTIVE);
        end

        function datum = getTestDatum(datum)
            dataset = load('data_acdw/data_dslr_600.mat');
            datum.in = {dataset.features'};
            datum.out = {dataset.labels'};
            datum.sampleNum = length(dataset.labels);
            datum.normalize(@DatasetACDW.normalizer);
            datum.normalize(Datum.FLAG_PASSIVE);
        end

        function blobs = getBlobs()
            blobs = [
                Blob( 800, Blob.IO_INPUT), ...
                Blob(1024, Blob.LU + Blob.OP_RELU), ...
                Blob(  31, Blob.LU + Blob.LOSS_SOFTMAX + Blob.IO_OUTPUT) ...
            ];
            blobs(1).setInput().setNext(blobs(2)).setNext(blobs(3)).setOutput();
        end

        function opt = getOpt(datum)
            opt = Opt();
            opt.batchSize = 256;
            opt.epochNum = 10;
            opt.dropout = 0;
            opt.learn = 0.1;
            opt.sampleNum = datum.sampleNum;
            opt.train = Opt.TRAIN_COMPACT;
            opt.test = Opt.TEST_MAX;
        end

        function in = normalizer(in)
            in = bsxfun(@rdivide, in, sum(in));
        end
    end
end
