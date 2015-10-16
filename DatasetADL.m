classdef DatasetADL < Dataset
    properties
        amazon;
        dslr;
        indices;
        labels;
    end

    methods(Static)
        function blobs = getBlobs()
            blobs = [ ...
                Blob( 800, Blob.IO_INPUT), ...
                Blob(  31, Blob.LU + Blob.OP_RELU), ...
                Blob( 600, Blob.LU + Blob.LOSS_SQUARED + Blob.IO_INPUT + Blob.IO_OUTPUT) ...
                Blob(1024, Blob.LU + Blob.OP_RELU), ...
                Blob(  31, Blob.LU + Blob.LOSS_SOFTMAX + Blob.IO_OUTPUT)
            ];
            blobs(1).setNext(blobs(2)) ...
                    .setNext(blobs(3)) ...
                    .setNext(blobs(4)) ...
                    .setNext(blobs(5));
        end
        function opt = getOpt()
            opt = Opt();
            opt.batchSize = 256;
            opt.sampleNum = 100000;
            opt.reportInterval = 1000;
            opt.collect = true;
            opt.learn = 0.005;
            opt.lambda = 0.001;
        end
    end

    methods
        function dataset = DatasetADL()
            dir = '../data_acdw/';
            amazon = load([dir, 'data_amazon.mat']);
            dslr = load([dir, 'data_dslr_600.mat']);
            indices = load([dir, 'rand_indices.mat']);

            indices.source_num = length(indices.source_index);
            indices.target_train_num = length(indices.target_training_index);
            indices.target_test_num = length(indices.target_test_index);

            amazon.dimension = size(amazon.features, 2);
            amazon.train_features = amazon.features(indices.source_index, :);
            [amazon.train_features, amazon.mu, amazon.sigma] = zscore(amazon.train_features);
            amazon.train_labels = amazon.labels(indices.source_index);

            dslr.dimension = size(dslr.features, 2);
            dslr.train_features = dslr.features(indices.target_training_index, :);
            [dslr.train_features, dslr.mu, dslr.sigma] = zscore(dslr.train_features);
            dslr.train_labels = dslr.labels(indices.target_training_index);

            dslr.test_features = dslr.features(indices.target_test_index, :);
            dslr.test_features = bsxfun(@rdivide, bsxfun(@minus, dslr.test_features, dslr.mu), dslr.sigma);
            dslr.test_labels = dslr.labels(indices.target_test_index);
            
            %
            dataset.amazon = amazon;
            dataset.dslr = dslr;
            dataset.indices = indices;
            dataset.labels = unique(dslr.train_labels);
        end
        function getTrainData(dataset)
            dataset.getTrainData@Dataset();

            dataset.in = {dataset.amazon.train_features', []};
            dataset.out = {dataset.dslr.train_features', ...
                           bsxfun(@eq, dataset.labels, dataset.amazon.train_labels')};
        end
        function getTestData(dataset)
            dataset.getTestData@Dataset();

            dataset.in = {[], dataset.dslr.test_features'};
            dataset.out = {[], bsxfun(@eq, dataset.labels, dataset.dslr.test_labels')};
            dataset.sampleNum = dataset.indices.target_test_num;
        end
        function [inBatch, outBatch, batchSize] = getBatch(dataset, opt) 
            switch(opt.flag)
                case Opt.TRAIN
                    batchSize = min([opt.sampleNum - dataset.totalSize, opt.batchSize]);
                    selLabel = dataset.labels(randi(length(dataset.labels), batchSize, 1));
                    [~, sourceSel] = max(bsxfun(@eq, dataset.amazon.train_labels, selLabel') + rand(dataset.indices.source_num, batchSize));
                    [~, targetSel] = max(bsxfun(@eq, dataset.dslr.train_labels, selLabel') + rand(dataset.indices.target_train_num, batchSize));

                    inBatch = {Dataset.slice(dataset.in{1}, sourceSel), ...
                               zeros(dataset.dslr.dimension, batchSize)};
                    outBatch = {Dataset.slice(dataset.out{1}, targetSel), ...
                                Dataset.slice(dataset.out{2}, sourceSel)};
                case Opt.TEST
                    batchSize = min([dataset.sampleNum - dataset.totalSize, opt.batchSize]);
                    targetSel = dataset.totalSize + (1:batchSize);

                    inBatch = {zeros(dataset.amazon.dimension, batchSize), ...
                               Dataset.slice(dataset.in{2}, targetSel)};
                    outBatch = {Dataset.slice(dataset.in{2}, targetSel), ...
                                Dataset.slice(dataset.out{2}, targetSel)};
            end
            dataset.totalSize = dataset.totalSize + batchSize;
        end
        function postTest(dataset, blobs)
            dataset.postTest@Dataset(blobs);
        end
        function showTestInfo(dataset)
            error = sum(dataset.dslr.test_labels ~= dataset.labels(dataset.predict{2})) / dataset.indices.target_test_num;
            fprintf('[DNN Testing] 0/1 Error: %.3f\n', error);
        end
    end
end


