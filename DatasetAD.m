classdef DatasetAD < Dataset
    properties
        amazon;
        dslr;
        indices;
        labels;
    end

    methods(Static)
        function blobs = getBlobs()
            blobs = [ ...
                Blob(800, [Blob.IO_INPUT]), ...
                Blob( 31, [Blob.LU, Blob.OP_RELU]), ...
                Blob(600, [Blob.LU, Blob.LOSS_SQUARED, Blob.IO_OUTPUT]) ...
            ];
            blobs(1).setNext(blobs(2)).setNext(blobs(3));
        end
        function opt = getOpt()
            opt = Opt();
            opt.batchSize = 256;
            opt.sampleNum = 100000;
            opt.reportInterval = 1024;
            opt.collect = false;
            opt.learn = 0.005;
            opt.lambda = 0.005;
        end
    end

    methods
        function dataset = DatasetAD()
            dir = '../data_acdw/';
            amazon = load([dir, 'data_amazon.mat']);
            dslr = load([dir, 'data_dslr_600.mat']);
            indices = load([dir, 'rand_indices.mat']);

            indices.source_num = length(indices.source_index);
            indices.target_train_num = length(indices.target_training_index);
            indices.target_test_num = length(indices.target_test_index);

            amazon.train_features = amazon.features(indices.source_index, :);
            [amazon.train_features, amazon.mu, amazon.sigma] = zscore(amazon.train_features);
            amazon.train_labels = amazon.labels(indices.source_index);

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

            dataset.in = {dataset.amazon.train_features'};
            dataset.out = {dataset.dslr.train_features'};
        end
        function getTestData(dataset)
            dataset.getTestData@Dataset();

            dataset.in = {dataset.amazon.train_features'};
            dataset.out = {dataset.dslr.train_features'};
            dataset.sampleNum = dataset.indices.source_num;
        end
        function [inBatch, outBatch, batchSize] = getBatch(dataset, opt) 
            switch(opt.flag)
                case Opt.TRAIN
                    batchSize = min([opt.sampleNum - dataset.totalSize, opt.batchSize]);
                    selLabel = dataset.labels(randi(length(dataset.labels), batchSize, 1));
                    [~, sourceSel] = max(bsxfun(@eq, dataset.amazon.train_labels, selLabel') + rand(dataset.indices.source_num, batchSize));
                    [~, targetSel] = max(bsxfun(@eq, dataset.dslr.train_labels, selLabel') + rand(dataset.indices.target_train_num, batchSize));
                case Opt.TEST
                    batchSize = min([dataset.sampleNum - dataset.totalSize, opt.batchSize]);
                    sourceSel = dataset.totalSize + (1:batchSize);
            end
            dataset.totalSize = dataset.totalSize + batchSize;
            inBatch = cellfun(@(x) Dataset.slice(x, sourceSel), dataset.in, 'UniformOutput', false);
            switch(opt.flag)
                case Opt.TRAIN
                    outBatch = cellfun(@(x) Dataset.slice(x, targetSel), dataset.out, 'UniformOutput', false);
                case Opt.TEST
                    outBatch = {};
            end
        end
        function postTest(dataset, blobs)
            dataset.predict = cellfun(@(x, y) [x, y], dataset.predict, {blobs.value}, 'UniformOutput', false);
        end
        function showTestInfo(dataset)
            source_features = dataset.predict{1}';
            source_labels = dataset.amazon.train_labels;
            target_train_features = dataset.dslr.train_features;
            target_train_labels = dataset.dslr.train_labels;
            target_test_features = dataset.dslr.test_features;
            target_test_labels = dataset.dslr.test_labels;

            save('projectionAD', 'source_features', 'source_labels', ...
                'target_train_features', 'target_train_labels', ...
                'target_test_features', 'target_test_labels');
        end
    end
end

