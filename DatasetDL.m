classdef DatasetDL < Dataset
    properties
        dslr;
        indices;
        labels;
    end

    methods(Static)
        function blobs = getBlobs()
            blobs = [ ...
                Blob( 600, Blob.IO_INPUT), ...
                Blob(1024, Blob.LU + Blob.OP_RELU), ...
                Blob(  31, Blob.LU + Blob.LOSS_SOFTMAX + Blob.IO_OUTPUT) ...
            ];
            blobs(1).setNext(blobs(2)).setNext(blobs(3));
        end
        function opt = getOpt()
            opt = Opt();
            opt.batchSize = 256;
            opt.sampleNum = 30000;
            opt.reportInterval = 1000;
            opt.collect = true;
            opt.learn = 0.01;
            opt.lambda = 0;
        end
    end

    methods
        function dataset = DatasetD()
            dir = '../data_acdw/';
            dslr = load([dir, 'data_dslr_600.mat']);
            indices = load([dir, 'rand_indices.mat']);

            indices.target_train_num = length(indices.target_training_index);
            indices.target_test_num = length(indices.target_test_index);

            dslr.train_features = dslr.features(indices.target_training_index, :);
            [dslr.train_features, dslr.mu, dslr.sigma] = zscore(dslr.train_features);
            dslr.train_labels = dslr.labels(indices.target_training_index);

            dslr.test_features = dslr.features(indices.target_test_index, :);
            dslr.test_features = bsxfun(@rdivide, bsxfun(@minus, dslr.test_features, dslr.mu), dslr.sigma);
            dslr.test_labels = dslr.labels(indices.target_test_index);
            
            %
            dataset.dslr = dslr;
            dataset.indices = indices;
            dataset.labels = unique(dslr.train_labels);
        end
        function getTrainData(dataset)
            dataset.getTrainData@Dataset();

            dataset.in = {dataset.dslr.train_features'};
            dataset.out = {bsxfun(@eq, dataset.labels, dataset.dslr.train_labels')};
            dataset.sampleNum = dataset.indices.target_train_num;
        end
        function getTestData(dataset)
            dataset.getTestData@Dataset();

            dataset.in = {dataset.dslr.test_features'};
            dataset.out = {bsxfun(@eq, dataset.labels, dataset.dslr.test_labels')};
            dataset.sampleNum = dataset.indices.target_test_num;
        end
        function showTestInfo(dataset)
            error = sum(dataset.dslr.test_labels ~= dataset.labels(dataset.predict{1})) / dataset.indices.target_test_num;
            fprintf('[DNN Testing] 0/1 Error: %.3f\n', error);
        end
    end
end
