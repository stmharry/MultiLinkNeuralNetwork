classdef DatasetADL < Dataset
    properties(Constant)
        PHASE = 0;
        AD  = DatasetADL.PHASE + 1;
        DL  = DatasetADL.PHASE + 2;
        ADL = DatasetADL.PHASE + 3;
    end

    properties
        amazon;
        dslr;
        indices;
        labels;
        match;

        phase;
    end

    methods(Static)
        function net = getNet()
            layers = [ ...
                Layer( 800, []), ...
                Layer(  31, [Layer.LU, Layer.OP_RELU]), ...
                Layer( 600, [Layer.LU, Layer.LOSS_SQUARED]) ...
                Layer(1024, [Layer.LU, Layer.OP_RELU]), ...
                Layer(  31, [Layer.LU, Layer.LOSS_CROSS_ENTROPY]), ...
            ];

            connections = [ ...
                Connection([Connection.GRADIENT_ADAGRAD]), ...
                Connection([Connection.GRADIENT_ADAGRAD]), ...
            ];

            layers(1).addNext(layers(2)) ...
                     .addNext(layers(3)) ...
                     .addNext(layers(4), connections(1)) ...
                     .addNext(layers(5), connections(2));

            net.layers = layers;
            net.connections = connections;
        end
        function opt = getOpt()
            opt = Opt();

            opt.batchSize      = 256;
            opt.sampleNum;
            opt.reportInterval = 1000;

            opt.flag;
            opt.provide = Opt.BATCH;
            opt.collect;

            opt.connectionType.gradient;
            opt.connectionType.regulator;
            opt.learn;
            opt.momentum;
            opt.lambda = 0.005;
        end
    end

    methods
        function dataset = DatasetADL()
            dir = '../data_acdw/';
            amazon = load([dir, 'data_amazon.mat']);
            dslr = load([dir, 'data_dslr_600.mat']);
            indices = load([dir, 'rand_indices.mat']);
            labels = unique(amazon.labels)';

            indices.source_num = length(indices.source_index);
            indices.target_train_num = length(indices.target_training_index);
            indices.target_test_num = length(indices.target_test_index);

            amazon.dimension = size(amazon.features, 2);
            amazon.train_features = amazon.features(indices.source_index, :)';
            [amazon.train_features, amazon.mu, amazon.sigma] = zscore(amazon.train_features, 0, 2);
            amazon.train_labels = amazon.labels(indices.source_index)';
            amazon.train_labels_expand = bsxfun(@eq, labels', amazon.train_labels);

            dslr.dimension = size(dslr.features, 2);
            dslr.train_features = dslr.features(indices.target_training_index, :)';
            [dslr.train_features, dslr.mu, dslr.sigma] = zscore(dslr.train_features, 0, 2);
            dslr.train_labels = dslr.labels(indices.target_training_index)';
            dslr.train_labels_expand = bsxfun(@eq, labels', dslr.train_labels);

            dslr.test_features = dslr.features(indices.target_test_index, :)';
            dslr.test_features = bsxfun(@rdivide, bsxfun(@minus, dslr.test_features, dslr.mu), dslr.sigma);
            dslr.test_labels = dslr.labels(indices.target_test_index)';
            dslr.test_labels_expand = bsxfun(@eq, labels', dslr.test_labels);
            
            %
            dataset.amazon = amazon;
            dataset.dslr = dslr;
            dataset.indices = indices;
            dataset.labels = labels;
            dataset.match = find(bsxfun(@eq, amazon.train_labels', dslr.train_labels));
        end
        function configureNN(dataset, nn, phase)
            dataset.phase = phase;
            switch(phase)
                case DatasetADL.AD
                    nn.layers(1).type.io = Layer.IO_INPUT;
                    nn.layers(3).type.io = Layer.IO_OUTPUT;
                    nn.layers(5).type.io = Layer.IO_DISABLE;
                    nn.opt.collect = false;
                case DatasetADL.DL
                    nn.layers(1).type.io = Layer.IO_DISABLE;
                    nn.layers(3).type.io = Layer.IO_INPUT;
                    nn.layers(5).type.io = Layer.IO_OUTPUT;
                    nn.layers(5).weight = 1;
                    nn.opt.collect = true;
                case DatasetADL.ADL
                    nn.layers(1).type.io = Layer.IO_INPUT;
                    nn.layers(3).type.io = Layer.IO_OUTPUT;
                    nn.layers(5).type.io = Layer.IO_OUTPUT;
                    nn.layers(5).weight = 0.5;
                    nn.opt.collect = true;
            end 
            nn.cache();
        end
        function batchSize = getDataBatch(dataset, opt) 
            switch(opt.flag)
                case Opt.TRAIN
                    batchSize = min([opt.sampleNum - dataset.totalSize, opt.batchSize]);
                    switch(dataset.phase)
                        case {DatasetADL.AD, DatasetADL.ADL}
                            sel = dataset.match(randi(length(dataset.match), batchSize, 1));
                            targetSel = floor((sel - 1) / dataset.indices.source_num) + 1;
                            sourceSel = sel - dataset.indices.source_num * (targetSel - 1);
                        case DatasetADL.DL
                            sel = randi(dataset.indices.target_train_num, batchSize, 1);
                    end
                    switch(dataset.phase)
                        case DatasetADL.AD
                            dataset.inBatch = {Dataset.slice(dataset.amazon.train_features, sourceSel)};
                            dataset.outBatch = {Dataset.slice(dataset.dslr.train_features, targetSel)};
                        case DatasetADL.DL
                            dataset.inBatch = {Dataset.slice(dataset.dslr.train_features, sel)};
                            dataset.outBatch = {Dataset.slice(dataset.dslr.train_labels_expand, sel)};
                        case DatasetADL.ADL
                            dataset.inBatch = {Dataset.slice(dataset.amazon.train_features, sourceSel)};
                            dataset.outBatch = {Dataset.slice(dataset.dslr.train_features, targetSel), ...
                                                Dataset.slice(dataset.amazon.train_labels_expand, sourceSel)};
                    end
                case Opt.TEST
                    switch(dataset.phase)
                        case DatasetADL.AD
                            batchSize = min([dataset.indices.source_num - dataset.totalSize, opt.batchSize]);
                            sel = dataset.totalSize + (1:batchSize);
                            dataset.inBatch = {Dataset.slice(dataset.amazon.train_features, sel)};
                        case DatasetADL.DL
                            batchSize = min([dataset.indices.target_test_num - dataset.totalSize, opt.batchSize]);
                            sel = dataset.totalSize + (1:batchSize);
                            dataset.inBatch = {Dataset.slice(dataset.dslr.test_features, sel)};
                            dataset.outBatch = {Dataset.slice(dataset.dslr.test_labels_expand, sel)};
                        case DatasetADL.ADL
                            error('Wrong configuration set!')
                    end
            end
            dataset.totalSize = dataset.totalSize + batchSize;
        end
        function preTest(dataset)
            dataset.totalSize = 0;
            dataset.predict = cell(1);
        end
        function processTestBatch(dataset, layers)
            switch(dataset.phase)
                case DatasetADL.AD
                    dataset.predict = cellfun(@(x, y) [x, y], dataset.predict, {layers.value}, 'UniformOutput', false);
                case DatasetADL.DL
                    dataset.processTestBatch@Dataset(layers);
                case DatasetADL.ADL
            end
        end
        function showTestInfo(dataset)
            switch(dataset.phase)
                case DatasetADL.AD
                    source_features = gather(dataset.predict{1});
                    source_labels = dataset.amazon.train_labels;
                    target_train_features = dataset.dslr.train_features;
                    target_train_labels = dataset.dslr.train_labels;
                    target_test_features = dataset.dslr.test_features;
                    target_test_labels = dataset.dslr.test_labels;
                    save('util/projection', 'source_features', 'source_labels', ...
                        'target_train_features', 'target_train_labels', ...
                        'target_test_features', 'target_test_labels');
                case DatasetADL.DL
                    error = sum(dataset.dslr.test_labels ~= dataset.labels(dataset.predict{1})) / dataset.indices.target_test_num;
                    fprintf('0/1 Error: %.3f\n', error);
                case DatasetADL.ADL
            end
        end
    end
end
