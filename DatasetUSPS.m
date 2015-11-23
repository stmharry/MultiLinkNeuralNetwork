classdef DatasetUSPS < Dataset
    properties
        u;
        l;
    end

    methods(Static)
        function net = getNet()
            opt = Opt([ ...
                Opt.BATCHSIZE, 256, ...
                Opt.SAMPLENUM, 0, ...
                Opt.REPORT_INTERVAL, 1000, ...
                Opt.PROVIDE, Opt.WHOLE, ...
            ]);

            layers = [ ...
                Layer(256, [Layer.IO_INPUT]), ...
                Layer(511, []), ...
                Layer( 10, [Layer.LOSS_DECISION_FOREST, ...
                            Layer.IO_OUTPUT]) ...
            ];
            
            connection = Connection([ ...
                Connection.CONNECT_FULL, ...
                Connection.GRADIENT_ADAGRAD, ...
                Connection.REGULATOR_DISABLE, ...
                Connection.GRADIENT_LEARN, 0.01, ...
            ]);

            connections = [
                Connection([], connection);
                Connection([ ...
                    Connection.CONNECT_DECISION_FOREST, ...
                    Connection.CONNECT_DECISION_FOREST_TREE, 16, ...
                    Connection.CONNECT_DECISION_FOREST_NUM, 127, ...
                    Connection.CONNECT_DECISION_FOREST_PRUNEOUT, 0, ...
                    Connection.GRADIENT_ADAGRAD, ...
                    Connection.GRADIENT_LEARN, 0.001]);
            ];

            layers(1).addNext(layers(2), connections(1));
            layers(2).addNext(layers(3), connections(2));

            net.layers = layers;
            net.connections = connections;
            net.opt = opt;
        end
    end

    methods
        function dataset = DatasetUSPS()
            dataset.u = load(['../USPSdata.mat']);
            dataset.u.X = zscore(dataset.u.X, 0, 2);
            dataset.u.n = length(dataset.u.Y);
            dataset.u.l = unique(dataset.u.Y);
            dataset.u.Z = Util.expand(dataset.u.Y, dataset.u.l);
        end
        function getDataWhole(dataset, opt)
            dataset.in = {dataset.u.X};
            dataset.out = {dataset.u.Z};
            dataset.sampleNum = dataset.u.n;
        end
        function preTest(dataset)
            dataset.sampleNum = dataset.u.n;
            dataset.predict = cell(1);
        end
        function processTestBatch(dataset, layers)
            index = cellfun(@Util.maxIndex, {layers.aux}, 'UniformOutput', false);
            dataset.predict = cellfun(@(x, y) [x, y], dataset.predict, index, 'UniformOutput', false);
        end
        function showTestInfo(dataset)
            e = sum(dataste.u.l(dataset.predict{1}) ~= dataset.u.Y) / dataset.u.n;
            fprintf('[0/1 error] %.3f\n', e);
        end
    end
end


