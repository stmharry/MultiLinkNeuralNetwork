classdef DatasetUSPS < Dataset
    properties
        X;
        Y;
        Y0;
        mu;
        sigma;
    end
    
    methods(Static)
        function blobs = getBlobs()
            blobs = [ ...
                Blob(256, Blob.IO_INPUT), ...
                Blob(256, Blob.LU + Blob.OP_RELU), ...
                Blob( 10, Blob.LU + Blob.LOSS_SOFTMAX + Blob.IO_OUTPUT) ...
            ];
            blobs(1).setInput().setNext(blobs(2)).setNext(blobs(3)).setOutput();
        end
        function opt = getOpt()
            opt = Opt();
            opt.batchSize = 64;
            opt.sampleNum = 1024;
            opt.reportInterval = 64;
            opt.learn = 0.1;
        end
    end

    methods
        function dataset = DatasetUSPS()
            dir = '../';
            usps = load([dir, 'USPSdata.mat']);
            dataset.X = double(usps.X);
            dataset.Y = double(usps.Y);
            dataset.Y0 = unique(usps.Y);    
        end
        function getTrainData(dataset)
            dataset.getTrainData@Dataset();

            numPerClass = 3;
            X = [];
            Y = [];
            for i = 1:length(dataset.Y0)
                sel = find(dataset.Y == dataset.Y0(i), numPerClass);
                X = [X, dataset.X(:, sel)];
                Y = [Y, dataset.Y(:, sel)];
            end
            [X, dataset.mu, dataset.sigma] = zscore(X, 0, 2);
            dataset.sigma(dataset.sigma == 0) = 1;

            dataset.in = {X};
            dataset.out = {bsxfun(@eq, Y, dataset.Y0')};
            dataset.sampleNum = length(Y);
        end
        function getTestData(dataset)
            dataset.getTestData@Dataset();

            X = bsxfun(@rdivide, bsxfun(@minus, dataset.X, dataset.mu), dataset.sigma);
            Y = dataset.Y;

            dataset.in = {X};
            dataset.out = {bsxfun(@eq, Y, dataset.Y0')};
            dataset.sampleNum = length(Y);
        end
    end
end
