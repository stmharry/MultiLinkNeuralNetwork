dataset = DatasetACDW();

blobs = dataset.getBlobs();
opt = dataset.getOpt();
nn = NN(blobs, opt);

datum = dataset.getTrainDatum();
nn.train(datum, opt);

%datum = dataset.getTestDatum(datum);
%nn.test(datum, opt);
