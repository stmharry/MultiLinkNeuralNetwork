function [datum, nn, opt] = train()
dataset = DatasetACDW();
datum = Datum();

datum = dataset.getTrainDatum(datum);
blobs = dataset.getBlobs();
opt = dataset.getOpt(datum);

nn = NN(blobs);
nn.setDefaultLink();
nn.initialize(opt);
nn.train(datum.in, datum.out, opt);

%datum = dataset.getTestDatum(datum);
datum.predicted = nn.test(datum.in, opt);
end
