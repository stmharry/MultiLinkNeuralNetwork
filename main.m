dataset = DatasetUSPS();

blobs = dataset.getBlobs();
opt = dataset.getOpt();
nn = NN(blobs, opt);

dataset.getTrainData();
nn.train(dataset, opt);

dataset.getTestData();
nn.test(dataset, opt);

dataset.showTestInfo();
