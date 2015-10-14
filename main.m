dataset = DatasetUSPS();

nn = NN(dataset);
dataset.getTrainData();
nn.train();
dataset.getTestData();
nn.test();
