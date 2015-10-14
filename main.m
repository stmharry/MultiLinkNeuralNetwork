dataset = DatasetUSPS();

nn = NN(dataset);
nn.train();
nn.test();
