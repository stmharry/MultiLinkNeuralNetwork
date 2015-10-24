dataset = DatasetDAL();
nn = NN(dataset);

dataset.configureNN(nn, DatasetDAL.DA);
nn.trainFor(100000);
%nn.test();

