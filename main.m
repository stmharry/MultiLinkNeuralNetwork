dataset = DatasetADL();
nn = NN(dataset);

dataset.configureNN(nn, DatasetADL.AD);
nn.trainFor(50000);

%{
dataset.configureNN(nn, DatasetADL.AD);
dataset.configureNN(nn, DatasetADL.ADL);
dataset.configureNN(nn, DatasetADL.AD);
nn.train();
nn.test();
%}
