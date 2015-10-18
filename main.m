dataset = DatasetADL();

%
nn = NN(dataset);
dataset.configureNN(nn, DatasetADL.DL);
nn.opt.sampleNum = 5000;
nn.train();
%{
dataset.configureNN(nn, DatasetADL.AD);
nn.opt.sampleNum = 6000;
nn.train();
dataset.configureNN(nn, DatasetADL.ADL);
nn.opt.sampleNum = 9000;
nn.train();

dataset.configureNN(nn, DatasetADL.DL);
nn.test();
%}
