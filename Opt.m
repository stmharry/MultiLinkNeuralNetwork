classdef Opt < handle
    properties(Constant)
        FLAG  = 0;
        TRAIN = Opt.FLAG + 1;
        TEST  = Opt.FLAG + 2;
    end

    properties
        batchSize = 256;
        sampleNum = 8192;
        reportInterval = 1024;

        init      = true;
        collect   = true;
        dropout   = 0.5;
        learn     = 0.1;
        lambda    = 0;
        momentum  = 0.9;

        flag;
    end
end
