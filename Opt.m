classdef Opt < handle
    properties(Constant)
        FLAG  = 0;
        TRAIN = Opt.FLAG + 1;
        TEST  = Opt.FLAG + 2;

        PROVIDE = 3;
        WHOLE = Opt.PROVIDE + 1;
        BATCH = Opt.PROVIDE + 2;
    end

    properties
        batchSize = 256;
        sampleNum = 8192;
        reportInterval = 1024;

        provide   = Opt.WHOLE;
        collect   = true;
        dropout   = 0.5;
        learn     = 0.1;
        lambda    = 0;
        momentum  = 0.9;

        flag;
    end
end
