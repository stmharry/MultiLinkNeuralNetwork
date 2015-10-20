classdef Opt < handle
    properties(Constant)
        FLAG  = 0;
        TRAIN = Opt.FLAG + 1;
        TEST  = Opt.FLAG + 2;

        PROVIDE = 3;
        WHOLE = Opt.PROVIDE + 1;
        BATCH = Opt.PROVIDE + 2;

        GRADIENT = 6;
        SGD = Opt.GRADIENT + 1;
        ADAGRAD = Opt.GRADIENT + 2;
    end

    properties
        batchSize      = 256;
        sampleNum      = 8192;
        reportInterval = 1024;
        
        flag     = Opt.FLAG;
        provide  = Opt.WHOLE;
        gradient = Opt.SGD;
        collect  = true;
        dropout  = 0.5;
        learn    = 0.1;
        lambda   = 0;
        momentum = 0.9;
    end
end
