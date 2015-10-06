classdef Opt < handle
    properties(Constant)
        TRAIN          = 0;
        TRAIN_COMPACT  = Opt.TRAIN + 1;
        TRAIN_EXPANDED = Opt.TRAIN + 2;
        TRAIN_HANDLE   = Opt.TRAIN + 3;
        TEST           = 7;
        TEST_RAW       = Opt.TEST + 1;
        TEST_MAX       = Opt.TEST + 2;
    end

    properties
        train     = Opt.TRAIN_EXPANDED;
        test      = Opt.TEST_MAX;

        batchSize = 256;
        epochNum  = 10;
        dropout   = 0.5;
        learn     = 0.1;
        lambda    = 0;
        momentum  = 0.9;

        sampleNum;
        flag;
        extra;
    end
end
