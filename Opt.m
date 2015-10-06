classdef Opt < handle
    properties(Constant)
        TRAIN          = 0;
        TRAIN_SLICE    = Opt.TRAIN + 2;
        TRAIN_GENERATE = Opt.TRAIN + 3;
        TEST           = 7;
        TEST_RAW       = Opt.TEST + 1;
        TEST_MAX       = Opt.TEST + 2;
    end

    properties
        train     = Opt.TRAIN_SLICE;
        test      = Opt.TEST_MAX;

        batchSize = 256;
        epochNum  = 10;
        dropout   = 0.5;
        learn     = 0.1;
        lambda    = 0;
        momentum  = 0.9;

        flag;
    end
end
