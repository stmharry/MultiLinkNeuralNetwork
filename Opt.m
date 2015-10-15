classdef Opt < handle
    properties
        batchSize = 256;
        sampleNum = 8192;
        reportInterval = 1024;

        init      = true;
        dropout   = 0.5;
        learn     = 0.1;
        lambda    = 0;
        momentum  = 0.9;

        flag;
    end
end
