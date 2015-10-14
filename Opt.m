classdef Opt < handle
    properties
        batchSize = 256;
        batchNum  = 128;
        epochNum  = 10;
        dropout   = 0.5;
        learn     = 0.1;
        lambda    = 0;
        momentum  = 0.9;

        flag;
    end
end
