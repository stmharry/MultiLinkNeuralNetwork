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
        batchSize      = 256;
        sampleNum      = 8192;
        reportInterval = 1024;
        
        flag    = Opt.FLAG;
        provide = Opt.WHOLE;
        collect = true;

        connectionType = struct( ...
            'gradient', Connection.GRADIENT_SGD, ...
            'regulator', Connection.REGULATOR_DISABLE);
        learn     = 0.01;
        momentum  = 0.9;
        lambda    = 0.01;

        % TODO: dropout for each layer
        dropout  = 0.5;
    end
end
