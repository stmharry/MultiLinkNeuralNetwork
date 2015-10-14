classdef(Abstract) Dataset < handle
    properties
        in;
        out;
        predict;
        sampleNum;
    end

    methods(Abstract)
        getTrainData(dataset);
        getTestData(dataset);
        showTestInfo(dataset);

        blobs = getBlobs(dataset);
        opt = getOpt(dataset);
    end
end
