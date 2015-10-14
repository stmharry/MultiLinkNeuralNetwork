classdef Dataset < handle
    properties
        in;
        out;
        predict;
        sampleNum;

        blobs;
        opt;
    end

    methods(Abstract)
        getTrainData(dataset);
        getTestData(dataset);

        postTest(dataset, blobs);
        showTestInfo(dataset);
    end
end
