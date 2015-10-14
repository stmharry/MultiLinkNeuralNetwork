classdef Dataset < handle
    properties
        in;
        out;
        predict;
        sampleNum;
    end
    
    methods(Abstract, Static)
        blobs = getBlobs();
        opt = getOpt();
    end

    methods(Abstract)
        getTrainData(dataset);
        getTestData(dataset);

        postTest(dataset, blobs);
        showTestInfo(dataset);
    end
end
