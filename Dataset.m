classdef(Abstract) Dataset < handle
    methods(Abstract, Static)
        datum = getTrainDatum(datum);
        datum = getTestDatum(datum);
        blobs = getBlobs();
        opt = getOpt(datum);
    end
end
