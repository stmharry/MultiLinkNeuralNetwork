classdef Blob < handle
    properties(Constant) 
        IO_INPUT        = 1;
        IO_OUTPUT       = bitshift(Blob.IO_INPUT, 1);
        LU              = bitshift(Blob.IO_OUTPUT, 1);
        OP_RELU         = bitshift(Blob.LU, 1);
        OP_FAST_SIGMOID = bitshift(Blob.OP_RELU, 1);
        DROPOUT         = bitshift(Blob.OP_FAST_SIGMOID, 1);
        LOSS_SQUARED    = bitshift(Blob.DROPOUT, 1);
        LOSS_SOFTMAX    = bitshift(Blob.LOSS_SQUARED, 1);
    end

    properties
        input;
        output;
        next;
        prev;

        dimension;
        type;

        value;
        error;
        extra;
    end

    methods
        function blob = Blob(dimension, type)
            blob.input = false;
            blob.output = false;
            blob.next = [];
            blob.prev = [];

            blob.dimension = dimension;
            blob.type = type;
        end

        function blob = setInput(blob)
            blob.input = true;
        end

        function blob = setOutput(blob)
            blob.output = true;
        end

        function next = setNext(blob, next)
            blob.next = [blob.next, next];
            next.prev = [next.prev, blob];
        end

        function prev = setPrev(blob, prev)
            blob.prev = [blob.prev, prev];
            prev.next = [prev.next, blob];
        end
    end
end
