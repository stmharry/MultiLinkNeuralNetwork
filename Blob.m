classdef Blob < handle
    properties(Constant)
        IO_DISABLE = 0
        IO_INPUT   = Blob.IO_DISABLE + 1;
        IO_OUTPUT  = Blob.IO_DISABLE + 2;

        LU_DISABLE = 3;
        LU         = Blob.LU_DISABLE + 1;

        OP_DISABLE      = 5;
        OP_RELU         = Blob.OP_DISABLE + 1;
        OP_FAST_SIGMOID = Blob.OP_DISABLE + 2;

        DROPOUT_DISABLE = 8;
        DROPOUT         = Blob.DROPOUT_DISABLE + 1;

        LOSS_DISABLE       = 10;
        LOSS_SQUARED       = Blob.LOSS_DISABLE + 1;
        LOSS_SQUARED_RATIO = Blob.LOSS_DISABLE + 2;
        LOSS_CROSS_ENTROPY = Blob.LOSS_DISABLE + 3;
        
        LOSS_OPT    = 14;
        LOSS_WEIGHT = Blob.LOSS_OPT + 1;
    end

    properties
        id;
        dimension;
        type;

        value;
        error;
        weight;
        dropout;
        aux;
        
        next;
        prev;
    end

    methods
        function blob = Blob(dimension, type)
            blob.dimension = dimension;
           
            blob.type.IO = Blob.IO_DISABLE;
            blob.type.LU = Blob.LU_DISABLE;
            blob.type.OP = Blob.OP_DISABLE;
            blob.type.DROPOUT = Blob.DROPOUT_DISABLE;
            blob.type.LOSS = Blob.LOSS_DISABLE;
            for i = 1:length(type)
                switch(type(i))
                    case {Blob.IO_DISABLE, Blob.IO_INPUT, Blob.IO_OUTPUT}
                        blob.type.IO = type(i);
                    case {Blob.LU_DISABLE, Blob.LU}
                        blob.type.LU = type(i);
                    case {Blob.OP_DISABLE, Blob.OP_RELU, Blob.OP_FAST_SIGMOID}
                        blob.type.OP = type(i);
                    case {Blob.DROPOUT_DISABLE, Blob.DROPOUT}
                        blob.type.DROPOUT = type(i);
                    case {Blob.LOSS_DISABLE, Blob.LOSS_SQUARED, Blob.LOSS_CROSS_ENTROPY}
                        blob.type.LOSS = type(i);
                        blob.weight = 1;
                    case {Blob.LOSS_WEIGHT}
                        i = i + 1;
                        blob.weight = type(i);
                end
            end

            blob.next = [];
            blob.prev = [];
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
