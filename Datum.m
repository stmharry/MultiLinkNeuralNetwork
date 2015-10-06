classdef Datum < handle
    properties(Constant)
       FLAG = 0;
       FLAG_ACTIVE = Datum.FLAG + 1;
       FLAG_PASSIVE = Datum.FLAG + 2;
    end

    properties
        in;
        out;
        predicted;
        sampleNum;

        mu;
        sigma;
    end

    methods
        function normalize(datum, arg)
            if(isa(arg, 'function_handle'))
                func = arg;
                datum.in = cellfun(func, datum.in, 'UniformOutput', false);
            elseif(isa(arg, 'numeric'))
                flag = arg;
                for i = 1:length(datum.in)
                    if(flag == Datum.FLAG_ACTIVE)
                        datum.mu{i} = mean(datum.in{i}, 2);
                        datum.sigma{i} = std(datum.in{i}, 0, 2);
                    end
                    datum.in{i} = bsxfun(@rdivide, bsxfun(@minus, datum.in{i}, datum.mu{i}), datum.sigma{i});
                end
            end
        end
    end
end
