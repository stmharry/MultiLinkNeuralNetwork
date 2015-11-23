classdef Util < handle
    properties(Constant)
        gpu = (gpuDeviceCount > 0);
        real = 'double';
        epsilon = 1e-9;
    end

    methods(Static)
        function in = cast(in)
            in = cast(in, Util.real);
            if(Util.gpu)
                in = gpuArray(in);
            end
        end

        function in = normalize(in)
            in = bsxfun(@rdivide, in, sum(in));
        end
        
        function out = expand(in, lab)
            out = bsxfun(@eq, lab', in);
        end
        
        function out = slice(in, sel)
            out = in(:, sel);
        end

        function out = maxIndex(in)
            [~, out] = max(in);
        end

        function out = zeros(x, y)
            if(Util.gpu)
                out = gpuArray.zeros(x, y, Util.real);
            else
                out = zeros(x, y, Util.real);
            end
        end

        function out = ones(x, y)
            if(Util.gpu)
                out = gpuArray.ones(x, y, Util.real);
            else
                out = ones(x, y, Util.real);
            end
        end

        function out = eye(x)
            if(Util.gpu)
                out = gpuArray.eye(x, x, Util.real);
            else
                out = eye(x);
            end
        end

        function out = rand(x, y)
            if(Util.gpu)
                out = gpuArray.rand(x, y, Util.real);
            else
                out = rand(x, y, Util.real);
            end
        end
        
        function out = pad(in, x)
            out = [Util.ones(1, x); in];
        end
    end
end
