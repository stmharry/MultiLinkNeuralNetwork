classdef Opt < handle
    properties(Constant)
        OPT             = 0;
        BATCHSIZE       = Opt.OPT + 1;
        SAMPLENUM       = Opt.OPT + 2;
        REPORT_INTERVAL = Opt.OPT + 3;
        COLLECT         = Opt.OPT + 4;

        PROVIDE = 10;
        WHOLE   = Opt.PROVIDE + 1;
        BATCH   = Opt.PROVIDE + 2;

        FLAG  = 0;
        TRAIN = Opt.FLAG + 1;
        TEST  = Opt.FLAG + 2;
    end

    properties
        batchSize;
        sampleNum;
        reportInterval;
        
        provide; 
        collect; 
        
        flag; 
    end

    methods
        function opt = Opt(type)
            i = 1;
            while(i <= length(type))
                switch(type(i))
                    case {Opt.BATCHSIZE}
                        i = i + 1;
                        opt.batchSize = type(i);
                    case {Opt.SAMPLENUM}
                        i = i + 1;
                        opt.sampleNum = type(i);
                    case {Opt.REPORT_INTERVAL}
                        i = i + 1;
                        opt.reportInterval = type(i);
                    case {Opt.COLLECT}
                        i = i + 1;
                        opt.collect = type(i);
                    case {Opt.PROVIDE}
                        i = i + 1;
                        opt.provide = type(i);
                end
                i = i + 1;
            end
        end
    end
end
