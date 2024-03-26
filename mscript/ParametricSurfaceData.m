% -------------------------------------------------------------------------
%
% Author: 
% Andreas Scholz
% Duisburg, 2022
% an.scho@t-online.de
%
% -------------------------------------------------------------------------

classdef ParametricSurfaceData
    
    properties
        
        u;
        v;
        
        x;
        xu;
        xv;
        
        xuu;
        xuv;
        xvv;
        
        N;
        
        FF1;
        FF2;
        
        K;
        
    end
    
    methods
        
        function [obj] = ParametricSurfaceData()
            
            obj.u = 0;
            obj.v = 0;
            
            obj.x  = [0 0 0];
            obj.xu = [0 0 0];
            obj.xv = [0 0 0];
            
            obj.xuu = [0 0 0];
            obj.xuv = [0 0 0];
            obj.xvv = [0 0 0];
            
            obj.FF1 = [0 0 0];
            obj.FF2 = [0 0 0];
            
            obj.K = 0;
        
        end
          
    end
    
end

