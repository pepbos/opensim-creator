% -------------------------------------------------------------------------
%
% Author: 
% Andreas Scholz
% Duisburg, 2022
% an.scho@t-online.de
%
% -------------------------------------------------------------------------

classdef GeodesicBoundaryPointFrame < RigidBody
    
    properties
        
        x;
        t;
        N;
        B;
        
    end
    
    methods
      
        function [obj] = GeodesicBoundaryPointFrame(r, R, v, w, x, t, N, B)
            
           obj = obj@RigidBody(r, R, v, w);
            
           obj.x = x;
           obj.t = t;
           obj.N = N;
           obj.B = B;
            
        end
        
    end
    
end

