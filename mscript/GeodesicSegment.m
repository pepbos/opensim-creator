% -------------------------------------------------------------------------
%
% Author: 
% Andreas Scholz
% Duisburg, 2022
% an.scho@t-online.de
%
% -------------------------------------------------------------------------

classdef GeodesicSegment < RigidBody
  
    properties
        
        KP;
        KQ;
        
        xLocal;
        xGlobal;
       
        l;
        
        aP;
        adP;
        
        aQ;
        adQ;
        
        rP;
        rdP;
        
        rQ;
        rdQ;
        
        tauP_tan;
        tauP_bin;
        
        tauQ_tan;
        tauQ_bin;
        
        kappaNP_tan;
        kappaNP_bin;
        
        kappaNQ_tan;
        kappaNQ_bin;
        
        kappaQ_alpha;
        kappaQ_rho;
        
    end
    
    methods
        
        function [obj] = GeodesicSegment(r, R, v, w)
            
            obj = obj@RigidBody(r, R, v, w);
            
            obj.KP = GeodesicBoundaryPointFrame(r, R, v, w, [], [], [], []);
            obj.KQ = GeodesicBoundaryPointFrame(r, R, v, w, [], [], [], []);
            
            obj.xLocal  = [];
            obj.xGlobal = [];
            
            obj.l  = [];
            
            obj.aP  = 1;
            obj.adP = 0;
            
            obj.aQ  = [];
            obj.adQ = [];
            
            obj.rP  = 0;
            obj.rdP = 1;
            
            obj.rQ  = [];
            obj.rdQ = [];
            
            obj.tauP_tan = [];
            obj.tauP_bin = [];
            
            obj.tauQ_tan = [];
            obj.tauQ_bin = [];
            
            obj.kappaNP_tan = [];
            obj.kappaNP_bin = []; 
        
            obj.kappaNQ_tan = []; 
            obj.kappaNQ_bin = [];
            
            obj.kappaQ_alpha = [];
            obj.kappaQ_rho = [];    
            
        end
        
        
        function [obj] = plotGeodesicSegment(obj, lineStyle, lineWidth)
            
           obj = obj.computeCurveInGlobalCoordinates();
           
           plot3(obj.xGlobal(1,:), ...
                 obj.xGlobal(2,:), ...
                 obj.xGlobal(3,:), ...
                 lineStyle, ... 
                 'linewidth', lineWidth);        
             
        end
        
        
        
        function [obj] = computeCurveInGlobalCoordinates(obj)
            
            [~, cols] = size(obj.xLocal);
            
            obj.xGlobal = zeros(3, cols);
            
            for i=1:cols
               
                obj.xGlobal(:,i) = obj.r + obj.R * obj.xLocal(:,i);
                
            end
            
        end
        
    end
       
end

