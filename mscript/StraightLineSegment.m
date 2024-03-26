% -------------------------------------------------------------------------
%
% Author: 
% Andreas Scholz
% Duisburg, 2022
% an.scho@t-online.de
%
% -------------------------------------------------------------------------

classdef StraightLineSegment
    
    properties
        
        startPoint;
        
        endPoint;
        
        e;
        
        l;
        
    end
    
    methods
        
        function [obj] = StraightLineSegment(startPoint, endPoint)
            
           obj = obj.update(startPoint, endPoint);
           
        end
        
        
        function [obj] = update(obj, startPoint, endPoint)
            
            obj.startPoint = startPoint;
           
            obj.endPoint = endPoint;
            
            obj.l = norm(obj.endPoint - obj.startPoint);
            
            obj.e = (obj.endPoint - obj.startPoint) / obj.l;
            
        end
        
        
        
        function [] = plotStraightLineSegment(obj, lineStyle, lineWidth)
            
            plot3([obj.startPoint(1,1), obj.endPoint(1,1)], ... 
                  [obj.startPoint(2,1), obj.endPoint(2,1)], ... 
                  [obj.startPoint(3,1), obj.endPoint(3,1)], ...
                  lineStyle, ...
                  'lineWidth', lineWidth);
            
        end
        
    end
    
end

