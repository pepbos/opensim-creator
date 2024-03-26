% -------------------------------------------------------------------------
%
% Author: 
% Andreas Scholz
% Duisburg, 2022
% an.scho@t-online.de
%
% -------------------------------------------------------------------------

classdef RigidBody
   
    properties
      
        r;
        R; 
        
        v;
        w;
     
    end
    
    methods 
        
        function [obj] = RigidBody(r, R, v, w)
            
            obj.r = r;
            obj.R = R;
            obj.v = v;
            obj.w = w;
            
        end
        
        
        function [obj] = performLinearSpatialMotion(obj, timeStep)
            
               obj.r = obj.r + obj.v*timeStep;
               
               Q = computeEulerAnglesFromRotationMatrix(obj.R);
              
               if norm(sin(Q(2))) > 0.1
                   
                    Qd = computeEulerAngleDerivativesFromAngularVelocity(obj.w);
                    
                    obj.R = computeRotationMatrixFromEulerAngles(Q+Qd*timeStep);
                    
               else
                   
                    Q = obj.computeBryantAnglesFromRotationMatrix(obj.R);
                    
                    Qd = obj.computeBryantAngleDerivativesFromAngularVelocity(obj.w);
                    
                    obj.R = obj.computeRotationMatrixFromBryantAngles(Q+Qd*timeStep);
                    
               end
    
        end
        
    end
    
end

