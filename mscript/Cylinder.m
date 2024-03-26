% -------------------------------------------------------------------------
%
% Author: 
% Andreas Scholz
% Duisburg, 2022
% an.scho@t-online.de
%
% -------------------------------------------------------------------------

classdef Cylinder < ParametricSurface

    properties
        
        rds;
        lngth;
        
    end
    
    
    methods
        
        function [obj]  = Cylinder(r, R, v, w, radius, length)
            
            obj = obj@ParametricSurface(r, R, v, w);
            
            obj.rds   = radius;
            obj.lngth = length;
            
             obj = obj.computeSurfaceMeshInLocalCoordinates();
            
             obj = obj.computeSurfaceMeshInGlobalCoordinates();
            
        end 
        
    
        function [obj] = computeSurfaceMeshInLocalCoordinates(obj)
            
            u = 2*pi*[0:0.05:1];
            v =      [-obj.lngth/2:0.5:obj.lngth/2];
            
            obj.xSurfMeshLocalCoordinates = zeros(length(v), length(u));
            obj.ySurfMeshLocalCoordinates = zeros(length(v), length(u));
            obj.zSurfMeshLocalCoordinates = zeros(length(v), length(u));
            
            for i=1:length(v)
                
                for j=1:length(u)
                    
                    obj.xSurfMeshLocalCoordinates(i,j) = obj.rds * cos(u(j));
                    obj.ySurfMeshLocalCoordinates(i,j) = obj.rds * sin(u(j));
                    obj.zSurfMeshLocalCoordinates(i,j) = v(i);
                    
                end
                
            end
            
        end
        
 
        function [obj] = evaluateSurface(obj, Q)
            
                sinu = sin(Q(1));
                cosu = cos(Q(1));
               
                obj.surfaceData.u = Q(1);
                
                obj.surfaceData.v = Q(2);
                
                obj.surfaceData.x   = [  obj.rds * cosu     ; ...
                                         obj.rds * sinu     ; ...
                                         obj.surfaceData.v  ];
               
                obj.surfaceData.xu  = [ -obj.rds * sinu ; ...
                                         obj.rds * cosu ; ...
                                         0.0            ];
                          
                obj.surfaceData.xv  = [  0.0 ; ...
                                         0.0 ; ...
                                         1.0 ];
                           
                obj.surfaceData.xuu = [ -obj.rds * cosu ; ...
                                        -obj.rds * sinu ; ...
                                         0.0            ];
                           
                obj.surfaceData.xuv = [  0.0 ; ...
                                         0.0 ; ...
                                         0.0 ];                     
                                     
                obj.surfaceData.xvv = [  0.0 ; ...
                                         0.0 ; ...
                                         0.0 ];
                          
                N = cross(obj.surfaceData.xu, obj.surfaceData.xv);
                
                obj.surfaceData.N = N / norm(N);
                
                obj = obj.evaluateFirstFundamentalForm();
                obj = obj.evaluateSecondFundamentalForm();
                obj = obj.evaluateGaussianCurvature();
                
         end
                  
    end
         
end

