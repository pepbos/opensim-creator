% -------------------------------------------------------------------------
%
% Author: 
% Andreas Scholz
% Duisburg, 2022
% an.scho@t-online.de
%
% -------------------------------------------------------------------------

classdef MuscleWrappingSystem
   
    properties
        
        O;
        I;
        
        q;
        
        eps;
        
        globalPathErrorNorm;
        
        Dxi;
        
        J;
        
        wrappingObstacles;
        
        geodesics;
        
        straightLineSegments;
        
        pathLength;
     
    end
    
    
    methods
        
        function [obj] = MuscleWrappingSystem(O, I)
            
           obj.O = O;
           obj.I = I;
           
           obj.q = [];
           
           obj.eps = [];
           
           obj.globalPathErrorNorm = [];
           
           obj.Dxi = [];
           
           obj.J = [];
           
           obj.wrappingObstacles = {};
           
           obj.geodesics = {};
           
           obj.straightLineSegments = {};
           
           obj.pathLength = [];
           
        end
        
        
        function [obj] = addWrappingObstacle(obj, wrappingObstacle, geodesicInitialConditions)

            n = length(obj.wrappingObstacles);
            
            obj.wrappingObstacles{n+1} = wrappingObstacle;
           
            obj.q((5*(n+1)-4):(5*(n+1)), 1) = geodesicInitialConditions;
            
            obj = obj.update();
            
        end
        
        
        function [obj] = update(obj)
            
            obj = obj.computeGeodesics();
            
            obj = obj.computeStraightLineSegments();
            
            obj = obj.computePathError();
            
            obj = obj.computePathErrorJacobian();
            
            obj = obj.computeNaturalGeodesicCorrections();
            
            obj = obj.computePathLength();
            
        end
        
        
        function [obj] = computeGeodesics(obj)
            
            for i=1:length(obj.wrappingObstacles)
                
                Q0(1,1:2)  = [obj.q(5*i-4,1), ...
                              obj.q(5*i-3,1)];
                
                Qd0(1,1:2) = [obj.q(5*i-2,1), ...
                              obj.q(5*i-1,1)];
                
                curveLength = obj.q(5*i,1);
                
                steps = 20;
                
                obj.geodesics{i} = obj.wrappingObstacles{i}.computeArcLengthParameterizedGeodesic(Q0, Qd0, curveLength, steps);
                
            end
            
        end
        
        
        function [obj] = computeStraightLineSegments(obj)
            
            n = length(obj.wrappingObstacles);
            
            if (n > 0)
                
                for i=1:n+1
                
                    if (i == 1)
                        startPoint = obj.O;
                        endPoint   = obj.geodesics{i}.r + obj.geodesics{i}.R * obj.geodesics{i}.KP.x;
                    end 
                    
                    if (i > 1 && i < n+1)
                        startPoint = obj.geodesics{i-1}.r + obj.geodesics{i-1}.R * obj.geodesics{i-1}.KQ.x;
                        endPoint   = obj.geodesics{i  }.r + obj.geodesics{i  }.R * obj.geodesics{i  }.KP.x;
                    end
                    
                    if (i == n+1)
                        startPoint = obj.geodesics{i-1}.r + obj.geodesics{i-1}.R * obj.geodesics{i-1}.KQ.x;
                        endPoint   = obj.I;
                    end
                       
                    obj.straightLineSegments{i} = StraightLineSegment(startPoint, endPoint);
                    
                end
               
            else
                
                obj.straightLineSegments{1} = StraightLineSegment(obj.O, obj.I);
                
            end
            
        end
        
        
        function [obj] = computePathLength(obj)
            
            n = length(obj.wrappingObstacles);
            
            if (n > 0)
                
                obj.pathLength = 0;
            
                for i=1:n
                    obj.pathLength = obj.pathLength + obj.straightLineSegments{i}.l + obj.geodesics{i}.l;
                end
            
                obj.pathLength = obj.pathLength + obj.straightLineSegments{i+1}.l;
                
            elseif (n == 0)
                
               obj.pathLength = obj.straightLineSegments{1}.l;
                
            end
            
        end
        
        
        function [obj] = computePathError(obj)
            
            n = length(obj.wrappingObstacles);
            
            if n > 0
                
                obj.eps = zeros(4*n,1);
              
                for i=1:n
                
                    eP = obj.straightLineSegments{i  }.e;
                    eQ = obj.straightLineSegments{i+1}.e;
                   
                    NP = obj.geodesics{i}.R * obj.geodesics{i}.KP.N;
                    BP = obj.geodesics{i}.R * obj.geodesics{i}.KP.B;
                    
                    NQ = obj.geodesics{i}.R * obj.geodesics{i}.KQ.N;
                    BQ = obj.geodesics{i}.R * obj.geodesics{i}.KQ.B;
                    
                    obj.eps(4*i-3,1) = dot(eP, NP);
                    obj.eps(4*i-2,1) = dot(eP, BP);
                    
                    obj.eps(4*i-1,1) = dot(eQ, NQ);
                    obj.eps(4*i  ,1) = dot(eQ, BQ);
                    
                end
                
            else
                
                obj.eps = 0;
            
            end
            
            obj.globalPathErrorNorm = norm(obj.eps);
            
        end
        
        
        function [obj] = computePathErrorJacobian(obj)
           
            n = length(obj.wrappingObstacles);
            
            obj.J = zeros(4*n, 4*n);
            
            if (n > 0)
               
                for i=1:n
                    
                    R = obj.geodesics{i}.R;
                    
                    tP = R * obj.geodesics{i}.KP.t;
                    NP = R * obj.geodesics{i}.KP.N;
                    BP = R * obj.geodesics{i}.KP.B;
                    
                    tQ = R * obj.geodesics{i}.KQ.t;
                    NQ = R * obj.geodesics{i}.KQ.N;
                    BQ = R * obj.geodesics{i}.KQ.B;
                    
                    eP = obj.straightLineSegments{i  }.e;
                    eQ = obj.straightLineSegments{i+1}.e;
                    
                    lP = obj.straightLineSegments{i  }.l;
                    lQ = obj.straightLineSegments{i+1}.l; 
                    
                    kappaNP_tan = obj.geodesics{i}.kappaNP_tan;
                    kappaNP_bin = obj.geodesics{i}.kappaNP_bin;
                    
                    kappaNQ_tan = obj.geodesics{i}.kappaNQ_tan;
                    kappaNQ_bin = obj.geodesics{i}.kappaNQ_bin;
                    
                    tauP_tan = obj.geodesics{i}.tauP_tan;
                    tauP_bin = obj.geodesics{i}.tauP_bin;
                    
                    tauQ_tan = obj.geodesics{i}.tauQ_tan;
                   
                    aQ  = obj.geodesics{i}.aQ;
                    adQ = obj.geodesics{i}.adQ;
                    
                    rQ  = obj.geodesics{i}.rQ;
                    rdQ = obj.geodesics{i}.rdQ;
                    
                    dNPdsP = -kappaNP_tan*tP - tauP_tan*BP;
                    dBPdsP =  tauP_tan*NP;
                    
                    dNQdsP = -kappaNQ_tan*tQ - tauQ_tan*BQ;
                    dBQdsP =  tauQ_tan*NQ;
   
                    dNPdbetaP =  tauP_bin*tP - kappaNP_bin*BP;
                    dBPdbetaP =  kappaNP_bin*NP;
                    
                    dNQdbetaP = -aQ*tauQ_tan*tQ - aQ*kappaNQ_bin*BQ;
                    dBQdbetaP = -adQ*tQ + aQ*kappaNQ_bin*NQ;
                    
                    dBPdtheta = -tP;
                    
                    dNQdtheta = -rQ*tauQ_tan*tQ - rQ*kappaNQ_bin*BQ;
                    dBQdtheta = -rdQ*tQ + rQ*kappaNQ_bin*NQ;
                    
                    dNQdl = dNQdsP;
                    dBQdl = dBQdsP;
                    
                    dePdsP    = (tP - dot(eP, tP)*eP) / lP;
                    dePdbetaP = (BP - dot(eP, BP)*eP) / lP;
                    
                    deQdsQ    = (dot(eQ, tQ)*eQ - tQ) / lQ;
                    deQdbetaQ = (dot(eQ, BQ)*eQ - BQ) / lQ;
                    
                    deQdsP = deQdsQ;
                    deQdl  = deQdsQ;
                    
                    deQdbetaP = deQdbetaQ*aQ;
                    deQdtheta = deQdbetaQ*rQ;
                    
                    m = 4*i;
                    
                    obj.J(m-3, m-3) = dot(dePdsP,    NP) + dot(eP, dNPdsP); 
                    obj.J(m-3, m-2) = dot(dePdbetaP, NP) + dot(eP, dNPdbetaP);
                    
                    obj.J(m-2, m-3) = dot(dePdsP,    BP) + dot(eP, dBPdsP);
                    obj.J(m-2, m-2) = dot(dePdbetaP, BP) + dot(eP, dBPdbetaP);
                    obj.J(m-2, m-1) =                      dot(eP, dBPdtheta);
                    
                    obj.J(m-1, m-3) = dot(deQdsP,    NQ) + dot(eQ, dNQdsP); 
                    obj.J(m-1, m-2) = dot(deQdbetaP, NQ) + dot(eQ, dNQdbetaP); 
                    obj.J(m-1, m-1) = dot(deQdtheta, NQ) + dot(eQ, dNQdtheta);
                    obj.J(m-1, m  ) = dot(deQdl,     NQ) + dot(eQ, dNQdl);
                    
                    obj.J(m  , m-3) = dot(deQdsP,    BQ) + dot(eQ, dBQdsP); 
                    obj.J(m  , m-2) = dot(deQdbetaP, BQ) + dot(eQ, dBQdbetaP); 
                    obj.J(m  , m-1) = dot(deQdtheta, BQ) + dot(eQ, dBQdtheta);
                    obj.J(m  , m  ) = dot(deQdl,     BQ) + dot(eQ, dBQdl);
                   
                    if (i > 1)
                       
                        aQ_left = obj.geodesics{i-1}.aQ;
                        rQ_left = obj.geodesics{i-1}.rQ;
                        
                        tQ_left = obj.geodesics{i-1}.R * obj.geodesics{i-1}.KQ.t;
                        BQ_left = obj.geodesics{i-1}.R * obj.geodesics{i-1}.KQ.B;
                        
                        dePdsQ_left    = (dot(eP, tQ_left)*eP - tQ_left) / lP;
                        dePdbetaQ_left = (dot(eP, BQ_left)*eP - BQ_left) / lP;
                        
                        dePdsP_left    = dePdsQ_left;
                        dePdbetaP_left = dePdbetaQ_left * aQ_left;
                        dePdtheta_left = dePdbetaQ_left * rQ_left;
                        dePdl_left     = dePdsQ_left;
                      
                        obj.J(m-3, m-7) = dot(dePdsP_left,    NP);
                        obj.J(m-3, m-6) = dot(dePdbetaP_left, NP);
                        obj.J(m-3, m-5) = dot(dePdtheta_left, NP);
                        obj.J(m-3, m-4) = dot(dePdl_left,     NP);
                        
                        obj.J(m-2, m-7) = dot(dePdsP_left,    BP);
                        obj.J(m-2, m-6) = dot(dePdbetaP_left, BP);
                        obj.J(m-2, m-5) = dot(dePdtheta_left, BP);
                        obj.J(m-2, m-4) = dot(dePdl_left,     BP);
                        
                    end
                   
                    if (i < n)
                       
                       tP_right = obj.geodesics{i+1}.R * obj.geodesics{i+1}.KP.t;
                       BP_right = obj.geodesics{i+1}.R * obj.geodesics{i+1}.KP.B;
                       
                       deQdsP_right    = (tP_right - dot(eQ, tP_right)*eQ) / lQ;
                       deQdbetaP_right = (BP_right - dot(eQ, BP_right)*eQ) / lQ;
                      
                       obj.J(m-1, m+1) = dot(NQ, deQdsP_right);
                       obj.J(m-1, m+2) = dot(NQ, deQdbetaP_right);
                       
                       obj.J(m  , m+1) = dot(BQ, deQdsP_right);
                       obj.J(m  , m+2) = dot(BQ, deQdbetaP_right);
                       
                    end
                    
                end
                
            end
            
        end
        

        function [obj] = computeNaturalGeodesicCorrections(obj)
            
            obj.Dxi = -obj.J \ obj.eps;
            
        end
        
        
        function [obj] = computeNewGeodesicParameters(obj)
            
            n = length(obj.wrappingObstacles);
            
            if (n > 0)
                
                for i=1:n
                  
                    DsP    = obj.Dxi(4*i-3, 1);
                    DbetaP = obj.Dxi(4*i-2, 1);
                    
                    QP_old = [ obj.q(5*i-4,1) ; ...
                               obj.q(5*i-3,1) ];
                    
                    obj.wrappingObstacles{i}.surface = obj.wrappingObstacles{i}.surface.evaluateSurface(QP_old);
                    
                    tP = obj.geodesics{i}.KP.t;
                    BP = obj.geodesics{i}.KP.B;
                    
                    xuP = obj.wrappingObstacles{i}.surface.surfaceData.xu;
                    xvP = obj.wrappingObstacles{i}.surface.surfaceData.xv;
                    
                    NP = obj.wrappingObstacles{i}.surface.surfaceData.N;
                    
                    xuP_perp = cross(xuP, NP);
                    xvP_perp = cross(xvP, NP);
                    
                    T = [ dot(xvP_perp, tP)/dot(xvP_perp, xuP), dot(xvP_perp, BP)/dot(xvP_perp, xuP) ;
                          dot(xuP_perp, tP)/dot(xuP_perp, xvP), dot(xuP_perp, BP)/dot(xuP_perp, xvP) ];
                      
                    QP_new = QP_old + T*[DsP; DbetaP];
                    
                    obj.q(5*i-4,1) = QP_new(1);
                    obj.q(5*i-3,1) = QP_new(2);
                    
                    Dtheta = obj.Dxi(4*i-1,1);
                    
                    t_rot = tP*cos(Dtheta) + BP*sin(Dtheta);
                    
                    [QdP_new, ~] = obj.wrappingObstacles{i}.projectVectorOntoTangentPlaneAndNormalize(QP_new, t_rot);
                    
                    obj.q(5*i-2,1) = QdP_new(1);
                    obj.q(5*i-1,1) = QdP_new(2);
                    
                    obj.q(5*i,1) = obj.q(5*i,1) + obj.Dxi(4*i,1);
                      
                end
                
            end
                        
        end
        
        
        function [obj] = doNewtonStep(obj)
           
            obj = obj.computeNaturalGeodesicCorrections();
            
            obj = obj.computeNewGeodesicParameters();
            
            obj = obj.update();
            
        end
        
        
        
        function [] = plotWrappingSystem(obj, ... 
                                         surfaceColor, ...
                                         straightLineSegmentStyle, ...
                                         geodesicSegmentStyle, ...
                                         pathLineWidth, ... 
                                         surfaceScale)
            
            n = length(obj.wrappingObstacles);
            
            if n > 0
            
                for i=1:n
                    
                    edgeColor = 'default';
                    obj.wrappingObstacles{i}.surface.plotSurface(surfaceColor, edgeColor, surfaceScale);
                
                    obj.geodesics{i}.plotGeodesicSegment(geodesicSegmentStyle, pathLineWidth);
                    
                    obj.straightLineSegments{i}.plotStraightLineSegment(straightLineSegmentStyle, pathLineWidth);
                    
                end
                
                obj.straightLineSegments{i+1}.plotStraightLineSegment(straightLineSegmentStyle, pathLineWidth);
           
            else
                
                obj.straightLineSegments{1}.plotStraightLineSegment(straightLineSegmentStyle, pathLineWidth);
                
            end
            
        end
      
    end
 
end

