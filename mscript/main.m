% -------------------------------------------------------------------------
%
% Author: 
% Andreas Scholz
% Duisburg, 2022
% an.scho@t-online.de
%
% -------------------------------------------------------------------------

  clc
  clear all
  close all
  
  format short
 
% -------------------------------------------------------------------------
% Settings.
% -------------------------------------------------------------------------
  numberOfPathCorrections = 4;
  
  klickButtonToIterate = true;
  
% -------------------------------------------------------------------------
% Definition of wrapping surfaces incl. position and orientation.
% -------------------------------------------------------------------------
  rCyl = [-5 0 0]';
  RCyl = computeRotationMatrixFromEulerAngles([0 0.4*pi 0]);
  cyl  = Cylinder(rCyl, RCyl, [0 0 0]', [0 0 0]', 1.5, 4);

  rEll = [0 0 1]';
  REll = computeRotationMatrixFromEulerAngles([0 0 0]);
  ell  = Ellipsoid(rEll, REll, [0 0 0]', [0 0 0]', 2, 2.5, 3.5);
 
  rTor = [5 0 0]';
  RTor = computeRotationMatrixFromEulerAngles([0.5*pi 0.4*pi 0]);
  tor  = Torus(rTor, RTor, [0 0 0]', [0 0 0]', 2, 1);
  
% -------------------------------------------------------------------------
% Creation of "wrapping obstacles" from the surfaces.
% -------------------------------------------------------------------------
  wrappingCyl = WrappingObstacle(cyl);
  wrappingEll = WrappingObstacle(ell);
  wrappingTor = WrappingObstacle(tor);
  
% -------------------------------------------------------------------------
% Positions of origin and insertion.
% -------------------------------------------------------------------------
  O = [-10 -1  -2]';
  I = [ 10  1  -1]';
  
% -------------------------------------------------------------------------
% Definition of the initial guess for the geodesics. The first two
% arguments define the start point position, the next two arguments define
% the direction, and the last argument defines the length.
% -------------------------------------------------------------------------
  qCyl = [0.75*pi  0         -1     -0.2     2]';
  qEll = [1.1      2          0     -1       1]';
  qTor = [0        1.25*pi    0     -1       1]';

% -------------------------------------------------------------------------
% Creation of a muscle wrapping system. Wrapping obstacles can be added
% (order matters) together with the initial guesses for the geodesic
% segments.
% -------------------------------------------------------------------------
  muscleWrappingSystem = MuscleWrappingSystem(O, I);
  
  muscleWrappingSystem = muscleWrappingSystem.addWrappingObstacle(wrappingCyl, qCyl);
  muscleWrappingSystem = muscleWrappingSystem.addWrappingObstacle(wrappingEll, qEll);
  muscleWrappingSystem = muscleWrappingSystem.addWrappingObstacle(wrappingTor, qTor);
 
  figure(1)
  hold on
  axis equal
  view([190 30])
  title('Muscle Wrapping System')
 
  globalPathErrorNorm = zeros(numberOfPathCorrections, 1);
  cylinderError       = zeros(numberOfPathCorrections, 1);
  ellipsoidError      = zeros(numberOfPathCorrections, 1);
  torusError          = zeros(numberOfPathCorrections, 1);
 
  display('The initial path-error norm is:')
  display(num2str(muscleWrappingSystem.globalPathErrorNorm));
  
  display(['### Do ' num2str(numberOfPathCorrections) ' path iterations ...'])
  
% -------------------------------------------------------------------------
% Here we do manual iterations by clicking the mouse so that the iterations
% become visible. 
% ------------------------------------------------------------------------- 
  for i = 1:numberOfPathCorrections+1
 
    muscleWrappingSystem.plotWrappingSystem('white', ...
                                            '-r',    ...
                                            '-r',    ...
                                            1.0,     ...
                                            1.0)
                              
    globalPathErrorNorm(i,1) = muscleWrappingSystem.globalPathErrorNorm;
    
    muscleWrappingSystem = muscleWrappingSystem.doNewtonStep();
  
    if (klickButtonToIterate == true)
        waitforbuttonpress();
    end
                                      
  end
  
  display('... done.')
  
  display('The final path-error norm is:')
  display(num2str(muscleWrappingSystem.globalPathErrorNorm));
  
  figure(2)
  hold on
  grid on
  plot(0:numberOfPathCorrections, globalPathErrorNorm, '-k')
  title('Path error over iterations')
  xlabel('number of path iterations')
  
  
  
 
  
  
                                      

 
                           
  
  
  
