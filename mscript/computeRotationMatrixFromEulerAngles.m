% -------------------------------------------------------------------------
%
% Author: 
% Andreas Scholz
% Duisburg, 2022
% an.scho@t-online.de
%
% -------------------------------------------------------------------------

function [ R ] = computeRotationMatrixFromEulerAngles( Q )

    [layers, ~] = size(Q);
            
    R = zeros(3, 3, layers);
           
    for i=1:length(Q(:,1))
                
        sinpsi   = sin(Q(i,1));
        cospsi   = cos(Q(i,1));
        sintheta = sin(Q(i,2));
        costheta = cos(Q(i,2));
        sinphi   = sin(Q(i,3));
        cosphi   = cos(Q(i,3));
               
        R(1:3, 1:3, i) = [ cospsi*cosphi-sinpsi*costheta*sinphi, -cospsi*sinphi-sinpsi*costheta*cosphi,  sinpsi*sintheta;
                           sinpsi*cosphi+cospsi*costheta*sinphi, -sinpsi*sinphi+cospsi*costheta*cosphi, -cospsi*sintheta; 
                           sintheta*sinphi,                       sintheta*cosphi,                       costheta ]; 
            
    end
    
end

