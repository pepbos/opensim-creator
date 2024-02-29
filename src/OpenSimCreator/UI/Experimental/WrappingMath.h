#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Geometry/Quaternion.h>

#include <vector>
#include <iostream>

namespace osc
{

using Rotation              = Eigen::Quaternion<double>;
using Mat3x3d                = Eigen::Matrix<double, 3, 3>;

/* static constexpr double INF = Eigen::Infinity; */
/* static constexpr double PI  = M_PI; */

using Vector3  = Eigen::Vector3d;

// Just here for this test sketch.
struct Transf
{
    /* Rotation orientation{1., 0., 0., 0.}; */
    Vector3 position{0., 0., 0.};
};

// TODO hold Simbody::Rotation inside?
struct DarbouxFrame
{
    DarbouxFrame() = default;

    DarbouxFrame(Vector3 surfaceTangent, Vector3 surfaceNormal);

    DarbouxFrame(Vector3 tangent, Vector3 normal, Vector3 binormal);

    Vector3 t{NAN, NAN, NAN};
    Vector3 n{NAN, NAN, NAN};
    Vector3 b{NAN, NAN, NAN};
};

//==============================================================================
//                      GEODESIC
//==============================================================================

// This is the result from shooting over a surface.
struct Geodesic
{
    struct BoundaryState
    {
        DarbouxFrame frame;
        Vector3 position{NAN, NAN, NAN};

        // Given the natural geodesic variations:
        //   0. Tangential
        //   1. Binormal
        //   2. InitialDirection
        //   3. Lengthening
        // we store the corresponding variations for the position and
        // DarbouxFrame as a position and rotation variation.
        //
        // We can ommit storing the lengthening variation, as it is equal to
        // zero for the start and equal to the tangential variation for the
        // end.

        // Natural geodesic position variation:
        std::array<Vector3, 4> v{
            Vector3{NAN, NAN, NAN}, // Tangential
            Vector3{NAN, NAN, NAN}, // Binormal
            Vector3{NAN, NAN, NAN}, // InitialDirection
            Vector3{NAN, NAN, NAN}, // TODO remove?
        };
        // Natural geodesic frame variation (a rotation):
        std::array<Vector3, 4> w = {
            Vector3{NAN, NAN, NAN}, // Tangential
            Vector3{NAN, NAN, NAN}, // Binormal
            Vector3{NAN, NAN, NAN}, // InitialDirection
            Vector3{NAN, NAN, NAN}, // TODO remove?
        };
    };

    BoundaryState start; // TODO Rename to K_P
    BoundaryState end;   // TODO Rename to K_Q
    double length;

    // Points and frames along the geodesic (TODO keep in local frame).
    std::vector<std::pair<Vector3, DarbouxFrame>> curveKnots;
};

}
