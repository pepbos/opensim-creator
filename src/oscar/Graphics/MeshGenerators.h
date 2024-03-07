#pragma once

#include <oscar/Graphics/Mesh.h>
#include <oscar/Maths/Angle.h>
#include <oscar/Maths/Vec.h>
#include <oscar/Maths/Vec2.h>
#include <oscar/Maths/Vec3.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace osc
{
    // generates 2D grid lines at Z == 0, X/Y == [-1,+1]
    Mesh GenerateGridLinesMesh(size_t nticks);

    // generates the *lines* of a cube with [-1,+1] in each dimension
    Mesh GenerateCubeLinesMesh();

    // generates a torus knot, the particular shape of which is defined by a pair of coprime integers
    // `p` and `q`. If `p` and `q` are not coprime, the result will be a torus link
    //
    // inspired by three.js's `TorusKnotGeometry`
    Mesh GenerateTorusKnotMesh(
        float torusRadius = 1.0f,
        float tubeRadius = 0.4f,
        size_t numTubularSegments = 64,
        size_t numRadialSegments = 8,
        size_t p = 2,
        size_t q = 3
    );

    // generates a rectangular cuboid with the given dimensions centered on the origin, with each
    // edge parallel to each axis
    //
    // `segments` affects how many 2-triangle quads may be generated along each dimension
    Mesh GenerateBoxMesh(
        float width = 1.0f,
        float height = 1.0f,
        float depth = 1.0f,
        size_t widthSegments = 1,
        size_t heightSegments = 1,
        size_t depthSegments = 1
    );

    // generates a 3D solid with flat faces by projecting triangle faces (`indicies`
    // indexes into `vertices` for each triangle) onto a sphere, followed by dividing
    // them up to the desired level of detail
    Mesh GeneratePolyhedronMesh(
        std::span<Vec3 const> vertices,
        std::span<uint32_t const> indices,
        float radius,
        size_t detail
    );

    Mesh GenerateIcosahedronMesh(
        float radius = 1.0f,
        size_t detail = 0
    );

    Mesh GenerateDodecahedronMesh(
        float radius = 1.0f,
        size_t detail = 0
    );

    Mesh GenerateOctahedronMesh(
        float radius = 1.0f,
        size_t detail = 0
    );

    Mesh GenerateTetrahedronMesh(
        float radius = 1.0f,
        size_t detail = 0
    );

    // returns a mesh with axial symmetry like vases. The lathe rotates around the Y axis.
    //
    // (ported from three.js:LatheGeometry)
    Mesh GenerateLatheMesh(
        std::span<Vec2 const> points = std::vector<Vec2>{{0.0f, -0.5f}, {0.5f, 0.0f}, {0.0f, 0.5f}},
        size_t segments = 12,
        Radians phiStart = Degrees{0},
        Radians phiLength = Degrees{360}
    );

    // returns a mesh representation of a solid circle
    //
    // (ported from three.js:CircleGeometry)
    Mesh GenerateCircleMesh(
        float radius = 1.0f,
        size_t segments = 32,
        Radians thetaStart = Degrees{0},
        Radians thetaLength = Degrees{360}
    );

    // returns a mesh representation of a ring
    //
    // (ported from three.js/RingGeometry)
    Mesh GenerateRingMesh(
        float innerRadius = 0.5f,
        float outerRadius = 1.0f,
        size_t thetaSegments = 32,
        size_t phiSegments = 1,
        Radians thetaStart = Degrees{0},
        Radians thetaLength = Degrees{360}
    );

    // (ported from three.js/TorusGeometry)
    Mesh GenerateTorusMesh(
        float radius = 1.0f,
        float tube = 0.4f,
        size_t radialSegments = 12,
        size_t tubularSegments = 48,
        Radians arc = Degrees{360}
    );

    // (ported from three.js/CylinderGeometry)
    Mesh GenerateCylinderMesh(
        float radiusTop = 1.0f,
        float radiusBottom = 1.0f,
        float height = 1.0f,
        size_t radialSegments = 32,
        size_t heightSegments = 1,
        bool openEnded = false,
        Radians thetaStart = Degrees{0},
        Radians thetaLength = Degrees{360}
    );

    // (ported from three.js/ConeGeometry)
    Mesh GenerateConeMesh(
        float radius = 1.0f,
        float height = 1.0f,
        size_t radialSegments = 32,
        size_t heightSegments = 1,
        bool openEnded = false,
        Radians thetaStart = Degrees{0},
        Radians thetaLength = Degrees{360}
    );

    // (ported from three.js/PlaneGeometry)
    Mesh GeneratePlaneMesh(
        float width = 1.0f,
        float height = 1.0f,
        size_t widthSegments = 1,
        size_t heightSegments = 1
    );

    // (ported from three.js/SphereGeometry)
    Mesh GenerateSphereMesh(
        float radius = 1.0f,
        size_t widthSegments = 32,
        size_t heightSegments = 16,
        Radians phiStart = Degrees{0},
        Radians phiLength = Degrees{360},
        Radians thetaStart = Degrees{0},
        Radians thetaLength = Degrees{180}
    );

    // (ported from three.js/WireframeGeometry)
    Mesh GenerateWireframeMesh(Mesh const&);
}
