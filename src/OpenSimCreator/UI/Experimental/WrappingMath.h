#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <iostream>
#include <vector>

namespace osc
{

    // The actual step of each natural geodesic variation:
    // {tangentiol, binormal, directional, lengthening}
    using GeodesicCorrection = std::array<double, 4>;

    using Rotation = Eigen::Quaternion<double>;
    using Mat3x3d  = Eigen::Matrix<double, 3, 3>;

    using Vector3 = Eigen::Vector3d;

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

    /*     DarbouxFrame operator*( */
    /*             const Rotation& lhs, */
    /*             const DarbouxFrame& rhs); */

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

    //==============================================================================
    //                      SURFACE
    //==============================================================================

    struct WrappingPath;

    // An abstract component class that you can calculate geodesics over.
    class Surface
    {
    public:

        virtual ~Surface() = default;

    protected:

        Surface()                              = default;
        Surface(Surface&&) noexcept            = default;
        Surface& operator=(Surface&&) noexcept = default;
        Surface(const Surface&)                = default;
        Surface& operator=(const Surface&)     = default;

    public:

        Geodesic calcGeodesic(
            Vector3 initPosition,
            Vector3 initVelocity,
            double length) const;

        using GetSurfaceFn = std::function<const Surface*(size_t)>;

        static WrappingPath calcNewWrappingPath(
            Vector3 pathStart,
            Vector3 pathEnd,
            GetSurfaceFn& GetSurface,
            double eps     = 1e-6,
            size_t maxIter = 50);

        static size_t calcUpdatedWrappingPath(
            WrappingPath& path,
            GetSurfaceFn& GetSurface,
            double eps     = 1e-6,
            size_t maxIter = 10);

        // TODO This is just here for the current test.
        void setOffsetFrame(Transf transform)
        {
            _transform = std::move(transform);
        }

        const Transf& getOffsetFrame() const;

    private:

        virtual Geodesic calcLocalGeodesicImpl(
            Vector3 initPosition,
            Vector3 initVelocity,
            double length) const = 0;

        // This would be a socket to an offset frame for example.
        Transf _transform;
    };

    //==============================================================================
    //                      ANALYTIC SPHERE SURFACE
    //==============================================================================

    // Concrete component.
    class AnalyticSphereSurface : public Surface
    {
    public:

        explicit AnalyticSphereSurface(double radius) : _radius(radius)
        {}

    private:

        Geodesic calcLocalGeodesicImpl(
            Vector3 initPosition,
            Vector3 initVelocity,
            double length) const override;

        double _radius;
    };

    //==============================================================================
    //                      ANALYTIC CYLINDER SURFACE
    //==============================================================================

    // Concrete component.
    class AnalyticCylinderSurface : public Surface
    {
    public:

        explicit AnalyticCylinderSurface(double radius) : _radius(radius)
        {}

    private:

        Geodesic calcLocalGeodesicImpl(
            Vector3 initPosition,
            Vector3 initVelocity,
            double length) const override;

        double _radius;
    };

    //==============================================================================
    //      WRAPPING PATH
    //==============================================================================

    // Captures the smoothness of the wrapping path.
    class PathContinuityError final
    {
    public:

        ~PathContinuityError()                                     = default;
        PathContinuityError()                                      = default;
        PathContinuityError(const PathContinuityError&)            = default;
        PathContinuityError(PathContinuityError&&) noexcept        = default;
        PathContinuityError& operator=(const PathContinuityError&) = default;
        PathContinuityError& operator=(PathContinuityError&&) noexcept =
            default;

        // Maximum alignment error of the tangents.
        double calcMaxPathError() const;

        // Maximum natural geodesic correction for reducing the path error.
        double calcMaxCorrectionStep() const;

        /* private: */
        // Pointer to first geodesic's correction.
        const GeodesicCorrection* begin() const;
        // Pointer to one past last geodesic's correction.
        const GeodesicCorrection* end() const;

        // Get access to the path error and path error jacobian.
        Eigen::VectorXd& updPathError();
        Eigen::MatrixXd& updPathErrorJacobian();

        // Compute the geodesic corrections from the path error and path error
        // jacobian.
        void calcPathCorrection();

        // Resize internal matrices to match the problem size (number of
        // surfaces).
        void resize(size_t nSurfaces);

        double _maxAngleDegrees = 5.;
        double _eps             = 1e-10;
        Eigen::VectorXd _solverError; // For debugging.
        Eigen::VectorXd _pathCorrections;
        Eigen::VectorXd _pathError;
        Eigen::MatrixXd _pathErrorJacobian;
        Eigen::JacobiSVD<Eigen::MatrixXd> _svd;
        size_t _nSurfaces = 0;

        friend Surface; // TODO change to whomever is calculating the path.
        friend WrappingPath calcNewWrappingPath(
            Vector3,
            Vector3,
            std::function<const Surface*(size_t)>,
            double,
            size_t);
        friend size_t calcUpdatedWrappingPath(
            WrappingPath& path,
            std::function<const Surface*(size_t)> surfaces,
            double,
            size_t);
    };

    // The result of computing a path over surfaces.
    struct WrappingPath
    {
        WrappingPath() = default;

        WrappingPath(Vector3 pStart, Vector3 pEnd) :
            startPoint(std::move(pStart)), endPoint(std::move(pEnd))
        {}

        Vector3 startPoint{
            NAN,
            NAN,
            NAN,
        };
        Vector3 endPoint{
            NAN,
            NAN,
            NAN,
        };
        std::vector<Geodesic> segments;
        PathContinuityError smoothness;
    };

} // namespace osc
