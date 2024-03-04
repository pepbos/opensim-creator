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
    using Mat3x3   = Eigen::Matrix<double, 3, 3>;

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

        Geodesic calcWrappingPath(Vector3 pointBefore, Vector3 pointAfter)
            const;

        using GetSurfaceFn = std::function<const Surface*(size_t)>;

        static WrappingPath calcNewWrappingPath(
            Vector3 pathStart,
            Vector3 pathEnd,
            GetSurfaceFn& GetSurface,
            double eps     = 1e-3,
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

        virtual std::vector<Vector3> makeSelfTestPoints() const;
        virtual std::vector<Vector3> makeSelfTestVelocities() const;
        virtual std::vector<double> makeSelfTestLengths() const;
        virtual double selfTestEquivalentRadius() const = 0;

        // This would be a socket to an offset frame for example.
        Transf _transform;

        void doSelfTest(
            Vector3 initPosition,
            Vector3 initVelocity,
            double length,
            double eps = 1e-3,
            double delta = 1e-4
            ) const;

    public:

        // Move out of class.
        void doSelfTests(double eps = 1e-3) const;
    };

    //==============================================================================
    //                      IMPLICIT SURFACE
    //==============================================================================

    class ImplicitSurface : public virtual Surface
    {
    public:

        // TODO Use symmetric matrix class.
        using Hessian = Mat3x3;

        virtual ~ImplicitSurface() = default;

    protected:

        ImplicitSurface()                                      = default;
        ImplicitSurface(const ImplicitSurface&)                = default;
        ImplicitSurface(ImplicitSurface&&) noexcept            = default;
        ImplicitSurface& operator=(const ImplicitSurface&)     = default;
        ImplicitSurface& operator=(ImplicitSurface&&) noexcept = default;

    public:

        // TODO put local in front of everything?

        double calcSurfaceConstraint(Vector3 position) const;
        Vector3 calcSurfaceConstraintGradient(Vector3 position) const;
        Hessian calcSurfaceConstraintHessian(Vector3 position) const;

    private:

        // Implicit surface constraint.
        virtual double calcSurfaceConstraintImpl(Vector3 position) const = 0;
        virtual Vector3 calcSurfaceConstraintGradientImpl(
            Vector3 position) const = 0;
        virtual Hessian calcSurfaceConstraintHessianImpl(
            Vector3 position) const = 0;

        Geodesic calcLocalGeodesicImpl(
            Vector3 initPosition,
            Vector3 initVelocity,
            double length) const override;

        // TODO would become obsolete with variable step integration.
        size_t _integratorSteps = 100;
    };

    //==============================================================================
    //                      IMPLICIT ELLIPSOID SURFACE
    //==============================================================================

    // Concrete component.
    class ImplicitEllipsoidSurface : public ImplicitSurface
    {
    public:

        ImplicitEllipsoidSurface() = default;

        explicit ImplicitEllipsoidSurface(double xRadius, double yRadius, double zRadius) :
            _xRadius(xRadius),
            _yRadius(yRadius),
            _zRadius(zRadius)
        {}

        Vector3 getRadii() const
        {
            return {_xRadius, _yRadius, _zRadius};
        }

        void setRadii(double xRadius, double yRadius, double zRadius)
        {
            _xRadius = xRadius;
            _yRadius = yRadius;
            _zRadius = zRadius;
        }

    private:

        // Implicit surface constraint.
        double calcSurfaceConstraintImpl(Vector3 position) const override;
        Vector3 calcSurfaceConstraintGradientImpl(
            Vector3 position) const override;
        Hessian calcSurfaceConstraintHessianImpl(
            Vector3 position) const override;

        double selfTestEquivalentRadius() const override {return
            std::min(_xRadius,  std::min(_yRadius, _zRadius));}

        double _xRadius = 1.;
        double _yRadius = 1.;
        double _zRadius = 1.;
    };

    //==============================================================================
    //                      IMPLICIT SPHERE SURFACE
    //==============================================================================

    // Concrete component.
    class ImplicitSphereSurface : public ImplicitSurface
    {
    public:

        explicit ImplicitSphereSurface(double radius) : _radius(radius)
        {}

        double getRadius() const
        {
            return _radius;
        }
        void setRadius(double radius)
        {
            _radius = radius;
        }

    private:

        // Implicit surface constraint.
        double calcSurfaceConstraintImpl(Vector3 position) const override;
        Vector3 calcSurfaceConstraintGradientImpl(
            Vector3 position) const override;
        Hessian calcSurfaceConstraintHessianImpl(
            Vector3 position) const override;

        double selfTestEquivalentRadius() const override {return _radius;}

        double _radius = 1.;
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

        double getRadius() const
        {
            return _radius;
        }
        void setRadius(double radius)
        {
            _radius = radius;
        }

    private:

        Geodesic calcLocalGeodesicImpl(
            Vector3 initPosition,
            Vector3 initVelocity,
            double length) const override;

        double selfTestEquivalentRadius() const override {return _radius;}

        double _radius = 1.;
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

        double getRadius() const
        {
            return _radius;
        }
        void setRadius(double radius)
        {
            _radius = radius;
        }

    private:

        Geodesic calcLocalGeodesicImpl(
            Vector3 initPosition,
            Vector3 initVelocity,
            double length) const override;


        double selfTestEquivalentRadius() const override {return _radius;}

        double _radius = 1.;
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
