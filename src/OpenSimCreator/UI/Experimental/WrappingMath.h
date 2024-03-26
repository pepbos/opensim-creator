#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>

namespace osc
{

// The actual step of each natural geodesic variation:
// {tangentiol, binormal, directional, lengthening}
static constexpr size_t GEODESIC_DIM = 4;
using GeodesicCorrection             = std::array<double, GEODESIC_DIM>;

using Rotation = Eigen::Quaternion<double>;
using Mat3x3   = Eigen::Matrix<double, 3, 3>;

using Vector3 = Eigen::Vector3d;

// Just here for this test sketch.
struct Transf
{
    /* Rotation orientation{1., 0., 0., 0.}; */
    Vector3 position{0., 0., 0.};
};

//==============================================================================
//                      RUNGE KUTTA 4
//==============================================================================

// Just here for this test sketch.
template <typename Y, typename D, typename DY = Y>
void RungeKutta4(Y& y, double& t, double dt, std::function<D(const Y&)> f)
{
    D k0, k1, k2, k3;

    {
        const Y& yk = y;
        k0          = f(yk);
    }

    {
        const double h = dt / 2.;
        Y yk           = y + (h * k0);
        k1             = f(yk);
    }

    {
        const double h = dt / 2.;
        Y yk           = y + (h * k1);
        k2             = f(yk);
    }

    {
        const double h = dt;
        Y yk           = y + (h * k2);
        k3             = f(yk);
    }

    const double w = dt / 6.;

    y = y + (w * k0 + (w * 2.) * k1 + (w * 2.) * k2 + w * k3);
    t += dt;
}

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
    enum Status
    {
        Ok                             = 0,
        InitialTangentParallelToNormal = 1, // TODO
        PrevLineSegmentInsideSurface   = 2, // TODO
        NextLineSegmentInsideSurface   = 4, // TODO
        NegativeLength                 = 8,
        LiftOff                        = 16, // TODO
        TouchDownFailed                = 32, // TODO
        /* IntegratorFailed               = 64, // TODO */
    };

    struct InitialConditions
    {
        Vector3 position{NAN, NAN, NAN};
        Vector3 velocity{NAN, NAN, NAN};
        double length = NAN;
    };

    using Sample = std::pair<Vector3, DarbouxFrame>;

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
    std::vector<Sample> samples;

    Status status = Status::Ok;
};

std::ostream& operator<<(std::ostream& os, const Geodesic& x);

std::ostream& operator<<(std::ostream& os, const Geodesic::Status& s);

inline Geodesic::Status operator|(Geodesic::Status lhs, Geodesic::Status rhs)
{
    using T = std::underlying_type_t<Geodesic::Status>;
    return static_cast<Geodesic::Status>(
        static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline Geodesic::Status operator&(Geodesic::Status lhs, Geodesic::Status rhs)
{
    using T = std::underlying_type_t<Geodesic::Status>;
    return static_cast<Geodesic::Status>(
        static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline Geodesic::Status operator~(Geodesic::Status s)
{
    using T = std::underlying_type_t<Geodesic::Status>;
    return static_cast<Geodesic::Status>(~static_cast<T>(s));
}

inline Geodesic::Status& operator|=(Geodesic::Status& lhs, Geodesic::Status rhs)
{
    using T = std::underlying_type_t<Geodesic::Status>;
    lhs     = static_cast<Geodesic::Status>(
        static_cast<T>(lhs) | static_cast<T>(rhs));
    return lhs;
}

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
    static constexpr double MIN_DIST_FROM_SURF =
        1e-3; // TODO this must be a setting.

    void calcGeodesic(
        Vector3 initPosition,
        Vector3 initVelocity,
        double length,
        Geodesic& geodesic) const;

    void calcGeodesic(
        Vector3 initPosition,
        Vector3 initVelocity,
        double length,
        Vector3 pointBefore,
        Vector3 pointAfter,
        Geodesic& geodesic) const;

    // TODO This is just here for the current test.
    void setOffsetFrame(Transf transform)
    {
        _transform = std::move(transform);
    }

    const Transf& getOffsetFrame() const;

    Vector3 getPathStartGuess() const;

    void setLocalPathStartGuess(Vector3 pathStartGuess);

    bool isAboveSurface(Vector3 point, double bound) const;

    size_t calcAccurateLocalSurfaceProjection(
        Vector3 pointInit,
        Vector3& point,
        DarbouxFrame& frame,
        double eps,
        size_t maxIter) const;

    void applyVariation(Geodesic& geodesic, const GeodesicCorrection& var) const;

    // For convenience:
    // (Assumes the point lies on the surface)
    Vector3 calcSurfaceNormal(Vector3 point) const;
    double calcNormalCurvature(Vector3 point, Vector3 tangent) const;
    double calcGeodesicTorsion(Vector3 point, Vector3 tangent) const;

private:
    virtual void calcLocalGeodesicImpl(
        Vector3 initPosition,
        Vector3 initVelocity,
        double length,
        Geodesic& geodesic) const = 0;

    // Required for touchdown.
    virtual size_t calcAccurateLocalSurfaceProjectionImpl(
        Vector3 pointInit,
        Vector3& point,
        DarbouxFrame& frame,
        double eps,
        size_t maxIter) const = 0;

    virtual bool isAboveSurfaceImpl(Vector3 point, double bound) const = 0;

    virtual Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const = 0;
    virtual double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const = 0;
    virtual double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const = 0;

    // This would be a socket to an offset frame for example.
    Transf _transform;

    // TODO this should not be surface dependent.
    // TODO weird guess (keep until fixed)
    Vector3 _pathLocalStartGuess = {1., 1., 1.};
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

    // TODO DO NOT USE THIS. FOR TESTING ONLY.
    // Computes the normal curvature. Arguments are in global coordinates.
    // Does not project the point and tangent to the surface.
    // Unsafe: for testing only.
    double testCalcNormalCurvature(Vector3 point, Vector3 tangent) const;

    // TODO DO NOT USE THIS. FOR TESTING ONLY.
    // Computes the geodesic torsion. Arguments are in global coordinates.
    // Does not project the point and tangent to the surface.
    // Unsafe: for testing only.
    double testCalcGeodesicTorsion(Vector3 point, Vector3 tangent) const;

    // TODO DO NOT USE THIS. FOR TESTING ONLY.
    // Computes the surface normal direction Argument is in global coordinates.
    // Does not project the point to the surface.
    // Unsafe: for testing only.
    Vector3 testCalcSurfaceNormal(Vector3 point) const;

    Vector3 testCalcAcceleration(Vector3 point, Vector3 tangent) const;

private:
    // Implicit surface constraint.
    virtual double calcSurfaceConstraintImpl(Vector3 position) const = 0;
    virtual Vector3 calcSurfaceConstraintGradientImpl(
        Vector3 position) const = 0;
    virtual Hessian calcSurfaceConstraintHessianImpl(
        Vector3 position) const = 0;

    void calcLocalGeodesicImpl(
        Vector3 initPosition,
        Vector3 initVelocity,
        double length,
        Geodesic& geodesic) const override;

    size_t calcAccurateLocalSurfaceProjectionImpl(
        Vector3 pointInit,
        Vector3& point,
        DarbouxFrame& frame,
        double eps,
        size_t maxIter) const override;

    Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const override;
    double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const override;
    double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const override;

    // TODO would become obsolete with variable step integration.
    size_t _integratorSteps = 1000;
};

//==============================================================================
//                      IMPLICIT ELLIPSOID SURFACE
//==============================================================================

// Concrete component.
class ImplicitEllipsoidSurface : public ImplicitSurface
{
public:
    ImplicitEllipsoidSurface() = default;

    explicit ImplicitEllipsoidSurface(
        double xRadius,
        double yRadius,
        double zRadius) :
        _xRadius(xRadius), _yRadius(yRadius), _zRadius(zRadius)
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
    Vector3 calcSurfaceConstraintGradientImpl(Vector3 position) const override;
    Hessian calcSurfaceConstraintHessianImpl(Vector3 position) const override;

    virtual bool isAboveSurfaceImpl(Vector3 point, double bound) const override;

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
    Vector3 calcSurfaceConstraintGradientImpl(Vector3 position) const override;
    Hessian calcSurfaceConstraintHessianImpl(Vector3 position) const override;

    virtual bool isAboveSurfaceImpl(Vector3 point, double bound) const override;

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
    void calcLocalGeodesicImpl(
        Vector3 initPosition,
        Vector3 initVelocity,
        double length,
        Geodesic& geodesic) const override;

    bool isAboveSurfaceImpl(Vector3 point, double bound) const override;

    size_t calcAccurateLocalSurfaceProjectionImpl(
        Vector3 pointInit,
        Vector3& point,
        DarbouxFrame& frame,
        double eps,
        size_t maxIter) const override;

    Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const override;
    double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const override;
    double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const override;

    double _radius = 1.;
};

//==============================================================================
//                      IMPLICIT CYLINDER SURFACE
//==============================================================================

// Concrete component.
class ImplicitCylinderSurface : public ImplicitSurface
{
public:
    ImplicitCylinderSurface() = default;

    explicit ImplicitCylinderSurface(double radius) : _radius(radius)
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
    Vector3 calcSurfaceConstraintGradientImpl(Vector3 position) const override;
    Hessian calcSurfaceConstraintHessianImpl(Vector3 position) const override;

    bool isAboveSurfaceImpl(Vector3 point, double bound) const override;

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
    void calcLocalGeodesicImpl(
        Vector3 initPosition,
        Vector3 initVelocity,
        double length,
        Geodesic& geodesic) const override;

    bool isAboveSurfaceImpl(Vector3 point, double bound) const override;

    size_t calcAccurateLocalSurfaceProjectionImpl(
        Vector3 pointInit,
        Vector3& point,
        DarbouxFrame& frame,
        double eps,
        size_t maxIter) const override;

    Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const override;
    double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const override;
    double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const override;

    double _radius = 1.;
};

//==============================================================================
//                      IMPLICIT TORUS SURFACE
//==============================================================================

// Concrete component.
class ImplicitTorusSurface : public ImplicitSurface
{
public:
    ImplicitTorusSurface() = default;

    explicit ImplicitTorusSurface(double smallRadius, double bigRadius) :
        _smallRadius(smallRadius),
        _bigRadius(bigRadius)
    {}

    double getBigRadius() const
    {
        return _smallRadius;
    }

    double getSmallRadius() const
    {
        return _bigRadius;
    }

    void setRadii(double a, double b)
    {
        _smallRadius = std::min(std::abs(a), std::abs(b));
        _bigRadius = std::max(std::abs(a), std::abs(b));
    }

private:
    // Implicit surface constraint.
    double calcSurfaceConstraintImpl(Vector3 position) const override;
    Vector3 calcSurfaceConstraintGradientImpl(Vector3 position) const override;
    Hessian calcSurfaceConstraintHessianImpl(Vector3 position) const override;

    bool isAboveSurfaceImpl(Vector3 point, double bound) const override;

    double _smallRadius = 0.1;
    double _bigRadius = 1.;
};

//==============================================================================
//      WRAPPING PATH
//==============================================================================

struct CorrectionBounds
{
    double maxAngleDegrees  = 300.;
    double maxLengthening   = 1e2;
    double maxRepositioning = 1e2;
};

// Captures the smoothness of the wrapping path.
class PathContinuityError final
{
public:
    ~PathContinuityError()                                         = default;
    PathContinuityError()                                          = default;
    PathContinuityError(const PathContinuityError&)                = default;
    PathContinuityError(PathContinuityError&&) noexcept            = default;
    PathContinuityError& operator=(const PathContinuityError&)     = default;
    PathContinuityError& operator=(PathContinuityError&&) noexcept = default;

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
    bool calcPathCorrection();

    // Resize internal matrices to match the problem size (number of
    // surfaces).
    void resize(size_t rows, size_t cols);

    static const size_t NUMBER_OF_CONSTRAINTS = 6;

    void resize(size_t nSurfaces);

    CorrectionBounds maxStep;

    double _eps = 1e-10;
    Eigen::VectorXd _solverError; // For debugging.
    Eigen::VectorXd _pathCorrections;
    Eigen::VectorXd _pathError;
    Eigen::MatrixXd _pathErrorJacobian;

    Eigen::MatrixXd _costP;
    Eigen::MatrixXd _costQ;
    Eigen::MatrixXd _costL;
    Eigen::VectorXd _vecL;

    Eigen::MatrixXd _mat;
    Eigen::VectorXd _vec;
    Eigen::VectorXd _solve;

    Eigen::MatrixXd _matSmall;
    Eigen::VectorXd _vecSmall;
    Eigen::VectorXd _solveSmall;

    double _length = 0.;
    Eigen::VectorXd _lengthJacobian;

    Eigen::JacobiSVD<Eigen::MatrixXd> _svd;
    size_t _nSurfaces = 0;

    friend Surface; // TODO change to whomever is calculating the path.
    friend WrappingPath;
};

// The result of computing a path over surfaces.
struct WrappingPath
{
    using GetSurfaceFn = std::function<const Surface*(size_t)>;

    WrappingPath() = default;

    WrappingPath(Vector3 pathStart, Vector3 pathEnd, GetSurfaceFn& GetSurface);

    size_t updPath(
        GetSurfaceFn& GetSurface,
        double eps     = 1e-6,
        size_t maxIter = 10);

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
    std::vector<Geodesic> segments = {};
    PathContinuityError smoothness = {};

    enum Status
    {
        Ok                     = 0,
        FailedToInvertJacobian = 1,
        ExceededMaxIterations  = 2,
    } status = Status::Ok;
};

struct SegmentIterator final
{
    struct Segment
    {
        Geodesic* prev    = nullptr;
        Geodesic* current = nullptr;
        Geodesic* next    = nullptr;
    };

    using iterator_category = std::input_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = Segment;
    using pointer           = const value_type*;
    using reference         = const value_type&;

    SegmentIterator() = default;

private:
    explicit SegmentIterator(WrappingPath& path) :
        m_Begin(&*path.segments.begin()), m_End(&*path.segments.end()),
        m_Prev(m_Begin), m_Next(m_Begin), m_Segment({nullptr, m_Begin, nullptr})
    {
        updNext();
    }

public:
    static SegmentIterator Begin(WrappingPath& path)
    {
        return SegmentIterator(path);
    }

    static SegmentIterator End(WrappingPath& path)
    {
        SegmentIterator end;
        end.m_Segment = {nullptr, &*path.segments.end(), nullptr};
        return end;
    }

    reference operator*() const
    {
        return m_Segment;
    }
    pointer operator->()
    {
        return &m_Segment;
    }

private:
    // TODO use algorithm from standard library.
    void updPrev()
    {
        for (; m_Prev != m_Segment.current; ++m_Prev) {
            // Check if segment is active.
            if ((m_Prev->status & Geodesic::Status::LiftOff) == 0) {
                m_Segment.prev = m_Prev;
            }
        }
    }

    // TODO use algorithm from standard library.
    void updNext()
    {
        while (m_End != ++m_Next) {
            // Check if segment is active.
            if ((m_Next->status & Geodesic::Status::LiftOff) == 0) {
                break;
            }
        }
        m_Segment.next = m_End == m_Next ? nullptr : m_Next;
    }

public:
    // Prefix increment
    SegmentIterator& operator++()
    {
        ++m_Segment.current;
        updPrev();
        updNext();
        return *this;
    }

    // Postfix increment
    SegmentIterator operator++(int)
    {
        SegmentIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(
        const SegmentIterator& lhs,
        const SegmentIterator& rhs)
    {
        return lhs.m_Segment.current == rhs.m_Segment.current;
    };
    friend bool operator!=(
        const SegmentIterator& lhs,
        const SegmentIterator& rhs)
    {
        return lhs.m_Segment.current != rhs.m_Segment.current;
    };

private:
    Geodesic* m_Begin = nullptr;
    Geodesic* m_End   = nullptr;

    Geodesic* m_Prev = nullptr;
    Geodesic* m_Next = nullptr;

    Segment m_Segment;
};

std::ostream& operator<<(std::ostream& os, const WrappingPath::Status& s);

inline WrappingPath::Status operator|(
    WrappingPath::Status lhs,
    WrappingPath::Status rhs)
{
    return static_cast<WrappingPath::Status>(
        static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline WrappingPath::Status operator&(
    WrappingPath::Status lhs,
    WrappingPath::Status rhs)
{
    return static_cast<WrappingPath::Status>(
        static_cast<int>(lhs) & static_cast<int>(rhs));
}

inline WrappingPath::Status operator~(WrappingPath::Status s)
{
    return static_cast<WrappingPath::Status>(~static_cast<int>(s));
}

} // namespace osc
