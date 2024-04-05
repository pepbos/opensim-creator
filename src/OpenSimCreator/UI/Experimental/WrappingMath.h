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
/* static constexpr size_t GEODESIC_DIM = 4; */
/* using GeodesicCorrection             = std::array<double, GEODESIC_DIM>; */

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

template <typename Y, typename DY = Y, typename S = Y>
class RungeKuttaMerson
{
    public:
    RungeKuttaMerson() = default;

    RungeKuttaMerson(double hMin, double hMax, double accuracy): _h0(hMin), _hMin(hMin), _hMax(hMax), _accuracy(accuracy) {}

    // Integrate y0, populating y1, y2.
    double step(double h, std::function<DY(const Y&)>& f);

    double stepTo(
            Y y0,
            double x1,
            std::function<DY(const Y&)>& f,
            std::function<void(Y&)>& g
            );

    struct Sample
    {
        Sample(double xk, const Y& yk) : x(xk), y({yk}) {}

        double x;
        S y;
    };

    const std::vector<Sample>& getSamples() const {return _samples;}
    size_t getNumberOfFailedSteps() const {return _failedCount;}

    private:
    static constexpr size_t ORDER = 5;

    std::array<DY, ORDER> _k {};
    std::array<Y, 3> _y {};

    std::vector<Sample> _samples {};

    double _h0 = 1e-6;
    double _hMin = 1e-6;
    double _hMax = 1e-1;
    double _accuracy = 1e-4;

    size_t _failedCount = 0;
};

class Darboux
{
    public:
    using Matrix = Eigen::Matrix<double, 3, 3, Eigen::ColMajor>;

    Darboux();

    static Darboux FromTangenGuessAndNormal(Vector3 tangentGuess, Vector3 normal);

    Darboux(Vector3 tangent, Vector3 normal, Vector3 binormal);

    const Vector3& t() const {return reinterpret_cast<const Vector3&>(*_rotation.data());}
    const Vector3& n() const {return reinterpret_cast<const Vector3&>(*(_rotation.data() + 3));}
    const Vector3& b() const {return reinterpret_cast<const Vector3&>(*(_rotation.data() + 6));}

    const Darboux::Matrix& matrix() const {return _rotation;}

    private:
    Eigen::Matrix<double, 3, 3, Eigen::ColMajor> _rotation;
};

class Trihedron
{
    public:
        Trihedron() = default;

        Trihedron(Vector3 point, Darboux rotation);

        Trihedron(Vector3 point,
                Vector3 tangent,
                Vector3 normal,
                Vector3 binormal);

        static Trihedron FromPointAndTangentGuessAndNormal(Vector3 point,
                Vector3 tangentGuess, Vector3 normal);

        const Vector3& t() const {return _rotation.t();}
        const Vector3& n() const {return _rotation.n();}
        const Vector3& b() const {return _rotation.b();}

        const Darboux::Matrix& R() const {return _rotation.matrix(); }
        const Vector3& p() const {return _point;}

        Vector3& updPoint() {return _point;}

    private:
        Vector3 _point = {NAN, NAN, NAN};
        Darboux _rotation;
};

//==============================================================================
//                      GEODESIC
//==============================================================================

struct Geodesic
{
    enum Status
    {
        Ok                             = 0,
        InitialTangentParallelToNormal = 1,
        PrevLineSegmentInsideSurface   = 2,
        NextLineSegmentInsideSurface   = 4,
        NegativeLength                 = 8,
        LiftOff                        = 16,
        TouchDownFailed                = 32,
        IntegratorFailed               = 64,
        Disabled                       = 128,
    };

    static constexpr size_t DOF = 4;

    using Variation  = Eigen::Matrix<double, 3, DOF>;
    using Correction = Eigen::Vector<double, DOF>;

    Trihedron K_P;
    Trihedron K_Q;

    // Variations (in local frame, not body frame).
    Variation v_P;
    Variation w_P;

    Variation v_Q;
    Variation w_Q;

    double length;

    // Points and frames along the geodesic (TODO keep in local frame).
    std::vector<Trihedron> samples;

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

struct WrappingArgs
{
    bool m_CostP = true;
    bool m_CostQ = true;
    bool m_CostL = false;
    bool m_CostT = true;
    bool m_CostN = false;
    bool m_CostB = false;
    bool m_Augment = false;
    bool m_Cache = false;
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
    static constexpr double MIN_DIST_FROM_SURF =
        1e-3; // TODO this must be a setting.

    void calcGeodesic(
        Vector3 initPosition,
        Vector3 initVelocity,
        double length,
        Geodesic& geodesic);

    void calcGeodesic(
        Vector3 initPosition,
        Vector3 initVelocity,
        double length,
        Vector3 pointBefore,
        Vector3 pointAfter,
        Geodesic& geodesic);

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
        Trihedron& K,
        double eps,
        size_t maxIter) const;

    std::pair<Vector3, size_t> calcPointOnLineNearSurface(Vector3 a, Vector3 b, double eps, size_t maxIter) const;
    std::pair<Vector3, size_t> calcLocalPointOnLineNearSurface(Vector3 a, Vector3 b, double eps, size_t maxIter) const;

    size_t calcPointOnSurfaceNearLine(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const;
    size_t calcLocalPointOnSurfaceNearLine(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const;

    size_t calcLocalTrihedronOnLineNearSurface(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const;

    void applyVariation(Geodesic& geodesic, const Geodesic::Correction& var) const;

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
        Geodesic& geodesic) = 0;

    // Required for touchdown.
    virtual size_t calcAccurateLocalSurfaceProjectionImpl(
        Vector3 pointInit,
        Trihedron& K,
        double eps,
        size_t maxIter) const = 0;

    virtual bool isAboveSurfaceImpl(Vector3 point, double bound) const = 0;

    virtual Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const = 0;
    virtual double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const = 0;
    virtual double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const = 0;

    virtual std::pair<Vector3, size_t> calcLocalPointOnLineNearSurfaceImpl(Vector3 a, Vector3 b, double eps, size_t maxIter) const = 0;
    virtual size_t calcLocalPointOnSurfaceNearLineImpl(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const = 0;
    virtual size_t calcLocalTrihedronOnLineNearSurfaceImpl(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const = 0;

    // This would be a socket to an offset frame for example.
    Transf _transform;

    // TODO this should not be surface dependent.
    // TODO weird guess (keep until fixed)
    Vector3 _pathLocalStartGuess = {1., 1., 1.};
};

//==============================================================================
//                         IMPLICIT SURFACE STATE
//==============================================================================

struct ImplicitGeodesicState
{
    ImplicitGeodesicState() = default;

    ImplicitGeodesicState(Vector3 aPosition, Vector3 aVelocity) :
        position(std::move(aPosition)), velocity(std::move(aVelocity)){};

    Vector3 position = {NAN, NAN, NAN};
    Vector3 velocity = {NAN, NAN, NAN};
    double a         = 1.;
    double aDot      = 0.;
    double r         = 0.;
    double rDot      = 1.;
};

struct ImplicitGeodesicStateDerivative
{
    Vector3 velocity     = {NAN, NAN, NAN};
    Vector3 acceleration = {NAN, NAN, NAN};
    double aDot          = NAN;
    double aDDot         = NAN;
    double rDot          = NAN;
    double rDDot         = NAN;
};

ImplicitGeodesicState operator*(
        double dt,
        const ImplicitGeodesicStateDerivative& dy);

ImplicitGeodesicState operator+(
        const ImplicitGeodesicState& lhs,
        const ImplicitGeodesicState& rhs);

ImplicitGeodesicState operator-(
        const ImplicitGeodesicState& lhs,
        const ImplicitGeodesicState& rhs);

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
        Geodesic& geodesic) override;

    size_t calcAccurateLocalSurfaceProjectionImpl(
        Vector3 pointInit,
        Trihedron& K,
        double eps,
        size_t maxIter) const override;

    std::pair<Vector3, size_t> calcLocalPointOnLineNearSurfaceImpl(Vector3 a, Vector3 b, double eps, size_t maxIter) const override;
    size_t calcLocalPointOnSurfaceNearLineImpl(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const override;
    size_t calcLocalTrihedronOnLineNearSurfaceImpl(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const override;

    Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const override;
    double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const override;
    double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const override;

    RungeKuttaMerson<ImplicitGeodesicState, ImplicitGeodesicStateDerivative> _rkm;
};

void RunIntegratorTests();

class SphereSurface final
{
    public:
    explicit SphereSurface(double radius) : _radius(radius) {}

    double getRadius() const;

    private:
    double _radius = NAN;
};

class WrapSurface
{
    public:
    struct GeodesicInitConditions {

    };

    struct GeodesicBoundaryState {
        GeodesicInitConditions applyVariation() const;

        // etc ...
        double lenght = NAN;

    };

    enum Status {

    };

    struct Sample {};

    // Interface:
    private:

    virtual void calcLocalGeodescImpl(GeodesicInitConditions q0) = 0;

    virtual bool calcLocalLineToSurfaceTouchdownPointImpl(Vector3& p) = 0;

    virtual const GeodesicBoundaryState& getLocalGeodesicBoundaryStateImpl() const = 0;

    virtual void writeLocalGeodesicImpl(std::vector<Sample>& samples) const = 0;

    // Supplied:
    public:

    Transf& updTransform();
    const Transf& getTransform();

    void calcLocalGeodesic(GeodesicInitConditions q0);
    void calcLocalLineToSurfaceTouchdownPoint();
    const GeodesicBoundaryState& getLocalGeodesicBoundaryState() const;
    void writeLocalGeodesic(std::vector<Sample>& samples) const;

    Status getStatus() const;

    private:
    Transf _transform;
    Status _status;
};

class ImplicitWrapSurface
{
    /* struct Sample { */
    /*     Sample(ImplicitGeodesicState q): p(q.position), t(q.velocity) {} */

    /*     Vector3 p = {NAN, NAN, NAN}; */
    /*     Vector3 t = {NAN, NAN, NAN}; */
    /* }; */

    /* RungeKuttaMerson<ImplicitGeodesicState, ImplicitGeodesicStateDerivative, Sample> */
};

class WrapPath
{
    void calcWrappingPath();

    std::vector<WrapSurface> updSurfaces();
    Vector3& updPathStart();
    Vector3& updPathEnd();
    WrappingArgs& updArgs();

    std::vector<WrapSurface> getSurfaces();

    const std::vector<std::vector<Vector3>>& getPathPoints() const;
    double getLength() const;
    double getLengtheningSpeed() const;
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
        Geodesic& geodesic) override;

    bool isAboveSurfaceImpl(Vector3 point, double bound) const override;

    size_t calcAccurateLocalSurfaceProjectionImpl(
        Vector3 pointInit,
        Trihedron& K,
        double eps,
        size_t maxIter) const override;

    Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const override;
    double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const override;
    double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const override;

    std::pair<Vector3, size_t> calcLocalPointOnLineNearSurfaceImpl(Vector3 a, Vector3 b, double eps, size_t maxIter) const override;
    size_t calcLocalPointOnSurfaceNearLineImpl(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const override;
    size_t calcLocalTrihedronOnLineNearSurfaceImpl(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const override;

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
        Geodesic& geodesic) override;

    bool isAboveSurfaceImpl(Vector3 point, double bound) const override;

    size_t calcAccurateLocalSurfaceProjectionImpl(
        Vector3 pointInit,
        Trihedron& K,
        double eps,
        size_t maxIter) const override;

    Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const override;
    double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const override;
    double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const override;

    std::pair<Vector3, size_t> calcLocalPointOnLineNearSurfaceImpl(Vector3 a, Vector3 b, double eps, size_t maxIter) const override;
    size_t calcLocalPointOnSurfaceNearLineImpl(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const override;
    size_t calcLocalTrihedronOnLineNearSurfaceImpl(Trihedron& K, Vector3 a, Vector3 b, double eps, size_t maxIter) const override;

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
    const Geodesic::Correction* begin() const;
    // Pointer to one past last geodesic's correction.
    const Geodesic::Correction* end() const;

    // Get access to the path error and path error jacobian.
    Eigen::VectorXd& updPathError();
    Eigen::MatrixXd& updPathErrorJacobian();

    // Compute the geodesic corrections from the path error and path error
    // jacobian.
    bool calcPathCorrection(const WrappingArgs& args);

    static const size_t NUMBER_OF_CONSTRAINTS = 6;

    void resize(size_t nSurfaces, const WrappingArgs& args);

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
    using GetSurfaceFn = std::function<Surface*(size_t)>;

    WrappingPath() = default;

    WrappingPath(Vector3 pathStart, Vector3 pathEnd, GetSurfaceFn& GetSurface);

    size_t updPath(
        GetSurfaceFn& GetSurface,
        const WrappingArgs& args,
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
