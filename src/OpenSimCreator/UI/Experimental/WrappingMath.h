#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <functional>
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

    struct InitialConditions
    {
        Vector3 p;
        Vector3 t;
        double l;
    };

    static constexpr size_t DOF = 4;

    using Variation  = Eigen::Matrix<double, 3, DOF>;
    using Correction = Eigen::Vector<double, DOF>;

    Trihedron K_P;
    Trihedron K_Q;

    Variation v_P;
    Variation w_P;

    Variation v_Q;
    Variation w_Q;

    double length;
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

class WrappingPath;

// An abstract component class that you can calculate geodesics over.
class Surface
{
public:
    static constexpr double MIN_DIST_FROM_SURF = 1e-3;

    virtual ~Surface() = default;

protected:
    Surface()                              = default;
    Surface(Surface&&) noexcept            = default;
    Surface& operator=(Surface&&) noexcept = default;
    Surface(const Surface&)                = default;
    Surface& operator=(const Surface&)     = default;

public:
    Geodesic::Status calcGeodesic(Geodesic::InitialConditions g0);
    const Geodesic& getGeodesic() {return _geodesic;}

    void applyVariation(const Geodesic::Correction& var);

    // Returns true if touchdown was detected, with the point written to argument p.
    bool calcLocalLineToSurfaceTouchdownPoint(Vector3 a, Vector3 b, Vector3& p, size_t maxIter, double eps);
    bool isAboveSurface(Vector3 point, double bound) const {return isAboveSurfaceImpl(std::move(point), bound);}

    Vector3 getPathStartGuess() const {return _pathLocalStartGuess;}
    void setLocalPathStartGuess(Vector3 pathStartGuess) {_pathLocalStartGuess = std::move(pathStartGuess);}

    Geodesic::Status getStatus() const {return _status;}
    Geodesic::Status& updStatus() {return _status;}

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

    virtual bool isAboveSurfaceImpl(Vector3 point, double bound) const = 0;

    virtual Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const = 0;
    virtual double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const = 0;
    virtual double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const = 0;

    virtual void calcLocalGeodesicPointsImpl(std::vector<Trihedron>& pts) const = 0;

    virtual std::pair<bool, size_t> calcLocalLineToSurfaceTouchdownPointImpl(Vector3 a, Vector3 b, Vector3& p, size_t maxIter, double eps) = 0;

    // TODO this should not be surface dependent.
    // TODO weird guess (keep until fixed)
    Vector3 _pathLocalStartGuess = {1., 1., 1.};

    Geodesic _geodesic {};
    Geodesic::Status _status = Geodesic::Status::Ok;
};

class WrapObstacle
{
    private:
    WrapObstacle() = default;
    WrapObstacle(
            std::shared_ptr<Transf> transform,
            std::unique_ptr<Surface> surface) :
        _transform(std::move(transform)),
        _surface(std::move(surface)) {}

    public:
    template<typename SURFACE, typename ...Args>
    static WrapObstacle Create(std::shared_ptr<Transf> transform, Args&&...args)
    {
        return WrapObstacle(
                std::move(transform),
                std::make_unique<SURFACE>(std::forward<Args>(args)...));
    }

    /* Surface* updSurface() {return _surface.get();} */
    /* const Surface& getSurface() const {return *_surface;} */

    Vector3 getPathStartGuess() const;
    void setLocalPathStartGuess(Vector3 point) {_surface->setLocalPathStartGuess(std::move(point));}

    Geodesic::Status& updStatus() {return _surface->updStatus();}
    Geodesic::Status getStatus() const {return _surface->getStatus();}

    const Geodesic& calcGeodesic(Geodesic::InitialConditions g0);
    const Geodesic& calcGeodesicInGround();
    const Geodesic& getGeodesic() const {return _geodesic;}

    void attemptTouchdown(const Vector3& p_O, const Vector3& p_I, size_t maxIter = 20, double eps = 1e-3);
    void detectLiftOff(const Vector3& p_O, const Vector3& p_I);
    bool isAboveSurface(const Vector3& p, double bound);

    const Transf& getOffsetFrame() const {return *_transform;}

    // For convenience:
    // (Assumes the point lies on the surface)
    Vector3 calcSurfaceNormal(Vector3 point) const;
    double calcNormalCurvature(Vector3 point, Vector3 tangent) const;
    double calcGeodesicTorsion(Vector3 point, Vector3 tangent) const;

    private:

    std::shared_ptr<Transf> _transform;
    std::unique_ptr<Surface> _surface;
    Geodesic _geodesic;
    std::vector<Trihedron> _samples;

    friend WrappingPath;
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

    Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const override;
    double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const override;
    double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const override;

    std::pair<bool, size_t> calcLocalLineToSurfaceTouchdownPointImpl(Vector3 a, Vector3 b, Vector3& p, size_t maxIter, double eps) override;

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

    Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const override;
    double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const override;
    double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const override;

    std::pair<bool, size_t> calcLocalLineToSurfaceTouchdownPointImpl(Vector3 a, Vector3 b, Vector3& p, size_t maxIter, double eps) override;

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

    Vector3 calcLocalSurfaceNormalImpl(Vector3 point) const override;
    double calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const override;
    double calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const override;

    std::pair<bool, size_t> calcLocalLineToSurfaceTouchdownPointImpl(Vector3 a, Vector3 b, Vector3& p, size_t maxIter, double eps) override;

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

struct LineSeg
{
    LineSeg(const Vector3& a, const Vector3& b) : l((b-a).norm()), d((b-a)/l) {}
    double l = NAN;
    Vector3 d {NAN, NAN, NAN};
};

// Captures the smoothness of the wrapping path.
class SolverT final
{
public:
    ~SolverT()                                         = default;
    SolverT()                                          = default;
    SolverT(const SolverT&)                = default;
    SolverT(SolverT&&) noexcept            = default;
    SolverT& operator=(const SolverT&)     = default;
    SolverT& operator=(SolverT&&) noexcept = default;

    // Maximum natural geodesic correction for reducing the path error.
    double calcMaxCorrectionStep() const;

    // Pointer to first geodesic's correction.
    const Geodesic::Correction* begin() const;
    // Pointer to one past last geodesic's correction.
    const Geodesic::Correction* end() const;

    // Get access to the path error and path error jacobian.

    // Compute the geodesic corrections from the path error and path error
    // jacobian.
    bool calcPathCorrection(
            const std::vector<WrapObstacle>& obs,
            const std::vector<LineSeg>& lines);

    bool calcNormalsCorrection(
            const std::vector<WrapObstacle>& obs,
            const std::vector<LineSeg>& lines);

    void resize(size_t nSurfaces);

    CorrectionBounds maxStep;

    Eigen::VectorXd _pathCorrections;

    Eigen::VectorXd _pathError;
    Eigen::MatrixXd _pathErrorJacobian;

    Eigen::MatrixXd _mat;
    Eigen::VectorXd _vec;
    Eigen::VectorXd _vecL;

    double _length = 0.;
    Eigen::VectorXd _lengthJacobian;

    double _weight = 0.;

    friend Surface; // TODO change to whomever is calculating the path.
    friend WrappingPath;
};

// The result of computing a path over surfaces.
class WrappingPath
{
    public:
    enum Status
    {
        Ok                     = 0,
        FailedToInvertJacobian = 1,
        ExceededMaxIterations  = 2,
    };

    WrappingPath() = default;

    WrappingPath(Vector3 pathStart, Vector3 pathEnd) : _startPoint(std::move(pathStart)), _endPoint(std::move(pathEnd)) {}

    std::vector<WrapObstacle>& updSegments() {return _segments;}

    const std::vector<WrapObstacle>& getSegments() const
    {return _segments;}

    const std::vector<LineSeg>& getLineSegments() const
    {return _lineSegments;}

    const SolverT& getSolver() const
    {return _smoothness;}

    SolverT& updSolver()
    {return _smoothness;}

    size_t calcInitPath(
        double eps     = 1e-6,
        size_t maxIter = 10);

    size_t calcPath(
        bool breakOnErr = false,
        double eps     = 1e-6,
        size_t maxIter = 10);

    const Vector3& getStart() const {return _startPoint;}
    Vector3& updStart() {return _startPoint;}

    const Vector3& getEnd() const {return _endPoint;}
    Vector3& updEnd() {return _endPoint;}

    Status getStatus() const {return _status;}
    Status& updStatus() {return _status;}

    double getLength() const;
    const std::vector<Vector3>& calcPathPoints() const;
    const std::vector<Vector3>& getPathPoints() const;

    WrappingArgs& updOpts() {return _opts;}
    const WrappingArgs& getOpts() {return _opts;}

    private:
    Vector3 _startPoint{
        NAN,
        NAN,
        NAN,
    };
    Vector3 _endPoint{
        NAN,
        NAN,
        NAN,
    };

    std::vector<WrapObstacle> _segments = {};
    std::vector<LineSeg> _lineSegments = {};
    std::vector<Vector3> _pathPoints = {};
    SolverT _smoothness = {};

    Status _status = Status::Ok;
    WrappingArgs _opts;
    double _pathError = NAN;
    double _pathErrorBound = std::abs(1. - cos(1. / 180. * M_PI));
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
