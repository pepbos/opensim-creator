#include "WrappingMath.h"

#include "oscar/Utils/Assertions.h"
#include <Eigen/src/Core/util/Constants.h>
#include <cstddef>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace osc;

//==============================================================================
//                      PRINTING
//==============================================================================

namespace osc
{

std::ostream& operator<<(std::ostream& os, const Trihedron& K)
{
    os << "{";
    os << "p:" << K.p().transpose() << ", ";
    os << "t:" << K.t().transpose() << ", ";
    os << "n:" << K.n().transpose() << ", ";
    os << "b:" << K.b().transpose() << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Geodesic::Status& s)
{
    os << "Status{";
    std::string delim;
    if (s == Geodesic::Status::Ok) {
        return os << "Status{Ok}";
    }
    if (s & Geodesic::Status::InitialTangentParallelToNormal) {
        os << delim << "InitialTangentParallelToNormal";
        delim = ", ";
    }
    if (s & Geodesic::Status::PrevLineSegmentInsideSurface) {
        os << delim << "PrevLineSegmentInsideSurface";
        delim = ", ";
    }
    if (s & Geodesic::Status::NextLineSegmentInsideSurface) {
        os << delim << "NextLineSegmentInsideSurface";
        delim = ", ";
    }
    if (s & Geodesic::Status::NegativeLength) {
        os << delim << "NegativeLength";
        delim = ", ";
    }
    if (s & Geodesic::Status::LiftOff) {
        os << delim << "LiftOff";
        delim = ", ";
    }
    if (s & Geodesic::Status::TouchDownFailed) {
        os << delim << "TouchDownFailed";
        delim = ", ";
    }
    if (s & Geodesic::Status::IntegratorFailed) {
        os << delim << "IntegratorFailed";
        delim = ", ";
    }
    if (s & Geodesic::Status::Disabled) {
        os << delim << "Disabled";
        delim = ", ";
    }
    return os << "}";
}

std::ostream& operator<<(std::ostream& os, const Geodesic& x)
{
    os << "Geodesic{\n";
    os << "    K_P: " << x.K_P << ",\n";
    os << "    K_Q: " << x.K_Q << ",\n";
    os << "    l: " << x.length << ",\n";
    return os << "}";
}

std::ostream& operator<<(std::ostream& os, const WrappingPath::Status& s)
{
    os << "Status{";
    std::string delim;
    if (s == WrappingPath::Status::Ok) {
        return os << "Status{Ok}";
    }
    if (s & WrappingPath::Status::FailedToInvertJacobian) {
        os << delim << "FailedToInvertJacobian";
        delim = ", ";
    }
    if (s & WrappingPath::Status::ExceededMaxIterations) {
        os << delim << "ExceededMaxIterations";
        delim = ", ";
    }
    return os << "}";
}

} // namespace osc

//==============================================================================
//                      ASSERTION HELPERS
//==============================================================================
namespace
{

void AssertEq(
    double lhs,
    double rhs,
    const std::string& msg,
    double eps = 1e-10)
{
    const bool isOk = std::abs(lhs - rhs) < eps;
    if (!isOk) {
        std::ostringstream os;
        os << "FAILED ASSERT: " << msg << std::endl;
        os << "    lhs = " << lhs << std::endl;
        os << "    rhs = " << rhs << std::endl;
        os << "    err = " << lhs - rhs << std::endl;
        os << "    bnd = " << eps << std::endl;
        std::string msg = os.str();
        throw std::runtime_error(msg.c_str());
        OSC_ASSERT(isOk && msg.c_str());
    }
}

void AssertEq(
    const Vector3& lhs,
    const Vector3& rhs,
    const std::string& msg,
    double eps = 1e-10)
{
    const bool isOk = (lhs - rhs).norm() < eps;
    if (!isOk) {
        std::ostringstream os;
        os << "FAILED ASSERT: " << msg << std::endl;
        os << "    lhs = " << lhs.transpose() << std::endl;
        os << "    rhs = " << rhs.transpose() << std::endl;
        os << "    err = " << (lhs - rhs).transpose() << std::endl;
        os << "    bnd = " << eps << std::endl;
        /* throw std::runtime_error(msg); */
        std::string msg = os.str();
        throw std::runtime_error(msg.c_str());
        OSC_ASSERT(isOk && msg.c_str());
    }
}

} // namespace

//==============================================================================
//                      DARBOUX FRAME
//==============================================================================

void AssertDarbouxFrame(const Darboux& frame)
{
    const Vector3 t = frame.t();
    const Vector3 n = frame.n();
    const Vector3 b = frame.b();

    AssertEq(t.dot(t), 1., "t norm = 1.");
    AssertEq(n.dot(n), 1., "n norm = 1.");
    AssertEq(b.dot(b), 1., "b norm = 1.");

    AssertEq(t.dot(n), 0., "t.dot(n) = 0.");
    AssertEq(t.dot(b), 0., "t.dot(b) = 0.");
    AssertEq(n.dot(b), 0., "n.dot(b) = 0.");

    AssertEq(t.cross(n), b, "t.cross(n) = b");
    AssertEq(b.cross(t), n, "b.cross(t) = n");
    AssertEq(n.cross(b), t, "n.cross(b) = t");
}

Darboux::Darboux()
{
    _rotation.fill(NAN);
}

Darboux Darboux::FromTangenGuessAndNormal(Vector3 tangentGuess, Vector3 normal)
{
    Vector3 n = std::move(normal);
    n         = n / n.norm();

    Vector3 t = std::move(tangentGuess);
    t         = t - n * n.dot(t);
    t         = t / t.norm();

    Vector3 b = t.cross(n);
    b         = b / b.norm();

    return Darboux(t, n, b);
}

Darboux::Darboux(Vector3 tangent, Vector3 normal, Vector3 binormal)
{
    _rotation.col(0) = tangent;
    _rotation.col(1) = normal;
    _rotation.col(2) = binormal;

    AssertDarbouxFrame(*this);
}

Trihedron::Trihedron(Vector3 point, Darboux rotation) :
    _point(std::move(point)), _rotation(std::move(rotation))
{}

Trihedron::Trihedron(
    Vector3 point,
    Vector3 tangent,
    Vector3 normal,
    Vector3 binormal) :
    Trihedron(
        std::move(point),
        Darboux(std::move(tangent), std::move(normal), std::move(binormal)))
{}

Trihedron Trihedron::FromPointAndTangentGuessAndNormal(
    Vector3 point,
    Vector3 tangentGuess,
    Vector3 normal)
{
    return {
        std::move(point),
        Darboux::FromTangenGuessAndNormal(
            std::move(tangentGuess),
            std::move(normal))};
}

namespace osc
{

Darboux operator*(Eigen::Quaterniond lhs, const Darboux& rhs)
{
    Eigen::Matrix<double, 3, 3> mat = lhs.toRotationMatrix();
    const Vector3 t                 = mat * rhs.t();
    const Vector3 n                 = mat * rhs.n();
    return {t, n, t.cross(n)};
}

Trihedron operator*(Eigen::Quaterniond lhs, const Trihedron& rhs)
{
    Eigen::Matrix<double, 3, 3> mat = lhs.toRotationMatrix();
    const Vector3 t                 = mat * rhs.t();
    const Vector3 n                 = mat * rhs.n();
    return {mat * rhs.p(), t, n, t.cross(n)};
}

} // namespace osc

//==============================================================================
//                      TRANSFORM
//==============================================================================

// TODO add rotation transform

Vector3 calcPointInLocal(const Transf& transform, Vector3 pointInGround)
{
    return pointInGround - transform.position;
}

Vector3 calcPointInGround(const Transf& transform, Vector3 pointInLocal)
{
    return pointInLocal + transform.position;
}

Vector3 calcVectorInLocal(const Transf&, Vector3 vecInGround)
{
    return vecInGround;
}

Vector3 calcVectorInGround(const Transf&, Vector3 vecInLocal)
{
    return vecInLocal;
}

void calcInGround(const Transf& transform, Trihedron& K)
{
    Vector3& p = K.updPoint();
    p          = calcPointInGround(transform, p);

    // TODO Rotation transform here.
}

void calcInGround(const Transf& transform, Geodesic& g)
{
    // TODO rotation transform here.
    calcInGround(transform, g.K_P);
    calcInGround(transform, g.K_Q);
}

void calcInLocal(const Transf& transform, Geodesic::InitialConditions& g0)
{
    g0.p = calcPointInLocal(transform, g0.p);
    g0.t = calcVectorInLocal(transform, g0.t);
}

//==============================================================================
//                      SOME MATHS
//==============================================================================

double calcPointOnLineNearOriginAsFactor(
    Eigen::Vector<double, 2> a,
    Eigen::Vector<double, 2> b)
{
    using Vector2 = Eigen::Vector<double, 2>;

    const Vector2 e  = b - a;
    const double eTe = e.dot(e);

    Vector2 d      = -a.dot(e) * e / eTe;
    const double c = e.dot(d) / eTe;
    return std::max(0., std::min(c, 1.));
}

Vector3 calcPointOnLineNearOrigin(Vector3 a, Vector3 b)
{
    const Vector3 e = b - a;

    Vector3 p = a - a.dot(e) * e / e.dot(e);

    const double d0 = a.dot(a);
    const double d1 = b.dot(b);
    const double d  = p.dot(p);

    if (d0 < d) {
        return a;
    }
    if (d1 < d) {
        return b;
    }
    return p;
};

Vector3 calcPointOnLineNearPoint(Vector3 a, Vector3 b, Vector3 point)
{
    return calcPointOnLineNearOrigin(a - point, b - point) + point;
};

double sign(double x)
{
    return static_cast<double>(x > 0.) - static_cast<double>(x < 0.);
}

template <typename VECTOR>
double calcInfNorm(const typename std::remove_reference<VECTOR>::type& vec)
{
    double infNorm = 0.;
    const auto n   = static_cast<size_t>(vec.rows());
    for (size_t i = 0; i < n; ++i) {
        infNorm = std::max(infNorm, std::abs(vec[i]));
    }
    return infNorm;
}

template <typename VECTOR>
double calcScaledToFit(
    typename std::remove_reference<VECTOR>::type& vec,
    double bound)
{
    const double c = std::abs(bound) / calcInfNorm<VECTOR>(vec);
    if (c < 1.) {
        vec *= c;
    }
    return std::min(c, 1.);
}

template <>
double calcInfNorm<ImplicitGeodesicState>(const ImplicitGeodesicState& y)
{
    double infNorm = 0.;

    infNorm = std::max(infNorm, y.position[0]);
    infNorm = std::max(infNorm, y.position[1]);
    infNorm = std::max(infNorm, y.position[2]);

    infNorm = std::max(infNorm, y.velocity[0]);
    infNorm = std::max(infNorm, y.velocity[1]);
    infNorm = std::max(infNorm, y.velocity[2]);

    infNorm = std::max(infNorm, y.a);
    infNorm = std::max(infNorm, y.aDot);

    infNorm = std::max(infNorm, y.r);
    infNorm = std::max(infNorm, y.rDot);

    return infNorm;
}

//==============================================================================
//                      CURVATURES
//==============================================================================

/// Calculates adjoint of matrix.
/// Assumes matrix is symmetric.
Mat3x3 calcAdjoint(const Mat3x3& mat)
{
    double fxx = mat(0, 0);
    double fyy = mat(1, 1);
    double fzz = mat(2, 2);

    double fxy = mat(0, 1);
    double fxz = mat(0, 2);
    double fyz = mat(1, 2);

    std::array<double, 9> elements = {
        fyy * fzz - fyz * fyz,
        fyz * fxz - fxy * fzz,
        fxy * fyz - fyy * fxz,
        fxz * fyz - fxy * fzz,
        fxx * fzz - fxz * fxz,
        fxy * fxz - fxx * fyz,
        fxy * fyz - fxz * fyy,
        fxy * fxz - fxx * fyz,
        fxx * fyy - fxy * fxy};
    Mat3x3 adj;
    size_t i = 0;
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            adj(r, c) = elements[i];
            ++i;
        }
    }
    return adj;
}

double calcNormalCurvature(
    const ImplicitSurface& s,
    Vector3 point,
    Vector3 tangent)
{
    const Vector3& p  = point;
    const Vector3& v  = tangent;
    const Vector3 g   = s.calcSurfaceConstraintGradient(p);
    const Vector3 h_v = s.calcSurfaceConstraintHessian(p) * v;
    // Sign flipped compared to thesis: kn = negative, see eq 3.63
    return -v.dot(h_v) / g.norm();
}

double calcGeodesicTorsion(
    const ImplicitSurface& s,
    Vector3 point,
    Vector3 tangent)
{
    // TODO verify this!
    const Vector3& p  = point;
    const Vector3& v  = tangent;
    const Vector3 g   = s.calcSurfaceConstraintGradient(p);
    const Vector3 h_v = s.calcSurfaceConstraintHessian(p) * v;
    const Vector3 gxv = g.cross(v);
    return -h_v.dot(gxv) / g.dot(g);
}

// TODO use normalized vector.
Vector3 calcSurfaceNormal(const ImplicitSurface& s, Vector3 point)
{
    const Vector3& p       = point;
    const Vector3 gradient = s.calcSurfaceConstraintGradient(p);
    return gradient / gradient.norm();
}

Vector3 calcAcceleration(
    const ImplicitSurface& s,
    Vector3 point,
    Vector3 tangent)
{
    // TODO Writing it out saves a root, but optimizers are smart.
    // Sign flipped compared to thesis: kn = negative, see eq 3.63
    return calcNormalCurvature(s, point, std::move(tangent)) *
           calcSurfaceNormal(s, point);
}

double calcGaussianCurvature(
    const ImplicitSurface& s,
    const ImplicitGeodesicState& q)
{
    const Vector3& p = q.position;
    Vector3 g        = s.calcSurfaceConstraintGradient(p);
    double gDotg     = g.dot(g);
    Mat3x3 adj       = calcAdjoint(s.calcSurfaceConstraintHessian(p));

    if (gDotg * gDotg < 1e-13) {
        throw std::runtime_error(
            "Gaussian curvature inaccurate: are we normal to surface?");
    }

    return (g.dot(adj * g)) / (gDotg * gDotg);
}

Trihedron calcTrihedron(
    const ImplicitSurface& s,
    const Vector3& p,
    const Vector3& t)
{
    return Trihedron::FromPointAndTangentGuessAndNormal(
        p,
        t,
        calcSurfaceNormal(s, p));
}

Trihedron calcTrihedron(
    const ImplicitSurface& s,
    const ImplicitGeodesicState& q)
{
    return Trihedron::FromPointAndTangentGuessAndNormal(
        q.position,
        q.velocity,
        calcSurfaceNormal(s, q.position));
}

//==============================================================================
//                      SURFACE PROJECTION
//==============================================================================

size_t calcFastPointProjectedToSurface(
    const ImplicitSurface& s,
    Vector3& position,
    double eps     = 1e-13,
    size_t maxIter = 10)
{
    Vector3 pk = position;
    for (size_t iteration = 0; iteration < maxIter; ++iteration) {
        const double c = s.calcSurfaceConstraint(pk);

        const double error = std::abs(c);

        if (error < eps) {
            position = pk;
            return iteration;
        }

        const Vector3 g = s.calcSurfaceConstraintGradient(pk);

        pk += -g * c / g.dot(g);
    }
    return maxIter;
}

size_t calcFastSurfaceProjection(
    const ImplicitSurface& s,
    Vector3& p,
    Vector3& v,
    double eps     = 1e-13,
    size_t maxIter = 100)
{
    size_t steps = calcFastPointProjectedToSurface(s, p, eps, maxIter);

    Vector3 n = s.calcSurfaceConstraintGradient(p);
    if (!(n.cross(v).norm() > 1e-13)) {
        throw std::runtime_error("Velocity and normal are perpendicular!");
    }

    v = v - n.dot(v) * n / n.dot(n);
    v = v / v.norm();

    return steps;
}

//==============================================================================
//                         IMPLICIT SURFACE STATE
//==============================================================================

namespace osc
{

ImplicitGeodesicStateDerivative calcImplicitGeodesicStateDerivative(
    const ImplicitGeodesicState& y,
    const Vector3& acceleration,
    double gaussianCurvature)
{
    ImplicitGeodesicStateDerivative dy;
    dy.velocity     = y.velocity;
    dy.acceleration = acceleration;
    dy.aDot         = y.aDot;
    dy.aDDot        = -y.a * gaussianCurvature;
    dy.rDot         = y.rDot;
    dy.rDDot        = -y.r * gaussianCurvature;
    return dy;
}

ImplicitGeodesicState operator*(
    double dt,
    const ImplicitGeodesicStateDerivative& dy)
{
    ImplicitGeodesicState y;
    y.position = dt * dy.velocity;
    y.velocity = dt * dy.acceleration;
    y.a        = dt * dy.aDot;
    y.aDot     = dt * dy.aDDot;
    y.r        = dt * dy.rDot;
    y.rDot     = dt * dy.rDDot;
    return y;
}

ImplicitGeodesicState operator+(
    const ImplicitGeodesicState& lhs,
    const ImplicitGeodesicState& rhs)
{
    ImplicitGeodesicState y;
    y.position = lhs.position + rhs.position;
    y.velocity = lhs.velocity + rhs.velocity;
    y.a        = lhs.a + rhs.a;
    y.aDot     = lhs.aDot + rhs.aDot;
    y.r        = lhs.r + rhs.r;
    y.rDot     = lhs.rDot + rhs.rDot;
    return y;
}

ImplicitGeodesicState operator-(
    const ImplicitGeodesicState& lhs,
    const ImplicitGeodesicState& rhs)
{
    ImplicitGeodesicState y;
    y.position = lhs.position - rhs.position;
    y.velocity = lhs.velocity - rhs.velocity;
    y.a        = lhs.a - rhs.a;
    y.aDot     = lhs.aDot - rhs.aDot;
    y.r        = lhs.r - rhs.r;
    y.rDot     = lhs.rDot - rhs.rDot;
    return y;
}
} // namespace osc

//==============================================================================
//                      IMPLICIT SURFACE HELPERS
//==============================================================================

// Compute generic geodesic state from implicit geodesic state.
void calcGeodesicBoundaryState(
    const ImplicitSurface& s,
    const ImplicitGeodesicState& q,
    bool isEnd,
    Trihedron& K,
    Geodesic::Variation& v,
    Geodesic::Variation& w)
{
    K = calcTrihedron(s, q);

    const Vector3 t = K.t();
    const Vector3 b = K.b();

    v.col(0) = t;
    v.col(1) = b * q.a;
    v.col(2) = b * q.r;
    v.col(3) = isEnd ? t : Vector3{0., 0., 0.};

    const double tau_g   = calcGeodesicTorsion(s, q.position, q.velocity);
    const double kappa_n = calcNormalCurvature(s, q.position, q.velocity);
    const double kappa_a = calcNormalCurvature(s, q.position, b);

    w.col(0) = Vector3{tau_g, 0., kappa_n};
    w.col(1) = Vector3{
        -q.a * kappa_a,
        -q.aDot,
        -q.a * tau_g,
    };
    w.col(2) = Vector3{-q.r * kappa_a, -q.rDot, -q.r * tau_g};
    w.col(3) = isEnd ? Vector3{tau_g, 0., kappa_n} : Vector3{0., 0., 0.};

    // Variation is in local frame.
    w = K.R() * w;
}

//==============================================================================
//                      PATH ERROR GRADIENT
//==============================================================================

using GeodesicJacobian = Eigen::Vector<double, Geodesic::DOF>;

GeodesicJacobian calcDirectionJacobian(
    const LineSeg& e,
    Vector3 axis,
    const Geodesic::Variation& v)
{
    Vector3 y = axis - e.d * e.d.dot(axis);
    y /= e.l;
    return y.transpose() * v;
}

GeodesicJacobian calcPathErrorJacobian(
    const LineSeg& line,
    Vector3 axis,
    const Geodesic::Variation& v,
    const Geodesic::Variation& w,
    bool invertV = false)
{
    GeodesicJacobian jacobian =
        calcDirectionJacobian(line, axis, v) * (invertV ? -1. : 1.);
    jacobian += axis.cross(line.d).transpose() * w;
    return jacobian;
}

//==============================================================================
//                      ACCURATE SURFACE PROJECTION
//==============================================================================

/* double calcMaxAlignmentError(double angleDeg) */
/* { */
/*     return std::abs(1. - cos(angleDeg / 180. * M_PI)); */
/* } */

// Find the touchdown point of a line to a surface.
size_t calcLineToImplicitSurfaceTouchdownPoint(
    const ImplicitSurface& s,
    Vector3 a,
    Vector3 b,
    Vector3& p,
    size_t maxIter,
    double eps)
{
    // Initial guess.
    double alpha = 0.;
    size_t iter  = 0;

    for (; iter < maxIter; ++iter) {
        // Touchdown point on line.
        const Vector3 d  = b - a;
        const Vector3 pl = a + (b - a) * alpha;

        // Constraint evaluation at touchdown point.
        const double c = s.calcSurfaceConstraint(pl);

        // Assumes a negative constraint evaluation means touchdown.
        if (std::abs(c) < eps)
            break;

        // Gradient at point on line.
        const Vector3 g                  = s.calcSurfaceConstraintGradient(pl);
        const ImplicitSurface::Hessian H = s.calcSurfaceConstraintHessian(pl);

        // Add a weight to the newton step to avoid large steps.
        constexpr double w = 0.5;

        // Update alpha.
        const double step = g.dot(d) / (d.dot(H * d) + w);

        // Stop when converged.
        if (std::abs(step) < eps)
            break;

        // Clamp the stepsize.
        constexpr double maxStep = 0.25;
        alpha -= std::min(std::max(-maxStep, step), maxStep);

        // Stop when leaving bounds.
        if (alpha < 0. || alpha > 1.)
            break;
    }

    alpha = std::max(std::min(1., alpha), 0.);
    p     = a + (b - a) * alpha;
    return iter;
}

std::pair<bool, size_t> ImplicitSurface::
    calcLocalLineToSurfaceTouchdownPointImpl(
        Vector3 a,
        Vector3 b,
        Vector3& p,
        size_t maxIter,
        double eps)
{
    const size_t iter =
        calcLineToImplicitSurfaceTouchdownPoint(*this, a, b, p, maxIter, eps);
    return {calcSurfaceConstraint(p) < 0., iter};
}

//==============================================================================
//                      IMPLICIT GEODESIC INTEGRATOR
//==============================================================================

namespace osc
{

template <typename Y, typename DY, typename S>
double RungeKuttaMerson<Y, DY, S>::step(
    double h,
    std::function<DY(const Y&)>& f)
{
    Y& yk = _y.at(1);

    // k1
    _k.at(0) = f(_y.at(0));

    // k2
    {
        yk       = _y.at(0) + (h / 3.) * _k.at(0);
        _k.at(1) = f(yk);
    }

    // k3
    {
        yk       = _y.at(0) + (h / 6.) * _k.at(0) + (h / 6.) * _k.at(1);
        _k.at(2) = f(yk);
    }

    // k4
    {
        yk = _y.at(0) + (1. / 8. * h) * _k.at(0) + (3. / 8. * h) * _k.at(2);
        _k.at(3) = f(yk);
    }

    // k5
    {
        yk = _y.at(0) + (1. / 2. * h) * _k.at(0) + (-3. / 2. * h) * _k.at(2) +
             (2. * h) * _k.at(3);
        _k.at(4) = f(yk);
    }

    // y1: Auxiliary --> Already updated in k5 computation.

    // y2: Final state.
    _y.at(2) = _y.at(0) + (1. / 6. * h) * _k.at(0) + (2. / 3. * h) * _k.at(3) +
               (1. / 6. * h) * _k.at(4);

    return calcInfNorm<Y>(_y.at(1) - _y.at(2)) * 0.2;
}

template <typename Y, typename DY, typename S>
double RungeKuttaMerson<Y, DY, S>::stepTo(
    Y y0,
    double x1,
    std::function<DY(const Y&)>& f,
    std::function<void(Y&)>& g)
{
    _y.at(0) = std::move(y0);
    double h = _h0;
    double x = 0.;
    double e = 0.;

    _failedCount = 0;

    _samples.clear();
    _samples.push_back(Sample(x, _y.at(0)));

    while (x < x1 - 1e-13) {
        const bool init = x == 0.;

        h = x + h > x1 ? x1 - x : h;

        // Attempt step.
        double err = step(h, f);

        // Reject if accuracy was not met.
        if (err > _accuracy) { // Rejected
            // Descrease stepsize.
            h /= 2.;
            _y.at(1) = _y.at(0);
            _y.at(2) = _y.at(0);
            ++_failedCount;
        } else {         // Accepted
            g(_y.at(2)); // Enforce constraints.
            _y.at(0) = _y.at(2);
            _y.at(1) = _y.at(2);
            x += h;
            _samples.push_back(Sample(x, _y.at(0)));
        }

        // Potentially increase stepsize.
        if (err < _accuracy / 64.) {
            h *= 2.;
        }

        e = std::max(e, err);
        h = std::min(_hMax, std::max(_hMin, h));

        if (init) {
            _h0 = h;
        }
    }

    return e;
}

void RunIntegratorTests()
{

    const Vector3 w{1., 2., 3.};
    std::function<Vector3(const Vector3&)> f =
        [&](const Vector3& yk) -> Vector3 { return w.cross(yk); };

    std::function<void(Vector3&)> g = [&](Vector3& yk) { yk.normalize(); };

    // Fixed step integrator test.
    {
        RungeKuttaMerson<Vector3, Vector3, Vector3> rkm(1e-3, 1e-3, 1e-6);

        const Vector3 y0{1., 0., 0.};
        const double x = 1.;

        const double e   = rkm.stepTo(y0, x, f, g);
        const Vector3 y1 = rkm.getSamples().back().y;

        // Check the result
        Vector3 axis = w / w.norm();
        double angle = w.norm() * x;
        Eigen::Quaterniond q(Eigen::AngleAxisd(angle, axis));

        const Vector3 y1_expected = q * y0;
        const double e_real       = (y1 - y1_expected).norm();

        if (e_real > 1e-10) {
            std::cout << "y0     = " << y0 << std::endl;
            std::cout << "y1     = " << y1.transpose() << std::endl;
            std::cout << "y1_exp = " << y1_expected.transpose() << std::endl;
            std::cout << "e      = " << e << std::endl;
            std::cout << "e_exp  = " << e_real << std::endl;
            throw std::runtime_error(
                "Failed RungeKuttaMerson fixed step integrator test");
        }
    }

    // Variable step integrator test.
    {
        const double accuracy = 1e-6;
        RungeKuttaMerson<Vector3, Vector3, Vector3> rkm(1e-5, 1e-1, accuracy);

        const Vector3 y0{1., 0., 0.};
        const double x = 1.;

        double e = NAN;
        Vector3 y1;

        for (size_t i = 0; i < 20; ++i) {
            e  = rkm.stepTo(y0, x, f, g);
            y1 = rkm.getSamples().back().y;
        }

        // Check the result
        Vector3 axis = w / w.norm();
        double angle = w.norm() * x;
        Eigen::Quaterniond q(Eigen::AngleAxisd(angle, axis));

        const Vector3 y1_expected = q * y0;
        const double e_real       = (y1 - y1_expected).norm();

        bool failed = false;
        failed |= rkm.getSamples().size() != 26;
        failed |= e_real > 10. * accuracy;
        if (failed) {
            std::cout << "n      = " << rkm.getSamples().size() << std::endl;
            std::cout << "failed = " << rkm.getNumberOfFailedSteps()
                      << std::endl;
            std::cout << "y0     = " << y0 << std::endl;
            std::cout << "y1     = " << y1.transpose() << std::endl;
            std::cout << "y1_exp = " << y1_expected.transpose() << std::endl;
            std::cout << "e      = " << e << std::endl;
            std::cout << "e_exp  = " << e_real << std::endl;
            throw std::runtime_error(
                "Failed RungeKuttaMerson variable step integrator test");
        }
    }
}

} // namespace osc

//==============================================================================
//                      GEODESIC STATUS FLAGS
//==============================================================================

namespace
{
bool isActive(Geodesic::Status s)
{
    return s == Geodesic::Status::Ok || s == Geodesic::Status::NegativeLength;
}

bool isError(Geodesic::Status s)
{
    return !isActive(s) || s == Geodesic::Status::Disabled;
}
} // namespace

//==============================================================================
//                      SURFACE
//==============================================================================

Geodesic::Status Surface::calcGeodesic(Geodesic::InitialConditions g0)
{
    _status = Geodesic::Status::Ok;
    if (g0.l < 0.) {
        _status |= Geodesic::Status::NegativeLength;
        g0.l = 0.;
    }
    try {
        calcLocalGeodesicImpl(g0.p, g0.t, g0.l, _geodesic);
    } catch (const std::exception& e) {
        _status |= Geodesic::Status::IntegratorFailed;
    }
    return _status;
}

Geodesic::InitialConditions applyNaturalGeodesicVariation(
    const Geodesic& g,
    const Geodesic::Correction& correction)
{
    Vector3 v = g.v_P * correction;
    Vector3 w = g.w_P * correction;

    Vector3 t = g.K_P.t() + w.cross(g.K_P.t());
    Vector3 p = g.K_P.p() + v;
    double l  = g.length + correction[Geodesic::DOF - 1];

    return {p, t, l};
}

void Surface::applyVariation(const Geodesic::Correction& c)
{
    calcGeodesic(applyNaturalGeodesicVariation(_geodesic, c));
}

bool Surface::calcLocalLineToSurfaceTouchdownPoint(
    Vector3 a,
    Vector3 b,
    Vector3& p,
    size_t maxIter,
    double eps)
{
    std::pair<bool, size_t> y =
        calcLocalLineToSurfaceTouchdownPointImpl(a, b, p, maxIter, eps);
    const bool touchdown = y.first;
    const size_t iter    = y.second;
    if (iter >= maxIter) {
        updStatus() |= Geodesic::Status::TouchDownFailed;
    }
    return touchdown;
}

Vector3 Surface::calcSurfaceNormal(Vector3 point) const
{
    return calcLocalSurfaceNormalImpl(std::move(point));
}

double Surface::calcNormalCurvature(Vector3 point, Vector3 tangent) const
{
    return calcLocalNormalCurvatureImpl(std::move(point), std::move(tangent));
}

double Surface::calcGeodesicTorsion(Vector3 point, Vector3 tangent) const
{
    return calcLocalGeodesicTorsionImpl(std::move(point), std::move(tangent));
}

//==============================================================================
//                      IMPLICIT SURFACE
//==============================================================================

double ImplicitSurface::calcSurfaceConstraint(Vector3 position) const
{
    return calcSurfaceConstraintImpl(position);
}

Vector3 ImplicitSurface::calcSurfaceConstraintGradient(Vector3 position) const
{
    return calcSurfaceConstraintGradientImpl(position);
}

ImplicitSurface::Hessian ImplicitSurface::calcSurfaceConstraintHessian(
    Vector3 position) const
{
    return calcSurfaceConstraintHessianImpl(position);
}

void ImplicitSurface::calcLocalGeodesicImpl(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length,
    Geodesic& geodesic)
{

    std::function<ImplicitGeodesicStateDerivative(const ImplicitGeodesicState&)>
        f = [&](const ImplicitGeodesicState& q)
        -> ImplicitGeodesicStateDerivative {
        return calcImplicitGeodesicStateDerivative(
            q,
            calcAcceleration(*this, q.position, q.velocity),
            calcGaussianCurvature(*this, q));
    };
    std::function<void(ImplicitGeodesicState&)> g =
        [&](ImplicitGeodesicState& q) {
            // Position reprojection.
            Vector3& p      = q.position;
            const double c  = calcSurfaceConstraint(p);
            const Vector3 g = calcSurfaceConstraintGradient(p);
            p += -g * c / g.dot(g);
            // Tangent reprojection.
            Vector3 n  = calcSurfaceConstraintGradient(p);
            Vector3& v = q.velocity;
            v          = v - n.dot(v) * n / n.dot(n);
            v          = v / v.norm();
        };

    calcFastSurfaceProjection(*this, initPosition, initVelocity);

    const ImplicitGeodesicState q0{initPosition, initVelocity};
    _rkm.stepTo(q0, length, f, g);

    if (_rkm.getSamples().empty()) {
        throw std::runtime_error("failed to shoot geodesic");
    }

    calcGeodesicBoundaryState(
        *this,
        _rkm.getSamples().front().y,
        false,
        geodesic.K_P,
        geodesic.v_P,
        geodesic.w_P);
    calcGeodesicBoundaryState(
        *this,
        _rkm.getSamples().back().y,
        true,
        geodesic.K_Q,
        geodesic.v_Q,
        geodesic.w_Q);
    geodesic.length = length;
}

Vector3 ImplicitSurface::calcLocalSurfaceNormalImpl(Vector3 point) const
{
    return ::calcSurfaceNormal(*this, point);
}

double ImplicitSurface::calcLocalGeodesicTorsionImpl(
    Vector3 point,
    Vector3 tangent) const
{
    return ::calcGeodesicTorsion(*this, std::move(point), std::move(tangent));
}

double ImplicitSurface::calcLocalNormalCurvatureImpl(
    Vector3 point,
    Vector3 tangent) const
{
    return ::calcNormalCurvature(*this, std::move(point), std::move(tangent));
}

void ImplicitSurface::calcPathPointsImpl(
    std::vector<Vector3>& points,
    Transf transform) const
{
    for (const auto& q : _rkm.getSamples()) {
        points.push_back(calcPointInGround(transform, q.y.position));
    }
}

//==============================================================================
//                      IMPLICIT ELLIPSOID SURFACE
//==============================================================================

double ImplicitEllipsoidSurface::calcSurfaceConstraintImpl(Vector3 p) const
{
    p.x() /= _xRadius;
    p.y() /= _yRadius;
    p.z() /= _zRadius;
    return p.dot(p) - 1.;
}

Vector3 ImplicitEllipsoidSurface::calcSurfaceConstraintGradientImpl(
    Vector3 p) const
{
    p.x() /= _xRadius * _xRadius;
    p.y() /= _yRadius * _yRadius;
    p.z() /= _zRadius * _zRadius;
    return 2. * p;
}

Mat3x3 ImplicitEllipsoidSurface::calcSurfaceConstraintHessianImpl(Vector3) const
{
    static constexpr size_t n = 3;

    Mat3x3 hessian;
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < n; ++c) {
            hessian(r, c) = r == c ? 2. : 0.;
        }
    }

    hessian(0, 0) /= _xRadius * _xRadius;
    hessian(1, 1) /= _yRadius * _yRadius;
    hessian(2, 2) /= _zRadius * _zRadius;

    return hessian;
}

bool ImplicitEllipsoidSurface::isAboveSurfaceImpl(Vector3 point, double bound)
    const
{
    point.x() /= (_xRadius + bound);
    point.y() /= (_yRadius + bound);
    point.z() /= (_zRadius + bound);
    return point.dot(point) - 1. > 0.;
}

//==============================================================================
//                      IMPLICIT SPHERE SURFACE
//==============================================================================

double ImplicitSphereSurface::calcSurfaceConstraintImpl(Vector3 p) const
{
    return p.dot(p) - _radius * _radius;
}

Vector3 ImplicitSphereSurface::calcSurfaceConstraintGradientImpl(
    Vector3 p) const
{
    return 2. * p;
}

Mat3x3 ImplicitSphereSurface::calcSurfaceConstraintHessianImpl(Vector3) const
{
    Mat3x3 hessian;
    for (size_t r = 0; r < hessian.rows(); ++r) {
        for (size_t c = 0; c < hessian.cols(); ++c) {
            hessian(r, c) = r == c ? 2. : 0.;
        }
    }
    return hessian;
}

bool ImplicitSphereSurface::isAboveSurfaceImpl(Vector3 point, double bound)
    const
{
    return point.dot(point) - (_radius + bound) * (_radius + bound) > 0.;
}

//==============================================================================
//                      ANALYTIC SPHERE SURFACE
//==============================================================================

std::pair<bool, size_t> AnalyticSphereSurface::
    calcLocalLineToSurfaceTouchdownPointImpl(
        Vector3 a,
        Vector3 b,
        Vector3& p,
        size_t,
        double)
{
    Vector3 pLine        = calcPointOnLineNearOrigin(a, b);
    const bool touchdown = pLine.dot(pLine) < _radius * _radius;
    if (touchdown) {
        p = pLine;
    }
    return {touchdown, 0};
}

void AnalyticSphereSurface::calcLocalGeodesicImpl(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length,
    Geodesic& geodesic)
{
    const double r     = _radius;
    const double angle = length / r;

    const double norm = initPosition.norm();
    if (norm < 1e-13) {
        throw std::runtime_error("Error: initial position at origin.");
    }

    Trihedron& K_P           = geodesic.K_P;
    Geodesic::Variation& v_P = geodesic.v_P;
    Geodesic::Variation& w_P = geodesic.w_P;

    K_P = Trihedron::FromPointAndTangentGuessAndNormal(
        initPosition / norm * r,
        initVelocity,
        initPosition);

    w_P.col(0) = -K_P.b() / r;
    w_P.col(1) = K_P.t() / r;
    w_P.col(2) = -K_P.n();
    w_P.col(3) = Vector3{0., 0., 0.};

    // Since position = normal * radius -> p = n * r
    // We have dp = dn * r
    // With: dn = w x n
    v_P.col(0) = K_P.t();
    v_P.col(1) = K_P.b();
    v_P.col(2) = Vector3{0., 0., 0.};
    v_P.col(3) = Vector3{0., 0., 0.};

    // Integrate to final trihedron: K_Q
    Trihedron& K_Q           = geodesic.K_Q;
    Geodesic::Variation& v_Q = geodesic.v_Q;
    Geodesic::Variation& w_Q = geodesic.w_Q;

    // Integration is a rotation over the axis by the angle.
    const Vector3 axis = -K_P.b(); // axis is negative of binormal
    const Eigen::Quaterniond dq{Eigen::AngleAxisd(angle, axis)};

    // Final frame: Rotate the input of the initial frame.
    K_Q = dq * K_P;

    // For a sphere the rotation of the initial frame directly rotates the final
    // frame:
    w_Q        = w_P;
    w_Q.col(3) = w_Q.col(0);

    // End frame position variation: dp = w x n * r
    for (size_t i = 0; i < 4; ++i) {
        v_Q.col(i) = w_Q.col(i).cross(K_Q.n()) * r;
    }

    geodesic.length = length;
}

bool AnalyticSphereSurface::isAboveSurfaceImpl(Vector3 point, double bound)
    const
{
    return point.dot(point) - (_radius + bound) * (_radius + bound) > 0.;
}

double AnalyticSphereSurface::calcLocalNormalCurvatureImpl(Vector3, Vector3)
    const
{
    return -1. / _radius;
}

Vector3 AnalyticSphereSurface::calcLocalSurfaceNormalImpl(Vector3 point) const
{
    return point / point.norm();
}

double AnalyticSphereSurface::calcLocalGeodesicTorsionImpl(Vector3, Vector3)
    const
{
    return 0.;
}

void AnalyticSphereSurface::calcPathPointsImpl(
    std::vector<Vector3>& points,
    Transf transform) const
{
    const Geodesic& g = getGeodesic();

    // Push the start point.
    Vector3 p = g.K_P.p();
    points.push_back(calcPointInGround(transform, p));

    const double angle = g.length / _radius;
    size_t n           = static_cast<size_t>(std::abs(angle / _sampleSpacing));
    n                  = std::min(_maxNrOfSamples, n);

    if (n > 0) {
        const Vector3 axis = -g.K_P.b();
        const double c     = 1. / static_cast<double>(n + 1);
        const Rotation dq{Eigen::AngleAxisd(angle * c, axis)};
        for (size_t i = 0; i < n; ++i) {
            p = dq * p;
            // TODO can be made more efficient by computing dq = R * dq * RT
            points.push_back(calcPointInGround(transform, p));
        }
    }

    // Push the end point.
    points.push_back(calcPointInGround(transform, g.K_Q.p()));
}

//==============================================================================
//                      IMPLICIT CYLINDER SURFACE
//==============================================================================

double ImplicitCylinderSurface::calcSurfaceConstraintImpl(Vector3 p) const
{
    return p.x() * p.x() + p.y() * p.y() - _radius * _radius;
}

Vector3 ImplicitCylinderSurface::calcSurfaceConstraintGradientImpl(
    Vector3 p) const
{
    return {2. * p.x(), 2. * p.y(), 0.};
}

Mat3x3 ImplicitCylinderSurface::calcSurfaceConstraintHessianImpl(Vector3) const
{
    Mat3x3 hessian;
    hessian.fill(0.);

    hessian(0, 0) = 2.;
    hessian(1, 1) = 2.;

    return hessian;
}

bool ImplicitCylinderSurface::isAboveSurfaceImpl(Vector3 point, double bound)
    const
{
    const double radialDistance = point.x() * point.x() + point.y() * point.y();
    const double radialBound    = (_radius + bound) * (_radius + bound);
    return radialDistance > radialBound;
}

//==============================================================================
//                      ANALYTIC CYLINDER SURFACE
//==============================================================================

void AnalyticCylinderSurface::calcLocalGeodesicImpl(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length,
    Geodesic& geodesic)
{
    const double r = _radius;
    const double l = length;

    // Cylinder axis assumed to be aligned with z-axis.
    const Vector3 x{1., 0., 0.};
    const Vector3 y{0., 1., 0.};
    const Vector3 z{0., 0., 1.};

    // Initial darboux frame.
    Trihedron& K_P           = geodesic.K_P;
    Geodesic::Variation& v_P = geodesic.v_P;
    Geodesic::Variation& w_P = geodesic.w_P;

    // Normal direction.
    Darboux f_P = Darboux::FromTangenGuessAndNormal(
        initVelocity,
        Vector3{initPosition.x(), initPosition.y(), 0.});

    // Initial position on surface.
    const Vector3 p_P = f_P.n() * r + z * initPosition.z();

    K_P = Trihedron(p_P, f_P);

    // Rotation angle between start and end frame about cylinder axis.
    const double alpha = -l / r * K_P.b()[2];
    AssertEq(
        alpha,
        (l / r) * z.cross(K_P.n()).dot(K_P.t()),
        "Rewriting alpha check");

    // Distance along cylinder axis between start and end frame.
    const double h = l * K_P.t()[2];
    AssertEq(h, l * z.dot(K_P.t()), "Rewriting h check");

    AssertEq(alpha * alpha * r * r + h * h, l * l, "(alpha * r)^2 + h^2 = l^2");

    AssertEq(
        (K_P.t().cross(z) * l).norm(),
        std::abs(alpha * r),
        "||t X z || * l = |alpha * r|");
    AssertEq(K_P.t().dot(z) * l, h, "t.T z * l = h");

    // Rotation angle variation to initial direction variation.
    const double dAlpha_dTheta = (l / r) * K_P.t()[2];
    AssertEq(
        dAlpha_dTheta,
        -(l / r) * z.cross(K_P.n()).dot(K_P.n().cross(K_P.t())),
        "Rewriting dAlpha_dTheta check");

    // Distance along axis variation to initial direction variation.
    const double dh_dTheta = l * K_P.b()[2];
    AssertEq(
        dh_dTheta,
        -l * z.dot(K_P.n().cross(K_P.t())),
        "Rewriting dh_dTheta check");

    // Rotation angle variation to length variation.
    const double dAlpha_dl = -1. / r * K_P.b()[2];
    AssertEq(
        dAlpha_dl,
        (1. / r) * z.cross(K_P.n()).dot(K_P.t()),
        "Rewriting dAlpha_dl");

    // Distance along axis variation to length variation.
    const double dh_dl = K_P.t()[2];

    // Start position variation.
    const Vector3 zeros{0., 0., 0.};

    v_P.col(0) = K_P.t();
    v_P.col(1) = K_P.b();
    v_P.col(2) = zeros;
    v_P.col(3) = zeros;

    // Start frame variation.
    w_P.col(0) = z * dAlpha_dl;
    w_P.col(1) = z * K_P.t()[2] / r;
    w_P.col(2) = -K_P.n();
    w_P.col(3) = zeros;

    AssertEq(w_P.col(0), z * K_P.t().dot(z.cross(K_P.n())) / r, "Rewriting w_P.col(0) check");

    // Integration of the angular rotation about cylinder axis.
    const Rotation dq{Eigen::AngleAxisd(alpha, z)};

    // Final darboux frame.
    const Vector3 t_Q = dq * K_P.t();
    const Vector3 n_Q = dq * K_P.n();
    Darboux f_Q{t_Q, n_Q, t_Q.cross(n_Q)};

    // Final position.
    const Vector3 p_Q = f_Q.n() * r + z * (p_P.z() + h);

    // Final trihedron.
    Trihedron& K_Q           = geodesic.K_Q;
    Geodesic::Variation& v_Q = geodesic.v_Q;
    Geodesic::Variation& w_Q = geodesic.w_Q;

    K_Q = Trihedron(p_Q, f_Q);

    // Final position variation.
    v_Q.col(0) = f_Q.t();
    v_Q.col(1) = f_Q.b();
    v_Q.col(2) = z.cross(p_Q) * dAlpha_dTheta + z * dh_dTheta;
    v_Q.col(3) = z.cross(p_Q) * dAlpha_dl + z * dh_dl;

    w_Q.col(0) = -z * f_Q.b()[2] / r;
    w_Q.col(1) = z * f_Q.t().dot(z) / r;
    w_Q.col(2) = dAlpha_dTheta * z - f_Q.n();
    w_Q.col(3) = dAlpha_dl * z;

    AssertEq(w_Q.col(0), z * f_Q.t().dot(z.cross(f_Q.n())) / r, "Rewriting w_Q.col(0) check");

    geodesic.length = length;
}

bool AnalyticCylinderSurface::isAboveSurfaceImpl(Vector3 point, double bound)
    const
{
    return point.x() * point.x() + point.y() * point.y() -
               (_radius + bound) * (_radius + bound) >
           0.;
}

double AnalyticCylinderSurface::calcLocalNormalCurvatureImpl(
    Vector3 point,
    Vector3 tangent) const
{
    AssertEq(ImplicitCylinderSurface(_radius).calcNormalCurvature(point, tangent), -4. * _radius, "Verify AnalyticCylinderSurface::kn");
    return - 4. * _radius;
}

Vector3 AnalyticCylinderSurface::calcLocalSurfaceNormalImpl(Vector3 point) const
{
    const double x = point.x();
    const double y = point.y();
    return Vector3{x, y, 0.} / std::sqrt(x * x + y * y);
}

double AnalyticCylinderSurface::calcLocalGeodesicTorsionImpl(
    Vector3 point,
    Vector3 tangent) const
{
    AssertEq(ImplicitCylinderSurface(_radius).calcGeodesicTorsion(point, tangent), 0., "Verify AnalyticCylinderSurface::tau_g");
    return 0.;
}

void AnalyticCylinderSurface::calcPathPointsImpl(std::vector<Vector3>& points, Transf transform)
    const
{
    const Geodesic& g = getGeodesic();

    // Push the start point.
    Vector3 p = g.K_P.p();
    points.push_back(calcPointInGround(transform, p));

    const double angle = g.length / _radius;
    size_t n = static_cast<size_t>(std::abs(angle / _sampleSpacing));
    n = std::min(_maxNrOfSamples, n);

    if (n > 0) {
        const Vector3 z {0., 0., 1.};
        const double d = 1. / static_cast<double>(n+1);

        const double s = sin(angle * d);
        const double c = cos(angle * d);

        const double dh = (g.K_Q.p()[2] - g.K_P.p()[2]) * d;
        for (size_t i = 0; i < n; ++i) {
            p = {c * p[0] - s * p[1], s * p[0] + c * p[1], p[2] + dh};

            // TODO can be made more efficient by computing dq = R * dq * RT
            points.push_back(calcPointInGround(transform, p));
        }
    }

    // Push the end point.
    points.push_back(calcPointInGround(transform, g.K_Q.p()));
}

std::pair<bool, size_t> AnalyticCylinderSurface::
    calcLocalLineToSurfaceTouchdownPointImpl(
        Vector3 a,
        Vector3 b,
        Vector3& p,
        size_t,
        double)
{
    using Vector2 = Eigen::Vector<double, 2>;
    const double c = calcPointOnLineNearOriginAsFactor(Vector2{a[0], a[1]}, Vector2{b[0], b[1]});

    const Vector3 pl = a + (b - a) * c;

    const bool touchdown = pl.dot(pl) < _radius * _radius;

    if (touchdown) {
        p = pl;
    }

    return {touchdown, 0};
}

//==============================================================================
//                      IMPLICIT TORUS SURFACE
//==============================================================================

double ImplicitTorusSurface::calcSurfaceConstraintImpl(Vector3 position) const
{
    const Vector3& p = position;
    const double x   = p.x();
    const double y   = p.y();

    const double r = _smallRadius;
    const double R = _bigRadius;

    const double c = (p.dot(p) + R * R - r * r);
    return c * c - 4. * R * R * (x * x + y * y);
}

Vector3 ImplicitTorusSurface::calcSurfaceConstraintGradientImpl(
    Vector3 position) const
{
    const Vector3& p = position;
    const double x   = p.x();
    const double y   = p.y();

    const double r = _smallRadius;
    const double R = _bigRadius;

    const double c = (p.dot(p) + R * R - r * r);
    return 4. * c * p - 8. * R * R * Vector3{x, y, 0.};
}

ImplicitSurface::Hessian ImplicitTorusSurface::calcSurfaceConstraintHessianImpl(
    Vector3 position) const
{
    ImplicitSurface::Hessian hessian;

    const Vector3& p = position;

    const double r = _smallRadius;
    const double R = _bigRadius;

    const double c = (p.dot(p) + R * R - r * r);

    hessian.setIdentity();
    hessian *= 4. * c;

    hessian += 8. * p * p.transpose();

    hessian(0, 0) -= 8. * R * R;
    hessian(1, 1) -= 8. * R * R;

    return hessian;
}

bool ImplicitTorusSurface::isAboveSurfaceImpl(Vector3 point, double bound) const
{
    const double x = point.x();
    const double y = point.y();
    const double z = point.z();

    const double r = _smallRadius + bound;
    const double R = _bigRadius;

    const double c = (sqrt(x * x + y * y) - R);
    return c * c + z * z - r * r > 0.;
}

//==============================================================================
//                      WRAP OBSTACLE
//==============================================================================

void WrapObstacle::applyVariation(const Geodesic::Correction& c)
{
    _surface->applyVariation(c);
    calcGeodesicInGround();
}

const Geodesic& WrapObstacle::calcGeodesic(Geodesic::InitialConditions g0)
{
    calcInLocal(getOffsetFrame(), g0);
    _surface->calcGeodesic(g0);
    return calcGeodesicInGround();
}

const Geodesic& WrapObstacle::calcGeodesicInGround()
{
    if (isActive(getStatus())) {
        _geodesic = _surface->getGeodesic();
        calcInGround(getOffsetFrame(), _geodesic);
    }
    return getGeodesic();
}

Vector3 WrapObstacle::getPathStartGuess() const
{
    return calcPointInGround(getOffsetFrame(), _surface->getPathStartGuess());
}

void WrapObstacle::attemptTouchdown(
    const Vector3& p_O,
    const Vector3& p_I,
    size_t maxIter,
    double eps)
{
    Geodesic::Status s = getStatus();
    const bool active  = isActive(s & ~Geodesic::Status::LiftOff);
    const bool liftoff = (s & Geodesic::Status::LiftOff) > 0;

    // Only attempt touchdown if segment is active, and in liftoff.
    if (!active || !liftoff)
        return;

    // Attempt touchdown.
    const Transf& q = getOffsetFrame();
    const Vector3 a = calcPointInLocal(q, p_O);
    const Vector3 b = calcPointInLocal(q, p_I);
    Vector3 p       = {NAN, NAN, NAN};

    if (!_surface
             ->calcLocalLineToSurfaceTouchdownPoint(a, b, p, maxIter, eps)) {
        // Stop if no touchdown detected.
        return;
    }

    Geodesic::InitialConditions g0{p, b - a, 0.};
    _surface->calcGeodesic(g0);
    calcGeodesicInGround();
}

void WrapObstacle::detectLiftOff(const Vector3& p_O, const Vector3& p_I)
{
    bool liftoff = true;

    // Only detect liftoff if surface is active and has negative length.
    Geodesic::Status s = getStatus();
    liftoff &= isActive(s);
    liftoff &= (s & Geodesic::Status::NegativeLength) > 0;

    // Use the normal to detect liftoff.
    const Geodesic g = getGeodesic();
    liftoff &= g.K_P.n().dot(p_O - g.K_P.p()) > 0.;
    liftoff &= g.K_Q.n().dot(p_I - g.K_Q.p()) > 0.;

    if (liftoff) {
        updStatus() |= Geodesic::Status::LiftOff;
    }
}

Vector3 WrapObstacle::calcSurfaceNormal(Vector3 point) const
{
    const Transf& offset = getOffsetFrame();
    return calcVectorInGround(
        offset,
        _surface->calcSurfaceNormal(calcPointInLocal(offset, point)));
}

double WrapObstacle::calcNormalCurvature(Vector3 point, Vector3 tangent) const
{
    const Transf& offset = getOffsetFrame();
    return _surface->calcNormalCurvature(
        calcPointInLocal(offset, point),
        calcVectorInLocal(offset, tangent));
}

double WrapObstacle::calcGeodesicTorsion(Vector3 point, Vector3 tangent) const
{
    const Transf& offset = getOffsetFrame();
    return _surface->calcGeodesicTorsion(
        calcPointInLocal(offset, point),
        calcVectorInLocal(offset, tangent));
}

void WrapObstacle::calcPathPoints(std::vector<Vector3>& points) const
{
    _surface->calcPathPoints(points, getOffsetFrame());
}

bool WrapObstacle::isAboveSurface(const Vector3& p, double bound) const
{
    return _surface->isAboveSurface(
        calcPointInLocal(getOffsetFrame(), p),
        bound);
}

//==============================================================================
//                      WRAP PATH
//==============================================================================

enum class PathErrorKind
{
    Tangent,
    Normal,
    Binormal,
};

Vector3 getAxis(const Trihedron& K, PathErrorKind kind)
{
    switch (kind) {
    case PathErrorKind::Tangent:
        return K.t();
    case PathErrorKind::Normal:
        return K.n();
    case PathErrorKind::Binormal:
        return K.b();
    default:
        break;
    }
    return {NAN, NAN, NAN};
}

double getOutput(PathErrorKind kind)
{
    return kind == PathErrorKind::Tangent ? 1. : 0.;
}

using ActiveLambda = std::function<void(size_t prev, size_t next, size_t current, bool isActive)>;

void CallMe(size_t prev, size_t next, size_t current, bool isActive, ActiveLambda& f)
{
    f(prev, next, current, isActive);
}

template<typename... FUNCS>
void CallMe(size_t prev, size_t next, size_t current, bool isActive, ActiveLambda& f, FUNCS&&... fs)
{
    f(prev, next, current, isActive);
    CallMe(prev, next, current, isActive, std::forward<FUNCS>(fs)...);
}

template<typename... FUNCS>
void forEachActive(
    const std::vector<WrapObstacle>& obs,
    ActiveLambda& f,
    FUNCS&&... fs)
{
    const ptrdiff_t n = obs.size();
    ptrdiff_t next    = 0;
    ptrdiff_t prev    = -1;

    for (ptrdiff_t i = 0; i < n; ++i) {
        // Find the active segment before the current.
        if (i > 0) {
            if (isActive(obs.at(i - 1).getStatus())) {
                prev = i - 1;
            }
        }

        // Find the active segment after the current.
        if (next <= i) {
            for (; ++next < n;) {
                const WrapObstacle& o = obs.at(next);
                if (isActive(o.getStatus())) {
                    break;
                }
            }
        }

        CallMe(prev < 0 ? n : prev, next, i, isActive(obs.at(i).getStatus()), f, std::forward<FUNCS>(fs)...);
    }
}

double calcPathError(const LineSeg& e, const Trihedron& K, PathErrorKind kind)
{
    return e.d.dot(getAxis(K, kind)) - getOutput(kind);
}

void calcPathError(
    const std::vector<WrapObstacle>& obs,
    const std::vector<LineSeg>& lines,
    Eigen::VectorXd& pathError,
    PathErrorKind kind)
{
    size_t i      = 0;
    ptrdiff_t row = -1;
    for (const WrapObstacle& o : obs) {
        const Geodesic& g = o.getGeodesic();
        pathError(++row)  = calcPathError(lines.at(i), g.K_P, kind);
        pathError(++row)  = calcPathError(lines.at(++i), g.K_Q, kind);
    }
}

void calcPathErrorJacobian(
    const std::vector<WrapObstacle>& obs,
    const std::vector<LineSeg>& lines,
    Eigen::MatrixXd& pathErrorJacobian,
    PathErrorKind kind)
{
    size_t row = 0;
    size_t col = 0;

    constexpr size_t Q = Geodesic::DOF;
    const size_t n     = obs.size();

    pathErrorJacobian.fill(0.);

    ActiveLambda f = [&](size_t prev, size_t next, size_t i, bool isActive) {
        if (!isActive) {
            return;
        }

        const LineSeg& l_P = lines.at(i);
        const LineSeg& l_Q = lines.at(i + 1);

        const Geodesic& g = obs.at(i).getGeodesic();
        const Vector3 a_P = getAxis(g.K_P, kind);
        const Vector3 a_Q = getAxis(g.K_Q, kind);

        pathErrorJacobian.block<1, Q>(row, col) =
            calcPathErrorJacobian(l_P, a_P, g.v_P, g.w_P);
        if (prev != n) {
            pathErrorJacobian.block<1, Q>(row, col - Q) =
                -calcDirectionJacobian(
                    l_P,
                    a_P,
                    obs.at(prev).getGeodesic().v_Q);
        }
        ++row;

        pathErrorJacobian.block<1, Q>(row, col) =
            calcPathErrorJacobian(l_Q, a_Q, g.v_Q, g.w_Q, true);
        if (next != n) {
            pathErrorJacobian.block<1, Q>(row, col + Q) =
                calcDirectionJacobian(l_Q, a_Q, obs.at(next).getGeodesic().v_P);
        }
        ++row;
        col += Q;
    };

    forEachActive(obs, f);
}

double calcPathLength(
    const std::vector<WrapObstacle>& obs,
    const std::vector<LineSeg>& lines)
{
    double lTot = 0.;
    for (const LineSeg& l : lines) {
        lTot += l.l;
    }

    for (const WrapObstacle& o : obs) {
        if (!isActive(o.getStatus()))
            continue;
        lTot += std::abs(o.getGeodesic().length);
    }
    return lTot;
}

void calcPathLengthJacobian(
    const std::vector<WrapObstacle>& obs,
    const std::vector<LineSeg>& lines,
    Eigen::VectorXd& lengthJacobian)
{
    size_t i           = 0;
    constexpr size_t Q = Geodesic::DOF;
    for (const WrapObstacle& o : obs) {
        if (!isActive(o.getStatus()))
            continue;
        const Geodesic& g = o.getGeodesic();

        // Length of curve segment.
        GeodesicJacobian jacobian = {0., 0., 0., 1.};

        // Length of previous line segment.
        const Vector3 e_P = lines.at(i).d;
        const Vector3 e_Q = lines.at(i + 1).d;

        jacobian += e_P.transpose() * g.v_P;
        jacobian -= e_Q.transpose() * g.v_Q;

        // Write jacobian elements.
        lengthJacobian.middleRows<Q>(i * Q) = jacobian;

        ++i;
    }
}

void calcLineSegments(
    Vector3 p_O,
    Vector3 p_I,
    const std::vector<WrapObstacle>& obs,
    std::vector<LineSeg>& lines)
{
    const size_t n = obs.size();
    lines.reserve(n+1);
    lines.clear();

    Vector3 a = std::move(p_O);
    for (size_t i = 0; i < n; ++i) {
        if (!isActive(obs.at(i).getStatus())) {
            continue;
        }

        const Geodesic& g = obs.at(i).getGeodesic();
        const Vector3 b   = g.K_P.p();
        lines.emplace_back(a, b);
        a = g.K_Q.p();
    }
    lines.emplace_back(a, p_I);
}

double calcMaxPathError(
    const std::vector<WrapObstacle>& obs,
    const std::vector<LineSeg>& lines,
    PathErrorKind kind)
{
    size_t i         = 0;
    double maxAbsErr = 0.;
    const double y   = getOutput(kind);
    for (const WrapObstacle& o : obs) {
        if (!isActive(o.getStatus())) {
            continue;
        }

        const Geodesic& g = o.getGeodesic();

        double err = lines.at(i).d.dot(getAxis(g.K_P, kind)) - y;
        maxAbsErr  = std::max(maxAbsErr, std::abs(err));

        err       = lines.at(++i).d.dot(getAxis(g.K_Q, kind)) - y;
        maxAbsErr = std::max(maxAbsErr, std::abs(err));
    }
    return maxAbsErr;
}

void calcInitZeroLengthGeodesics(
    const Vector3& p_O,
    std::vector<WrapObstacle>& obs)
{
    Vector3 prev = p_O;
    for (WrapObstacle& o : obs) {
        if (!isActive(o.getStatus())) {
            continue;
        }

        Vector3 initPositon  = o.getPathStartGuess();
        Vector3 initVelocity = (initPositon - prev);

        // Shoot a zero-length geodesic as initial guess.
        o.calcGeodesic({initPositon, initVelocity, 0.});
        prev = o.getGeodesic().K_Q.p();
    }
}

//==============================================================================
//                      SOLVER
//==============================================================================

Geodesic::Correction calcClamped(
    const CorrectionBounds& bnds,
    const Geodesic::Correction& correction)
{
    auto Clamp = [](double bnd, double x) {
        if (std::abs(x) > std::abs(bnd)) {
            return x / std::abs(x) * std::abs(bnd);
        }
        return x;
    };
    const double maxAngle = bnds.maxAngleDegrees / 180. * M_PI;

    return {
        Clamp(bnds.maxRepositioning, correction(0)),
        Clamp(bnds.maxRepositioning, correction(1)),
        Clamp(maxAngle, correction(2)),
        Clamp(bnds.maxLengthening, correction(3)),
    };
}

double WrappingPathSolver::calcMaxCorrectionStep() const
{
    double maxCorr = 0.;
    for (int i = 0; i < _pathCorrections.rows(); ++i) {
        maxCorr = std::max(maxCorr, std::abs(_pathCorrections[i]));
    }
    return maxCorr;
}

// Apply max angle limit to the path error.
void clampPathError(Eigen::VectorXd& pathError, double maxAngleDegrees)
{

    static constexpr double PI = M_PI;
    const double cap           = 1. - cos(maxAngleDegrees / 180. * PI);

    /* std::cout << "before pathError =" << pathError.transpose() << std::endl;
     */

    for (int r = 0; r < pathError.rows(); ++r) {
        double& err = pathError[r];
        err         = (std::abs(err) > cap) ? err / std::abs(err) * cap : err;
    }

    /* std::cout << "after pathError =" << pathError.transpose() << std::endl;
     */
}

const Geodesic::Correction* WrappingPathSolver::begin() const
{
    static_assert(
        sizeof(Geodesic::Correction) == sizeof(double) * Geodesic::DOF);
    return reinterpret_cast<const Geodesic::Correction*>(&_pathCorrections(0));
}

size_t countActive(const std::vector<WrapObstacle>& obs)
{
    size_t n = 0;
    for (const WrapObstacle& o : obs) {
        if (isActive(o.getStatus())) {
            ++n;
        }
    }
    return n;
}

void WrappingPathSolver::resize(size_t n)
{
    constexpr size_t Q = Geodesic::DOF;

    _pathCorrections.resize(n * Q);
    _vec.resize(n * Q);
    _mat.resize(n * Q, n * Q);

    _pathCorrections.fill(NAN);
    _vec.fill(0.);
    _mat.fill(0.);
}

void OptSolver::resize(size_t n)
{
    WrappingPathSolver::resize(n);

    constexpr size_t Q = Geodesic::DOF;
    constexpr size_t C = 2;

    _gT.resize(n * C);
    _qT.resize(n * Q);
    _JT.resize(n * C, n * Q);
    _gN.resize(n * C);
    _JN.resize(n * C, n * Q);
    _qN.resize(n * Q);
    _gB.resize(n * C);
    _JB.resize(n * C, n * Q);
    _qB.resize(n * Q);
    _JL.resize(n * Q);

    _P.resize(n * Q, n * Q);
    _d.resize(n * Q);

    _vecL.resize(n * Q);

    // Reset values.

    _JT.fill(0.);
    _gT.fill(0.);
    _qT.fill(0.);
    _JN.fill(0.);
    _gN.fill(0.);
    _qN.fill(0.);
    _JB.fill(0.);
    _gB.fill(0.);
    _qB.fill(0.);
    _JL.fill(0.);
    _l = 0.;
    _P.setIdentity();
    _d.fill(0.);

    _pathCorrections.fill(NAN);
}

void WrappingPath::Solver::resize(size_t n)
{
    WrappingPathSolver::resize(n);

    constexpr size_t Q = Geodesic::DOF;
    constexpr size_t C = 2;

    _g.resize(n * C);
    _J.resize(n * C, n * Q);

    _g.fill(NAN);
    _J.fill(0.);
}

void addPathErrorToCost(
    const std::vector<WrapObstacle>& obs,
    const std::vector<LineSeg>& lines,
    Eigen::VectorXd& fun,
    Eigen::MatrixXd& jac,
    Eigen::VectorXd& vec,
    Eigen::MatrixXd& mat,
    PathErrorKind kind)
{
    calcPathErrorJacobian(obs, lines, jac, kind);
    calcPathError(obs, lines, fun, kind);

    mat += jac.transpose() * jac;
    vec += jac.transpose() * fun;
}

bool WrappingPathSolver::calcPathCorrection(
    const std::vector<WrapObstacle>& obs,
    const std::vector<LineSeg>& lines,
    double pathErr,
    double pathErrBnd)
{
    resize(countActive(obs));
    calcPathCorrectionImpl(obs, lines, pathErr, pathErrBnd);
    solve();
    return true;
}

void WrappingPath::Solver::calcPathCorrectionImpl(
    const std::vector<WrapObstacle>& obs,
    const std::vector<LineSeg>& lines,
    double pathErr,
    double)
{
    for (int i = 0; i < _mat.rows(); ++i) {
        _mat(i, i) = pathErr;
    }

    addPathErrorToCost(obs, lines, _g, _J, _vec, _mat, PathErrorKind::Normal);
    addPathErrorToCost(obs, lines, _g, _J, _vec, _mat, PathErrorKind::Binormal);
}

void WrappingPathSolver::solve()
{
    _pathCorrections = -_mat.colPivHouseholderQr().solve(_vec);
}

void WrappingPathSolver::print(std::ostream& os) const
{
    os << "WrappingPathSolver{\n";
    os << "    maxStep: {max... },\n";
    os << "}\n";
}

void OptSolver::print(std::ostream& os) const
{
    os << "OptSolver{\n";
    os << "    maxStep: {max... },\n";
    os << "}\n";
}

void WrappingPath::Solver::print(std::ostream& os) const
{
    os << "Solver{\n";
    os << "    maxStep: {max... },\n";
    os << "    g: " << _g.transpose() << "\n";
    os << "    J:\n" << _J << "\n";
    os << "    vec: " << _vec.transpose() << "\n";
    os << "    mat:\n" << _mat << "\n";
    os << "    COR: " << _pathCorrections.transpose() << "\n";
    os << "}\n";
}

void OptSolver::calcPathCorrectionImpl(
    const std::vector<WrapObstacle>& obs,
    const std::vector<LineSeg>& lines,
    double pathErr,
    double)
{
    _P *= _opts.m_CostW ? (pathErr + 1e-4) : 0.;

    if (_opts.m_CostL) {
        calcPathLengthJacobian(obs, lines, _JL);
        _l = calcPathLength(obs, lines);
        _d += _JL * 0.1;
        /* _P += _JL * _JL.transpose(); */
        for (int i = 0; i < _P.rows(); ++i) {
            _P(i, i) += 1.;
        }
    }

    if (_opts.m_CostT) {
        calcPathErrorJacobian(obs, lines, _JT, PathErrorKind::Tangent);
        calcPathError(obs, lines, _gT, PathErrorKind::Tangent);

        _P += _JT.transpose() * _JT;
        _d += _JT.transpose() * _gT;
    }

    if (_opts.m_CostN || _opts.m_AugN) {
        calcPathErrorJacobian(obs, lines, _JN, PathErrorKind::Normal);
        calcPathError(obs, lines, _gN, PathErrorKind::Normal);

        if (_opts.m_CostN) {
            _P += _JN.transpose() * _JN;
            _d += _JN.transpose() * _gN;
        }
    }

    if (_opts.m_CostB) {
        calcPathErrorJacobian(obs, lines, _JB, PathErrorKind::Binormal);
        calcPathError(obs, lines, _gB, PathErrorKind::Binormal);

        _P += _JB.transpose() * _JB;
        _d += _JB.transpose() * _gB;
    }

    _vec = _P.colPivHouseholderQr().solve(_d);
    if (_opts.m_AugN) {
        _JJT    = _JN * _JN.transpose();
        _JJTinv = _JJT.colPivHouseholderQr().inverse();
        _pathCorrections =
            _JN.transpose() * _JJTinv * (_JN * _vec - _gN) - _vec;
    } else {
        _pathCorrections = -_vec;
    }
}

Geodesic::Status clearErrorFlag()
{
    return ~(
        Geodesic::Status::InitialTangentParallelToNormal |
        Geodesic::Status::PrevLineSegmentInsideSurface |
        Geodesic::Status::NextLineSegmentInsideSurface |
        Geodesic::Status::TouchDownFailed | Geodesic::Status::IntegratorFailed);
}

void WrappingPath::calcInitPath(double eps, size_t maxIter)
{
    calcInitZeroLengthGeodesics(getStart(), updSegments());

    return;
    calcPath(false, eps, maxIter, true);
}

void WrappingPath::calcPath(
    bool breakOnErr,
    double,
    size_t maxIter,
    bool preventLiftOff)
{
    updStatus()    = WrappingPath::Status::Ok;
    const size_t n = getSegments().size();
    if (n == 0) {
        return;
    }

    // Clear the error flags.
    static const Geodesic::Status clearErrMap = clearErrorFlag();
    for (WrapObstacle& o : updSegments()) {
        o.updStatus() = o.getStatus() & clearErrMap;
    }

    // Update the frame offsets.
    for (WrapObstacle& o : _segments) {
        o.calcGeodesicInGround();
    }

    auto GetPrevPoint = [&](size_t i) {
        return i == n
                                ? getStart()
                                : getSegments().at(i).getGeodesic().K_Q.p();
    };
    auto GetNextPoint = [&](size_t i) {
        return i == n
                                ? getEnd()
                                : getSegments().at(i).getGeodesic().K_P.p();
    };

    // Helper for detecting if start/end points lie inside the surfacce.
    ActiveLambda DetectInsideSurfaceError = [&](size_t prev, size_t next, size_t i, bool) {
        const Vector3 p_O = GetPrevPoint(prev);
        const Vector3 p_I = GetNextPoint(next);
        if (!getSegments().at(i).isAboveSurface(p_O, 0.)) {
            updSegments().at(i).updStatus() |=
                Geodesic::Status::PrevLineSegmentInsideSurface;
        }
        if (!getSegments().at(i).isAboveSurface(p_I, 0.)) {
            updSegments().at(i).updStatus() |=
                Geodesic::Status::NextLineSegmentInsideSurface;
        }
    };

    // Helper for detecting touchdown and liftoff.
    ActiveLambda TouchdownAndLiftOff = [&](size_t prev, size_t next, size_t i, bool) {
        const Vector3 p_O = GetPrevPoint(prev);
        const Vector3 p_I = GetNextPoint(next);
        updSegments().at(i).attemptTouchdown(p_O, p_I);
        updSegments().at(i).detectLiftOff(p_O, p_I);
    };

    const size_t prevLoopIter = loopIter;
    for (loopIter = 0; loopIter < maxIter; ++loopIter) {

        // Detect touchdown & liftoff.
        if (preventLiftOff) {
            forEachActive(getSegments(), DetectInsideSurfaceError);
        } else {
            forEachActive(getSegments(), DetectInsideSurfaceError, TouchdownAndLiftOff);
        }

        // Compute the line segments.
        calcLineSegments(getStart(), getEnd(), getSegments(), _lineSegments);

        // Evaluate path error, and stop when converged.
        _pathError = calcMaxPathError(
            getSegments(),
            getLineSegments(),
            PathErrorKind::Tangent);
        if (_pathError < _pathErrorBound) {
            loopIter = loopIter == 0 ? prevLoopIter : loopIter;
            return;
        }

        // Optionally return on error.
        if (breakOnErr) {
            for (const WrapObstacle& o : getSegments()) {
                if (isError(o.getStatus())) {
                    return;
                }
            }
        }

        // Compute path corrections.
        if (!updSolver().calcPathCorrection(
                getSegments(),
                _lineSegments,
                _pathError,
                _pathErrorBoundWide)) {
            updStatus() = WrappingPath::Status::FailedToInvertJacobian;
            return;
        }

        // Apply path corrections.
        const Geodesic::Correction* corrIt = getSolver().begin();
        for (WrapObstacle& o : updSegments()) {
            if (!isActive(o.getStatus())) {
                continue;
            }
            o.applyVariation(*corrIt);
            ++corrIt;
        }
    }

    updStatus() = Status::ExceededMaxIterations;
}

const std::vector<Vector3>& WrappingPath::calcPathPoints()
{
    _pathPoints.clear();
    _pathPoints.push_back(getStart());
    for (const WrapObstacle& o : getSegments()) {
        if (!isActive(o.getStatus()))
            continue;
        o.calcPathPoints(_pathPoints);
    }
    _pathPoints.push_back(getEnd());
    return _pathPoints;
}

double WrappingPath::getLength() const {
    double l = 0.;
    for (const LineSeg& s: getLineSegments()) {
        l += s.l;
    }
    for (const WrapObstacle& o: getSegments()) {
        l += o.getGeodesic().length;
    }
    return l;
}
