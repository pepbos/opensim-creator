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
    return os << "}";
}

std::ostream& operator<<(std::ostream& os, const Geodesic& x)
{
    os << "Geodesic{\n";
    os << "    K_P: " << x.K_P << ",\n";
    os << "    K_Q: " << x.K_Q << ",\n";
    os << "    l: " << x.length << ",\n";
    os << "    s: " << x.status << ",\n";
    os << "    nKnts: " << x.samples.size();
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
        /* throw std::runtime_error(msg); */
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

namespace
{
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

ImplicitGeodesicState operator*(double dt, ImplicitGeodesicStateDerivative& dy)
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

} // namespace

//==============================================================================
//                      DARBOUX FRAME
//==============================================================================

Darboux::Matrix stackCols(Vector3 t, Vector3 n, Vector3 b)
{
    Darboux::Matrix mat;
    mat.col(0) = t;
    mat.col(1) = n;
    mat.col(2) = b;
    return mat;
}

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
    b = b / b.norm();

    return Darboux(t, n, b);
}

Darboux::Darboux(Vector3 tangent, Vector3 normal, Vector3 binormal)
{
    _rotation.col(0) = tangent;
    _rotation.col(1) = normal;
    _rotation.col(2) = binormal;

    AssertDarbouxFrame(*this);
}

Trihedron::Trihedron(Vector3 point, Darboux rotation)
    : _point(std::move(point)), _rotation(std::move(rotation))
{}

Trihedron::Trihedron(Vector3 point, Vector3 tangent, Vector3 normal, Vector3 binormal)
    : Trihedron(std::move(point), Darboux(tangent, normal, binormal))
{}

Trihedron Trihedron::FromPointAndTangentGuessAndNormal(
        Vector3 point,
        Vector3 tangentGuess, Vector3 normal)
{
    return Trihedron(std::move(point), Darboux::FromTangenGuessAndNormal(tangentGuess, normal));
}

namespace
{

Darboux operator*(Eigen::Quaterniond lhs, const Darboux& rhs)
{
    Eigen::Matrix<double, 3, 3> mat = lhs.toRotationMatrix();
    const Vector3 t = mat * rhs.t();
    const Vector3 n = mat * rhs.n();
    return {t, n, t.cross(n)};
}

Trihedron operator*(Eigen::Quaterniond lhs, const Trihedron& rhs)
{
    Eigen::Matrix<double, 3, 3> mat = lhs.toRotationMatrix();
    const Vector3 t = mat * rhs.t();
    const Vector3 n = mat * rhs.n();
    return {mat * rhs.p(), t, n, t.cross(n)};
}

} // namespace

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

void calcInGroundFrame(const Transf& transform, Trihedron& K)
{
    Vector3& p = K.updPoint();
    p = calcPointInGround(transform, p);

    // TODO Rotation transform here.
}

void calcGeodesicInGlobal(const Transf& transform, Geodesic& g)
{
    calcInGroundFrame(transform, g.K_P);
    calcInGroundFrame(transform, g.K_Q);
    for (Trihedron& Ki: g.samples) {
        calcInGroundFrame(transform, Ki);
    }
}

//==============================================================================
//                      SOME MATHS
//==============================================================================

Vector3 calcPointOnLineNearPoint(Vector3 a, Vector3 b, Vector3 point)
{
    Vector3 p0 = a - point;
    Vector3 p1 = b - point;

    const Vector3 e = p1 - p0;

    Vector3 p = p0 - p0.dot(e) * e / e.dot(e);

    const double d0 = p0.dot(p0);
    const double d1 = p1.dot(p1);
    const double d  = p.dot(p);

    if (d0 < d) {
        return a;
    }
    if (d1 < d) {
        return b;
    }
    return p + point;
};

double sign(double x)
{
    return static_cast<double>(x > 0.) - static_cast<double>(x < 0.);
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
Vector3 calcSurfaceNormal(
    const ImplicitSurface& s,
    Vector3 point)
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
    return calcNormalCurvature(s, point, std::move(tangent)) * calcSurfaceNormal(s, point);
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
    return Trihedron::FromPointAndTangentGuessAndNormal(p, t, calcSurfaceNormal(s, p));
}

Trihedron calcTrihedron(
    const ImplicitSurface& s,
    const ImplicitGeodesicState& q)
{
    return Trihedron::FromPointAndTangentGuessAndNormal(q.position, q.velocity, calcSurfaceNormal(s, q.position));
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

void RungeKutta4(
    const ImplicitSurface& s,
    ImplicitGeodesicState& q,
    double& t,
    double dt)
{
    RungeKutta4<ImplicitGeodesicState, ImplicitGeodesicStateDerivative>(
        q,
        t,
        dt,
        [&](const ImplicitGeodesicState& qk)
            -> ImplicitGeodesicStateDerivative {
            return calcImplicitGeodesicStateDerivative(
                qk,
                calcAcceleration(s, qk.position, qk.velocity),
                calcGaussianCurvature(s, qk));
        });
}

// Implicit geodesic shooter.
std::pair<ImplicitGeodesicState, ImplicitGeodesicState>
calcLocalImplicitGeodesic(
    const ImplicitSurface& s,
    Vector3 positionInit,
    Vector3 velocityInit,
    double length,
    size_t steps,
    std::function<void(const ImplicitGeodesicState&)>& Monitor)
{
    ImplicitGeodesicState xStart{
        std::move(positionInit),
        std::move(velocityInit)};
    constexpr size_t maxIter = 10;
    if (calcFastSurfaceProjection(s, xStart.position, xStart.velocity, 1e-10, maxIter) >= maxIter)
    {
        throw std::runtime_error("failed to project initial point to surface");
    }

    ImplicitGeodesicState xEnd(xStart);
    Monitor(xEnd);

    if (length <= 0.) {
        return {xStart, xEnd};
    }

    double l  = 0.;
    double dl = length / static_cast<double>(steps);

    for (size_t k = 0; k < steps; ++k) {
        RungeKutta4(s, xEnd, l, dl);

        if(calcFastSurfaceProjection(s, xEnd.position, xEnd.velocity, 1e-10, maxIter) >= maxIter) {
            throw std::runtime_error("failed to project point to surface during integration");
        }

        Monitor(xEnd);
    }

    AssertEq(
        length,
        l,
        "Total length does not match integrated length",
        1e-10); // TODO this should be flagged, and with high precision.

    return {xStart, xEnd};
}

//==============================================================================
//                      PATH ERROR GRADIENT
//==============================================================================

using GeodesicJacobian  = Eigen::Vector<double, Geodesic::DOF>;

GeodesicJacobian calcDirectionJacobian(Vector3 d, double l, Vector3 axis, const Geodesic::Variation& v)
{
    AssertEq(d.dot(d), 1., "direction must be unit");

    Vector3 y = axis - d * d.dot(axis);
    y /= l;
    return y.transpose() * v;
}

GeodesicJacobian calcPathErrorJacobian(
    Vector3 e,
    double l,
    Vector3 axis,
    const Geodesic::Variation& v,
    const Geodesic::Variation& w,
    bool invertV = false)
{
    GeodesicJacobian jacobian = calcDirectionJacobian(e, l, axis, v) * (invertV ? -1. : 1.);
    jacobian += axis.cross(e).transpose() * w;
    return jacobian;
}

//==============================================================================
//                      ACCURATE SURFACE PROJECTION
//==============================================================================

double calcMaxAlignmentError(double angleDeg)
{
    return std::abs(1. - cos(angleDeg / 180. * M_PI));
}

template <typename VECTOR>
double calcInfNorm(const typename std::remove_reference<VECTOR>::type& vec)
{
    double maxError = 0.;
    const auto n    = static_cast<size_t>(vec.rows());
    for (size_t i = 0; i < n; ++i) {
        maxError = std::max(maxError, std::abs(vec[i]));
    }
    return maxError;
}

template <typename VECTOR>
double calcScaledToFit(typename std::remove_reference<VECTOR>::type& vec, double bound)
{
    const double c = std::abs(bound) / calcInfNorm<VECTOR>(vec);
    if (c < 1.) {
        vec *= c;
    }
    return std::min(c, 1.);
}

template <double>
double calcInfNorm(double x)
{
    return std::abs(x);
}

size_t calcAccurateSurfaceProjection(
    const ImplicitSurface& s,
    Vector3 point,
    Trihedron& K_P,
    double eps,
    size_t maxIter)
{
    using Vector2 = Eigen::Vector2d;

    // Initial guess.
    const Vector3& p0 = point;

    size_t iter = 0;

    const double maxCost = calcMaxAlignmentError(10.);
    const double minCost = calcMaxAlignmentError(1.); // Move these bounds

    // TODO move to arguments? However, not need to compute all elements...
    Geodesic::Variation v_P;
    Geodesic::Variation w_P;

    Vector3 pk = K_P.p();;
    Vector3 tk = K_P.t();;

    for (; iter < maxIter; ++iter) {
        // Project directly to surface.
        iter += calcFastSurfaceProjection(s, pk, tk, eps / 10., maxIter - iter);

        // Now that the point lies on the surface, compute the error and
        // gradient.
        calcGeodesicBoundaryState(s, {pk, tk}, false, K_P, v_P, w_P);
        pk = K_P.p();
        tk = K_P.t();

        // Distance to original point.
        const double l = (p0 - K_P.p()).norm();

        if (std::abs(l) < eps) {
            break;
        }

        // Error vector from surface point to oginial point.
        const Vector3 e = (p0 - K_P.p()) / l;

        const double cosAngle = e.dot(K_P.n());

        if (sign(cosAngle) < 0.) {
            // Point is below surface, so we stop. TODO does that make sense? or
            // should we continue?
            break;
        }

        // The costfunction to minimize.
        double cost = 1. - cosAngle;

        // Stop if the costfunction is small enough.
        if (std::abs(cost) < minCost) {
            break;
        }

        // Compute gradient of cost.
        GeodesicJacobian g = -calcPathErrorJacobian(e, l, K_P.n(), v_P, w_P, true);
        Vector2 df{g[0], g[1]}; // TODO no need to compute entire jacobian.

        // Compute step to minimize the cost.
        const double weight = std::min(1. / (1e-2 + cost), 1.);
        cost                = std::min(cost, maxCost);

        Vector2 step = df * cost / df.dot(df) * weight;

        pk -= K_P.t() * step[0] + K_P.b() * step[1];
    }

    return iter;
}

// TODO this point init is annoying
size_t ImplicitSurface::calcAccurateLocalSurfaceProjectionImpl(
    Vector3 pointInit,
    Trihedron& K,
    double eps,
    size_t maxIter) const
{
    size_t iter = calcAccurateSurfaceProjection(
        *this,
        pointInit, K,
        eps,
        maxIter);
    return iter;
}

size_t Surface::calcAccurateLocalSurfaceProjection(
    Vector3 pointInit,
    Trihedron& K,
    double eps,
    size_t maxIter) const
{
    return calcAccurateLocalSurfaceProjectionImpl(
        std::move(pointInit),
        K,
        eps,
        maxIter);
}

size_t calcTouchdown(
    const Surface& s,
    Trihedron& K,
    Vector3 prev,
    Vector3 next,
    double eps,
    size_t maxIter)
{
    for (size_t iter = 0; iter < maxIter; ++iter) {
        const Vector3 pl = calcPointOnLineNearPoint(prev, next, K.p());
        K = Trihedron::FromPointAndTangentGuessAndNormal(K.p(), K.p() - prev, K.n());
        iter += s.calcAccurateLocalSurfaceProjection(
            pl,
            K,
            eps,
            maxIter - iter);

        // Detect touchdown.
        const Vector3 d    = (pl - K.p());
        const double dDotN = d.dot(K.n());
        if (dDotN < 0.) {
            return iter;
        }

        const double minCost  = calcMaxAlignmentError(5.); // Move these bounds
        const double cosAngle = dDotN / d.norm();
        const double cost     = 1. - cosAngle;
        if (cost < minCost) {
            return iter;
        }
    }

    return maxIter;
}

//==============================================================================
//                      GEODESIC STATUS FLAGS
//==============================================================================

namespace
{
using GS = Geodesic::Status;

GS isPrevOrNextLineSegmentInsideSurface(
    const Surface& surface,
    Vector3 prevPoint,
    Vector3 nextPoint)
{
    GS a = surface.isAboveSurface(
               std::move(prevPoint),
               Surface::MIN_DIST_FROM_SURF)
               ? GS::Ok
               : GS::PrevLineSegmentInsideSurface;

    GS b = surface.isAboveSurface(
               std::move(nextPoint),
               Surface::MIN_DIST_FROM_SURF)
               ? GS::Ok
               : GS::NextLineSegmentInsideSurface;

    return a | b;
}

GS calcLiftoff(
    Trihedron* begin,
    Trihedron* end,
    Vector3 prevPoint,
    Vector3 nextPoint)
{
    auto Liftoff = [&](const Trihedron& K,
                       Vector3 point) -> bool {
        return K.n().dot(point - K.p()) > 0.;
    };

    bool liftoff = true;
    for (Trihedron* it = begin;
         it != end && (liftoff &= Liftoff(*it, prevPoint));
         ++it) {
    }
    for (Trihedron* it = end;
         begin != it && (liftoff &= Liftoff(*(it - 1), nextPoint));
         --it) {
    }

    return liftoff ? GS::LiftOff : GS::Ok;
}

bool isActive(Geodesic::Status s) {
    return s == Geodesic::Status::Ok;
}

size_t countActive(const std::vector<Geodesic>& segments)
{
    size_t count = 0;
    for (const Geodesic& s : segments) {
        if (!isActive(s.status)) {
            continue;
        }
        ++count;
    }
    return count;
}

ptrdiff_t findPrevSegmentIndex(
    const std::vector<Geodesic>& segments,
    ptrdiff_t idx)
{
    ptrdiff_t prev = -1;
    for (ptrdiff_t i = 0; i < idx; ++i) {
        if (isActive(segments.at(i).status)) {
            prev = i;
        }
    }
    return prev;
}

ptrdiff_t findNextSegmentIndex(
    const std::vector<Geodesic>& segments,
    ptrdiff_t idx)
{
    for (ptrdiff_t i = idx + 1; i < static_cast<ptrdiff_t>(segments.size());
         ++i) {
        if (isActive(segments.at(i).status)) {
            return i;
        }
    }
    return -1;
}

Vector3 findPrevSegmentEndPoint(
    const Vector3& pathStart,
    const std::vector<Geodesic>& segments,
    ptrdiff_t idx)
{
    ptrdiff_t prev = findPrevSegmentIndex(segments, idx);
    return prev < 0 ? pathStart : segments.at(prev).K_Q.p();
}

Vector3 findNextSegmentStartPoint(
    const Vector3& pathEnd,
    const std::vector<Geodesic>& segments,
    ptrdiff_t idx)
{
    ptrdiff_t next = findNextSegmentIndex(segments, idx);
    return next < 0 ? pathEnd : segments.at(next).K_P.p();
}

} // namespace

//==============================================================================
//                      WRAPPING STATUS FLAGS
//==============================================================================

namespace
{
using WS = WrappingPath::Status;

void setStatusFlag(WS& current, WS flag, bool value = true)
{
    if (value) {
        current = current | flag;
    } else {
        current = current & ~flag;
    }
}

} // namespace

//==============================================================================
//                      SURFACE
//==============================================================================

bool Surface::isAboveSurface(Vector3 point, double bound) const
{
    return isAboveSurfaceImpl(std::move(point), bound);
}

void updGeodesicStatus(
    const Surface& s,
    Geodesic& geodesic,
    Vector3 prev,
    Vector3 next)
{
    geodesic.status |= isPrevOrNextLineSegmentInsideSurface(s, prev, next);

    if (geodesic.length < 0.) {
        geodesic.status |= Geodesic::Status::NegativeLength;
        geodesic.status |= Geodesic::Status::LiftOff;
        geodesic.length = 0.;
    }
    geodesic.status |= calcLiftoff(
        &*geodesic.samples.begin(),
        &*geodesic.samples.end(),
        prev,
        next);

    if (geodesic.status & Geodesic::Status::LiftOff) {
        geodesic.length = 0.;
        geodesic.samples.clear();
    }

    if (geodesic.samples.empty()) {
        size_t maxIter = 10;
        if (calcTouchdown(
                s,
                geodesic.K_P,
                prev,
                next,
                1e-3,
                maxIter) == maxIter) {
            geodesic.status |= Geodesic::Status::TouchDownFailed;
        }
    }
}

void Surface::calcGeodesic(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length,
    Geodesic& geodesic) const
{
    Vector3 p0 = calcPointInLocal(_transform, std::move(initPosition));
    Vector3 v0 = calcVectorInLocal(_transform, std::move(initVelocity));

    geodesic.samples.clear();
    calcLocalGeodesicImpl(p0, v0, length, geodesic);

    calcGeodesicInGlobal(_transform, geodesic);
}

void Surface::calcGeodesic(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length,
    Vector3 pointBefore,
    Vector3 pointAfter,
    Geodesic& geodesic) const
{
    Vector3 p0 = calcPointInLocal(_transform, std::move(initPosition));
    Vector3 v0 = calcVectorInLocal(_transform, std::move(initVelocity));

    Vector3 prev = calcPointInLocal(_transform, std::move(pointBefore));
    Vector3 next = calcPointInLocal(_transform, std::move(pointAfter));

    geodesic.status = Geodesic::Status::Ok;
    geodesic.samples.clear();
    try {
        calcLocalGeodesicImpl(p0, v0, length, geodesic);
    } catch (const std::exception&) {
        geodesic.status |= Geodesic::Status::IntegratorFailed;
        geodesic.samples.clear();
        geodesic.length = 0.;
    }

    // Reset status flags.
    updGeodesicStatus(*this, geodesic, prev, next); // TODO Flip the order.

    calcGeodesicInGlobal(_transform, geodesic);
}

const Transf& Surface::getOffsetFrame() const
{
    return _transform;
}

void Surface::setLocalPathStartGuess(Vector3 pathStartGuess)
{
    _pathLocalStartGuess = std::move(pathStartGuess);
}

Vector3 Surface::getPathStartGuess() const
{
    return calcPointInGround(_transform, _pathLocalStartGuess);
}

Vector3 Surface::calcSurfaceNormal(Vector3 point) const
{
    return calcLocalSurfaceNormalImpl(calcPointInLocal(_transform, std::move(point)));
}

double Surface::calcNormalCurvature(Vector3 point, Vector3 tangent) const
{
    return calcLocalNormalCurvatureImpl(
            calcPointInLocal(_transform, std::move(point)),
            calcVectorInLocal(_transform, std::move(tangent)));
}

double Surface::calcGeodesicTorsion(Vector3 point, Vector3 tangent) const
{
    return calcLocalGeodesicTorsionImpl(
            calcPointInLocal(_transform, std::move(point)),
            calcVectorInLocal(_transform, std::move(tangent)));
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
    Geodesic& geodesic) const
{
    std::function<void(const ImplicitGeodesicState&)> Monitor =
        [&](const ImplicitGeodesicState& q) {
            geodesic.samples.emplace_back(
                    calcTrihedron(*this, q));
        };

    std::pair<ImplicitGeodesicState, ImplicitGeodesicState> out =
        calcLocalImplicitGeodesic(
            *this,
            initPosition,
            initVelocity,
            length,
            _integratorSteps,
            Monitor);

    calcGeodesicBoundaryState(*this, out.first, false, geodesic.K_P, geodesic.v_P, geodesic.w_P);
    calcGeodesicBoundaryState(*this, out.second, true, geodesic.K_Q, geodesic.v_Q, geodesic.w_Q);
    geodesic.length = length;
}

Vector3 ImplicitSurface::calcLocalSurfaceNormalImpl(Vector3 point) const
{
    return ::calcSurfaceNormal(*this, point);
}

double ImplicitSurface::calcLocalGeodesicTorsionImpl(Vector3 point, Vector3 tangent) const
{
    return ::calcGeodesicTorsion(*this, std::move(point), std::move(tangent));
}

double ImplicitSurface::calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const
{
    return ::calcNormalCurvature(*this, std::move(point), std::move(tangent));
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

void AnalyticSphereSurface::calcLocalGeodesicImpl(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length,
    Geodesic& geodesic) const
{
    const double r     = _radius;
    const double angle = length / r;

    const double norm =  initPosition.norm();
    if (norm < 1e-13) {
        throw std::runtime_error("Error: initial position at origin.");
    }

    Trihedron& K_P = geodesic.K_P;
    Geodesic::Variation& v_P = geodesic.v_P;
    Geodesic::Variation& w_P = geodesic.w_P;

    K_P = Trihedron::FromPointAndTangentGuessAndNormal(initPosition / norm * r, initVelocity, initPosition);

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
    Trihedron& K_Q = geodesic.K_Q;
    Geodesic::Variation& v_Q = geodesic.v_Q;
    Geodesic::Variation& w_Q = geodesic.w_Q;

    // Integration is a rotation over the axis by the angle.
    const Vector3 axis = -K_P.b(); // axis is negative of binormal
    const Rotation dq{Eigen::AngleAxisd(angle, axis)};

    // Final frame: Rotate the input of the initial frame.
    K_Q = dq * K_P;

    // For a sphere the rotation of the initial frame directly rotates the final
    // frame:
    w_Q = w_P;
    w_Q.col(3) = w_Q.col(0);

    // End frame position variation: dp = w x n * r
    for (size_t i = 0; i < 4; ++i) {
        v_Q.col(i) = w_Q.col(i).cross(K_Q.n()) * r;
    }

    // TODO seems a waste...
    size_t nSamples = 10;
    for (size_t i = 0; i < nSamples; ++i) {
        const double angle_i =
            angle * static_cast<double>(i) / static_cast<double>(nSamples);
        const Rotation dq{Eigen::AngleAxisd(angle_i, axis)};
        geodesic.samples.emplace_back(dq * K_P);
    }

    geodesic.length = length;
}

bool AnalyticSphereSurface::isAboveSurfaceImpl(Vector3 point, double bound)
    const
{
    return point.dot(point) - (_radius + bound) * (_radius + bound) > 0.;
}

size_t AnalyticSphereSurface::calcAccurateLocalSurfaceProjectionImpl(
    Vector3,
    Trihedron& K,
    double,
    size_t) const
{
    K = Trihedron::FromPointAndTangentGuessAndNormal(K.p() / K.p().norm() * _radius, K.t(), K.p());
    return 0;
}

double AnalyticSphereSurface::calcLocalNormalCurvatureImpl(Vector3, Vector3) const
{
    return -1. / _radius;
}

Vector3 AnalyticSphereSurface::calcLocalSurfaceNormalImpl(Vector3 point) const
{
    return point / point.norm();
}

double AnalyticSphereSurface::calcLocalGeodesicTorsionImpl(Vector3, Vector3) const
{
    return 0.;
}

//==============================================================================
//                      IMPLICIT CYLINDER SURFACE
//==============================================================================

double ImplicitCylinderSurface::calcSurfaceConstraintImpl(Vector3 p) const
{
    return p.x() * p.x() + p.y() * p.y() - _radius * _radius;
}

Vector3 ImplicitCylinderSurface::calcSurfaceConstraintGradientImpl(Vector3 p) const
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
    Geodesic& geodesic) const
{
    const double r = _radius;
    const double l = length;

    // Cylinder axis assumed to be aligned with z-axis.
    const Vector3 x{1., 0., 0.};
    const Vector3 y{0., 1., 0.};
    const Vector3 z{0., 0., 1.};

    // Initial darboux frame.
    Trihedron& K_P = geodesic.K_P;
    Geodesic::Variation& v_P = geodesic.v_P;
    Geodesic::Variation& w_P = geodesic.w_P;

    // Normal direction.
    Darboux f_P = Darboux::FromTangenGuessAndNormal(initVelocity, Vector3{initPosition.x(), initPosition.y(), 0.});

    // Initial position on surface.
    const Vector3 p_P = f_P.n() * r + z * initPosition.z();

    K_P = Trihedron(p_P, f_P);

    // Rotation angle between start and end frame about cylinder axis.
    const double alpha = (l / r) * z.cross(K_P.n()).dot(K_P.t());

    // Distance along cylinder axis between start and end frame.
    const double h = l * z.dot(K_P.t());

    AssertEq(alpha * alpha * r * r + h * h, l * l, "(alpha * r)^2 + h^2 = l^2");

    AssertEq(
        (K_P.t().cross(z) * l).norm(),
        std::abs(alpha * r),
        "||t X z || * l = |alpha * r|");
    AssertEq(K_P.t().dot(z) * l, h, "t.T z * l = h");

    // Rotation angle variation to initial direction variation.
    const double dAlpha_dTheta =
        -(l / r) * z.cross(K_P.n()).dot(K_P.n().cross(K_P.t()));

    // Distance along axis variation to initial direction variation.
    const double dh_dTheta = -l * z.dot(K_P.n().cross(K_P.t()));

    // Rotation angle variation to length variation.
    const double dAlpha_dl = (1. / r) * z.cross(K_P.n()).dot(K_P.t());

    // Distance along axis variation to length variation.
    const double dh_dl = z.dot(K_P.t());

    // Start position variation.
    const Vector3 zeros{0., 0., 0.};

    v_P.col(0) = K_P.t();
    v_P.col(1) = K_P.b();
    v_P.col(2) = zeros;
    v_P.col(3) = zeros;

    // Start frame variation.
    w_P.col(0) =  z * K_P.t().dot(z.cross(K_P.n())) / r;
    w_P.col(1) =  z * K_P.t().dot(z) / r;
    w_P.col(2) =  -K_P.n();
    w_P.col(3) =  zeros;

    // Integration of the angular rotation about cylinder axis.
    const Rotation dq{Eigen::AngleAxisd(alpha, z)};

    // Final darboux frame.
    const Vector3 t_Q = dq * K_P.t();
    const Vector3 n_Q = dq * K_P.n();
    Darboux f_Q{t_Q, n_Q, t_Q.cross(n_Q)};

    // Final position.
    const Vector3 p_Q = f_Q.n() * r + z * (p_P.z() + h);

    // Final trihedron.
    Trihedron& K_Q = geodesic.K_Q;
    Geodesic::Variation& v_Q = geodesic.v_Q;
    Geodesic::Variation& w_Q = geodesic.w_Q;

    K_Q = Trihedron(p_Q, f_Q);

    // Final position variation.
    v_Q.col(0) = f_Q.t();
    v_Q.col(1) = f_Q.b();
    v_Q.col(2) = z.cross(p_Q) * dAlpha_dTheta + z * dh_dTheta;
    v_Q.col(3) = z.cross(p_Q) * dAlpha_dl + z * dh_dl;

    w_Q.col(0) = z * f_Q.t().dot(z.cross(f_Q.n())) / r;
    w_Q.col(1) = z * f_Q.t().dot(z) / r;
    w_Q.col(2) = dAlpha_dTheta * z - f_Q.n();
    w_Q.col(3) = dAlpha_dl * z;

    size_t nSamples = 10;
    for (size_t i = 0; i < nSamples; ++i) {
        const double factor =
            static_cast<double>(i) / static_cast<double>(nSamples);
        const double angle_i = alpha * factor;
        const double h_i     = h * factor;
        const Rotation dq{Eigen::AngleAxisd(angle_i, z)};
        const Darboux f = dq * f_P;
        const Vector3 p_i    = dq * K_P.p() + h_i * z;
        geodesic.samples.emplace_back(p_i, f);
    }

    geodesic.length = length;
}

size_t AnalyticCylinderSurface::calcAccurateLocalSurfaceProjectionImpl(
    Vector3 pointInit,
    Trihedron& K,
    double,
    size_t) const
{
    const double x = pointInit.x();
    const double y = pointInit.y();
    const double z = pointInit.z();
    const Darboux f = Darboux::FromTangenGuessAndNormal(K.t(), {x, y, 0.});
    K = Trihedron(Vector3{0., 0., z} + _radius * f.n(), f);
    return 0;
}

bool AnalyticCylinderSurface::isAboveSurfaceImpl(Vector3 point, double bound)
    const
{
    return point.x() * point.x() + point.y() * point.y() -
               (_radius + bound) * (_radius + bound) >
           0.;
}

double AnalyticCylinderSurface::calcLocalNormalCurvatureImpl(Vector3 point, Vector3 tangent) const
{
    const Vector3& p = point;
    const Vector3& t = tangent;
    return - (t.x() * p.y() -t.y() * p.x() ) / _radius;
}

Vector3 AnalyticCylinderSurface::calcLocalSurfaceNormalImpl(Vector3 point) const
{
    const double x = point.x();
    const double y = point.y();
    return Vector3{x, y, 0.} / std::sqrt(x*x + y*y);
}

double AnalyticCylinderSurface::calcLocalGeodesicTorsionImpl(Vector3, Vector3) const
{
    return 0.;
}

//==============================================================================
//                      IMPLICIT TORUS SURFACE
//==============================================================================

double ImplicitTorusSurface::calcSurfaceConstraintImpl(Vector3 position) const
{
    const Vector3& p = position;
    const double x = p.x();
    const double y = p.y();

    const double r = _smallRadius;
    const double R = _bigRadius;

    const double c = (p.dot(p) + R*R - r*r);
    return c*c - 4. * R*R * (x*x + y*y);
}

Vector3 ImplicitTorusSurface::calcSurfaceConstraintGradientImpl(Vector3 position) const
{
    const Vector3& p = position;
    const double x = p.x();
    const double y = p.y();

    const double r = _smallRadius;
    const double R = _bigRadius;

    const double c = (p.dot(p) + R*R - r*r);
    return 4.*c*p - 8. * R*R * Vector3{x, y, 0.};
}

ImplicitSurface::Hessian ImplicitTorusSurface::calcSurfaceConstraintHessianImpl(Vector3 position) const
{
    ImplicitSurface::Hessian hessian;

    const Vector3& p = position;

    const double r = _smallRadius;
    const double R = _bigRadius;

    const double c = (p.dot(p) + R*R - r*r);

    hessian.setIdentity();
    hessian *= 4. * c;

    hessian += 8. * p * p.transpose();

    hessian(0,0) -= 8. * R*R;
    hessian(1,1) -= 8. * R*R;

    return hessian;
}

bool ImplicitTorusSurface::isAboveSurfaceImpl(Vector3 point, double bound) const
{
    const double x = point.x();
    const double y = point.y();
    const double z = point.z();

    const double r = _smallRadius + bound;
    const double R = _bigRadius;

    const double c =  (sqrt(x*x + y*y) - R);
    return c*c + z*z - r*r > 0.;
}

//==============================================================================
//                      PATH CONTINUITY ERROR
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

void PathContinuityError::resize(size_t nSurfaces)
{
    constexpr size_t C = NUMBER_OF_CONSTRAINTS;
    constexpr size_t Q = Geodesic::DOF;
    const size_t n = _nSurfaces = nSurfaces;

    _pathError.resize(n * C);
    _solverError.resize(n * Q);
    _pathErrorJacobian.resize(n * C, n * Q);

    _costP.resize(n * Q, n * Q);
    _costQ.resize(n * Q, n * Q);
    _costL.resize(n * Q, n * Q);
    _vecL.resize(n * Q);

    _matSmall.resize(n * Q, n * Q);
    _vecSmall.resize(n * Q);
    _solveSmall.resize(_matSmall.cols());

    _mat.resize(n * (Q + C), n * (Q + C));
    _vec.resize(n * (Q + C));
    _solve.resize(_mat.cols());

    _pathCorrections.resize(n * Q);

    _length = 0.;
    _lengthJacobian.resize(n * Q);

    // Reset values.
    _pathCorrections.fill(NAN);
    _pathError.fill(NAN);
    _solverError.fill(NAN);
    _vec.fill(NAN);
    _vecSmall.fill(NAN);
    _solve.fill(NAN);
    _solveSmall.fill(NAN);
    _lengthJacobian.fill(0.);

    _costP.fill(0.);
    _costQ.fill(0.);
    _costL.fill(NAN);
    _vecL.fill(NAN);

    // Fill with zeros because it is sparse.
    _pathErrorJacobian.fill(0.);
    _mat.fill(0.);
    _matSmall.fill(0.);
}

Eigen::VectorXd& PathContinuityError::updPathError()
{
    return _pathError;
}
Eigen::MatrixXd& PathContinuityError::updPathErrorJacobian()
{
    return _pathErrorJacobian;
}

double PathContinuityError::calcMaxPathError() const
{
    double maxPathError = 0.;
    for (int i = 0; i < _pathError.rows(); ++i) {
        maxPathError = std::max(maxPathError, std::abs(_pathError[i]));
    }
    return maxPathError;
}

double PathContinuityError::calcMaxCorrectionStep() const
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

bool PathContinuityError::calcPathCorrection()
{
    const bool augmented = false;

    const double weight = calcMaxPathError() / 2. + 1e-3;

    const double weightLength = 0. / _length * 1e-4;
    /* std::cout << "weight =" << weight << std::endl; */

    // TODO Clamp the path error?
    calcScaledToFit<Eigen::VectorXd>(_pathError, 10.);
    /* std::cout << "\n"; */
    /* std::cout << "_pathError =" << _pathError.transpose() << std::endl; */
    /* std::cout << "_pathErrorJacobian =\n" << _pathErrorJacobian << std::endl;
     */

    /* std::cout << "_lengthJacobian =" << _lengthJacobian.transpose() <<
     * std::endl; */
    /* std::cout << "_length =" << _length << std::endl; */

    /* _matSmall = _pathErrorJacobian * _pathErrorJacobian.transpose(); */
    /* _vecSmall =  _matSmall.colPivHouseholderQr().solve(_pathError); */

    const size_t n     = _nSurfaces;
    constexpr size_t Q = Geodesic::DOF;
    constexpr size_t C = NUMBER_OF_CONSTRAINTS;

    /* _mat *= weight; */

    /* _matSmall.fill(0.); */
    /* _matSmall = _pathErrorJacobian * _pathErrorJacobian.transpose() * weight;
     */

    if (augmented) {
        // Set cost function.
        {
            /* _mat.topLeftCorner(n * Q, n * Q) *= weight; */
            _mat.topLeftCorner(n * Q, n * Q) +=
                _lengthJacobian * _lengthJacobian.transpose() * weightLength;
            /* _vec.topRows(n * Q).fill(0.); */
            _vec.topRows(n * Q) = _lengthJacobian * _length * weightLength;
        }

        // Set constraints
        {
            _mat.bottomLeftCorner(n * C, n * Q) = _pathErrorJacobian;
            _mat.topRightCorner(n * Q, n * C) = _pathErrorJacobian.transpose();
            _vec.bottomRows(n * C)            = _pathError;
        }

        _solve           = -_mat.colPivHouseholderQr().solve(_vec);
        _pathCorrections = _solve.topRows(n * Q);
    } else {
        /* for (size_t i = 0; i < n * Q; ++i) { */
        /*     _matSmall(i,i) = weight; */
        /* } */
        _costL = _lengthJacobian * _lengthJacobian.transpose();
        _vecL  = _lengthJacobian * _length;

        _matSmall = _costP + _costQ;
        _matSmall *= weight;

        /* _matSmall += _costL * weightLength; */

        _matSmall += _pathErrorJacobian.transpose() * _pathErrorJacobian;

        _vecSmall = _pathErrorJacobian.transpose() * _pathError;
        /* const bool singular = ((calcInfNorm(_vecSmall) < bnd / 100.) &&
         * (calcInfNorm(_pathError) > bnd)); */
        /* _vecSmall += _vecL * weightLength; */

        _solveSmall = -_matSmall.colPivHouseholderQr().solve(_vecSmall);

        _pathCorrections = _solveSmall;
    }

    /* _mat += _pathErrorJacobian.transpose() * _pathErrorJacobian; */
    /* _vec = _pathErrorJacobian.transpose() * _pathError; */

    /* _mat = (_pathErrorJacobian *
     * _pathErrorJacobian.transpose()).colPivHouseholderQr(). */

    /* std::cout << "_vec =" << _vec.transpose() << std::endl; */
    /* std::cout << "_mat =\n" << _mat << std::endl; */

    // Compute singular value decomposition. TODO or other decomposition?
    /* constexpr bool useSvd = false; */
    /* if (useSvd) { */
    /*     _svd.compute(_mat, Eigen::ComputeThinU | Eigen::ComputeThinV); */
    /*     _pathCorrections = -_svd.solve(_vec); */
    /* } else { */
    /*     _pathCorrections = -_mat.colPivHouseholderQr().solve(_vec); */
    /* } */
    /* std::cout << "solverCorr = " << _pathCorrections.transpose() <<
     * std::endl; */

    /* _solverError = _mat * _pathCorrections + _vec; */
    /* std::cout << "_solverError = " << _solverError.transpose() << std::endl;
     */

    /* throw std::runtime_error("stop"); */
    /* return _solverError.norm() < _eps; */
    return true;
}

const Geodesic::Correction* PathContinuityError::begin() const
{
    static_assert(sizeof(Geodesic::Correction) == sizeof(double) * Geodesic::DOF);
    return reinterpret_cast<const Geodesic::Correction*>(&_pathCorrections(0));
}

const Geodesic::Correction* PathContinuityError::end() const
{
    return begin() + _nSurfaces;
}

//==============================================================================
//                      SOLVING THE WRAPPING PROBLEM
//==============================================================================

std::vector<Geodesic> calcInitWrappingPathGuess(
    const Vector3& pathStart,
    std::function<const Surface*(size_t)>& getSurface)
{
    std::vector<Geodesic> geodesics;
    Vector3 pointBefore = pathStart;
    for (size_t i = 0; getSurface(i); ++i) {
        Vector3 initPositon  = getSurface(i)->getPathStartGuess();
        Vector3 initVelocity = (initPositon - pointBefore);

        // Shoot a zero-length geodesic as initial guess.
        geodesics.push_back({});
        Geodesic& g = geodesics.back();
        getSurface(i)->calcGeodesic(initPositon, initVelocity, 0., g);

        if (g.length != 0. || g.samples.empty()) {
            throw std::runtime_error("Failed to shoot a zero-length geodesic");
        }

        pointBefore = initPositon;
    }
    return geodesics;
}

WrappingPath::WrappingPath(
    Vector3 pathStart,
    Vector3 pathEnd,
    GetSurfaceFn& GetSurface) :
    startPoint(pathStart), endPoint(pathEnd),
    segments(calcInitWrappingPathGuess(startPoint, GetSurface))
{}

void applyNaturalGeodesicVariation(
    Trihedron& K_P,
    const Geodesic::Variation& v_P,
    const Geodesic::Variation& w_P,
    const Geodesic::Correction& correction)
{
    Vector3 v = v_P * correction;
    Vector3 w = w_P * correction;
    Vector3 t = K_P.t() + w.cross(K_P.t());
    Vector3 n = K_P.n() + w.cross(K_P.n());

    K_P = Trihedron(K_P.p() + v, Darboux::FromTangenGuessAndNormal(t, n));
}

void Surface::applyVariation(Geodesic& geodesic, const Geodesic::Correction& correction)
    const
{
    applyNaturalGeodesicVariation(geodesic.K_P, geodesic.v_P, geodesic.w_P, correction);
    geodesic.length += correction[Geodesic::DOF-1];
    calcGeodesic(
        geodesic.K_P.p(),
        geodesic.K_P.t(),
        geodesic.length,
        geodesic);
}

void calcSegmentPathErrorJacobian(
    const Geodesic::Variation* v_Q_prev,
    const Trihedron& K,
    const Geodesic::Variation& v,
    const Geodesic::Variation& w,
    const Geodesic::Variation* v_P_next,
    const Vector3& point,
    Eigen::VectorXd& pathError,
    Eigen::MatrixXd& pathErrorJacobian,
    size_t& row,
    bool isFirst)
{
    constexpr size_t DIM = Geodesic::DOF;

    const double l  = (K.p() - point).norm();
    const Vector3 e = (K.p() - point) / l;

    const size_t col =
        (row / PathContinuityError::NUMBER_OF_CONSTRAINTS) * Geodesic::DOF;

    auto UpdatePathErrorElementAndJacobian = [&](const Vector3& m, double y) {
        pathError[row] = e.dot(m) + y;

        const Vector3 de = (m - e * e.dot(m)) / l;

        for (size_t i = 0; i < Geodesic::DOF; ++i) {

            const Vector3 dm = w.col(i).cross(m);

            pathErrorJacobian(row, col + i) = de.dot(v.col(i)) + e.dot(dm);

            // Check if other end was connected to a geodesic.
            if (v_Q_prev) {
                pathErrorJacobian(row, col - DIM + i) =
                    -de.dot(v_Q_prev->col(i));
            }

            if (v_P_next) {
                pathErrorJacobian(row, col + DIM + i) =
                    -de.dot(v_P_next->col(i));
            }
        }

        ++row;
    };

    UpdatePathErrorElementAndJacobian(K.t(), isFirst ? -1. : 1.);
    UpdatePathErrorElementAndJacobian(K.n(), 0.);
    UpdatePathErrorElementAndJacobian(K.b(), 0.);
}

size_t calcPathErrorJacobian(WrappingPath& path)
{
    size_t nActiveSegments = countActive(path.segments);
    /* std::cout << "nActiveSegments = " << nActiveSegments << "\n"; */

    path.smoothness.resize(nActiveSegments);

    // Active segment count.
    size_t row               = 0;
    size_t rowLengthJacobian = 3;
    /* SegmentIterator end = SegmentIterator::End(path); */
    /* for (SegmentIterator it = SegmentIterator::Begin(path); it != end; ++it)
     */
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(path.segments.size());
         ++idx) {
        const Geodesic& s = path.segments.at(idx);
        if (!isActive(s.status)) {
            continue;
        }

        {
            ptrdiff_t prev = findPrevSegmentIndex(path.segments, idx);

            /* std::cout << "idx = " << idx << "\n"; */
            /* std::cout << "row = " << row << "\n"; */
            /* std::cout << "col = " << row << "\n"; */
            calcSegmentPathErrorJacobian(
                prev < 0 ? nullptr : &path.segments.at(prev).v_Q,
                path.segments.at(idx).K_P,
                path.segments.at(idx).v_P,
                path.segments.at(idx).w_P,
                nullptr,
                prev < 0 ? path.startPoint
                         : path.segments.at(prev).K_Q.p(),
                path.smoothness.updPathError(),
                path.smoothness.updPathErrorJacobian(),
                row,
                true);
        }

        {
            ptrdiff_t next = findNextSegmentIndex(path.segments, idx);
            calcSegmentPathErrorJacobian(
                nullptr,
                path.segments.at(idx).K_Q,
                path.segments.at(idx).v_Q,
                path.segments.at(idx).w_Q,
                next < 0 ? nullptr : &path.segments.at(next).v_P,
                next < 0 ? path.endPoint
                         : path.segments.at(next).K_P.p(),
                path.smoothness.updPathError(),
                path.smoothness.updPathErrorJacobian(),
                row,
                false);
        }

        path.smoothness._length += std::abs(s.length);
        path.smoothness._lengthJacobian(rowLengthJacobian) += sign(s.length);
        rowLengthJacobian += Geodesic::DOF;

        for (size_t i = 0; i < Geodesic::DOF; ++i) {
            for (size_t j = 0; j < Geodesic::DOF; ++j) {
                const size_t r = i + Geodesic::DOF * idx;
                const size_t c = j + Geodesic::DOF * idx;
                path.smoothness._costP(r, c) =
                    s.v_P.col(i).dot(s.v_P.col(j));
                path.smoothness._costQ(r, c) +=
                    s.v_Q.col(i).dot(s.v_Q.col(j));
            }
        }
    }
    return nActiveSegments;
}

size_t WrappingPath::updPath(
    GetSurfaceFn& GetSurface,
    double eps,
    size_t maxIter)
{
    const size_t nSurfaces = segments.size();

    if (nSurfaces == 0) {
        return 0;
    }

    /* std::cout << "START WrapSolver::calcPath\n"; */
    for (size_t loopIter = 0; loopIter < maxIter; ++loopIter) {

        const ptrdiff_t nTouchdown = countActive(segments);

        if (nTouchdown > 0) {
            // Fill the path error jacobian.
            /* std::cout << "Calc Patherror Jacobian" << std::endl; */
            calcPathErrorJacobian(*this);
            /* std::cout << "    ===== ERRR ==== = " <<
             * path.smoothness.calcMaxPathError() << "\n"; */

            if (smoothness.calcMaxPathError() < eps) {
                return loopIter;
            }

            // Process the path errors.
            // TODO handle failing to invert jacobian.
            setStatusFlag(
                status,
                WrappingPath::Status::FailedToInvertJacobian,
                !(smoothness.calcPathCorrection()));

            // Obtain the computed geodesic corrections from the path errors.
            const Geodesic::Correction* corrIt  = smoothness.begin();
            const Geodesic::Correction* corrEnd = smoothness.end();
            if (corrEnd - corrIt != nTouchdown) {
                throw std::runtime_error(
                    "Number of geodesic-corrections not equal to "
                    "number of geodesics");
            }

            // Apply corrections.
            for (Geodesic& s : segments) {
                if (!isActive(s.status)) {
                    continue;
                }
                const Geodesic::Correction correction = *corrIt;

                applyNaturalGeodesicVariation(s.K_P, s.v_P, s.w_P, correction);

                // TODO last field of correction must be lengthening.
                s.length += correction[3];

                ++corrIt;
            }
        }

        /* size_t idx = 0; */
        /* SegmentIterator end = SegmentIterator::End(path); */
        /* for (SegmentIterator it = SegmentIterator::Begin(path); it != end;
         * ++it, ++idx) { */
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(segments.size());
             ++idx) {
            // Shoot a new geodesic.
            /* Geodesic& s = *it->current; */
            const Geodesic& s = segments.at(idx);
            GetSurface(idx)->calcGeodesic(
                s.K_P.p(),
                s.K_P.t(),
                s.length,
                findPrevSegmentEndPoint(startPoint, segments, idx),
                findNextSegmentStartPoint(endPoint, segments, idx),
                segments.at(idx));
        }
    }

    // TODO handle failing to converge.
    /* std::cout << "Exceeded max iterations\n"; */
    return maxIter;
    throw std::runtime_error(
        "failed to find wrapping path: exceeded max number of iterations");
}
