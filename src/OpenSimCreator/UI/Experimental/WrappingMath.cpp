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

static constexpr bool RUNTIME_UNIT_TESTS = true;

//==============================================================================
//                      PRINTING
//==============================================================================

namespace osc
{

std::ostream& operator<<(std::ostream& os, const DarbouxFrame& frame)
{
    return os << "DarbouxFrame{" << "t:" << frame.t.transpose() << ", "
              << "n:" << frame.n.transpose() << ", " << "b:" << frame.b.transpose()
              << "}";
}

std::ostream& operator<<(std::ostream& os, const Geodesic::BoundaryState& x)
{
    // TODO remove indentation from printing.

    os << "t: " << x.frame.t.transpose() << ", ";
    os << "n: " << x.frame.n.transpose() << ", ";
    os << "b: " << x.frame.b.transpose() << ", ";
    os << "r: " << x.position.transpose() << "\n";

    std::string delim = "         v: {";
    for (const Vector3& vi : x.v) {
        os << delim << vi.transpose();
        delim = ", ";
    }
    delim = "}, \n         w: {";
    for (const Vector3& wi : x.w) {
        os << delim << wi.transpose();
        delim = ", ";
    }
    return os << "}";
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
    return os << "}";
}

std::ostream& operator<<(std::ostream& os, const Geodesic& x)
{
    os << "Geodesic{\n";
    os << "    K_P: " << x.start << ",\n";
    os << "    K_Q: " << x.end << ",\n";
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
    double eps = 1e-13)
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
    double eps = 1e-13)
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

bool AssertEq(
    const Vector3& lhs,
    const Vector3& rhs,
    const std::string& msg,
    std::ostream& os,
    double eps = 1e-13)
{
    const bool isOk = (lhs - rhs).norm() < eps;
    if (!isOk) {
        os << "FAILED ASSERT: " << msg << std::endl;
        os << "    lhs = " << lhs.transpose() << std::endl;
        os << "    rhs = " << rhs.transpose() << std::endl;
        os << "    err = " << (lhs - rhs).transpose() << std::endl;
        os << "    bnd = " << eps << std::endl;
    }
    return isOk;
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

void AssertDarbouxFrame(const DarbouxFrame& frame)
{
    auto& t = frame.t;
    auto& n = frame.n;
    auto& b = frame.b;

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

DarbouxFrame::DarbouxFrame(Vector3 surfaceTangent, Vector3 surfaceNormal) :
    t(std::move(surfaceTangent)), n(std::move(surfaceNormal)), b(t.cross(n))
{
    t = t / t.norm();
    n = n / n.norm();
    b = b / b.norm();

    AssertDarbouxFrame(*this);
}

DarbouxFrame::DarbouxFrame(Vector3 tangent, Vector3 normal, Vector3 binormal) :
    t(std::move(tangent)), n(std::move(normal)), b(std::move(binormal))
{
    AssertDarbouxFrame(*this);
}

DarbouxFrame calcDarbouxFromTangentGuessAndNormal(
    Vector3 tangentGuess,
    Vector3 surfaceNormal)
{
    Vector3 n = std::move(surfaceNormal);
    n         = n / n.norm();

    Vector3 t = std::move(tangentGuess);
    t         = t - n * n.dot(t);
    t         = t / t.norm();

    Vector3 b = t.cross(n);

    return {t, n, b};
}

namespace
{
DarbouxFrame operator*(const Rotation& lhs, const DarbouxFrame& rhs)
{
    return {
        lhs * rhs.t,
        lhs * rhs.n,
        lhs * rhs.b,
    };
}
} // namespace

//==============================================================================
//                      TRANSFORM
//==============================================================================

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

void calcDarbouxFrameInGlobal(const Transf& transform, DarbouxFrame& frame)
{
    frame.t = calcVectorInGround(transform, frame.t);
    frame.n = calcVectorInGround(transform, frame.n);
    frame.b = calcVectorInGround(transform, frame.b);
    AssertDarbouxFrame(frame);
}

void calcBoundaryStateInGlobal(
    const Transf& transform,
    Geodesic::BoundaryState& x)
{
    x.position = calcPointInGround(transform, x.position);
    calcDarbouxFrameInGlobal(transform, x.frame);
    for (Vector3& vi : x.v) {
        vi = calcVectorInGround(transform, vi);
    }
    for (Vector3& wi : x.w) {
        wi = calcVectorInGround(transform, wi);
    }
}

void calcGeodesicInGlobal(const Transf& transform, Geodesic& geodesic)
{
    calcBoundaryStateInGlobal(transform, geodesic.start);
    calcBoundaryStateInGlobal(transform, geodesic.end);

    // TODO this is a bit wasteful.
    for (std::pair<Vector3, DarbouxFrame>& knot : geodesic.samples) {
        knot.first = calcPointInGround(transform, knot.first);
        calcDarbouxFrameInGlobal(transform, knot.second);
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

DarbouxFrame calcDarbouxFrame(
    const ImplicitSurface& s,
    const ImplicitGeodesicState& q)
{
    return {q.velocity, calcSurfaceNormal(s, q.position)};
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
    throw std::runtime_error("Failed to project point to surface");
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
Geodesic::BoundaryState calcGeodesicBoundaryState(
    const ImplicitSurface& s,
    const ImplicitGeodesicState& x,
    bool isEnd)
{
    Geodesic::BoundaryState y;
    y.frame = calcDarbouxFrame(s, x);

    y.position = x.position;

    // Implicit surface uses natural geodesic variations:
    //   0. Tangential
    //   1. Binormal
    //   2. InitialDirection
    //   3. Lengthening
    y.v = {
        y.frame.t,
        y.frame.b * x.a,
        y.frame.b * x.r,
        isEnd ? y.frame.t : Vector3{0., 0., 0.},
    };

    const double tau_g   = calcGeodesicTorsion(s, x.position, x.velocity);
    const double kappa_n = calcNormalCurvature(s, x.position, x.velocity);
    const double kappa_a = calcNormalCurvature(s, x.position, y.frame.b);

    y.w = {
        Vector3{tau_g, 0., kappa_n},
        Vector3{
                -x.a * kappa_a,
                -x.aDot,
                -x.a * tau_g,
                },
        Vector3{-x.r * kappa_a, -x.rDot, -x.r * tau_g},
        isEnd ? Vector3{tau_g, 0., kappa_n}
        : Vector3{0., 0., 0.},
    };
    return y;
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
    calcFastSurfaceProjection(s, xStart.position, xStart.velocity);

    ImplicitGeodesicState xEnd(xStart);
    Monitor(xEnd);

    if (length <= 0.) {
        return {xStart, xEnd};
    }

    double l  = 0.;
    double dl = length / static_cast<double>(steps);

    for (size_t k = 0; k < steps; ++k) {
        RungeKutta4(s, xEnd, l, dl);

        calcFastSurfaceProjection(s, xEnd.position, xEnd.velocity);

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

Vector3 calcDirectionDerivative(Vector3 d, double l, Vector3 v)
{
    return (v - d * d.dot(v)) / l;
}

double calcPathErrorDerivative(
    Vector3 e,
    double l,
    Vector3 v,
    Vector3 d,
    Vector3 w,
    DarbouxFrame& f) // TODO write w in inertial frame.
{
    const Vector3 w0 = w[0] * f.t + w[1] * f.n + w[2] * f.b;
    return calcDirectionDerivative(e, l, v).dot(d) + e.dot(w0.cross(d));
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
    Vector3& projectedPoint,
    Vector3& projectedTangent,
    double eps,
    size_t maxIter)
{
    using Vector2 = Eigen::Vector2d;

    Vector3& pk = projectedPoint;
    Vector3& vk = projectedTangent;

    // Initial guess.
    const Vector3& p0 = point;

    size_t iter = 0;

    const double maxCost = calcMaxAlignmentError(10.);
    const double minCost = calcMaxAlignmentError(1.); // Move these bounds
    for (; iter < maxIter; ++iter) {
        // Project directly to surface.
        iter += calcFastSurfaceProjection(s, pk, vk, eps, maxIter - iter);

        // Now that the point lies on the surface, compute the error and
        // gradient.
        Geodesic::BoundaryState K_P =
            calcGeodesicBoundaryState(s, {pk, vk}, false);

        // Distance to original point.
        const double l = (p0 - K_P.position).norm();

        if (std::abs(l) < eps) {
            break;
        }

        // Error vector from surface point to oginial point.
        const Vector3 e = (p0 - K_P.position) / l;

        const double cosAngle = e.dot(K_P.frame.n);

        if (sign(cosAngle) < 0.) {
            // Point is below surface, so we stop. TODO does that make sense? or
            // should we continue?
            return iter;
        }

        // The costfunction to minimize.
        double cost = 1. - e.dot(K_P.frame.n);

        // Stop if the costfunction is small enough.
        if (std::abs(cost) < minCost) {
            return iter;
        }

        // Compute gradient of cost.
        double df_dt = -calcPathErrorDerivative(
            e,
            l,
            -K_P.v.at(0),
            K_P.frame.n,
            K_P.w.at(0),
            K_P.frame);
        double df_dB = -calcPathErrorDerivative(
            e,
            l,
            -K_P.v.at(1),
            K_P.frame.n,
            K_P.w.at(1),
            K_P.frame);
        Vector2 df{df_dt, df_dB};

        // Compute step to minimize the cost.
        const double weight = 1. / (1. + cost);
        cost                = std::min(cost, maxCost);

        Vector2 step = df * cost / df.dot(df) * weight;

        pk -= K_P.frame.t * step[0] + K_P.frame.b * step[1];
    }

    return iter;
}

size_t ImplicitSurface::calcAccurateLocalSurfaceProjectionImpl(
    Vector3 pointInit,
    Vector3& point,
    DarbouxFrame& frame,
    double eps,
    size_t maxIter) const
{
    size_t iter = calcAccurateSurfaceProjection(
        *this,
        pointInit,
        point,
        frame.t,
        eps,
        maxIter);
    frame = calcDarbouxFromTangentGuessAndNormal(
        frame.t,
        calcSurfaceConstraintGradient(point));
    return iter;
}

size_t Surface::calcAccurateLocalSurfaceProjection(
    Vector3 pointInit,
    Vector3& point,
    DarbouxFrame& frame,
    double eps,
    size_t maxIter) const
{
    return calcAccurateLocalSurfaceProjectionImpl(
        std::move(pointInit),
        point,
        frame,
        eps,
        maxIter);
}

size_t calcTouchdown(
    const Surface& s,
    Vector3& point,
    DarbouxFrame& frame,
    Vector3 prev,
    Vector3 next,
    double eps,
    size_t maxIter)
{
    for (size_t iter = 0; iter < maxIter; ++iter) {
        const Vector3 pl = calcPointOnLineNearPoint(prev, next, point);
        frame = calcDarbouxFromTangentGuessAndNormal(point - prev, frame.n);
        iter += s.calcAccurateLocalSurfaceProjection(
            pl,
            point,
            frame,
            eps,
            maxIter - iter);

        // Detect touchdown.
        const Vector3 d    = (pl - point);
        const double dDotN = d.dot(frame.n);
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
    Geodesic::Sample* begin,
    Geodesic::Sample* end,
    Vector3 prevPoint,
    Vector3 nextPoint)
{
    auto Liftoff = [&](const std::pair<Vector3, DarbouxFrame>& frame,
                       Vector3 point) -> bool {
        return frame.second.n.dot(point - frame.first) > 0.;
    };

    bool liftoff = true;
    for (Geodesic::Sample* it = begin;
         it != end && (liftoff &= Liftoff(*it, prevPoint));
         ++it) {
    }
    for (Geodesic::Sample* it = end;
         begin != it && (liftoff &= Liftoff(*(it - 1), nextPoint));
         --it) {
    }

    return liftoff ? GS::LiftOff : GS::Ok;
}
} // namespace

ptrdiff_t findPrevSegmentIndex(
    const std::vector<Geodesic>& segments,
    ptrdiff_t idx)
{
    ptrdiff_t prev = -1;
    for (ptrdiff_t i = 0; i < idx; ++i) {
        if ((segments.at(i).status & Geodesic::Status::LiftOff) == 0) {
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
        if ((segments.at(i).status & Geodesic::Status::LiftOff) == 0) {
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
    return prev < 0 ? pathStart : segments.at(prev).end.position;
}

Vector3 findNextSegmentStartPoint(
    const Vector3& pathEnd,
    const std::vector<Geodesic>& segments,
    ptrdiff_t idx)
{
    ptrdiff_t next = findNextSegmentIndex(segments, idx);
    return next < 0 ? pathEnd : segments.at(next).start.position;
}

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
    if (geodesic.samples.empty()) {
        throw std::runtime_error("no samples in geodesic");
    }
    geodesic.status |= isPrevOrNextLineSegmentInsideSurface(s, prev, next);

    if (geodesic.length < 0.) {
        geodesic.status |= Geodesic::Status::NegativeLength;
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

        size_t maxIter = 10;
        if (calcTouchdown(
                s,
                geodesic.start.position,
                geodesic.start.frame,
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

    geodesic.samples.clear();
    calcLocalGeodesicImpl(p0, v0, length, geodesic);

    // Reset status flags.
    geodesic.status = Geodesic::Status::Ok;
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
                Geodesic::Sample{q.position, calcDarbouxFrame(*this, q)});
        };

    std::pair<ImplicitGeodesicState, ImplicitGeodesicState> out =
        calcLocalImplicitGeodesic(
            *this,
            initPosition,
            initVelocity,
            length,
            _integratorSteps,
            Monitor);

    geodesic.start  = calcGeodesicBoundaryState(*this, out.first, false);
    geodesic.end    = calcGeodesicBoundaryState(*this, out.second, true);
    geodesic.length = length;
}

// TODO DO NOT USE THIS. FOR TESTING ONLY.
double ImplicitSurface::testCalcGeodesicTorsion(Vector3 point, Vector3 tangent) const
{
    return calcGeodesicTorsion(*this, std::move(point), std::move(tangent));
}

// TODO DO NOT USE THIS. FOR TESTING ONLY.
double ImplicitSurface::testCalcNormalCurvature(Vector3 point, Vector3 tangent) const
{
    return calcNormalCurvature(*this, std::move(point), std::move(tangent));
}

// TODO DO NOT USE THIS. FOR TESTING ONLY.
Vector3 ImplicitSurface::testCalcSurfaceNormal(Vector3 point) const
{
    return calcSurfaceNormal(*this, std::move(point));
}

// TODO DO NOT USE THIS. FOR TESTING ONLY.
Vector3 ImplicitSurface::testCalcAcceleration(Vector3 point, Vector3 tangent) const
{
    return calcAcceleration(*this, std::move(point), std::move(tangent));
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

DarbouxFrame testRotationIntegration(DarbouxFrame f_P, double angle)
{
    Vector3 axis = -f_P.b;
    auto f       = [=](const Vector3& frameAxis) -> Vector3 {
        return axis.cross(frameAxis);
    };

    size_t nSteps   = 1000;
    const double dt = angle / static_cast<double>(nSteps);

    DarbouxFrame f_Q = f_P;
    for (size_t i = 0; i < nSteps; ++i) {
        double t = 0.;
        RungeKutta4<Vector3, Vector3>(f_Q.n, t, dt, f);
        RungeKutta4<Vector3, Vector3>(f_Q.t, t, dt, f);
        RungeKutta4<Vector3, Vector3>(f_Q.b, t, dt, f);
    }

    return f_Q;
}

void AnalyticSphereSurface::calcLocalGeodesicImpl(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length,
    Geodesic& geodesic) const
{

    // Make sure we are not changing the variation dimension.
    /* static_assert(Geodesic::BoundaryState.w::size() == 4); */
    /* static_assert(Geodesic::BoundaryState.v.size() == 4); */

    const double r     = _radius;
    const double angle = length / r;

    if (initPosition.norm() < 1e-13) {
        throw std::runtime_error("Error: initial position at origin.");
    }

    // Initial darboux frame.
    DarbouxFrame f_P =
        calcDarbouxFromTangentGuessAndNormal(initVelocity, initPosition);

    // Initial trihedron: K_P
    Geodesic::BoundaryState K_P;

    K_P.w = {
        -f_P.b / r,
        f_P.t / r,
        -f_P.n,
        Vector3{0., 0., 0.}
    };

    // Since position = normal * radius -> p = n * r
    // We have dp = dn * r
    // With: dn = w x n
    for (size_t i = 0; i < 4; ++i) {
        K_P.v.at(i) = K_P.w.at(i).cross(f_P.n) * r;
    }

    K_P.position = f_P.n * r;
    K_P.frame    = f_P;

    // Integrate to final trihedron: K_Q
    Geodesic::BoundaryState K_Q;

    // Integration is a rotation over the axis by the angle.
    const Vector3 axis = -f_P.b; // axis is negative of binormal
    const Rotation dq{Eigen::AngleAxisd(angle, axis)};

    // Final darboux frame: Rotate the input of the initial frame.
    DarbouxFrame f_Q{
        dq * f_P.t,
        dq * f_P.n,
    };

    // Final position is normal times radius.
    K_Q.position = f_Q.n * r;
    K_Q.frame    = f_Q;

    // For a sphere the rotation of the initial frame directly rotates the final
    // frame:
    K_Q.w       = K_P.w;
    K_Q.w.at(3) = -f_Q.b / r;

    // End frame position variation is the same: dp = w x n * r
    for (size_t i = 0; i < 4; ++i) {
        K_Q.v.at(i) = K_Q.w.at(i).cross(f_Q.n) * r;
    }

    // Test the above:
    if (RUNTIME_UNIT_TESTS) {

        DarbouxFrame f_Q0 = testRotationIntegration(f_P, angle);

        const double eps = 1e-9;
        AssertEq(
            f_Q.t,
            f_Q0.t,
            "Numerical integration of tangent direction did not match",
            eps);
        AssertEq(
            f_Q.n,
            f_Q0.n,
            "Numerical integration of normal direction did not match",
            eps);
        AssertEq(
            f_Q.b,
            f_Q0.b,
            "Numerical integration of binormal direction did not match",
            eps);
    }

    if (false) {

        for (size_t i = 0; i < 4; ++i) {
            const double eta = 1e-3;
            const double d   = 1e-3;

            // Apply variation to initial velocity and position.
            /* Vector3 dv_P = f_P.t + K_P.w.at(i).cross(f_P.t) * d; */
            /* Vector3 dp_P = f_P.n + K_P.w.at(i).cross(f_P.n) * d; */

            const Vector3 dAxis = K_P.w.at(i) * d;
            const double dAngle = dAxis.norm();
            const Rotation dW =
                dAngle < 1e-13
                    ? Rotation::Identity()
                    : Rotation(
                          Eigen::AngleAxis<double>(dAngle, dAxis / dAngle));

            const Vector3 dv_P = dW * f_P.t;
            const Vector3 dp_P = dW * f_P.n;

            // Redo the integration, and verify the variation.
            DarbouxFrame df_P(dv_P, dp_P);
            AssertEq(
                r * (df_P.n - f_P.n) / d,
                K_P.v.at(i),
                "Failed to verify initial position variation",
                eta);
            AssertEq(
                (df_P.b - f_P.b) / d,
                K_P.w.at(i).cross(f_P.b),
                "Failed to verify initial binormal variation",
                eta);

            const double dangle = i == 3 ? angle + d : angle;
            DarbouxFrame df_Q   = testRotationIntegration(df_P, dangle);

            AssertEq(
                df_Q.t,
                dW * f_Q.t,
                "Failed to verify final tangent variation",
                eta);
            AssertEq(
                df_Q.n,
                dW * f_Q.n,
                "Failed to verify final normal variation",
                eta);
            AssertEq(
                df_Q.b,
                dW * f_Q.b,
                "Failed to verify final binormal variation",
                eta);

            AssertEq(
                r * (df_Q.n - f_Q.n) / d,
                K_Q.v.at(i),
                "Failed to verify final position variation",
                eta);
            AssertEq(
                (df_Q.t - f_Q.t) / d,
                K_Q.w.at(i).cross(f_Q.t),
                "Failed to verify final tangent variation",
                eta);
            AssertEq(
                (df_Q.n - f_Q.n) / d,
                K_Q.w.at(i).cross(f_Q.n),
                "Failed to verify final normal variation",
                eta);
            AssertEq(
                (df_Q.b - f_Q.b) / d,
                K_Q.w.at(i).cross(f_Q.b),
                "Failed to verify final binormal variation",
                eta);
        }
    }

    auto ApplyAsTransform = [&](const DarbouxFrame& f, Vector3 x) -> Vector3 {
        return Vector3{f.t.dot(x), f.n.dot(x), f.b.dot(x)};
    };

    for (size_t i = 0; i < 4; ++i) {
        K_P.w.at(i) = ApplyAsTransform(K_P.frame, K_P.w.at(i));
        K_Q.w.at(i) = ApplyAsTransform(K_Q.frame, K_Q.w.at(i));
    }

    size_t nSamples = 10;
    for (size_t i = 0; i < nSamples; ++i) {
        const double angle_i =
            angle * static_cast<double>(i) / static_cast<double>(nSamples);
        const Rotation dq{Eigen::AngleAxisd(angle_i, axis)};
        const DarbouxFrame f = dq * K_P.frame;
        geodesic.samples.emplace_back(
            std::pair<Vector3, DarbouxFrame>{dq * K_P.position, f});
    }

    geodesic.start  = std::move(K_P);
    geodesic.end    = std::move(K_Q);
    geodesic.length = length;
}

bool AnalyticSphereSurface::isAboveSurfaceImpl(Vector3 point, double bound)
    const
{
    return point.dot(point) - (_radius + bound) * (_radius + bound) > 0.;
}

size_t AnalyticSphereSurface::calcAccurateLocalSurfaceProjectionImpl(
    Vector3 pointInit,
    Vector3& point,
    DarbouxFrame& frame,
    double,
    size_t) const
{
    point = pointInit / pointInit.norm() * _radius;
    frame = calcDarbouxFromTangentGuessAndNormal(frame.t, point);
    return 0;
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
    Geodesic& geodesic) const
{
    const double r = _radius;
    const double l = length;

    // Cylinder axis assumed to be aligned with z-axis.
    const Vector3 x{1., 0., 0.};
    const Vector3 y{0., 1., 0.};
    const Vector3 z{0., 0., 1.};

    // Initial darboux frame.
    DarbouxFrame f_P = calcDarbouxFromTangentGuessAndNormal(
        initVelocity,
        Vector3{initPosition.x(), initPosition.y(), 0.});

    // Initial position on surface.
    const Vector3 p_P = f_P.n * r + z * initPosition.z();

    // Rotation angle between start and end frame about cylinder axis.
    const double alpha = (l / r) * z.cross(f_P.n).dot(f_P.t);

    // Distance along cylinder axis between start and end frame.
    const double h = l * z.dot(f_P.t);

    AssertEq(alpha * alpha * r * r + h * h, l * l, "(alpha * r)^2 + h^2 = l^2");

    AssertEq(
        (f_P.t.cross(z) * l).norm(),
        std::abs(alpha * r),
        "||t X z || * l = |alpha * r|");
    AssertEq(f_P.t.dot(z) * l, h, "t.T z * l = h");

    // Rotation angle variation to initial direction variation.
    const double dAlpha_dTheta =
        -(l / r) * z.cross(f_P.n).dot(f_P.n.cross(f_P.t));

    // Distance along axis variation to initial direction variation.
    const double dh_dTheta = -l * z.dot(f_P.n.cross(f_P.t));

    // Rotation angle variation to length variation.
    const double dAlpha_dl = (1. / r) * z.cross(f_P.n).dot(f_P.t);

    // Distance along axis variation to length variation.
    const double dh_dl = z.dot(f_P.t);

    // Initial trihedron: K_P
    Geodesic::BoundaryState K_P;

    // Start position variation.
    const Vector3 zeros{0., 0., 0.};
    K_P.v = {
        f_P.t,
        f_P.b,
        zeros,
        zeros,
    };

    // Start frame variation.
    K_P.w = {
        z * f_P.t.dot(z.cross(f_P.n)) / r,
        z * f_P.t.dot(z) / r,
        -f_P.n,
        zeros,
    };

    // Set start frame fields.
    K_P.frame    = f_P;
    K_P.position = p_P;

    // Integration of the angular rotation about cylinder axis.
    const Rotation dq{Eigen::AngleAxisd(alpha, z)};

    // Final darboux frame.
    const Vector3 t_Q = dq * f_P.t;
    const Vector3 n_Q = dq * f_P.n;
    DarbouxFrame f_Q{t_Q, n_Q, t_Q.cross(n_Q)};

    // Final position.
    const Vector3 p_Q = f_Q.n * r + z * (p_P.z() + h);

    // Final trihedron.
    Geodesic::BoundaryState K_Q;

    // Final position variation.
    K_Q.v = {
        f_Q.t,
        f_Q.b,
        z.cross(p_Q) * dAlpha_dTheta + z * dh_dTheta,
        z.cross(p_Q) * dAlpha_dl + z * dh_dl,
    };

    K_Q.w = {
        z * f_Q.t.dot(z.cross(f_Q.n)) / r,
        z * f_Q.t.dot(z) / r,
        dAlpha_dTheta * z - f_Q.n,
        dAlpha_dl * z,
    };

    // Set start frame fields.
    K_Q.frame    = f_Q;
    K_Q.position = p_Q;

    auto ApplyAsTransform = [&](const DarbouxFrame& f, Vector3 x) -> Vector3 {
        return Vector3{f.t.dot(x), f.n.dot(x), f.b.dot(x)};
    };

    for (size_t i = 0; i < 4; ++i) {
        K_P.w.at(i) = ApplyAsTransform(K_P.frame, K_P.w.at(i));
        K_Q.w.at(i) = ApplyAsTransform(K_Q.frame, K_Q.w.at(i));
    }

    size_t nSamples = 10;
    for (size_t i = 0; i < nSamples; ++i) {
        const double factor =
            static_cast<double>(i) / static_cast<double>(nSamples);
        const double angle_i = alpha * factor;
        const double h_i     = h * factor;
        const Rotation dq{Eigen::AngleAxisd(angle_i, z)};
        const DarbouxFrame f = dq * K_P.frame;
        const Vector3 p_i    = dq * K_P.position + h_i * z;
        geodesic.samples.emplace_back(std::pair<Vector3, DarbouxFrame>{p_i, f});
    }

    geodesic.start  = std::move(K_P);
    geodesic.end    = std::move(K_Q);
    geodesic.length = length;
}

size_t AnalyticCylinderSurface::calcAccurateLocalSurfaceProjectionImpl(
    Vector3 pointInit,
    Vector3& point,
    DarbouxFrame& frame,
    double,
    size_t) const
{
    const double x = pointInit.x();
    const double y = pointInit.y();
    const double z = pointInit.z();
    const double c = _radius / std::sqrt(x * x + y * y);
    point          = Vector3{x * c, y * c, z};
    frame          = calcDarbouxFromTangentGuessAndNormal(frame.t, point);
    return 0;
}

bool AnalyticCylinderSurface::isAboveSurfaceImpl(Vector3 point, double bound)
    const
{
    return point.x() * point.x() + point.y() * point.y() -
               (_radius + bound) * (_radius + bound) >
           0.;
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

GeodesicCorrection calcClamped(
    const CorrectionBounds& bnds,
    const GeodesicCorrection& correction)
{
    auto Clamp = [](double bnd, double x) {
        if (std::abs(x) > std::abs(bnd)) {
            return x / std::abs(x) * std::abs(bnd);
        }
        return x;
    };
    const double maxAngle = bnds.maxAngleDegrees / 180. * M_PI;

    return {
        Clamp(bnds.maxRepositioning, correction.at(0)),
        Clamp(bnds.maxRepositioning, correction.at(1)),
        Clamp(maxAngle, correction.at(2)),
        Clamp(bnds.maxLengthening, correction.at(3)),
    };
}

void PathContinuityError::resize(size_t nSurfaces)
{
    constexpr size_t C = NUMBER_OF_CONSTRAINTS;
    constexpr size_t Q = GEODESIC_DIM;
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
    constexpr size_t Q = GEODESIC_DIM;
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

        _matSmall += _costL * weightLength;

        _matSmall += _pathErrorJacobian.transpose() * _pathErrorJacobian;

        _vecSmall = _pathErrorJacobian.transpose() * _pathError;
        /* const bool singular = ((calcInfNorm(_vecSmall) < bnd / 100.) &&
         * (calcInfNorm(_pathError) > bnd)); */
        _vecSmall += _vecL * weightLength;

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

const GeodesicCorrection* PathContinuityError::begin() const
{
    return reinterpret_cast<const GeodesicCorrection*>(&_pathCorrections(0));
}

const GeodesicCorrection* PathContinuityError::end() const
{
    return begin() + _nSurfaces;
}

//==============================================================================
//                      SOLVING THE WRAPPING PROBLEM
//==============================================================================

Vector3 calcNormalDerivative(const DarbouxFrame& frame, const Vector3& rate)
{
    Vector3 nDot = -rate[2] * frame.t + rate[0] * frame.b;

    if (RUNTIME_UNIT_TESTS) {
        const Vector3 w =
            (rate[0] * frame.t + rate[1] * frame.n + rate[2] * frame.b);
        AssertEq(nDot, w.cross(frame.n), "nDot = w.cross(n)");
    }

    return nDot;
}

Vector3 calcBinormalDerivative(const DarbouxFrame& frame, const Vector3& rate)
{
    Vector3 bDot = rate[1] * frame.t - rate[0] * frame.n;

    if (RUNTIME_UNIT_TESTS) {
        const Vector3 w =
            (rate[0] * frame.t + rate[1] * frame.n + rate[2] * frame.b);
        AssertEq(bDot, w.cross(frame.b), "bDot = w.cross(b)");
    }

    return bDot;
}

Vector3 calcTangentDerivative(const DarbouxFrame& frame, const Vector3& rate)
{
    Vector3 tDot = -rate[1] * frame.b + rate[2] * frame.n;

    if (RUNTIME_UNIT_TESTS) {
        const Vector3 w =
            (rate[0] * frame.t + rate[1] * frame.n + rate[2] * frame.b);
        AssertEq(tDot, w.cross(frame.t), "tDot = w.cross(t)");
    }

    return tDot;
}

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
    Geodesic::BoundaryState& geodesicStart,
    const GeodesicCorrection& correction)
{
    // Darboux frame:
    Vector3 t        = geodesicStart.frame.t;
    Vector3 n        = geodesicStart.frame.n;
    const Vector3& b = geodesicStart.frame.b;

    Vector3 dp = correction.at(1) * b + correction.at(0) * t;

    // TODO use start frame to rotate both vectors properly.
    // TODO overload vor ANALYTIC?
    geodesicStart.position += dp;

    /* Vector3 velocity = cos(correction.at(2)) * t + sin(correction.at(2)) * b;
     */
    /* geodesicStart.frame.t = velocity; */

    for (size_t i = 0; i < 4; ++i) {
        const Vector3& w = geodesicStart.w.at(i);
        const double c   = correction.at(i);

        t += calcTangentDerivative(geodesicStart.frame, w * c);
        n += calcNormalDerivative(geodesicStart.frame, w * c);
    }
    geodesicStart.frame = calcDarbouxFromTangentGuessAndNormal(t, n);
}

void Surface::applyVariation(Geodesic& geodesic, const GeodesicCorrection& var)
    const
{
    applyNaturalGeodesicVariation(geodesic.start, var);
    geodesic.length += var.back();
    calcGeodesic(
        geodesic.start.position,
        geodesic.start.frame.t,
        geodesic.length,
        geodesic);
}

size_t countActive(const std::vector<Geodesic>& segments)
{
    size_t count = 0;
    for (const Geodesic& s : segments) {
        /* std::cout << "status = " << s.status << "\n"; */
        /* std::cout << "active = " << ((s.status & Geodesic::Status::LiftOff) >
         * 0) << "\n"; */
        if ((s.status & Geodesic::Status::LiftOff) > 0) {
            continue;
        }
        ++count;
    }
    return count;
}

void calcSegmentPathErrorJacobian(
    const Geodesic::BoundaryState* KQ_prev,
    const Geodesic::BoundaryState& K,
    const Geodesic::BoundaryState* KP_next,
    const Vector3& point,
    Eigen::VectorXd& pathError,
    Eigen::MatrixXd& pathErrorJacobian,
    double& length,
    Eigen::VectorXd& lengthJacobian,
    size_t& row,
    bool isFirst)
{
    constexpr size_t DIM = GEODESIC_DIM;

    const double l  = (K.position - point).norm();
    const Vector3 e = (K.position - point) / l;

    const size_t col =
        (row / PathContinuityError::NUMBER_OF_CONSTRAINTS) * GEODESIC_DIM;

    /* std::cout << "row = " << row << "\n"; */
    /* std::cout << "col = " << col << "\n"; */
    /* std::cout << "prev = " << KQ_prev << "\n"; */
    /* std::cout << "next = " << KP_next << "\n"; */
    // Update the path length.
    length += l;

    // Update path length jacobian.
    for (size_t i = 0; i < GEODESIC_DIM; ++i) {
        /* std::cout << "setting length jacobian index " << col + i  << "\n"; */
        lengthJacobian(col + i) += e.dot(K.v.at(i));
        if (KQ_prev) {
            lengthJacobian(col - DIM + i) += -e.dot(KQ_prev->v.at(i));
        }
        if (KP_next) {
            lengthJacobian(col + DIM + i) += -e.dot(KP_next->v.at(i));
        }
    }
    /* std::cout << "    l =" << l << "\n"; */
    /* std::cout << "    lengthJacobian =" << lengthJacobian.transpose() <<
     * "\n"; */

    auto UpdatePathErrorElementAndJacobian = [&](const Vector3& m, double y) {
        pathError[row] = e.dot(m) + y;

        const Vector3 de = (m - e * e.dot(m)) / l;

        for (size_t i = 0; i < GEODESIC_DIM; ++i) {

            const Vector3& w = K.w.at(i);
            // TODO store in inertial frame.
            const Vector3 dm =
                (w[0] * K.frame.t + w[1] * K.frame.n + w[2] * K.frame.b)
                    .cross(m);

            pathErrorJacobian(row, col + i) = de.dot(K.v.at(i)) + e.dot(dm);

            // Check if other end was connected to a geodesic.
            if (KQ_prev) {
                pathErrorJacobian(row, col - DIM + i) =
                    -de.dot(KQ_prev->v.at(i));
            }

            if (KP_next) {
                pathErrorJacobian(row, col + DIM + i) =
                    -de.dot(KP_next->v.at(i));
            }
        }

        ++row;
        /* std::cout << "    pathError=" << pathError.transpose() << "\n"; */
        /* std::cout << "    pathErrorJcobian=\n" << pathErrorJacobian << "\n";
         */
    };

    UpdatePathErrorElementAndJacobian(K.frame.t, isFirst ? -1. : 1.);
    UpdatePathErrorElementAndJacobian(K.frame.n, 0.);
    UpdatePathErrorElementAndJacobian(K.frame.b, 0.);
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
        if ((s.status & Geodesic::Status::LiftOff) > 0) {
            continue;
        }

        {
            ptrdiff_t prev = findPrevSegmentIndex(path.segments, idx);

            /* std::cout << "idx = " << idx << "\n"; */
            /* std::cout << "row = " << row << "\n"; */
            /* std::cout << "col = " << row << "\n"; */
            calcSegmentPathErrorJacobian(
                prev < 0 ? nullptr : &path.segments.at(prev).end,
                path.segments.at(idx).start,
                nullptr,
                prev < 0 ? path.startPoint
                         : path.segments.at(prev).end.position,
                path.smoothness.updPathError(),
                path.smoothness.updPathErrorJacobian(),
                path.smoothness._length,
                path.smoothness._lengthJacobian,
                row,
                true);
        }

        {
            ptrdiff_t next = findNextSegmentIndex(path.segments, idx);
            calcSegmentPathErrorJacobian(
                nullptr,
                path.segments.at(idx).end,
                next < 0 ? nullptr : &path.segments.at(next).start,
                next < 0 ? path.endPoint
                         : path.segments.at(next).start.position,
                path.smoothness.updPathError(),
                path.smoothness.updPathErrorJacobian(),
                path.smoothness._length,
                path.smoothness._lengthJacobian,
                row,
                false);
        }

        path.smoothness._length += std::abs(s.length);
        path.smoothness._lengthJacobian(rowLengthJacobian) += sign(s.length);
        rowLengthJacobian += GEODESIC_DIM;

        for (size_t i = 0; i < GEODESIC_DIM; ++i) {
            for (size_t j = 0; j < GEODESIC_DIM; ++j) {
                const size_t r = i + GEODESIC_DIM * idx;
                const size_t c = j + GEODESIC_DIM * idx;
                path.smoothness._costP(r, c) =
                    s.start.v.at(i).dot(s.start.v.at(j));
                path.smoothness._costQ(r, c) +=
                    s.end.v.at(i).dot(s.end.v.at(j));
            }
        }
    }
    /* std::cout << "    pathError=" <<
     * path.smoothness.updPathError().transpose() << "\n"; */
    /* std::cout << "    pathErrorJcobian=\n" <<
     * path.smoothness.updPathErrorJacobian() << "\n"; */
    /* throw std::runtime_error("stop"); */
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
                /* std::cout << "   Wrapping path solved in " << loopIter <<
                 * "steps\n"; */
                return loopIter;
            }

            // Process the path errors.
            // TODO handle failing to invert jacobian.
            /* std::cout << "Calc path error correction" << std::endl; */
            setStatusFlag(
                status,
                WrappingPath::Status::FailedToInvertJacobian,
                !(smoothness.calcPathCorrection()));
            /* std::cout << "    ===== CORR ==== = " <<
             * path.smoothness.calcMaxCorrectionStep() << "\n"; */

            // Obtain the computed geodesic corrections from the path errors.
            const GeodesicCorrection* corrIt  = smoothness.begin();
            const GeodesicCorrection* corrEnd = smoothness.end();
            if (corrEnd - corrIt != nTouchdown) {
                throw std::runtime_error(
                    "Number of geodesic-corrections not equal to "
                    "number of geodesics");
            }

            // Apply corrections.
            for (Geodesic& s : segments) {
                if ((s.status & Geodesic::Status::LiftOff) > 0) {
                    continue;
                }
                // TODO remove this?
                /* const GeodesicCorrection correction =
                 * calcClamped(path.smoothness.maxStep, *corrIt); */
                const GeodesicCorrection correction = *corrIt;

                /* std::cout << "s.length_before = " << s.length << "\n"; */
                applyNaturalGeodesicVariation(s.start, correction);

                /* for (double c: correction) { */
                /*     std::cout << "ci = " << c << "\n"; */
                /* } */
                /* std::cout << "s.start_after = " << s.start << "\n"; */

                // TODO last field of correction must be lengthening.
                s.length += correction.at(3);
                if (s.length < 0.) {
                    std::cout << "negative path length: " << s.length << "\n";
                }

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
            /* std::cout << "Shooting s.length = " << s.length << "\n"; */
            GetSurface(idx)->calcGeodesic(
                s.start.position,
                s.start.frame.t,
                s.length,
                findPrevSegmentEndPoint(startPoint, segments, idx),
                findNextSegmentStartPoint(endPoint, segments, idx),
                segments.at(idx));
            /* std::cout << "Returned s.length = " << s.length << "\n"; */
        }
    }

    // TODO handle failing to converge.
    /* std::cout << "Exceeded max iterations\n"; */
    return maxIter;
    throw std::runtime_error(
        "failed to find wrapping path: exceeded max number of iterations");
}

//==============================================================================
//                      SOLVING THE WRAPPING PROBLEM
//==============================================================================

namespace osc
{

std::vector<Vector3> Surface::makeSelfTestPoints() const
{
    std::vector<Vector3> points;
    std::array<double, 5> values = {-3., 3., 1., -1., 0.};
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values.size(); ++j) {
            for (size_t k = 0; k < values.size(); ++k) {
                points.push_back(
                    Vector3{values.at(i), values.at(j), values.at(k)} +
                    Vector3{
                        -1. / 11.,
                        1. / 12.,
                        1. / 13.,
                    });
            }
        }
    }
    // Remove the one filled with zeros.
    points.pop_back();
    return points;
}

std::vector<Vector3> Surface::makeSelfTestVelocities() const
{
    std::vector<Vector3> velocities;
    std::array<double, 3> values = {1., -1., 0.};
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values.size(); ++j) {
            for (size_t k = 0; k < values.size(); ++k) {
                velocities.push_back(
                    Vector3{values.at(i), values.at(j), values.at(k)} +
                    Vector3{
                        1. / 14.,
                        1. / 15.,
                        1. / 16.,
                    });
            }
        }
    }
    return velocities;
}

std::vector<double> Surface::makeSelfTestLengths() const
{
    const double r = selfTestEquivalentRadius();
    std::vector<double> lengths;
    lengths.push_back(0.);
    for (size_t i = 1; i < 4; ++i) {
        lengths.push_back(static_cast<double>(i) * r);
    }
    return lengths;
}

void Surface::doSelfTests(const std::string name, double eps) const
{
    for (Vector3 r_P : makeSelfTestPoints()) {
        for (Vector3 v_P : makeSelfTestVelocities()) {
            for (double l : makeSelfTestLengths()) {
                // TODO Skip tests with parallel position and velocity: will
                // very likely fail.
                if (r_P.cross(v_P).norm() < 1e-9) {
                    continue;
                    ;
                }
                auto transform = getOffsetFrame();
                doSelfTest(
                    name,
                    calcPointInGround(transform, r_P),
                    calcVectorInGround(transform, v_P),
                    l,
                    eps);
            }
        }
    }
}

// Test variation effects on start and end frames.
void Surface::doSelfTest(
    const std::string name,
    Vector3 r_P,
    Vector3 v_P,
    double l,
    double eps,
    double delta) const
{

    // Shoot a zero length geodesic.
    Geodesic gZero;
    calcGeodesic(r_P, v_P, l, gZero);

    // To check the local behavior of the geodesic variation, we apply a
    // small variation to the start point, and see the effect on the
    // geodesic.

    // For debugging.
    std::string msg;

    bool allTestsPassed = true;
    std::ostringstream errs;
    for (size_t i = 0; i < 4; ++i) {
        GeodesicCorrection c{0., 0., 0., 0.};
        c.at(i) = delta;

        Geodesic gOne;
        {
            Geodesic::BoundaryState dK_P = gZero.start;
            applyNaturalGeodesicVariation(dK_P, c);

            // Shoot a new geodesic with the applied variation.
            double dl =
                i == 3 ? c.at(i) + l : l; // TODO encode this in the struct.
            calcGeodesic(dK_P.position, dK_P.frame.t, dl, gOne);
        }

        std::ostringstream os;
        os << "testing variation = ";
        os << "{" << c.at(0) << "," << c.at(1) << "," << c.at(2) << ","
           << c.at(3) << "}";
        os << " with l = " << l;

        {
            const Geodesic::BoundaryState K0 = gZero.start;
            const Geodesic::BoundaryState K1 = gOne.start;

            const Vector3 dp = K0.v.at(i);

            const Vector3 dt = calcTangentDerivative(K0.frame, K0.w.at(i));
            const Vector3 dn = calcNormalDerivative(K0.frame, K0.w.at(i));
            const Vector3 db = calcBinormalDerivative(K0.frame, K0.w.at(i));

            allTestsPassed &= AssertEq(
                (K1.position - K0.position) / delta,
                dp,
                name + ": Failed start position variation " + os.str(),
                errs,
                eps);

            allTestsPassed &= AssertEq(
                (K1.frame.t - K0.frame.t) / delta,
                dt,
                name + ": Failed start tangent variation  " + os.str(),
                errs,
                eps);
            allTestsPassed &= AssertEq(
                (K1.frame.n - K0.frame.n) / delta,
                dn,
                name + ": Failed start normal variation   " + os.str(),
                errs,
                eps);
            allTestsPassed &= AssertEq(
                (K1.frame.b - K0.frame.b) / delta,
                db,
                name + ": Failed start binormal variation " + os.str(),
                errs,
                eps);
        }

        {
            const Geodesic::BoundaryState K0 = gZero.end;
            const Geodesic::BoundaryState K1 = gOne.end;

            const Vector3 dp = K0.v.at(i);

            const Vector3 dt = calcTangentDerivative(K0.frame, K0.w.at(i));
            const Vector3 dn = calcNormalDerivative(K0.frame, K0.w.at(i));
            const Vector3 db = calcBinormalDerivative(K0.frame, K0.w.at(i));

            allTestsPassed &= AssertEq(
                (K1.position - K0.position) / delta,
                dp,
                name + ": Failed end position variation" + os.str(),
                errs,
                eps);

            allTestsPassed &= AssertEq(
                (K1.frame.t - K0.frame.t) / delta,
                dt,
                name + ": Failed end tangent variation " + os.str(),
                errs,
                eps);
            allTestsPassed &= AssertEq(
                (K1.frame.n - K0.frame.n) / delta,
                dn,
                name + ": Failed end normal variation  " + os.str(),
                errs,
                eps);
            allTestsPassed &= AssertEq(
                (K1.frame.b - K0.frame.b) / delta,
                db,
                name + ": Failed end binormal variation" + os.str(),
                errs,
                eps);
        }
    }
    if (!allTestsPassed) {
        throw std::runtime_error(errs.str());
    }
}

bool WrappingTester(
    const WrappingPath& path,
    WrappingPath::GetSurfaceFn& GetSurface,
    std::ostream& os,
    double d,
    double eps)
{
    WrappingPath pathZero = path;
    calcPathErrorJacobian(pathZero);

    const size_t n = path.segments.size();
    os << "Start test for path error jacobian:\n";
    bool success = true;
    for (size_t i = 0; i < n; ++i) {
        os << "    Start testing Surface " << i << "\n";
        for (size_t j = 0; j < 9; ++j) {
            WrappingPath pathOne = path;

            const Surface* surface = GetSurface(i);

            GeodesicCorrection correction{0., 0., 0., 0.};

            Eigen::VectorXd correctionVector(n * 4);
            correctionVector.fill(0.);

            if (j < 8) {
                correction.at(j % 4)              = (j < 4) ? d : -d;
                correctionVector[i * 4 + (j % 4)] = (j < 4) ? d : -d;
            }
            os << "        d" << i << " = " << correctionVector.transpose()
               << "\n";

            Geodesic::BoundaryState start = pathOne.segments.at(i).start;
            applyNaturalGeodesicVariation(start, correction);

            const double length =
                pathOne.segments.at(i).length + correction.at(3);
            surface->calcGeodesic(
                start.position,
                start.frame.t,
                length,
                pathOne.segments.at(i));

            calcPathErrorJacobian(pathOne);

            Eigen::VectorXd dErrExpected =
                pathZero.smoothness._pathErrorJacobian * correctionVector;

            Eigen::VectorXd dErr =
                pathOne.smoothness._pathError - pathZero.smoothness._pathError;

            os << "        dErrExp" << i << " = "
               << dErrExpected.transpose() / d << "\n";
            os << "        dErr   " << i << " = " << dErr.transpose() / d
               << "\n";
            os << "        ErrZero" << i << " = "
               << pathZero.smoothness._pathError.transpose() << "\n";
            os << "        ErrOne" << i << "  = "
               << pathOne.smoothness._pathError.transpose() << "\n";

            for (int k = 0; k < dErr.rows(); ++k) {
                if (std::abs(dErrExpected[k] / d - dErr[k] / d) > eps) {
                    os << "    FAILED TEST FOR SURFACE " << i
                       << " with d = " << correctionVector.transpose() << "\n";
                    success = false;
                    break;
                }
            }
        }
    }
    return success;
}

} // namespace osc
