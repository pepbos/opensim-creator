#include "WrappingMath.h"

#include "oscar/Utils/Assertions.h"
#include <Eigen/src/Core/util/Constants.h>
#include <iostream>
#include <sstream>
#include <string>

using namespace osc;

static constexpr bool RUNTIME_UNIT_TESTS = true;

//==============================================================================
//                      PRINTING
//==============================================================================
namespace
{

    struct Print3
    {
        Vector3 x;
    };

    std::ostream& operator<<(std::ostream& os, Print3 p)
    {
        os << "[";
        std::string delim;
        for (size_t r = 0; r < 3; ++r) {
            os << delim << p.x(r);
            delim = ", ";
        }
        return os << "]";
    }

    /* std::ostream& operator<<(std::ostream& os, const DarbouxFrame& frame) */
    /* { */
    /*     return os << "DarbouxFrame{" */
    /*               << "t:" << Print3{frame.t} << ", " */
    /*               << "n:" << Print3{frame.n} << ", " */
    /*               << "b:" << Print3{frame.b} << "}"; */
    /* } */

    /* std::ostream& operator<<(std::ostream& os, const Geodesic::BoundaryState&
     * x) */
    /* { */
    /*     os << "GeodesicState{" */
    /*           "frame:" */
    /*        << x.frame << "position: " << Print3{x.position}; */
    /*     std::string delim = ", v: {"; */
    /*     for (const Vector3& vi : x.v) { */
    /*         os << delim << Print3{vi}; */
    /*         delim = ", "; */
    /*     } */
    /*     delim = "}, w: {"; */
    /*     for (const Vector3& wi : x.w) { */
    /*         os << delim << Print3{wi}; */
    /*         delim = ", "; */
    /*     } */
    /*     return os << "}"; */
    /* } */

    /* std::ostream& operator<<(std::ostream& os, const Geodesic& x) */
    /* { */
    /*     os << "Geodesic{"; */
    /*     os << "start: " << x.start << ", "; */
    /*     os << "end: " << x.start << ", "; */
    /*     os << "length: " << x.start << ", "; */
    /*     os << "curveKnots.size(): " << x.curveKnots.size(); */
    /*     return os << "}"; */
    /* } */

} // namespace

//==============================================================================
//                      ASSERTION HELPERS
//==============================================================================
namespace
{

    void AssertEq(
        const Vector3& lhs,
        double norm,
        const std::string& msg,
        double eps = 1e-13)
    {
        const bool cond = std::abs(lhs.norm() - norm) > eps;
        if (cond) {
            std::ostringstream os;
            os << "FAILED ASSERT: " << msg << std::endl;
            os << "    lhs.norm() = " << Print3{lhs}
               << ".norm() = " << lhs.norm() << std::endl;
            os << "    expected = " << norm << std::endl;
            os << "    err = " << lhs.norm() - norm << std::endl;
            std::string msg = os.str();
            throw std::runtime_error(msg.c_str());
            /* OSC_ASSERT(cond && msg.c_str()); */
        }
    }

    void AssertEq(
        double lhs,
        double rhs,
        const std::string& msg,
        double eps = 1e-13)
    {
        const bool cond = std::abs(lhs - rhs) > eps;
        if (cond) {
            std::ostringstream os;
            os << "FAILED ASSERT: " << msg << std::endl;
            os << "    lhs = " << lhs << std::endl;
            os << "    rhs = " << rhs << std::endl;
            os << "    err = " << lhs - rhs << std::endl;
            /* throw std::runtime_error(msg); */
            std::string msg = os.str();
            throw std::runtime_error(msg.c_str());
            OSC_ASSERT(cond && msg.c_str());
        }
    }

    void AssertEq(
        const Vector3& lhs,
        const Vector3& rhs,
        const std::string& msg,
        double eps = 1e-13)
    {
        const bool cond = (lhs - rhs).norm() > eps;
        if (cond) {
            std::ostringstream os;
            os << "FAILED ASSERT: " << msg << std::endl;
            os << "    lhs = " << Print3{lhs} << std::endl;
            os << "    rhs = " << Print3{rhs} << std::endl;
            os << "    err = " << Print3{lhs - rhs} << std::endl;
            /* throw std::runtime_error(msg); */
            std::string msg = os.str();
            throw std::runtime_error(msg.c_str());
            OSC_ASSERT(cond && msg.c_str());
        }
    }

    /* bool AssertEq( */
    /*     double lhs, */
    /*     double rhs, */
    /*     const std::string& msg, */
    /*     std::ostream& os, */
    /*     double eps = 1e-13) */
    /* { */
    /*     const bool failed = std::abs(lhs - rhs) > eps; */
    /*     if (failed) { */
    /*         os << "FAILED ASSERT: " << msg << std::endl; */
    /*         os << "    lhs = " << lhs << std::endl; */
    /*         os << "    rhs = " << rhs << std::endl; */
    /*         os << "    err = " << lhs - rhs << std::endl; */
    /*     } */
    /*     return !failed; */
    /* } */

    bool AssertEq(
        const Vector3& lhs,
        const Vector3& rhs,
        const std::string& msg,
        std::ostream& os,
        double eps = 1e-13)
    {
        const bool failed = (lhs - rhs).norm() > eps;
        if (failed) {
            os << "FAILED ASSERT: " << msg << std::endl;
            os << "    lhs = " << Print3{lhs} << std::endl;
            os << "    rhs = " << Print3{rhs} << std::endl;
            os << "    err = " << Print3{lhs - rhs} << std::endl;
        }
        return !failed;
    }

} // namespace

//==============================================================================
//                      RUNGE KUTTA 4
//==============================================================================
namespace
{

    template<typename Y, typename D, typename DY = Y>
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

} // namespace


namespace{
//==============================================================================
//                         IMPLICIT SURFACE STATE
//==============================================================================

struct ImplicitGeodesicState
{
    ImplicitGeodesicState() = default;

    ImplicitGeodesicState(Vector3 aPosition, Vector3 aVelocity):
        position(std::move(aPosition)),
        velocity(std::move(aVelocity)){};

    Vector3 position = {NAN, NAN, NAN};
    Vector3 velocity = {NAN, NAN, NAN};
    double a      = 1.;
    double aDot   = 0.;
    double r      = 0.;
    double rDot   = 1.;
};

struct ImplicitGeodesicStateDerivative
{
    Vector3 velocity     = {NAN, NAN, NAN};
    Vector3 acceleration = {NAN, NAN, NAN};
    double aDot       = NAN;
    double aDDot      = NAN;
    double rDot       = NAN;
    double rDDot      = NAN;
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

/* std::ostream& operator<<(std::ostream& os, const ImplicitGeodesicState& x) */
/* { */
/*     return os << "ImplicitGeodesicState{" */
/*               << "p: " << Print3{x.position} << ", " */
/*               << "v: " << Print3{x.velocity} << ", " */
/*               << "a: " << x.a << ", " */
/*               << "aDot: " << x.a << ", " */
/*               << "r: " << x.r << ", " */
/*               << "rDot: " << x.rDot << "}"; */
/* } */
}

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
    for (std::pair<Vector3, DarbouxFrame>& knot : geodesic.curveKnots) {
        knot.first = calcPointInGround(transform, knot.first);
        calcDarbouxFrameInGlobal(transform, knot.second);
    }
}

//==============================================================================
//                      SURFACE
//==============================================================================

Geodesic Surface::calcGeodesic(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length) const
{
    Vector3 p0        = calcPointInLocal(_transform, initPosition);
    Vector3 v0        = calcVectorInLocal(_transform, initVelocity);
    Geodesic geodesic = calcLocalGeodesicImpl(p0, v0, length);
    calcGeodesicInGlobal(_transform, geodesic);
    return geodesic;
}

const Transf& Surface::getOffsetFrame() const
{
    return _transform;
}

Geodesic Surface::calcWrappingPath(Vector3 pointBefore, Vector3 pointAfter) const
{
    GetSurfaceFn getSurface = [&](size_t idx)->const Surface*{
        return idx == 0 ? this : nullptr;
    };
    return calcNewWrappingPath(pointBefore, pointAfter, getSurface).segments.front();
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
    const ImplicitGeodesicState& q)
{
    const Vector3& p  = q.position;
    const Vector3& v  = q.velocity;
    const Vector3 g   = s.calcSurfaceConstraintGradient(p);
    const Vector3 h_v = s.calcSurfaceConstraintHessian(p) * v;
    // Sign flipped compared to thesis: kn = negative, see eq 3.63
    return -v.dot(h_v) / g.norm();
}

double calcGeodesicTorsion(
    const ImplicitSurface& s,
    const ImplicitGeodesicState& q)
{
    // TODO verify this!
    const Vector3& p  = q.position;
    const Vector3& v  = q.velocity;
    const Vector3 g   = s.calcSurfaceConstraintGradient(p);
    const Vector3 h_v = s.calcSurfaceConstraintHessian(p) * v;
    const Vector3 gxv = g.cross(v);
    return h_v.dot(gxv) / g.dot(g);
}

// TODO use normalized vector.
Vector3 calcSurfaceNormal(const ImplicitSurface& s, const ImplicitGeodesicState& q)
{
    const Vector3& p       = q.position;
    const Vector3 gradient = s.calcSurfaceConstraintGradient(p);
    return gradient / gradient.norm();
}

Vector3 calcAcceleration(const ImplicitSurface& s, const ImplicitGeodesicState& q)
{
    // TODO Writing it out saves a root, but optimizers are smart.
    // Sign flipped compared to thesis: kn = negative, see eq 3.63
    return calcNormalCurvature(s, q) * calcSurfaceNormal(s, q);
}

double calcGaussianCurvature(
    const ImplicitSurface& s,
    const ImplicitGeodesicState& q)
{
    const Vector3& p = q.position;
    Vector3 g        = s.calcSurfaceConstraintGradient(p);
    double gDotg  = g.dot(g);
    Mat3x3 adj    = calcAdjoint(s.calcSurfaceConstraintHessian(p));

    if(gDotg * gDotg < 1e-13) {
        throw std::runtime_error("Gaussian curvature inaccurate: are we normal to surface?");
    }

    return (g.dot(adj * g)) / (gDotg * gDotg);
}

DarbouxFrame calcDarbouxFrame(
    const ImplicitSurface& s,
    const ImplicitGeodesicState& q)
{
    return {q.velocity, calcSurfaceNormal(s, q)};
}

size_t calcPointProjectedToSurface(
    const ImplicitSurface& s,
    Vector3& position,
    double eps = 1e-13,
    size_t maxIter = 10)
{
    Vector3 p0 = position;
    Vector3 pk = position;
    for (size_t iteration = 0; iteration < maxIter; ++iteration) {
        const double c = s.calcSurfaceConstraint(pk);

        const double error = std::abs(c);

        if (error < eps) {
            position = pk;
            return iteration;
        }

        const Vector3 g = s.calcSurfaceConstraintGradient(pk);

        pk += - g * c / g.dot(g);
    }
    throw std::runtime_error("Failed to project point to surface");
}

size_t calcProjectedToSurface(
    const ImplicitSurface& s,
    ImplicitGeodesicState& q,
    double eps = 1e-13,
    size_t maxIter = 100)
{
    /* std::cout << "START calcProjectedToSurface\n"; */
    size_t steps = calcPointProjectedToSurface(s, q.position, eps, maxIter);
    /* std::cout << "    point projected in " << steps << " steps."; */

    Vector3 n = calcSurfaceNormal(s, q);
    if (!(n.cross(q.velocity).norm() > 1e-13)) {
        throw std::runtime_error("Velocity and normal are perpendicular!");
    }

    q.velocity = q.velocity - n.dot(q.velocity) * n;
    q.velocity = q.velocity / q.velocity.norm();

    return steps;
}

//==============================================================================
//                      IMPLICIT SURFACE HELPERS
//==============================================================================

// Compute generic geodesic state from implicit geodesic state.
Geodesic::BoundaryState calcGeodesicBoundaryState(
        const ImplicitSurface& s,
        const ImplicitGeodesicState& x, bool isEnd)
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

    const double tau_g = calcGeodesicTorsion(s, x);
    const double kappa_n = calcNormalCurvature(s, x);
    ImplicitGeodesicState qB {x.position, y.frame.b};
    const double kappa_a = calcNormalCurvature(s, qB);

    y.w = {
        Vector3{tau_g, 0., kappa_n},
        Vector3{ -x.a * kappa_a, -x.aDot, -x.a * tau_g, },
        Vector3{ -x.r * kappa_a, -x.rDot, -x.r * tau_g},
        isEnd ? Vector3{tau_g, 0., kappa_n} : Vector3{0., 0., 0.},
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
        [&](const ImplicitGeodesicState& qk) -> ImplicitGeodesicStateDerivative {
        return calcImplicitGeodesicStateDerivative(qk,
                calcAcceleration(s, qk),
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
    std::vector<std::pair<Vector3, DarbouxFrame>>& log)
{
    ImplicitGeodesicState xStart{positionInit, velocityInit};
    calcProjectedToSurface(s, xStart);

    double l  = 0.;
    double dl = length / static_cast<double>(steps);
    /* std::cout << "ImplicitSurface::calcLocalGeodesic" << std::endl; */
    ImplicitGeodesicState xEnd(xStart);
    /* std::cout << "    xStart = " << xEnd << std::endl; */

    log.push_back({xEnd.position, calcDarbouxFrame(s, xEnd)});

    // For unit testing.
    for (size_t k = 0; k < steps; ++k) {
        RungeKutta4(s, xEnd, l, dl);

        calcProjectedToSurface(s, xEnd);

        // TODO don't log all points.
        log.push_back({xEnd.position, calcDarbouxFrame(s, xEnd)});
    }

    AssertEq(length, l, "Total length does not match integrated length", 1e-6);

    return {xStart, xEnd};
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

Geodesic ImplicitSurface::calcLocalGeodesicImpl(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length) const
{
    Geodesic geodesic;

    std::pair<ImplicitGeodesicState, ImplicitGeodesicState> out =
        calcLocalImplicitGeodesic(
            *this,
            initPosition, initVelocity,
            length,
            _integratorSteps,
            geodesic.curveKnots);

    geodesic.start  = calcGeodesicBoundaryState(*this, out.first, false);
    geodesic.end  = calcGeodesicBoundaryState(*this, out.second, true);
    geodesic.length = length;

    return geodesic;
}

//==============================================================================
//                      IMPLICIT SPHERE SURFACE
//==============================================================================

double ImplicitSphereSurface::calcSurfaceConstraintImpl(Vector3 p) const
{
    return p.dot(p) - _radius * _radius;
}

Vector3 ImplicitSphereSurface::calcSurfaceConstraintGradientImpl(Vector3 p) const
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

Geodesic AnalyticSphereSurface::calcLocalGeodesicImpl(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length) const
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

    std::vector<std::pair<Vector3, DarbouxFrame>> curveKnots;
    size_t nSamples = 10;
    for (size_t i = 0; i < nSamples; ++i) {
        const double angle_i =
            angle * static_cast<double>(i) / static_cast<double>(nSamples);
        const Rotation dq{Eigen::AngleAxisd(angle_i, axis)};
        const DarbouxFrame f = dq * K_P.frame;
        curveKnots.emplace_back(
            std::pair<Vector3, DarbouxFrame>{dq * K_P.position, f});
    }

    return {K_P, K_Q, length, std::move(curveKnots)};
}

//==============================================================================
//                      ANALYTIC CYLINDER SURFACE
//==============================================================================

Geodesic AnalyticCylinderSurface::calcLocalGeodesicImpl(
    Vector3 initPosition,
    Vector3 initVelocity,
    double length) const
{
    if (initPosition.norm() < 1e-13) {
        throw std::runtime_error("Error: initial position at origin.");
    }

    const double r = _radius;
    const double l = length;

    // Cylinder axis assumed to be aligned with z-axis.
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

    // Rotation angle variation to initial direction variation.
    const double dAlpha_dTheta =
        (l / r) * z.cross(f_P.n).dot(f_P.n.cross(f_P.t));

    // Distance along axis variation to initial direction variation.
    const double dh_dTheta = l * z.dot(f_P.n.cross(f_P.t));

    // Rotation angle variation to length variation.
    const double dAlpha_dl = (1. / r) * z.cross(f_P.n).dot(f_P.t);

    // Distance along axis variation to length variation.
    const double dh_dl = z.dot(f_P.t);

    // Initial trihedron: K_P
    Geodesic::BoundaryState K_P;

    // Start position variation.
    const Vector3 zeros{0., 0., 0.};
    K_P.v = {
        z.cross(f_P.n) * r,
        z,
        zeros,
        zeros,
    };

    // Start frame variation.
    K_P.w = {
        z,
        zeros,
        zeros,
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
        z.cross(p_Q),
        z,
        z.cross(p_Q) * dAlpha_dTheta + z * dh_dTheta,
        z.cross(p_Q) * dAlpha_dl + z * dh_dl,
    };

    K_Q.w = {
        z,
        zeros,
        dAlpha_dTheta * z + f_Q.n,
        dAlpha_dl * z + f_Q.n,
    };

    std::vector<std::pair<Vector3, DarbouxFrame>> curveKnots;
    size_t nSamples = 10;
    for (size_t i = 0; i < nSamples; ++i) {
        const double factor =
            static_cast<double>(i) / static_cast<double>(nSamples);
        const double angle_i = alpha * factor;
        const double h_i     = h * factor;
        const Rotation dq{Eigen::AngleAxisd(angle_i, z)};
        const DarbouxFrame f = dq * K_P.frame;
        const Vector3 p_i    = dq * K_P.position + h_i * z;
        curveKnots.emplace_back(std::pair<Vector3, DarbouxFrame>{p_i, f});
    }

    return {K_P, K_Q, length, std::move(curveKnots)};
}

//==============================================================================
//                      PATH CONTINUITY ERROR
//==============================================================================

void PathContinuityError::resize(size_t nSurfaces)
{
    const size_t dim = nSurfaces * 4;
    _pathError.resize(dim);
    _pathCorrections.resize(dim);
    _pathErrorJacobian.resize(dim, dim);

    // Reset values.
    _pathCorrections.fill(NAN);
    _pathError.fill(NAN);
    // Fill with zeros because it is sparse.
    _pathErrorJacobian.fill(0.);
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
    const double cap           = sin(maxAngleDegrees / 180. * PI);

    for (int r = 0; r < pathError.rows(); ++r) {
        double& err = pathError[r];
        err         = (std::abs(err) > cap) ? err / std::abs(err) * cap : err;
    }

    /* std::cout << "pathError =\n" << pathError << std::endl; */
}

void PathContinuityError::calcPathCorrection()
{
    clampPathError(_pathError, _maxAngleDegrees);

    // Compute singular value decomposition. TODO or other decomposition?
    _svd.compute(_pathErrorJacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    _pathCorrections = -_svd.solve(_pathError);
    /* std::cout << "solverCorr =\n" << _pathCorrections << std::endl; */

    _solverError = _pathErrorJacobian.transpose()
                   * (_pathErrorJacobian * _pathCorrections + _pathError);
    /* std::cout << "solverError =\n" << _solverError << std::endl; */
    if (_solverError.norm() > _eps) {
        std::cout << "solver error = " << _solverError.norm() << std::endl;
        throw std::runtime_error("failed to compute corrections: failed to "
                                 "invert path error jacobian");
    }
}

const GeodesicCorrection* PathContinuityError::begin() const
{
    return reinterpret_cast<const GeodesicCorrection*>(&_pathCorrections(0));
}

const GeodesicCorrection* PathContinuityError::end() const
{
    return begin() + _pathCorrections.rows() / 4;
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
    Vector3 tDot = - rate[1] * frame.b + rate[2] * frame.n;

    if (RUNTIME_UNIT_TESTS) {
        const Vector3 w =
            (rate[0] * frame.t + rate[1] * frame.n + rate[2] * frame.b);
        AssertEq(tDot, w.cross(frame.t), "tDot = w.cross(t)");
    }

    return tDot;
}

double calcPathErrorJacobian(
    const Geodesic::BoundaryState& s0,
    const Vector3& position,
    Eigen::VectorXd& pathError,
    Eigen::MatrixXd& pathErrorJacobian,
    size_t idx)
{
    // Compute the length of the straight line segment.
    const Vector3 d = position - s0.position;
    const double l  = d.norm(); // segment length.

    // Define the error as the direction from surface to point.
    const Vector3 e = d / l;

    // Fill the correct row of the path-error-vector and path-error-jacobian:
    // The first straight line segment will add two constraints, and each
    // following segment will add 4 constraints.
    // We therefore start at row:
    size_t row = idx * 4 + 2;

    // Normal and binormal of surface.
    const Vector3& N0 = s0.frame.n;
    const Vector3& B0 = s0.frame.b;

    // Compute path error as projection of error vector onto the normal and
    // binormal.
    pathError[row]     = e.dot(N0);
    pathError[row + 1] = e.dot(B0);

    for (size_t i = 0; i < 4; ++i) {
        // Linear variation of position given ith natural geodesic variation.
        const Vector3 v0 = s0.v.at(i);

        // Angular variation of frame given ith natural geodesic variation.
        const Vector3 w0 = s0.w.at(i);

        // Frame variation given angular variation.
        const Vector3 dN0 = calcNormalDerivative(s0.frame, w0);
        const Vector3 dB0 = calcBinormalDerivative(s0.frame, w0);

        // Variation of error vector.
        const Vector3 de0 = -(v0 - e * e.dot(v0)) / l;

        // Variation of path error values.
        const size_t col                = idx * 4 + i;
        pathErrorJacobian(row, col)     = de0.dot(N0) + e.dot(dN0);
        pathErrorJacobian(row + 1, col) = de0.dot(B0) + e.dot(dB0);
        /* std::cout << "    pathErrorJcobian[" << row <<"," << i << "] = " <<
         * pathErrorJacobian(row, i) << "\n"; */
    }

    // Return the length of the straight line segment.
    return l;
}

double calcPathErrorJacobian(
    const Vector3& position,
    const Geodesic::BoundaryState& s1,
    Eigen::VectorXd& pathError,
    Eigen::MatrixXd& pathErrorJacobian)
{
    /* std::cout << "\n"; */
    /* std::cout << "START calcPathErrorJacobian (start)\n"; */
    /* std::cout << "    p0 = " << Print3{position} << "\n"; */
    /* std::cout << "    s1 = " << s1 << "\n"; */

    const double l  = (s1.position - position).norm();
    const Vector3 e = (s1.position - position) / l;

    /* std::cout << "    l0 = " << l << "\n"; */
    /* std::cout << "    e0 = " << Print3{e} << "\n"; */
    /* std::cout << "    t1 = " << Print3{s1.frame.t} << "\n"; */

    const Vector3 N1 = s1.frame.n;
    const Vector3 B1 = s1.frame.b;

    // Compute path error.
    pathError[0] = e.dot(N1);
    pathError[1] = e.dot(B1);
    /* std::cout << "    eN1 = " << pathError[0] << "\n"; */
    /* std::cout << "    eB1 = " << pathError[1] << "\n"; */

    // Compute jacobian to natural geodesic variation.
    for (size_t i = 0; i < 4; ++i) {
        const Vector3 v1    = s1.v.at(i);
        const Vector3 n1Dot = calcNormalDerivative(s1.frame, s1.w.at(i));
        const Vector3 b1Dot = calcBinormalDerivative(s1.frame, s1.w.at(i));

        const Vector3 de1 = (v1 - e * e.dot(v1)) / l;

        pathErrorJacobian(0, i) = de1.dot(N1) + e.dot(n1Dot);
        pathErrorJacobian(1, i) = de1.dot(B1) + e.dot(b1Dot);

        /* std::cout << "        J0" */
        /*           << "," << i << " = " << pathErrorJacobian(0, i) << "\n"; */
        /* std::cout << "        J1" */
        /*           << "," << i << " = " << pathErrorJacobian(1, i) << "\n"; */
    }

    /* std::cout << "\n"; */
    return l;
}

double calcPathErrorJacobian(
    const Geodesic::BoundaryState& s0,
    const Geodesic::BoundaryState& s1,
    Eigen::VectorXd& pathError,
    Eigen::MatrixXd& pathErrorJacobian,
    size_t idx0)
{
    Vector3 e      = s1.position - s0.position;
    const double l = e.norm();
    e              = e / l;

    const Vector3& N0 = s0.frame.n;
    const Vector3& B0 = s0.frame.b;

    const Vector3& N1 = s1.frame.n;
    const Vector3& B1 = s1.frame.b;

    size_t row = idx0 * 4 + 2;

    // Compute path error.
    pathError[row]     = e.dot(N0);
    pathError[row + 1] = e.dot(B0);

    pathError[row + 2] = e.dot(N1);
    pathError[row + 3] = e.dot(B1);

    /* std::cout << "    pathError = " << row << "\n"; */

    // Compute jacobian to natural geodesic variation.
    for (size_t i = 0; i < 4; ++i) {
        const size_t col = idx0 * 4 + i;
        const Vector3 v0 = s0.v.at(i);
        const Vector3 v1 = s1.v.at(i);

        const Vector3 w0 = s0.w.at(i);
        const Vector3 w1 = s1.w.at(i);

        const Vector3 dN0 = calcNormalDerivative(s0.frame, w0);
        const Vector3 dB0 = calcBinormalDerivative(s0.frame, w0);

        const Vector3 dN1 = calcNormalDerivative(s1.frame, w1);
        const Vector3 dB1 = calcBinormalDerivative(s1.frame, w1);

        const Vector3 de0 = -(v0 - e * e.dot(v0)) / l;
        const Vector3 de1 = (v1 - e * e.dot(v1)) / l;

        {
            pathErrorJacobian(row, col)     = de0.dot(N0) + e.dot(dN0);
            pathErrorJacobian(row + 1, col) = de0.dot(B0) + e.dot(dB0);

            pathErrorJacobian(row, col + 4)     = de1.dot(N0);
            pathErrorJacobian(row + 1, col + 4) = de1.dot(B0);
        }

        {
            pathErrorJacobian(row + 2, col) = de0.dot(N1);
            pathErrorJacobian(row + 3, col) = de0.dot(B1);

            pathErrorJacobian(row + 2, col + 4) = de1.dot(N1) + e.dot(dN1);
            pathErrorJacobian(row + 3, col + 4) = de1.dot(B1) + e.dot(dB1);
        }
    }

    return l;
}

// TODO reuse existin functions.
Vector3 calcPointClosestToPointOnEdge(
    const Vector3& edgePoint0,
    const Vector3& edgePoint1,
    const Vector3& targetPoint)
{
    Vector3 p0 = edgePoint0 - targetPoint;
    Vector3 p1 = edgePoint1 - targetPoint;

    const Vector3 e = p1 - p0;

    Vector3 p = p0 - p0.dot(e) * e / e.dot(e);

    const double d0 = p0.dot(p0);
    const double d1 = p1.dot(p1);
    const double d  = p.dot(p);

    if (d0 < d) {
        return edgePoint0;
    }
    if (d1 < d) {
        return edgePoint1;
    }
    return p + targetPoint;
}

std::vector<Geodesic> calcInitWrappingPathGuess(
    const Vector3& pathStart,
    const Vector3& pathEnd,
    std::function<const Surface*(size_t)>& getSurface)
{
    std::vector<Geodesic> geodesics;
    for (size_t i = 0; getSurface(i); ++i) {
        Vector3 originSurface = getSurface(i)->getOffsetFrame().position;
        const Vector3 pointBefore =
            i == 0 ? pathStart : geodesics.back().end.position;
        const Vector3& pointAfter = pathEnd;

        Vector3 initPositon = calcPointClosestToPointOnEdge(
            pointBefore,
            pointAfter,
            originSurface);
        Vector3 initVelocity = (pathEnd - pathStart);

        // Shoot a zero-length geodesic as initial guess.
        geodesics.push_back(
            getSurface(i)->calcGeodesic(initPositon, initVelocity, 0.));

        if (geodesics.back().length != 0.) {
            throw std::runtime_error("Failed to shoot a zero-length geodesic");
        }
    }
    return geodesics;
}

WrappingPath Surface::calcNewWrappingPath(
    Vector3 pathStart,
    Vector3 pathEnd,
    Surface::GetSurfaceFn& GetSurface,
    double eps,
    size_t maxIter)
{
    PathContinuityError smoothness;

    size_t nSurfaces = 0;
    for (; GetSurface(nSurfaces); ++nSurfaces) {
    }
    /* std::cout << "nSurfaces="<<nSurfaces << std::endl; */

    WrappingPath path(pathStart, pathEnd);
    path.smoothness.resize(nSurfaces);
    path.segments = calcInitWrappingPathGuess(pathStart, pathEnd, GetSurface);

    Surface::calcUpdatedWrappingPath(path, GetSurface, eps, maxIter);

    return path;
}


double calcClamped(double x, double bnd) {
    if (std::abs(x) > std::abs(bnd)) {
        return x / std::abs(x) * std::abs(bnd);
    }
    return x;
}

void applyNaturalGeodesicVariation(
    Geodesic::BoundaryState& geodesicStart,
    const GeodesicCorrection& correction)
{
    // Darboux frame:
    const Vector3& t = geodesicStart.frame.t;
    /* const Vector3& n = geodesicStart.frame.n; */
    const Vector3& b = geodesicStart.frame.b;

    const double maxStep = Eigen::Infinity;
    Vector3 dp = calcClamped(correction.at(1), maxStep) * b + calcClamped(correction.at(0), maxStep) * t;

    // TODO use start frame to rotate both vectors properly.
    // TODO overload vor ANALYTIC?
    geodesicStart.position += dp;

    Vector3 velocity = cos(calcClamped(correction.at(2), maxStep)) * t + sin(calcClamped(correction.at(2), maxStep)) * b;
    geodesicStart.frame.t = velocity;
    /* throw std::runtime_error("failed to compute corrections: failed to "); */
}

size_t Surface::calcUpdatedWrappingPath(
    WrappingPath& path,
    Surface::GetSurfaceFn& GetSurface,
    double eps,
    size_t maxIter)
{
    const ptrdiff_t nSurfaces = path.segments.size();
    /* for (; at(nSurfaces); ++nSurfaces); */
    /*     throw std::runtime_error("Number of geodesic segments in path does
     * not match number of provided getSurface to wrap over"); */
    /* } */

    if (nSurfaces == 0) {
        return 0;
    }

    /* std::cout << "START WrapSolver::calcPath\n"; */
    for (size_t loopIter = 0; loopIter < maxIter; ++loopIter) {
        // Setup path error for first segment.
        calcPathErrorJacobian(
            path.startPoint,
            path.segments.front().start,
            path.smoothness.updPathError(),
            path.smoothness.updPathErrorJacobian());

        // Setup path error between segments.
        for (ptrdiff_t idx = 0; idx + 1 < nSurfaces; ++idx) {
            /* std::cout << "START WrapSolver::intermediate segment" <<
             * std::endl; */
            Geodesic::BoundaryState start = path.segments.at(idx).end;
            Geodesic::BoundaryState end   = path.segments.at(idx + 1).start;
            calcPathErrorJacobian(
                start,
                end,
                path.smoothness.updPathError(),
                path.smoothness.updPathErrorJacobian(),
                idx++);
        }

        // Setup path error for last segment.
        calcPathErrorJacobian(
            path.segments.back().end,
            path.endPoint,
            path.smoothness.updPathError(),
            path.smoothness.updPathErrorJacobian(),
            nSurfaces - 1);

        // Process the path errors.
        path.smoothness.calcPathCorrection();
        /* std::cout << "    ===== STEP ==== = " <<
         * path.smoothness.calcMaxPathError() << "\n"; */

        if (path.smoothness.calcMaxPathError() < eps) {
            /* std::cout << "   Wrapping path solved in " << loopIter <<
             * "steps\n"; */
            return loopIter;
        }

        // Obtain the computed geodesic corrections from the path errors.
        const auto* corrIt  = path.smoothness.begin();
        const auto* corrEnd = path.smoothness.end();
        if (corrEnd - corrIt != nSurfaces) {
            std::runtime_error("Number of geodesic-corrections not equal ot "
                               "number of geodesics");
        }

        // Apply corrections.
        for (ptrdiff_t i = 0; i < nSurfaces; ++i) {
            const GeodesicCorrection& correction = *(corrIt + i);
            applyNaturalGeodesicVariation(
                path.segments.at(i).start,
                correction);

            // TODO last field of correction must be lengthening.
            path.segments.at(i).length += correction.at(3);
            /* std::cout << "    state after correction: " <<
             * path.segments.at(i).start << "\n"; */
        }

        // Shoot a the new geodesics.
        for (ptrdiff_t i = 0; i < nSurfaces; ++i) {
            /* std::cout << "    shoot a new geodesic over " << i << "th
             * surface\n"; */
            path.segments.at(i) = GetSurface(i)->calcGeodesic(
                path.segments.at(i).start.position,
                path.segments.at(i).start.frame.t,
                path.segments.at(i).length);
        }
    }
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
        return {
            {1., 0., 0.},
            {0., 1., 0.},
            {0., 0., 1.},
            {-1., 0., 0.},
            {0., -1., 0.},
            {0., 0., -1.},
        };
    }

    std::vector<Vector3> Surface::makeSelfTestVelocities() const
    {
        return {
            {1., 0., 0.},
            {0., 1., 0.},
            {0., 0., 1.},
            {-1., 0., 0.},
            {0., -1., 0.},
            {0., 0., -1.},
        };
    }

    std::vector<double> Surface::makeSelfTestLengths() const
    {
        const double r = selfTestEquivalentRadius();
        std::vector<double> lengths;
        lengths.push_back(0.);
        for (size_t i = 1; i < 6; ++i){
            lengths.push_back(static_cast<double>(i) * r);
        }
        return lengths;
    }

    void Surface::doSelfTests() const
    {
        for (Vector3 r_P : makeSelfTestPoints())
        {
        for (Vector3 v_P : makeSelfTestVelocities())
        {
        for (double l : makeSelfTestLengths())
        {
            auto transform  = getOffsetFrame();
            doSelfTest(
                     calcPointInGround(transform, r_P),
                    calcVectorInGround(transform, v_P), l);
        }
        }
        }
    }

    // Test variation effects on start and end frames.
    void Surface::doSelfTest(
            Vector3 r_P,
            Vector3 v_P,
            double l,
            double delta,
            double eps
            ) const
    {
        if (r_P.cross(v_P).norm() < 1e-13) {
            // Skip tests with parallel position and velocity: will very likely fail.
            return;
        }

        // Shoot a zero length geodesic.
        const Geodesic gZero = calcGeodesic(r_P, v_P, l);

        // To check the local behavior of the geodesic variation, we apply a
        // small variation to the start point, and see the effect on the
        // geodesic.

        // For debugging.
        std::string msg;

        bool allTestsPassed = true;
        std::ostringstream errs;
        for (size_t i = 0; i < 4; ++i) {
            GeodesicCorrection c {0., 0., 0., 0.};
            c.at(i) = delta;

            Geodesic gOne;
            {
                Geodesic::BoundaryState dK_P = gZero.start;
                applyNaturalGeodesicVariation(dK_P, c);

                // Shoot a new geodesic with the applied variation.
                double dl = i == 3 ? c.at(i) + l : l; // TODO encode this in the struct.
                gOne = calcGeodesic(dK_P.position, dK_P.frame.t, dl);
            }

            std::ostringstream os;
            os << "testing variation = ";
            os << "{" << c.at(0) << "," << c.at(1) << "," << c.at(2) << "," << c.at(3) << "}";
            os << " with l = " << l;

            {
                const Geodesic::BoundaryState K0 = gZero.start;
                const Geodesic::BoundaryState K1 = gOne.start;

                const Vector3 dp = K0.v.at(i);

                const Vector3 dt = calcTangentDerivative(K0.frame, K0.w.at(i));
                const Vector3 dn = calcNormalDerivative(K0.frame, K0.w.at(i));
                const Vector3 db = calcBinormalDerivative(K0.frame, K0.w.at(i));

                allTestsPassed &= AssertEq((K1.position - K0.position) / delta, dp, "Failed start position variation " + os.str(), errs, eps);

                allTestsPassed &= AssertEq((K1.frame.t - K0.frame.t) / delta, dt, "Failed start tangent variation  " + os.str(), errs, eps);
                allTestsPassed &= AssertEq((K1.frame.n - K0.frame.n) / delta, dn, "Failed start normal variation   " + os.str(), errs, eps);
                allTestsPassed &= AssertEq((K1.frame.b - K0.frame.b) / delta, db, "Failed start binormal variation " + os.str(), errs, eps);
            }

            {
                const Geodesic::BoundaryState K0 = gZero.end;
                const Geodesic::BoundaryState K1 = gOne.end;

                const Vector3 dp = K0.v.at(i);

                const Vector3 dt = calcTangentDerivative(K0.frame, K0.w.at(i));
                const Vector3 dn = calcNormalDerivative(K0.frame, K0.w.at(i));
                const Vector3 db = calcBinormalDerivative(K0.frame, K0.w.at(i));

                allTestsPassed &= AssertEq((K1.position - K0.position) / delta, dp, "Failed end position variation" + os.str(), errs, eps);

                allTestsPassed &= AssertEq((K1.frame.t - K0.frame.t) / delta, dt, "Failed end tangent variation " + os.str(), errs, eps);
                allTestsPassed &= AssertEq((K1.frame.n - K0.frame.n) / delta, dn, "Failed end normal variation  " + os.str(), errs, eps);
                allTestsPassed &= AssertEq((K1.frame.b - K0.frame.b) / delta, db, "Failed end binormal variation" + os.str(), errs, eps);
            }
        }
        if (!allTestsPassed) {
            throw std::runtime_error(errs.str());
        }

    }

}
