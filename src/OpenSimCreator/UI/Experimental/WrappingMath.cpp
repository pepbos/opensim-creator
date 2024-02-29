#include "WrappingMath.h"

#include "oscar/Utils/Assertions.h"
#include <iostream>
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
        os << "Vector3{";
        std::string delim;
        for (size_t r = 0; r < 3; ++r) {
            os << delim << p.x(r);
            delim = ", ";
        }
        return os << "}";
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

    /* void AssertEq( */
    /*     const Vector3& lhs, */
    /*     double norm, */
    /*     const std::string& msg, */
    /*     double eps = 1e-13) */
    /* { */
    /*     const bool cond = std::abs(lhs.norm() - norm) > eps; */
    /*     if (cond) { */
    /*         std::ostringstream os; */
    /*         os << "FAILED ASSERT: " << msg << std::endl; */
    /*         os << "    lhs.norm() = " << Print3{lhs} */
    /*                   << ".norm() = " << lhs.norm() << std::endl; */
    /*         os << "    expected = " << norm << std::endl; */
    /*         os << "    err = " << lhs.norm() - norm << std::endl; */
    /*         std::string msg = os.str(); */
    /*         OSC_ASSERT(cond && msg.c_str()); */
    /*     } */
    /* } */

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
            OSC_ASSERT(cond && msg.c_str());
        }
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

namespace{
DarbouxFrame
operator*(
        const Rotation& lhs,
        const DarbouxFrame& rhs)
{
    return {
        lhs * rhs.t,
        lhs * rhs.n,
        lhs * rhs.b,
    };
}
}

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
    x.position = calcPointInLocal(transform, x.position);
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
    Vector3 v0        = calcPointInLocal(_transform, initVelocity);
    Geodesic geodesic = calcLocalGeodesicImpl(p0, v0, length);
    calcGeodesicInGlobal(_transform, geodesic);
    return geodesic;
}

const Transf& Surface::getOffsetFrame() const
{
    return _transform;
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

    // Project onto surface.
    initVelocity = initVelocity
                   - initPosition * initPosition.dot(initVelocity)
                         / initPosition.dot(initPosition);

    // Initial darboux frame.
    DarbouxFrame f_P(initVelocity, initPosition);

    // Initial trihedron: K_P
    Geodesic::BoundaryState K_P;

    K_P.w = {
        -f_P.b,
        f_P.t,
        f_P.n,
        Vector3{0., 0., 0.}
    };

    // Since position = normal * radius -> p = n * r
    // We have dp = dn * r
    // With: dn = w x n
    for (size_t i = 0; i < 4; ++i) {
        K_P.v.at(i) = K_P.w.at(i).cross(f_P.n) * r;
    }

    K_P.position = f_P.n * _radius;
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
    K_Q.position = f_Q.n * _radius;
    K_Q.frame    = f_Q;

    // For a sphere the rotation of the initial frame directly rotates the final
    // frame:
    K_Q.w       = K_P.w;
    K_Q.w.at(3) = K_P.w.at(0);

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
                (df_P.n - f_P.n) / d,
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
                (df_Q.n - f_Q.n) / d,
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

    std::vector<std::pair<Vector3, DarbouxFrame>> curveKnots;
    size_t nSamples = 10;
    for (size_t i =0; i < nSamples; ++i)
    {
        const double angle_i =
            static_cast<double>(i) /
            static_cast<double>(nSamples);
        const Rotation dq{Eigen::AngleAxisd(angle_i, axis)};
        const DarbouxFrame f = dq * K_P.frame;
        curveKnots.emplace_back(
            std::pair<Vector3, DarbouxFrame>{dq * K_P.position, f});
    }

    return {K_P, K_Q, length, std::move(curveKnots)};
}
