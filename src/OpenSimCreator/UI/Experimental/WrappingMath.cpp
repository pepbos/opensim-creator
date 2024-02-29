#include "WrappingMath.h"

#include "oscar/Utils/Assertions.h"
#include <iostream>
#include <string>

using namespace osc;

/* static constexpr bool RUNTIME_UNIT_TESTS = true; */

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

/* std::ostream& operator<<(std::ostream& os, const Geodesic::BoundaryState& x) */
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

}

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
    const bool  cond =std::abs(lhs - rhs) > eps;
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
    const bool  cond =(lhs - rhs).norm() > eps;
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
}

//==============================================================================
//                      DARBOUX FRAME
//==============================================================================

void AssertDarbouxFrame(const DarbouxFrame& frame) {
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

DarbouxFrame:: DarbouxFrame(Vector3 tangent, Vector3 normal, Vector3 binormal) :
    t(std::move(tangent)), n(std::move(normal)), b(std::move(binormal)) {
        AssertDarbouxFrame(*this);}

DarbouxFrame calcDarbouxFromTangentGuessAndNormal(
        Vector3 tangentGuess,
        Vector3 surfaceNormal)
{
    Vector3 n = std::move(surfaceNormal);
    n = n / n.norm();

    Vector3 t = std::move(tangentGuess);
    t = t - n * n.dot(t);
    t = t / t.norm();

    Vector3 b = t.cross(n);

    return{t, n, b};
}
