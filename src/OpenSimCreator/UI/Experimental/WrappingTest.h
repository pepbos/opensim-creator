#pragma once

#include "OpenSimCreator/UI/Experimental/WrappingMath.h"
#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>

namespace osc
{
    struct GeodesicTestBounds
    {
        double eps = 1e-8;
        double numericalDarbouxDrift = 5e-3;
        double variation = 1e-5;
        double varEps = 1e-3; // Take as variation / 10?
        size_t integratorSteps = 2000;
    };

    class TestRapport
    {
        public:
        TestRapport(const std::string& name);

        void print(std::ostream& os);

        void newSection(const std::string& section);

        void newSubSection(const std::string& subSection);

        void assertEq(
                Vector3 lhs,
                Vector3 rhs,
                const std::string& msg,
                double eps);

        void assertEq(
                double lhs,
                double rhs,
                const std::string& msg,
                double eps);

        void assertEq(
                size_t lhs,
                size_t rhs,
                const std::string& msg);

        void assertEq(
                Vector3 lhs,
                double rhs,
                const std::string& msg,
                double eps);

        void assertValue(
                bool value,
                const std::string& msg);

        void finalize();

        bool success() const {return _success;}

        void process(bool result);

        std::ostringstream _oss;

        std::string _indent;
        std::string _name;
        std::string _section;
        std::string _subSection;

        bool _success = true;
        bool _sectionSuccess = true;
        bool _subSectionSuccess = true;
        size_t _countFail = 0;
        size_t _count = 0;
        bool _verbose = false;
    };

bool RunAllWrappingTests(std::ostream& os);

bool RunImplicitGeodesicTest(
    const ImplicitSurface& s,
    const Geodesic& g,
    GeodesicTestBounds bnds,
    const std::string& name,
    std::ostream& os);

std::vector<Trihedron> calcImplicitTestSamples(
    const ImplicitSurface& s,
    Trihedron K,
    double l, size_t steps = 1000);

}

/* DarbouxFrame testRotationIntegration(DarbouxFrame f_P, double angle) */
/* { */
/*     Vector3 axis = -f_P.b; */
/*     auto f       = [=](const Vector3& frameAxis) -> Vector3 { */
/*         return axis.cross(frameAxis); */
/*     }; */

/*     size_t nSteps   = 1000; */
/*     const double dt = angle / static_cast<double>(nSteps); */

/*     DarbouxFrame f_Q = f_P; */
/*     for (size_t i = 0; i < nSteps; ++i) { */
/*         double t = 0.; */
/*         RungeKutta4<Vector3, Vector3>(f_Q.n, t, dt, f); */
/*         RungeKutta4<Vector3, Vector3>(f_Q.t, t, dt, f); */
/*         RungeKutta4<Vector3, Vector3>(f_Q.b, t, dt, f); */
/*     } */

/*     return f_Q; */
/* } */

 // namespace osc

/* // Test variation effects on start and end frames. */
/* void Surface::doSelfTest( */
/*     const std::string name, */
/*     Vector3 r_P, */
/*     Vector3 v_P, */
/*     double l, */
/*     double eps, */
/*     double delta) const */
/* { */

/*     // Shoot a zero length geodesic. */
/*     Geodesic gZero; */
/*     calcGeodesic(r_P, v_P, l, gZero); */

/*     // To check the local behavior of the geodesic variation, we apply a */
/*     // small variation to the start point, and see the effect on the */
/*     // geodesic. */

/*     // For debugging. */
/*     std::string msg; */

/*     bool allTestsPassed = true; */
/*     std::ostringstream errs; */
/*     for (size_t i = 0; i < 4; ++i) { */
/*         GeodesicCorrection c{0., 0., 0., 0.}; */
/*         c.at(i) = delta; */

/*         Geodesic gOne; */
/*         { */
/*             Geodesic::BoundaryState dK_P = gZero.start; */
/*             applyNaturalGeodesicVariation(dK_P, c); */

/*             // Shoot a new geodesic with the applied variation. */
/*             double dl = */
/*                 i == 3 ? c.at(i) + l : l; // TODO encode this in the struct. */
/*             calcGeodesic(dK_P.position, dK_P.frame.t, dl, gOne); */
/*         } */

/*         std::ostringstream os; */
/*         os << "testing variation = "; */
/*         os << "{" << c.at(0) << "," << c.at(1) << "," << c.at(2) << "," */
/*            << c.at(3) << "}"; */
/*         os << " with l = " << l; */

/*         { */
/*             const Geodesic::BoundaryState K0 = gZero.start; */
/*             const Geodesic::BoundaryState K1 = gOne.start; */

/*             const Vector3 dp = K0.v.at(i); */

/*             const Vector3 dt = calcTangentDerivative(K0.frame, K0.w.at(i)); */
/*             const Vector3 dn = calcNormalDerivative(K0.frame, K0.w.at(i)); */
/*             const Vector3 db = calcBinormalDerivative(K0.frame, K0.w.at(i)); */

/*             allTestsPassed &= AssertEq( */
/*                 (K1.position - K0.position) / delta, */
/*                 dp, */
/*                 name + ": Failed start position variation " + os.str(), */
/*                 errs, */
/*                 eps); */

/*             allTestsPassed &= AssertEq( */
/*                 (K1.frame.t - K0.frame.t) / delta, */
/*                 dt, */
/*                 name + ": Failed start tangent variation  " + os.str(), */
/*                 errs, */
/*                 eps); */
/*             allTestsPassed &= AssertEq( */
/*                 (K1.frame.n - K0.frame.n) / delta, */
/*                 dn, */
/*                 name + ": Failed start normal variation   " + os.str(), */
/*                 errs, */
/*                 eps); */
/*             allTestsPassed &= AssertEq( */
/*                 (K1.frame.b - K0.frame.b) / delta, */
/*                 db, */
/*                 name + ": Failed start binormal variation " + os.str(), */
/*                 errs, */
/*                 eps); */
/*         } */

/*         { */
/*             const Geodesic::BoundaryState K0 = gZero.end; */
/*             const Geodesic::BoundaryState K1 = gOne.end; */

/*             const Vector3 dp = K0.v.at(i); */

/*             const Vector3 dt = calcTangentDerivative(K0.frame, K0.w.at(i)); */
/*             const Vector3 dn = calcNormalDerivative(K0.frame, K0.w.at(i)); */
/*             const Vector3 db = calcBinormalDerivative(K0.frame, K0.w.at(i)); */

/*             allTestsPassed &= AssertEq( */
/*                 (K1.position - K0.position) / delta, */
/*                 dp, */
/*                 name + ": Failed end position variation" + os.str(), */
/*                 errs, */
/*                 eps); */

/*             allTestsPassed &= AssertEq( */
/*                 (K1.frame.t - K0.frame.t) / delta, */
/*                 dt, */
/*                 name + ": Failed end tangent variation " + os.str(), */
/*                 errs, */
/*                 eps); */
/*             allTestsPassed &= AssertEq( */
/*                 (K1.frame.n - K0.frame.n) / delta, */
/*                 dn, */
/*                 name + ": Failed end normal variation  " + os.str(), */
/*                 errs, */
/*                 eps); */
/*             allTestsPassed &= AssertEq( */
/*                 (K1.frame.b - K0.frame.b) / delta, */
/*                 db, */
/*                 name + ": Failed end binormal variation" + os.str(), */
/*                 errs, */
/*                 eps); */
/*         } */
/*     } */
/*     if (!allTestsPassed) { */
/*         throw std::runtime_error(errs.str()); */
/*     } */
/* } */

/* bool WrappingTester( */
/*     const WrappingPath& path, */
/*     WrappingPath::GetSurfaceFn& GetSurface, */
/*     std::ostream& os, */
/*     double d, */
/*     double eps) */
/* { */
/*     WrappingPath pathZero = path; */
/*     calcPathErrorJacobian(pathZero); */

/*     const size_t n = path.segments.size(); */
/*     os << "Start test for path error jacobian:\n"; */
/*     bool success = true; */
/*     for (size_t i = 0; i < n; ++i) { */
/*         os << "    Start testing Surface " << i << "\n"; */
/*         for (size_t j = 0; j < 9; ++j) { */
/*             WrappingPath pathOne = path; */

/*             const Surface* surface = GetSurface(i); */

/*             GeodesicCorrection correction{0., 0., 0., 0.}; */

/*             Eigen::VectorXd correctionVector(n * 4); */
/*             correctionVector.fill(0.); */

/*             if (j < 8) { */
/*                 correction.at(j % 4)              = (j < 4) ? d : -d; */
/*                 correctionVector[i * 4 + (j % 4)] = (j < 4) ? d : -d; */
/*             } */
/*             os << "        d" << i << " = " << correctionVector.transpose() */
/*                << "\n"; */

/*             Geodesic::BoundaryState start = pathOne.segments.at(i).start; */
/*             applyNaturalGeodesicVariation(start, correction); */

/*             const double length = */
/*                 pathOne.segments.at(i).length + correction.at(3); */
/*             surface->calcGeodesic( */
/*                 start.position, */
/*                 start.frame.t, */
/*                 length, */
/*                 pathOne.segments.at(i)); */

/*             calcPathErrorJacobian(pathOne); */

/*             Eigen::VectorXd dErrExpected = */
/*                 pathZero.smoothness._pathErrorJacobian * correctionVector; */

/*             Eigen::VectorXd dErr = */
/*                 pathOne.smoothness._pathError - pathZero.smoothness._pathError; */

/*             os << "        dErrExp" << i << " = " */
/*                << dErrExpected.transpose() / d << "\n"; */
/*             os << "        dErr   " << i << " = " << dErr.transpose() / d */
/*                << "\n"; */
/*             os << "        ErrZero" << i << " = " */
/*                << pathZero.smoothness._pathError.transpose() << "\n"; */
/*             os << "        ErrOne" << i << "  = " */
/*                << pathOne.smoothness._pathError.transpose() << "\n"; */

/*             for (int k = 0; k < dErr.rows(); ++k) { */
/*                 if (std::abs(dErrExpected[k] / d - dErr[k] / d) > eps) { */
/*                     os << "    FAILED TEST FOR SURFACE " << i */
/*                        << " with d = " << correctionVector.transpose() << "\n"; */
/*                     success = false; */
/*                     break; */
/*                 } */
/*             } */
/*         } */
/*     } */
/*     return success; */
/* } */
