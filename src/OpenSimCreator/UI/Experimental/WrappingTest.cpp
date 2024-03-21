#include "WrappingTest.h"

#include "WrappingMath.h"

using namespace osc;

//==============================================================================
//                      UNIT TESTING
//==============================================================================

namespace
{

//==============================================================================
//                      GEODESIC FORWARD INTEGRATION
//==============================================================================

// Darboux trihedron along a geodesic curve.
struct Trihedron
{
    static Trihedron FromPointAndTangentGuessAndNormal(
        Vector3 point,
        Vector3 tangentGuess,
        Vector3 normal)
    {
        Trihedron K;
        K.p = std::move(point);

        K.n = std::move(normal);
        K.t = std::move(tangentGuess);

        K.t = K.t - K.n * K.n.dot(K.t);
        K.t = K.t / K.t.norm();

        K.b = K.t.cross(K.n);
        K.b = K.b / K.b.norm();

        return K;
    }

    Vector3 p = {NAN, NAN, NAN};
    Vector3 t = {NAN, NAN, NAN};
    Vector3 n = {NAN, NAN, NAN};
    Vector3 b = {NAN, NAN, NAN};
};

struct TrihedronDerivative
{
    Vector3 pDot = {NAN, NAN, NAN};
    Vector3 tDot = {NAN, NAN, NAN};
    Vector3 nDot = {NAN, NAN, NAN};
    Vector3 bDot = {NAN, NAN, NAN};
};

TrihedronDerivative calcTrihedronDerivative(
    const Trihedron& y,
    double kn,
    double tau_g)
{
    TrihedronDerivative dy;

    dy.pDot = y.t;
    dy.tDot = kn * y.n;
    dy.nDot = -kn * y.t - tau_g * y.b;
    dy.bDot = tau_g * y.n;

    return dy;
}

Trihedron operator*(double dt, TrihedronDerivative& dy)
{
    Trihedron y;
    y.p = dt * dy.pDot;
    y.t = dt * dy.tDot;
    y.n = dt * dy.nDot;
    y.b = dt * dy.bDot;
    return y;
}

Trihedron operator+(const Trihedron& lhs, const Trihedron& rhs)
{
    Trihedron y;
    y.p = lhs.p + rhs.p;
    y.t = lhs.t + rhs.t;
    y.n = lhs.n + rhs.n;
    y.b = lhs.b + rhs.b;
    return y;
}

void TrihedronStep(const ImplicitSurface& s, Trihedron& q, double& l, double dl)
{
    RungeKutta4<Trihedron, TrihedronDerivative>(
        q,
        l,
        dl,
        [&](const Trihedron& qk) -> TrihedronDerivative {
            return calcTrihedronDerivative(
                qk,
                s.testCalcNormalCurvature(qk.p, qk.t),
                s.testCalcGeodesicTorsion(qk.p, qk.t));
        });
}

std::vector<Trihedron> calcImplicitGeodesic(
    const ImplicitSurface& s,
    Trihedron q,
    double l,
    size_t n)
{
    std::vector<Trihedron> samples;
    samples.push_back(q);

    double dl = l / static_cast<double>(n);
    for (size_t k = 0; k < n; ++k) {
        TrihedronStep(s, q, l, dl);
        samples.push_back(q);
    }

    return samples;
}

//==============================================================================
//                      UNIT TESTING
//==============================================================================

bool RunTrihedronTest(const Trihedron& y, const std::string& msg, TestRapport& o, double eps)
{
    const Vector3& t = y.t;
    const Vector3& n = y.n;
    const Vector3& b = y.b;

    bool success = true;

    o.assertEq(t.dot(t), 1., msg + ": t norm = 1.", eps);
    o.assertEq(n.dot(n), 1., msg + ": n norm = 1.", eps);
    o.assertEq(b.dot(b), 1., msg + ": b norm = 1.", eps);

    o.assertEq(t.dot(n), 0., msg + ": t.dot(n) = 0.", eps);
    o.assertEq(t.dot(b), 0., msg + ": t.dot(b) = 0.", eps);
    o.assertEq(n.dot(b), 0., msg + ": n.dot(b) = 0.", eps);

    o.assertEq(t.cross(n), b, msg + ": t.cross(n) = b", eps);
    o.assertEq(b.cross(t), n, msg + ": b.cross(t) = n", eps);
    o.assertEq(n.cross(b), t, msg + ": n.cross(b) = t", eps);

    return success;
}

bool RunRungeKutta4Test(std::ostream& os)
{
    os << "Start all wrapping tests";

    return true;
}

std::vector<Trihedron> RunImplicitGeodesicShooterTest(
        const ImplicitSurface& s,
        const Geodesic& g,
        GeodesicTestBounds bnds,
        TestRapport& o)
{
    const Vector3 p_P = g.start.position;
    const Vector3 v_P = g.start.frame.t;
    const double l = g.length;
    const size_t n = bnds.integratorSteps;

    // Initial trihedron.
    Trihedron K_P {g.start.position, g.start.frame.t, g.start.frame.n, g.start.frame.b};

    // Final trihedron.
    std::vector<Trihedron> samples = calcImplicitGeodesic(s, K_P, l, n);

    // Basic tests:
    {
        o.newSubSection("Shooter basics");
        o.assertEq(samples.size(), n+1, "Number of samples from integrator");

        // Darboux frame directions must be perpendicular.
        RunTrihedronTest(K_P, "Initial trihedron", o, bnds.eps);
        RunTrihedronTest(samples.back(), "Final trihedron", o, bnds.eps);
        for (const Trihedron& y: samples) {
            if (!o.success()) { break; }
            RunTrihedronTest(y, "Intermediate trihedron", o, bnds.eps);
        }
    }

    o.newSubSection("Shooter surface drift");
    {
        // Assert if points lie on the surface.
        o.assertEq(s.calcSurfaceConstraint(samples.front().p), 0., "Start position on surface", bnds.eps);
        o.assertEq(s.calcSurfaceConstraint(samples.back().p), 0., "End position on surface", bnds.eps);
        for (const Trihedron& K: samples) {
            if (!o.success()) { break; }
            o.assertEq(s.calcSurfaceConstraint(K.p), 0., "Intermediate position on surface", bnds.eps);
        }

    }

    o.newSubSection("Shooter darboux drift");
    if (samples.size() > 2)
    {

        auto TrihedronAssertionHelper = [&](
                const Trihedron& prev,
                const Trihedron& next,
                const std::string msg)
        {
            Trihedron K_est;
            K_est.p = prev.p;
            K_est.t = (next.p - prev.p) / (next.p - prev.p).norm();
            K_est.n = s.testCalcSurfaceNormal(prev.p);
            K_est.b = K_est.t.cross(K_est.n);
            K_est.b = K_est.b / K_est.b.norm();

            RunTrihedronTest(K_est, msg + " Numerical trihedron from motion", o, bnds.numericalDarbouxDrift);

            o.assertEq(K_est.t, prev.t, msg + " Tangents  match", bnds.numericalDarbouxDrift);
            o.assertEq(K_est.n, prev.n, msg + " Normals   match", bnds.numericalDarbouxDrift);
            o.assertEq(K_est.b, prev.b, msg + " Binormals match", bnds.numericalDarbouxDrift);
        };

        // Assert Darboux frame along curve.

        TrihedronAssertionHelper(samples.front(), samples.at(1), "Start");
        TrihedronAssertionHelper(samples.at(samples.size() - 2), samples.back(), "End");

        Trihedron prev = samples.front();
        for (size_t i = 1; i < samples.size(); ++i) {
            if (!o.success()) { break; }
            const Trihedron& next = samples.at(i);
            TrihedronAssertionHelper(prev, next, "Intermediate");
            prev = next;
        }
    }

    // Compare integrated to implicit version.
    o.newSubSection("Shooter vs implicit integrator");
    {
        // Start trihedron test.
        {
            Trihedron K_P = samples.front();
            o.assertEq(K_P.p, g.start.position, "Start position match", bnds.eps);
            o.assertEq(K_P.t, g.start.frame.t,  "Start tangent  match", bnds.eps);
            o.assertEq(K_P.n, g.start.frame.n,  "Start normal   match", bnds.eps);
            o.assertEq(K_P.b, g.start.frame.b,  "Start binormal match", bnds.eps);
        }
        // End trihedron test.
        {
            Trihedron K_Q = samples.back();
            o.assertEq(K_Q.p, g.end.position,"End position match", bnds.eps);
            o.assertEq(K_Q.t, g.end.frame.t, "End tangent  match", bnds.eps);
            o.assertEq(K_Q.n, g.end.frame.n, "End normal   match", bnds.eps);
            o.assertEq(K_Q.b, g.end.frame.b, "End binormal match", bnds.eps);
        }
    }

    return samples;
}

void RunImplicitGeodesicVariationTest(
    const ImplicitSurface& s,
    const Geodesic& gZero,
    GeodesicTestBounds bnds,
    TestRapport& o)
{
    // Verify geodesic numerically.
    o.newSection("Unconstrained comparison");
    std::vector<Trihedron> samplesZero = RunImplicitGeodesicShooterTest(s, gZero, bnds, o);

    Trihedron K_P_zero = samplesZero.front();
    Trihedron K_Q_zero = samplesZero.back();

    auto ToTrihedron = [](const DarbouxFrame& q, Vector3 v) -> Vector3
    {
        return q.t * v[0] + q.n * v[1] + q.b * v[2];
    };

    // Stop if already failed.
    if (!o.success()) {
        return;
    }

    // Variation test.
    for (size_t i = 0; i < 8; ++i)
    {

        // Make a copy of the geodesic.
        Geodesic gOne = gZero;

        GeodesicCorrection c = {0., 0., 0., 0.};
        const double d =  i < 4 ? bnds.variation : -bnds.variation;
        c.at(i%4) = d;
        s.applyVariation(gOne, c);

        {
            std::ostringstream oss;
            oss << "Geodesic variation: ";
            for (size_t j = 0; j < 4; ++j) {
                oss << c.at(j) << ",";
            }
            o.newSection(oss.str());
        }

        std::vector<Trihedron> samples = RunImplicitGeodesicShooterTest(s, gOne, bnds, o);

        // Verify that variation had the expected effect at start and end frames.
        {
            Trihedron K_P_one = samples.front();
            const Vector3 v_P = gZero.start.v.at(i%4);
            const Vector3 w_P = ToTrihedron(gZero.start.frame, gZero.start.w.at(i%4));

            o.assertEq((K_P_one.p - K_P_zero.p) / d, v_P, "Shooter p_P1 - p_P0 = v_P", bnds.varEps);
            o.assertEq((K_P_one.t - K_P_zero.t) / d, w_P.cross(K_P_zero.t), "Shooter t_P1 - t_P0 = w_P x t_P0", bnds.varEps);
            o.assertEq((K_P_one.n - K_P_zero.n) / d, w_P.cross(K_P_zero.n), "Shooter n_P1 - n_P0 = w_P x n_P0", bnds.varEps);
            o.assertEq((K_P_one.b - K_P_zero.b) / d, w_P.cross(K_P_zero.b), "Shooter b_P1 - b_P0 = w_P x b_P0", bnds.varEps);
        }

        {
            Trihedron K_Q_one = samples.back();
            const Vector3 v_Q = gZero.end.v.at(i%4);
            const Vector3 w_Q = ToTrihedron(gZero.end.frame, gZero.end.w.at(i%4));

            o.assertEq((K_Q_one.p - K_Q_zero.p) / d, v_Q, "Shooter p_Q1 - p_Q0 = v_Q", bnds.varEps);
            o.assertEq((K_Q_one.t - K_Q_zero.t) / d, w_Q.cross(K_Q_zero.t), "Shooter t_Q1 - t_Q0 = w_Q x t_Q0", bnds.varEps);
            o.assertEq((K_Q_one.n - K_Q_zero.n) / d, w_Q.cross(K_Q_zero.n), "Shooter n_Q1 - n_Q0 = w_Q x n_Q0", bnds.varEps);
            o.assertEq((K_Q_one.b - K_Q_zero.b) / d, w_Q.cross(K_Q_zero.b), "Shooter b_Q1 - b_Q0 = w_Q x b_Q0", bnds.varEps);
        }
    }
}

} // namespace

namespace osc
{

    TestRapport::TestRapport(const std::string& name) : _name(name)
    {
        _oss << "\n";
        size_t n = 50;
        for (size_t i = 0; i < n; ++i) _oss << '=';
        _oss << "\n";

        _oss << "==== START TEST RAPPORT FOR: " << _name << " ====\n";

        for (size_t i = 0; i < n; ++i) _oss << '=';
        _oss << "\n";
        _oss << "\n";
    }

    void TestRapport::print(std::ostream& os) {
        os << _oss.str();
    }

    void TestRapport::newSection(const std::string& section) {
        newSubSection("");
        if (!_section.empty())
        {
            _oss << "END SECTION: " << _section << "\n";
            if (_sectionSuccess) {
                if (_countFail != 0) {
                    throw std::runtime_error("ERROR: fail count should have been zero");
                }
                _oss << " success (" << _count << ")\n";
            } else {
                _oss << " FAILED (" << _countFail << "/" << _count + _countFail << ")\n";
            }
        }
        _sectionSuccess = true;
        _section = section;
        if (!_section.empty()) {
            _indent = "    ";
            _oss << "START SECTION: " << section << "\n";
        }
        _countFail = 0;
        _count = 0;
    }

    void TestRapport::newSubSection(const std::string& subSection) {
        if (!_subSection.empty())
        {
            _oss << "    " << "end  subsection: " << _subSection;
            if (_subSectionSuccess) {
                _oss << " success (" << _count << ")\n";
            } else {
                _oss << " FAILED (" << _countFail << "/" << _count << ")\n";
            }
        }
        _subSectionSuccess = true;
        _subSection = subSection;
        if (!_subSection.empty()) {
            _indent = "        ";
            _oss << "    " << "start subsection: " << subSection << "\n";
        }
    }

    void TestRapport::finalize() {
        newSubSection("");
        newSection("");

        _oss << "\n";
        size_t n = 50;
        for (size_t i = 0; i < n; ++i) _oss << '=';
        _oss << "\n";

        _oss << "==== FINISHED TEST RAPPORT FOR: " << _name << ": ";

        if (_success) {
            _oss << " success\n";
        } else {
            _oss << " FAILED\n";
        }

        for (size_t i = 0; i < n; ++i) _oss << '=';
        _oss << "\n";
    }

    void TestRapport::assertEq(
                Vector3 lhs,
                Vector3 rhs,
                const std::string& msg,
                double eps)
    {
        const bool isOk = (lhs - rhs).norm() < eps;
        if (!isOk) {
            _oss << _indent << "ASSERTION FAILED:   " << msg << "\n";
            _oss << _indent << "    lhs = " << lhs.transpose() << "\n";
            _oss << _indent << "    rhs = " << rhs.transpose() << "\n";
            _oss << _indent << "    err = " << (lhs - rhs).transpose() << "\n";
            _oss << _indent << "    bnd = " << eps << "\n";
        }
        process(isOk);
    }

    void TestRapport::assertEq(
                Vector3 lhs,
                double rhs,
                const std::string& msg,
                double eps)
    {
        const bool isOk = std::abs(lhs.norm() - rhs) < eps;
        if (!isOk) {
            _oss << _indent << "ASSERTION FAILED:   " << msg << "\n";
            _oss << _indent << "    lhs  = " << lhs.transpose() << "\n";
            _oss << _indent << "    norm = " << lhs.norm() << "\n";
            _oss << _indent << "    rhs  = " << rhs << "\n";
            _oss << _indent << "    err  = " << std::abs(lhs.norm() - rhs) << "\n";
            _oss << _indent << "    bnd  = " << eps << "\n";
        }
        process(isOk);
    }

    void TestRapport::assertEq(
                double lhs,
                double rhs,
                const std::string& msg,
                double eps)
    {
        const bool isOk = std::abs(lhs - rhs) < eps;
        if (!isOk) {
            _oss << _indent << "ASSERTION FAILED:   " << msg << "\n";
            _oss << _indent << "    lhs  = " << lhs << "\n";
            _oss << _indent << "    rhs  = " << rhs << "\n";
            _oss << _indent << "    err  = " << std::abs(lhs - rhs) << "\n";
            _oss << _indent << "    bnd  = " << eps << "\n";
        }
        process(isOk);
    }

    void TestRapport::assertEq(
                size_t lhs,
                size_t rhs,
                const std::string& msg)
    {
        const bool isOk = lhs == rhs;
        if (!isOk) {
            _oss << _indent << "ASSERTION FAILED:   " << msg << "\n";
            _oss << _indent << "    lhs  = " << lhs << "\n";
            _oss << _indent << "    rhs  = " << rhs << "\n";
        }
        process(isOk);
    }

    void TestRapport::assertValue(
            bool value,
            const std::string& msg)
    {
        if (!value) {
            _oss << _indent << "ASSERTION FAILED:   " << msg << "\n";
        }
        process(value);
    }

    void TestRapport::process(bool result)
    {
        _subSectionSuccess &= result;
        _sectionSuccess &= result;
        _success &= result;
        if (!result) {
            ++_countFail;
        }
        ++_count;
    }

bool RunImplicitGeodesicTest(
        const ImplicitSurface& s,
        const Geodesic& g,
        GeodesicTestBounds bnds,
        const std::string& name,
        std::ostream& os)
{
    TestRapport o(name);
    RunImplicitGeodesicVariationTest(s, g, bnds, o);
    o.finalize();
    o.print(os);
    return o.success();
}

bool RunAllWrappingTests(std::ostream& os)
{
    os << "Start all wrapping tests";

    bool success = true;

    success &= RunRungeKutta4Test(os);

    return success;
}

std::vector<Geodesic::Sample> calcImplicitTestSamples(
    const ImplicitSurface& s,
    Vector3 p,
    DarbouxFrame f,
    double l,
    size_t steps)
{
    Trihedron K_P {std::move(p), f.t, f.n, f.b};
    std::vector<Geodesic::Sample> samples;
    for (const Trihedron& K: calcImplicitGeodesic(s, K_P, l, steps))
    {
        samples.push_back({K.p, {K.t, K.n, K.b}});
    }
    return samples;
}

} // namespace osc
