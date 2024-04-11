#include "WrappingTest.h"

#include "WrappingMath.h"
#include <stdexcept>

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
struct Tri
{
    static Tri FromPointAndTangentGuessAndNormal(
        Vector3 point,
        Vector3 tangentGuess,
        Vector3 normal)
    {
        Tri K;
        K.p = std::move(point);

        K.n = std::move(normal);
        K.t = std::move(tangentGuess);

        K.t = K.t - K.n * K.n.dot(K.t);
        K.t = K.t / K.t.norm();

        K.b = K.t.cross(K.n);
        K.b = K.b / K.b.norm();

        K.v = K.t;

        return K;
    }

    Vector3 p = {NAN, NAN, NAN};
    Vector3 v = {NAN, NAN, NAN};
    Vector3 t = {NAN, NAN, NAN};
    Vector3 n = {NAN, NAN, NAN};
    Vector3 b = {NAN, NAN, NAN};
};

struct TrihedronDerivative
{
    Vector3 pDot = {NAN, NAN, NAN};
    Vector3 vDot = {NAN, NAN, NAN};
    Vector3 tDot = {NAN, NAN, NAN};
    Vector3 nDot = {NAN, NAN, NAN};
    Vector3 bDot = {NAN, NAN, NAN};
};

TrihedronDerivative calcTrihedronDerivative(
    const Tri& y,
    Vector3 a,
    double kn,
    double tau_g)
{
    /* std::cout << "tau_g = " << tau_g << "\n"; */
    /* std::cout << "a = " << a << "\n"; */
    TrihedronDerivative dy;

    dy.pDot = y.v;
    dy.vDot = std::move(a);
    dy.tDot = kn * y.n;
    dy.nDot = -kn * y.t + tau_g * y.b;
    dy.bDot = -tau_g * y.n;

    return dy;
}

Tri operator*(double dt, TrihedronDerivative& dy)
{
    Tri y;
    y.p = dt * dy.pDot;
    y.v = dt * dy.vDot;
    y.t = dt * dy.tDot;
    y.n = dt * dy.nDot;
    y.b = dt * dy.bDot;
    return y;
}

Tri operator+(const Tri& lhs, const Tri& rhs)
{
    Tri y;
    y.p = lhs.p + rhs.p;
    y.v = lhs.v + rhs.v;
    y.t = lhs.t + rhs.t;
    y.n = lhs.n + rhs.n;
    y.b = lhs.b + rhs.b;
    return y;
}

void TrihedronStep(const WrapObstacle& s, Tri& q, double& l, double dl)
{
    RungeKutta4<Tri, TrihedronDerivative>(
        q,
        l,
        dl,
        [&](const Tri& qk) -> TrihedronDerivative {
        const double kn = s.calcNormalCurvature(qk.p, qk.v);
        const double tau_g = s.calcGeodesicTorsion(qk.p, qk.v);
        const Vector3 n = s.calcSurfaceNormal(qk.p);
            return calcTrihedronDerivative(
                qk,
                n * kn,
                kn,
                tau_g);
        });
}

std::vector<Tri> calcImplicitGeodesic(
    const WrapObstacle& s,
    Tri q,
    double l,
    size_t n)
{
    std::vector<Tri> samples;
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

bool RunTrihedronTest(const Tri& y, const std::string& msg, TestRapport& o, double eps)
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

using Mat33d = Eigen::Matrix<double, 3,3>;

Mat33d AsMat3(const Tri& q) {
    Mat33d mat;
    mat(0,0) = q.t(0);
    mat(1,0) = q.t(1);
    mat(2,0) = q.t(2);

    mat(0,1) = q.n(0);
    mat(1,1) = q.n(1);
    mat(2,1) = q.n(2);

    mat(0,2) = q.b(0);
    mat(1,2) = q.b(1);
    mat(2,2) = q.b(2);
    return mat;
}

Vector3 calcRotationVector(Eigen::Quaterniond q)
{
    const double sinHalfAngle = q.vec().norm();
    if (sinHalfAngle < 1e-16) {
        return {0., 0., 0.};
    }
    const double angle = std::atan(sinHalfAngle / q.w()) * 2.;
    return q.vec() / sinHalfAngle * angle;
}

Vector3 calcApproxRate(const Mat33d& prev, const Mat33d& next, double dt) {
    using Q = Eigen::Quaterniond;
    Q qa = Q(prev);
    Q qb = Q(next);

    qa.normalize();
    qb.normalize();

    return calcRotationVector(qa.inverse() * qb) / dt;
}

Vector3 calcApproxRate(const Trihedron& prev, const Trihedron& next, double dt) {
    return calcApproxRate(prev.R(), next.R(), dt);
}

Vector3 calcApproxRate(const Tri& prev, const Tri& next, double dt) {
    return calcApproxRate(AsMat3(prev), AsMat3(next), dt);
}

bool RunRungeKutta4Test(std::ostream& os)
{
    os << "Start all wrapping tests";

    return true;
}

std::vector<Tri> RunImplicitGeodesicShooterTest(
        const WrapObstacle& s,
        GeodesicTestBounds bnds,
        TestRapport& o)
{
    const Geodesic g = s.getGeodesic();
    const double l = g.length;
    const size_t n = bnds.integratorSteps;

    Tri K_P {g.K_P.p(), g.K_P.t(), g.K_P.t(), g.K_P.n(), g.K_P.b()};

    std::vector<Tri> samples = calcImplicitGeodesic(s, K_P, l, n);

    // Basic tests:
    {
        o.newSubSection("Shooter basics");
        o.assertEq(samples.size(), n+1, "Number of samples from integrator");

        // Darboux frame directions must be perpendicular.
        RunTrihedronTest(K_P, "Initial trihedron", o, bnds.eps);
        RunTrihedronTest(samples.back(), "Final trihedron", o, bnds.eps);
        for (const Tri& y: samples) {
            if (!o.success()) { break; }
            RunTrihedronTest(y, "Intermediate trihedron", o, bnds.eps);
        }
    }

    o.newSubSection("Shooter surface drift");
    {
        // Assert if points lie on the surface.
        o.assertEq(s.calcSurfaceConstraint(samples.front().p), 0., "Start position on surface", bnds.eps);
        o.assertEq(s.calcSurfaceConstraint(samples.back().p), 0., "End position on surface", bnds.eps);
        for (const Tri& K: samples) {
            if (!o.success()) { break; }
            o.assertEq(s.calcSurfaceConstraint(K.p), 0., "Intermediate position on surface", bnds.eps);
        }

    }

    o.newSubSection("Shooter darboux drift");
    if (samples.size() > 2)
    {

        auto TrihedronAssertionHelper = [&](
                const Tri& prev,
                const Tri& next,
                const std::string msg)
        {
            Tri K_est;
            K_est.p = prev.p;
            K_est.v = prev.v;
            K_est.t = (next.p - prev.p) / (next.p - prev.p).norm();
            K_est.n = s.calcSurfaceNormal(prev.p);
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

        Tri prev = samples.front();
        for (size_t i = 1; i < samples.size(); ++i) {
            if (!o.success()) { break; }
            const Tri& next = samples.at(i);
            TrihedronAssertionHelper(prev, next, "Intermediate");
            prev = next;
        }
    }

    // Compare integrated to implicit version.
    o.newSubSection("Shooter vs implicit integrator");
    {
        // Start trihedron test.
        {
            Tri K_P = samples.front();
            o.assertEq(K_P.p, g.K_P.p(), "Start position match", bnds.eps);
            o.assertEq(K_P.t, g.K_P.t(), "Start tangent  match", bnds.eps);
            o.assertEq(K_P.n, g.K_P.n(), "Start normal   match", bnds.eps);
            o.assertEq(K_P.b, g.K_P.b(), "Start binormal match", bnds.eps);
        }
        // End trihedron test.
        {
            Tri K_Q = samples.back();
            o.assertEq(K_Q.p, g.K_Q.p(), "End position match", bnds.eps);
            o.assertEq(K_Q.t, g.K_Q.t(), "End tangent  match", bnds.eps);
            o.assertEq(K_Q.n, g.K_Q.n(), "End normal   match", bnds.eps);
            o.assertEq(K_Q.b, g.K_Q.b(), "End binormal match", bnds.eps);
        }
    }

    return samples;
}

void BinormalVariationTest(
    WrapObstacle& s,
    std::vector<Tri> samplesZero,
    Geodesic::Correction c,
    GeodesicTestBounds bnds,
    const std::string& msg,
    TestRapport& o)
{
    const Geodesic gZero = s.getGeodesic();

    auto ToDarbouxFrame = [&](const Trihedron q, Vector3 v) -> Vector3
    {
        return Vector3{q.t().dot(v), q.n().dot(v), q.b().dot(v)};
    };

    const double kn_P = s.calcNormalCurvature(gZero.K_P.p(), gZero.K_P.t());
    const double tau_P = s.calcGeodesicTorsion(gZero.K_P.p(), gZero.K_P.t());

    Vector3 v_P = ToDarbouxFrame(gZero.K_P, gZero.v_P * c);
    Vector3 w_P = ToDarbouxFrame(gZero.K_P, gZero.w_P * c);
    Vector3 v_Q = ToDarbouxFrame(gZero.K_Q, gZero.v_Q * c);
    Vector3 w_Q = ToDarbouxFrame(gZero.K_Q, gZero.w_Q * c);

    const double kn_Q = s.calcNormalCurvature(gZero.K_Q.p(), gZero.K_Q.t());
    const double tau_Q = s.calcGeodesicTorsion(gZero.K_Q.p(), gZero.K_Q.t());

    o._oss << o._indent << "kn_P  = " << kn_P << "\n";
    o._oss << o._indent << "tau_P = " << tau_P << "\n";
    o._oss << o._indent << "kn_Q  = " << kn_Q << "\n";
    o._oss << o._indent << "tau_Q = " << tau_Q << "\n";

    const double d = bnds.variation;
    c *= d;

    s.applyVariation(c);

    auto ToTriFrame = [&](const Tri q, Vector3 v) -> Vector3
    {
        return Vector3{q.t.dot(v), q.n.dot(v), q.b.dot(v)};
    };

    std::vector<Tri> samplesOne = RunImplicitGeodesicShooterTest(s, bnds, o);

    o._verbose = true;
    o.newSubSection(msg + " => shooter v_P, w_P");
    {
        const Tri& K_P_zero = samplesZero.front();
        const Tri& K_P_one = samplesOne.front();
        o.assertEq(ToTriFrame(K_P_zero, K_P_one.p - K_P_zero.p) / d, v_P,       "Shooter v_P", bnds.varEps);
        o.assertEq(calcApproxRate(K_P_zero, K_P_one, d), w_P, "Shooter w_P", bnds.varEps);
    }

    o.newSubSection(msg + " => shooter v_Q, w_Q");
    {
        const Tri& K_Q_zero = samplesZero.back();
        const Tri& K_Q_one = samplesOne.back();
        o.assertEq(ToTriFrame(K_Q_zero, K_Q_one.p - K_Q_zero.p) / d, v_Q,       "Shooter v_Q", bnds.varEps);
        o.assertEq(calcApproxRate(K_Q_zero, K_Q_one, d), w_Q, "Shooter w_Q", bnds.varEps);
    }

    const Geodesic gOne = s.getGeodesic();

    o.newSubSection(msg + " => geodesic v_P, w_P");
    {
        const Trihedron& K_P_zero = gZero.K_P;
        const Trihedron& K_P_one  = gOne.K_P;
        const Vector3 dp = gOne.K_P.p() - gZero.K_P.p();
        o.assertEq(ToDarbouxFrame(K_P_zero, dp) / d, v_P,                           "Geodesic v_P", bnds.varEps);
        o.assertEq(calcApproxRate(K_P_zero, K_P_one, d), w_P, "Geodesic w_P", bnds.varEps);
    }

    o.newSubSection(msg + " => geodesic v_Q, w_Q");
    {
        const Trihedron& K_Q_zero = gZero.K_Q;
        const Trihedron& K_Q_one  = gOne.K_Q;
        const Vector3 dp = gOne.K_Q.p() - gZero.K_Q.p();
        o.assertEq(ToDarbouxFrame(K_Q_zero, dp) / d, v_Q,                           "Geodesic v_Q", bnds.varEps);
        o.assertEq(calcApproxRate(K_Q_zero, K_Q_one, d), w_Q, "Geodesic w_Q", bnds.varEps);
    }
    o._verbose = false;

    // Reset to original geodesic.
    s.calcGeodesic({gZero.K_P.p(), gZero.K_P.t(), gZero.length});
}

void RunImplicitGeodesicVariationTest(
    WrapObstacle& s,
    GeodesicTestBounds bnds,
    TestRapport& o)
{
    // Verify geodesic numerically.
    o.newSection("Unconstrained comparison");
    std::vector<Tri> samplesZero = RunImplicitGeodesicShooterTest(s, bnds, o);

    if (!o.success()) {
        return;
    }

    o.newSection("Tangential variation");
    BinormalVariationTest(s, samplesZero, {1., 0., 0., 0.}, bnds, "ds", o);
    o.newSection("Binormal variation");
    BinormalVariationTest(s, samplesZero, {0., 1., 0., 0.}, bnds, "dB", o);
    o.newSection("Directional variation");
    BinormalVariationTest(s, samplesZero, {0., 0., 1., 0.}, bnds, "do", o);
    o.newSection("Lengthening variation");
    BinormalVariationTest(s, samplesZero, {0., 0., 0., 1.}, bnds, "dl", o);
    o.newSection("Mixed variation");
    BinormalVariationTest(s, samplesZero, {1., 1., 1., 1.}, bnds, "dl", o);
    o.newSection("Mixed signed variation");
    BinormalVariationTest(s, samplesZero, {1., -1., 1., -1.}, bnds, "dl", o);
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
        }
        if (isOk && _verbose) {
            _oss << _indent << "ASSERTION SUCCESS:   " << msg << "\n";
        }
        if (!isOk || _verbose) {
            _oss << _indent << "    lhs = " << lhs.transpose() << "\n";
            _oss << _indent << "    rhs = " << rhs.transpose() << "\n";
            _oss << _indent << "    err = " << (lhs - rhs).transpose() << " --> " << (lhs - rhs).norm() << "\n";
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
        }
        if (isOk && _verbose) {
            _oss << _indent << "ASSERTION SUCCESS:   " << msg << "\n";
        }
        if (!isOk || _verbose) {
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
        }
        if (isOk && _verbose) {
            _oss << _indent << "ASSERTION SUCCESS:   " << msg << "\n";
        }
        if (!isOk || _verbose) {
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
        }
        if (isOk && _verbose) {
            _oss << _indent << "ASSERTION SUCCESS:   " << msg << "\n";
        }
        if (!isOk || _verbose) {
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

bool RunGeodesicTest(
        WrapObstacle& s,
        GeodesicTestBounds bnds,
        const std::string& name,
        std::ostream& os)
{
    TestRapport o(name);
    RunImplicitGeodesicVariationTest(s, bnds, o);
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

std::vector<Trihedron> calcImplicitTestSamples(
    const WrapObstacle& s,
    Trihedron K,
    double l,
    size_t steps)
{
    Tri K_P {K.p(), K.t(), K.t(), K.n(), K.b()};
    std::vector<Trihedron> samples;
    for (const Tri& K: calcImplicitGeodesic(s, K_P, l, steps))
    {
        samples.push_back({K.p, {K.t, K.n, K.b}});
    }
    return samples;
}

} // namespace osc
