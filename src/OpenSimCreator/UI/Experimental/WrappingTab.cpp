#include "WrappingTab.h"

#include "OpenSimCreator/UI/Experimental/WrappingTest.h"
#include "WrappingMath.h"
#include <SDL_events.h>
#include <array>
#include <cstdint>
#include <eigen3/Eigen/Geometry>
#include <limits>
#include <memory>
#include <optional>
#include <oscar/oscar.h>
#include <vector>

using namespace osc;



namespace
{
    Mesh GenerateXYZToXYZLineMesh()
    {
        Mesh data;
        data.setVerts({{0.0f, 0.0f, 0.0f}, {+1.0f, +1.0f, +1.0f}});
        data.setNormals({{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}});
        data.setIndices({0, 1});
        data.setTopology(MeshTopology::Lines);
        return data;
    }

    Vector3 ToVector3(Vec3 v)
    {
        return Vector3{
            static_cast<double>(v.x),
            static_cast<double>(v.y),
            static_cast<double>(v.z),
        };
    }

    Vec3 ToVec3(const Vector3& v)
    {
        return Vec3{
            static_cast<float>(v.x()),
            static_cast<float>(v.y()),
            static_cast<float>(v.z()),
        };
    }

    constexpr CStringView c_TabStringID = "OpenSim/Experimental/Wrapping";

    constexpr size_t NUM_SURFACES = 4;
    constexpr std::array<CStringView, NUM_SURFACES> c_SurfNames = {"Ellipsoid", "Cylinder", "Torus", "Sphere"};

    struct SceneSurface final
    {
        std::unique_ptr<Surface> m_Surface;
        Mesh m_Mesh{};
        Vec3 m_Scale {1., 1., 1.};
        Quat m_Quat {1., 0., 0., 0.};
    };

    SceneSurface CreateSphere(double radius)
    {
        return SceneSurface{
            std::make_unique<ImplicitSphereSurface>(ImplicitSphereSurface(radius)),
                SphereGeometry(1.),
                Vec3{static_cast<float>(radius)},
                Quat{1., 0., 0., 0.}};
    }

    SceneSurface CreateCylinder(float radius, float length)
    {
        SceneSurface out;
        out.m_Surface = std::make_unique<ImplicitCylinderSurface>(ImplicitCylinderSurface(radius));
        out.m_Mesh    = CylinderGeometry(1., 1., 1., 45, 1, false, Radians{0.}, Radians{2. * M_PI});
        out.m_Scale   = Vec3{radius, length, radius};
        Rotation q    = Rotation(Eigen::AngleAxisd(M_PI/2., Vector3{1., 0., 0.}));
        out.m_Quat    = Quat{static_cast<float>(q.w()),
                static_cast<float>(q.x()),
                static_cast<float>(q.y()),
                static_cast<float>(q.z()),};
        return out;
    }

    SceneSurface CreateEllipsoid(float rx, float ry, float rz)
    {
        SceneSurface out;
        out.m_Surface = std::make_unique<ImplicitEllipsoidSurface>(ImplicitEllipsoidSurface(rx, ry, rz));
        out.m_Mesh    = SphereGeometry(1.);
        out.m_Scale   = Vec3{rx, ry, rz};
        return out;
    }

    SceneSurface CreateTorus(float r, float R)
    {
        SceneSurface out;
        out.m_Surface = std::make_unique<ImplicitTorusSurface>(ImplicitTorusSurface(r, R));
        out.m_Mesh    = TorusGeometry(1., r / R, 12);
        out.m_Scale   = Vec3{R};
        return out;
    }

    void RenderSurface(const SceneSurface& x, const MeshBasicMaterial& material, Camera& camera,
        const MeshBasicMaterial::PropertyBlock& color)
    {
        Graphics::DrawMesh(
                x.m_Mesh,
                {
                .scale    = x.m_Scale,
                .position = ToVec3(
                        x.m_Surface->getOffsetFrame().position),
                },
                material,
                camera,
                color);
    }

    struct PathTerminalPoint
    {
        float min = -10.;
        float max = 10.;
        Vec3 point {0., 0., 0.};
    };

    void UpdWrappingPath(
            const std::vector<SceneSurface>& s,
            WrappingPath& p,
            const WrappingArgs& args,
            const Vec3 start,
            const Vec3 end)
    {
        const size_t n = s.size();
        WrappingPath::GetSurfaceFn GetSurface = [&](size_t i) -> const Surface*
        {
            if (i >= n) {
                return nullptr;
            }
            return s.at(i).m_Surface.get();
        };
        if (n == 0) {
            return;
        }
        if (n != p.segments.size() || !args.m_Cache) {
            p = WrappingPath(ToVector3(start), ToVector3(end), GetSurface);
        } else {
            p.startPoint = ToVector3(start);
            p.endPoint = ToVector3(end);
            p.updPath(GetSurface);
        }
    }

} // namespace

class osc::WrappingTab::Impl final : public StandardTabImpl
{

public:

    Impl() : StandardTabImpl{c_TabStringID}
    {
        // Add an ellipsoid.
        {
            m_Surface.emplace_back(CreateEllipsoid(1., 5., 2.));
            m_Surface.back().m_Surface->setLocalPathStartGuess({-1., 1., 1});
            m_Surface.back().m_Surface->setOffsetFrame({{-3., 0., 0.}});
        }
        const bool ok = false;

        if(ok)
        {
            m_Surface.emplace_back(CreateSphere(1.));
            m_Surface.back().m_Surface->setLocalPathStartGuess({-2., 2., 2});
            m_Surface.back().m_Surface->setOffsetFrame({{-1., 1., 1.}});
        }

        if(ok)
        {
            m_Surface.emplace_back(CreateTorus(1., 0.1f));
            m_Surface.back().m_Surface->setLocalPathStartGuess({-1., 1., 1});
            m_Surface.back().m_Surface->setOffsetFrame({{-3., 0., 0.}});
        }

        if(ok)
        {
            m_Surface.emplace_back(CreateCylinder(1., 10.));
            m_Surface.back().m_Surface->setLocalPathStartGuess({-1., 1., 1});
            m_Surface.back().m_Surface->setOffsetFrame({{-3., 0., 0.}});
        }

        // Configure sphere material.
        m_TransparantMaterial.setTransparent(true);

        m_Camera.setBackgroundColor({1.0f, 1.0f, 1.0f, 0.0f});
        m_Camera.setPosition({0., 0., 5.});
    }

private:

    void implOnMount() final
    {
        App::upd().makeMainEventLoopPolling();
        m_IsMouseCaptured = true;
    }

    void implOnUnmount() final
    {
        m_IsMouseCaptured = false;
        App::upd().makeMainEventLoopWaiting();
        App::upd().setShowCursor(true);
    }

    bool implOnEvent(SDL_Event const& e) final
    {
        if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) {
            m_IsMouseCaptured = false;
            return true;
        } else if (
            e.type == SDL_MOUSEBUTTONDOWN
            && ui::IsMouseInMainViewportWorkspaceScreenRect()) {
            m_IsMouseCaptured = true;
            return true;
        }
        return false;
    }

    void implOnTick() final
    {

        // Create path anew, or start from previous.
        if (m_SingleStep) {
            m_FreezePath = false;
        }

        if (!m_FreezePath) {
            UpdWrappingPath(m_Surface, m_WrappingPath, m_WrappingArgs, m_StartPoint.point, m_EndPoint.point);
        }

        if (m_SingleStep && false) {
            m_FreezePath = true;
            m_SingleStep = false;

            // Do the self test.
            {
                GeodesicTestBounds bnds;
                const size_t n = m_WrappingPath.segments.size();
                if (n == m_Surface.size()) {
                    for (size_t i = 0; i < n; ++i) {
                        std::cout << "\n";
                        const ImplicitEllipsoidSurface* s = dynamic_cast<ImplicitEllipsoidSurface*>(m_Surface.back().m_Surface.get());
                        if (!s) {
                            std::cout << "Skipping self test\n";
                            continue;
                        }
                        const Geodesic& g = m_WrappingPath.segments.at(i);
                        RunImplicitGeodesicTest(*s, g, bnds, "surface", std::cout);
                    }
                }
            }

            std::cout << "\n";
            std::cout << "pathError: " << m_WrappingPath.smoothness.updPathError().transpose() << "\n";
            std::cout << "pathJacobian:\n";
            std::cout << m_WrappingPath.smoothness.updPathErrorJacobian() << "\n";
            std::cout << "CORRECTION: " << m_WrappingPath.smoothness._pathCorrections.transpose() << "\n";

            std::cout << "\n";

            std::cout << "_costL:\n";
            std::cout << m_WrappingPath.smoothness._costL << "\n";
            std::cout << "_vecL: ";
            std::cout << m_WrappingPath.smoothness._vecL.transpose() << "\n";
            std::cout << "_lengthJacobian: ";
            std::cout << m_WrappingPath.smoothness._lengthJacobian.transpose() << "\n";
            std::cout << "_length: ";
            std::cout << m_WrappingPath.smoothness._length << "\n";

            std::cout << "_costP:\n";
            std::cout << m_WrappingPath.smoothness._costP << "\n";
            std::cout << "_costQ:\n";
            std::cout << m_WrappingPath.smoothness._costQ << "\n";

            std::cout << "\n";

            std::cout << "_matSmall:\n";
            std::cout << m_WrappingPath.smoothness._matSmall << "\n";
            std::cout << "_vecSmall: " << m_WrappingPath.smoothness._vecSmall.transpose() << "\n";

            std::cout << "\n";

            std::cout << "mat:\n";
            std::cout << m_WrappingPath.smoothness._mat << "\n";
            std::cout << "vec: " << m_WrappingPath.smoothness._vec.transpose() << "\n";

            std::cout << "\n";

            std::cout << "status" << m_WrappingPath.status << "\n";
            for (const Geodesic& s: m_WrappingPath.segments) {
                std::cout << "    " << s << "\n";
            }
            std::cout << "\n";
            std::cout << "\n";
        }

        bool error = m_WrappingPath.status > 0;
        for (const Geodesic& s: m_WrappingPath.segments) {
            /* error |= s.status > 0; */
            error |= (s.status & Geodesic::Status::InitialTangentParallelToNormal) > 0;
            error |= (s.status & Geodesic::Status::PrevLineSegmentInsideSurface) > 0;
            error |= (s.status & Geodesic::Status::NextLineSegmentInsideSurface) > 0;
            /* error |= (s.status & Geodesic::Status::NegativeLength) > 0; */
            /* error |= (s.status & Geodesic::Status::LiftOff) > 0; */
            /* error |= (s.status & Geodesic::Status::TouchDownFailed) > 0; */
            error |= (s.status & Geodesic::Status::IntegratorFailed) > 0;
        }
        if (error && !m_FreezePath) {
            std::cout << "    " << m_WrappingPath.status << "\n";
            for (const Geodesic& s: m_WrappingPath.segments) {
                std::cout << "    " << s << "\n";
            }
            m_FreezePath = true;
        }
    }

    void implOnDraw() final
    {
        // handle mouse capturing
        if (m_IsMouseCaptured) {
            ui::UpdateEulerCameraFromImGuiUserInput(m_Camera, m_CameraEulers);
            ui::SetMouseCursor(ImGuiMouseCursor_None);
            App::upd().setShowCursor(false);
        } else {
            ui::SetMouseCursor(ImGuiMouseCursor_Arrow);
            App::upd().setShowCursor(true);
        }

        if (ui::Begin("viewer")) {
            // Control path terminal points.
            ImGui::SliderFloat3("p_O", m_StartPoint.point.begin(), m_EndPoint.min, m_EndPoint.max);
            ImGui::SliderFloat3("p_I", m_EndPoint.point.begin(), m_EndPoint.min, m_EndPoint.max);

            // Set wrapping args.
            ui::Checkbox("Cache path", &m_WrappingArgs.m_Cache);

            // Some simulation args.
            ui::Checkbox("Freeze", &m_FreezePath);
            ui::Checkbox("Single", &m_SingleStep);

            // Surface specific stuff.
            {
                for (size_t i = 0; i < m_Surface.size(); ++i) {
                    if (m_Surface.size() != m_WrappingPath.segments.size()) break;

                    ui::Text(c_SurfNames.at(i));

                    Geodesic& g = m_WrappingPath.segments.at(i);
                    Surface* s = m_Surface.at(i).m_Surface.get();

                    // Enable / disable surface.
                    bool enabled = (g.status & Geodesic::Status::Disabled) == 0;
                    ui::Checkbox("Active", &enabled);
                    g.status = enabled
                        ? g.status & ~Geodesic::Status::Disabled
                        : g.status | Geodesic::Status::Disabled;

                    Transf transform = s->getOffsetFrame();
                    Vec3 offset = ToVec3(transform.position);
                    ImGui::SliderFloat3("p_S", offset.begin(), m_EndPoint.max, m_EndPoint.min);
                    transform.position = ToVector3(offset);
                    s->setOffsetFrame(transform);
                }
            }
        }

        // Render surfaces.
        {
            for (const SceneSurface& s: m_Surface) {
                RenderSurface(s, m_TransparantMaterial, m_Camera, m_GreenColorMaterialProps);
            }
        }

        // Render path.
        {
            for (size_t i = 0; i < m_WrappingPath.segments.size(); ++i) {
                if (m_Surface.size() != m_WrappingPath.segments.size()) break;
                Geodesic& g = m_WrappingPath.segments.at(i);

                if (g.samples.empty()) continue;
                /* Surface* s = m_Surface.at(i).m_Surface.get(); */

                Graphics::DrawMesh(
                        m_SphereMesh,
                        {
                        .scale    = {0.01, 0.01, 0.01},
                        .position = ToVec3(g.K_P.p()),
                        },
                        m_Material,
                        m_Camera,
                        m_RedColorMaterialProps);
            }
        }

        // render curve
        auto DrawCurveSegmentMesh = [&](Vec3 p0, Vec3 p1, const MeshBasicMaterial::PropertyBlock& color) {
            Graphics::DrawMesh(
                    m_LineMesh,
                    {.scale = p1 - p0, .position = p0},
                    m_Material,
                    m_Camera,
                    color);
        };
        {
            Vec3 prev              = m_StartPoint.point;
            for (const Geodesic& geodesic : m_WrappingPath.segments) {

                // Iterate over the logged points in the Geodesic.
                for (const Trihedron& s: geodesic.samples) {
                    const Vec3 next = ToVec3(s.p());
                    DrawCurveSegmentMesh(prev, next, m_RedColorMaterialProps);
                    prev = next;
                }
            }
            const Vec3 next = m_EndPoint.point;
            DrawCurveSegmentMesh(prev, next, m_RedColorMaterialProps);

            for (const Geodesic& g: m_WrappingPath.segments) {
                if (g.samples.empty()) continue;
                DrawCurveSegmentMesh(ToVec3(g.K_P.p()), ToVec3(g.K_P.p() + g.K_P.n()), m_GreenColorMaterialProps);
            }
        }

        Rect const viewport = ui::GetMainViewportWorkspaceScreenRect();

        // draw scene to screen
        m_Camera.setPixelRect(viewport);
        m_Camera.renderToScreen();

    }

    // Wrapping stuff.
    PathTerminalPoint m_StartPoint {};
    PathTerminalPoint m_EndPoint {};

    WrappingPath m_WrappingPath = WrappingPath();
    WrappingArgs m_WrappingArgs {};
    std::vector<SceneSurface> m_Surface {};

    ResourceLoader m_Loader = App::resource_loader();
    Camera m_Camera;
    MeshBasicMaterial m_TransparantMaterial{};
    MeshBasicMaterial m_Material{};

    Mesh m_SphereMesh = SphereGeometry(1.0f, 24, 24);
    Mesh m_LineMesh   = GenerateXYZToXYZLineMesh();

    MeshBasicMaterial::PropertyBlock m_BlackColorMaterialProps
        {Color::black()};
    MeshBasicMaterial::PropertyBlock  m_BlueColorMaterialProps
        {Color::blue()};
    MeshBasicMaterial::PropertyBlock m_GreenColorMaterialProps
        {Color::dark_green().with_alpha(0.2f)};
    MeshBasicMaterial::PropertyBlock m_RedColorMaterialProps
        {Color::red()};
    MeshBasicMaterial::PropertyBlock m_GreyColorMaterialProps
        {Color::half_grey().with_alpha(0.2f)};
    MeshBasicMaterial::PropertyBlock m_VariationColorMaterialProps {Color::purple()};

    // scene state
    bool m_FreezePath = false;
    bool m_SingleStep = false;

    bool m_IsMouseCaptured = false;

    Eulers m_CameraEulers{};
};

// public API

CStringView osc::WrappingTab::id()
{
    return c_TabStringID;
}

osc::WrappingTab::WrappingTab(ParentPtr<ITabHost> const&) :
    m_Impl{std::make_unique<Impl>()}
{}

osc::WrappingTab::WrappingTab(WrappingTab&&) noexcept                 = default;
osc::WrappingTab& osc::WrappingTab::operator=(WrappingTab&&) noexcept = default;
osc::WrappingTab::~WrappingTab() noexcept                             = default;

UID osc::WrappingTab::implGetID() const
{
    return m_Impl->getID();
}

CStringView osc::WrappingTab::implGetName() const
{
    return m_Impl->getName();
}

void osc::WrappingTab::implOnMount()
{
    m_Impl->onMount();
}

void osc::WrappingTab::implOnUnmount()
{
    m_Impl->onUnmount();
}

bool osc::WrappingTab::implOnEvent(SDL_Event const& e)
{
    return m_Impl->onEvent(e);
}

void osc::WrappingTab::implOnTick()
{
    m_Impl->onTick();
}

void osc::WrappingTab::implOnDraw()
{
    m_Impl->onDraw();
}
