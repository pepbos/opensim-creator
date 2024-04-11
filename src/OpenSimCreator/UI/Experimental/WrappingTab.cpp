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
#include <stdexcept>
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
    constexpr std::array<CStringView, 2> c_PathNames = {"Red", "Green"};

    struct SceneSurface final
    {
        std::shared_ptr<Transf> m_Offset;
        Mesh m_Mesh{};
        Vec3 m_Scale {1., 1., 1.};
        Quat m_Quat {1., 0., 0., 0.};
        Vec3 m_OffsetPos {0., 0., 0.};
    };

    std::pair<SceneSurface, WrapObstacle> CreateSphere(double radius)
    {
        Transf o = {Vector3{0., 0., 0.}};
        std::shared_ptr<Transf> offset = std::make_shared<Transf>(o);
        WrapObstacle obs  = WrapObstacle::Create<AnalyticSphereSurface>(offset, radius);
        return std::make_pair<SceneSurface, WrapObstacle>(
            SceneSurface{
            offset,
            SphereGeometry(1.),
            Vec3{static_cast<float>(radius)},
            Quat{1., 0., 0., 0.}}, std::move(obs));
    }

    std::pair<SceneSurface, WrapObstacle>
    CreateCylinder(float radius, float length)
    {
        Transf o = {Vector3{0., 0., 0.}};
        std::shared_ptr<Transf> offset = std::make_shared<Transf>(o);
        WrapObstacle obs  = WrapObstacle::Create<ImplicitCylinderSurface>(offset, radius);
        Rotation q    = Rotation(Eigen::AngleAxisd(M_PI/2., Vector3{1., 0., 0.}));

        return std::make_pair<SceneSurface, WrapObstacle>(
                SceneSurface{
                offset,
                CylinderGeometry(1., 1., 1., 45, 1, false, Radians{0.}, Radians{2. * M_PI}),
                Vec3{radius, length, radius},
                Quat{static_cast<float>(q.w()), static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()),}},
                std::move(obs));
    }

    std::pair<SceneSurface, WrapObstacle>
    CreateEllipsoid(float rx, float ry, float rz, double accuracy)
    {
        Transf o = {Vector3{0., 0., 0.}};
        std::shared_ptr<Transf> offset = std::make_shared<Transf>(o);
        WrapObstacle obs  = WrapObstacle::Create<ImplicitEllipsoidSurface>(offset, rx, ry, rz, 1e-9, 1e-1, accuracy);

        return std::make_pair<SceneSurface, WrapObstacle>(
                SceneSurface{
                offset,
                SphereGeometry(1.),
                Vec3{rx, ry, rz},
                Quat{1., 0., 0., 0.}}, std::move(obs));
    }

    std::pair<SceneSurface, WrapObstacle>
    CreateTorus(float r, float R)
    {
        Transf o = {Vector3{0., 0., 0.}};
        std::shared_ptr<Transf> offset = std::make_shared<Transf>(o);
        WrapObstacle obs  = WrapObstacle::Create<ImplicitTorusSurface>(offset, r, r);
        return std::make_pair<SceneSurface, WrapObstacle>(
                SceneSurface{
                offset,
                TorusGeometry(1., r / R, 12),
                Vec3{R},
                Quat{1., 0., 0., 0.}}, std::move(obs));
    }

    void RenderSurface(const SceneSurface& x, const MeshBasicMaterial& material, Camera& camera,
        const MeshBasicMaterial::PropertyBlock& color)
    {
        Graphics::DrawMesh(
                x.m_Mesh,
                {
                .scale    = x.m_Scale,
                .rotation = x.m_Quat,
                .position = ToVec3(
                        x.m_Offset->position),
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

} // namespace

class osc::WrappingTab::Impl final : public StandardTabImpl
{

public:

    Impl() : StandardTabImpl{c_TabStringID}
    {
        auto AppendSurface = [&](std::pair<SceneSurface, WrapObstacle>&& s, Vector3 pathStartGuess, bool var = false)
        {
            size_t i = var ? 0 : 1;
            m_Surface.at(i).push_back(std::move(s.first));
            m_WrappingPath.at(i).updSegments().push_back(std::move(s.second));
            m_WrappingPath.at(i).updSegments().back().setLocalPathStartGuess(pathStartGuess);
        };
        // Add an ellipsoid.
        {
            AppendSurface(CreateEllipsoid(1., 5., 2., m_IntegratorAccuracy), {-1., 1., 1.});
            AppendSurface(CreateEllipsoid(1., 5., 2., m_IntegratorAccuracy), {-1., 1., 1.}, true);
        }
        const bool ok = false;

        if(ok)
        {
            AppendSurface(CreateSphere(1.), {-2., 2., 2});
            AppendSurface(CreateSphere(1.), {-2., 2., 2}, true);
        }

        if(ok)
        {
            AppendSurface(CreateTorus(1., 0.1f),{-2., 2., 2});
            AppendSurface(CreateTorus(1., 0.1f),{-2., 2., 2}, true);
        }

        if(ok)
        {
            AppendSurface(CreateCylinder(1., 10.), {-2., 2., 2});
            AppendSurface(CreateCylinder(1., 10.), {-2., 2., 2}, true);
        }

        // Configure sphere material.
        m_TransparantMaterial.setTransparent(true);

        m_Camera.setBackgroundColor({1.0f, 1.0f, 1.0f, 0.0f});
        m_Camera.setPosition({0., 0., 5.});

        m_StartPoint.point = {-5., 0., 0.};
        m_EndPoint.point = {5., 0.1, 0.1};
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
        RunIntegratorTests();

        // Create path anew, or start from previous.
        if (m_SingleStep) {
            m_FreezePath = false;
        }

        for (size_t i = 0; i < 2; ++i) {
            if (!m_FreezePath) {
                m_WrappingPath.at(i).updStart() = ToVector3(m_StartPoint.point);
                m_WrappingPath.at(i).updEnd() = ToVector3(m_EndPoint.point);
                if (m_Cache) {
                    if (m_SingleStep) {
                        m_WrappingPath.at(i).calcPath(true, 1e-6, 1);
                    } else {
                        m_WrappingPath.at(i).calcPath();
                    }
                } else {
                    m_WrappingPath.at(i).calcInitPath();
                }
            }
        }

        if (m_SingleStep) {
            m_FreezePath = true;
            m_SingleStep = false;

            std::ostringstream oss;
            for (size_t i = 0; i < 2; ++i) {
                std::cout << "\n\n\n";
                size_t j = 0;
                for (WrapObstacle& o: m_WrappingPath.at(i).updSegments()) {
                    std::cout << "\n\n";
                    oss << "Path[" << i << "]";
                    oss << "(Surface[" << j << "])";
                    RunGeodesicTest(o, m_TestBnds,oss.str(), std::cout);
                    oss.clear();
                }
                ++j;
            }

            for (size_t i = 0; i < 2; ++i) {
                m_WrappingPath.at(i).getSolver().print(std::cout);
                std::cout << "status" << m_WrappingPath.at(i).getStatus() << "\n";
                for (const WrapObstacle& o: m_WrappingPath.at(i).getSegments()) {
                    std::cout << "    " << o.getStatus() << "\n";
                    std::cout << "    " << o.getGeodesic() << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        for (size_t i = 0; i < 2; ++i) {
            bool error = m_WrappingPath.at(i).getStatus() > 0;
            for (const WrapObstacle& o: m_WrappingPath.at(i).getSegments()) {
                Geodesic::Status s = o.getStatus();
                /* error |= s.status > 0; */
                error |= (s & Geodesic::Status::InitialTangentParallelToNormal) > 0;
                error |= (s & Geodesic::Status::PrevLineSegmentInsideSurface) > 0;
                error |= (s & Geodesic::Status::NextLineSegmentInsideSurface) > 0;
                /* error |= (s & Geodesic::Status::NegativeLength) > 0; */
                /* error |= (s & Geodesic::Status::LiftOff) > 0; */
                /* error |= (s & Geodesic::Status::TouchDownFailed) > 0; */
                error |= (s & Geodesic::Status::IntegratorFailed) > 0;
            }
            if (error && !m_FreezePath) {
                m_FreezePath = true;
            }
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
            /* ImGui::SliderFloat3("p_I", m_EndPoint.point.begin(), m_EndPoint.min, m_EndPoint.max); */
            ImGui::SliderFloat3("p_I", m_Angles.begin(), -10., 10.);

            {
            float angle  = m_Angles[0];
            float radius = m_Angles[1];
            float height = m_Angles[2];
            m_EndPoint.point = { cos(angle) * radius, height, sin(angle) * radius};
            }

            // Set wrapping args.
            ui::Checkbox("Cache path", &m_Cache);

            // Some simulation args.
            ui::Checkbox("Freeze", &m_FreezePath);
            ui::Checkbox("Single", &m_SingleStep);

            for (size_t i = 0; i < 2; ++i) {
                if (ui::Begin(c_PathNames.at(i))) {

                    /* ui::Checkbox("CostP", &m_WrappingPath.at(i).updOpts().m_CostP); */
                    /* ui::Checkbox("CostQ", &m_WrappingPath.at(i).updOpts().m_CostQ); */

                    /* ui::Checkbox("CostW", &m_WrappingPath.at(i).updOpts().m_CostW); */
                    /* ui::Checkbox("CostT", &m_WrappingPath.at(i).updOpts().m_CostT); */
                    /* ui::Checkbox("CostN", &m_WrappingPath.at(i).updOpts().m_CostN); */
                    /* ui::Checkbox("CostB", &m_WrappingPath.at(i).updOpts().m_CostB); */
                    /* ui::Checkbox("CostL", &m_WrappingPath.at(i).updOpts().m_CostL); */
                    /* ui::Checkbox("AugN", &m_WrappingPath.at(i).updOpts().m_AugN); */

                    char buffer[32];
                    snprintf(buffer, sizeof(buffer), "%zu", m_WrappingPath.at(i).getLoopIter());
                    ui::Text(buffer);

                    snprintf(buffer, sizeof(buffer), "%g", m_WrappingPath.at(i).getPathError());
                    ui::Text(buffer);

                    snprintf(buffer, sizeof(buffer), "%g", m_WrappingPath.at(i).getPathErrorBound());

                    snprintf(buffer, sizeof(buffer), "%zu", m_WrappingPath.at(i).getPathPoints().size());
                    ui::Text(buffer);
                }
            }

            // Surface specific stuff.
            if(false)
            {
                size_t i = 0;
                for (WrapObstacle& o: m_WrappingPath.at(0).updSegments()) {
                    ui::Text(c_SurfNames.at(i));

                    // Enable / disable surface.
                    bool enabled = (o.getStatus() & Geodesic::Status::Disabled) == 0;
                    ui::Checkbox("Active", &enabled);
                    o.updStatus() |= enabled
                        ? o.getStatus() & ~Geodesic::Status::Disabled
                        : o.getStatus() | Geodesic::Status::Disabled;

                    ImGui::SliderFloat3("p_S", m_Surface.at(0).at(i).m_OffsetPos.begin(), m_EndPoint.max, m_EndPoint.min);
                    m_Surface.at(0).at(i).m_Offset->position = ToVector3(m_Surface.at(0).at(i).m_OffsetPos);
                    ++i;
                }
            }
        }

        // Render surfaces.
        for (size_t i = 0; i < 1; ++i) {
            for (const SceneSurface& s: m_Surface.at(i)) {
                RenderSurface(s, m_TransparantMaterial, m_Camera, m_GreenColorMaterialProps);
            }
        }

        float r = 0.05f;

        Graphics::DrawMesh(
                m_SphereMesh,
                {
                .scale    = {r, r, r},
                .position = m_StartPoint.point,
                },
                m_Material,
                m_Camera,
                m_BlueColorMaterialProps);

        Graphics::DrawMesh(
                m_SphereMesh,
                {
                .scale    = {r, r, r},
                .position = m_EndPoint.point,
                },
                m_Material,
                m_Camera,
                m_GreenColorMaterialProps);

        r = 0.02f;

        // render curve
        for (size_t i = 0; i < 2; ++i) {
            if(!m_WrappingPath.at(i).calcPathPoints().empty())
            {
                auto DrawCurveSegmentMesh = [&](Vec3 p0, Vec3 p1, const MeshBasicMaterial::PropertyBlock& color) {
                    Graphics::DrawMesh(
                            m_LineMesh,
                            {.scale = p1 - p0, .position = p0},
                            m_Material,
                            m_Camera,
                            color);
                    Graphics::DrawMesh(
                            m_SphereMesh,
                            {
                            .scale    = {r, r, r},
                            .position = p1,
                            },
                            m_Material,
                            m_Camera,
                            color);
                };
                Vec3 prev              = ToVec3(m_WrappingPath.at(i).getPathPoints().front());
                for (const Vector3& p : m_WrappingPath.at(i).getPathPoints()) {
                    const Vec3 next = ToVec3(p);
                    DrawCurveSegmentMesh(prev, next, i == 0 ? m_RedColorMaterialProps : m_GreenColorMaterialProps);
                    prev = next;
                }
            } else {
                throw std::runtime_error("path points empty");
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

    std::array<WrappingPath,2> m_WrappingPath = {WrappingPath(), WrappingPath()};
    std::array<std::vector<SceneSurface>,2> m_Surface = {};

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
    bool m_Cache = false;
    bool m_SingleStep = false;

    bool m_IsMouseCaptured = false;

    Eulers m_CameraEulers{};
    Vec3 m_Angles = {0., 10., 0.};

    double m_IntegratorAccuracy = 1e-6;
    GeodesicTestBounds m_TestBnds {
        1e-6,
    };
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
