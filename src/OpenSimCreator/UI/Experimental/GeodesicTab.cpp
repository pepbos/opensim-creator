#include "GeodesicTab.h"

#include "OpenSimCreator/UI/Experimental/WrappingTest.h"
#include "WrappingMath.h"
#include <Eigen/src/Geometry/AngleAxis.h>
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
    data.setVerts({
        { 0.0f,  0.0f,  0.0f},
        {+1.0f, +1.0f, +1.0f}
    });
    data.setNormals({
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f}
    });
    data.setIndices({0, 1});
    data.setTopology(MeshTopology::Lines);
    return data;
}
Vec3 ToVec3(const Vector3& v)
{
    return Vec3{
        static_cast<float>(v.x()),
        static_cast<float>(v.y()),
        static_cast<float>(v.z()),
    };
}

constexpr CStringView c_TabStringID = "OpenSim/Experimental/Geodesic";

MaterialPropertyBlock GeneratePropertyBlock(Color const& color)
{
    MaterialPropertyBlock p;
    p.setColor("uColor", color);
    return p;
}

struct StartPoint
{
    double radius              = 1.;
    float length               = 1.;
    std::array<float, 3> angle = {0., 0., 0.};
};

Vector3 ComputePoint(const StartPoint& pt)
{
    Rotation qz =
        Rotation(Eigen::AngleAxisd(pt.angle.at(0), Vector3{0., 0., 1.}));
    Rotation qy =
        Rotation(Eigen::AngleAxisd(pt.angle.at(1), Vector3{0., 1., 0.}));
    return qz * qy * Vector3{-pt.radius, 0., 0.};
}

Vector3 ComputeVelocity(const StartPoint& pt)
{
    Rotation qx =
        Rotation(Eigen::AngleAxisd(pt.angle.at(0), Vector3{1., 0., 0.}));
    Rotation qy =
        Rotation(Eigen::AngleAxisd(pt.angle.at(1), Vector3{0., 1., 0.}));
    Rotation qz =
        Rotation(Eigen::AngleAxisd(pt.angle.at(2), Vector3{0., 0., 1.}));
    return qz * qy * qx * Vector3{0., 0., 1.};
}

enum class ShowSurface
{
    AnalyticSphere,
    ImplicitSphere,
    ImplicitEllipsoid,
    ImplicitCylinder,
    AnalyticCylinder,
    ImplicitTorus,
};

} // namespace

class osc::GeodesicTab::Impl final : public StandardTabImpl
{

public:
    Impl() : StandardTabImpl{c_TabStringID}
    {
        // Set some surface params.
        {
            m_ImplicitCylinderSurface.setRadius(0.75);
            m_ImplicitEllipsoidSurface.setRadii(0.5, 0.75, 1.);
            m_ImplicitSphereSurface.setRadius(0.75);

            m_AnalyticCylinderSurface.setRadius(0.75);
            m_AnalyticSphereSurface.setRadius(0.75);

            {
                const float r = 0.1f;
                const float R = 1.0f;
                m_TorusMesh   = TorusGeometry(R, r, 12);
                m_ImplicitTorusSurface.setRadii(R, r);
            }
        }

        // Choose geodesic start point.
        {
            m_StartPoint.radius = 1.2;
        }

        // Initialize the wrapping path.
        {
            /* m_Geodesic; */
        }

        // Configure sphere material.
        m_Material.setTransparent(true);

        m_Camera.setBackgroundColor({1.0f, 1.0f, 1.0f, 0.0f});
        m_Camera.setPosition({0., 0., 2.});
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
            e.type == SDL_MOUSEBUTTONDOWN &&
            ui::IsMouseInMainViewportWorkspaceScreenRect()) {
            m_IsMouseCaptured = true;
            return true;
        }
        return false;
    }

    void implOnTick() final
    {
        // Analytic geodesic computation.
        Vector3 p0 = ComputePoint(m_StartPoint);
        Vector3 v0 = ComputeVelocity(m_StartPoint);
        double l   = m_StartPoint.length;
        const GeodesicTestBounds bnds;

        switch (m_ShowSurface) {
        case ShowSurface::AnalyticSphere: {
            m_AnalyticSphereSurface.calcGeodesic(p0, v0, l, m_Geodesic);
            break;
        }
        case ShowSurface::ImplicitSphere: {
            m_ImplicitSphereSurface.calcGeodesic(p0, v0, l, m_Geodesic);
            if (m_RunTestReport) {
                RunImplicitGeodesicTest(m_ImplicitSphereSurface, m_Geodesic, bnds, "Sphere", std::cout);
                m_RunTestReport = false;
            }
            break;
        }
        case ShowSurface::ImplicitEllipsoid: {
            m_ImplicitEllipsoidSurface.calcGeodesic(p0, v0, l, m_Geodesic);
            break;
        }
        case ShowSurface::ImplicitCylinder: {
            m_ImplicitCylinderSurface.calcGeodesic(p0, v0, l, m_Geodesic);
            break;
        }
        case ShowSurface::ImplicitTorus: {
            m_ImplicitTorusSurface.calcGeodesic(p0, v0, l, m_Geodesic);
            break;
        }
        default:
            throw std::runtime_error("unknown surface");
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
            ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
            App::upd().setShowCursor(true);
        }

        if (ImGui::Begin("viewer")) {
            ImGui::Checkbox("Report", &m_RunTestReport);
            ImGui::SliderAngle("phi", &m_StartPoint.angle.at(0));
            ImGui::SliderAngle("theta", &m_StartPoint.angle.at(1));
            ImGui::SliderAngle("psi", &m_StartPoint.angle.at(2));
            ImGui::SliderAngle("length", &m_StartPoint.length);
        }

        // render mesh

        switch (m_ShowSurface) {
        case ShowSurface::AnalyticSphere: {
            Graphics::DrawMesh(
                m_SphereMesh,
                {
                    .scale    = Vec3{static_cast<float>(
                        m_AnalyticSphereSurface.getRadius())},
                    .position = ToVec3(
                        m_AnalyticSphereSurface.getOffsetFrame().position),
                },
                m_Material,
                m_Camera,
                m_BlueColorMaterialProps);
            break;
        }
        case ShowSurface::ImplicitSphere: {
            Graphics::DrawMesh(
                m_SphereMesh,
                {
                    .scale    = Vec3{static_cast<float>(
                        m_ImplicitSphereSurface.getRadius())},
                    .position = ToVec3(
                        m_ImplicitSphereSurface.getOffsetFrame().position),
                },
                m_Material,
                m_Camera,
                m_GreenColorMaterialProps);
            break;
        }
        case ShowSurface::ImplicitEllipsoid: {
            Graphics::DrawMesh(
                m_SphereMesh,
                {
                    .scale    = ToVec3(m_ImplicitEllipsoidSurface.getRadii()),
                    .position = ToVec3(
                        m_ImplicitEllipsoidSurface.getOffsetFrame().position),
                },
                m_Material,
                m_Camera,
                m_GreenColorMaterialProps);
            break;
        }
        case ShowSurface::ImplicitCylinder: {
            Rotation q =
                Rotation(Eigen::AngleAxisd(M_PI / 2., Vector3{1., 0., 0.}));
            Quat qf = Quat{
                static_cast<float>(q.w()),
                static_cast<float>(q.x()),
                static_cast<float>(q.y()),
                static_cast<float>(q.z()),
            };
            Graphics::DrawMesh(
                m_CylinderMesh,
                {
                    .scale    = Vec3{static_cast<float>(
                        m_ImplicitCylinderSurface.getRadius())},
                    .rotation = qf,
                    .position = ToVec3(
                        m_ImplicitCylinderSurface.getOffsetFrame().position),
                },
                m_Material,
                m_Camera,
                m_GreenColorMaterialProps);
            break;
        }
        case ShowSurface::ImplicitTorus: {
            Graphics::DrawMesh(
                m_TorusMesh,
                {
                    .position = ToVec3(
                        m_ImplicitTorusSurface.getOffsetFrame().position),
                },
                m_Material,
                m_Camera,
                m_GreenColorMaterialProps);
            break;
        }
        default:
            throw std::runtime_error("unknown surface");
        }

        // render curve
        {
            const Geodesic& g         = m_Geodesic;
            auto DrawCurveSegmentMesh = [&](Vec3 p0, Vec3 p1, bool red = true) {
                Graphics::DrawMesh(
                    m_LineMesh,
                    {.scale = p1 - p0, .position = p0},
                    m_Material,
                    m_Camera,
                    red ? m_RedColorMaterialProps : m_BlackColorMaterialProps);
            };
            Vector3 prev = g.start.position;
            DrawCurveSegmentMesh(
                ToVec3(ComputePoint(m_StartPoint)),
                ToVec3(prev),
                false);

            // Draw no
            DrawCurveSegmentMesh(
                ToVec3(prev),
                ToVec3(prev + g.samples.front().second.t),
                false);
            // Iterate over the logged points in the Geodesic.
            const double tLen = 1e-1;
            for (const std::pair<Vector3, DarbouxFrame>& knot : g.samples) {
                const Vector3 next = knot.first;
                DrawCurveSegmentMesh(ToVec3(prev), ToVec3(next));
                DrawCurveSegmentMesh(
                    ToVec3(next - knot.second.b * tLen),
                    ToVec3(next + knot.second.b * tLen),
                    false);
                prev = next;
            }
            const Vector3 next = g.end.position;
            DrawCurveSegmentMesh(ToVec3(prev), ToVec3(next));
        }

        Rect const viewport = ui::GetMainViewportWorkspaceScreenRect();

        // draw scene to screen
        m_Camera.setPixelRect(viewport);
        m_Camera.renderToScreen();
    }

    // Wrapping stuff.
    StartPoint m_StartPoint = {};

    Geodesic m_Geodesic = {};
    Geodesic m_TestGeodesic = {};

    ResourceLoader m_Loader = App::resource_loader();
    Camera m_Camera;
    Material m_Material{
        Shader{
               m_Loader.slurp("oscar_demos/shaders/SolidColor.vert"),
               m_Loader.slurp("oscar_demos/shaders/SolidColor.frag"),
               }
    };

    Mesh m_SphereMesh   = SphereGeometry(1.0f, 12, 12);
    Mesh m_CylinderMesh = CylinderGeometry(
        1.,
        1.,
        10.,
        45,
        1.,
        false,
        Radians{0.},
        Radians{2. * M_PI});
    Mesh m_LineMesh  = GenerateXYZToXYZLineMesh();
    Mesh m_TorusMesh = TorusGeometry(1., 0.1f);

    MaterialPropertyBlock m_BlackColorMaterialProps =
        GeneratePropertyBlock({0.0f, 0.0f, 0.0f, 1.0f});
    MaterialPropertyBlock m_BlueColorMaterialProps =
        GeneratePropertyBlock({0.0f, 0.0f, 0.5f, 0.5f});
    MaterialPropertyBlock m_GreenColorMaterialProps =
        GeneratePropertyBlock({0.0f, 0.2f, 0.0f, 0.2f});
    MaterialPropertyBlock m_RedColorMaterialProps =
        GeneratePropertyBlock({1.0f, 0.0f, 0.0f, 1.0f});
    MaterialPropertyBlock m_GreyColorMaterialProps =
        GeneratePropertyBlock({0.5f, 0.5f, 0.5f, 0.5f});

    // scene state
    AnalyticCylinderSurface m_AnalyticCylinderSurface =
        AnalyticCylinderSurface(1.);
    AnalyticSphereSurface m_AnalyticSphereSurface = AnalyticSphereSurface(1.);

    ImplicitCylinderSurface m_ImplicitCylinderSurface =
        ImplicitCylinderSurface(1.);
    ImplicitEllipsoidSurface m_ImplicitEllipsoidSurface =
        ImplicitEllipsoidSurface(1., 1., 1.);
    ImplicitSphereSurface m_ImplicitSphereSurface = ImplicitSphereSurface(1.);
    ImplicitTorusSurface m_ImplicitTorusSurface = ImplicitTorusSurface(1., 0.1);

    bool m_IsMouseCaptured = false;
    Eulers m_CameraEulers{};

    ShowSurface m_ShowSurface = ShowSurface::ImplicitSphere;
    bool m_RunTestReport = false;
};

// public API

CStringView osc::GeodesicTab::id()
{
    return c_TabStringID;
}

osc::GeodesicTab::GeodesicTab(ParentPtr<ITabHost> const&) :
    m_Impl{std::make_unique<Impl>()}
{}

osc::GeodesicTab::GeodesicTab(GeodesicTab&&) noexcept                 = default;
osc::GeodesicTab& osc::GeodesicTab::operator=(GeodesicTab&&) noexcept = default;
osc::GeodesicTab::~GeodesicTab() noexcept                             = default;

UID osc::GeodesicTab::implGetID() const
{
    return m_Impl->getID();
}

CStringView osc::GeodesicTab::implGetName() const
{
    return m_Impl->getName();
}

void osc::GeodesicTab::implOnMount()
{
    m_Impl->onMount();
}

void osc::GeodesicTab::implOnUnmount()
{
    m_Impl->onUnmount();
}

bool osc::GeodesicTab::implOnEvent(SDL_Event const& e)
{
    return m_Impl->onEvent(e);
}

void osc::GeodesicTab::implOnTick()
{
    m_Impl->onTick();
}

void osc::GeodesicTab::implOnDraw()
{
    m_Impl->onDraw();
}
