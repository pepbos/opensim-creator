#include "WrappingTab.h"

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
    Vec3 ToVec3(const Vector3& v)
    {
        return Vec3{
            static_cast<float>(v.x()),
            static_cast<float>(v.y()),
            static_cast<float>(v.z()),
        };
    }

    constexpr CStringView c_TabStringID = "OpenSim/Experimental/Wrapping";

    struct SceneSphereSurface final
    {

        explicit SceneSphereSurface(Vec3 pos_, double r_) :
            surface{AnalyticSphereSurface(r_)}, pos{pos_}, r{r_ * 1.0}
        {
            surface.setOffsetFrame(Transf{
                {
                 pos.x,
                 pos.y,
                 pos.z,
                 }
            });
        }

        Vec3 getPosition() const
        {
            return Vec3{
                pos.x,
                pos.y,
                pos.z,
            };
        }

        AnalyticSphereSurface surface;
        Vec3 pos;
        double r;
    };

    struct SceneSphere final
    {

        explicit SceneSphere(Vec3 pos_) : pos{pos_}
        {}

        Vec3 pos;
        bool isHovered = false;
    };

    MaterialPropertyBlock GeneratePropertyBlock(Color const& color)
    {
        MaterialPropertyBlock p;
        p.setColor("uColor", color);
        return p;
    }

    struct PathTerminalPoint
    {
        float radius = 1.;
        float phi    = 0.;
        float theta  = 0.;
    };

    Vector3 ComputePoint(const PathTerminalPoint& pt)
    {
        return {
            pt.radius * cos(pt.phi),                  // z
            pt.radius * sin(pt.phi) * sin(pt.theta),  // y
            -pt.radius * sin(pt.phi) * cos(pt.theta), // x
        };
    }

} // namespace

class osc::WrappingTab::Impl final : public StandardTabImpl
{

    const Surface* getWrapSurfaceHelper(size_t i)
    {
        switch (i) {
            /* case 0: return &m_ImplicitSphereSurface; */
            case 0: return &m_ImplicitEllipsoidSurface;
            /* case 2: return &m_AnalyticSphereSurface; */
            default: return nullptr;
        }
    }

public:

    Impl() : StandardTabImpl{c_TabStringID}
    {
        // Set some surface params.
        {
            // Set surface position.
            m_AnalyticSphereSurface.setOffsetFrame(Transf{
                Vector3{3., 0., 0.1}
            });
            // Set radius.
            m_AnalyticSphereSurface.setRadius(0.5);

            m_ImplicitSphereSurface.setOffsetFrame(Transf{
                Vector3{-3., 0.1, 0.1}
            });
            m_ImplicitSphereSurface.setRadius(0.75);

            m_ImplicitEllipsoidSurface.setOffsetFrame(Transf{
                Vector3{0.2, 0.1, 0.1}
            });
            m_ImplicitEllipsoidSurface.setRadii(1., 1., 3.);
        }

        // Make sure to do all surface self tests (TODO terrible place for it, but whatever.
        {
            m_ImplicitSphereSurface.doSelfTests();
            std::cout << "ImplicitSphereSurface self test OK\n";
            m_AnalyticSphereSurface.doSelfTests();
            std::cout << "AnalyticSphereSurface self test OK\n";
            m_ImplicitEllipsoidSurface.doSelfTests(1e-2);
            std::cout << "ImplicitEllipsoidSurface self test OK\n";
        }

        // Choose wrapping terminal points.
        {
            m_StartPoint                     = {-5., 0.1, 0.};
            m_EndPoint.radius                = 5.;
        }

        // Initialize the wrapping path.
        {
            Surface::GetSurfaceFn GetSurface = [&](size_t i) -> const Surface* {
                return getWrapSurfaceHelper(i);
            };
            m_WrappingPath = Surface::calcNewWrappingPath(
                m_StartPoint,
                ComputePoint(m_EndPoint),
                GetSurface);
        }

        // Configure sphere material.
        m_Material.setTransparent(true);

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
            && IsMouseInMainViewportWorkspaceScreenRect()) {
            m_IsMouseCaptured = true;
            return true;
        }
        return false;
    }

    void implOnTick() final
    {
        // Analytic geodesic computation.
        Surface::GetSurfaceFn GetSurface = [&](size_t i) -> const Surface* {
            return getWrapSurfaceHelper(i);
        };
        m_WrappingPath = Surface::calcNewWrappingPath(
            m_StartPoint,
            ComputePoint(m_EndPoint),
            GetSurface);
        /* m_WrappingPath.endPoint = ComputePoint(m_EndPoint); */
        /* Surface::calcUpdatedWrappingPath(m_WrappingPath, GetSurface); */
    }

    void implOnDraw() final
    {
        // handle mouse capturing
        if (m_IsMouseCaptured) {
            UpdateEulerCameraFromImGuiUserInput(m_Camera, m_CameraEulers);
            ImGui::SetMouseCursor(ImGuiMouseCursor_None);
            App::upd().setShowCursor(false);
        } else {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
            App::upd().setShowCursor(true);
        }

        if (ImGui::Begin("viewer")) {
            ImGui::SliderAngle("phi", &m_EndPoint.phi);
            ImGui::SliderAngle("theta", &m_EndPoint.theta);
        }

        // render sphere && ellipsoid
        {
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

            Graphics::DrawMesh(
                m_SphereMesh,
                {
                    .scale    = ToVec3(m_ImplicitEllipsoidSurface.getRadii()),
                    .position = ToVec3(m_ImplicitEllipsoidSurface.getOffsetFrame().position),
                },
                m_Material,
                m_Camera,
                m_GreenColorMaterialProps);
        }

        // render curve
        {
            Vector3 prev              = m_StartPoint;
            auto DrawCurveSegmentMesh = [&](Vec3 p0, Vec3 p1, bool red = true) {
                Graphics::DrawMesh(
                    m_LineMesh,
                    {.scale = p1 - p0, .position = p0},
                    m_Material,
                    m_Camera,
                    red ? m_RedColorMaterialProps : m_BlackColorMaterialProps);
            };
            for (const Geodesic& geodesic : m_WrappingPath.segments) {

                // Iterate over the logged points in the Geodesic.
                for (const std::pair<Vector3, DarbouxFrame>& knot :
                     geodesic.curveKnots) {
                    const Vector3 next = knot.first;
                    DrawCurveSegmentMesh(ToVec3(prev), ToVec3(next));
                    prev = next;
                }
            }
            const Vector3 next = m_WrappingPath.endPoint;
            DrawCurveSegmentMesh(ToVec3(prev), ToVec3(next));

            DrawCurveSegmentMesh(ToVec3(m_StartPoint), {0., 0., 0.}, false);

            DrawCurveSegmentMesh(
                {0., 0., 0.},
                ToVec3(ComputePoint(m_EndPoint)),
                false);
        }

        Rect const viewport = GetMainViewportWorkspaceScreenRect();

        // draw scene to screen
        m_Camera.setPixelRect(viewport);
        m_Camera.renderToScreen();
    }

    // Wrapping stuff.
    Vector3 m_StartPoint{
        NAN,
        NAN,
        NAN,
    };
    PathTerminalPoint m_EndPoint = {};

    WrappingPath m_WrappingPath = WrappingPath();

    ResourceLoader m_Loader = App::resource_loader();
    Camera m_Camera;
    Material m_Material{
        Shader{
               m_Loader.slurp("oscar_demos/shaders/SolidColor.vert"),
               m_Loader.slurp("oscar_demos/shaders/SolidColor.frag"),
               }
    };

    Mesh m_SphereMesh = GenerateUVSphereMesh(12, 12);
    Mesh m_LineMesh   = GenerateXYZToXYZLineMesh();

    MaterialPropertyBlock m_BlackColorMaterialProps =
        GeneratePropertyBlock({0.0f, 0.0f, 0.0f, 1.0f});
    MaterialPropertyBlock m_BlueColorMaterialProps =
        GeneratePropertyBlock({0.0f, 0.0f, 0.5f, 0.5f});
    MaterialPropertyBlock m_GreenColorMaterialProps =
        GeneratePropertyBlock({0.0f, 0.5f, 0.0f, 0.5f});
    MaterialPropertyBlock m_RedColorMaterialProps =
        GeneratePropertyBlock({1.0f, 0.0f, 0.0f, 1.0f});
    MaterialPropertyBlock m_GreyColorMaterialProps =
        GeneratePropertyBlock({0.5f, 0.5f, 0.5f, 0.5f});

    // scene state
    AnalyticSphereSurface m_AnalyticSphereSurface = AnalyticSphereSurface(1.);
    ImplicitSphereSurface m_ImplicitSphereSurface = ImplicitSphereSurface(1.);

    ImplicitEllipsoidSurface m_ImplicitEllipsoidSurface;

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
