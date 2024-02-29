#include "WrappingMath.h"
#include "WrappingTab.h"

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
    constexpr CStringView c_TabStringID = "OpenSim/Experimental/Wrapping";

    struct SceneSphereSurface final
    {

        explicit SceneSphereSurface(Vec3 pos_, double r_) : pos{pos_}, r{r_}
        {}

        Vec3 pos;
        double r;
    };

    struct SceneCurveSegment final
    {

        explicit SceneCurveSegment(Vec3 pos0, Vec3 pos1) :
            pos{pos0}, diff{pos1 - pos0}
        {}

        Vec3 pos;
        Vec3 diff;
    };

    std::vector<SceneCurveSegment> GenerateSceneCurveSegments()
    {
        std::vector<SceneCurveSegment> rv;
        rv.emplace_back(SceneCurveSegment{
            Vec3{0., 0., 0.},
            Vec3{5., 1., 1.},
        });
        return rv;
    }

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

} // namespace

class osc::WrappingTab::Impl final : public StandardTabImpl
{
public:

    Impl() : StandardTabImpl{c_TabStringID}
    {
        m_Camera.setBackgroundColor({1.0f, 1.0f, 1.0f, 0.0f});
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

        // render sphere
        Graphics::DrawMesh(
            m_SphereMesh,
            {.position = m_SceneSphereSurface.pos},
            m_Material,
            m_Camera,
            m_RedColorMaterialProps);

        // render curve
        for (SceneCurveSegment const& curve : m_SceneCurveSegments) {
            Graphics::DrawMesh(
                m_LineMesh,
                {.scale = curve.diff, .position = curve.pos},
                m_Material,
                m_Camera,
                m_BlackColorMaterialProps);
        }

        Rect const viewport = GetMainViewportWorkspaceScreenRect();

        // draw scene to screen
        m_Camera.setPixelRect(viewport);
        m_Camera.renderToScreen();
    }

    // Wrapping stuff.
    Geodesic  m_Geodesic = Geodesic();

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
        GeneratePropertyBlock({0.0f, 0.0f, 1.0f, 1.0f});
    MaterialPropertyBlock m_RedColorMaterialProps =
        GeneratePropertyBlock({1.0f, 0.0f, 0.0f, 1.0f});

    // scene state
    SceneSphereSurface m_SceneSphereSurface = SceneSphereSurface(Vec3{0., 0., 0.}, 1.);
    std::vector<SceneCurveSegment> m_SceneCurveSegments =
        GenerateSceneCurveSegments();
    bool m_IsMouseCaptured                  = false;
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
