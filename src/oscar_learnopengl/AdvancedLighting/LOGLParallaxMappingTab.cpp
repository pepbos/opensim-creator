#include "LOGLParallaxMappingTab.h"

#include <oscar_learnopengl/LearnOpenGLHelpers.h>
#include <oscar_learnopengl/MouseCapturingCamera.h>

#include <oscar/oscar.h>
#include <SDL_events.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

using namespace osc::literals;
using namespace osc;

namespace
{
    constexpr CStringView c_TabStringID = "LearnOpenGL/ParallaxMapping";

    // matches the quad used in LearnOpenGL's parallax mapping tutorial
    Mesh GenerateQuad()
    {
        Mesh rv;
        rv.setVerts({
            {-1.0f,  1.0f, 0.0f},
            {-1.0f, -1.0f, 0.0f},
            { 1.0f, -1.0f, 0.0f},
            { 1.0f,  1.0f, 0.0f},
        });
        rv.setNormals({
            {0.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 1.0f},
        });
        rv.setTexCoords({
            {0.0f, 1.0f},
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f},
        });
        rv.setIndices({
            0, 1, 2,
            0, 2, 3,
        });
        rv.recalculateTangents();
        return rv;
    }

    MouseCapturingCamera CreateCamera()
    {
        MouseCapturingCamera rv;
        rv.setPosition({0.0f, 0.0f, 3.0f});
        rv.setVerticalFOV(45_deg);
        rv.setNearClippingPlane(0.1f);
        rv.setFarClippingPlane(100.0f);
        return rv;
    }

    Material CreateParallaxMappingMaterial(IResourceLoader& rl)
    {
        Texture2D diffuseMap = LoadTexture2DFromImage(
            rl.open("oscar_learnopengl/textures/bricks2.jpg"),
            ColorSpace::sRGB
        );
        Texture2D normalMap = LoadTexture2DFromImage(
            rl.open("oscar_learnopengl/textures/bricks2_normal.jpg"),
            ColorSpace::Linear
        );
        Texture2D displacementMap = LoadTexture2DFromImage(
            rl.open("oscar_learnopengl/textures/bricks2_disp.jpg"),
            ColorSpace::Linear
        );

        Material rv{Shader{
            rl.slurp("oscar_learnopengl/shaders/AdvancedLighting/ParallaxMapping.vert"),
            rl.slurp("oscar_learnopengl/shaders/AdvancedLighting/ParallaxMapping.frag"),
        }};
        rv.setTexture("uDiffuseMap", diffuseMap);
        rv.setTexture("uNormalMap", normalMap);
        rv.setTexture("uDisplacementMap", displacementMap);
        rv.setFloat("uHeightScale", 0.1f);
        return rv;
    }

    Material CreateLightCubeMaterial(IResourceLoader& rl)
    {
        return Material{Shader{
            rl.slurp("oscar_learnopengl/shaders/LightCube.vert"),
            rl.slurp("oscar_learnopengl/shaders/LightCube.frag"),
        }};
    }
}

class osc::LOGLParallaxMappingTab::Impl final : public StandardTabImpl {
public:
    Impl() : StandardTabImpl{c_TabStringID}
    {}

private:
    void implOnMount() final
    {
        m_Camera.onMount();
    }

    void implOnUnmount() final
    {
        m_Camera.onUnmount();
    }

    bool implOnEvent(SDL_Event const& e) final
    {
        return m_Camera.onEvent(e);
    }

    void implOnDraw() final
    {
        m_Camera.onDraw();

        // clear screen and ensure camera has correct pixel rect
        App::upd().clearScreen({0.1f, 0.1f, 0.1f, 1.0f});

        // draw normal-mapped quad
        {
            m_ParallaxMappingMaterial.setVec3("uLightWorldPos", m_LightTransform.position);
            m_ParallaxMappingMaterial.setVec3("uViewWorldPos", m_Camera.getPosition());
            m_ParallaxMappingMaterial.setBool("uEnableMapping", m_IsMappingEnabled);
            Graphics::DrawMesh(m_QuadMesh, m_QuadTransform, m_ParallaxMappingMaterial, m_Camera);
        }

        // draw light source cube
        {
            m_LightCubeMaterial.setColor("uLightColor", Color::white());
            Graphics::DrawMesh(m_CubeMesh, m_LightTransform, m_LightCubeMaterial, m_Camera);
        }

        m_Camera.setPixelRect(GetMainViewportWorkspaceScreenRect());
        m_Camera.renderToScreen();

        ImGui::Begin("controls");
        ImGui::Checkbox("normal mapping", &m_IsMappingEnabled);
        ImGui::End();
    }

    ResourceLoader m_Loader = App::resource_loader();

    // rendering state
    Material m_ParallaxMappingMaterial = CreateParallaxMappingMaterial(m_Loader);
    Material m_LightCubeMaterial = CreateLightCubeMaterial(m_Loader);
    Mesh m_CubeMesh = GenerateLearnOpenGLCubeMesh();
    Mesh m_QuadMesh = GenerateQuad();

    // scene state
    MouseCapturingCamera m_Camera = CreateCamera();
    Transform m_QuadTransform;
    Transform m_LightTransform = {
        .scale = Vec3{0.2f},
        .position = {0.5f, 1.0f, 0.3f},
    };
    bool m_IsMappingEnabled = true;
};


// public API

CStringView osc::LOGLParallaxMappingTab::id()
{
    return c_TabStringID;
}

osc::LOGLParallaxMappingTab::LOGLParallaxMappingTab(ParentPtr<ITabHost> const&) :
    m_Impl{std::make_unique<Impl>()}
{
}

osc::LOGLParallaxMappingTab::LOGLParallaxMappingTab(LOGLParallaxMappingTab&&) noexcept = default;
osc::LOGLParallaxMappingTab& osc::LOGLParallaxMappingTab::operator=(LOGLParallaxMappingTab&&) noexcept = default;
osc::LOGLParallaxMappingTab::~LOGLParallaxMappingTab() noexcept = default;

UID osc::LOGLParallaxMappingTab::implGetID() const
{
    return m_Impl->getID();
}

CStringView osc::LOGLParallaxMappingTab::implGetName() const
{
    return m_Impl->getName();
}

void osc::LOGLParallaxMappingTab::implOnMount()
{
    m_Impl->onMount();
}

void osc::LOGLParallaxMappingTab::implOnUnmount()
{
    m_Impl->onUnmount();
}

bool osc::LOGLParallaxMappingTab::implOnEvent(SDL_Event const& e)
{
    return m_Impl->onEvent(e);
}

void osc::LOGLParallaxMappingTab::implOnDraw()
{
    m_Impl->onDraw();
}
