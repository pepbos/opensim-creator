#include "RendererCoordinateSystemsTab.hpp"

#include "src/Bindings/ImGuiHelpers.hpp"
#include "src/Graphics/MeshGen.hpp"
#include "src/Graphics/Renderer.hpp"
#include "src/Graphics/Texturing.hpp"
#include "src/Maths/Constants.hpp"
#include "src/Maths/Geometry.hpp"
#include "src/Maths/Transform.hpp"
#include "src/Platform/App.hpp"
#include "src/Widgets/PerfPanel.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <SDL_events.h>

#include <cstdint>
#include <numeric>
#include <string>
#include <utility>

static char const g_VertexShader[] =
R"(
    #version 330 core

    uniform mat4 uModelMat;
    uniform mat4 uViewMat;
    uniform mat4 uProjMat;

    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 FragTexCoord;

    void main()
    {
	    gl_Position = uProjMat * uViewMat * uModelMat * vec4(aPos, 1.0);
	    FragTexCoord = aTexCoord;
    }
)";

static char const g_FragmentShader[] =
R"(
    #version 330 core

    uniform sampler2D uTexture1;
    uniform sampler2D uTexture2;

    in vec2 FragTexCoord;
    out vec4 FragColor;

    void main()
    {
	    FragColor = mix(texture(uTexture1, FragTexCoord), texture(uTexture2, FragTexCoord), 0.2);
    }
)";

// worldspace positions of each cube (step 2)
static glm::vec3 const g_CubePositions[] =
{
    { 0.0f,  0.0f,  0.0f },
    { 2.0f,  5.0f, -15.0f},
    {-1.5f, -2.2f, -2.5f },
    {-3.8f, -2.0f, -12.3f},
    { 2.4f, -0.4f, -3.5f },
    {-1.7f,  3.0f, -7.5f },
    { 1.3f, -2.0f, -2.5f },
    { 1.5f,  2.0f, -2.5f },
    { 1.5f,  0.2f, -1.5f },
    {-1.3f,  1.0f, -1.5  },
};

// generate a texture-mapped cube
static osc::experimental::Mesh GenerateMesh()
{
    auto generatedCube = osc::GenCube();

    for (glm::vec3& vert : generatedCube.verts)
    {
        vert *= 0.5f;  // makes the verts match LearnOpenGL
    }

    osc::experimental::Mesh m;
    m.setVerts(generatedCube.verts);
    m.setTexCoords(generatedCube.texcoords);
    m.setIndices(generatedCube.indices);
    return m;
}

class osc::RendererCoordinateSystemsTab::Impl final {
public:

    Impl(TabHost* parent) : m_Parent{parent}
    {
        m_Camera.setPosition({0.0f, 0.0f, 3.0f});
        m_Camera.setCameraFOV(glm::radians(45.0f));
        m_Camera.setNearClippingPlane(0.1f);
        m_Camera.setFarClippingPlane(100.0f);
        m_Material.setTexture("uTexture1", osc::experimental::LoadTexture2DFromImageResource("container.jpg"));
        m_Material.setTexture("uTexture2", osc::experimental::LoadTexture2DFromImageResource("awesomeface.png"));
    }

    UID getID() const
    {
        return m_ID;
    }

    CStringView getName() const
    {
        return "Coordinate Systems (LearnOpenGL)";
    }

    TabHost* getParent() const
    {
        return m_Parent;
    }

    void onMount()
    {
        osc::App::upd().makeMainEventLoopPolling();
        m_IsMouseCaptured = true;
    }

    void onUnmount()
    {
        m_IsMouseCaptured = false;
        osc::App::upd().setShowCursor(true);
        osc::App::upd().makeMainEventLoopWaiting();
    }

    bool onEvent(SDL_Event const& e)
    {
        if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)
        {
            m_IsMouseCaptured = false;
            return true;
        }
        else if (e.type == SDL_MOUSEBUTTONDOWN && osc::IsMouseInMainViewportWorkspaceScreenRect())
        {
            m_IsMouseCaptured = true;
            return true;
        }
        return false;
    }

    void onTick()
    {
        float const rotationSpeed = glm::radians(50.0f);
        float const dt = osc::App::get().getDeltaSinceAppStartup().count();
        float const angle = rotationSpeed * dt;
        glm::vec3 const axis = glm::normalize(glm::vec3{0.5f, 1.0f, 0.0f});

        m_Step1.rotation = glm::angleAxis(angle, axis);
    }

    void onDrawMainMenu()
    {
    }

    void onDraw()
    {
        // handle mouse capturing
        if (m_IsMouseCaptured)
        {
            UpdateEulerCameraFromImGuiUserInput(m_Camera, m_CameraEulers);
            ImGui::SetMouseCursor(ImGuiMouseCursor_None);
            App::upd().setShowCursor(false);
        }
        else
        {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
            App::upd().setShowCursor(true);
        }

        // clear screen and ensure camera has correct pixel rect
        osc::App::upd().clearScreen({0.2f, 0.3f, 0.3f, 1.0f});
        m_Camera.setPixelRect(osc::GetMainViewportWorkspaceScreenRect());

        // draw 3D scene
        if (m_ShowStep1)
        {
            experimental::Graphics::DrawMesh(m_Mesh, m_Step1, m_Material, m_Camera);
        }
        else
        {
            int i = 0;
            glm::vec3 const axis = glm::normalize(glm::vec3{1.0f, 0.3f, 0.5f});
            for (glm::vec3 const& pos : g_CubePositions)
            {
                float const angle = glm::radians(i++ * 20.0f);

                Transform t;
                t.rotation = glm::angleAxis(angle, axis);
                t.position = pos;

                experimental::Graphics::DrawMesh(m_Mesh, t, m_Material, m_Camera);
            }
        }
        m_Camera.render();

        // draw UI extras
        {
            ImGui::Begin("Tutorial Step");
            ImGui::Checkbox("step1", &m_ShowStep1);
            if (m_IsMouseCaptured)
            {
                ImGui::Text("mouse captured (esc to uncapture)");
            }

            glm::vec3 cameraPos = m_Camera.getPosition();
            ImGui::Text("camera pos = (%f, %f, %f)", cameraPos.x, cameraPos.y, cameraPos.z);
            glm::vec3 cameraEulers = glm::degrees(m_CameraEulers);
            ImGui::Text("camera eulers = (%f, %f, %f)", cameraEulers.x, cameraEulers.y, cameraEulers.z);
            ImGui::End();

            m_PerfPanel.draw();
        }
    }

private:
    UID m_ID;
    TabHost* m_Parent;
    experimental::Shader m_Shader{g_VertexShader, g_FragmentShader};
    experimental::Material m_Material{m_Shader};
    experimental::Mesh m_Mesh = GenerateMesh();
    experimental::Camera m_Camera;
    glm::vec3 m_CameraEulers = {0.0f, 0.0f, 0.0f};
    bool m_ShowStep1 = false;
    bool m_IsMouseCaptured = false;
    Transform m_Step1;
    PerfPanel m_PerfPanel{"perf"};
};


// public API (PIMPL)

osc::RendererCoordinateSystemsTab::RendererCoordinateSystemsTab(TabHost* parent) :
    m_Impl{new Impl{std::move(parent)}}
{
}

osc::RendererCoordinateSystemsTab::RendererCoordinateSystemsTab(RendererCoordinateSystemsTab&& tmp) noexcept :
    m_Impl{std::exchange(tmp.m_Impl, nullptr)}
{
}

osc::RendererCoordinateSystemsTab& osc::RendererCoordinateSystemsTab::operator=(RendererCoordinateSystemsTab&& tmp) noexcept
{
    std::swap(m_Impl, tmp.m_Impl);
    return *this;
}

osc::RendererCoordinateSystemsTab::~RendererCoordinateSystemsTab() noexcept
{
    delete m_Impl;
}

osc::UID osc::RendererCoordinateSystemsTab::implGetID() const
{
    return m_Impl->getID();
}

osc::CStringView osc::RendererCoordinateSystemsTab::implGetName() const
{
    return m_Impl->getName();
}

osc::TabHost* osc::RendererCoordinateSystemsTab::implParent() const
{
    return m_Impl->getParent();
}

void osc::RendererCoordinateSystemsTab::implOnMount()
{
    m_Impl->onMount();
}

void osc::RendererCoordinateSystemsTab::implOnUnmount()
{
    m_Impl->onUnmount();
}

bool osc::RendererCoordinateSystemsTab::implOnEvent(SDL_Event const& e)
{
    return m_Impl->onEvent(e);
}

void osc::RendererCoordinateSystemsTab::implOnTick()
{
    m_Impl->onTick();
}

void osc::RendererCoordinateSystemsTab::implOnDrawMainMenu()
{
    m_Impl->onDrawMainMenu();
}

void osc::RendererCoordinateSystemsTab::implOnDraw()
{
    m_Impl->onDraw();
}
