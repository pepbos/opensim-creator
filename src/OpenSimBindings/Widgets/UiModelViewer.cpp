#include "UiModelViewer.hpp"

#include "src/Bindings/ImGuiHelpers.hpp"
#include "src/Formats/DAE.hpp"
#include "src/Graphics/GraphicsHelpers.hpp"
#include "src/Graphics/Icon.hpp"
#include "src/Graphics/IconCache.hpp"
#include "src/Graphics/MeshCache.hpp"
#include "src/Graphics/SceneDecoration.hpp"
#include "src/Graphics/SceneRenderer.hpp"
#include "src/Graphics/SceneRendererParams.hpp"
#include "src/Graphics/ShaderCache.hpp"
#include "src/Maths/AABB.hpp"
#include "src/Maths/BVH.hpp"
#include "src/Maths/Constants.hpp"
#include "src/Maths/MathHelpers.hpp"
#include "src/Maths/Line.hpp"
#include "src/Maths/RayCollision.hpp"
#include "src/Maths/Rect.hpp"
#include "src/Maths/PolarPerspectiveCamera.hpp"
#include "src/OpenSimBindings/Rendering/CachedModelRenderer.hpp"
#include "src/OpenSimBindings/Rendering/CustomDecorationOptions.hpp"
#include "src/OpenSimBindings/Rendering/CustomRenderingOptions.hpp"
#include "src/OpenSimBindings/Rendering/ModelRendererParams.hpp"
#include "src/OpenSimBindings/Rendering/OpenSimDecorationGenerator.hpp"
#include "src/OpenSimBindings/Rendering/MuscleColoringStyle.hpp"
#include "src/OpenSimBindings/Rendering/MuscleDecorationStyle.hpp"
#include "src/OpenSimBindings/Rendering/MuscleSizingStyle.hpp"
#include "src/OpenSimBindings/Widgets/BasicWidgets.hpp"
#include "src/OpenSimBindings/OpenSimHelpers.hpp"
#include "src/OpenSimBindings/VirtualConstModelStatePair.hpp"
#include "src/Platform/App.hpp"
#include "src/Platform/os.hpp"
#include "src/Utils/Algorithms.hpp"
#include "src/Utils/Perf.hpp"
#include "src/Utils/UID.hpp"
#include "src/Widgets/GuiRuler.hpp"

#include <glm/mat3x3.hpp>
#include <glm/mat4x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <imgui.h>
#include <nonstd/span.hpp>
#include <OpenSim/Common/Component.h>
#include <OpenSim/Common/ComponentPath.h>
#include <OpenSim/Simulation/Model/Model.h>
#include <IconsFontAwesome5.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace osc
{
    void DrawAllImGuiOverlays(
        osc::ModelRendererParams& params,
        nonstd::span<osc::SceneDecoration const> drawlist,
        std::optional<osc::AABB> maybeSceneAABB,
        osc::Rect const& renderRect,
        osc::IconCache& iconCache,
        osc::GuiRuler& ruler)
    {
        // draw the top overlays
        ImGui::SetCursorScreenPos(renderRect.p1 + glm::vec2{ImGui::GetStyle().WindowPadding});
        DrawViewerTopButtonRow(params, drawlist, iconCache, ruler);

        // compute bottom overlay positions
        ImGuiStyle const& style = ImGui::GetStyle();
        glm::vec2 const alignmentAxesDims = osc::CalcAlignmentAxesDimensions();
        glm::vec2 const axesTopLeft =
        {
            renderRect.p1.x + style.WindowPadding.x,
            renderRect.p2.y - style.WindowPadding.y - alignmentAxesDims.y
        };

        // draw the bottom overlays
        ImGui::SetCursorScreenPos(axesTopLeft);
        DrawAlignmentAxes(
            params.camera.getViewMtx()
        );
        DrawCameraControlButtons(
            params.camera,
            renderRect,
            maybeSceneAABB,
            iconCache
        );
    }

    osc::Line UnprojectMouseCursor(
        osc::PolarPerspectiveCamera const& camera,
        glm::vec2 mousePos,
        osc::Rect const& renderRect)
    {
        glm::vec2 const posInRender = mousePos - renderRect.p1;
        return camera.unprojectTopLeftPosToWorldRay(posInRender, osc::Dimensions(renderRect));
    }

    bool IsSceneDecorationIDed(osc::SceneDecoration const& dec)
    {
        return !dec.id.empty();
    }

    bool ShouldAllowInteraction(osc::ImGuiItemHittestResult const& htResult)
    {
        if (!htResult.isHovered ||
            ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
            ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
            ImGui::IsMouseDragging(ImGuiMouseButton_Right))
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    void HandleUserInputs(
        osc::PolarPerspectiveCamera& camera,
        osc::Rect const& viewportRect,
        std::optional<osc::AABB> maybeSceneAABB)
    {
        bool ctrlDown = osc::IsCtrlOrSuperDown();

        if (ImGui::IsKeyReleased(ImGuiKey_X))
        {
            if (ctrlDown)
            {
                FocusAlongMinusX(camera);
            } else
            {
                FocusAlongX(camera);
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Y))
        {
            if (!ctrlDown)
            {
                FocusAlongY(camera);
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F))
        {
            if (ctrlDown)
            {
                if (maybeSceneAABB)
                {
                    osc::AutoFocus(
                        camera,
                        *maybeSceneAABB,
                        osc::AspectRatio(viewportRect)
                    );
                }
            }
            else
            {
                Reset(camera);
            }
        }
        if (ctrlDown && (ImGui::IsKeyPressed(ImGuiKey_8)))
        {
            if (maybeSceneAABB)
            {
                osc::AutoFocus(
                    camera,
                    *maybeSceneAABB,
                    osc::AspectRatio(viewportRect)
                );
            }
        }
    }

    std::optional<SceneCollision> HittestModelRenderer(
        osc::PolarPerspectiveCamera const& camera,
        glm::vec2 mouseScreenPos,
        osc::Rect const& viewportScreenRect,
        osc::CachedModelRenderer const& modelRenderer)
    {
        OSC_PERF("scene hittest");

        // un-project 2D mouse cursor into 3D scene as a ray
        Line const cameraRay = UnprojectMouseCursor(
            camera,
            mouseScreenPos,
            viewportScreenRect
        );

        // perform hittest only on IDed scene elements
        return modelRenderer.getClosestCollision(
            cameraRay,
            IsSceneDecorationIDed
        );
    }
}

class osc::UiModelViewer::Impl final {
public:

    bool isLeftClicked() const
    {
        return m_RenderedImageHittest.isLeftClickReleasedWithoutDragging;
    }

    bool isRightClicked() const
    {
        return m_RenderedImageHittest.isRightClickReleasedWithoutDragging;
    }

    bool isMousedOver() const
    {
        return m_RenderedImageHittest.isHovered;
    }

    std::optional<SceneCollision> draw(VirtualConstModelStatePair const& rs)
    {
        OSC_PERF("UiModelViewer/draw");

        // inputs: the camera may always be moved around while hovering
        if (m_RenderedImageHittest.isHovered)
        {
            UpdatePolarCameraFromImGuiUserInput(
                Dimensions(m_RenderedImageHittest.rect),
                m_Params.camera
            );
        }

        // inputs: other user inputs have extra restrictions
        bool const allowInteraction = ShouldAllowInteraction(m_RenderedImageHittest);
        if (allowInteraction)
        {
            HandleUserInputs(
                m_Params.camera,
                m_RenderedImageHittest.rect,
                m_CachedModelRenderer.getRootAABB()
            );
        }

        m_CachedModelRenderer.populate(rs, m_Params);

        // if this is the first frame being rendered, auto-focus the scene
        if (m_IsRenderingFirstFrame && m_CachedModelRenderer.getRootAABB())
        {
            AutoFocus(
                m_Params.camera,
                *m_CachedModelRenderer.getRootAABB(),
                AspectRatio(ImGui::GetContentRegionAvail())
            );
            m_IsRenderingFirstFrame = false;
        }

        // render into texture
        {
            OSC_PERF("UiModelViewer/draw/render");

            m_CachedModelRenderer.draw(
                rs,
                m_Params,
                ImGui::GetContentRegionAvail(),
                App::get().getMSXAASamplesRecommended()
            );
        }

        // blit texture as an ImGui::Image
        DrawTextureAsImGuiImage(
            m_CachedModelRenderer.updRenderTexture(),
            ImGui::GetContentRegionAvail()
        );
        m_RenderedImageHittest = osc::HittestLastImguiItem();

        std::optional<SceneCollision> hittestResult;
        if (allowInteraction)
        {
            hittestResult = HittestModelRenderer(
                m_Params.camera,
                ImGui::GetMousePos(),
                m_RenderedImageHittest.rect,
                m_CachedModelRenderer
            );
        }

        // draw any ImGui-based overlays over the image
        {
            OSC_PERF("UiModelViewer/draw/overlays");

            DrawAllImGuiOverlays(
                m_Params,
                m_CachedModelRenderer.getDrawlist(),
                m_CachedModelRenderer.getRootAABB(),
                m_RenderedImageHittest.rect,
                *m_IconCache,
                m_Ruler
            );

            if (m_Ruler.isMeasuring())
            {
                std::optional<GuiRulerMouseHit> maybeHit;
                if (hittestResult)
                {
                    maybeHit.emplace(hittestResult->decorationID, hittestResult->worldspaceLocation);
                }
                m_Ruler.draw(m_Params.camera, m_RenderedImageHittest.rect, maybeHit);
            }
        }

        // handle return value
        if (m_Ruler.isMeasuring())
        {
            // disable hittest while measuring
            return std::nullopt;
        }
        else
        {
            return hittestResult;
        }
    }

private:

    // rendering-related data
    ModelRendererParams m_Params;
    CachedModelRenderer m_CachedModelRenderer
    {
        App::get().getConfig(),
        App::singleton<MeshCache>(),
        *App::singleton<ShaderCache>(),
    };
    osc::ImGuiItemHittestResult m_RenderedImageHittest;

    // overlay-related data
    std::shared_ptr<IconCache> m_IconCache = osc::App::singleton<osc::IconCache>(
        App::resource("icons/"),
        ImGui::GetTextLineHeight()/128.0f
    );
    GuiRuler m_Ruler;

    // a flag that will auto-focus the main scene camera the next time it's used
    //
    // initialized `true`, so that the initially-loaded model is autofocused (#520)
    bool m_IsRenderingFirstFrame = true;
};


// public API (PIMPL)

osc::UiModelViewer::UiModelViewer() :
    m_Impl{std::make_unique<Impl>()}
{
}
osc::UiModelViewer::UiModelViewer(UiModelViewer&&) noexcept = default;
osc::UiModelViewer& osc::UiModelViewer::operator=(UiModelViewer&&) noexcept = default;
osc::UiModelViewer::~UiModelViewer() noexcept = default;

bool osc::UiModelViewer::isLeftClicked() const
{
    return m_Impl->isLeftClicked();
}

bool osc::UiModelViewer::isRightClicked() const
{
    return m_Impl->isRightClicked();
}

bool osc::UiModelViewer::isMousedOver() const
{
    return m_Impl->isMousedOver();
}

std::optional<osc::SceneCollision> osc::UiModelViewer::draw(VirtualConstModelStatePair const& rs)
{
    return m_Impl->draw(rs);
}
