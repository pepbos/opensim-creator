#pragma once

#include <OpenSimCreator/UI/MeshWarper/MeshWarpingTabSharedState.hpp>

#include <imgui.h>
#include <oscar/Bindings/ImGuiHelpers.hpp>
#include <oscar/Graphics/Color.hpp>
#include <oscar/Maths/Vec3.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace osc
{
    // widget: bottom status bar (shows status messages, hover information, etc.)
    class MeshWarpingTabStatusBar final {
    public:
        MeshWarpingTabStatusBar(
            std::string_view label,
            std::shared_ptr<MeshWarpingTabSharedState const> tabState_) :

            m_Label{label},
            m_State{std::move(tabState_)}
        {
        }

        void onDraw()
        {
            if (osc::BeginMainViewportBottomBar(m_Label))
            {
                drawContent();
            }
            ImGui::End();
        }

    private:
        void drawContent()
        {
            if (m_State->currentHover)
            {
                drawCurrentHoverInfo(*m_State->currentHover);
            }
            else
            {
                ImGui::TextDisabled("(nothing hovered)");
            }
        }

        void drawCurrentHoverInfo(MeshWarpingTabHover const& hover)
        {
            drawColorCodedXYZ(hover.getWorldspaceLocation());
            ImGui::SameLine();
            if (hover.isHoveringASceneElement())
            {
                ImGui::TextDisabled("(Click: select %s)", FindElementNameOr(m_State->getScratch(), *hover.getSceneElementID()).c_str());
            }
            else if (hover.getInput() == TPSDocumentInputIdentifier::Source)
            {
                ImGui::TextDisabled("(Click: add a landmark, Ctrl+Click: add non-participating landmark)");
            }
            else
            {
                static_assert(NumOptions<TPSDocumentInputIdentifier>() == 2);
                ImGui::TextDisabled("(Click: add a landmark)");
            }
        }

        void drawColorCodedXYZ(Vec3 pos)
        {
            ImGui::TextUnformatted("(");
            ImGui::SameLine();
            for (int i = 0; i < 3; ++i)
            {
                Color color = {0.5f, 0.5f, 0.5f, 1.0f};
                color[i] = 1.0f;

                osc::PushStyleColor(ImGuiCol_Text, color);
                ImGui::Text("%f", pos[i]);
                ImGui::SameLine();
                osc::PopStyleColor();
            }
            ImGui::TextUnformatted(")");
        }

        std::string m_Label;
        std::shared_ptr<MeshWarpingTabSharedState const> m_State;
    };
}