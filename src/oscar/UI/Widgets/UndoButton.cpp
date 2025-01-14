#include "UndoButton.h"

#include <oscar/UI/oscimgui.h>
#include <oscar/Utils/UndoRedo.h>

#include <IconsFontAwesome5.h>

#include <memory>

osc::UndoButton::UndoButton(std::shared_ptr<UndoRedoBase> undoRedo_) :
    m_UndoRedo{std::move(undoRedo_)}
{
}

osc::UndoButton::~UndoButton() noexcept = default;

void osc::UndoButton::onDraw()
{
    int imguiID = 0;

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.0f, 0.0f});

    bool wasDisabled = false;
    if (!m_UndoRedo->canUndo())
    {
        ImGui::BeginDisabled();
        wasDisabled = true;
    }
    if (ImGui::Button(ICON_FA_UNDO))
    {
        m_UndoRedo->undo();
    }

    ImGui::SameLine();

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0.0f, ImGui::GetStyle().FramePadding.y});
    ImGui::Button(ICON_FA_CARET_DOWN);
    ImGui::PopStyleVar();

    if (wasDisabled)
    {
        ImGui::EndDisabled();
    }

    if (ImGui::BeginPopupContextItem("##OpenUndoMenu", ImGuiPopupFlags_MouseButtonLeft))
    {
        for (ptrdiff_t i = 0; i < m_UndoRedo->getNumUndoEntriesi(); ++i)
        {
            ImGui::PushID(imguiID++);
            if (ImGui::Selectable(m_UndoRedo->getUndoEntry(i).message().c_str()))
            {
                m_UndoRedo->undoTo(i);
            }
            ImGui::PopID();
        }
        ImGui::EndPopup();
    }

    ImGui::PopStyleVar();
}
