#include "IconWithMenu.h"

#include <oscar/UI/Icon.h>
#include <oscar/UI/oscimgui.h>
#include <oscar/UI/Widgets/IconWithoutMenu.h>
#include <oscar/Utils/CStringView.h>

#include <functional>
#include <string>
#include <utility>

osc::IconWithMenu::IconWithMenu(
    Icon icon,
    CStringView title,
    CStringView description,
    std::function<bool()> contentRenderer) :

    m_IconWithoutMenu{std::move(icon), title, description},
    m_ContextMenuID{"##" + m_IconWithoutMenu.getIconID()},
    m_ContentRenderer{std::move(contentRenderer)}
{
}

bool osc::IconWithMenu::onDraw()
{
    if (m_IconWithoutMenu.onDraw())
    {
        ImGui::OpenPopup(m_ContextMenuID.c_str());
    }

    bool rv = false;
    if (ImGui::BeginPopup(m_ContextMenuID.c_str(),ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings))
    {
        ImGui::TextDisabled("%s", m_IconWithoutMenu.getTitle().c_str());
        ImGui::Dummy({0.0f, 0.5f*ImGui::GetTextLineHeight()});
        rv = m_ContentRenderer();
        ImGui::EndPopup();
    }

    return rv;
}
