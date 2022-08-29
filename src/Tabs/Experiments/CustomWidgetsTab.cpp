#include "CustomWidgetsTab.hpp"

#include <IconsFontAwesome5.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <SDL_events.h>

#include <string>
#include <utility>

namespace
{
    float GetWidgetTitleDescriptionHeight(const char *title,
        const char *description)
    {
        //ImGui::PushFont(g_font_mgr.m_menu_font_medium);
        float h = ImGui::GetFrameHeight();
        //ImGui::PopFont();

        if (description) {
            ImGuiStyle &style = ImGui::GetStyle();
            h += style.ItemInnerSpacing.y;
            //ImGui::PushFont(g_font_mgr.m_default_font);
            h += ImGui::GetTextLineHeight();
            //ImGui::PopFont();
        }

        return h;
    }

    void WidgetTitleDescription(const char *title, const char *description, ImVec2 pos)
    {
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGuiStyle &style = ImGui::GetStyle();

        ImVec2 text_pos = pos;
        text_pos.x += style.FramePadding.x;
        text_pos.y += style.FramePadding.y;

        //ImGui::PushFont(g_font_mgr.m_menu_font_medium);
        float title_height = ImGui::GetTextLineHeight();
        draw_list->AddText(text_pos, ImGui::GetColorU32(ImGuiCol_Text), title);
        //ImGui::PopFont();

        if (description) {
            text_pos.y += title_height + style.ItemInnerSpacing.y;

            //ImGui::PushFont(g_font_mgr.m_default_font);
            draw_list->AddText(text_pos, ImGui::GetColorU32(ImVec4(0.94f, 0.94f, 0.94f, 0.70f)), description);
            //ImGui::PopFont();
        }
    }

    float GetSliderRadius(ImVec2 size)
    {
        return size.y * 0.5f;
    }

    void DrawSlider(float v, bool hovered, ImVec2 pos, ImVec2 size)
    {
        ImDrawList *draw_list = ImGui::GetWindowDrawList();

        float radius = GetSliderRadius(size);
        float rounding = size.y * 0.25f;
        float slot_half_height = size.y * 0.125f;
        const bool circular_grab = false;

        ImU32 bg = hovered ? ImGui::GetColorU32(ImGuiCol_FrameBgActive)
            : ImGui::GetColorU32(ImGuiCol_CheckMark);

        glm::vec2 pmid(pos.x + radius + v*(size.x - radius*2), pos.y + size.y / 2);
        ImVec2 smin(pos.x + rounding, pmid.y - slot_half_height);
        ImVec2 smax(pmid.x, pmid.y + slot_half_height);
        draw_list->AddRectFilled(smin, smax, bg, rounding);

        bg = hovered ? ImGui::GetColorU32(ImGuiCol_FrameBgHovered)
            : ImGui::GetColorU32(ImGuiCol_FrameBg);

        smin.x = pmid.x;
        smax.x = pos.x + size.x - rounding;
        draw_list->AddRectFilled(smin, smax, bg, rounding);

        if (circular_grab) {
            draw_list->AddCircleFilled(pmid, radius * 0.8f, ImGui::GetColorU32(ImGuiCol_SliderGrab));
        } else {
            glm::vec2 offs(radius*0.8, radius*0.8);
            draw_list->AddRectFilled(pmid - offs, pmid + offs, ImGui::GetColorU32(ImGuiCol_SliderGrab), rounding);
        }
    }

    float GetSliderTrackXOffset(ImVec2 size)
    {
        return GetSliderRadius(size);
    }

    float GetSliderTrackWidth(ImVec2 size)
    {
        return size.x - GetSliderRadius(size) * 2;
    }

    float GetSliderValueForMousePos(ImVec2 mouse, ImVec2 pos, ImVec2 size)
    {
        return (mouse.x - pos.x - GetSliderTrackXOffset(size)) /
            GetSliderTrackWidth(size);
    }

    void Slider(const char *str_id, float *v, const char *description)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32_BLACK_TRANS);

        ImGuiStyle &style = ImGui::GetStyle();
        ImGuiWindow *window = ImGui::GetCurrentWindow();

        //ImGui::PushFont(g_font_mgr.m_menu_font_medium);
        float title_height = ImGui::GetTextLineHeight();
        //ImGui::PopFont();

        ImVec2 p = ImGui::GetCursorScreenPos();
        ImVec2 size(ImGui::GetColumnWidth(), GetWidgetTitleDescriptionHeight(str_id, description));
        WidgetTitleDescription(str_id, description, p);

        // XXX: Internal API
        ImVec2 wpos = ImGui::GetCursorPos();
        ImRect bb(p, ImVec2(p.x + size.x, p.y + size.y));
        ImGui::ItemSize(size, 0.0f);
        ImGui::ItemAdd(bb, 0);
        ImGui::SetItemAllowOverlap();
        ImGui::SameLine(0, 0);

        ImVec2 slider_size(size.x * 0.4f, title_height * 0.9f);
        ImVec2 slider_pos(bb.Max.x - slider_size.x - style.FramePadding.x,
            p.y + (title_height - slider_size.y)/2 + style.FramePadding.y);

        ImGui::SetCursorPos(ImVec2(wpos.x + size.x - slider_size.x - style.FramePadding.x,
            wpos.y));

        ImGui::InvisibleButton("###slider", slider_size, 0);


        if (ImGui::IsItemHovered())
        {
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
            {
                *v -= 0.05f;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
            {
                *v += 0.05f;
            }

            if (ImGui::IsKeyDown(ImGuiKey_LeftArrow) ||
                ImGui::IsKeyDown(ImGuiKey_RightArrow))
            {
                ImGui::NavMoveRequestCancel();
            }
        }

        if (ImGui::IsItemActive())
        {
            ImVec2 mouse = ImGui::GetMousePos();
            *v = GetSliderValueForMousePos(mouse, slider_pos, slider_size);
        }

        *v = fmax(0.0f, fmin(*v, 1.0f));

        DrawSlider(*v, ImGui::IsItemHovered() || ImGui::IsItemActive(), slider_pos, slider_size);

        ImVec2 slider_max = ImVec2(slider_pos.x + slider_size.x, slider_pos.y + slider_size.y);
        ImGui::RenderNavHighlight(ImRect(slider_pos, slider_max), window->GetID("###slider"));

        ImGui::PopStyleColor();
    }


    void DrawToggle(bool enabled, bool hovered, ImVec2 pos, ImVec2 size)
    {
        ImDrawList *draw_list = ImGui::GetWindowDrawList();

        float radius = size.y * 0.5f;
        float rounding = size.y * 0.25f;
        float slot_half_height = size.y * 0.5f;
        const bool circular_grab = false;

        ImU32 bg = hovered ? ImGui::GetColorU32(enabled ? ImGuiCol_FrameBgActive : ImGuiCol_FrameBgHovered)
            : ImGui::GetColorU32(enabled ? ImGuiCol_CheckMark : ImGuiCol_FrameBg);

        glm::vec2 pmid(pos.x + radius + (int)enabled * (size.x - radius * 2), pos.y + size.y / 2.0f);
        ImVec2 smin(pos.x, pmid.y - slot_half_height);
        ImVec2 smax(pos.x + size.x, pmid.y + slot_half_height);
        draw_list->AddRectFilled(smin, smax, bg, rounding);

        if (circular_grab) {
            draw_list->AddCircleFilled(pmid, radius * 0.8f, ImGui::GetColorU32(ImGuiCol_SliderGrab));
        } else {
            glm::vec2 offs(radius*0.8f, radius*0.8f);
            draw_list->AddRectFilled(pmid - offs, pmid + offs, ImGui::GetColorU32(ImGuiCol_SliderGrab), rounding);
        }
    }

    bool Toggle(const char *str_id, bool *v, const char *description)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32_BLACK_TRANS);

        ImGuiStyle &style = ImGui::GetStyle();

        //ImGui::PushFont(g_font_mgr.m_menu_font_medium);
        float title_height = ImGui::GetTextLineHeight();
        //ImGui::PopFont();

        ImVec2 p = ImGui::GetCursorScreenPos();
        ImVec2 bb(ImGui::GetColumnWidth(),
            GetWidgetTitleDescriptionHeight(str_id, description));
        ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0, 0));
        ImGui::PushID(str_id);
        bool status = ImGui::Button("###toggle_button", bb);
        if (status) {
            *v = !*v;
        }
        ImGui::PopID();
        ImGui::PopStyleVar();
        const ImVec2 p_min = ImGui::GetItemRectMin();
        const ImVec2 p_max = ImGui::GetItemRectMax();

        WidgetTitleDescription(str_id, description, p);

        float toggle_height = title_height * 0.9f;
        ImVec2 toggle_size(toggle_height * 1.75f, toggle_height);
        ImVec2 toggle_pos(p_max.x - toggle_size.x - style.FramePadding.x,
            p_min.y + (title_height - toggle_size.y)/2.0f + style.FramePadding.y);
        DrawToggle(*v, ImGui::IsItemHovered(), toggle_pos, toggle_size);

        ImGui::PopStyleColor();

        return status;
    }
}

class osc::CustomWidgetsTab::Impl final {
public:

    Impl(TabHost* parent) : m_Parent{std::move(parent)}
    {
    }

    UID getID() const
    {
        return m_ID;
    }

    CStringView getName() const
    {
        return m_Name;
    }

    TabHost* parent()
    {
        return m_Parent;
    }

    void onMount()
    {

    }

    void onUnmount()
    {

    }

    bool onEvent(SDL_Event const&)
    {
        return false;
    }

    void onTick()
    {

    }

    void onDrawMainMenu()
    {

    }

    void onDraw()
    {
        ImGui::Begin("window");
        Slider("lol", &m_Value, "some desc");
        Toggle("toggle", &m_Toggle, "some toggle");
        ImGui::End();
    }


private:
    UID m_ID;
    std::string m_Name = ICON_FA_COOKIE " CustomWidgetsTab";
    TabHost* m_Parent;
    float m_Value = 0.0f;
    bool m_Toggle = false;
};


// public API

osc::CustomWidgetsTab::CustomWidgetsTab(TabHost* parent) :
    m_Impl{new Impl{std::move(parent)}}
{
}

osc::CustomWidgetsTab::CustomWidgetsTab(CustomWidgetsTab&& tmp) noexcept :
    m_Impl{std::exchange(tmp.m_Impl, nullptr)}
{
}

osc::CustomWidgetsTab& osc::CustomWidgetsTab::operator=(CustomWidgetsTab&& tmp) noexcept
{
    std::swap(m_Impl, tmp.m_Impl);
    return *this;
}

osc::CustomWidgetsTab::~CustomWidgetsTab() noexcept
{
    delete m_Impl;
}

osc::UID osc::CustomWidgetsTab::implGetID() const
{
    return m_Impl->getID();
}

osc::CStringView osc::CustomWidgetsTab::implGetName() const
{
    return m_Impl->getName();
}

osc::TabHost* osc::CustomWidgetsTab::implParent() const
{
    return m_Impl->parent();
}

void osc::CustomWidgetsTab::implOnMount()
{
    m_Impl->onMount();
}

void osc::CustomWidgetsTab::implOnUnmount()
{
    m_Impl->onUnmount();
}

bool osc::CustomWidgetsTab::implOnEvent(SDL_Event const& e)
{
    return m_Impl->onEvent(e);
}

void osc::CustomWidgetsTab::implOnTick()
{
    m_Impl->onTick();
}

void osc::CustomWidgetsTab::implOnDrawMainMenu()
{
    m_Impl->onDrawMainMenu();
}

void osc::CustomWidgetsTab::implOnDraw()
{
    m_Impl->onDraw();
}