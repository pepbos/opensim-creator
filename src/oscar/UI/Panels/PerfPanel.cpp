#include "PerfPanel.h"

#include <oscar/Platform/App.h>
#include <oscar/UI/oscimgui.h>
#include <oscar/UI/Panels/StandardPanelImpl.h>
#include <oscar/Utils/Perf.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <memory>
#include <string_view>
#include <vector>

using namespace osc;

namespace
{
    bool LexographicallyHighestLabel(PerfMeasurement const& a, PerfMeasurement const& b)
    {
        return std::string_view{a.getLabel()} > std::string_view{b.getLabel()};
    }
}

class osc::PerfPanel::Impl final : public StandardPanelImpl {
public:

    explicit Impl(std::string_view panelName) :
        StandardPanelImpl{panelName}
    {
    }

private:
    void implDrawContent() final
    {
        ui::Columns(2);
        ui::TextUnformatted("FPS");
        ui::NextColumn();
        ui::Text("%.0f", static_cast<double>(ImGui::GetIO().Framerate));
        ui::NextColumn();
        ui::Columns();

        {
            bool waiting = App::get().isMainLoopWaiting();
            if (ui::Checkbox("waiting", &waiting))
            {
                App::upd().setMainLoopWaiting(waiting);
            }
        }
        {
            bool vsync = App::get().isVsyncEnabled();
            if (ui::Checkbox("VSYNC", &vsync))
            {
                App::upd().setVsync(vsync);
            }
        }
        if (ui::Button("clear measurements"))
        {
            ClearAllPerfMeasurements();
        }
        ui::Checkbox("pause", &m_IsPaused);

        std::vector<PerfMeasurement> measurements;
        if (!m_IsPaused)
        {
            measurements = GetAllPerfMeasurements();
            std::sort(measurements.begin(), measurements.end(), LexographicallyHighestLabel);
        }

        ImGuiTableFlags flags =
            ImGuiTableFlags_NoSavedSettings |
            ImGuiTableFlags_Resizable |
            ImGuiTableFlags_BordersInner;
        if (ImGui::BeginTable("measurements", 6, flags))
        {
            ImGui::TableSetupColumn("Label");
            ImGui::TableSetupColumn("Source File");
            ImGui::TableSetupColumn("Num Calls");
            ImGui::TableSetupColumn("Last Duration");
            ImGui::TableSetupColumn("Average Duration");
            ImGui::TableSetupColumn("Total Duration");
            ImGui::TableHeadersRow();

            for (PerfMeasurement const& pm : measurements)
            {
                if (pm.getCallCount() <= 0)
                {
                    continue;
                }

                int column = 0;
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(column++);
                ui::TextUnformatted(pm.getLabel());
                ImGui::TableSetColumnIndex(column++);
                ui::Text("%s:%u", pm.getFilename().c_str(), pm.getLine());
                ImGui::TableSetColumnIndex(column++);
                ui::Text("%" PRId64, pm.getCallCount());
                ImGui::TableSetColumnIndex(column++);
                ui::Text("%ld us", static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(pm.getLastDuration()).count()));
                ImGui::TableSetColumnIndex(column++);
                ui::Text("%ld us", static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(pm.getAvgDuration()).count()));
                ImGui::TableSetColumnIndex(column++);
                ui::Text("%ld us", static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(pm.getTotalDuration()).count()));
            }

            ImGui::EndTable();
        }
    }

    bool m_IsPaused = false;
};


// public API

osc::PerfPanel::PerfPanel(std::string_view panelName) :
    m_Impl{std::make_unique<Impl>(panelName)}
{
}

osc::PerfPanel::PerfPanel(PerfPanel&&) noexcept = default;
osc::PerfPanel& osc::PerfPanel::operator=(PerfPanel&&) noexcept = default;
osc::PerfPanel::~PerfPanel() noexcept = default;

CStringView osc::PerfPanel::implGetName() const
{
    return m_Impl->getName();
}

bool osc::PerfPanel::implIsOpen() const
{
    return m_Impl->isOpen();
}

void osc::PerfPanel::implOpen()
{
    return m_Impl->open();
}

void osc::PerfPanel::implClose()
{
    m_Impl->close();
}

void osc::PerfPanel::implOnDraw()
{
    m_Impl->onDraw();
}
