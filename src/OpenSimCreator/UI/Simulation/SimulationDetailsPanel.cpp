#include "SimulationDetailsPanel.h"

#include <OpenSimCreator/Documents/Simulation/Simulation.h>
#include <OpenSimCreator/UI/Shared/BasicWidgets.h>
#include <OpenSimCreator/UI/Simulation/SimulationOutputPlot.h>

#include <IconsFontAwesome5.h>
#include <oscar/Platform/os.h>
#include <oscar/UI/ImGuiHelpers.h>
#include <oscar/UI/oscimgui.h>
#include <oscar/UI/Panels/StandardPanelImpl.h>
#include <oscar/Utils/Perf.h>

#include <filesystem>
#include <memory>
#include <string_view>
#include <utility>

using namespace osc;

class osc::SimulationDetailsPanel::Impl final : public StandardPanelImpl {
public:
    Impl(
        std::string_view panelName,
        ISimulatorUIAPI* simulatorUIAPI,
        std::shared_ptr<Simulation const> simulation) :

        StandardPanelImpl{panelName},
        m_SimulatorUIAPI{simulatorUIAPI},
        m_Simulation{std::move(simulation)}
    {
    }

private:
    void implDrawContent() final
    {
        {
            ImGui::Dummy({0.0f, 1.0f});
            ImGui::TextUnformatted("info:");
            ImGui::SameLine();
            DrawHelpMarker("Top-level info about the simulation");
            ImGui::Separator();
            ImGui::Dummy({0.0f, 2.0f});

            ImGui::Columns(2);
            ImGui::Text("num reports");
            ImGui::NextColumn();
            ImGui::Text("%zu", m_Simulation->getNumReports());
            ImGui::NextColumn();
            ImGui::Columns();
        }

        {
            OSC_PERF("draw simulation params");
            DrawSimulationParams(m_Simulation->getParams());
        }

        ImGui::Dummy({0.0f, 10.0f});

        {
            OSC_PERF("draw simulation stats");
            drawSimulationStatPlots(*m_Simulation);
        }
    }

    void drawSimulationStatPlots(Simulation const& sim)
    {
        auto outputs = sim.getOutputs();

        if (outputs.empty())
        {
            ImGui::TextDisabled("(no simulator output plots available for this simulation)");
            return;
        }

        ImGui::Dummy({0.0f, 1.0f});
        ImGui::Columns(2);
        ImGui::TextUnformatted("plots:");
        ImGui::SameLine();
        DrawHelpMarker("Various statistics collected when the simulation was ran");
        ImGui::NextColumn();
        if (std::any_of(outputs.begin(), outputs.end(), [](OutputExtractor const& o) { return o.getOutputType() == OutputType::Float; }))
        {
            ImGui::Button(ICON_FA_SAVE " Save All " ICON_FA_CARET_DOWN);
            if (ImGui::BeginPopupContextItem("##exportoptions", ImGuiPopupFlags_MouseButtonLeft))
            {
                if (ImGui::MenuItem("as CSV"))
                {
                    TryPromptAndSaveOutputsAsCSV(*m_SimulatorUIAPI, outputs);
                }

                if (ImGui::MenuItem("as CSV (and open)"))
                {
                    std::filesystem::path p = TryPromptAndSaveOutputsAsCSV(*m_SimulatorUIAPI, outputs);
                    if (!p.empty())
                    {
                        OpenPathInOSDefaultApplication(p);
                    }
                }

                ImGui::EndPopup();
            }
        }

        ImGui::NextColumn();
        ImGui::Columns();
        ImGui::Separator();
        ImGui::Dummy({0.0f, 2.0f});

        int imguiID = 0;
        ImGui::Columns(2);
        for (OutputExtractor const& output : sim.getOutputs())
        {
            ImGui::PushID(imguiID++);
            DrawOutputNameColumn(output, false);
            ImGui::NextColumn();
            SimulationOutputPlot plot{m_SimulatorUIAPI, output, 32.0f};
            plot.onDraw();
            ImGui::NextColumn();
            ImGui::PopID();
        }
        ImGui::Columns();
    }

    ISimulatorUIAPI* m_SimulatorUIAPI;
    std::shared_ptr<Simulation const> m_Simulation;
};


// public API (PIMPL)

osc::SimulationDetailsPanel::SimulationDetailsPanel(
    std::string_view panelName,
    ISimulatorUIAPI* simulatorUIAPI,
    std::shared_ptr<Simulation const> simulation) :

    m_Impl{std::make_unique<Impl>(panelName, simulatorUIAPI, std::move(simulation))}
{
}

osc::SimulationDetailsPanel::SimulationDetailsPanel(SimulationDetailsPanel&&) noexcept = default;
osc::SimulationDetailsPanel& osc::SimulationDetailsPanel::operator=(SimulationDetailsPanel&&) noexcept = default;
osc::SimulationDetailsPanel::~SimulationDetailsPanel() noexcept = default;

CStringView osc::SimulationDetailsPanel::implGetName() const
{
    return m_Impl->getName();
}

bool osc::SimulationDetailsPanel::implIsOpen() const
{
    return m_Impl->isOpen();
}

void osc::SimulationDetailsPanel::implOpen()
{
    m_Impl->open();
}

void osc::SimulationDetailsPanel::implClose()
{
    m_Impl->close();
}

void osc::SimulationDetailsPanel::implOnDraw()
{
    m_Impl->onDraw();
}
