#include "SelectionDetailsPanel.h"

#include <OpenSimCreator/Documents/Simulation/SimulationModelStatePair.h>
#include <OpenSimCreator/OutputExtractors/ComponentOutputExtractor.h>
#include <OpenSimCreator/UI/Simulation/ComponentDetails.h>
#include <OpenSimCreator/UI/Simulation/ISimulatorUIAPI.h>
#include <OpenSimCreator/UI/Simulation/SimulationOutputPlot.h>

#include <OpenSim/Common/Component.h>
#include <oscar/UI/oscimgui.h>
#include <oscar/UI/Panels/StandardPanelImpl.h>

#include <memory>
#include <string_view>
#include <utility>

using namespace osc;

class osc::SelectionDetailsPanel::Impl final : public StandardPanelImpl {
public:
    Impl(
        std::string_view panelName,
        ISimulatorUIAPI* simulatorUIAPI) :

        StandardPanelImpl{panelName},
        m_SimulatorUIAPI{simulatorUIAPI}
    {
    }

private:
    void implDrawContent() final
    {
        SimulationModelStatePair* maybeShownState = m_SimulatorUIAPI->tryGetCurrentSimulationState();

        if (!maybeShownState)
        {
            ImGui::TextDisabled("(no simulation selected)");
            return;
        }

        SimulationModelStatePair& ms = *maybeShownState;

        OpenSim::Component const* selected = ms.getSelected();

        if (!selected)
        {
            ImGui::TextDisabled("(nothing selected)");
            return;
        }

        m_ComponentDetailsWidget.onDraw(ms.getState(), selected);

        if (ImGui::CollapsingHeader("outputs"))
        {
            int imguiID = 0;
            ImGui::Columns(2);
            for (auto const& [outputName, aoPtr] : selected->getOutputs())
            {
                ImGui::PushID(imguiID++);

                ImGui::Text("%s", outputName.c_str());
                ImGui::NextColumn();
                SimulationOutputPlot plot
                {
                    m_SimulatorUIAPI,
                    OutputExtractor{ComponentOutputExtractor{*aoPtr}},
                    ImGui::GetTextLineHeight(),
                };
                plot.onDraw();
                ImGui::NextColumn();

                ImGui::PopID();
            }
            ImGui::Columns();
        }
    }

    ISimulatorUIAPI* m_SimulatorUIAPI;
    ComponentDetails m_ComponentDetailsWidget;
};

osc::SelectionDetailsPanel::SelectionDetailsPanel(std::string_view panelName, ISimulatorUIAPI* simulatorUIAPI) :
    m_Impl{std::make_unique<Impl>(panelName, simulatorUIAPI)}
{
}

osc::SelectionDetailsPanel::SelectionDetailsPanel(SelectionDetailsPanel&&) noexcept = default;
osc::SelectionDetailsPanel& osc::SelectionDetailsPanel::operator=(SelectionDetailsPanel&&) noexcept = default;
osc::SelectionDetailsPanel::~SelectionDetailsPanel() noexcept = default;

CStringView osc::SelectionDetailsPanel::implGetName() const
{
    return m_Impl->getName();
}

bool osc::SelectionDetailsPanel::implIsOpen() const
{
    return m_Impl->isOpen();
}

void osc::SelectionDetailsPanel::implOpen()
{
    m_Impl->open();
}

void osc::SelectionDetailsPanel::implClose()
{
    m_Impl->close();
}

void osc::SelectionDetailsPanel::implOnDraw()
{
    m_Impl->onDraw();
}
