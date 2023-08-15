#pragma once

#include <oscar/Panels/Panel.hpp>
#include <oscar/Utils/CStringView.hpp>

#include <memory>
#include <string_view>

namespace osc { class MainUIStateAPI; }
namespace osc { template<typename T> class ParentPtr; }
namespace osc { class VirtualModelStatePair; }

namespace osc
{
    class SimulationViewerPanel final : public Panel {
    public:
        SimulationViewerPanel(
            std::string_view panelName,
            std::shared_ptr<VirtualModelStatePair>,
            ParentPtr<MainUIStateAPI> const&
        );
        SimulationViewerPanel(SimulationViewerPanel const&) = delete;
        SimulationViewerPanel(SimulationViewerPanel&&) noexcept;
        SimulationViewerPanel& operator=(SimulationViewerPanel const&) = delete;
        SimulationViewerPanel& operator=(SimulationViewerPanel&&) noexcept;
        ~SimulationViewerPanel() noexcept;

    private:
        CStringView implGetName() const final;
        bool implIsOpen() const final;
        void implOpen() final;
        void implClose() final;
        void implOnDraw() final;

        class Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
