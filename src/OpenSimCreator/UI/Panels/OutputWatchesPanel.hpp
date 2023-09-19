#pragma once

#include <oscar/UI/Panels/Panel.hpp>
#include <oscar/Utils/CStringView.hpp>

#include <memory>
#include <string_view>

namespace osc { class MainUIStateAPI; }
namespace osc { template<typename T> class ParentPtr; }
namespace osc { class UndoableModelStatePair; }

namespace osc
{
    class OutputWatchesPanel final : public Panel {
    public:
        OutputWatchesPanel(
            std::string_view panelName,
            std::shared_ptr<UndoableModelStatePair const>,
            ParentPtr<MainUIStateAPI> const&
        );
        OutputWatchesPanel(OutputWatchesPanel const&) = delete;
        OutputWatchesPanel(OutputWatchesPanel&&) noexcept;
        OutputWatchesPanel& operator=(OutputWatchesPanel const&) = delete;
        OutputWatchesPanel& operator=(OutputWatchesPanel&&) noexcept;
        ~OutputWatchesPanel() noexcept;

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