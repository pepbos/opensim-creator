#pragma once

#include <oscar/UI/Widgets/Popup.hpp>

#include <memory>
#include <string_view>

namespace OpenSim { class Component; }
namespace osc { class PopupAPI; }
namespace osc { class UndoableModelStatePair; }

namespace osc
{
    class AddComponentPopup final : public Popup {
    public:
        AddComponentPopup(
            std::string_view popupName,
            PopupAPI*,
            std::shared_ptr<UndoableModelStatePair>,
            std::unique_ptr<OpenSim::Component> prototype
        );
        AddComponentPopup(AddComponentPopup const&) = delete;
        AddComponentPopup(AddComponentPopup&&) noexcept;
        AddComponentPopup& operator=(AddComponentPopup const&) = delete;
        AddComponentPopup& operator=(AddComponentPopup&&) noexcept;
        ~AddComponentPopup() noexcept;

    private:
        bool implIsOpen() const final;
        void implOpen() final;
        void implClose() final;
        bool implBeginPopup() final;
        void implOnDraw() final;
        void implEndPopup() final;

        class Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}