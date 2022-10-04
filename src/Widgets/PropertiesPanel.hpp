#pragma once

#include <memory>

namespace osc { class EditorAPI; }
namespace osc { class UndoableModelStatePair; }

namespace osc
{
    class PropertiesPanel final {
    public:
        explicit PropertiesPanel(EditorAPI*, std::shared_ptr<UndoableModelStatePair>);
        PropertiesPanel(PropertiesPanel const&) = delete;
        PropertiesPanel(PropertiesPanel&&) noexcept;
        PropertiesPanel& operator=(PropertiesPanel const&) = delete;
        PropertiesPanel& operator=(PropertiesPanel&&) noexcept;
        ~PropertiesPanel() noexcept;

        void draw();

    private:
        class Impl;
        Impl* m_Impl;
    };
}