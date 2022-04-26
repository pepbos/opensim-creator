#pragma once

#include "src/Platform/Screen.hpp"

#include <SDL_events.h>

#include <memory>

namespace osc
{
    class MainEditorState;
}

namespace osc
{
    // shows forward-dynamic simulations
    class SimulatorScreen final : public Screen {
    public:
        SimulatorScreen(std::shared_ptr<MainEditorState>);
        SimulatorScreen(SimulatorScreen const&) = delete;
        SimulatorScreen(SimulatorScreen&&) noexcept;
        SimulatorScreen& operator=(SimulatorScreen const&) = delete;
        SimulatorScreen& operator=(SimulatorScreen&&) noexcept;
        ~SimulatorScreen() noexcept override;

        void onMount() override;
        void onUnmount() override;
        void onEvent(SDL_Event const&) override;
        void tick(float) override;
        void draw() override;

        class Impl;
    private:
        Impl* m_Impl;
    };
}
