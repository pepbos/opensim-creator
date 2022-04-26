#pragma once

#include "src/Platform/Screen.hpp"

#include <SDL_events.h>

#include <memory>

namespace osc
{
    // basic test for graphics backend: can it display a triangle
    class HelloTriangleScreen final : public Screen {
    public:
        HelloTriangleScreen();
        HelloTriangleScreen(HelloTriangleScreen const&) = delete;
        HelloTriangleScreen(HelloTriangleScreen&&) noexcept;
        HelloTriangleScreen& operator=(HelloTriangleScreen const&) = delete;
        HelloTriangleScreen& operator=(HelloTriangleScreen&&) noexcept;
        ~HelloTriangleScreen() noexcept override;

        void onEvent(SDL_Event const&) override;
        void tick(float) override;
        void draw() override;

        class Impl;
    private:
        Impl* m_Impl;
    };
}
