#pragma once

#include "src/Tabs/Tab.hpp"
#include "src/Utils/CStringView.hpp"
#include "src/Utils/UID.hpp"

#include <SDL_events.h>

#include <memory>

namespace osc
{
	class MainEditorState;
}

namespace osc
{
	class TabHost;

	class SimulatorTab final : public Tab {
	public:
		SimulatorTab(TabHost*, std::shared_ptr<MainEditorState>);
		SimulatorTab(SimulatorTab const&) = delete;
		SimulatorTab(SimulatorTab&&) noexcept;
		SimulatorTab& operator=(SimulatorTab const&) = delete;
		SimulatorTab& operator=(SimulatorTab&&) noexcept;
		~SimulatorTab() noexcept override;

	private:
		UID implGetID() const override;
		CStringView implGetName() const override;
		TabHost* implParent() const override;
		void implOnMount() override;
		void implOnUnmount() override;
		bool implOnEvent(SDL_Event const&) override;
		void implOnTick() override;
		void implOnDrawMainMenu() override;
		void implOnDraw() override;

		class Impl;
		Impl* m_Impl;
	};
}