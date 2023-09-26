#include "OscarDemosTabRegistry.hpp"

#include <oscar_demos/CustomWidgetsTab.hpp>
#include <oscar_demos/HittestTab.hpp>
#include <oscar_demos/ImGuiDemoTab.hpp>
#include <oscar_demos/ImGuizmoDemoTab.hpp>
#include <oscar_demos/ImPlotDemoTab.hpp>
#include <oscar_demos/MandelbrotTab.hpp>
#include <oscar_demos/MeshGenTestTab.hpp>

#include <oscar/UI/Tabs/TabRegistry.hpp>
#include <oscar/UI/Tabs/TabRegistryEntry.hpp>
#include <oscar/Utils/ParentPtr.hpp>

#include <memory>

namespace
{
    template<typename TabType>
    void RegisterTab(osc::TabRegistry& registry)
    {
        osc::TabRegistryEntry entry
        {
            TabType::id(),
            [](osc::ParentPtr<osc::TabHost> const& h) { return std::make_unique<TabType>(h); },
        };
        registry.registerTab(entry);
    }
}

void osc::RegisterDemoTabs(TabRegistry& registry)
{
    RegisterTab<CustomWidgetsTab>(registry);
    RegisterTab<HittestTab>(registry);
    RegisterTab<ImGuiDemoTab>(registry);
    RegisterTab<ImPlotDemoTab>(registry);
    RegisterTab<ImGuizmoDemoTab>(registry);
    RegisterTab<MandelbrotTab>(registry);
    RegisterTab<MeshGenTestTab>(registry);
}
