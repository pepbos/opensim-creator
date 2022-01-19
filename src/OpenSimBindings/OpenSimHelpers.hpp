#pragma once

#include <OpenSim/Common/Component.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <string_view>

namespace OpenSim
{
    class AbstractSocket;
}

namespace osc {
    class ComponentPathPtrs {
        static constexpr size_t max_component_depth = 16;
        using container = std::array<OpenSim::Component const*, max_component_depth>;

        container els{};
        size_t n;

    public:
        explicit ComponentPathPtrs(OpenSim::Component const& c) : n{0} {
            OpenSim::Component const* cp = &c;

            els[n++] = cp;
            while (cp->hasOwner()) {
                if (n >= max_component_depth) {
                    throw std::runtime_error{
                        "cannot traverse hierarchy to a component: it is deeper than 32 levels in the component tree, which isn't currently supported by osc"};
                }

                cp = &cp->getOwner();
                els[n++] = cp;
            }
            std::reverse(els.begin(), els.begin() + n);
        }

        [[nodiscard]] constexpr container::const_iterator begin() const noexcept {
            return els.begin();
        }

        [[nodiscard]] constexpr container::const_iterator end() const noexcept {
            return els.begin() + n;
        }

        [[nodiscard]] constexpr bool empty() const noexcept {
            return n == 0;
        }
    };

    inline ComponentPathPtrs path_to(OpenSim::Component const& c) {
        return ComponentPathPtrs{c};
    }

    std::vector<OpenSim::AbstractSocket const*> GetAllSockets(OpenSim::Component&);
    std::vector<OpenSim::AbstractSocket const*> GetSocketsWithTypeName(OpenSim::Component& c, std::string_view);
    std::vector<OpenSim::AbstractSocket const*> GetPhysicalFrameSockets(OpenSim::Component& c);

    // returns non-nullptr if the given path resolves a component relative to root
    OpenSim::Component const* FindComponent(OpenSim::Component const& root, OpenSim::ComponentPath const&);

    // return non-nullptr if the given path resolves a component of type T relative to root
    template<typename T>
    T const* FindComponent(OpenSim::Component const& root, OpenSim::ComponentPath const& cp)
    {
        return dynamic_cast<T const*>(FindComponent(root, cp));
    }

    // returns true if the path resolves to a component within root
    bool ContainsComponent(OpenSim::Component const& root, OpenSim::ComponentPath const&);
}
