#pragma once

#include <nonstd/span.hpp>

#include <memory>

namespace OpenSim {
    class Joint;
    class Constraint;
    class ContactGeometry;
    class Force;
}

namespace osmv {

    // static registry of types. The registry is guaranteed to:
    //
    // - return entries in constant time
    // - return entries contiguously in memory
    // - return entires in a format that's useful for downstream (e.g. contiguous strings for ImGui)
    template<typename T>
    struct Type_registry {
        [[nodiscard]] static nonstd::span<std::unique_ptr<T const> const> prototypes() noexcept;
        [[nodiscard]] static nonstd::span<char const* const> names() noexcept;
        [[nodiscard]] static nonstd::span<char const* const> descriptions() noexcept;
    };

    struct joint : Type_registry<OpenSim::Joint> {};
    struct contact_geom : Type_registry<OpenSim::ContactGeometry> {};
    struct constraint : Type_registry<OpenSim::Constraint> {};
    struct force : Type_registry<OpenSim::Force> {};
}
