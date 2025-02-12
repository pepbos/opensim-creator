#pragma once

#include <oscar/Maths/Vec3.h>

#include <iosfwd>

namespace osc
{
    // represents an axis-aligned bounding box (AABB)
    struct AABB final {

        friend bool operator==(AABB const&, AABB const&) = default;

        Vec3 min{};
        Vec3 max{};
    };

    std::ostream& operator<<(std::ostream&, AABB const&);
}
