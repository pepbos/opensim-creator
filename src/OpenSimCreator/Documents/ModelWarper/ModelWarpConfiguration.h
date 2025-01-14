#pragma once

#include <oscar/Maths/CommonFunctions.h>

#include <filesystem>

namespace OpenSim { class Model; }

namespace osc::mow
{
    class ModelWarpConfiguration final {
    public:
        ModelWarpConfiguration() = default;
        ModelWarpConfiguration(
            std::filesystem::path const& osimFileLocation,
            OpenSim::Model const&
        );

        float getWarpBlendingFactor() const { return m_WarpBlendingFactor; }
        void setWarpBlendingFactor(float v) { m_WarpBlendingFactor = saturate(v); }

    private:
        float m_WarpBlendingFactor = 1.0f;
    };
}
