#pragma once

#include "src/Maths/PolarPerspectiveCamera.hpp"
#include "src/OpenSimBindings/Graphics/CustomDecorationOptions.hpp"
#include "src/OpenSimBindings/Graphics/CustomRenderingOptions.hpp"

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace osc
{
    struct ModelRendererParams final {
        ModelRendererParams();

        CustomDecorationOptions decorationOptions;
        CustomRenderingOptions renderingOptions;
        glm::vec3 lightColor;
        glm::vec4 backgroundColor;
        glm::vec3 floorLocation;
        PolarPerspectiveCamera camera;
    };
}