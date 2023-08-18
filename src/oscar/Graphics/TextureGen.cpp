#include "TextureGen.hpp"

#include "oscar/Graphics/ColorSpace.hpp"
#include "oscar/Graphics/GraphicsHelpers.hpp"
#include "oscar/Graphics/Rgba32.hpp"
#include "oscar/Graphics/TextureFormat.hpp"

#include <glm/vec2.hpp>

#include <array>
#include <cstddef>
#include <optional>


osc::Texture2D osc::GenChequeredFloorTexture()
{
    constexpr size_t chequerWidth = 32;
    constexpr size_t chequerHeight = 32;
    constexpr size_t textureWidth = 2 * chequerWidth;
    constexpr size_t textureHeight = 2 * chequerHeight;
    constexpr Rgba32 onColor = {0xff, 0xff, 0xff, 0xff};
    constexpr Rgba32 offColor = {0xf3, 0xf3, 0xf3, 0xff};

    std::array<Rgba32, textureWidth * textureHeight> pixels{};
    for (size_t row = 0; row < textureHeight; ++row)
    {
        size_t const rowStart = row * textureWidth;
        bool const yOn = (row / chequerHeight) % 2 == 0;
        for (size_t col = 0; col < textureWidth; ++col)
        {
            bool const xOn = (col / chequerWidth) % 2 == 0;
            pixels[rowStart + col] = yOn ^ xOn ? onColor : offColor;
        }
    }

    Texture2D rv
    {
        glm::vec2{textureWidth, textureHeight},
        TextureFormat::RGBA32,
        nonstd::span<uint8_t const>{&pixels.front().r, sizeof(pixels)},
        ColorSpace::sRGB,
    };
    rv.setFilterMode(TextureFilterMode::Mipmap);
    rv.setWrapMode(TextureWrapMode::Repeat);
    return rv;
}
