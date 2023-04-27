#include "src/Graphics/GraphicsHelpers.hpp"

#include "src/Graphics/ColorSpace.hpp"
#include "src/Graphics/Texture2D.hpp"
#include "src/Graphics/Image.hpp"
#include "src/Platform/Config.hpp"

#include <gtest/gtest.h>

TEST(GraphicsHelpers, ToTexture2DPropagatesSRGBColorSpace)
{
	uint8_t data[] = {0xff};
	osc::Image srgbImage{{1, 1},  data, 1, osc::ColorSpace::sRGB};

	osc::Texture2D rv = osc::ToTexture2D(srgbImage);

	ASSERT_EQ(rv.getColorSpace(), osc::ColorSpace::sRGB);
}

TEST(GraphicsHelpers, LoadTexture2DFromImageRespectsSRGBColorSpace)
{
	auto config = osc::Config::load();
	auto path = config->getResourceDir() / "textures" / "awesomeface.png";

	osc::Texture2D const rv = osc::LoadTexture2DFromImage(path, osc::ColorSpace::sRGB);

	ASSERT_EQ(rv.getColorSpace(), osc::ColorSpace::sRGB);
}

TEST(GraphicsHelpers, LoadTexture2DFromImageRespectsLinearColorSpace)
{
	auto config = osc::Config::load();
	auto path = config->getResourceDir() / "textures" / "awesomeface.png";

	osc::Texture2D const rv = osc::LoadTexture2DFromImage(path, osc::ColorSpace::Linear);

	ASSERT_EQ(rv.getColorSpace(), osc::ColorSpace::Linear);
}