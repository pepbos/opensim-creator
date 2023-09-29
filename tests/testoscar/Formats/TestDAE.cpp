#include "oscar/Formats/DAE.hpp"

#include "testoscar/testoscarconfig.hpp"

#include "oscar/Graphics/MeshGen.hpp"
#include "oscar/Graphics/SceneDecoration.hpp"
#include "oscar/Utils/StringHelpers.hpp"

#include <gtest/gtest.h>

#include <sstream>

TEST(DAE, WriteDecorationsAsDAEWorksForEmptyScene)
{
    osc::DAEMetadata metadata{TESTOSCAR_APPNAME_STRING, TESTOSCAR_APPNAME_STRING};

    std::stringstream ss;
    osc::WriteDecorationsAsDAE(ss, {}, metadata);

    ASSERT_FALSE(ss.str().empty());
}

TEST(DAE, WriteDecorationsAsDAEWorksForNonEmptyScene)
{
    osc::DAEMetadata metadata{TESTOSCAR_APPNAME_STRING, TESTOSCAR_APPNAME_STRING};

    osc::SceneDecoration dec{osc::GenCube()};

    std::stringstream ss;
    osc::WriteDecorationsAsDAE(ss, {&dec, 1}, metadata);

    ASSERT_FALSE(ss.str().empty());
}

TEST(DAE, SetAuthorWritesAuthorToOutput)
{
    osc::DAEMetadata metadata{TESTOSCAR_APPNAME_STRING, TESTOSCAR_APPNAME_STRING};
    metadata.author = "TestThis";

    std::stringstream ss;
    osc::WriteDecorationsAsDAE(ss, {}, metadata);

    ASSERT_TRUE(osc::Contains(ss.str(), metadata.author));
}

TEST(DAE, SetAuthoringToolsWritesAuthoringToolToOutput)
{
    osc::DAEMetadata metadata{TESTOSCAR_APPNAME_STRING, TESTOSCAR_APPNAME_STRING};
    metadata.authoringTool = "TestThis";

    std::stringstream ss;
    osc::WriteDecorationsAsDAE(ss, {}, metadata);

    ASSERT_TRUE(osc::Contains(ss.str(), metadata.authoringTool));
}