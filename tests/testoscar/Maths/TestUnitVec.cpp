#include <oscar/Maths/UnitVec.h>
#include <oscar/Maths/UnitVec3.h>

#include <oscar/Maths/GeometricFunctions.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace osc;

TEST(UnitVec, DefaultConstructorFillsWithNaNs)
{
    ASSERT_TRUE(std::isnan(UnitVec3{}[0]));
    ASSERT_TRUE(std::isnan(UnitVec3{}[1]));
    ASSERT_TRUE(std::isnan(UnitVec3{}[2]));
}

TEST(UnitVec, IsConstexprDefaultConstructible)
{
    [[maybe_unused]] constexpr UnitVec3 v;
}

TEST(UnitVec, IsConstexprNegateable)
{
    constexpr UnitVec3 v;
    [[maybe_unused]] constexpr UnitVec3 neg = -v;
}

TEST(UnitVec, IsConstexprPositiveable)
{
    constexpr UnitVec3 v;
    [[maybe_unused]] constexpr UnitVec3 neg = +v;
}

TEST(UnitVec, NormalizesVecArgs)
{
    ASSERT_EQ(UnitVec3(2.0f, 0.0f, 0.0f), UnitVec3(1.0f, 0.0f, 0.0f));
    ASSERT_EQ(UnitVec3(0.0f, 3.0f, 0.0f), UnitVec3(0.0f, 1.0f, 0.0f));
}
