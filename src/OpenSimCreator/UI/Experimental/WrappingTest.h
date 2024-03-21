#pragma once

#include "OpenSimCreator/UI/Experimental/WrappingMath.h"
#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>

namespace osc
{
    struct GeodesicTestBounds
    {
        double eps = 1e-9;
        double varEps = 1e-9;
        double variation = 1e-4;
        size_t integratorSteps = 1000;
    };

    class TestRapport
    {
        public:
        TestRapport(const std::string& name);

        void print(std::ostream& os);

        void newSection(const std::string& section);

        void newSubSection(const std::string& subSection);

        void assertEq(
                Vector3 lhs,
                Vector3 rhs,
                const std::string& msg,
                double eps);

        void assertEq(
                double lhs,
                double rhs,
                const std::string& msg,
                double eps);

        void assertEq(
                Vector3 lhs,
                double rhs,
                const std::string& msg,
                double eps);

        void assertValue(
                bool value,
                const std::string& msg);

        void finalize();

        bool success() const {return _success;}

        void process(bool result);

        std::ostringstream _oss;

        std::string _indent;
        std::string _name;
        std::string _section;
        std::string _subSection;

        bool _success = true;
        bool _sectionSuccess = true;
        bool _subSectionSuccess = true;
        size_t _countFail = 0;
        size_t _count = 0;
    };

bool RunAllWrappingTests(std::ostream& os);

bool RunImplicitGeodesicTest(
    const ImplicitSurface& s,
    const Geodesic& g,
    GeodesicTestBounds bnds,
    const std::string& name,
    std::ostream& os);

} // namespace osc
