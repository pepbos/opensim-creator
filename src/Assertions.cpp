#include "Assertions.hpp"

#include "src/Log.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

void osc::onAssertionFailure(char const* failing_code,
                             char const* func,
                             char const* file,
                             unsigned int line)
{
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s:%s:%u: Assertion '%s' failed", file, func, line, failing_code);
    throw std::runtime_error{buf};
}
