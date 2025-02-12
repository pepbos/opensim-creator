#include "Perf.h"

#include <oscar/Utils/HashHelpers.h>
#include <oscar/Utils/SynchronizedValue.h>

#include <unordered_map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

using namespace osc;

namespace
{
    int64_t GenerateID(
        std::string_view label,
        std::string_view filename,
        unsigned int line)
    {
        return static_cast<int64_t>(HashOf(label, filename, line));
    }

    SynchronizedValue<std::unordered_map<int64_t, PerfMeasurement>>& GetMeasurementStorage()
    {
        static SynchronizedValue<std::unordered_map<int64_t, PerfMeasurement>> s_Measurements;
        return s_Measurements;
    }
}


// public API

int64_t osc::detail::AllocateMeasurementID(std::string_view label, std::string_view filename, unsigned int line)
{
    int64_t id = GenerateID(label, filename, line);
    auto metadata = std::make_shared<PerfMeasurementMetadata>(id, label, filename, line);

    auto guard = GetMeasurementStorage().lock();
    guard->emplace(std::piecewise_construct, std::tie(id), std::tie(metadata));
    return id;
}

void osc::detail::SubmitMeasurement(int64_t id, PerfClock::time_point start, PerfClock::time_point end)
{
    auto guard = GetMeasurementStorage().lock();
    auto it = guard->find(id);

    if (it != guard->end())
    {
        it->second.submit(start, end);
    }
}

void osc::ClearAllPerfMeasurements()
{
    auto guard = GetMeasurementStorage().lock();

    for (auto& [id, data] : *guard)
    {
        data.clear();
    }
}

std::vector<PerfMeasurement> osc::GetAllPerfMeasurements()
{
    auto guard = GetMeasurementStorage().lock();

    std::vector<PerfMeasurement> rv;
    rv.reserve(guard->size());
    for (auto const& [id, measurement] : *guard)
    {
        rv.push_back(measurement);
    }
    return rv;
}
