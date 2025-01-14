#include "IntegratorOutputExtractor.h"

#include <OpenSimCreator/Documents/Simulation/SimulationReport.h>
#include <OpenSimCreator/OutputExtractors/IOutputExtractor.h>

#include <oscar/Maths/Constants.h>
#include <oscar/Utils/Assertions.h>
#include <oscar/Utils/HashHelpers.h>
#include <oscar/Utils/UID.h>
#include <simmath/Integrator.h>

#include <cmath>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

using namespace osc;

namespace
{
    std::vector<OutputExtractor> ConstructIntegratorOutputExtractors()
    {
        std::vector<OutputExtractor> rv;
        rv.emplace_back(IntegratorOutputExtractor{
            "AccuracyInUse",
            "The accuracy which is being used for error control. Usually this is the same value that was specified to setAccuracy()",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getAccuracyInUse()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "PredictedNextStepSize",
            "The step size that will be attempted first on the next call to stepTo() or stepBy().",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getPredictedNextStepSize()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumStepsAttempted",
            "The total number of steps that have been attempted (successfully or unsuccessfully)",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumStepsAttempted()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumStepsTaken",
            "The total number of steps that have been successfully taken",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumStepsTaken()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumRealizations",
            "The total number of state realizations that have been performed",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumRealizations()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumQProjections",
            "The total number of times a state positions Q have been projected",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumQProjections()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumUProjections",
            "The total number of times a state velocities U have been projected",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumUProjections()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumErrorTestFailures",
            "The number of attempted steps that have failed due to the error being unacceptably high",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumErrorTestFailures()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumConvergenceTestFailures",
            "The number of attempted steps that failed due to non-convergence of internal step iterations. This is most common with iterative methods but can occur if for some reason a step can't be completed.",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumConvergenceTestFailures()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumRealizationFailures",
            "The number of attempted steps that have failed due to an error when realizing the state",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumRealizationFailures()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumQProjectionFailures",
            "The number of attempted steps that have failed due to an error when projecting the state positions (Q)",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumQProjectionFailures()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumUProjectionFailures",
            "The number of attempted steps that have failed due to an error when projecting the state velocities (U)",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumUProjectionFailures()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumProjectionFailures",
            "The number of attempted steps that have failed due to an error when projecting the state (either a Q- or U-projection)",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumProjectionFailures()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumConvergentIterations",
            "For iterative methods, the number of internal step iterations in steps that led to convergence (not necessarily successful steps).",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumConvergentIterations()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumDivergentIterations",
            "For iterative methods, the number of internal step iterations in steps that did not lead to convergence.",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumDivergentIterations()); }
        });
        rv.emplace_back(IntegratorOutputExtractor{
            "NumIterations",
            "For iterative methods, this is the total number of internal step iterations taken regardless of whether those iterations led to convergence or to successful steps. This is the sum of the number of convergent and divergent iterations which are available separately.",
            [](SimTK::Integrator const& inter) { return static_cast<float>(inter.getNumIterations()); }
        });
        return rv;
    }

    std::vector<OutputExtractor> const& GetAllIntegratorOutputExtractors()
    {
        static std::vector<OutputExtractor> const s_IntegratorOutputs = ConstructIntegratorOutputExtractors();
        return s_IntegratorOutputs;
    }
}

// public API

osc::IntegratorOutputExtractor::IntegratorOutputExtractor(std::string_view name,
                                                          std::string_view description,
                                                          ExtractorFn extractor) :
    m_Name{name},
    m_Description{description},
    m_Extractor{extractor}
{
}

CStringView osc::IntegratorOutputExtractor::getName() const
{
    return m_Name;
}

CStringView osc::IntegratorOutputExtractor::getDescription() const
{
    return m_Description;
}

OutputType osc::IntegratorOutputExtractor::getOutputType() const
{
    return OutputType::Float;
}

float osc::IntegratorOutputExtractor::getValueFloat(OpenSim::Component const&, SimulationReport const& report) const
{
    return report.getAuxiliaryValue(m_AuxiliaryDataID).value_or(quiet_nan_v<float>);
}

void osc::IntegratorOutputExtractor::getValuesFloat(OpenSim::Component const&,
                                           std::span<SimulationReport const> reports,
                                           std::span<float> out) const
{
    OSC_ASSERT_ALWAYS(reports.size() == out.size());
    for (size_t i = 0; i < reports.size(); ++i)
    {
        out[i] = reports[i].getAuxiliaryValue(m_AuxiliaryDataID).value_or(quiet_nan_v<float>);
    }
}

std::string osc::IntegratorOutputExtractor::getValueString(OpenSim::Component const&, SimulationReport const& report) const
{
    return std::to_string(report.getAuxiliaryValue(m_AuxiliaryDataID).value_or(quiet_nan_v<float>));
}

UID osc::IntegratorOutputExtractor::getAuxiliaryDataID() const
{
    return m_AuxiliaryDataID;
}

IntegratorOutputExtractor::ExtractorFn osc::IntegratorOutputExtractor::getExtractorFunction() const
{
    return m_Extractor;
}

std::size_t osc::IntegratorOutputExtractor::getHash() const
{
    return HashOf(m_AuxiliaryDataID, m_Name, m_Description, m_Extractor);
}

bool osc::IntegratorOutputExtractor::equals(IOutputExtractor const& other) const
{
    if (this == &other)
    {
        return true;
    }

    auto const* const otherT = dynamic_cast<IntegratorOutputExtractor const*>(&other);
    if (!otherT)
    {
        return false;
    }

    return
        m_AuxiliaryDataID == otherT->m_AuxiliaryDataID &&
        m_Name == otherT->m_Name &&
        m_Description == otherT->m_Description &&
        m_Extractor == otherT->m_Extractor;
}

int osc::GetNumIntegratorOutputExtractors()
{
    return static_cast<int>(GetAllIntegratorOutputExtractors().size());
}

IntegratorOutputExtractor const& osc::GetIntegratorOutputExtractor(int idx)
{
    return dynamic_cast<IntegratorOutputExtractor const&>(GetAllIntegratorOutputExtractors().at(static_cast<size_t>(idx)).getInner());
}

OutputExtractor osc::GetIntegratorOutputExtractorDynamic(int idx)
{
    return GetAllIntegratorOutputExtractors().at(static_cast<size_t>(idx));
}
