#include "MeshWarpPairingLookup.hpp"

#include <OpenSimCreator/Utils/OpenSimHelpers.hpp>

#include <OpenSim/Simulation/Model/Geometry.h>
#include <OpenSim/Simulation/Model/Model.h>
#include <oscar/Platform/Log.hpp>

#include <filesystem>
#include <sstream>
#include <utility>

using osc::mow::MeshWarpPairing;
using osc::FindGeometryFileAbsPath;
using osc::log_error;

namespace
{
    std::unordered_map<std::string, MeshWarpPairing> CreateLut(
        std::filesystem::path const& modelFileLocation,
        OpenSim::Model const& model)
    {
        std::unordered_map<std::string, MeshWarpPairing> rv;
        rv.reserve(osc::GetNumChildren<OpenSim::Mesh>(model));

        // go through each mesh in the `OpenSim::Model` and attempt to load its landmark pairings
        for (auto const& mesh : model.getComponentList<OpenSim::Mesh>())
        {
            if (auto meshPath = FindGeometryFileAbsPath(model, mesh))
            {
                rv.try_emplace(
                    mesh.getAbsolutePathString(),
                    modelFileLocation,
                    std::move(meshPath).value()
                );
            }
            else
            {
                std::stringstream ss;
                ss << mesh.getGeometryFilename() << ": could not find this mesh file: skipping";
                log_error(std::move(ss).str());
            }
        }

        return rv;
    }
}

osc::mow::MeshWarpPairingLookup::MeshWarpPairingLookup(
    std::filesystem::path const& modelFileLocation,
    OpenSim::Model const& model) :

    m_ComponentAbsPathToMeshPairing{CreateLut(modelFileLocation, model)}
{
}
