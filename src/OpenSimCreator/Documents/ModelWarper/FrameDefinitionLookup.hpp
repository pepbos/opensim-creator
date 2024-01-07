#pragma once

#include <OpenSimCreator/Documents/Frames/FrameDefinition.hpp>
#include <OpenSimCreator/Documents/Frames/FramesFile.hpp>

#include <oscar/Utils/ClonePtr.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace OpenSim { class Model; }
namespace osc::mow { class ModelWarpConfiguration; }

namespace osc::mow
{
    class FrameDefinitionLookup final {
    public:
        FrameDefinitionLookup() = default;

        FrameDefinitionLookup(
            std::filesystem::path const& modelPath,
            OpenSim::Model const&
        );

        bool hasFrameDefinitionFile() const;
        std::filesystem::path recommendedFrameDefinitionFilepath() const;
        bool hasFramesFileLoadError() const;
        std::optional<std::string> getFramesFileLoadError() const;

        frames::FrameDefinition const* lookup(std::string const& frameComponentAbsPath) const;

    private:
        struct DefaultInitialized final {};
        struct FileDoesntExist final {};
        using InnerVariant = std::variant<
            DefaultInitialized,
            frames::FramesFile,
            FileDoesntExist,
            std::string
        >;
        static InnerVariant TryLoadFramesFile(std::filesystem::path const&);

        std::filesystem::path m_ExpectedFrameDefinitionFilepath;
        InnerVariant m_FramesFileOrLoadError = DefaultInitialized{};
    };
}