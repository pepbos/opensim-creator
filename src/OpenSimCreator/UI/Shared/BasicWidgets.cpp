#include "BasicWidgets.h"

#include <OpenSimCreator/Documents/Model/IModelStatePair.h>
#include <OpenSimCreator/Documents/Model/UndoableModelActions.h>
#include <OpenSimCreator/Documents/Model/UndoableModelStatePair.h>
#include <OpenSimCreator/Documents/Simulation/IntegratorMethod.h>
#include <OpenSimCreator/Documents/Simulation/SimulationModelStatePair.h>
#include <OpenSimCreator/Graphics/CustomRenderingOptions.h>
#include <OpenSimCreator/Graphics/ModelRendererParams.h>
#include <OpenSimCreator/Graphics/MuscleDecorationStyle.h>
#include <OpenSimCreator/Graphics/MuscleSizingStyle.h>
#include <OpenSimCreator/Graphics/OpenSimDecorationGenerator.h>
#include <OpenSimCreator/Graphics/OpenSimDecorationOptions.h>
#include <OpenSimCreator/OutputExtractors/ComponentOutputExtractor.h>
#include <OpenSimCreator/OutputExtractors/OutputExtractor.h>
#include <OpenSimCreator/Platform/RecentFile.h>
#include <OpenSimCreator/Platform/RecentFiles.h>
#include <OpenSimCreator/UI/IMainUIStateAPI.h>
#include <OpenSimCreator/Utils/OpenSimHelpers.h>
#include <OpenSimCreator/Utils/ParamBlock.h>
#include <OpenSimCreator/Utils/ParamValue.h>
#include <OpenSimCreator/Utils/SimTKHelpers.h>

#include <IconsFontAwesome5.h>
#include <OpenSim/Common/Component.h>
#include <OpenSim/Common/ComponentOutput.h>
#include <OpenSim/Simulation/Model/Frame.h>
#include <OpenSim/Simulation/Model/Geometry.h>
#include <OpenSim/Simulation/Model/Model.h>
#include <OpenSim/Simulation/Model/Point.h>
#include <oscar/Formats/DAE.h>
#include <oscar/Formats/OBJ.h>
#include <oscar/Formats/STL.h>
#include <oscar/Graphics/Color.h>
#include <oscar/Graphics/Mesh.h>
#include <oscar/Graphics/Scene/SceneCache.h>
#include <oscar/Graphics/Scene/SceneDecoration.h>
#include <oscar/Maths/MathHelpers.h>
#include <oscar/Maths/Rect.h>
#include <oscar/Maths/VecFunctions.h>
#include <oscar/Maths/Vec2.h>
#include <oscar/Maths/Vec3.h>
#include <oscar/Platform/App.h>
#include <oscar/Platform/AppMetadata.h>
#include <oscar/Platform/Log.h>
#include <oscar/Platform/os.h>
#include <oscar/UI/IconCache.h>
#include <oscar/UI/ImGuiHelpers.h>
#include <oscar/UI/oscimgui.h>
#include <oscar/UI/oscimgui_internal.h>
#include <oscar/UI/Widgets/IconWithMenu.h>
#include <oscar/UI/Widgets/CameraViewAxes.h>
#include <oscar/Utils/ParentPtr.h>
#include <oscar/Utils/StringHelpers.h>
#include <SimTKcommon/basics.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

using namespace osc::literals;
using namespace osc;

// export utils
namespace
{
    // prompts the user for a save location and then exports a DAE file containing the 3D scene
    void TryPromptUserToSaveAsDAE(std::span<SceneDecoration const> scene)
    {
        std::optional<std::filesystem::path> maybeDAEPath =
            PromptUserForFileSaveLocationAndAddExtensionIfNecessary("dae");

        if (!maybeDAEPath)
        {
            return;  // user cancelled out
        }
        std::filesystem::path const& daePath = *maybeDAEPath;

        std::ofstream outfile{daePath};

        if (!outfile)
        {
            log_error("cannot save to %s: IO error", daePath.string().c_str());
            return;
        }

        AppMetadata const& appMetadata = App::get().getMetadata();
        DAEMetadata daeMetadata
        {
            GetBestHumanReadableApplicationName(appMetadata),
            CalcFullApplicationNameWithVersionAndBuild(appMetadata),
        };

        WriteDecorationsAsDAE(outfile, scene, daeMetadata);
        log_info("wrote scene as a DAE file to %s", daePath.string().c_str());
    }

    void DrawOutputTooltip(OpenSim::AbstractOutput const& o)
    {
        DrawTooltip(o.getTypeName());
    }

    bool DrawOutputWithSubfieldsMenu(IMainUIStateAPI& api, OpenSim::AbstractOutput const& o)
    {
        bool outputAdded = false;
        OutputSubfield supportedSubfields = GetSupportedSubfields(o);

        // can plot suboutputs
        if (ImGui::BeginMenu(("  " + o.getName()).c_str()))
        {
            for (OutputSubfield f : GetAllSupportedOutputSubfields())
            {
                if (f & supportedSubfields)
                {
                    if (auto label = GetOutputSubfieldLabel(f); label && ImGui::MenuItem(label->c_str()))
                    {
                        api.addUserOutputExtractor(OutputExtractor{ComponentOutputExtractor{o, f}});
                        outputAdded = true;
                    }
                }
            }
            ImGui::EndMenu();
        }

        if (ImGui::IsItemHovered())
        {
            DrawOutputTooltip(o);
        }

        return outputAdded;
    }

    bool DrawOutputWithNoSubfieldsMenuItem(IMainUIStateAPI& api, OpenSim::AbstractOutput const& o)
    {
        // can only plot top-level of output

        bool outputAdded = false;

        if (ImGui::MenuItem(("  " + o.getName()).c_str()))
        {
            api.addUserOutputExtractor(OutputExtractor{ComponentOutputExtractor{o}});
            outputAdded = true;
        }

        if (ImGui::IsItemHovered())
        {
            DrawOutputTooltip(o);
        }

        return outputAdded;
    }

    bool DrawRequestOutputMenuOrMenuItem(IMainUIStateAPI& api, OpenSim::AbstractOutput const& o)
    {
        if (GetSupportedSubfields(o) == OutputSubfield::None)
        {
            return DrawOutputWithNoSubfieldsMenuItem(api, o);
        }
        else
        {
            return DrawOutputWithSubfieldsMenu(api, o);
        }
    }

    void DrawSimulationParamValue(ParamValue const& v)
    {
        if (std::holds_alternative<double>(v))
        {
            ImGui::Text("%f", static_cast<float>(std::get<double>(v)));
        }
        else if (std::holds_alternative<IntegratorMethod>(v))
        {
            ImGui::Text("%s", GetIntegratorMethodString(std::get<IntegratorMethod>(v)).c_str());
        }
        else if (std::holds_alternative<int>(v))
        {
            ImGui::Text("%i", std::get<int>(v));
        }
        else
        {
            ImGui::Text("(unknown value type)");
        }
    }

    Transform CalcTransformWithRespectTo(
        OpenSim::Mesh const& mesh,
        OpenSim::Frame const& frame,
        SimTK::State const& state)
    {
        Transform rv = decompose_to_transform(mesh.getFrame().findTransformBetween(state, frame));
        rv.scale = ToVec3(mesh.get_scale_factors());
        return rv;
    }

    void ActionReexportMeshOBJWithRespectTo(
        OpenSim::Model const& model,
        SimTK::State const& state,
        OpenSim::Mesh const& openSimMesh,
        OpenSim::Frame const& frame)
    {
        // prompt user for a save location
        std::optional<std::filesystem::path> const maybeUserSaveLocation =
            PromptUserForFileSaveLocationAndAddExtensionIfNecessary("obj");
        if (!maybeUserSaveLocation)
        {
            return;  // user didn't select a save location
        }
        std::filesystem::path const& userSaveLocation = *maybeUserSaveLocation;

        // load raw mesh data into an osc mesh for processing
        Mesh oscMesh = ToOscMesh(model, state, openSimMesh);

        // bake transform into mesh data
        oscMesh.transformVerts(CalcTransformWithRespectTo(openSimMesh, frame, state));

        // write transformed mesh to output
        std::ofstream outputFileStream
        {
            userSaveLocation,
            std::ios_base::out | std::ios_base::trunc | std::ios_base::binary,
        };
        if (!outputFileStream)
        {
            std::string const error = CurrentErrnoAsString();
            log_error("%s: could not save obj output: %s", userSaveLocation.string().c_str(), error.c_str());
            return;
        }

        AppMetadata const& appMetadata = App::get().getMetadata();
        ObjMetadata const objMetadata
        {
            CalcFullApplicationNameWithVersionAndBuild(appMetadata),
        };

        WriteMeshAsObj(
            outputFileStream,
            oscMesh,
            objMetadata,
            ObjWriterFlags::NoWriteNormals
        );
    }

    void ActionReexportMeshSTLWithRespectTo(
        OpenSim::Model const& model,
        SimTK::State const& state,
        OpenSim::Mesh const& openSimMesh,
        OpenSim::Frame const& frame)
    {
        // prompt user for a save location
        std::optional<std::filesystem::path> const maybeUserSaveLocation =
            PromptUserForFileSaveLocationAndAddExtensionIfNecessary("stl");
        if (!maybeUserSaveLocation)
        {
            return;  // user didn't select a save location
        }
        std::filesystem::path const& userSaveLocation = *maybeUserSaveLocation;

        // load raw mesh data into an osc mesh for processing
        Mesh oscMesh = ToOscMesh(model, state, openSimMesh);

        // bake transform into mesh data
        oscMesh.transformVerts(CalcTransformWithRespectTo(openSimMesh, frame, state));

        // write transformed mesh to output
        std::ofstream outputFileStream
        {
            userSaveLocation,
            std::ios_base::out | std::ios_base::trunc | std::ios_base::binary,
        };
        if (!outputFileStream)
        {
            std::string const error = CurrentErrnoAsString();
            log_error("%s: could not save obj output: %s", userSaveLocation.string().c_str(), error.c_str());
            return;
        }

        AppMetadata const& appMetadata = App::get().getMetadata();
        StlMetadata const stlMetadata
        {
            CalcFullApplicationNameWithVersionAndBuild(appMetadata),
        };

        WriteMeshAsStl(outputFileStream, oscMesh, stlMetadata);
    }
}


// public API

void osc::DrawNothingRightClickedContextMenuHeader()
{
    ImGui::TextDisabled("(nothing selected)");
}

void osc::DrawContextMenuHeader(CStringView title, CStringView subtitle)
{
    ImGui::TextUnformatted(title.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("%s", subtitle.c_str());
}

void osc::DrawRightClickedComponentContextMenuHeader(OpenSim::Component const& c)
{
    DrawContextMenuHeader(Ellipsis(c.getName(), 15), c.getConcreteClassName());
}

void osc::DrawContextMenuSeparator()
{
    ImGui::Separator();
    ImGui::Dummy({0.0f, 3.0f});
}

void osc::DrawComponentHoverTooltip(OpenSim::Component const& hovered)
{
    BeginTooltip();

    ImGui::TextUnformatted(hovered.getName().c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("%s", hovered.getConcreteClassName().c_str());

    EndTooltip();
}

void osc::DrawSelectOwnerMenu(IModelStatePair& model, OpenSim::Component const& selected)
{
    if (ImGui::BeginMenu("Select Owner"))
    {
        model.setHovered(nullptr);

        for (
            OpenSim::Component const* owner = GetOwner(selected);
            owner != nullptr;
            owner = GetOwner(*owner))
        {
            std::string const menuLabel = [&owner]()
            {
                std::stringstream ss;
                ss << owner->getName() << '(' << owner->getConcreteClassName() << ')';
                return std::move(ss).str();
            }();

            if (ImGui::MenuItem(menuLabel.c_str()))
            {
                model.setSelected(owner);
            }
            if (ImGui::IsItemHovered())
            {
                model.setHovered(owner);
            }
        }

        ImGui::EndMenu();
    }
}

bool osc::DrawWatchOutputMenu(IMainUIStateAPI& api, OpenSim::Component const& c)
{
    bool outputAdded = false;

    if (ImGui::BeginMenu("Watch Output"))
    {
        DrawHelpMarker("Watch the selected output. This makes it appear in the 'Output Watches' window in the editor panel and the 'Output Plots' window during a simulation");

        // iterate from the selected component upwards to the root
        int imguiId = 0;
        for (OpenSim::Component const* p = &c; p; p = GetOwner(*p))
        {
            ImGui::PushID(imguiId++);

            ImGui::Dummy({0.0f, 2.0f});
            ImGui::TextDisabled("%s (%s)", p->getName().c_str(), p->getConcreteClassName().c_str());
            ImGui::Separator();

            if (p->getNumOutputs() == 0)
            {
                ImGui::TextDisabled("  (has no outputs)");
            }
            else
            {
                for (auto const& [name, output] : p->getOutputs())
                {
                    if (DrawRequestOutputMenuOrMenuItem(api, *output))
                    {
                        outputAdded = true;
                    }
                }
            }

            ImGui::PopID();
        }

        ImGui::EndMenu();
    }

    return outputAdded;
}

void osc::DrawSimulationParams(ParamBlock const& params)
{
    ImGui::Dummy({0.0f, 1.0f});
    ImGui::TextUnformatted("parameters:");
    ImGui::SameLine();
    DrawHelpMarker("The parameters used when this simulation was launched. These must be set *before* running the simulation");
    ImGui::Separator();
    ImGui::Dummy({0.0f, 2.0f});

    ImGui::Columns(2);
    for (int i = 0, len = params.size(); i < len; ++i)
    {
        std::string const& name = params.getName(i);
        std::string const& description = params.getDescription(i);
        ParamValue const& value = params.getValue(i);

        ImGui::TextUnformatted(name.c_str());
        ImGui::SameLine();
        DrawHelpMarker(name, description);
        ImGui::NextColumn();

        DrawSimulationParamValue(value);
        ImGui::NextColumn();
    }
    ImGui::Columns();
}

void osc::DrawSearchBar(std::string& out)
{
    if (!out.empty())
    {
        if (ImGui::Button("X"))
        {
            out.clear();
        }
        DrawTooltipBodyOnlyIfItemHovered("Clear the search string");
    }
    else
    {
        ImGui::Text(ICON_FA_SEARCH);
    }

    // draw search bar

    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    InputString("##hirarchtsearchbar", out);
}

void osc::DrawOutputNameColumn(
    IOutputExtractor const& output,
    bool centered,
    SimulationModelStatePair* maybeActiveSate)
{
    if (centered)
    {
        TextCentered(output.getName());
    }
    else
    {
        ImGui::TextUnformatted(output.getName().c_str());
    }

    // if it's specifically a component ouptut, then hover/clicking the text should
    // propagate to the rest of the UI
    //
    // (e.g. if the user mouses over the name of a component output it should make
    // the associated component the current hover to provide immediate feedback to
    // the user)
    if (auto const* co = dynamic_cast<ComponentOutputExtractor const*>(&output); co && maybeActiveSate)
    {
        if (ImGui::IsItemHovered())
        {
            maybeActiveSate->setHovered(FindComponent(maybeActiveSate->getModel(), co->getComponentAbsPath()));
        }

        if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
        {
            maybeActiveSate->setSelected(FindComponent(maybeActiveSate->getModel(), co->getComponentAbsPath()));
        }
    }

    if (!output.getDescription().empty())
    {
        ImGui::SameLine();
        DrawHelpMarker(output.getName(), output.getDescription());
    }
}

void osc::DrawWithRespectToMenuContainingMenuPerFrame(
    OpenSim::Component const& root,
    std::function<void(OpenSim::Frame const&)> const& onFrameMenuOpened)
{
    ImGui::TextDisabled("With Respect to:");
    ImGui::Separator();

    int imguiID = 0;
    for (OpenSim::Frame const& frame : root.getComponentList<OpenSim::Frame>())
    {
        ImGui::PushID(imguiID++);
        if (ImGui::BeginMenu(frame.getName().c_str()))
        {
            onFrameMenuOpened(frame);
            ImGui::EndMenu();
        }
        ImGui::PopID();
    }
}

void osc::DrawWithRespectToMenuContainingMenuItemPerFrame(
    OpenSim::Component const& root,
    std::function<void(OpenSim::Frame const&)> const& onFrameMenuItemClicked)
{
    ImGui::TextDisabled("With Respect to:");
    ImGui::Separator();

    int imguiID = 0;
    for (OpenSim::Frame const& frame : root.getComponentList<OpenSim::Frame>())
    {
        ImGui::PushID(imguiID++);
        if (ImGui::MenuItem(frame.getName().c_str()))
        {
            onFrameMenuItemClicked(frame);
        }
        ImGui::PopID();
    }
}

void osc::DrawPointTranslationInformationWithRespectTo(
    OpenSim::Frame const& frame,
    SimTK::State const& state,
    Vec3 locationInGround)
{
    SimTK::Transform const groundToFrame = frame.getTransformInGround(state).invert();
    Vec3 position = ToVec3(groundToFrame * ToSimTKVec3(locationInGround));

    ImGui::Text("translation");
    ImGui::SameLine();
    DrawHelpMarker("translation", "Translational offset (in meters) of the point expressed in the chosen frame");
    ImGui::SameLine();
    ImGui::InputFloat3("##translation", value_ptr(position), "%.6f", ImGuiInputTextFlags_ReadOnly);
}

void osc::DrawDirectionInformationWithRepsectTo(
    OpenSim::Frame const& frame,
    SimTK::State const& state,
    Vec3 directionInGround)
{
    SimTK::Transform const groundToFrame = frame.getTransformInGround(state).invert();
    Vec3 direction = ToVec3(groundToFrame.xformBaseVecToFrame(ToSimTKVec3(directionInGround)));

    ImGui::Text("direction");
    ImGui::SameLine();
    DrawHelpMarker("direction", "a unit vector expressed in the given frame");
    ImGui::SameLine();
    ImGui::InputFloat3("##direction", value_ptr(direction), "%.6f", ImGuiInputTextFlags_ReadOnly);
}

void osc::DrawFrameInformationExpressedIn(
    OpenSim::Frame const& parent,
    SimTK::State const& state,
    OpenSim::Frame const& otherFrame)
{
    SimTK::Transform const xform = parent.findTransformBetween(state, otherFrame);
    Vec3 position = ToVec3(xform.p());
    Vec3 rotationEulers = ToVec3(xform.R().convertRotationToBodyFixedXYZ());

    ImGui::Text("translation");
    ImGui::SameLine();
    DrawHelpMarker("translation", "Translational offset (in meters) of the frame's origin expressed in the chosen frame");
    ImGui::SameLine();
    ImGui::InputFloat3("##translation", value_ptr(position), "%.6f", ImGuiInputTextFlags_ReadOnly);

    ImGui::Text("orientation");
    ImGui::SameLine();
    DrawHelpMarker("orientation", "Orientation offset (in radians) of the frame, expressed in the chosen frame as a frame-fixed x-y-z rotation sequence");
    ImGui::SameLine();
    ImGui::InputFloat3("##orientation", value_ptr(rotationEulers), "%.6f", ImGuiInputTextFlags_ReadOnly);
}

bool osc::BeginCalculateMenu(CalculateMenuFlags flags)
{
    CStringView const label = flags & CalculateMenuFlags::NoCalculatorIcon ?
        "Calculate" :
        ICON_FA_CALCULATOR " Calculate";
    return ImGui::BeginMenu(label.c_str());
}

void osc::EndCalculateMenu()
{
    ImGui::EndMenu();
}

void osc::DrawCalculatePositionMenu(
    OpenSim::Component const& root,
    SimTK::State const& state,
    OpenSim::Point const& point)
{
    if (ImGui::BeginMenu("Position"))
    {
        auto const onFrameMenuOpened = [&state, &point](OpenSim::Frame const& frame)
        {
            DrawPointTranslationInformationWithRespectTo(
                frame,
                state,
                ToVec3(point.getLocationInGround(state))
            );
        };

        DrawWithRespectToMenuContainingMenuPerFrame(root, onFrameMenuOpened);
        ImGui::EndMenu();
    }
}

void osc::DrawCalculateMenu(
    OpenSim::Component const& root,
    SimTK::State const& state,
    OpenSim::Point const& point,
    CalculateMenuFlags flags)
{
    if (BeginCalculateMenu(flags))
    {
        DrawCalculatePositionMenu(root, state, point);
        EndCalculateMenu();
    }
}

void osc::DrawCalculateTransformMenu(
    OpenSim::Component const& root,
    SimTK::State const& state,
    OpenSim::Frame const& frame)
{
    if (ImGui::BeginMenu("Transform"))
    {
        auto const onFrameMenuOpened = [&state, &frame](OpenSim::Frame const& otherFrame)
        {
            DrawFrameInformationExpressedIn(frame, state, otherFrame);
        };
        DrawWithRespectToMenuContainingMenuPerFrame(root, onFrameMenuOpened);
        ImGui::EndMenu();
    }
}

void osc::DrawCalculateMenu(
    OpenSim::Component const& root,
    SimTK::State const& state,
    OpenSim::Frame const& frame,
    CalculateMenuFlags flags)
{
    if (BeginCalculateMenu(flags))
    {
        DrawCalculateTransformMenu(root, state, frame);
        EndCalculateMenu();
    }
}

void osc::DrawCalculateOriginMenu(
    OpenSim::Component const& root,
    SimTK::State const& state,
    OpenSim::Sphere const& sphere)
{
    if (ImGui::BeginMenu("origin"))
    {
        Vec3 const posInGround = ToVec3(sphere.getFrame().getPositionInGround(state));
        auto const onFrameMenuOpened = [&state, posInGround](OpenSim::Frame const& otherFrame)
        {
            DrawPointTranslationInformationWithRespectTo(otherFrame, state, posInGround);
        };
        DrawWithRespectToMenuContainingMenuPerFrame(root, onFrameMenuOpened);

        ImGui::EndMenu();
    }
}

void osc::DrawCalculateRadiusMenu(
    OpenSim::Component const&,
    SimTK::State const&,
    OpenSim::Sphere const& sphere)
{
    if (ImGui::BeginMenu("radius"))
    {
        double d = sphere.get_radius();
        ImGui::InputDouble("radius", &d);
        ImGui::EndMenu();
    }
}

void osc::DrawCalculateVolumeMenu(
    OpenSim::Component const&,
    SimTK::State const&,
    OpenSim::Sphere const& sphere)
{
    if (ImGui::BeginMenu("volume"))
    {
        double const r = sphere.get_radius();
        double v = 4.0/3.0 * SimTK::Pi * r*r*r;
        ImGui::InputDouble("volume", &v);
        ImGui::EndMenu();
    }
}

void osc::DrawCalculateMenu(
    OpenSim::Component const& root,
    SimTK::State const& state,
    OpenSim::Geometry const& geom,
    CalculateMenuFlags flags)
{
    if (BeginCalculateMenu(flags))
    {
        if (auto const* spherePtr = dynamic_cast<OpenSim::Sphere const*>(&geom))
        {
            DrawCalculateOriginMenu(root, state, *spherePtr);
            DrawCalculateRadiusMenu(root, state, *spherePtr);
            DrawCalculateVolumeMenu(root, state, *spherePtr);
        }
        else
        {
            DrawCalculateTransformMenu(root, state, geom.getFrame());
        }
        EndCalculateMenu();
    }
}

void osc::TryDrawCalculateMenu(
    OpenSim::Component const& root,
    SimTK::State const& state,
    OpenSim::Component const& selected,
    CalculateMenuFlags flags)
{
    if (auto const* const frame = dynamic_cast<OpenSim::Frame const*>(&selected))
    {
        DrawCalculateMenu(root, state, *frame, flags);
    }
    else if (auto const* const point = dynamic_cast<OpenSim::Point const*>(&selected))
    {
        DrawCalculateMenu(root, state, *point, flags);
    }
}

bool osc::DrawMuscleRenderingOptionsRadioButtions(OpenSimDecorationOptions& opts)
{
    MuscleDecorationStyle const currentStyle = opts.getMuscleDecorationStyle();
    bool edited = false;
    for (auto const& metadata : GetAllMuscleDecorationStyleMetadata())
    {
        if (ImGui::RadioButton(metadata.label.c_str(), metadata.value == currentStyle))
        {
            opts.setMuscleDecorationStyle(metadata.value);
            edited = true;
        }
    }
    return edited;
}

bool osc::DrawMuscleSizingOptionsRadioButtons(OpenSimDecorationOptions& opts)
{
    MuscleSizingStyle const currentStyle = opts.getMuscleSizingStyle();
    bool edited = false;
    for (auto const& metadata : GetAllMuscleSizingStyleMetadata())
    {
        if (ImGui::RadioButton(metadata.label.c_str(), metadata.value == currentStyle))
        {
            opts.setMuscleSizingStyle(metadata.value);
            edited = true;
        }
    }
    return edited;
}

bool osc::DrawMuscleColoringOptionsRadioButtons(OpenSimDecorationOptions& opts)
{
    MuscleColoringStyle const currentStyle = opts.getMuscleColoringStyle();
    bool edited = false;
    for (auto const& metadata : GetAllMuscleColoringStyleMetadata())
    {
        if (ImGui::RadioButton(metadata.label.c_str(), metadata.value == currentStyle))
        {
            opts.setMuscleColoringStyle(metadata.value);
            edited = true;
        }
    }
    return edited;
}

bool osc::DrawMuscleDecorationOptionsEditor(OpenSimDecorationOptions& opts)
{
    int id = 0;
    bool edited = false;

    ImGui::PushID(id++);
    ImGui::TextDisabled("Rendering");
    edited = DrawMuscleRenderingOptionsRadioButtions(opts) || edited;
    ImGui::PopID();

    ImGui::Dummy({0.0f, 0.25f*ImGui::GetTextLineHeight()});
    ImGui::PushID(id++);
    ImGui::TextDisabled("Sizing");
    edited = DrawMuscleSizingOptionsRadioButtons(opts) || edited;
    ImGui::PopID();

    ImGui::Dummy({0.0f, 0.25f*ImGui::GetTextLineHeight()});
    ImGui::PushID(id++);
    ImGui::TextDisabled("Coloring");
    edited = DrawMuscleColoringOptionsRadioButtons(opts) || edited;
    ImGui::PopID();

    return edited;
}

bool osc::DrawRenderingOptionsEditor(CustomRenderingOptions& opts)
{
    bool edited = false;
    ImGui::TextDisabled("Rendering");
    for (size_t i = 0; i < opts.getNumOptions(); ++i)
    {
        bool value = opts.getOptionValue(i);
        if (ImGui::Checkbox(opts.getOptionLabel(i).c_str(), &value))
        {
            opts.setOptionValue(i, value);
            edited = true;
        }
    }
    return edited;
}

bool osc::DrawOverlayOptionsEditor(OverlayDecorationOptions& opts)
{
    std::optional<CStringView> lastGroupLabel;
    bool edited = false;
    for (size_t i = 0; i < opts.getNumOptions(); ++i)
    {
        // print header, if necessary
        CStringView const groupLabel = opts.getOptionGroupLabel(i);
        if (groupLabel != lastGroupLabel)
        {
            if (lastGroupLabel)
            {
                ImGui::Dummy({0.0f, 0.25f*ImGui::GetTextLineHeight()});
            }
            ImGui::TextDisabled("%s", groupLabel.c_str());
            lastGroupLabel = groupLabel;
        }

        bool value = opts.getOptionValue(i);
        if (ImGui::Checkbox(opts.getOptionLabel(i).c_str(), &value))
        {
            opts.setOptionValue(i, value);
            edited = true;
        }
    }
    return edited;
}

bool osc::DrawCustomDecorationOptionCheckboxes(OpenSimDecorationOptions& opts)
{
    int imguiID = 0;
    bool edited = false;
    for (size_t i = 0; i < opts.getNumOptions(); ++i)
    {
        ImGui::PushID(imguiID++);

        bool v = opts.getOptionValue(i);
        if (ImGui::Checkbox(opts.getOptionLabel(i).c_str(), &v))
        {
            opts.setOptionValue(i, v);
            edited = true;
        }
        if (std::optional<CStringView> description = opts.getOptionDescription(i))
        {
            ImGui::SameLine();
            DrawHelpMarker(*description);
        }

        ImGui::PopID();
    }
    return edited;
}

bool osc::DrawAdvancedParamsEditor(
    ModelRendererParams& params,
    std::span<SceneDecoration const> drawlist)
{
    bool edited = false;

    if (ImGui::Button("Export to .dae"))
    {
        TryPromptUserToSaveAsDAE(drawlist);
    }
    DrawTooltipBodyOnlyIfItemHovered("Try to export the 3D scene to a portable DAE file, so that it can be viewed in 3rd-party modelling software, such as Blender");

    ImGui::Dummy({0.0f, 10.0f});
    ImGui::Text("advanced camera properties:");
    ImGui::Separator();
    edited = SliderMetersFloat("radius", params.camera.radius, 0.0f, 10.0f) || edited;
    edited = SliderAngle("theta", params.camera.theta, 0_deg, 360_deg) || edited;
    edited = SliderAngle("phi", params.camera.phi, 0_deg, 360_deg) || edited;
    edited = SliderAngle("fov", params.camera.vertical_fov, 0_deg, 360_deg) || edited;
    edited = InputMetersFloat("znear", params.camera.znear) || edited;
    edited = InputMetersFloat("zfar", params.camera.zfar) || edited;
    ImGui::NewLine();
    edited = SliderMetersFloat("pan_x", params.camera.focusPoint.x, -100.0f, 100.0f) || edited;
    edited = SliderMetersFloat("pan_y", params.camera.focusPoint.y, -100.0f, 100.0f) || edited;
    edited = SliderMetersFloat("pan_z", params.camera.focusPoint.z, -100.0f, 100.0f) || edited;

    ImGui::Dummy({0.0f, 10.0f});
    ImGui::Text("advanced scene properties:");
    ImGui::Separator();
    edited = ImGui::ColorEdit3("light_color", value_ptr(params.lightColor)) || edited;
    edited = ImGui::ColorEdit3("background color", value_ptr(params.backgroundColor)) || edited;
    edited = InputMetersFloat3("floor location", params.floorLocation) || edited;
    DrawTooltipBodyOnlyIfItemHovered("Set the origin location of the scene's chequered floor. This is handy if you are working on smaller models, or models that need a floor somewhere else");

    return edited;
}

bool osc::DrawVisualAidsContextMenuContent(ModelRendererParams& params)
{
    bool edited = false;

    // generic rendering options
    edited = DrawRenderingOptionsEditor(params.renderingOptions) || edited;

    // overlay options
    edited = DrawOverlayOptionsEditor(params.overlayOptions) || edited;

    // OpenSim-specific extra rendering options
    ImGui::Dummy({0.0f, 0.25f*ImGui::GetTextLineHeight()});
    ImGui::TextDisabled("OpenSim");
    edited = DrawCustomDecorationOptionCheckboxes(params.decorationOptions) || edited;

    return edited;
}

bool osc::DrawViewerTopButtonRow(
    ModelRendererParams& params,
    std::span<SceneDecoration const>,
    IconCache& iconCache,
    std::function<bool()> const& drawExtraElements)
{
    bool edited = false;

    IconWithMenu muscleStylingButton
    {
        iconCache.getIcon("muscle_coloring"),
        "Muscle Styling",
        "Affects how muscles appear in this visualizer panel",
        [&params]() { return DrawMuscleDecorationOptionsEditor(params.decorationOptions); },
    };
    edited = muscleStylingButton.onDraw() || edited;
    ImGui::SameLine();

    IconWithMenu vizAidsButton
    {
        iconCache.getIcon("viz_aids"),
        "Visual Aids",
        "Affects what's shown in the 3D scene",
        [&params]() { return DrawVisualAidsContextMenuContent(params); },
    };
    edited = vizAidsButton.onDraw() || edited;

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // caller-provided extra buttons (usually, context-dependent)
    edited = drawExtraElements() || edited;

    return edited;
}

bool osc::DrawCameraControlButtons(
    ModelRendererParams& params,
    std::span<SceneDecoration const> drawlist,
    Rect const& viewerScreenRect,
    std::optional<AABB> const& maybeSceneAABB,
    IconCache& iconCache,
    Vec2 desiredTopCentroid)
{
    IconWithoutMenu zoomOutButton{
        iconCache.getIcon("zoomout"),
        "Zoom Out Camera",
        "Moves the camera one step away from its focus point (Hotkey: -)",
    };
    IconWithoutMenu zoomInButton{
        iconCache.getIcon("zoomin"),
        "Zoom in Camera",
        "Moves the camera one step towards its focus point (Hotkey: =)",
    };
    IconWithoutMenu autoFocusButton{
        iconCache.getIcon("zoomauto"),
        "Auto-Focus Camera",
        "Try to automatically adjust the camera's zoom etc. to suit the model's dimensions (Hotkey: Ctrl+F)",
    };
    IconWithMenu sceneSettingsButton{
        iconCache.getIcon("gear"),
        "Scene Settings",
        "Change advanced scene settings",
        [&params, drawlist]() { return DrawAdvancedParamsEditor(params, drawlist); },
    };

    auto c = ImGui::GetStyle().Colors[ImGuiCol_Button];
    c.w *= 0.9f;
    ImGui::PushStyleColor(ImGuiCol_Button, c);

    float const spacing = ImGui::GetStyle().ItemSpacing.x;
    float width = zoomOutButton.dimensions().x + spacing + zoomInButton.dimensions().x + spacing + autoFocusButton.dimensions().x;
    Vec2 const topleft = {desiredTopCentroid.x - 0.5f*width, desiredTopCentroid.y + 2.0f*ImGui::GetStyle().ItemSpacing.y};
    ImGui::SetCursorScreenPos(topleft);

    bool edited = false;
    if (zoomOutButton.onDraw()) {
        ZoomOut(params.camera);
        edited = true;
    }
    ImGui::SameLine();
    if (zoomInButton.onDraw()) {
        ZoomIn(params.camera);
        edited = true;
    }
    ImGui::SameLine();
    if (autoFocusButton.onDraw() && maybeSceneAABB) {
        AutoFocus(params.camera, *maybeSceneAABB, AspectRatio(viewerScreenRect));
        edited = true;
    }

    // next line (centered)
    {
        Vec2 const tl = {
            desiredTopCentroid.x - 0.5f*sceneSettingsButton.dimensions().x,
            ImGui::GetCursorScreenPos().y,
        };
        ImGui::SetCursorScreenPos(tl);
        if (sceneSettingsButton.onDraw()) {
            edited = true;
        }
    }

    ImGui::PopStyleColor();

    return edited;
}

bool osc::DrawViewerImGuiOverlays(
    ModelRendererParams& params,
    std::span<SceneDecoration const> drawlist,
    std::optional<AABB> maybeSceneAABB,
    Rect const& renderRect,
    IconCache& iconCache,
    std::function<bool()> const& drawExtraElementsInTop)
{
    ImGuiStyle const& style = ImGui::GetStyle();

    bool edited = false;

    // draw top-left buttons
    ImGui::SetCursorScreenPos(renderRect.p1 + Vec2{style.WindowPadding});
    edited = DrawViewerTopButtonRow(params, drawlist, iconCache, drawExtraElementsInTop) || edited;

    // draw top-right camera manipulators
    CameraViewAxes axes;
    Vec2 const renderDims = dimensions(renderRect);
    Vec2 const axesDims = axes.dimensions();
    Vec2 const axesTopLeft = {
        renderRect.p1.x + renderDims.x - style.WindowPadding.x - axesDims.x,
        renderRect.p1.y + style.WindowPadding.y,
    };

    // draw the bottom overlays
    ImGui::SetCursorScreenPos(axesTopLeft);
    edited = axes.draw(params.camera) || edited;

    Vec2 const cameraButtonsTopLeft = axesTopLeft + Vec2{0.0f, axesDims.y};
    ImGui::SetCursorScreenPos(cameraButtonsTopLeft);
    edited = DrawCameraControlButtons(
        params,
        drawlist,
        renderRect,
        maybeSceneAABB,
        iconCache,
        {axesTopLeft.x + 0.5f*axesDims.x, axesTopLeft.y + axesDims.y}
    ) || edited;

    return edited;
}

bool osc::BeginToolbar(CStringView label, std::optional<Vec2> padding)
{
    if (padding)
    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, *padding);
    }

    float const height = ImGui::GetFrameHeight() + 2.0f*ImGui::GetStyle().WindowPadding.y;
    ImGuiWindowFlags const flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;
    bool open = BeginMainViewportTopBar(label, height, flags);
    if (padding)
    {
        ImGui::PopStyleVar();
    }
    return open;
}

void osc::DrawNewModelButton(ParentPtr<IMainUIStateAPI> const& api)
{
    if (ImGui::Button(ICON_FA_FILE))
    {
        ActionNewModel(api);
    }
    DrawTooltipIfItemHovered("New Model", "Creates a new OpenSim model in a new tab");
}

void osc::DrawOpenModelButtonWithRecentFilesDropdown(
    std::function<void(std::optional<std::filesystem::path>)> const& onUserClickedOpenOrSelectedFile)
{
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {2.0f, 0.0f});
    if (ImGui::Button(ICON_FA_FOLDER_OPEN))
    {
        onUserClickedOpenOrSelectedFile(std::nullopt);
    }
    DrawTooltipIfItemHovered("Open Model", "Opens an existing osim file in a new tab");
    ImGui::SameLine();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {1.0f, ImGui::GetStyle().FramePadding.y});
    ImGui::Button(ICON_FA_CARET_DOWN);
    DrawTooltipIfItemHovered("Open Recent File", "Opens a recently-opened osim file in a new tab");
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();

    if (ImGui::BeginPopupContextItem("##RecentFilesMenu", ImGuiPopupFlags_MouseButtonLeft))
    {
        auto const recentFiles = App::singleton<RecentFiles>();
        int imguiID = 0;

        for (RecentFile const& rf : *recentFiles)
        {
            ImGui::PushID(imguiID++);
            if (ImGui::Selectable(rf.path.filename().string().c_str()))
            {
                onUserClickedOpenOrSelectedFile(rf.path);
            }
            ImGui::PopID();
        }

        ImGui::EndPopup();
    }
}

void osc::DrawOpenModelButtonWithRecentFilesDropdown(ParentPtr<IMainUIStateAPI> const& api)
{
    DrawOpenModelButtonWithRecentFilesDropdown([&api](auto maybeFile)
    {
        if (maybeFile)
        {
            ActionOpenModel(api, *maybeFile);
        }
        else
        {
            ActionOpenModel(api);
        }
    });
}

void osc::DrawSaveModelButton(
    ParentPtr<IMainUIStateAPI> const& api,
    UndoableModelStatePair& model)
{
    if (ImGui::Button(ICON_FA_SAVE))
    {
        ActionSaveModel(*api, model);
    }
    DrawTooltipIfItemHovered("Save Model", "Saves the model to an osim file");
}

void osc::DrawReloadModelButton(UndoableModelStatePair& model)
{
    if (!HasInputFileName(model.getModel()))
    {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f * ImGui::GetStyle().Alpha);
    }

    if (ImGui::Button(ICON_FA_RECYCLE))
    {
        ActionReloadOsimFromDisk(model, *App::singleton<SceneCache>());
    }

    if (!HasInputFileName(model.getModel()))
    {
        ImGui::PopItemFlag();
        ImGui::PopStyleVar();
    }

    DrawTooltipIfItemHovered("Reload Model", "Reloads the model from its source osim file");
}

void osc::DrawUndoButton(UndoableModelStatePair& model)
{
    int itemFlagsPushed = 0;
    int styleVarsPushed = 0;
    if (!model.canUndo())
    {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ++itemFlagsPushed;
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f * ImGui::GetStyle().Alpha);
        ++styleVarsPushed;
    }

    if (ImGui::Button(ICON_FA_UNDO))
    {
        ActionUndoCurrentlyEditedModel(model);
    }

    PopItemFlags(itemFlagsPushed);
    ImGui::PopStyleVar(styleVarsPushed);

    DrawTooltipIfItemHovered("Undo", "Undo the model to an earlier version");
}

void osc::DrawRedoButton(UndoableModelStatePair& model)
{
    int itemFlagsPushed = 0;
    int styleVarsPushed = 0;
    if (!model.canRedo())
    {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ++itemFlagsPushed;
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f * ImGui::GetStyle().Alpha);
        ++styleVarsPushed;
    }

    if (ImGui::Button(ICON_FA_REDO))
    {
        ActionRedoCurrentlyEditedModel(model);
    }

    PopItemFlags(itemFlagsPushed);
    ImGui::PopStyleVar(styleVarsPushed);

    DrawTooltipIfItemHovered("Redo", "Redo the model to an undone version");
}

void osc::DrawUndoAndRedoButtons(UndoableModelStatePair& model)
{
    DrawUndoButton(model);
    ImGui::SameLine();
    DrawRedoButton(model);
}

void osc::DrawToggleFramesButton(UndoableModelStatePair& model, IconCache& icons)
{
    Icon const& icon = icons.getIcon(IsShowingFrames(model.getModel()) ? "frame_colored" : "frame_bw");
    if (ImageButton("##toggleframes", icon.getTexture(), icon.getDimensions(), icon.getTextureCoordinates()))
    {
        ActionToggleFrames(model);
    }
    DrawTooltipIfItemHovered("Toggle Rendering Frames", "Toggles whether frames (coordinate systems) within the model should be rendered in the 3D scene.");
}

void osc::DrawToggleMarkersButton(UndoableModelStatePair& model, IconCache& icons)
{
    Icon const& icon = icons.getIcon(IsShowingMarkers(model.getModel()) ? "marker_colored" : "marker");
    if (ImageButton("##togglemarkers", icon.getTexture(), icon.getDimensions(), icon.getTextureCoordinates()))
    {
        ActionToggleMarkers(model);
    }
    DrawTooltipIfItemHovered("Toggle Rendering Markers", "Toggles whether markers should be rendered in the 3D scene");
}

void osc::DrawToggleWrapGeometryButton(UndoableModelStatePair& model, IconCache& icons)
{
    Icon const& icon = icons.getIcon(IsShowingWrapGeometry(model.getModel()) ? "wrap_colored" : "wrap");
    if (ImageButton("##togglewrapgeom", icon.getTexture(), icon.getDimensions(), icon.getTextureCoordinates()))
    {
        ActionToggleWrapGeometry(model);
    }
    DrawTooltipIfItemHovered("Toggle Rendering Wrap Geometry", "Toggles whether wrap geometry should be rendered in the 3D scene.\n\nNOTE: This is a model-level property. Individual wrap geometries *within* the model may have their visibility set to 'false', which will cause them to be hidden from the visualizer, even if this is enabled.");
}

void osc::DrawToggleContactGeometryButton(UndoableModelStatePair& model, IconCache& icons)
{
    Icon const& icon = icons.getIcon(IsShowingContactGeometry(model.getModel()) ? "contact_colored" : "contact");
    if (ImageButton("##togglecontactgeom", icon.getTexture(), icon.getDimensions(), icon.getTextureCoordinates()))
    {
        ActionToggleContactGeometry(model);
    }
    DrawTooltipIfItemHovered("Toggle Rendering Contact Geometry", "Toggles whether contact geometry should be rendered in the 3D scene");
}

void osc::DrawAllDecorationToggleButtons(UndoableModelStatePair& model, IconCache& icons)
{
    DrawToggleFramesButton(model, icons);
    ImGui::SameLine();
    DrawToggleMarkersButton(model, icons);
    ImGui::SameLine();
    DrawToggleWrapGeometryButton(model, icons);
    ImGui::SameLine();
    DrawToggleContactGeometryButton(model, icons);
}

void osc::DrawSceneScaleFactorEditorControls(UndoableModelStatePair& model)
{
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.0f, 0.0f});
    ImGui::TextUnformatted(ICON_FA_EXPAND_ALT);
    DrawTooltipIfItemHovered("Scene Scale Factor", "Rescales decorations in the model by this amount. Changing this can be handy when working on extremely small/large models.");
    ImGui::SameLine();

    {
        float scaleFactor = model.getFixupScaleFactor();
        ImGui::SetNextItemWidth(ImGui::CalcTextSize("0.00000").x);
        if (ImGui::InputFloat("##scaleinput", &scaleFactor))
        {
            ActionSetModelSceneScaleFactorTo(model, scaleFactor);
        }
    }
    ImGui::PopStyleVar();

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {2.0f, 0.0f});
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_EXPAND_ARROWS_ALT))
    {
        ActionAutoscaleSceneScaleFactor(model);
    }
    ImGui::PopStyleVar();
    DrawTooltipIfItemHovered("Autoscale Scale Factor", "Try to autoscale the model's scale factor based on the current dimensions of the model");
}

void osc::DrawMeshExportContextMenuContent(
    UndoableModelStatePair const& model,
    OpenSim::Mesh const& mesh)
{
    ImGui::TextDisabled("Format:");
    ImGui::Separator();

    if (ImGui::BeginMenu(".obj"))
    {
        auto const onFrameMenuItemClicked = [&model, &mesh](OpenSim::Frame const& frame)
        {
            ActionReexportMeshOBJWithRespectTo(
                model.getModel(),
                model.getState(),
                mesh,
                frame
            );
        };

        DrawWithRespectToMenuContainingMenuItemPerFrame(model.getModel(), onFrameMenuItemClicked);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(".stl"))
    {
        auto const onFrameMenuItemClicked = [model, &mesh](OpenSim::Frame const& frame)
        {
            ActionReexportMeshSTLWithRespectTo(
                model.getModel(),
                model.getState(),
                mesh,
                frame
            );
        };

        DrawWithRespectToMenuContainingMenuItemPerFrame(model.getModel(), onFrameMenuItemClicked);
        ImGui::EndMenu();
    }
}
