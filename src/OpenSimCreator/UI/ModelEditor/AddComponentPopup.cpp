#include "AddComponentPopup.h"

#include <OpenSimCreator/Documents/Model/UndoableModelActions.h>
#include <OpenSimCreator/Documents/Model/UndoableModelStatePair.h>
#include <OpenSimCreator/UI/Shared/ObjectPropertiesEditor.h>
#include <OpenSimCreator/Utils/OpenSimHelpers.h>

#include <IconsFontAwesome5.h>
#include <OpenSim/Common/Component.h>
#include <OpenSim/Common/ComponentList.h>
#include <OpenSim/Common/ComponentPath.h>
#include <OpenSim/Common/ComponentSocket.h>
#include <OpenSim/Simulation/Model/AbstractPathPoint.h>
#include <OpenSim/Simulation/Model/Model.h>
#include <OpenSim/Simulation/Model/PathActuator.h>
#include <OpenSim/Simulation/Model/PathPoint.h>
#include <OpenSim/Simulation/Model/PhysicalFrame.h>
#include <OpenSim/Simulation/Model/Station.h>
#include <oscar/Graphics/Color.h>
#include <oscar/Platform/App.h>
#include <oscar/UI/ImGuiHelpers.h>
#include <oscar/UI/oscimgui.h>
#include <oscar/UI/Widgets/StandardPopup.h>
#include <oscar/Utils/StringHelpers.h>
#include <SimTKcommon/SmallMatrix.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace OpenSim { class AbstractProperty; }

namespace
{
    struct PathPoint final {

        PathPoint(OpenSim::ComponentPath userChoice_,
            OpenSim::ComponentPath actualFrame_,
            SimTK::Vec3 const& locationInFrame_) :
            userChoice{std::move(userChoice_)},
            actualFrame{std::move(actualFrame_)},
            locationInFrame{locationInFrame_}
        {
        }

        // what the user chose when the clicked in the UI
        OpenSim::ComponentPath userChoice;

        // what the actual frame is that will be attached to
        //
        // (can be different from user choice because the user is permitted to click a station)
        OpenSim::ComponentPath actualFrame;

        // location of the point within the frame
        SimTK::Vec3 locationInFrame;
    };
}

class osc::AddComponentPopup::Impl final : public StandardPopup {
public:
    Impl(
        std::string_view popupName,
        IPopupAPI* api,
        std::shared_ptr<UndoableModelStatePair> uum,
        std::unique_ptr<OpenSim::Component> prototype) :

        StandardPopup{popupName},
        m_Uum{std::move(uum)},
        m_Proto{std::move(prototype)},
        m_PrototypePropertiesEditor{api, m_Uum, [proto = m_Proto]() { return proto.get(); }}
    {}

private:

    std::unique_ptr<OpenSim::Component> tryCreateComponentFromState()
    {
        OpenSim::Model const& model = m_Uum->getModel();

        if (m_Name.empty())
        {
            return nullptr;
        }

        if (m_ProtoSockets.size() != m_SocketConnecteePaths.size())
        {
            return nullptr;
        }

        // clone prototype
        std::unique_ptr<OpenSim::Component> rv = Clone(*m_Proto);

        // set name
        rv->setName(m_Name);

        // assign sockets
        for (size_t i = 0; i < m_ProtoSockets.size(); ++i)
        {
            OpenSim::AbstractSocket const& socket = *m_ProtoSockets[i];
            OpenSim::ComponentPath const& connecteePath = m_SocketConnecteePaths[i];

            OpenSim::Component const* connectee = FindComponent(model, connecteePath);

            if (!connectee)
            {
                return nullptr;  // invalid connectee slipped through
            }

            rv->updSocket(socket.getName()).connect(*connectee);
        }

        // assign path points (if applicable)
        if (auto* pa = dynamic_cast<OpenSim::PathActuator*>(rv.get()))
        {
            if (m_PathPoints.size() < 2)
            {
                return nullptr;
            }

            for (size_t i = 0; i < m_PathPoints.size(); ++i)
            {
                auto const& pp = m_PathPoints[i];

                if (IsEmpty(pp.actualFrame))
                {
                    return nullptr;  // invalid path slipped through
                }

                auto const* pof = FindComponent<OpenSim::PhysicalFrame>(model, pp.actualFrame);
                if (!pof)
                {
                    return nullptr;  // invalid path slipped through
                }

                std::stringstream ppName;
                ppName << pa->getName() << "-P" << (i+1);

                pa->addNewPathPoint(ppName.str(), *pof, pp.locationInFrame);
            }
        }

        return rv;
    }

    bool isAbleToAddComponentFromCurrentState() const
    {
        OpenSim::Model const& model = m_Uum->getModel();

        bool hasName = !m_Name.empty();
        bool allSocketsAssigned = std::all_of(
            m_SocketConnecteePaths.begin(),
            m_SocketConnecteePaths.end(),
            [&model](OpenSim::ComponentPath const& cp)
            {
                return ContainsComponent(model, cp);
            }
        );
        bool hasEnoughPathPoints =
            dynamic_cast<OpenSim::PathActuator const*>(m_Proto.get()) == nullptr ||
            m_PathPoints.size() >= 2;

        return hasName && allSocketsAssigned && hasEnoughPathPoints;
    }

    void drawNameEditor()
    {
        ImGui::Columns(2);

        ImGui::TextUnformatted("name");
        ImGui::SameLine();
        DrawHelpMarker("Name the newly-added component will have after being added into the model. Note: this is used to derive the name of subcomponents (e.g. path points)");
        ImGui::NextColumn();

        InputString("##componentname", m_Name);
        App::upd().addFrameAnnotation("AddComponentPopup::ComponentNameInput", GetItemRect());

        ImGui::NextColumn();

        ImGui::Columns();
    }

    void drawPropertyEditors()
    {
        ImGui::TextUnformatted("Properties");
        ImGui::SameLine();
        DrawHelpMarker("These are properties of the OpenSim::Component being added. Their datatypes, default values, and help text are defined in the source code (see OpenSim_DECLARE_PROPERTY in OpenSim's C++ source code, if you want the details). Their default values are typically sane enough to let you add the component directly into your model.");
        ImGui::Separator();

        ImGui::Dummy({0.0f, 3.0f});

        auto maybeUpdater = m_PrototypePropertiesEditor.onDraw();
        if (maybeUpdater)
        {
            OpenSim::AbstractProperty* prop = FindPropertyMut(*m_Proto, maybeUpdater->getPropertyName());
            if (prop)
            {
                maybeUpdater->apply(*prop);
            }
        }
    }

    void drawSocketEditors()
    {
        if (m_ProtoSockets.empty())
        {
            return;
        }

        ImGui::TextUnformatted("Socket assignments (required)");
        ImGui::SameLine();
        DrawHelpMarker("The OpenSim::Component being added has `socket`s that connect to other components in the model. You must specify what these sockets should be connected to; otherwise, the component cannot be added to the model.\n\nIn OpenSim, a Socket formalizes the dependency between a Component and another object (typically another Component) without owning that object. While Components can be composites (of multiple components) they often depend on unrelated objects/components that are defined and owned elsewhere. The object that satisfies the requirements of the Socket we term the 'connectee'. When a Socket is satisfied by a connectee we have a successful 'connection' or is said to be connected.");
        ImGui::Separator();

        ImGui::Dummy({0.0f, 1.0f});

        // for each socket in the prototype (cached), check if the user has chosen a
        // connectee for it yet and provide a UI for selecting them
        for (size_t i = 0; i < m_ProtoSockets.size(); ++i)
        {
            drawIthSocketEditor(i);
            ImGui::Dummy({0.0f, 0.5f*ImGui::GetTextLineHeight()});
        }
    }

    void drawIthSocketEditor(size_t i)
    {
        OpenSim::AbstractSocket const& socket = *m_ProtoSockets[i];
        OpenSim::ComponentPath& connectee = m_SocketConnecteePaths[i];

        ImGui::Columns(2);

        ImGui::TextUnformatted(socket.getName().c_str());
        ImGui::SameLine();
        DrawHelpMarker(m_Proto->getPropertyByName("socket_" + socket.getName()).getComment());
        ImGui::TextDisabled("%s", socket.getConnecteeTypeName().c_str());
        ImGui::NextColumn();

        // rhs: search and connectee choices
        ImGui::PushID(static_cast<int>(i));
        ImGui::TextUnformatted(ICON_FA_SEARCH);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        InputString("##search", m_SocketSearchStrings[i]);
        ImGui::BeginChild("##pfselector", {ImGui::GetContentRegionAvail().x, 128.0f});

        // iterate through potential connectees in model and print connect-able options
        int innerID = 0;
        for (OpenSim::Component const& c : m_Uum->getModel().getComponentList())
        {
            if (!IsAbleToConnectTo(socket, c)) {
                continue;  // can't connect to it
            }

            if (dynamic_cast<OpenSim::Station const*>(&c) && IsChildOfA<OpenSim::Muscle>(c)) {
                continue;  // it's a muscle point: don't present it (noisy)
            }

            if (!ContainsCaseInsensitive(c.getName(), m_SocketSearchStrings[i])) {
                continue;  // not part of the user-enacted search set
            }

            OpenSim::ComponentPath const absPath = GetAbsolutePath(c);
            bool selected = absPath == connectee;

            ImGui::PushID(innerID++);
            if (ImGui::Selectable(c.getName().c_str(), selected))
            {
                connectee = absPath;
            }

            Rect const selectableRect = GetItemRect();
            DrawTooltipIfItemHovered(absPath.toString());

            ImGui::PopID();

            if (selected)
            {
                App::upd().addFrameAnnotation(c.toString(), selectableRect);
            }
        }

        ImGui::EndChild();
        ImGui::PopID();
        ImGui::NextColumn();
        ImGui::Columns();
    }

    void drawPathPointEditorChoices()
    {
        OpenSim::Model const& model = m_Uum->getModel();

        // show list of choices
        ImGui::BeginChild("##pf_ppchoices", {ImGui::GetContentRegionAvail().x, 128.0f});

        // choices
        for (OpenSim::Component const& c : model.getComponentList())
        {
            auto const isSameUserChoiceAsComponent = [&c](PathPoint const& p)
            {
                return p.userChoice == GetAbsolutePath(c);
            };
            if (std::any_of(m_PathPoints.begin(), m_PathPoints.end(), isSameUserChoiceAsComponent))
            {
                continue;  // already selected
            }

            OpenSim::Component const* userChoice = nullptr;
            OpenSim::PhysicalFrame const* actualFrame = nullptr;
            SimTK::Vec3 locationInFrame = {0.0, 0.0, 0.0};

            // careful here: the order matters
            //
            // various OpenSim classes compose some of these. E.g. subclasses of
            // AbstractPathPoint *also* contain a station object, but named with a
            // plain name
            if (auto const* pof = dynamic_cast<OpenSim::PhysicalFrame const*>(&c))
            {
                userChoice = pof;
                actualFrame = pof;
            }
            else if (auto const* pp = dynamic_cast<OpenSim::PathPoint const*>(&c))
            {
                userChoice = pp;
                actualFrame = &pp->getParentFrame();
                locationInFrame = pp->get_location();
            }
            else if (auto const* app = dynamic_cast<OpenSim::AbstractPathPoint const*>(&c))
            {
                userChoice = app;
                actualFrame = &app->getParentFrame();
            }
            else if (auto const* station = dynamic_cast<OpenSim::Station const*>(&c))
            {
                // check name because it might be a child of one of the above and we
                // don't want to double-count it
                if (station->getName() != "station")
                {
                    userChoice = station;
                    actualFrame = &station->getParentFrame();
                    locationInFrame = station->get_location();
                }
            }

            if (!userChoice || !actualFrame)
            {
                continue;  // can't attach a point to it
            }

            if (!ContainsCaseInsensitive(c.getName(), m_PathSearchString))
            {
                continue;  // search failed
            }

            if (ImGui::Selectable(c.getName().c_str()))
            {
                m_PathPoints.emplace_back(
                    GetAbsolutePath(*userChoice),
                    GetAbsolutePath(*actualFrame),
                    locationInFrame
                );
            }
            DrawTooltipIfItemHovered(c.getName(), (GetAbsolutePathString(c) + " " + c.getConcreteClassName()));
        }

        ImGui::EndChild();
    }

    void drawPathPointEditorAlreadyChosenPoints()
    {
        OpenSim::Model const& model = m_Uum->getModel();

        ImGui::BeginChild("##pf_pathpoints", {ImGui::GetContentRegionAvail().x, 128.0f});

        std::optional<ptrdiff_t> maybeIndexToErase;
        for (ptrdiff_t i = 0; i < std::ssize(m_PathPoints); ++i)
        {
            PushID(i);

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0.0f, 0.0f});

            if (ImGui::Button(ICON_FA_TRASH))
            {
                maybeIndexToErase = i;
            }

            ImGui::SameLine();

            if (i <= 0)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button(ICON_FA_ARROW_UP) && i > 0)
            {
                std::swap(m_PathPoints[i], m_PathPoints[i-1]);
            }
            if (i <= 0)
            {
                ImGui::EndDisabled();
            }

            ImGui::SameLine();

            if (i >= std::ssize(m_PathPoints) - 1)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button(ICON_FA_ARROW_DOWN) && i < std::ssize(m_PathPoints) - 1)
            {
                std::swap(m_PathPoints[i], m_PathPoints[i+1]);
            }
            if (i >= std::ssize(m_PathPoints) - 1)
            {
                ImGui::EndDisabled();
            }

            ImGui::PopStyleVar();
            ImGui::SameLine();

            ImGui::Text("%s", m_PathPoints[i].userChoice.getComponentName().c_str());
            if (ImGui::IsItemHovered())
            {
                if (OpenSim::Component const* c = FindComponent(model, m_PathPoints[i].userChoice))
                {
                    DrawTooltip(c->getName(), GetAbsolutePathString(*c));
                }
            }

            PopID();
        }

        if (maybeIndexToErase)
        {
            m_PathPoints.erase(m_PathPoints.begin() + *maybeIndexToErase);
        }

        ImGui::EndChild();
    }

    void drawPathPointEditor()
    {
        auto* protoAsPA = dynamic_cast<OpenSim::PathActuator*>(m_Proto.get());
        if (!protoAsPA)
        {
            return;  // not a path actuator
        }

        // header
        ImGui::TextUnformatted("Path Points (at least 2 required)");
        ImGui::SameLine();
        DrawHelpMarker("The Component being added is (effectively) a line that connects physical frames (e.g. bodies) in the model. For example, an OpenSim::Muscle can be described as an actuator that connects bodies in the model together. You **must** specify at least two physical frames on the line in order to add a PathActuator component.\n\nDetails: in OpenSim, some `Components` are `PathActuator`s. All `Muscle`s are defined as `PathActuator`s. A `PathActuator` is an `Actuator` that actuates along a path. Therefore, a `Model` containing a `PathActuator` with zero or one points would be invalid. This is why it is required that you specify at least two points");
        ImGui::Separator();

        InputString(ICON_FA_SEARCH " search", m_PathSearchString);

        ImGui::Columns(2);
        int imguiID = 0;

        ImGui::PushID(imguiID++);
        drawPathPointEditorChoices();
        ImGui::PopID();
        ImGui::NextColumn();

        ImGui::PushID(imguiID++);
        drawPathPointEditorAlreadyChosenPoints();
        ImGui::PopID();
        ImGui::NextColumn();

        ImGui::Columns();
    }

    void drawBottomButtons()
    {
        if (ImGui::Button("cancel"))
        {
            requestClose();
        }

        if (!isAbleToAddComponentFromCurrentState())
        {
            return;  // can't add anything yet
        }

        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_PLUS " add"))
        {
            std::unique_ptr<OpenSim::Component> rv = tryCreateComponentFromState();
            if (rv)
            {
                if (ActionAddComponentToModel(*m_Uum, std::move(rv), m_CurrentErrors))
                {
                    requestClose();
                }
            }
        }
    }

    void drawAnyErrorMessages()
    {
        if (!m_CurrentErrors.empty())
        {
            PushStyleColor(ImGuiCol_Text, Color::red());
            ImGui::Dummy({0.0f, 2.0f});
            ImGui::TextWrapped("Error adding component to model: %s", m_CurrentErrors.c_str());
            ImGui::Dummy({0.0f, 2.0f});
            PopStyleColor();
        }
    }

    void implDrawContent() final
    {
        drawNameEditor();

        drawPropertyEditors();

        ImGui::Dummy({0.0f, 3.0f});

        drawSocketEditors();

        ImGui::Dummy({0.0f, 1.0f});

        drawPathPointEditor();

        drawAnyErrorMessages();

        ImGui::Dummy({0.0f, 1.0f});

        drawBottomButtons();
    }

    // the model that the component should be added to
    std::shared_ptr<UndoableModelStatePair> m_Uum;

    // a prototypical version of the component being added
    std::shared_ptr<OpenSim::Component> m_Proto;  // (may be shared with editor popups etc)

    // cached sequence of OpenSim::PhysicalFrame sockets in the prototype
    std::vector<OpenSim::AbstractSocket const*> m_ProtoSockets{GetAllSockets(*m_Proto)};

    // user-assigned name for the to-be-added component
    std::string m_Name{m_Proto->getConcreteClassName()};

    // a property editor for the prototype's properties
    ObjectPropertiesEditor m_PrototypePropertiesEditor;

    // user-enacted search strings for each socket input (used to filter each list)
    std::vector<std::string> m_SocketSearchStrings{m_ProtoSockets.size()};

    // absolute paths to user-selected connectees of the prototype's sockets
    std::vector<OpenSim::ComponentPath> m_SocketConnecteePaths{m_ProtoSockets.size()};

    // absolute paths to user-selected physical frames that should be used as path points
    std::vector<PathPoint> m_PathPoints;

    // search string that user edits to search through possible path point locations
    std::string m_PathSearchString;

    // storage for any addition errors
    std::string m_CurrentErrors;
};


// public API

osc::AddComponentPopup::AddComponentPopup(
    std::string_view popupName,
    IPopupAPI* api,
    std::shared_ptr<UndoableModelStatePair> uum,
    std::unique_ptr<OpenSim::Component> prototype) :

    m_Impl{std::make_unique<Impl>(popupName, api, std::move(uum), std::move(prototype))}
{
}

osc::AddComponentPopup::AddComponentPopup(AddComponentPopup&&) noexcept = default;
osc::AddComponentPopup& osc::AddComponentPopup::operator=(AddComponentPopup&&) noexcept = default;
osc::AddComponentPopup::~AddComponentPopup() noexcept = default;

bool osc::AddComponentPopup::implIsOpen() const
{
    return m_Impl->isOpen();
}

void osc::AddComponentPopup::implOpen()
{
    m_Impl->open();
}

void osc::AddComponentPopup::implClose()
{
    m_Impl->close();
}

bool osc::AddComponentPopup::implBeginPopup()
{
    return m_Impl->beginPopup();
}

void osc::AddComponentPopup::implOnDraw()
{
    m_Impl->onDraw();
}

void osc::AddComponentPopup::implEndPopup()
{
    m_Impl->endPopup();
}

