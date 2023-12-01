#include "GeometryPathPropertyEditorPopup.hpp"

#include <OpenSimCreator/Documents/Model/UndoableModelStatePair.hpp>
#include <OpenSimCreator/Utils/OpenSimHelpers.hpp>

#include <IconsFontAwesome5.h>
#include <imgui.h>
#include <OpenSim/Simulation/Model/AbstractPathPoint.h>
#include <OpenSim/Simulation/Model/GeometryPath.h>
#include <OpenSim/Simulation/Model/Model.h>
#include <OpenSim/Simulation/Model/PathPoint.h>
#include <oscar/Bindings/ImGuiHelpers.hpp>
#include <oscar/Platform/Log.hpp>
#include <oscar/UI/Widgets/StandardPopup.hpp>
#include <oscar/Utils/CStringView.hpp>

#include <array>
#include <functional>
#include <memory>
#include <string_view>
#include <utility>

namespace
{
    constexpr auto c_LocationInputIDs = std::to_array<osc::CStringView>({ "##xinput", "##yinput", "##zinput" });
    static_assert(c_LocationInputIDs.size() == 3);

    OpenSim::GeometryPath InitGeometryPathFromPropOrDefault(std::function<OpenSim::ObjectProperty<OpenSim::GeometryPath> const* ()> const& accessor)
    {
        OpenSim::ObjectProperty<OpenSim::GeometryPath> const* maybeGeomPath = accessor();
        if (!maybeGeomPath)
        {
            return OpenSim::GeometryPath{};  // default it
        }
        else if (!maybeGeomPath->empty())
        {
            return maybeGeomPath->getValue(0);
        }
        else
        {
            return OpenSim::GeometryPath{};  // ignore lists of geometry paths (too complicated)
        }
    }

    struct RequestedAction final {
        enum class Type {
            MoveUp,
            MoveDown,
            Delete,
            None,
        };

        RequestedAction() = default;

        RequestedAction(
            Type type_,
            ptrdiff_t pathPointIndex_) :

            type{type_},
            pathPointIndex{pathPointIndex_}
        {
        }

        void reset()
        {
            *this = {};
        }

        Type type = Type::None;
        ptrdiff_t pathPointIndex = -1;
    };

    void ActionMovePathPointUp(OpenSim::PathPointSet& pps, ptrdiff_t i)
    {
        if (1 <= i && i < osc::ssize(pps))
        {
            auto tmp = osc::Clone(osc::At(pps, i));
            osc::Assign(pps, i, osc::At(pps, i-1));
            osc::Assign(pps, i-1, std::move(tmp));
        }
    }

    void ActionMovePathPointDown(OpenSim::PathPointSet& pps, ptrdiff_t i)
    {
        if (0 <= i && i < osc::ssize(pps)-1)
        {
            auto tmp = osc::Clone(osc::At(pps, i));
            osc::Assign(pps, i, osc::At(pps, i+1));
            osc::Assign(pps, i+1, std::move(tmp));
        }
    }

    void ActionDeletePathPoint(OpenSim::PathPointSet& pps, ptrdiff_t i)
    {
        if (0 <= i && i < osc::ssize(pps))
        {
            osc::EraseAt(pps, i);
        }
    }

    void ActionSetPathPointFramePath(
        OpenSim::PathPointSet& pps,
        ptrdiff_t i,
        std::string const& frameAbsPath)
    {
        osc::At(pps, i).updSocket("parent_frame").setConnecteePath(frameAbsPath);
    }

    void ActionAddNewPathPoint(OpenSim::PathPointSet& pps)
    {
        std::string const frame = osc::empty(pps) ?
            "/ground" :
            osc::At(pps, osc::size(pps)-1).getSocket("parent_frame").getConnecteePath();

        auto pp = std::make_unique<OpenSim::PathPoint>();
        pp->updSocket("parent_frame").setConnecteePath(frame);

        osc::Append(pps, std::move(pp));
    }

    std::function<void(OpenSim::AbstractProperty&)> MakeGeometryPathPropertyOverwriter(
        OpenSim::GeometryPath const& editedPath)
    {
        return [editedPath](OpenSim::AbstractProperty& prop)
        {
            if (auto* gpProp = dynamic_cast<OpenSim::ObjectProperty<OpenSim::GeometryPath>*>(&prop))
            {
                if (!gpProp->empty())
                {
                    gpProp->updValue() = editedPath;  // just overwrite it
                }
            }
        };
    }

    osc::ObjectPropertyEdit MakeObjectPropertyEdit(
        OpenSim::ObjectProperty<OpenSim::GeometryPath> const& prop,
        OpenSim::GeometryPath const& editedPath)
    {
        return {prop, MakeGeometryPathPropertyOverwriter(editedPath)};
    }
}

class osc::GeometryPathPropertyEditorPopup::Impl final : public osc::StandardPopup {
public:
    Impl(
        std::string_view popupName_,
        std::shared_ptr<UndoableModelStatePair const> targetModel_,
        std::function<OpenSim::ObjectProperty<OpenSim::GeometryPath> const*()> accessor_,
        std::function<void(ObjectPropertyEdit)> onEditCallback_) :

        StandardPopup{popupName_, {768.0f, 0.0f}, ImGuiWindowFlags_AlwaysAutoResize},
        m_TargetModel{std::move(targetModel_)},
        m_Accessor{std::move(accessor_)},
        m_OnEditCallback{std::move(onEditCallback_)},
        m_EditedGeometryPath{InitGeometryPathFromPropOrDefault(m_Accessor)}
    {
    }
private:
    void implDrawContent() final
    {
        if (m_Accessor() == nullptr)
        {
            // edge-case: the geometry path that this popup is editing no longer
            // exists (e.g. because a muscle was deleted or similar), so it should
            // announce the problem and close itself
            ImGui::Text("The GeometryPath no longer exists - closing this popup");
            requestClose();
            return;
        }
        // else: the geometry path exists, but this UI should edit the cached
        // `m_EditedGeometryPath`, which is independent of the original data
        // and the target model (so that edits can be applied transactionally)

        ImGui::Text("Path Points:");
        ImGui::Separator();
        drawPathPointEditorTable();
        ImGui::Separator();
        drawAddPathPointButton();
        ImGui::NewLine();
        drawBottomButtons();
    }

    void drawPathPointEditorTable()
    {
        OpenSim::PathPointSet& pps = m_EditedGeometryPath.updPathPointSet();

        if (ImGui::BeginTable("##GeometryPathEditorTable", 6))
        {
            ImGui::TableSetupColumn("Actions");
            ImGui::TableSetupColumn("Type");
            ImGui::TableSetupColumn("X");
            ImGui::TableSetupColumn("Y");
            ImGui::TableSetupColumn("Z");
            ImGui::TableSetupColumn("Frame");
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();

            for (ptrdiff_t i = 0; i < osc::ssize(pps); ++i)
            {
                osc::PushID(i);
                drawIthPathPointTableRow(pps, i);
                osc::PopID();
            }

            ImGui::EndTable();
        }

        // perform any actions after rendering the table: in case the action would
        // in some way screw with rendering (e.g. deleting a point midway
        // through rendering a row is probably a bad idea)
        tryExecuteRequestedAction(pps);
    }

    void drawAddPathPointButton()
    {
        if (ImGui::Button(ICON_FA_PLUS_CIRCLE " Add Point"))
        {
            ActionAddNewPathPoint(m_EditedGeometryPath.updPathPointSet());
        }
    }

    void drawIthPathPointTableRow(OpenSim::PathPointSet& pps, ptrdiff_t i)
    {
        int column = 0;

        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(column++);
        drawIthPathPointActionsCell(pps, i);

        ImGui::TableSetColumnIndex(column++);
        drawIthPathPointTypeCell(pps, i);

        tryDrawIthPathPointLocationEditorCells(pps, i, column);

        ImGui::TableSetColumnIndex(column++);
        drawIthPathPointFrameCell(pps, i);
    }

    void drawIthPathPointActionsCell(OpenSim::PathPointSet& pps, ptrdiff_t i)
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {2.0f, 0.0f});

        if (i <= 0)
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::SmallButton(ICON_FA_ARROW_UP))
        {
            m_RequestedAction = RequestedAction{RequestedAction::Type::MoveUp, i};
        }
        if (i <= 0)
        {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();

        if (i+1 >= osc::ssize(pps))
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::SmallButton(ICON_FA_ARROW_DOWN))
        {
            m_RequestedAction = RequestedAction{RequestedAction::Type::MoveDown, i};
        }
        if (i+1 >= osc::ssize(pps))
        {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Text, {0.7f, 0.0f, 0.0f, 1.0f});
        if (ImGui::SmallButton(ICON_FA_TIMES))
        {
            m_RequestedAction = RequestedAction{RequestedAction::Type::Delete, i};
        }
        ImGui::PopStyleColor();

        ImGui::PopStyleVar();
    }

    void drawIthPathPointTypeCell(OpenSim::PathPointSet const& pps, ptrdiff_t i)
    {
        ImGui::TextDisabled("%s", At(pps, i).getConcreteClassName().c_str());
    }

    // try, because the path point type might not actually have a set location
    //
    // (e.g. `MovingPathPoint`s)
    void tryDrawIthPathPointLocationEditorCells(OpenSim::PathPointSet& pps, ptrdiff_t i, int& column)
    {
        OpenSim::AbstractPathPoint& app = At(pps, i);

        if (auto* const pp = dynamic_cast<OpenSim::PathPoint*>(&app))
        {
            float const inputWidth = ImGui::CalcTextSize("0.00000").x;

            SimTK::Vec3& location = pp->upd_location();

            static_assert(c_LocationInputIDs.size() == 3);
            for (size_t dim = 0; dim < c_LocationInputIDs.size(); ++dim)
            {
                auto v = static_cast<float>(location[static_cast<int>(dim)]);

                ImGui::TableSetColumnIndex(column++);
                ImGui::SetNextItemWidth(inputWidth);
                if (ImGui::InputFloat(c_LocationInputIDs[dim].c_str(), &v))
                {
                    location[static_cast<int>(dim)] = static_cast<double>(v);
                }
            }
        }
        else
        {
            // it's some other kind of path point, with no editable X, Y, or Z
            ImGui::TableSetColumnIndex(column++);
            ImGui::TableSetColumnIndex(column++);
            ImGui::TableSetColumnIndex(column++);
        }
    }

    void drawIthPathPointFrameCell(OpenSim::PathPointSet& pps, ptrdiff_t i)
    {
        float const width = ImGui::CalcTextSize("/bodyset/a_typical_body_name").x;

        std::string const& label = At(pps, i).getSocket("parent_frame").getConnecteePath();

        ImGui::SetNextItemWidth(width);
        if (ImGui::BeginCombo("##framesel", label.c_str()))
        {
            for (OpenSim::Frame const& frame : m_TargetModel->getModel().getComponentList<OpenSim::Frame>())
            {
                std::string const absPath = frame.getAbsolutePathString();
                if (ImGui::Selectable(absPath.c_str()))
                {
                    ActionSetPathPointFramePath(pps, i, absPath);
                }
            }
            ImGui::EndCombo();
        }
    }

    void drawBottomButtons()
    {
        if (ImGui::Button("cancel"))
        {
            requestClose();
        }

        ImGui::SameLine();

        if (ImGui::Button("save"))
        {
            OpenSim::ObjectProperty<OpenSim::GeometryPath> const* maybeProp = m_Accessor();
            if (maybeProp)
            {
                m_OnEditCallback(MakeObjectPropertyEdit(*maybeProp, m_EditedGeometryPath));
            }
            else
            {
                log::error("cannot update geometry path: it no longer exists");
            }
            requestClose();
        }
    }

    void tryExecuteRequestedAction(OpenSim::PathPointSet& pps)
    {
        if (!(0 <= m_RequestedAction.pathPointIndex && m_RequestedAction.pathPointIndex < osc::ssize(pps)))
        {
            // edge-case: if the index is out of range, ignore the action
            m_RequestedAction.reset();
            return;
        }

        switch (m_RequestedAction.type)
        {
        case RequestedAction::Type::MoveUp:
            ActionMovePathPointUp(pps, m_RequestedAction.pathPointIndex);
            break;
        case RequestedAction::Type::MoveDown:
            ActionMovePathPointDown(pps, m_RequestedAction.pathPointIndex);
            break;
        case RequestedAction::Type::Delete:
            ActionDeletePathPoint(pps, m_RequestedAction.pathPointIndex);
            break;
        default:
            break;  // (unhandled/do nothing)
        }

        m_RequestedAction.reset();  // action handled: reset
    }

    std::shared_ptr<UndoableModelStatePair const> m_TargetModel;
    std::function<OpenSim::ObjectProperty<OpenSim::GeometryPath> const*()> m_Accessor;
    std::function<void(ObjectPropertyEdit)> m_OnEditCallback;

    OpenSim::GeometryPath m_EditedGeometryPath;
    RequestedAction m_RequestedAction;
};


// public API (PIMPL)

osc::GeometryPathPropertyEditorPopup::GeometryPathPropertyEditorPopup(
    std::string_view popupName_,
    std::shared_ptr<UndoableModelStatePair const> targetModel_,
    std::function<OpenSim::ObjectProperty<OpenSim::GeometryPath> const*()> accessor_,
    std::function<void(ObjectPropertyEdit)> onEditCallback_) :

    m_Impl{std::make_unique<Impl>(popupName_, std::move(targetModel_), std::move(accessor_), std::move(onEditCallback_))}
{
}
osc::GeometryPathPropertyEditorPopup::GeometryPathPropertyEditorPopup(GeometryPathPropertyEditorPopup&&) noexcept = default;
osc::GeometryPathPropertyEditorPopup& osc::GeometryPathPropertyEditorPopup::operator=(GeometryPathPropertyEditorPopup&&) noexcept = default;
osc::GeometryPathPropertyEditorPopup::~GeometryPathPropertyEditorPopup() noexcept = default;

bool osc::GeometryPathPropertyEditorPopup::implIsOpen() const
{
    return m_Impl->isOpen();
}
void osc::GeometryPathPropertyEditorPopup::implOpen()
{
    m_Impl->open();
}
void osc::GeometryPathPropertyEditorPopup::implClose()
{
    m_Impl->close();
}
bool osc::GeometryPathPropertyEditorPopup::implBeginPopup()
{
    return m_Impl->beginPopup();
}
void osc::GeometryPathPropertyEditorPopup::implOnDraw()
{
    m_Impl->onDraw();
}
void osc::GeometryPathPropertyEditorPopup::implEndPopup()
{
    m_Impl->endPopup();
}