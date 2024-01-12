#pragma once

#include <OpenSimCreator/Documents/MeshImporter/Document.hpp>
#include <OpenSimCreator/Documents/MeshImporter/MIObject.hpp>

#include <oscar/Maths/Transform.hpp>
#include <oscar/Maths/Vec3.hpp>
#include <oscar/Scene/SceneDecorationFlags.hpp>
#include <oscar/Utils/CStringView.hpp>
#include <oscar/Utils/UID.hpp>

#include <concepts>
#include <string>
#include <optional>
#include <unordered_set>
#include <vector>

namespace osc::mi { class Body; }
namespace osc::mi { class Joint; }

namespace osc::mi
{
    // returns `true` if `body` participates in any joint in the document
    bool IsAChildAttachmentInAnyJoint(Document const&, MIObject const&);

    // returns `true` if a Joint is complete b.s.
    bool IsGarbageJoint(Document const&, Joint const&);

    // returns `true` if a body is indirectly or directly attached to ground
    bool IsBodyAttachedToGround(
        Document const&,
        Body const&,
        std::unordered_set<UID>& previouslyVisitedJoints
    );

    // returns `true` if `joint` is indirectly or directly attached to ground via its parent
    bool IsJointAttachedToGround(
        Document const&,
        Joint const&,
        std::unordered_set<UID>&
    );

    // returns `true` if `body` is attached to ground
    bool IsBodyAttachedToGround(
        Document const&,
        Body const&,
        std::unordered_set<UID>&
    );

    // returns `true` if the document contains issues and populates the output vector with messages
    bool GetIssues(
        Document const&,
        std::vector<std::string>&
    );

    // returns a string representing the subheader of an object
    std::string GetContextMenuSubHeaderText(
        Document const&,
        MIObject const&
    );

    // returns true if the given object ('s ID) is in the "selection group" of the parent
    bool IsInSelectionGroupOf(
        Document const&,
        UID parent,
        UID id
    );

    template<std::invocable<UID> Consumer>
    void ForEachIDInSelectionGroup(
        Document const& doc,
        UID parent,
        Consumer f)
    {
        for (MIObject const& obj : doc.iter())
        {
            UID const id = obj.getID();

            if (IsInSelectionGroupOf(doc, parent, id))
            {
                f(id);
            }
        }
    }

    void SelectAnythingGroupedWith(Document&, UID);

    // returns the ID of the thing the station should attach to when trying to
    // attach to something in the document
    UID GetStationAttachmentParent(Document const&, MIObject const&);

    // points an axis of a given object towards some other object in the document
    void PointAxisTowards(
        Document&,
        UID,
        int axis,
        UID
    );

    // returns decoration flags for an object in the document
    SceneDecorationFlags computeFlags(
        Document const&,
        UID id,
        std::optional<UID> maybeHoverID = std::nullopt
    );
}
