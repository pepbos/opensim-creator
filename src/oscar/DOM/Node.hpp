#pragma once

#include <oscar/DOM/NodePath.hpp>
#include <oscar/DOM/Object.hpp>

#include <oscar/Utils/ClonePtr.hpp>
#include <oscar/Utils/CStringView.hpp>

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace osc { class Variant; }
namespace osc { class PropertyDescriptions; }

namespace osc
{
    class Node : public Object {
    protected:
        Node();
        Node(Node const&);
        Node(Node&&) noexcept;
        Node& operator=(Node const&);
        Node& operator=(Node&&) noexcept;

    public:
        std::unique_ptr<Node> clone() const
        {
            return std::unique_ptr<Node>{static_cast<Node*>(static_cast<Object const&>(*this).clone().release())};
        }

        CStringView getName() const;
        void setName(std::string_view);

        template<
            typename TNode = Node,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode const* getParent() const
        {
            return getParent(Identity<TNode>{});
        }

        template<
            typename TNode = Node,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode* updParent()
        {
            return updParent(Identity<TNode>{});
        }


        size_t getNumChildren() const;

        template<
            typename TNode = Node,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode const* getChild(size_t i) const
        {
            return getChild(i, Identity<TNode>{});
        }

        template<
            typename TNode = Node,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode const* getChild(std::string_view childName) const
        {
            return getChild(childName, Identity<TNode>{});
        }

        template<
            typename TNode = Node,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode* updChild(size_t i)
        {
            return updChild(i, Identity<TNode>{});
        }

        template<
            typename TNode = Node,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode* updChild(std::string_view childName)
        {
            return updChild(childName, Identity<TNode>{});
        }

        template<
            typename TNode,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode& addChild(std::unique_ptr<TNode> p)
        {
            return addChild(std::move(p), Identity<TNode>{});
        }

        template<
            typename TNode,
            typename... Args,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode& emplaceChild(Args&&... args)
        {
            return addChild(std::make_unique<TNode>(std::forward<Args>(args)...));
        }

        bool removeChild(size_t);
        bool removeChild(Node&);
        bool removeChild(std::string_view childName);

        NodePath getAbsolutePath() const;

        template<
            typename TNode = Node,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode const* find(NodePath const& p) const
        {
            return find(p, Identity<TNode>{});
        }

        template<
            typename TNode = Node,
            typename std::enable_if_t<std::is_base_of_v<Node, TNode>, bool> = true
        >
        TNode* findMut(NodePath const& p)
        {
            return dynamic_cast<TNode*>(findMut(p, Identity<TNode>{}));
        }

    private:
        // You might be (rightly) wondering why this implementation goes through the
        // bother of using `Identity<T>` classes to distinguish overloads etc. rather
        // than just specializing a template function.
        //
        // It's because standard C++ doesn't allow template specialization on class
        // member functions. See: https://stackoverflow.com/a/3057522

        template<typename T>
        struct Identity { using type = T; };

        template<typename TDerived>
        TDerived const* getParent(Identity<TDerived>) const
        {
            return dynamic_cast<TDerived const*>(getParent(Identity<Node>{}));
        }
        Node const* getParent(Identity<Node>) const;

        template<typename TDerived>
        TDerived* updParent(Identity<TDerived>)
        {
            return dynamic_cast<TDerived*>(updParent(Identity<Node>{}));
        }
        Node* updParent(Identity<Node>);

        template<typename TDerived>
        TDerived const* getChild(size_t i, Identity<TDerived>) const
        {
            return dynamic_cast<TDerived const*>(getChild(i, Identity<Node>{}));
        }
        Node const* getChild(size_t, Identity<Node>) const;

        template<typename TDerived>
        TDerived const* getChild(std::string_view childName, Identity<TDerived>) const
        {
            return dynamic_cast<TDerived const*>(getChild(childName, Identity<Node>{}));
        }
        Node const* getChild(std::string_view, Identity<Node>) const;

        template<typename TDerived>
        TDerived* updChild(size_t i, Identity<TDerived>)
        {
            return dynamic_cast<TDerived*>(updChild(i, Identity<Node>{}));
        }
        Node* updChild(size_t, Identity<Node>);

        template<typename TDerived>
        TDerived* updChild(std::string_view childName, Identity<TDerived>)
        {
            return dynamic_cast<TDerived*>(updChild(childName, Identity<Node>{}));
        }
        Node* updChild(std::string_view childName, Identity<Node>);

        template<typename TDerived>
        TDerived& addChild(std::unique_ptr<TDerived> p, Identity<TDerived>)
        {
            TDerived& rv = *p;
            addChild(std::move(p), Identity<Node>{});
            return rv;
        }
        Node& addChild(std::unique_ptr<Node>, Identity<Node>);

        template<typename TDerived>
        TDerived const* find(NodePath const& p, Identity<TDerived>) const
        {
            return dynamic_cast<TDerived const*>(find(p, Identity<Node>{}));
        }
        Node const* find(NodePath const&, Identity<Node>) const;

        template<typename TDerived>
        TDerived* findMut(NodePath const& p, Identity<TDerived>)
        {
            return dynamic_cast<TDerived*>(findMut(p, Identity<Node>{}));
        }
        Node* findMut(NodePath const&, Identity<Node>);

        // lifetime
        // lifetimed ptr to parent
        // children
    };
}
