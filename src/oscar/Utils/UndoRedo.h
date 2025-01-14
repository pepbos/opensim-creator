#pragma once

#include <oscar/Utils/CStringView.h>
#include <oscar/Utils/UID.h>

#include <chrono>
#include <concepts>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// undo/redo algorithm support
//
// snapshot-based, rather than command-pattern based. Designed to be reference-counted, and
// allow for implementations that don't need to know what, or how, the data is actually
// stored in memory
namespace osc
{
    template<typename T>
    concept Undoable = std::destructible<T> && std::copy_constructible<T>;

    // internal storage details
    namespace detail
    {
        // a base class for storing undo/redo metadata
        class UndoRedoEntryMetadata {
        protected:
            explicit UndoRedoEntryMetadata(std::string_view message_) :
                m_Message{message_}
            {}
            UndoRedoEntryMetadata(UndoRedoEntryMetadata const&) = default;
            UndoRedoEntryMetadata(UndoRedoEntryMetadata&&) noexcept = default;
            UndoRedoEntryMetadata& operator=(UndoRedoEntryMetadata const&) = default;
            UndoRedoEntryMetadata& operator=(UndoRedoEntryMetadata&&) noexcept = default;
        public:
            virtual ~UndoRedoEntryMetadata() noexcept = default;

            UID id() const { return m_ID; }
            std::chrono::system_clock::time_point time() const { return m_Time; }
            CStringView message() const { return m_Message; }
        private:
            UID m_ID;
            std::chrono::system_clock::time_point m_Time = std::chrono::system_clock::now();
            std::string m_Message;
        };

        // concrete implementation of storage for a complete undo/redo entry (metadata + value)
        template<Undoable T>
        class UndoRedoEntryData final : public UndoRedoEntryMetadata {
        public:
            template<typename... Args>
            UndoRedoEntryData(std::string_view message_, Args&&... args)
                requires std::constructible_from<T, Args&&...> :

                UndoRedoEntryMetadata{std::move(message_)},
                m_Value{std::forward<Args>(args)...}
            {}

            T const& value() const { return m_Value; }

        private:
            T m_Value;
        };
    }

    // type-erased, const, and reference-counted storage for undo/redo entry data
    //
    // can be safely copied, sliced, etc. from the derived class, enabling type-erased
    // implementation code
    class UndoRedoEntryBase {
    protected:
        explicit UndoRedoEntryBase(std::shared_ptr<detail::UndoRedoEntryMetadata const> data_) :
            m_Data{std::move(data_)}
        {}

    public:
        UID id() const { return m_Data->id(); }
        std::chrono::system_clock::time_point time() const { return m_Data->time(); }
        CStringView message() const { return m_Data->message(); }

    protected:
        std::shared_ptr<detail::UndoRedoEntryMetadata const> m_Data;
    };

    // concrete, known-to-hold-type-T version of `UndoRedoEntry`
    template<Undoable T>
    class UndoRedoEntry final : public UndoRedoEntryBase {
    public:
        template<typename... Args>
        UndoRedoEntry(std::string_view message_, Args&&... args)
            requires std::constructible_from<T, Args&&...> :

            UndoRedoEntryBase{std::make_shared<detail::UndoRedoEntryData<T>>(std::move(message_), std::forward<Args>(args)...)}
        {}

        T const& value() const { return static_cast<detail::UndoRedoEntryData<T> const&>(*m_Data).value(); }
    };

    // type-erased base class for undo/redo storage
    //
    // this base class stores undo/redo entries as type-erased pointers, so that the
    // code here, and in other generic downstream classes, doesn't need to know what's
    // actually being stored
    class UndoRedoBase {
    protected:
        explicit UndoRedoBase(UndoRedoEntryBase initialCommit_);
        UndoRedoBase(UndoRedoBase const&);
        UndoRedoBase(UndoRedoBase&&) noexcept;
        UndoRedoBase& operator=(UndoRedoBase const&);
        UndoRedoBase& operator=(UndoRedoBase&&) noexcept;

    public:
        virtual ~UndoRedoBase() noexcept;

        void commitScratch(std::string_view commitMsg);
        UndoRedoEntryBase const& getHead() const;
        UID getHeadID() const;

        size_t getNumUndoEntries() const;
        ptrdiff_t getNumUndoEntriesi() const;
        UndoRedoEntryBase const& getUndoEntry(ptrdiff_t i) const;
        void undoTo(ptrdiff_t nthEntry);
        bool canUndo() const;
        void undo();

        size_t getNumRedoEntries() const;
        ptrdiff_t getNumRedoEntriesi() const;
        UndoRedoEntryBase const& getRedoEntry(ptrdiff_t i) const;
        bool canRedo() const;
        void redoTo(ptrdiff_t nthEntry);
        void redo();

    private:
        virtual UndoRedoEntryBase implCreateCommitFromScratch(std::string_view commitMsg) = 0;
        virtual void implAssignScratchFromCommit(UndoRedoEntryBase const&) = 0;

        std::vector<UndoRedoEntryBase> m_Undo;
        std::vector<UndoRedoEntryBase> m_Redo;
        UndoRedoEntryBase m_Head;
    };

    // concrete class for undo/redo storage
    //
    // - there is a "scratch" space that other code can edit
    // - other code can "commit" the scratch space to storage via `commit(message)`
    // - there is always at least one commit (the "head") in storage, for rollback support
    template<Undoable T>
    class UndoRedo final : public UndoRedoBase {
    public:
        template<typename... Args>
        UndoRedo(Args&&... args)
            requires std::constructible_from<T, Args&&...> :

            UndoRedoBase(UndoRedoEntry<T>{"created document", std::forward<Args>(args)...}),
            m_Scratch{static_cast<UndoRedoEntry<T> const&>(getHead()).value()}
        {}

        T const& getScratch() const { return m_Scratch; }

        T& updScratch() { return m_Scratch; }

        UndoRedoEntry<T> const& getUndoEntry(ptrdiff_t i) const
        {
            return static_cast<UndoRedoEntry<T> const&>(static_cast<UndoRedoBase const&>(*this).getUndoEntry(i));
        }

        UndoRedoEntry<T> const& getRedoEntry(ptrdiff_t i) const
        {
            return static_cast<UndoRedoEntry<T> const&>(static_cast<UndoRedoBase const&>(*this).getRedoEntry(i));
        }

    private:
        virtual UndoRedoEntryBase implCreateCommitFromScratch(std::string_view commitMsg)
        {
            return UndoRedoEntry<T>{std::move(commitMsg), m_Scratch};
        }

        virtual void implAssignScratchFromCommit(UndoRedoEntryBase const& commit)
        {
            m_Scratch = static_cast<UndoRedoEntry<T> const&>(commit).value();
        }

        T m_Scratch;
    };
}
