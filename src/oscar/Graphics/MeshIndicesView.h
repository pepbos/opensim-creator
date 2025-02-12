#pragma once

#include <oscar/Utils/Assertions.h>
#include <oscar/Utils/Concepts.h>

#include <cstdint>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <span>
#include <stdexcept>

namespace osc
{
    // a span-like view over mesh indices
    //
    // for perf reasons, runtime mesh indices can be stored in either a 16-bit or 32-bit format
    // the mesh class exposes this fact by returning this view class, which must be checked at
    // runtime by calling code

    class MeshIndicesView final {
    private:
        union U32PtrOrU16Ptr {
            uint16_t const* u16;
            uint32_t const* u32;

            U32PtrOrU16Ptr() : u16{nullptr} {}
            U32PtrOrU16Ptr(uint16_t const* ptr) : u16{ptr} {}
            U32PtrOrU16Ptr(uint32_t const* ptr) : u32{ptr} {}
        };
    public:
        class Iterator final {
        public:
            using difference_type = size_t;
            using value_type = uint32_t;
            using pointer = void;
            using reference = value_type;
            using iterator_category = std::forward_iterator_tag;

            Iterator(U32PtrOrU16Ptr ptr, bool isU32) :
                m_Ptr{ptr},
                m_IsU32{isU32}
            {
            }

            uint32_t operator*() const
            {
                return m_IsU32 ? *m_Ptr.u32 : static_cast<uint32_t>(*m_Ptr.u16);
            }

            friend bool operator==(Iterator const& lhs, Iterator const& rhs)
            {
                return lhs.m_Ptr.u16 == rhs.m_Ptr.u16 && lhs.m_IsU32 == rhs.m_IsU32;
            }

            Iterator& operator++()
            {
                if (m_IsU32) { ++m_Ptr.u32; } else { ++m_Ptr.u16; }
                return *this;
            }
        private:
            U32PtrOrU16Ptr m_Ptr;
            bool m_IsU32;
        };

        MeshIndicesView() :
            m_Ptr{},
            m_Size{0},
            m_IsU32{false}
        {
        }

        MeshIndicesView(uint16_t const* ptr, size_t size) :
            m_Ptr{ptr},
            m_Size{size},
            m_IsU32{false}
        {
        }

        MeshIndicesView(uint32_t const* ptr, size_t size) :
            m_Ptr{ptr},
            m_Size{size},
            m_IsU32{true}
        {
        }

        template<std::ranges::contiguous_range Range>
        MeshIndicesView(Range const& range)
            requires IsAnyOf<typename Range::value_type, uint16_t, uint32_t>

            : MeshIndicesView{std::ranges::data(range), std::ranges::size(range)}
        {
        }

        bool isU16() const
        {
            return !m_IsU32;
        }

        bool isU32() const
        {
            return m_IsU32;
        }

        [[nodiscard]] bool empty() const
        {
            return size() == 0;
        }

        size_t size() const
        {
            return m_Size;
        }

        std::span<uint16_t const> toU16Span() const
        {
            OSC_ASSERT(!m_IsU32);
            return {m_Ptr.u16, m_Size};
        }

        std::span<uint32_t const> toU32Span() const
        {
            OSC_ASSERT(m_IsU32);
            return {m_Ptr.u32, m_Size};
        }

        uint32_t operator[](ptrdiff_t i) const
        {
            return !m_IsU32 ? static_cast<uint32_t>(m_Ptr.u16[i]) : m_Ptr.u32[i];
        }

        uint32_t at(ptrdiff_t i) const
        {
            if (i >= static_cast<ptrdiff_t>(size())) {
                throw std::out_of_range{"attempted to access a MeshIndicesView with an invalid index"};
            }
            return this->operator[](i);
        }

        Iterator begin() const
        {
            return Iterator{m_Ptr, m_IsU32};
        }

        Iterator end() const
        {
            return Iterator{m_IsU32 ? U32PtrOrU16Ptr{m_Ptr.u32 + m_Size} : U32PtrOrU16Ptr{m_Ptr.u16 + m_Size}, m_IsU32};
        }

    private:
        U32PtrOrU16Ptr m_Ptr;
        size_t m_Size;
        bool m_IsU32;
    };
}
