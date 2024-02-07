#pragma once

#include <concepts>
#include <filesystem>
#include <string>
#include <utility>

namespace osc
{
    class ResourcePath final {
    public:
        template<typename... Args>
        ResourcePath(Args&&... args)
            requires std::constructible_from<std::filesystem::path, Args&&...> :
            m_Path{std::forward<Args>(args)...}
        {}

        std::string string() const
        {
            return m_Path.string();
        }

        friend bool operator==(ResourcePath const&, ResourcePath const&) = default;
        friend ResourcePath operator/(ResourcePath const& lhs, ResourcePath const& rhs)
        {
            return ResourcePath{lhs.m_Path / rhs.m_Path};
        }
        friend ResourcePath operator/(ResourcePath const& lhs, std::string_view rhs)
        {
            return ResourcePath{lhs.m_Path / rhs};
        }
    private:
        friend struct std::hash<osc::ResourcePath>;
        std::filesystem::path m_Path;
    };
}

template<>
struct std::hash<osc::ResourcePath> final {
    size_t operator()(osc::ResourcePath const& p) const
    {
        return std::filesystem::hash_value(p.m_Path);
    }
};
