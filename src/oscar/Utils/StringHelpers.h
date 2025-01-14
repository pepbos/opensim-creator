#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace osc
{
    // returns true if `sv` contains `substr`
    bool Contains(std::string_view sv, std::string_view substr);

    // returns true if `sv` contains `c`
    bool Contains(std::string_view sv, std::string_view::value_type c);

    // returns true if `sv` constains `substr` (case-insensitive)
    bool ContainsCaseInsensitive(std::string_view sv, std::string_view substr);

    // returns true if `b` is lexographically greater than `a`, ignoring case
    bool IsStringCaseInsensitiveGreaterThan(std::string_view a, std::string_view b);

    // returns true if `a` is equal to `b` (case-insensitive)
    bool IsEqualCaseInsensitive(std::string_view a, std::string_view b);

    // returns true if:
    //
    // - `sv` is nonempty
    // - the first character of `sv` is in the set [_a-zA-Z]
    // - the remaining characters of `sv` are in the set [_a-zA-Z0-9]
    // - (sorry UTF8ers)
    //
    // (i.e. it would be a valid identifier in, say, a scripting language or tree)
    bool IsValidIdentifier(std::string_view sv);

    // returns a substring of `sv` without leading/trailing whitespace
    std::string_view TrimLeadingAndTrailingWhitespace(std::string_view sv);

    // (tries to) convert `sv` to a floating point number
    //
    // - strips leading and trailing whitespace
    //
    // - parses the remaining characters as a locale-dependent floating point
    //   number, internally using something like std::strtof (which depends
    //   on C locale - careful)
    //
    // returns the resulting float if sucessful, or std::nullopt if it fails
    //
    // the reason this function exists is because, at time of writing, C++'s
    // <charconv> `std::from_chars` function isn't implemented in Mac OSX
    // or linux. When they are, feel free to nuke this from orbit.
    //
    // see the unittest suite for some of the more unusual things to consider
    std::optional<float> FromCharsStripWhitespace(std::string_view);

    // returns a string that *may* be truncated with ellipsis (...) if the length
    // of the input character sequence exceeds the given maximum length
    std::string Ellipsis(std::string_view, size_t maxLen);

    // returns the end of the string between the last occurance of the delimiter and
    // the end of the string, or the input string if the delimiter does not occur within
    // the string.
    std::string_view SubstringAfterLast(std::string_view, std::string_view::value_type delimiter);

    // converts the given byte into a 2-length hex character representation
    //
    // e.g. 0x00 --> ('0', '0')
    //      0xf0 --> ('f', '0')
    //      0x02 --> ('0', '2')
    std::pair<char, char> ToHexChars(uint8_t);
    std::optional<uint8_t> TryParseHexCharsAsByte(char, char);
}
