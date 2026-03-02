#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct ToolError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& where, const std::string& message) {
    throw ToolError(where + ": " + message);
}

struct JsonValue {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;

    Kind kind = Kind::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    Array array_value;
    Object object_value;

    static JsonValue make_null() {
        JsonValue v;
        v.kind = Kind::Null;
        return v;
    }

    static JsonValue make_bool(bool b) {
        JsonValue v;
        v.kind = Kind::Bool;
        v.bool_value = b;
        return v;
    }

    static JsonValue make_number(double d) {
        JsonValue v;
        v.kind = Kind::Number;
        v.number_value = d;
        return v;
    }

    static JsonValue make_string(std::string s) {
        JsonValue v;
        v.kind = Kind::String;
        v.string_value = std::move(s);
        return v;
    }

    static JsonValue make_array(Array a) {
        JsonValue v;
        v.kind = Kind::Array;
        v.array_value = std::move(a);
        return v;
    }

    static JsonValue make_object(Object o) {
        JsonValue v;
        v.kind = Kind::Object;
        v.object_value = std::move(o);
        return v;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string text, std::string where)
        : text_(std::move(text)), where_(std::move(where)) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (!eof()) {
            parse_fail("unexpected trailing content");
        }
        return v;
    }

private:
    std::string text_;
    std::string where_;
    std::size_t pos_ = 0;

    [[noreturn]] void parse_fail(const std::string& message) const {
        std::ostringstream oss;
        oss << "invalid JSON at byte " << pos_ << ": " << message;
        fail(where_, oss.str());
    }

    bool eof() const { return pos_ >= text_.size(); }

    char peek() const {
        if (eof()) {
            return '\0';
        }
        return text_[pos_];
    }

    char take() {
        if (eof()) {
            parse_fail("unexpected end of input");
        }
        return text_[pos_++];
    }

    void expect(char ch) {
        if (take() != ch) {
            std::ostringstream oss;
            oss << "expected '" << ch << "'";
            parse_fail(oss.str());
        }
    }

    void skip_ws() {
        while (!eof() && std::isspace(static_cast<unsigned char>(peek()))) {
            ++pos_;
        }
    }

    bool consume(std::string_view literal) {
        if (text_.size() - pos_ < literal.size()) {
            return false;
        }
        if (text_.compare(pos_, literal.size(), literal.data(), literal.size()) == 0) {
            pos_ += literal.size();
            return true;
        }
        return false;
    }

    JsonValue parse_value() {
        skip_ws();
        if (eof()) {
            parse_fail("expected value");
        }

        const char ch = peek();
        if (ch == '{') {
            return parse_object();
        }
        if (ch == '[') {
            return parse_array();
        }
        if (ch == '"') {
            return JsonValue::make_string(parse_string());
        }
        if (ch == 't') {
            if (!consume("true")) {
                parse_fail("invalid token");
            }
            return JsonValue::make_bool(true);
        }
        if (ch == 'f') {
            if (!consume("false")) {
                parse_fail("invalid token");
            }
            return JsonValue::make_bool(false);
        }
        if (ch == 'n') {
            if (!consume("null")) {
                parse_fail("invalid token");
            }
            return JsonValue::make_null();
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
            return JsonValue::make_number(parse_number());
        }

        parse_fail("unexpected character while parsing value");
    }

    JsonValue parse_object() {
        expect('{');
        skip_ws();
        JsonValue::Object obj;
        if (peek() == '}') {
            take();
            return JsonValue::make_object(std::move(obj));
        }

        while (true) {
            skip_ws();
            if (peek() != '"') {
                parse_fail("expected string key");
            }
            std::string key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            JsonValue value = parse_value();

            auto [it, inserted] = obj.emplace(std::move(key), std::move(value));
            (void)it;
            if (!inserted) {
                parse_fail("duplicate object key");
            }

            skip_ws();
            if (peek() == '}') {
                take();
                break;
            }
            expect(',');
            skip_ws();
        }

        return JsonValue::make_object(std::move(obj));
    }

    JsonValue parse_array() {
        expect('[');
        skip_ws();
        JsonValue::Array arr;
        if (peek() == ']') {
            take();
            return JsonValue::make_array(std::move(arr));
        }

        while (true) {
            skip_ws();
            arr.emplace_back(parse_value());
            skip_ws();
            if (peek() == ']') {
                take();
                break;
            }
            expect(',');
            skip_ws();
        }

        return JsonValue::make_array(std::move(arr));
    }

    std::string parse_string() {
        expect('"');
        std::string out;

        while (true) {
            if (eof()) {
                parse_fail("unterminated string");
            }
            char ch = take();
            if (ch == '"') {
                break;
            }
            if (ch == '\\') {
                if (eof()) {
                    parse_fail("unterminated escape");
                }
                const char esc = take();
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        // Minimal \u handling for BMP codepoints.
                        std::uint32_t codepoint = 0;
                        for (int i = 0; i < 4; ++i) {
                            if (eof()) {
                                parse_fail("unterminated \\u escape");
                            }
                            char hx = take();
                            codepoint <<= 4;
                            if (hx >= '0' && hx <= '9') {
                                codepoint |= static_cast<std::uint32_t>(hx - '0');
                            } else if (hx >= 'a' && hx <= 'f') {
                                codepoint |= static_cast<std::uint32_t>(10 + (hx - 'a'));
                            } else if (hx >= 'A' && hx <= 'F') {
                                codepoint |= static_cast<std::uint32_t>(10 + (hx - 'A'));
                            } else {
                                parse_fail("invalid hex in \\u escape");
                            }
                        }
                        if (codepoint <= 0x7F) {
                            out.push_back(static_cast<char>(codepoint));
                        } else if (codepoint <= 0x7FF) {
                            out.push_back(static_cast<char>(0xC0u | ((codepoint >> 6) & 0x1Fu)));
                            out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
                        } else {
                            out.push_back(static_cast<char>(0xE0u | ((codepoint >> 12) & 0x0Fu)));
                            out.push_back(static_cast<char>(0x80u | ((codepoint >> 6) & 0x3Fu)));
                            out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
                        }
                        break;
                    }
                    default:
                        parse_fail("unsupported escape sequence");
                }
            } else {
                out.push_back(ch);
            }
        }

        return out;
    }

    double parse_number() {
        const std::size_t start = pos_;

        if (peek() == '-') {
            ++pos_;
        }

        if (eof()) {
            parse_fail("invalid number");
        }

        if (peek() == '0') {
            ++pos_;
        } else if (std::isdigit(static_cast<unsigned char>(peek()))) {
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        } else {
            parse_fail("invalid number integer part");
        }

        if (!eof() && peek() == '.') {
            ++pos_;
            if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) {
                parse_fail("invalid number fractional part");
            }
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        }

        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            ++pos_;
            if (!eof() && (peek() == '+' || peek() == '-')) {
                ++pos_;
            }
            if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) {
                parse_fail("invalid exponent");
            }
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        }

        const std::string token = text_.substr(start, pos_ - start);
        std::istringstream iss(token);
        double d = 0.0;
        iss >> d;
        if (!iss || !iss.eof()) {
            parse_fail("failed to parse number");
        }
        return d;
    }
};

std::string read_text_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail(path.string(), "failed to open file");
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void write_text_file(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        fail(path.string(), "failed to open output file");
    }
    out << content;
    if (!out) {
        fail(path.string(), "failed to write output file");
    }
}

JsonValue parse_json_file(const fs::path& path) {
    return JsonParser(read_text_file(path), path.string()).parse();
}

const JsonValue& require_kind(const JsonValue& v, JsonValue::Kind expected, const std::string& where, const std::string& field) {
    if (v.kind != expected) {
        fail(where, field + " has unexpected JSON type");
    }
    return v;
}

bool has_key(const JsonValue::Object& obj, std::string_view key) {
    return obj.find(std::string(key)) != obj.end();
}

const JsonValue& require_key(const JsonValue::Object& obj, std::string_view key, const std::string& where) {
    const auto it = obj.find(std::string(key));
    if (it == obj.end()) {
        fail(where, "missing key: " + std::string(key));
    }
    return it->second;
}

int require_int(const JsonValue& v, const std::string& where, const std::string& field) {
    if (v.kind != JsonValue::Kind::Number) {
        fail(where, field + " must be a number");
    }
    const double d = v.number_value;
    const double rounded = std::round(d);
    if (std::abs(d - rounded) > 1e-9) {
        fail(where, field + " must be an integer");
    }
    return static_cast<int>(rounded);
}

std::string require_nonempty_string(const JsonValue& v, const std::string& where, const std::string& field) {
    if (v.kind != JsonValue::Kind::String || v.string_value.empty()) {
        fail(where, field + " must be a non-empty string");
    }
    return v.string_value;
}

std::string join_strings(const std::vector<std::string>& values, const std::string& sep) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) {
            oss << sep;
        }
        oss << values[i];
    }
    return oss.str();
}

std::vector<std::string> sorted_keys(const JsonValue::Object& obj) {
    std::vector<std::string> keys;
    keys.reserve(obj.size());
    for (const auto& [k, _] : obj) {
        (void)_;
        keys.push_back(k);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

bool is_ascii_id(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        const bool ok =
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '.' || c == '-';
        if (!ok) {
            return false;
        }
    }
    return true;
}

bool is_pattern_token(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    const char first = s.front();
    const bool first_ok = (first == '_') || (first >= 'A' && first <= 'Z') || (first >= 'a' && first <= 'z');
    if (!first_ok) {
        return false;
    }
    for (char c : s) {
        const bool ok =
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == ':';
        if (!ok) {
            return false;
        }
    }
    return true;
}

bool is_canonical_preset_pattern(const std::string& s) {
    constexpr std::string_view kPrefix = "axp::l4::preset::";
    if (s.rfind(kPrefix, 0) != 0) {
        return false;
    }
    return is_pattern_token(s);
}

bool parse_u64_literal(const std::string& text, std::uint64_t& out) {
    if (text.empty()) {
        return false;
    }
    if (text.front() == '-') {
        return false;
    }

    int base = 10;
    std::string_view digits = text;
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        base = 16;
        digits = std::string_view(text).substr(2);
        if (digits.empty()) {
            return false;
        }
    }

    std::uint64_t value = 0;
    const char* begin = digits.data();
    const char* end = digits.data() + digits.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value, base);
    if (ec != std::errc() || ptr != end) {
        return false;
    }

    out = value;
    return true;
}

std::string normalize_hash(const fs::path& path, const JsonValue& value, const std::string& field, std::uint64_t* out_u64 = nullptr) {
    const std::string raw = require_nonempty_string(value, path.string(), field);
    std::uint64_t parsed = 0;
    if (!parse_u64_literal(raw, parsed)) {
        fail(path.string(), field + " must parse as u64 hex/integer, got '" + raw + "'");
    }
    std::ostringstream oss;
    oss << "0x" << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << parsed;
    if (out_u64 != nullptr) {
        *out_u64 = parsed;
    }
    return oss.str();
}

constexpr std::array<const char*, 3> kSupportedArches = {"sm89", "sm90", "sm100"};
constexpr std::array<const char*, 2> kSupportedProfiles = {"dev_fast", "proof_full"};

bool is_supported_arch(const std::string& s) {
    return std::find(kSupportedArches.begin(), kSupportedArches.end(), s) != kSupportedArches.end();
}

bool is_supported_profile(const std::string& s) {
    return std::find(kSupportedProfiles.begin(), kSupportedProfiles.end(), s) != kSupportedProfiles.end();
}

int cap_index(const std::string& cap) {
    for (std::size_t i = 0; i < kSupportedArches.size(); ++i) {
        if (cap == kSupportedArches[i]) {
            return static_cast<int>(i);
        }
    }
    return 999;
}

int profile_index(const std::string& profile) {
    for (std::size_t i = 0; i < kSupportedProfiles.size(); ++i) {
        if (profile == kSupportedProfiles[i]) {
            return static_cast<int>(i);
        }
    }
    return 999;
}

std::string cap_to_cpp(const std::string& cap) {
    if (cap == "sm89") return "iro::cap::sm89";
    if (cap == "sm90") return "iro::cap::sm90";
    if (cap == "sm100") return "iro::cap::sm100";
    fail("cap_to_cpp", "unsupported cap: " + cap);
}

std::string cap_to_guard(const std::string& cap) {
    if (cap == "sm89") return "AXP_ENABLE_SM89";
    if (cap == "sm90") return "AXP_ENABLE_SM90";
    if (cap == "sm100") return "AXP_ENABLE_SM100";
    fail("cap_to_guard", "unsupported cap: " + cap);
}

std::string profile_to_cpp(const std::string& profile) {
    if (profile == "dev_fast") return "axp::l4::profile::dev_fast";
    if (profile == "proof_full") return "axp::l4::profile::proof_full";
    fail("profile_to_cpp", "unsupported profile: " + profile);
}

std::string base_graph_id(const std::string& kernel_id) {
    constexpr std::string_view dev_suffix = "_dev_fast";
    constexpr std::string_view full_suffix = "_proof_full";
    if (kernel_id.size() > dev_suffix.size() &&
        kernel_id.compare(kernel_id.size() - dev_suffix.size(), dev_suffix.size(), dev_suffix) == 0) {
        return kernel_id.substr(0, kernel_id.size() - dev_suffix.size());
    }
    if (kernel_id.size() > full_suffix.size() &&
        kernel_id.compare(kernel_id.size() - full_suffix.size(), full_suffix.size(), full_suffix) == 0) {
        return kernel_id.substr(0, kernel_id.size() - full_suffix.size());
    }
    return kernel_id;
}

std::string sanitize_identifier(const std::string& value) {
    std::string normalized;
    normalized.reserve(value.size());
    bool prev_us = false;

    for (char c : value) {
        char out = c;
        const bool is_alnum = (c >= '0' && c <= '9') ||
                              (c >= 'A' && c <= 'Z') ||
                              (c >= 'a' && c <= 'z');
        if (!(is_alnum || c == '_')) {
            out = '_';
        }

        if (out == '_') {
            if (prev_us) {
                continue;
            }
            prev_us = true;
        } else {
            prev_us = false;
        }
        normalized.push_back(out);
    }

    while (!normalized.empty() && normalized.front() == '_') {
        normalized.erase(normalized.begin());
    }
    while (!normalized.empty() && normalized.back() == '_') {
        normalized.pop_back();
    }

    if (normalized.empty()) {
        normalized = "graph";
    }
    if (!normalized.empty() && normalized.front() >= '0' && normalized.front() <= '9') {
        normalized = "g_" + normalized;
    }
    return normalized;
}

std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 0x20) {
                    oss << "\\u"
                        << std::hex << std::nouppercase << std::setfill('0') << std::setw(4)
                        << static_cast<int>(c)
                        << std::dec;
                } else {
                    oss << static_cast<char>(c);
                }
                break;
        }
    }
    return oss.str();
}

struct KernelRecord {
    std::string id;
    std::string op_family;
    std::string capability;
    std::string profile;
    std::string realization_key;
    std::string graph_hash;
    std::uint64_t graph_hash_u64 = 0;
    std::string pattern;
};

struct ManifestRecord {
    fs::path path;
    std::string arch;
    std::vector<KernelRecord> kernels;
};

struct GraphBinding {
    std::string capability;
    std::string profile;

    bool operator<(const GraphBinding& other) const {
        const int c0 = cap_index(capability);
        const int c1 = cap_index(other.capability);
        if (c0 != c1) return c0 < c1;
        const int p0 = profile_index(profile);
        const int p1 = profile_index(other.profile);
        if (p0 != p1) return p0 < p1;
        if (capability != other.capability) return capability < other.capability;
        return profile < other.profile;
    }
};

struct GraphGroup {
    std::string graph_hash;
    std::uint64_t graph_hash_u64 = 0;
    std::string pattern;
    std::string op_family;
    std::string realization_key;
    std::set<std::string> graph_id_candidates;
    std::set<std::string> capabilities;
    std::set<std::string> profiles;
    std::set<GraphBinding> bindings;
    std::string graph_id;
};

struct RegistryGraph {
    std::string graph_id;
    std::string graph_hash;
    std::uint64_t graph_hash_u64 = 0;
    std::string pattern;
    std::string op_family;
    std::string realization_key;
    std::set<std::string> capabilities;
    std::set<std::string> profiles;
    std::set<GraphBinding> bindings;
};

struct RegistryIndex {
    std::vector<RegistryGraph> graphs;
    std::map<std::string, RegistryGraph> by_hash;
};

void validate_top_keys_exact(const JsonValue::Object& obj, const fs::path& path,
                             const std::set<std::string>& required_keys,
                             const std::string& label) {
    const std::vector<std::string> keys = sorted_keys(obj);
    std::vector<std::string> required(required_keys.begin(), required_keys.end());
    std::sort(required.begin(), required.end());
    if (keys != required) {
        fail(path.string(), label + " keys must be exactly [" + join_strings(required, ", ") + "]");
    }
}

KernelRecord parse_kernel_record(const fs::path& path, const std::string& arch, const JsonValue& kernel_value,
                                 std::size_t idx, bool strict_id) {
    const std::string where = path.string();
    const std::string prefix = "kernels[" + std::to_string(idx) + "]";
    require_kind(kernel_value, JsonValue::Kind::Object, where, prefix);
    const auto& obj = kernel_value.object_value;

    const std::set<std::string> required = {
        "id", "op_family", "capability", "profile", "config", "realization_key", "graph_hash", "pattern"
    };

    for (const std::string& key : required) {
        if (!has_key(obj, key)) {
            fail(where, prefix + " missing required key: " + key);
        }
    }

    for (const auto& [key, _] : obj) {
        (void)_;
        if (!required.count(key)) {
            fail(where, prefix + " has unsupported key: " + key);
        }
    }

    KernelRecord row;
    row.id = require_nonempty_string(require_key(obj, "id", where), where, prefix + ".id");
    if (strict_id && !is_ascii_id(row.id)) {
        fail(where, prefix + ".id must be ASCII [A-Za-z0-9_.-], got '" + row.id + "'");
    }

    row.op_family = require_nonempty_string(require_key(obj, "op_family", where), where, prefix + ".op_family");
    row.capability = require_nonempty_string(require_key(obj, "capability", where), where, prefix + ".capability");
    row.profile = require_nonempty_string(require_key(obj, "profile", where), where, prefix + ".profile");
    row.realization_key = require_nonempty_string(require_key(obj, "realization_key", where), where, prefix + ".realization_key");
    row.pattern = require_nonempty_string(require_key(obj, "pattern", where), where, prefix + ".pattern");

    if (!is_canonical_preset_pattern(row.pattern)) {
        fail(where, prefix + ".pattern must be canonical axp::l4::preset::*, got '" + row.pattern + "'");
    }

    row.graph_hash = normalize_hash(path, require_key(obj, "graph_hash", where), prefix + ".graph_hash", &row.graph_hash_u64);

    const JsonValue& config = require_key(obj, "config", where);
    if (config.kind != JsonValue::Kind::Object) {
        fail(where, prefix + ".config must be an object");
    }

    if (!is_supported_arch(row.capability)) {
        fail(where, prefix + ".capability must be one of [sm89, sm90, sm100]");
    }
    if (row.capability != arch) {
        fail(where, prefix + ".capability '" + row.capability + "' must match top-level arch '" + arch + "'");
    }

    if (!is_supported_profile(row.profile)) {
        fail(where, prefix + ".profile must be one of [dev_fast, proof_full]");
    }

    return row;
}

ManifestRecord parse_manifest(const fs::path& path, bool strict_id_checks) {
    const std::string where = path.string();
    const JsonValue root = parse_json_file(path);
    require_kind(root, JsonValue::Kind::Object, where, "manifest root");
    const auto& obj = root.object_value;

    validate_top_keys_exact(obj, path, {"schema_version", "arch", "kernels"}, "top-level");

    const int schema_version = require_int(require_key(obj, "schema_version", where), where, "schema_version");
    if (schema_version != 2) {
        fail(where, "schema_version must be 2, got " + std::to_string(schema_version));
    }

    ManifestRecord manifest;
    manifest.path = path;
    manifest.arch = require_nonempty_string(require_key(obj, "arch", where), where, "arch");
    if (!is_supported_arch(manifest.arch)) {
        fail(where, "arch must be one of [sm89, sm90, sm100], got '" + manifest.arch + "'");
    }

    const JsonValue& kernels_value = require_key(obj, "kernels", where);
    require_kind(kernels_value, JsonValue::Kind::Array, where, "kernels");

    manifest.kernels.reserve(kernels_value.array_value.size());
    for (std::size_t i = 0; i < kernels_value.array_value.size(); ++i) {
        manifest.kernels.push_back(parse_kernel_record(path, manifest.arch, kernels_value.array_value[i], i, strict_id_checks));
    }

    return manifest;
}

std::vector<GraphGroup> collect_groups(const std::vector<fs::path>& manifests) {
    std::map<std::string, GraphGroup> by_hash;

    for (const auto& manifest_path : manifests) {
        const ManifestRecord manifest = parse_manifest(manifest_path, false);

        for (const auto& kernel : manifest.kernels) {
            auto it = by_hash.find(kernel.graph_hash);
            if (it == by_hash.end()) {
                GraphGroup g;
                g.graph_hash = kernel.graph_hash;
                g.graph_hash_u64 = kernel.graph_hash_u64;
                g.pattern = kernel.pattern;
                g.op_family = kernel.op_family;
                g.realization_key = kernel.realization_key;
                g.graph_id_candidates.insert(base_graph_id(kernel.id));
                g.capabilities.insert(kernel.capability);
                g.profiles.insert(kernel.profile);
                g.bindings.insert(GraphBinding{kernel.capability, kernel.profile});
                by_hash.emplace(kernel.graph_hash, std::move(g));
            } else {
                GraphGroup& g = it->second;
                if (g.pattern != kernel.pattern) {
                    fail(manifest_path.string(), "pattern conflict for graph_hash " + kernel.graph_hash);
                }
                if (g.op_family != kernel.op_family) {
                    fail(manifest_path.string(), "op_family conflict for graph_hash " + kernel.graph_hash);
                }
                if (g.realization_key != kernel.realization_key) {
                    fail(manifest_path.string(), "realization_key conflict for graph_hash " + kernel.graph_hash);
                }
                g.graph_id_candidates.insert(base_graph_id(kernel.id));
                g.capabilities.insert(kernel.capability);
                g.profiles.insert(kernel.profile);
                g.bindings.insert(GraphBinding{kernel.capability, kernel.profile});
            }
        }
    }

    std::vector<GraphGroup> rows;
    rows.reserve(by_hash.size());
    for (auto& [_, row] : by_hash) {
        rows.push_back(std::move(row));
    }

    std::sort(rows.begin(), rows.end(), [](const GraphGroup& a, const GraphGroup& b) {
        return a.graph_hash_u64 < b.graph_hash_u64;
    });

    std::set<std::string> used_graph_ids;
    for (auto& row : rows) {
        if (row.graph_id_candidates.empty()) {
            fail("collect_groups", "graph has no id candidates");
        }
        std::string base = sanitize_identifier(*row.graph_id_candidates.begin());
        std::string candidate = base;
        if (used_graph_ids.count(candidate)) {
            candidate = base + "_" + row.graph_hash.substr(2, 8);
        }
        int suffix = 1;
        while (used_graph_ids.count(candidate)) {
            ++suffix;
            candidate = base + "_" + std::to_string(suffix);
        }
        row.graph_id = candidate;
        used_graph_ids.insert(candidate);
    }

    return rows;
}

std::string render_registry_json(const std::vector<GraphGroup>& rows) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema_version\": 2,\n";
    out << "  \"graphs\": [\n";

    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        out << "    {\n";
        out << "      \"graph_id\": \"" << json_escape(row.graph_id) << "\",\n";
        out << "      \"graph_hash\": \"" << row.graph_hash << "\",\n";
        out << "      \"pattern\": \"" << json_escape(row.pattern) << "\",\n";
        out << "      \"op_family\": \"" << json_escape(row.op_family) << "\",\n";
        out << "      \"realization_key\": \"" << json_escape(row.realization_key) << "\",\n";

        out << "      \"capabilities\": [\n";
        {
            std::vector<std::string> caps(row.capabilities.begin(), row.capabilities.end());
            std::sort(caps.begin(), caps.end(), [](const std::string& a, const std::string& b) {
                return cap_index(a) < cap_index(b);
            });
            for (std::size_t j = 0; j < caps.size(); ++j) {
                out << "        \"" << caps[j] << "\"";
                out << (j + 1 < caps.size() ? ",\n" : "\n");
            }
        }
        out << "      ],\n";

        out << "      \"profiles\": [\n";
        {
            std::vector<std::string> profiles(row.profiles.begin(), row.profiles.end());
            std::sort(profiles.begin(), profiles.end(), [](const std::string& a, const std::string& b) {
                return profile_index(a) < profile_index(b);
            });
            for (std::size_t j = 0; j < profiles.size(); ++j) {
                out << "        \"" << profiles[j] << "\"";
                out << (j + 1 < profiles.size() ? ",\n" : "\n");
            }
        }
        out << "      ],\n";

        out << "      \"bindings\": [\n";
        {
            std::vector<GraphBinding> bindings(row.bindings.begin(), row.bindings.end());
            std::sort(bindings.begin(), bindings.end());
            for (std::size_t j = 0; j < bindings.size(); ++j) {
                out << "        {\n";
                out << "          \"capability\": \"" << bindings[j].capability << "\",\n";
                out << "          \"profile\": \"" << bindings[j].profile << "\"\n";
                out << "        }";
                out << (j + 1 < bindings.size() ? ",\n" : "\n");
            }
        }
        out << "      ]\n";
        out << "    }";
        out << (i + 1 < rows.size() ? ",\n" : "\n");
    }

    out << "  ]\n";
    out << "}\n";
    return out.str();
}

std::string guard_for_cap(const std::string& cap, const std::string& body) {
    return "#if defined(" + cap_to_guard(cap) + ")\n" + body + "\n#endif";
}

std::string render_registry_header(const std::vector<GraphGroup>& rows) {
    std::vector<std::string> hash_lines;
    hash_lines.reserve(rows.size());
    for (const auto& row : rows) {
        hash_lines.push_back("inline constexpr iro::util::u64 " + row.graph_id + " = " + row.graph_hash + "ULL;");
    }

    std::map<std::string, std::string> pattern_to_key;
    std::map<std::string, std::string> key_to_pattern;
    for (const auto& row : rows) {
        auto it = pattern_to_key.find(row.pattern);
        if (it == pattern_to_key.end()) {
            pattern_to_key.emplace(row.pattern, row.realization_key);
        } else if (it->second != row.realization_key) {
            fail("render_registry_header", "inconsistent realization_key for pattern " + row.pattern);
        }
        auto kit = key_to_pattern.find(row.realization_key);
        if (kit == key_to_pattern.end()) {
            key_to_pattern.emplace(row.realization_key, row.pattern);
        } else if (kit->second != row.pattern) {
            fail("render_registry_header",
                 "realization_key '" + row.realization_key + "' maps to multiple patterns (" +
                 kit->second + ", " + row.pattern + ")");
        }
    }

    std::vector<std::string> tie_break_specs;
    tie_break_specs.reserve(pattern_to_key.size());
    for (const auto& [pattern, realization_key] : pattern_to_key) {
        tie_break_specs.push_back(
            "template<> struct axp::l4::manifest::tie_break_key<" + pattern +
            "> { static constexpr auto value = iro::util::fnv1a_64_cstr(\"" + realization_key + "\"); };"
        );
    }

    std::set<std::pair<std::string, std::string>> seen_enabled;
    std::set<std::pair<std::string, std::string>> seen_overrides;
    std::map<std::pair<std::string, std::string>, std::string> override_hash_by_key;
    std::vector<std::string> enabled_specs;
    std::vector<std::string> entry_specs;
    std::vector<std::string> override_specs;

    for (const auto& row : rows) {
        const std::string hash_ref = "axp::l4::graph_registry::hashes::" + row.graph_id;
        std::vector<GraphBinding> bindings(row.bindings.begin(), row.bindings.end());
        std::sort(bindings.begin(), bindings.end());

        for (const auto& binding : bindings) {
            const std::string cap_cpp = cap_to_cpp(binding.capability);
            const std::string profile_cpp = profile_to_cpp(binding.profile);

            const auto enabled_key = std::make_pair(row.pattern, cap_cpp);
            if (!seen_enabled.count(enabled_key)) {
                seen_enabled.insert(enabled_key);
                enabled_specs.push_back(
                    guard_for_cap(binding.capability,
                        "template<> struct axp::l4::manifest::enabled<" + row.pattern + ", " + cap_cpp + "> : std::true_type {};")
                );
            }

            const auto override_key = std::make_pair(row.realization_key, cap_cpp);
            auto oit = override_hash_by_key.find(override_key);
            if (oit == override_hash_by_key.end()) {
                override_hash_by_key.emplace(override_key, row.graph_hash);
            } else if (oit->second != row.graph_hash) {
                fail("render_registry_header",
                     "lowering identity collision for (realization_key, capability)=(" +
                     row.realization_key + ", " + cap_cpp + ")");
            }
            if (!seen_overrides.count(override_key)) {
                seen_overrides.insert(override_key);
                override_specs.push_back(
                    guard_for_cap(binding.capability,
                        "AXP_GRAPH_HASH_OVERRIDE(" + hash_ref + ", " + row.pattern + ", " + cap_cpp + ");")
                );
            }

            entry_specs.push_back(
                guard_for_cap(binding.capability,
                    "AXP_GRAPH_ENTRY(" + hash_ref + ", " + row.pattern + ", " + cap_cpp + ", " + profile_cpp + ");")
            );
        }
    }

    auto join_block = [](const std::vector<std::string>& lines, const std::string& sep) {
        std::ostringstream oss;
        for (std::size_t i = 0; i < lines.size(); ++i) {
            if (i) {
                oss << sep;
            }
            oss << lines[i];
        }
        return oss.str();
    };

    const std::string hash_block = join_block(hash_lines, "\n");
    const std::string tie_break_block = join_block(tie_break_specs, "\n");
    const std::string enabled_block = join_block(enabled_specs, "\n");
    const std::string specs_block = join_block(entry_specs, "\n\n");
    const std::string overrides_block = join_block(override_specs, "\n\n");

    std::ostringstream out;
    out << "// Generated file. DO NOT EDIT.\n";
    out << "// Regenerate with: tools/ax_manifest_tool.cpp\n\n";
    out << "#pragma once\n\n";
    out << "#if !defined(AXP_LIBRARY_BUILD)\n";
    out << "#error \"axp/l4/graph_registry_index.hpp is library-only; use axp/l4.hpp in application code.\"\n";
    out << "#endif\n\n";
    out << "#include \"../l4.hpp\"\n";
    out << "#include \"bind_key.hpp\"\n";
    out << "#include \"../graph/hash.hpp\"\n";
    out << "#include <type_traits>\n\n";
    out << "namespace axp::l4::graph_registry::hashes {\n\n";
    out << hash_block << "\n\n";
    out << "} // namespace axp::l4::graph_registry::hashes\n\n";
    out << "namespace axp::l4::graph_registry {\n\n";
    out << "template<iro::util::u64 GraphHash, class Cap, class ProfileT, class = void>\n";
    out << "struct entry {\n";
    out << "    static constexpr bool enabled = false;\n";
    out << "    using pattern = void;\n";
    out << "    static constexpr iro::util::u64 realization_key = 0;\n";
    out << "};\n\n";
    out << "template<iro::util::u64 GraphHash, class Cap, class ProfileT>\n";
    out << "inline constexpr bool enabled_v = entry<GraphHash, Cap, ProfileT>::enabled;\n\n";
    out << "} // namespace axp::l4::graph_registry\n\n";
    out << tie_break_block << "\n\n";
    out << enabled_block << "\n\n";
    out << "#define AXP_GRAPH_ENTRY(GRAPH_HASH, ENTRY, CAP, PROFILE)                                      \\\n";
    out << "template<>                                                                                    \\\n";
    out << "struct axp::l4::graph_registry::entry<GRAPH_HASH, CAP, PROFILE> {                           \\\n";
    out << "    static constexpr bool enabled = true;                                                     \\\n";
    out << "    using pattern = ENTRY;                                                                     \\\n";
    out << "    static constexpr iro::util::u64 realization_key =                                         \\\n";
    out << "        axp::l4::manifest::tie_break_key<pattern>::value;                                     \\\n";
    out << "}\n\n";
    out << "#define AXP_GRAPH_HASH_OVERRIDE(GRAPH_HASH, ENTRY, CAP)                                                   \\\n";
    out << "template<>                                                                                                \\\n";
    out << "struct axp::graph::graph_hash_override<                                                           \\\n";
    out << "    axp::level3::registry::Select<axp::l4::lowering::to_l3_pattern_t<ENTRY>, CAP>> {            \\\n";
    out << "    static constexpr bool enabled = true;                                                     \\\n";
    out << "    static constexpr iro::util::u64 value = GRAPH_HASH;                                       \\\n";
    out << "}\n\n";
    out << specs_block << "\n\n";
    out << overrides_block << "\n\n";
    out << "#undef AXP_GRAPH_ENTRY\n";
    out << "#undef AXP_GRAPH_HASH_OVERRIDE\n";

    return out.str();
}

RegistryIndex load_registry_index(const fs::path& path) {
    const std::string where = path.string();
    const JsonValue root = parse_json_file(path);
    require_kind(root, JsonValue::Kind::Object, where, "registry root");
    const auto& obj = root.object_value;

    const int schema_version = require_int(require_key(obj, "schema_version", where), where, "schema_version");
    if (schema_version != 2) {
        fail(where, "registry schema_version must be 2");
    }

    const JsonValue& graphs_value = require_key(obj, "graphs", where);
    require_kind(graphs_value, JsonValue::Kind::Array, where, "graphs");

    RegistryIndex index;
    for (std::size_t i = 0; i < graphs_value.array_value.size(); ++i) {
        const std::string prefix = "graphs[" + std::to_string(i) + "]";
        const JsonValue& graph_value = graphs_value.array_value[i];
        require_kind(graph_value, JsonValue::Kind::Object, where, prefix);
        const auto& gobj = graph_value.object_value;

        const std::set<std::string> required = {
            "graph_id", "graph_hash", "pattern", "op_family", "realization_key",
            "capabilities", "profiles", "bindings"
        };

        for (const std::string& key : required) {
            if (!has_key(gobj, key)) {
                fail(where, prefix + " missing required key: " + key);
            }
        }

        for (const auto& [key, _] : gobj) {
            (void)_;
            if (!required.count(key)) {
                fail(where, prefix + " has unsupported key: " + key);
            }
        }

        RegistryGraph graph;
        graph.graph_id = require_nonempty_string(require_key(gobj, "graph_id", where), where, prefix + ".graph_id");
        graph.graph_hash = normalize_hash(path, require_key(gobj, "graph_hash", where), prefix + ".graph_hash", &graph.graph_hash_u64);
        graph.pattern = require_nonempty_string(require_key(gobj, "pattern", where), where, prefix + ".pattern");
        if (!is_canonical_preset_pattern(graph.pattern)) {
            fail(where, prefix + ".pattern must be canonical axp::l4::preset::*");
        }

        graph.op_family = require_nonempty_string(require_key(gobj, "op_family", where), where, prefix + ".op_family");
        graph.realization_key = require_nonempty_string(require_key(gobj, "realization_key", where), where, prefix + ".realization_key");

        const JsonValue& caps_v = require_key(gobj, "capabilities", where);
        require_kind(caps_v, JsonValue::Kind::Array, where, prefix + ".capabilities");
        if (caps_v.array_value.empty()) {
            fail(where, prefix + ".capabilities must be a non-empty array");
        }
        for (std::size_t ci = 0; ci < caps_v.array_value.size(); ++ci) {
            const std::string cap = require_nonempty_string(caps_v.array_value[ci], where, prefix + ".capabilities[" + std::to_string(ci) + "]");
            if (!is_supported_arch(cap)) {
                fail(where, prefix + ".capabilities contains unsupported cap: " + cap);
            }
            graph.capabilities.insert(cap);
        }

        const JsonValue& profiles_v = require_key(gobj, "profiles", where);
        require_kind(profiles_v, JsonValue::Kind::Array, where, prefix + ".profiles");
        if (profiles_v.array_value.empty()) {
            fail(where, prefix + ".profiles must be a non-empty array");
        }
        for (std::size_t pi = 0; pi < profiles_v.array_value.size(); ++pi) {
            const std::string profile = require_nonempty_string(profiles_v.array_value[pi], where, prefix + ".profiles[" + std::to_string(pi) + "]");
            if (!is_supported_profile(profile)) {
                fail(where, prefix + ".profiles contains unsupported profile: " + profile);
            }
            graph.profiles.insert(profile);
        }

        const JsonValue& bindings_v = require_key(gobj, "bindings", where);
        require_kind(bindings_v, JsonValue::Kind::Array, where, prefix + ".bindings");
        if (bindings_v.array_value.empty()) {
            fail(where, prefix + ".bindings must be a non-empty array");
        }

        for (std::size_t bi = 0; bi < bindings_v.array_value.size(); ++bi) {
            const std::string bprefix = prefix + ".bindings[" + std::to_string(bi) + "]";
            const JsonValue& bval = bindings_v.array_value[bi];
            require_kind(bval, JsonValue::Kind::Object, where, bprefix);
            const auto& bobj = bval.object_value;
            validate_top_keys_exact(bobj, path, {"capability", "profile"}, bprefix);
            const std::string cap = require_nonempty_string(require_key(bobj, "capability", where), where, bprefix + ".capability");
            const std::string profile = require_nonempty_string(require_key(bobj, "profile", where), where, bprefix + ".profile");
            if (!is_supported_arch(cap)) {
                fail(where, bprefix + ".capability unsupported: " + cap);
            }
            if (!is_supported_profile(profile)) {
                fail(where, bprefix + ".profile unsupported: " + profile);
            }
            const GraphBinding binding{cap, profile};
            if (graph.bindings.count(binding)) {
                fail(where, prefix + " duplicate binding pair");
            }
            graph.bindings.insert(binding);
        }

        std::set<std::string> derived_caps;
        std::set<std::string> derived_profiles;
        for (const auto& b : graph.bindings) {
            derived_caps.insert(b.capability);
            derived_profiles.insert(b.profile);
        }
        if (derived_caps != graph.capabilities) {
            fail(where, prefix + ".capabilities must match bindings-derived capabilities");
        }
        if (derived_profiles != graph.profiles) {
            fail(where, prefix + ".profiles must match bindings-derived profiles");
        }

        if (index.by_hash.count(graph.graph_hash)) {
            fail(where, "duplicate graph_hash in registry: " + graph.graph_hash);
        }
        index.by_hash.emplace(graph.graph_hash, graph);
        index.graphs.push_back(std::move(graph));
    }

    return index;
}

void validate_manifest_against_registry(const ManifestRecord& manifest, const RegistryIndex& registry) {
    std::set<std::string> seen_ids;
    std::set<std::tuple<std::string, std::string, std::string, std::string>> seen_bind_keys;

    for (std::size_t i = 0; i < manifest.kernels.size(); ++i) {
        const auto& kernel = manifest.kernels[i];
        const std::string prefix = "kernels[" + std::to_string(i) + "]";

        if (!is_ascii_id(kernel.id)) {
            fail(manifest.path.string(), prefix + ".id must be ASCII [A-Za-z0-9_.-], got '" + kernel.id + "'");
        }
        if (seen_ids.count(kernel.id)) {
            fail(manifest.path.string(), "duplicate kernel id '" + kernel.id + "'");
        }
        seen_ids.insert(kernel.id);

        const auto bind_tuple = std::make_tuple(kernel.graph_hash, kernel.capability, kernel.profile, kernel.realization_key);
        if (seen_bind_keys.count(bind_tuple)) {
            fail(manifest.path.string(), "duplicate bind tuple (graph_hash, capability, profile, realization_key)");
        }
        seen_bind_keys.insert(bind_tuple);

        auto it = registry.by_hash.find(kernel.graph_hash);
        if (it == registry.by_hash.end()) {
            fail(manifest.path.string(), prefix + ".graph_hash '" + kernel.graph_hash + "' not present in generated registry index");
        }
        const RegistryGraph& graph = it->second;

        if (kernel.op_family != graph.op_family) {
            fail(manifest.path.string(), prefix + ".op_family does not match registry");
        }
        if (kernel.realization_key != graph.realization_key) {
            fail(manifest.path.string(), prefix + ".realization_key does not match registry");
        }
        if (kernel.pattern != graph.pattern) {
            fail(manifest.path.string(), prefix + ".pattern does not match registry");
        }
        if (!graph.capabilities.count(kernel.capability)) {
            fail(manifest.path.string(), prefix + " capability not allowed for graph_hash");
        }
        if (!graph.profiles.count(kernel.profile)) {
            fail(manifest.path.string(), prefix + " profile not allowed for graph_hash");
        }
        if (!graph.bindings.count(GraphBinding{kernel.capability, kernel.profile})) {
            fail(manifest.path.string(), prefix + " binding (capability, profile) not allowed for graph_hash");
        }
    }

    std::vector<std::string> ids;
    ids.reserve(manifest.kernels.size());
    for (const auto& k : manifest.kernels) {
        ids.push_back(k.id);
    }
    std::vector<std::string> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    if (ids != sorted) {
        fail(manifest.path.string(), "kernels must be sorted by lexical id for deterministic generation");
    }
}

std::string render_instantiation_tu(const std::string& manifest_file,
                                    const KernelRecord& kernel,
                                    const RegistryGraph& graph_meta,
                                    const std::string& cap_type,
                                    const std::string& profile_type) {
    std::ostringstream out;
    out << "// Generated file. DO NOT EDIT.\n";
    out << "// Instantiation wrapper only; no CUDA kernel definitions here.\n";
    out << "// source: " << manifest_file << "\n";
    out << "// kernel_id: " << kernel.id << "\n";
    out << "// graph_hash: " << kernel.graph_hash << "\n";
    out << "// realization_key: " << kernel.realization_key << "\n";
    out << "// profile: " << kernel.profile << "\n";
    out << "// schema_version: 2\n\n";
    out << "#include <type_traits>\n";
    out << "#include \"iro_cuda_ax_core.hpp\"\n";
    out << "#include <axp/l4.hpp>\n";
    out << "#include <axp/l4/resolve.hpp>\n\n";
    out << "namespace axp_generated_inst {\n\n";
    out << "inline constexpr iro::util::u64 kGraphHash = " << kernel.graph_hash << "ULL;\n";
    out << "using Cap = " << cap_type << ";\n";
    out << "using Profile = " << profile_type << ";\n";
    out << "using GraphEntry = axp::l4::graph_registry::entry<kGraphHash, Cap, Profile>;\n";
    out << "static_assert(GraphEntry::enabled,\n";
    out << "              \"manifest graph row is not enabled for this capability/profile\");\n";
    out << "using L4Pattern = " << graph_meta.pattern << ";\n";
    out << "static_assert(std::is_same_v<typename GraphEntry::pattern, L4Pattern>,\n";
    out << "              \"registry graph row resolves to unexpected pattern\");\n";
    out << "static_assert(axp::l4::manifest::enabled_v<L4Pattern, Cap>,\n";
    out << "              \"manifest row references a pattern disabled for this capability\");\n";
    out << "static_assert(axp::l4::manifest::tie_break_key<L4Pattern>::value == "
        << "iro::util::fnv1a_64_cstr(\"" << kernel.realization_key << "\"),\n";
    out << "              \"manifest realization_key mismatch for resolved pattern\");\n\n";
    out << "using Pattern = axp::l4::lowering::to_l3_pattern_t<L4Pattern>;\n\n";

    out << "using Graph = axp::level3::registry::Select<Pattern, Cap>;\n";
    out << "template struct axp::l4::resolve<Graph, Cap, Profile>;\n";
    out << "template struct axp::level3::registry::resolve_impl<Pattern, Cap>;\n";
    out << "using Selected = axp::l4::Select<Graph, Cap, Profile>;\n";
    out << "static_assert(std::is_same_v<Selected, Graph>,\n";
    out << "              \"graph-hash resolve must return registry graph type directly\");\n\n";
    out << "} // namespace axp_generated_inst\n";
    return out.str();
}

std::string cap_type_for_arch(const std::string& arch) {
    if (arch == "sm89") return "iro::cap::sm89";
    if (arch == "sm90") return "iro::cap::sm90";
    if (arch == "sm100") return "iro::cap::sm100";
    fail("cap_type_for_arch", "unsupported arch: " + arch);
}

struct ParsedOptions {
    std::map<std::string, std::vector<std::string>> values;
};

ParsedOptions parse_options(int argc, char** argv, int start_index) {
    ParsedOptions opts;
    for (int i = start_index; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            fail("argv", "expected option starting with '--', got '" + key + "'");
        }
        if (i + 1 >= argc) {
            fail("argv", "missing value for option '" + key + "'");
        }
        key = key.substr(2);
        const std::string value = argv[++i];
        opts.values[key].push_back(value);
    }
    return opts;
}

std::optional<std::string> get_one(const ParsedOptions& opts, const std::string& key) {
    auto it = opts.values.find(key);
    if (it == opts.values.end()) {
        return std::nullopt;
    }
    if (it->second.size() != 1) {
        fail("argv", "option '--" + key + "' must appear exactly once");
    }
    return it->second.front();
}

std::vector<std::string> get_many(const ParsedOptions& opts, const std::string& key) {
    auto it = opts.values.find(key);
    if (it == opts.values.end()) {
        return {};
    }
    return it->second;
}

std::vector<fs::path> default_manifest_paths() {
    return {
        fs::path("manifests/kernels_sm89.json"),
        fs::path("manifests/kernels_sm90.json"),
        fs::path("manifests/kernels_sm100.json"),
    };
}

int cmd_gen_registry_index(const ParsedOptions& opts) {
    const fs::path json_out = get_one(opts, "json-out").value_or("tools/generated/graph_registry_index.json");
    const fs::path header_out = get_one(opts, "header-out").value_or("include/axp/l4/graph_registry_index.hpp");

    std::vector<std::string> manifest_values = get_many(opts, "manifest");
    std::vector<fs::path> manifests;
    if (manifest_values.empty()) {
        manifests = default_manifest_paths();
    } else {
        manifests.reserve(manifest_values.size());
        for (const auto& s : manifest_values) {
            manifests.emplace_back(s);
        }
    }

    for (const auto& path : manifests) {
        if (!fs::exists(path)) {
            fail(path.string(), "manifest path does not exist");
        }
    }

    const auto groups = collect_groups(manifests);
    write_text_file(header_out, render_registry_header(groups));
    write_text_file(json_out, render_registry_json(groups));

    std::cout << "generated: " << fs::absolute(header_out).string() << "\n";
    std::cout << "generated: " << fs::absolute(json_out).string() << "\n";
    return 0;
}

int cmd_validate_manifest(const ParsedOptions& opts) {
    const auto registry_path = fs::path(get_one(opts, "registry").value_or("tools/generated/graph_registry_index.json"));
    const auto manifests_raw = get_many(opts, "manifest");
    if (manifests_raw.empty()) {
        fail("argv", "--manifest is required for validate-manifest");
    }

    const RegistryIndex registry = load_registry_index(registry_path);

    for (const auto& m : manifests_raw) {
        const fs::path manifest_path = m;
        const ManifestRecord manifest = parse_manifest(manifest_path, true);
        validate_manifest_against_registry(manifest, registry);
        std::cout << "validated: " << fs::absolute(manifest_path).string() << "\n";
    }

    return 0;
}

int cmd_gen_instantiations(const ParsedOptions& opts) {
    const auto manifest_opt = get_one(opts, "manifest");
    const auto out_dir_opt = get_one(opts, "out-dir");
    if (!manifest_opt.has_value() || !out_dir_opt.has_value()) {
        fail("argv", "gen-instantiations requires --manifest and --out-dir");
    }

    const fs::path manifest_path = *manifest_opt;
    const fs::path out_dir = *out_dir_opt;
    const fs::path registry_path = get_one(opts, "registry").value_or("tools/generated/graph_registry_index.json");
    const std::string profile = get_one(opts, "profile").value_or("proof_full");
    if (!is_supported_profile(profile)) {
        fail("argv", "--profile must be one of [dev_fast, proof_full]");
    }

    const RegistryIndex registry = load_registry_index(registry_path);
    const ManifestRecord manifest = parse_manifest(manifest_path, true);
    validate_manifest_against_registry(manifest, registry);

    fs::create_directories(out_dir);
    for (const auto& e : fs::directory_iterator(out_dir)) {
        if (!e.is_regular_file()) {
            continue;
        }
        if (e.path().extension() == ".cu") {
            fs::remove(e.path());
        }
    }

    std::vector<KernelRecord> kernels;
    for (const auto& k : manifest.kernels) {
        if (k.profile == profile) {
            kernels.push_back(k);
        }
    }

    if (kernels.empty()) {
        fail(manifest_path.string(),
             "gen-instantiations: profile '" + profile + "' exists but selected zero manifest rows");
    }

    const std::string cap_type = cap_type_for_arch(manifest.arch);

    for (const auto& kernel : kernels) {
        auto it = registry.by_hash.find(kernel.graph_hash);
        if (it == registry.by_hash.end()) {
            fail(manifest_path.string(), "graph_hash not found in registry for kernel " + kernel.id);
        }
        const auto& graph_meta = it->second;
        const std::string profile_type = profile_to_cpp(kernel.profile);
        const std::string content = render_instantiation_tu(
            manifest_path.filename().string(),
            kernel,
            graph_meta,
            cap_type,
            profile_type
        );

        const fs::path output = out_dir / (kernel.id + ".cu");
        write_text_file(output, content);
        std::cout << "generated: " << fs::absolute(output).string() << "\n";
    }

    return 0;
}

void print_usage() {
    std::cerr
        << "usage:\n"
        << "  ax_manifest_tool gen-registry-index [--json-out PATH] [--header-out PATH] [--manifest PATH ...]\n"
        << "  ax_manifest_tool validate-manifest --manifest PATH [--manifest PATH ...] [--registry PATH]\n"
        << "  ax_manifest_tool gen-instantiations --manifest PATH --out-dir DIR [--registry PATH] [--profile dev_fast|proof_full]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage();
            return 1;
        }

        const std::string cmd = argv[1];
        const ParsedOptions opts = parse_options(argc, argv, 2);

        if (cmd == "gen-registry-index") {
            return cmd_gen_registry_index(opts);
        }
        if (cmd == "validate-manifest") {
            return cmd_validate_manifest(opts);
        }
        if (cmd == "gen-instantiations") {
            return cmd_gen_instantiations(opts);
        }

        print_usage();
        return 1;
    } catch (const ToolError& err) {
        std::cerr << err.what() << "\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << "error: " << err.what() << "\n";
        return 1;
    }
}
