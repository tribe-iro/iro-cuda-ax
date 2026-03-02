#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    out << text;
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

std::string level1_text() {
    return R"(#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/level1/passthrough.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

// Generated file: do not edit manually.
// Source: tools/gen_layer_adapters.cpp

#include "../level0/index.hpp"
#include "../protocol/index.hpp"

namespace axp::level1 {

// Canonical L1 pass-through views over L0/protocol atoms.
namespace low = axp::level0;
namespace proto = axp::protocol;

} // namespace axp::level1
)";
}

std::string level2_text() {
    return R"(#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/level2/passthrough.hpp is internal substrate; use axp/l4.hpp in application code."
#endif

// Generated file: do not edit manually.
// Source: tools/gen_layer_adapters.cpp

#include "../level1/passthrough.hpp"

namespace axp::level2 {

// Canonical L2 pass-through views over L1 interfaces.
namespace low = axp::level1::low;
namespace proto = axp::level1::proto;

} // namespace axp::level2
)";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <repo-root>\n";
        return 2;
    }

    try {
        const std::filesystem::path root = std::filesystem::path(argv[1]);
        const auto level1_out = root / "include" / "axp" / "level1" / "passthrough.hpp";
        const auto level2_out = root / "include" / "axp" / "level2" / "passthrough.hpp";

        write_file(level1_out, level1_text());
        write_file(level2_out, level2_text());

        std::cout << level1_out.string() << "\n";
        std::cout << level2_out.string() << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "gen_layer_adapters: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
