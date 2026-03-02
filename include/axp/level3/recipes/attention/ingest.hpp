#pragma once

namespace axp::level3::recipes::attention::ingest {

// Ingest-plane composition entrypoint marker for attention recipes.
// Concrete ingest operators are currently defined in process.hpp while
// decomposition is migrated to explicit per-plane components.
struct marker {};

} // namespace axp::level3::recipes::attention::ingest
