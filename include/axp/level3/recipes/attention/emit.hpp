#pragma once

namespace axp::level3::recipes::attention::emit {

// Emit-plane composition entrypoint marker for attention recipes.
// Concrete emit operators are currently defined in process.hpp while
// decomposition is migrated to explicit per-plane components.
struct marker {};

} // namespace axp::level3::recipes::attention::emit
