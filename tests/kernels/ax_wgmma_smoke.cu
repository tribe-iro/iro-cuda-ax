// Minimal compile-only check for generated WGMMA entry points.
#include <cstdint>
#include <axp/realize/detail/wgmma_generated.hpp>

__global__ void ax_wgmma_smoke() {
}
