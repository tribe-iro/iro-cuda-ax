#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::intent {

namespace memory_pattern {
struct None {};
struct Optimized {};
} // namespace memory_pattern

namespace load_mode {
struct Streaming {};
struct AsyncPrefetch {};
} // namespace load_mode

namespace schedule {
struct ProducerConsumer {};
struct Pipelined {};
struct BulkSynchronous {};
} // namespace schedule

namespace tile_skip {
struct None {};
struct Causal {};
} // namespace tile_skip

} // namespace axp::intent
