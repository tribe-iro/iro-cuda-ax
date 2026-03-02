#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::intent {

namespace memory_pattern {
struct None {};
struct Optimized {};
struct Persistent {};
struct RingBuffered {};
} // namespace memory_pattern

namespace load_mode {
struct Streaming {};
struct AsyncPrefetch {};
struct DeterministicLatency {};
} // namespace load_mode

namespace schedule {
struct ProducerConsumer {};
struct Pipelined {};
struct BulkSynchronous {};
struct PersistentPipeline {};
struct DeterministicLatency {};
} // namespace schedule

namespace tile_skip {
struct None {};
struct Causal {};
} // namespace tile_skip

} // namespace axp::intent
