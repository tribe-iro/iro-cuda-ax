#pragma once

#include "common.hpp"

namespace preset {

using StreamingMicrobatch1024 = axp::l4::StreamingMicrobatchPattern<
    axp::recipe::F32Exact,
    StreamValuePayload,
    StreamIndexPayload,
    StreamStateTile,
    StreamValueSubj,
    StreamIndexSubj,
    StreamStateSubj,
    StreamAtomicOutSubj,
    StreamDependEventTag,
    StreamPhaseProcessTag,
    StreamDoneEventTag,
    iro::exec::block
>;

} // namespace preset
