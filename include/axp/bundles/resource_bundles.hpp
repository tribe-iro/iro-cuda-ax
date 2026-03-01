#pragma once

#include <iro_cuda_ax_core.hpp>
#include "../naming/tags.hpp"
#include "../protocol/stage/resources.hpp"

namespace axp::bundle {

template<class Tag, int Slots, long long BytesPerSlot, int AlignBytes>
using SmemPipe = iro::contract::res::smem_pipeline<Tag, Slots, BytesPerSlot, AlignBytes>;

template<class Tag, long long Bytes, int AlignBytes>
using SmemArena = iro::contract::res::smem_arena<Tag, Bytes, AlignBytes>;

// Common GEMM pipelines (A/B double-buffered)
template<class TagA, class TagB>
using GemmAB_Pipe2_128x64_F16 = iro::util::type_list<
    axp::protocol::stage::Pipe2_128x64_F16<TagA>,
    axp::protocol::stage::Pipe2_128x64_F16<TagB>
>;

template<class TagA, class TagB>
using GemmAB_Pipe2_128x128_F16 = iro::util::type_list<
    axp::protocol::stage::Pipe2_128x128_F16<TagA>,
    axp::protocol::stage::Pipe2_128x128_F16<TagB>
>;

template<class TagA, class TagB>
using GemmAB_Pipe2_64x64_F16 = iro::util::type_list<
    axp::protocol::stage::Pipe2_64x64_F16<TagA>,
    axp::protocol::stage::Pipe2_64x64_F16<TagB>
>;

// Attention pipelines (Q single-buffer, KV double-buffer)
template<class TagQ, class TagK>
using AttnQKV_Pipes = iro::util::type_list<
    axp::protocol::stage::PipeQ_64x128_F16<TagQ>,
    axp::protocol::stage::PipeKV_64x128_F16<TagK>
>;

// Reduction scratch (arena)
template<class Tag>
using ReduceScratch_16K = SmemArena<Tag, 16 * 1024, 16>;

} // namespace axp::bundle
