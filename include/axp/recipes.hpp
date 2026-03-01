#pragma once

#include <iro_cuda_ax_core.hpp>

namespace axp::recipe {

using F16AccF32Fast = iro::recipe::Precision<
    iro::elem::f16, iro::elem::f32, iro::elem::f16, 16, iro::recipe::Fast>;
using F16AccF32Exact = iro::recipe::Precision<
    iro::elem::f16, iro::elem::f32, iro::elem::f16, 16, iro::recipe::Exact>;
using F16AccF16Fast = iro::recipe::Precision<
    iro::elem::f16, iro::elem::f16, iro::elem::f16, 16, iro::recipe::Fast>;
using BF16AccF32Fast = iro::recipe::Precision<
    iro::elem::bf16, iro::elem::f32, iro::elem::bf16, 16, iro::recipe::Fast>;
using BF16AccF32Exact = iro::recipe::Precision<
    iro::elem::bf16, iro::elem::f32, iro::elem::bf16, 16, iro::recipe::Exact>;
using F32Fast = iro::recipe::Precision<
    iro::elem::f32, iro::elem::f32, iro::elem::f32, 4, iro::recipe::Fast>;
using F32Exact = iro::recipe::Precision<
    iro::elem::f32, iro::elem::f32, iro::elem::f32, 4, iro::recipe::Exact>;

} // namespace axp::recipe
