#include <axp/swizzle.hpp>
#include <axp/level0/stage.hpp>
#include <axp/realize/l0.hpp>
#include <axp/l4.hpp>
#include <axp/l4/preset/elementwise_norm_sort_hist.hpp>
#include <axp/level3/elementwise.hpp>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace {

using L4Pattern = axp::l4::preset::VectorizedElementwise16x16;
using L3Pattern = axp::l4::lowering::to_l3_pattern_t<L4Pattern>;
using Graph = axp::level3::registry::Select<L3Pattern, iro::cap::sm89>;

template<class Obligation>
struct realization_from_obligation;

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class InExtra, class OutExtra>
struct realization_from_obligation<axp::level0::LdGlobal<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup,
    CachePolicy, InDist, OutDist, InExtra, OutExtra>> {
    using type = axp::realize::l0::LdGlobal<
        Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup,
        CachePolicy, InDist, OutDist, InExtra, OutExtra>;
};

template<class Recipe, class InTile, class OutTile, class InSubj, class OutSubj, class ExecGroup,
         class CachePolicy, class InDist, class OutDist, class InExtra, class OutExtra>
struct realization_from_obligation<axp::level0::StGlobal<
    Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup,
    CachePolicy, InDist, OutDist, InExtra, OutExtra>> {
    using type = axp::realize::l0::StGlobal<
        Recipe, InTile, OutTile, InSubj, OutSubj, ExecGroup,
        CachePolicy, InDist, OutDist, InExtra, OutExtra>;
};

using Obligations = typename Graph::obligations;
static_assert(iro::util::size_v<Obligations> == 2,
              "Expected LdGlobal + StGlobal composition for elementwise copy");

using LdOb = iro::util::at_t<Obligations, 0>;
using StOb = iro::util::at_t<Obligations, 1>;
using LdReal = typename realization_from_obligation<LdOb>::type;
using StReal = typename realization_from_obligation<StOb>::type;

using RegTile = typename iro::util::at_t<typename LdOb::outputs, 0>::payload;
using Storage = typename RegTile::elem::storage_t;
static_assert(std::is_same_v<Storage, float>,
              "This example expects float storage for VectorizedElementwise16x16");

constexpr int kElems = RegTile::shape::size;
static_assert(kElems == 16 * 16, "Unexpected tile size for VectorizedElementwise16x16");

inline void check_cuda(const cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(status));
        std::fflush(stderr);
        std::exit(1);
    }
}

__global__ void ax_l4_elementwise_copy_kernel(const float* in, float* out) {
    Storage reg_tile[kElems];
    LdReal::execute(in, reg_tile);
    StReal::execute(reg_tile, out);
}

} // namespace

int main() {
    constexpr int n = kElems;
    std::vector<float> h_in(static_cast<std::size_t>(n));
    std::vector<float> h_out(static_cast<std::size_t>(n), 0.0f);

    for (int i = 0; i < n; ++i) {
        h_in[static_cast<std::size_t>(i)] = static_cast<float>(i) * 1.25f - 3.0f;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    check_cuda(cudaMalloc(&d_in, bytes), "cudaMalloc(d_in)");
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc(d_out)");

    check_cuda(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D in");

    ax_l4_elementwise_copy_kernel<<<1, 1>>>(d_in, d_out);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel sync");

    check_cuda(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H out");

    float max_abs_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float expect = h_in[static_cast<std::size_t>(i)];
        const float got = h_out[static_cast<std::size_t>(i)];
        const float err = std::fabs(expect - got);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }

    std::printf("AX SM89 L4 example\n");
    std::printf("preset=axp::l4::preset::VectorizedElementwise16x16\n");
    std::printf("flow=L4 pattern -> L3 graph -> graph obligations -> realizations\n");
    std::printf("graph_obligations=%d launch_blocks=1 launch_threads=1\n", iro::util::size_v<Obligations>);
    std::printf("out[0]=%.3f out[1]=%.3f out[2]=%.3f\n", h_out[0], h_out[1], h_out[2]);
    std::printf("max_abs_err=%.8f\n", max_abs_err);

    check_cuda(cudaFree(d_in), "cudaFree(d_in)");
    check_cuda(cudaFree(d_out), "cudaFree(d_out)");
    return 0;
}
