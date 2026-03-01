/**
 * @file ax_asserts.cu
 * @brief Compile-time verification for iro-cuda-ax CORE v0.2.4.
 *
 * This translation unit validates that the iro-cuda-ax contract system
 * compiles correctly with nvcc and all static assertions pass.
 */

#include "iro_cuda_ax_core.hpp"
#include <axp/graph/index.hpp>

// =============================================================================
// Verify iro::util
// =============================================================================

// Type list operations
static_assert(iro::util::size_v<iro::util::type_list<int, float, double>> == 3);
static_assert(std::is_same_v<iro::util::at_t<iro::util::type_list<int, float, double>, 0>, int>);
static_assert(std::is_same_v<iro::util::at_t<iro::util::type_list<int, float, double>, 1>, float>);
static_assert(std::is_same_v<iro::util::at_t<iro::util::type_list<int, float, double>, 2>, double>);

// Contains
static_assert(iro::util::contains_v<iro::util::type_list<int, float>, int>);
static_assert(iro::util::contains_v<iro::util::type_list<int, float>, float>);
static_assert(!iro::util::contains_v<iro::util::type_list<int, float>, double>);

// FNV-1a stable identity
static_assert(iro::util::fnv1a_64_cstr("test") != 0);
static_assert(iro::util::fnv1a_64_cstr("test") == iro::util::fnv1a_64_cstr("test"));
static_assert(iro::util::fnv1a_64_cstr("test") != iro::util::fnv1a_64_cstr("Test"));

// =============================================================================
// Verify iro::contract::Shape
// =============================================================================

static_assert(iro::contract::Shape<16>::rank == 1);
static_assert(iro::contract::Shape<16>::size == 16);
static_assert(iro::contract::Shape<16, 32>::rank == 2);
static_assert(iro::contract::Shape<16, 32>::size == 512);
static_assert(iro::contract::Shape<4, 8, 16>::rank == 3);
static_assert(iro::contract::Shape<4, 8, 16>::size == 512);
static_assert(iro::contract::Shape<>::rank == 0);
static_assert(iro::contract::Shape<>::size == 1);

// =============================================================================
// Verify iro::elem
// =============================================================================

static_assert(iro::elem::f16::bytes == 2);
static_assert(iro::elem::f16::align == 2);
static_assert(iro::elem::f32::bytes == 4);
static_assert(iro::elem::f32::align == 4);
static_assert(iro::elem::f64::bytes == 8);
static_assert(iro::elem::f64::align == 8);
static_assert(iro::elem::bf16::bytes == 2);
static_assert(iro::elem::e4m3::bytes == 1);
static_assert(iro::elem::e5m2::bytes == 1);

// Integer types
static_assert(iro::elem::i8::bytes == 1);
static_assert(iro::elem::i16::bytes == 2);
static_assert(iro::elem::i32::bytes == 4);
static_assert(iro::elem::i64::bytes == 8);
static_assert(iro::elem::u8::bytes == 1);
static_assert(iro::elem::u32::bytes == 4);

// =============================================================================
// Verify iro::contract::Align
// =============================================================================

static_assert(iro::contract::Align<4>::bytes == 4);
static_assert(iro::contract::Align<16>::bytes == 16);
static_assert(iro::contract::Align<128>::bytes == 128);

// =============================================================================
// Verify iro::scope ordering (normative per v0.2.4)
// =============================================================================

static_assert(iro::scope::lane::level == 0);
static_assert(iro::scope::warp::level == 1);
static_assert(iro::scope::warpgroup::level == 2);
static_assert(iro::scope::block::level == 3);
static_assert(iro::scope::cluster::level == 4);
static_assert(iro::scope::device::level == 5);

static_assert(iro::scope::lane::level < iro::scope::warp::level);
static_assert(iro::scope::warp::level < iro::scope::block::level);
static_assert(iro::scope::block::level < iro::scope::device::level);

// =============================================================================
// Verify iro::verify::scope_subsumes
// =============================================================================

static_assert(iro::verify::scope_subsumes(iro::scope::block::level, iro::scope::warp::level));
static_assert(iro::verify::scope_subsumes(iro::scope::device::level, iro::scope::block::level));
static_assert(!iro::verify::scope_subsumes(iro::scope::warp::level, iro::scope::block::level));
static_assert(iro::verify::scope_subsumes(iro::scope::warp::level, iro::scope::warp::level));

// =============================================================================
// Verify iro::token types
// =============================================================================

namespace token_test {

struct TestSubject { static constexpr auto id = iro::util::fnv1a_64_cstr("test.subject"); };

using VisibleWarp = iro::token::visible_at<TestSubject, iro::scope::warp>;
using VisibleBlock = iro::token::visible_at<TestSubject, iro::scope::block>;

static_assert(VisibleWarp::id != 0);
static_assert(VisibleBlock::id != 0);
static_assert(VisibleWarp::id != VisibleBlock::id);

// Token satisfaction: block scope subsumes warp scope
static_assert(iro::verify::token_satisfies<VisibleBlock, VisibleWarp>());
static_assert(!iro::verify::token_satisfies<VisibleWarp, VisibleBlock>());

// lanes_valid tokens
using Lanes32 = iro::token::lanes_valid<TestSubject, 32>;
using Lanes16 = iro::token::lanes_valid<TestSubject, 16>;

static_assert(iro::verify::token_satisfies<Lanes32, Lanes16>());
static_assert(!iro::verify::token_satisfies<Lanes16, Lanes32>());

// alive tokens
using AliveBlock = iro::token::alive<TestSubject, iro::token::lifetime::block>;
using AliveWarp = iro::token::alive<TestSubject, iro::token::lifetime::warp>;

static_assert(iro::verify::token_satisfies<AliveBlock, AliveWarp>());
static_assert(!iro::verify::token_satisfies<AliveWarp, AliveBlock>());

// sync_at tokens
using SyncBlock = iro::token::sync_at<TestSubject, iro::scope::block>;
using SyncWarp = iro::token::sync_at<TestSubject, iro::scope::warp>;

static_assert(iro::verify::token_satisfies<SyncBlock, SyncWarp>());

// version tokens
using Ver1 = iro::token::version<TestSubject, 1>;
using Ver1b = iro::token::version<TestSubject, 1>;
using Ver2 = iro::token::version<TestSubject, 2>;
static_assert(iro::verify::token_satisfies<Ver1, Ver1b>());
static_assert(!iro::verify::token_satisfies<Ver1, Ver2>());

}  // namespace token_test

// =============================================================================
// Verify iro::contract::Tile
// =============================================================================

namespace tile_test {

using TestShape = iro::contract::Shape<64, 64>;
using TestElem = iro::elem::f16;
using TestLayout = iro::contract::layout::RowMajor<64>;
using TestSpace = iro::contract::space::shared;
using TestAlign = iro::contract::Align<16>;

using TestTile = iro::contract::Tile<
    TestShape, TestElem, TestLayout, TestSpace, TestAlign
>;

static_assert(TestTile::bytes == 64 * 64 * 2);  // 8KB
static_assert(std::is_same_v<TestTile::shape, TestShape>);
static_assert(std::is_same_v<TestTile::elem, TestElem>);

}  // namespace tile_test

// =============================================================================
// Verify iro::contract::res
// =============================================================================

namespace res_test {

struct PipelineTag { static constexpr auto id = iro::util::fnv1a_64_cstr("test.pipeline"); };
struct RegionTag { static constexpr auto id = iro::util::fnv1a_64_cstr("test.region"); };

using TestPipeline = iro::contract::res::smem_pipeline<PipelineTag, 4, 1024, 128>;
static_assert(TestPipeline::slots == 4);
static_assert(TestPipeline::bytes_per_slot == 1024);
static_assert(TestPipeline::total_bytes == 4096);

using TestRegion = iro::contract::res::smem_region<RegionTag, 2048, 64>;
static_assert(TestRegion::bytes == 2048);
static_assert(TestRegion::align == 64);

}  // namespace res_test

// =============================================================================
// Verify iro::cap
// =============================================================================

static_assert(iro::cap::sm89::sm_version == 89);
static_assert(iro::cap::sm90::sm_version == 90);
static_assert(iro::cap::sm100::sm_version == 100);

static_assert(iro::cap::sm90::has_wgmma);
static_assert(iro::cap::sm90::has_tma);
static_assert(!iro::cap::sm89::has_wgmma);
static_assert(!iro::cap::sm89::has_f16_atomics);
static_assert(iro::cap::sm90::has_f16_atomics);

static_assert(iro::cap::cap_supports<iro::cap::sm90, iro::cap::sm89>());
static_assert(!iro::cap::cap_supports<iro::cap::sm89, iro::cap::sm90>());

// =============================================================================
// Verify iro::schema concepts
// =============================================================================

static_assert(iro::schema::ElemTag<iro::elem::f16>);
static_assert(iro::schema::ElemTag<iro::elem::f32>);
static_assert(iro::schema::ElemTag<iro::elem::bf16>);

static_assert(iro::schema::Layout<iro::contract::layout::Contiguous>);
static_assert(iro::schema::Layout<iro::contract::layout::RowMajor<64>>);
static_assert(iro::schema::Layout<iro::contract::layout::ColMajor<64>>);

// =============================================================================
// Verify iro::diag
// =============================================================================

static_assert(iro::diag::OK::is_error == false);
static_assert(iro::diag::Diagnostic<iro::diag::Code::MISSING_VISIBLE_AT>::is_error == true);
static_assert(iro::diag::Diagnostic<iro::diag::Code::MISSING_SYNC_AT>::is_error == true);

// =============================================================================
// Verify token list canonicality (§4.3)
// =============================================================================

namespace token_canonical_test {

struct Subject1 { static constexpr auto id = iro::util::fnv1a_64_cstr("canonical.subject1"); };
struct Subject2 { static constexpr auto id = iro::util::fnv1a_64_cstr("canonical.subject2"); };

using Tok1 = iro::token::visible_at<Subject1, iro::scope::warp>;
using Tok2 = iro::token::alive<Subject1, iro::token::lifetime::block>;
using Tok3 = iro::token::visible_at<Subject1, iro::scope::block>;  // Same kind+subject as Tok1
using Tok4 = iro::token::visible_at<Subject2, iro::scope::warp>;  // Different subject

// Canonical: different kind+subject pairs
static_assert(iro::verify::token_list_canonical<iro::util::type_list<Tok1, Tok2>>());
static_assert(iro::verify::token_list_canonical<iro::util::type_list<Tok1, Tok4>>());
static_assert(iro::verify::token_list_canonical<iro::util::type_list<Tok1, Tok2, Tok4>>());

// Non-canonical: duplicate kind+subject (both visible_at<Subject1>)
static_assert(!iro::verify::token_list_canonical<iro::util::type_list<Tok1, Tok3>>());
static_assert(!iro::verify::token_list_canonical<iro::util::type_list<Tok1, Tok2, Tok3>>());

}  // namespace token_canonical_test

// =============================================================================
// Verify resource list canonicality (§5.1)
// =============================================================================

namespace resource_canonical_test {

struct Tag1 { static constexpr auto id = iro::util::fnv1a_64_cstr("res.tag1"); };
struct Tag2 { static constexpr auto id = iro::util::fnv1a_64_cstr("res.tag2"); };

using Res1 = iro::contract::res::smem_region<Tag1, 1024, 16>;
using Res2 = iro::contract::res::smem_region<Tag2, 1024, 16>;
using Res1Dup = iro::contract::res::smem_region<Tag1, 1024, 16>;  // Same as Res1

// Canonical: different ids
static_assert(iro::verify::resource_list_canonical<iro::util::type_list<Res1, Res2>>());

// Non-canonical: duplicate ids
static_assert(!iro::verify::resource_list_canonical<iro::util::type_list<Res1, Res1Dup>>());

}  // namespace resource_canonical_test

// =============================================================================
// Verify graph hash ordering invariance (manifest v2 contract)
// =============================================================================

namespace graph_hash_order_test {

using RecipeF32 = iro::recipe::Precision<iro::elem::f32, iro::elem::f32, iro::elem::f32, 4, iro::recipe::Exact>;
using ScalarF32 = iro::contract::ScalarDesc<iro::elem::f32, iro::dist::lane>;

struct SubjX { static constexpr auto id = iro::util::fnv1a_64_cstr("graph_hash_order.subj_x"); };
struct SubjY { static constexpr auto id = iro::util::fnv1a_64_cstr("graph_hash_order.subj_y"); };

using SrcOut = iro::contract::OutputPort<
    ScalarF32, SubjX, iro::exec::warp, iro::util::type_list<>, iro::dist::lane, RecipeF32>;
using MidIn = iro::contract::InputPort<
    ScalarF32, SubjX, iro::exec::warp, iro::util::type_list<>, iro::dist::lane, RecipeF32>;
using MidOut = iro::contract::OutputPort<
    ScalarF32, SubjY, iro::exec::warp, iro::util::type_list<>, iro::dist::lane, RecipeF32>;
using DstIn = iro::contract::InputPort<
    ScalarF32, SubjY, iro::exec::warp, iro::util::type_list<>, iro::dist::lane, RecipeF32>;

using Src = iro::contract::Obligation<
    iro::util::type_list<>,
    iro::util::type_list<SrcOut>,
    iro::util::type_list<>
>;
using Mid = iro::contract::Obligation<
    iro::util::type_list<MidIn>,
    iro::util::type_list<MidOut>,
    iro::util::type_list<>
>;
using Dst = iro::contract::Obligation<
    iro::util::type_list<DstIn>,
    iro::util::type_list<>,
    iro::util::type_list<>
>;

using EdgeSrcMid = iro::compose::Edge<
    iro::compose::out_port_ref<Src, 0>,
    iro::compose::in_port_ref<Mid, 0>
>;
using EdgeMidDst = iro::compose::Edge<
    iro::compose::out_port_ref<Mid, 0>,
    iro::compose::in_port_ref<Dst, 0>
>;

using GraphAB = iro::compose::CompositionAutoResources<
    iro::util::type_list<Src, Mid, Dst>,
    iro::util::type_list<EdgeSrcMid, EdgeMidDst>,
    iro::profile::BudgetMax,
    iro::cap::sm90
>;

using GraphPermuted = iro::compose::CompositionAutoResources<
    iro::util::type_list<Dst, Src, Mid>,
    iro::util::type_list<EdgeMidDst, EdgeSrcMid>,
    iro::profile::BudgetMax,
    iro::cap::sm90
>;

static_assert(axp::graph::verify_structure<GraphAB>());
static_assert(axp::graph::verify_contract_flow<GraphAB>());
static_assert(axp::graph::verify<GraphAB>());
static_assert(axp::graph::graph_hash_v<GraphAB> == axp::graph::graph_hash_v<GraphPermuted>);

} // namespace graph_hash_order_test
