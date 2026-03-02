#pragma once
// Generated file. Upstream-generated WGMMA entry points for SM90 realization.

namespace axp::realize::sm90::detail {

__device__ __forceinline__ void wgmma_m64n8k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3];
    asm volatile(
        "wgmma.mma_async.aligned.m64n8k16.f16.f16.f32 "
        "{%0,%1,%2,%3}, %4, %5, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n16k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7];
    asm volatile(
        "wgmma.mma_async.aligned.m64n16k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n24k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11];
    asm volatile(
        "wgmma.mma_async.aligned.m64n24k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, %12, %13, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n32k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15];
    asm volatile(
        "wgmma.mma_async.aligned.m64n32k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n40k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19];
    asm volatile(
        "wgmma.mma_async.aligned.m64n40k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, %20, %21, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n48k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23];
    asm volatile(
        "wgmma.mma_async.aligned.m64n48k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n56k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27];
    asm volatile(
        "wgmma.mma_async.aligned.m64n56k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, %28, %29, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n64k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31];
    asm volatile(
        "wgmma.mma_async.aligned.m64n64k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n72k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35];
    asm volatile(
        "wgmma.mma_async.aligned.m64n72k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, %36, %37, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n80k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39];
    asm volatile(
        "wgmma.mma_async.aligned.m64n80k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n88k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43];
    asm volatile(
        "wgmma.mma_async.aligned.m64n88k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43}, %44, %45, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n96k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47];
    asm volatile(
        "wgmma.mma_async.aligned.m64n96k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n104k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51];
    asm volatile(
        "wgmma.mma_async.aligned.m64n104k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %52, %53, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n112k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55];
    asm volatile(
        "wgmma.mma_async.aligned.m64n112k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n120k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59];
    asm volatile(
        "wgmma.mma_async.aligned.m64n120k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59}, %60, %61, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n128k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63];
    asm volatile(
        "wgmma.mma_async.aligned.m64n128k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n136k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67];
    asm volatile(
        "wgmma.mma_async.aligned.m64n136k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67}, %68, %69, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n144k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71];
    asm volatile(
        "wgmma.mma_async.aligned.m64n144k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, %72, %73, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n152k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75];
    asm volatile(
        "wgmma.mma_async.aligned.m64n152k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75}, %76, %77, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n160k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79];
    asm volatile(
        "wgmma.mma_async.aligned.m64n160k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79}, %80, %81, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n168k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83];
    asm volatile(
        "wgmma.mma_async.aligned.m64n168k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83}, %84, %85, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n176k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87];
    asm volatile(
        "wgmma.mma_async.aligned.m64n176k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87}, %88, %89, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n184k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91];
    asm volatile(
        "wgmma.mma_async.aligned.m64n184k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91}, %92, %93, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n192k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95];
    asm volatile(
        "wgmma.mma_async.aligned.m64n192k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n200k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99];
    asm volatile(
        "wgmma.mma_async.aligned.m64n200k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99}, %100, %101, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n208k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103];
    asm volatile(
        "wgmma.mma_async.aligned.m64n208k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103}, %104, %105, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n216k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107];
    asm volatile(
        "wgmma.mma_async.aligned.m64n216k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107}, %108, %109, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n224k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111];
    asm volatile(
        "wgmma.mma_async.aligned.m64n224k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111}, %112, %113, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n232k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115];
    asm volatile(
        "wgmma.mma_async.aligned.m64n232k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115}, %116, %117, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n240k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119];
    asm volatile(
        "wgmma.mma_async.aligned.m64n240k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119}, %120, %121, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n248k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123];
    asm volatile(
        "wgmma.mma_async.aligned.m64n248k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123}, %124, %125, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n256k16_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123], d124 = acc[124], d125 = acc[125], d126 = acc[126], d127 = acc[127];
    asm volatile(
        "wgmma.mma_async.aligned.m64n256k16.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123), "+f"(d124), "+f"(d125), "+f"(d126), "+f"(d127)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
    acc[124] = d124;
    acc[125] = d125;
    acc[126] = d126;
    acc[127] = d127;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n8k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3];
    asm volatile(
        "wgmma.mma_async.aligned.m64n8k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, %4, %5, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n16k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7];
    asm volatile(
        "wgmma.mma_async.aligned.m64n16k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n24k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11];
    asm volatile(
        "wgmma.mma_async.aligned.m64n24k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, %12, %13, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n32k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15];
    asm volatile(
        "wgmma.mma_async.aligned.m64n32k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n40k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19];
    asm volatile(
        "wgmma.mma_async.aligned.m64n40k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, %20, %21, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n48k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23];
    asm volatile(
        "wgmma.mma_async.aligned.m64n48k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n56k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27];
    asm volatile(
        "wgmma.mma_async.aligned.m64n56k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, %28, %29, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n64k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31];
    asm volatile(
        "wgmma.mma_async.aligned.m64n64k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n72k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35];
    asm volatile(
        "wgmma.mma_async.aligned.m64n72k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, %36, %37, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n80k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39];
    asm volatile(
        "wgmma.mma_async.aligned.m64n80k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n88k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43];
    asm volatile(
        "wgmma.mma_async.aligned.m64n88k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43}, %44, %45, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n96k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47];
    asm volatile(
        "wgmma.mma_async.aligned.m64n96k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n104k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51];
    asm volatile(
        "wgmma.mma_async.aligned.m64n104k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %52, %53, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n112k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55];
    asm volatile(
        "wgmma.mma_async.aligned.m64n112k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n120k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59];
    asm volatile(
        "wgmma.mma_async.aligned.m64n120k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59}, %60, %61, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n128k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63];
    asm volatile(
        "wgmma.mma_async.aligned.m64n128k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n136k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67];
    asm volatile(
        "wgmma.mma_async.aligned.m64n136k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67}, %68, %69, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n144k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71];
    asm volatile(
        "wgmma.mma_async.aligned.m64n144k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, %72, %73, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n152k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75];
    asm volatile(
        "wgmma.mma_async.aligned.m64n152k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75}, %76, %77, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n160k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79];
    asm volatile(
        "wgmma.mma_async.aligned.m64n160k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79}, %80, %81, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n168k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83];
    asm volatile(
        "wgmma.mma_async.aligned.m64n168k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83}, %84, %85, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n176k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87];
    asm volatile(
        "wgmma.mma_async.aligned.m64n176k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87}, %88, %89, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n184k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91];
    asm volatile(
        "wgmma.mma_async.aligned.m64n184k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91}, %92, %93, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n192k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95];
    asm volatile(
        "wgmma.mma_async.aligned.m64n192k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n200k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99];
    asm volatile(
        "wgmma.mma_async.aligned.m64n200k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99}, %100, %101, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n208k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103];
    asm volatile(
        "wgmma.mma_async.aligned.m64n208k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103}, %104, %105, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n216k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107];
    asm volatile(
        "wgmma.mma_async.aligned.m64n216k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107}, %108, %109, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n224k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111];
    asm volatile(
        "wgmma.mma_async.aligned.m64n224k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111}, %112, %113, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n232k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115];
    asm volatile(
        "wgmma.mma_async.aligned.m64n232k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115}, %116, %117, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n240k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119];
    asm volatile(
        "wgmma.mma_async.aligned.m64n240k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119}, %120, %121, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n248k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123];
    asm volatile(
        "wgmma.mma_async.aligned.m64n248k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123}, %124, %125, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n256k16_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123], d124 = acc[124], d125 = acc[125], d126 = acc[126], d127 = acc[127];
    asm volatile(
        "wgmma.mma_async.aligned.m64n256k16.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123), "+f"(d124), "+f"(d125), "+f"(d126), "+f"(d127)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
    acc[124] = d124;
    acc[125] = d125;
    acc[126] = d126;
    acc[127] = d127;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n8k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3];
    asm volatile(
        "wgmma.mma_async.aligned.m64n8k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, %4, %5, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n16k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7];
    asm volatile(
        "wgmma.mma_async.aligned.m64n16k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n24k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11];
    asm volatile(
        "wgmma.mma_async.aligned.m64n24k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, %12, %13, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n32k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15];
    asm volatile(
        "wgmma.mma_async.aligned.m64n32k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n40k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19];
    asm volatile(
        "wgmma.mma_async.aligned.m64n40k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, %20, %21, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n48k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23];
    asm volatile(
        "wgmma.mma_async.aligned.m64n48k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n56k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27];
    asm volatile(
        "wgmma.mma_async.aligned.m64n56k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, %28, %29, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n64k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31];
    asm volatile(
        "wgmma.mma_async.aligned.m64n64k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n72k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35];
    asm volatile(
        "wgmma.mma_async.aligned.m64n72k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, %36, %37, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n80k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39];
    asm volatile(
        "wgmma.mma_async.aligned.m64n80k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n88k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43];
    asm volatile(
        "wgmma.mma_async.aligned.m64n88k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43}, %44, %45, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n96k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47];
    asm volatile(
        "wgmma.mma_async.aligned.m64n96k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n104k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51];
    asm volatile(
        "wgmma.mma_async.aligned.m64n104k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %52, %53, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n112k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55];
    asm volatile(
        "wgmma.mma_async.aligned.m64n112k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n120k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59];
    asm volatile(
        "wgmma.mma_async.aligned.m64n120k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59}, %60, %61, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n128k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63];
    asm volatile(
        "wgmma.mma_async.aligned.m64n128k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n136k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67];
    asm volatile(
        "wgmma.mma_async.aligned.m64n136k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67}, %68, %69, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n144k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71];
    asm volatile(
        "wgmma.mma_async.aligned.m64n144k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, %72, %73, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n152k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75];
    asm volatile(
        "wgmma.mma_async.aligned.m64n152k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75}, %76, %77, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n160k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79];
    asm volatile(
        "wgmma.mma_async.aligned.m64n160k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79}, %80, %81, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n168k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83];
    asm volatile(
        "wgmma.mma_async.aligned.m64n168k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83}, %84, %85, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n176k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87];
    asm volatile(
        "wgmma.mma_async.aligned.m64n176k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87}, %88, %89, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n184k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91];
    asm volatile(
        "wgmma.mma_async.aligned.m64n184k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91}, %92, %93, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n192k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95];
    asm volatile(
        "wgmma.mma_async.aligned.m64n192k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n200k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99];
    asm volatile(
        "wgmma.mma_async.aligned.m64n200k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99}, %100, %101, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n208k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103];
    asm volatile(
        "wgmma.mma_async.aligned.m64n208k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103}, %104, %105, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n216k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107];
    asm volatile(
        "wgmma.mma_async.aligned.m64n216k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107}, %108, %109, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n224k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111];
    asm volatile(
        "wgmma.mma_async.aligned.m64n224k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111}, %112, %113, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n232k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115];
    asm volatile(
        "wgmma.mma_async.aligned.m64n232k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115}, %116, %117, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n240k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119];
    asm volatile(
        "wgmma.mma_async.aligned.m64n240k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119}, %120, %121, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n248k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123];
    asm volatile(
        "wgmma.mma_async.aligned.m64n248k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123}, %124, %125, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n256k8_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123], d124 = acc[124], d125 = acc[125], d126 = acc[126], d127 = acc[127];
    asm volatile(
        "wgmma.mma_async.aligned.m64n256k8.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123), "+f"(d124), "+f"(d125), "+f"(d126), "+f"(d127)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
    acc[124] = d124;
    acc[125] = d125;
    acc[126] = d126;
    acc[127] = d127;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n8k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3];
    asm volatile(
        "wgmma.mma_async.aligned.m64n8k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, %4, %5, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n16k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7];
    asm volatile(
        "wgmma.mma_async.aligned.m64n16k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n24k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11];
    asm volatile(
        "wgmma.mma_async.aligned.m64n24k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, %12, %13, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n32k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15];
    asm volatile(
        "wgmma.mma_async.aligned.m64n32k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n40k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19];
    asm volatile(
        "wgmma.mma_async.aligned.m64n40k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, %20, %21, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n48k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23];
    asm volatile(
        "wgmma.mma_async.aligned.m64n48k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n56k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27];
    asm volatile(
        "wgmma.mma_async.aligned.m64n56k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, %28, %29, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n64k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31];
    asm volatile(
        "wgmma.mma_async.aligned.m64n64k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n72k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35];
    asm volatile(
        "wgmma.mma_async.aligned.m64n72k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, %36, %37, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n80k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39];
    asm volatile(
        "wgmma.mma_async.aligned.m64n80k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n88k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43];
    asm volatile(
        "wgmma.mma_async.aligned.m64n88k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43}, %44, %45, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n96k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47];
    asm volatile(
        "wgmma.mma_async.aligned.m64n96k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n104k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51];
    asm volatile(
        "wgmma.mma_async.aligned.m64n104k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %52, %53, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n112k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55];
    asm volatile(
        "wgmma.mma_async.aligned.m64n112k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n120k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59];
    asm volatile(
        "wgmma.mma_async.aligned.m64n120k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59}, %60, %61, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n128k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63];
    asm volatile(
        "wgmma.mma_async.aligned.m64n128k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n136k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67];
    asm volatile(
        "wgmma.mma_async.aligned.m64n136k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67}, %68, %69, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n144k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71];
    asm volatile(
        "wgmma.mma_async.aligned.m64n144k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, %72, %73, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n152k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75];
    asm volatile(
        "wgmma.mma_async.aligned.m64n152k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75}, %76, %77, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n160k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79];
    asm volatile(
        "wgmma.mma_async.aligned.m64n160k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79}, %80, %81, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n168k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83];
    asm volatile(
        "wgmma.mma_async.aligned.m64n168k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83}, %84, %85, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n176k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87];
    asm volatile(
        "wgmma.mma_async.aligned.m64n176k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87}, %88, %89, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n184k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91];
    asm volatile(
        "wgmma.mma_async.aligned.m64n184k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91}, %92, %93, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n192k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95];
    asm volatile(
        "wgmma.mma_async.aligned.m64n192k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n200k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99];
    asm volatile(
        "wgmma.mma_async.aligned.m64n200k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99}, %100, %101, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n208k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103];
    asm volatile(
        "wgmma.mma_async.aligned.m64n208k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103}, %104, %105, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n216k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107];
    asm volatile(
        "wgmma.mma_async.aligned.m64n216k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107}, %108, %109, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n224k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111];
    asm volatile(
        "wgmma.mma_async.aligned.m64n224k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111}, %112, %113, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n232k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115];
    asm volatile(
        "wgmma.mma_async.aligned.m64n232k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115}, %116, %117, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n240k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119];
    asm volatile(
        "wgmma.mma_async.aligned.m64n240k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119}, %120, %121, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n248k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123];
    asm volatile(
        "wgmma.mma_async.aligned.m64n248k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123}, %124, %125, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n256k32_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123], d124 = acc[124], d125 = acc[125], d126 = acc[126], d127 = acc[127];
    asm volatile(
        "wgmma.mma_async.aligned.m64n256k32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123), "+f"(d124), "+f"(d125), "+f"(d126), "+f"(d127)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
    acc[124] = d124;
    acc[125] = d125;
    acc[126] = d126;
    acc[127] = d127;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n8k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3];
    asm volatile(
        "wgmma.mma_async.aligned.m64n8k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3}, %4, %5, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n16k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7];
    asm volatile(
        "wgmma.mma_async.aligned.m64n16k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n24k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11];
    asm volatile(
        "wgmma.mma_async.aligned.m64n24k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, %12, %13, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n32k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15];
    asm volatile(
        "wgmma.mma_async.aligned.m64n32k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n40k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19];
    asm volatile(
        "wgmma.mma_async.aligned.m64n40k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, %20, %21, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n48k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23];
    asm volatile(
        "wgmma.mma_async.aligned.m64n48k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n56k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27];
    asm volatile(
        "wgmma.mma_async.aligned.m64n56k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, %28, %29, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n64k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31];
    asm volatile(
        "wgmma.mma_async.aligned.m64n64k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n72k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35];
    asm volatile(
        "wgmma.mma_async.aligned.m64n72k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, %36, %37, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n80k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39];
    asm volatile(
        "wgmma.mma_async.aligned.m64n80k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n88k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43];
    asm volatile(
        "wgmma.mma_async.aligned.m64n88k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43}, %44, %45, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n96k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47];
    asm volatile(
        "wgmma.mma_async.aligned.m64n96k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n104k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51];
    asm volatile(
        "wgmma.mma_async.aligned.m64n104k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %52, %53, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n112k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55];
    asm volatile(
        "wgmma.mma_async.aligned.m64n112k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n120k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59];
    asm volatile(
        "wgmma.mma_async.aligned.m64n120k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59}, %60, %61, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n128k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63];
    asm volatile(
        "wgmma.mma_async.aligned.m64n128k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n136k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67];
    asm volatile(
        "wgmma.mma_async.aligned.m64n136k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67}, %68, %69, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n144k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71];
    asm volatile(
        "wgmma.mma_async.aligned.m64n144k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, %72, %73, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n152k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75];
    asm volatile(
        "wgmma.mma_async.aligned.m64n152k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75}, %76, %77, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n160k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79];
    asm volatile(
        "wgmma.mma_async.aligned.m64n160k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79}, %80, %81, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n168k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83];
    asm volatile(
        "wgmma.mma_async.aligned.m64n168k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83}, %84, %85, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n176k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87];
    asm volatile(
        "wgmma.mma_async.aligned.m64n176k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87}, %88, %89, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n184k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91];
    asm volatile(
        "wgmma.mma_async.aligned.m64n184k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91}, %92, %93, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n192k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95];
    asm volatile(
        "wgmma.mma_async.aligned.m64n192k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n200k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99];
    asm volatile(
        "wgmma.mma_async.aligned.m64n200k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99}, %100, %101, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n208k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103];
    asm volatile(
        "wgmma.mma_async.aligned.m64n208k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103}, %104, %105, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n216k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107];
    asm volatile(
        "wgmma.mma_async.aligned.m64n216k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107}, %108, %109, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n224k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111];
    asm volatile(
        "wgmma.mma_async.aligned.m64n224k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111}, %112, %113, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n232k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115];
    asm volatile(
        "wgmma.mma_async.aligned.m64n232k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115}, %116, %117, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n240k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119];
    asm volatile(
        "wgmma.mma_async.aligned.m64n240k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119}, %120, %121, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n248k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123];
    asm volatile(
        "wgmma.mma_async.aligned.m64n248k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123}, %124, %125, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n256k32_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123], d124 = acc[124], d125 = acc[125], d126 = acc[126], d127 = acc[127];
    asm volatile(
        "wgmma.mma_async.aligned.m64n256k32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123), "+f"(d124), "+f"(d125), "+f"(d126), "+f"(d127)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
    acc[124] = d124;
    acc[125] = d125;
    acc[126] = d126;
    acc[127] = d127;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n8k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3];
    asm volatile(
        "wgmma.mma_async.aligned.m64n8k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3}, %4, %5, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n16k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7];
    asm volatile(
        "wgmma.mma_async.aligned.m64n16k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n24k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11];
    asm volatile(
        "wgmma.mma_async.aligned.m64n24k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, %12, %13, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n32k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15];
    asm volatile(
        "wgmma.mma_async.aligned.m64n32k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n40k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19];
    asm volatile(
        "wgmma.mma_async.aligned.m64n40k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, %20, %21, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n48k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23];
    asm volatile(
        "wgmma.mma_async.aligned.m64n48k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n56k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27];
    asm volatile(
        "wgmma.mma_async.aligned.m64n56k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, %28, %29, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n64k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31];
    asm volatile(
        "wgmma.mma_async.aligned.m64n64k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n72k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35];
    asm volatile(
        "wgmma.mma_async.aligned.m64n72k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, %36, %37, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n80k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39];
    asm volatile(
        "wgmma.mma_async.aligned.m64n80k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n88k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43];
    asm volatile(
        "wgmma.mma_async.aligned.m64n88k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43}, %44, %45, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n96k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47];
    asm volatile(
        "wgmma.mma_async.aligned.m64n96k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n104k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51];
    asm volatile(
        "wgmma.mma_async.aligned.m64n104k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %52, %53, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n112k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55];
    asm volatile(
        "wgmma.mma_async.aligned.m64n112k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n120k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59];
    asm volatile(
        "wgmma.mma_async.aligned.m64n120k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59}, %60, %61, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n128k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63];
    asm volatile(
        "wgmma.mma_async.aligned.m64n128k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n136k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67];
    asm volatile(
        "wgmma.mma_async.aligned.m64n136k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67}, %68, %69, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n144k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71];
    asm volatile(
        "wgmma.mma_async.aligned.m64n144k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, %72, %73, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n152k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75];
    asm volatile(
        "wgmma.mma_async.aligned.m64n152k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75}, %76, %77, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n160k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79];
    asm volatile(
        "wgmma.mma_async.aligned.m64n160k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79}, %80, %81, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n168k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83];
    asm volatile(
        "wgmma.mma_async.aligned.m64n168k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83}, %84, %85, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n176k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87];
    asm volatile(
        "wgmma.mma_async.aligned.m64n176k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87}, %88, %89, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n184k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91];
    asm volatile(
        "wgmma.mma_async.aligned.m64n184k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91}, %92, %93, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n192k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95];
    asm volatile(
        "wgmma.mma_async.aligned.m64n192k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n200k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99];
    asm volatile(
        "wgmma.mma_async.aligned.m64n200k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99}, %100, %101, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n208k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103];
    asm volatile(
        "wgmma.mma_async.aligned.m64n208k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103}, %104, %105, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n216k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107];
    asm volatile(
        "wgmma.mma_async.aligned.m64n216k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107}, %108, %109, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n224k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111];
    asm volatile(
        "wgmma.mma_async.aligned.m64n224k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111}, %112, %113, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n232k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115];
    asm volatile(
        "wgmma.mma_async.aligned.m64n232k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115}, %116, %117, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n240k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119];
    asm volatile(
        "wgmma.mma_async.aligned.m64n240k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119}, %120, %121, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n248k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123];
    asm volatile(
        "wgmma.mma_async.aligned.m64n248k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123}, %124, %125, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n256k32_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123], d124 = acc[124], d125 = acc[125], d126 = acc[126], d127 = acc[127];
    asm volatile(
        "wgmma.mma_async.aligned.m64n256k32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123), "+f"(d124), "+f"(d125), "+f"(d126), "+f"(d127)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
    acc[124] = d124;
    acc[125] = d125;
    acc[126] = d126;
    acc[127] = d127;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n8k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3];
    asm volatile(
        "wgmma.mma_async.aligned.m64n8k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3}, %4, %5, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n16k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7];
    asm volatile(
        "wgmma.mma_async.aligned.m64n16k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n24k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11];
    asm volatile(
        "wgmma.mma_async.aligned.m64n24k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, %12, %13, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n32k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15];
    asm volatile(
        "wgmma.mma_async.aligned.m64n32k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n40k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19];
    asm volatile(
        "wgmma.mma_async.aligned.m64n40k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, %20, %21, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n48k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23];
    asm volatile(
        "wgmma.mma_async.aligned.m64n48k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n56k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27];
    asm volatile(
        "wgmma.mma_async.aligned.m64n56k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, %28, %29, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n64k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31];
    asm volatile(
        "wgmma.mma_async.aligned.m64n64k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n72k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35];
    asm volatile(
        "wgmma.mma_async.aligned.m64n72k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, %36, %37, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n80k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39];
    asm volatile(
        "wgmma.mma_async.aligned.m64n80k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n88k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43];
    asm volatile(
        "wgmma.mma_async.aligned.m64n88k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43}, %44, %45, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n96k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47];
    asm volatile(
        "wgmma.mma_async.aligned.m64n96k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, %48, %49, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n104k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51];
    asm volatile(
        "wgmma.mma_async.aligned.m64n104k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, %52, %53, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n112k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55];
    asm volatile(
        "wgmma.mma_async.aligned.m64n112k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n120k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59];
    asm volatile(
        "wgmma.mma_async.aligned.m64n120k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59}, %60, %61, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n128k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63];
    asm volatile(
        "wgmma.mma_async.aligned.m64n128k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n136k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67];
    asm volatile(
        "wgmma.mma_async.aligned.m64n136k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67}, %68, %69, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n144k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71];
    asm volatile(
        "wgmma.mma_async.aligned.m64n144k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71}, %72, %73, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n152k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75];
    asm volatile(
        "wgmma.mma_async.aligned.m64n152k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75}, %76, %77, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n160k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79];
    asm volatile(
        "wgmma.mma_async.aligned.m64n160k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79}, %80, %81, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n168k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83];
    asm volatile(
        "wgmma.mma_async.aligned.m64n168k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83}, %84, %85, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n176k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87];
    asm volatile(
        "wgmma.mma_async.aligned.m64n176k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87}, %88, %89, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n184k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91];
    asm volatile(
        "wgmma.mma_async.aligned.m64n184k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91}, %92, %93, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n192k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95];
    asm volatile(
        "wgmma.mma_async.aligned.m64n192k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, %96, %97, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n200k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99];
    asm volatile(
        "wgmma.mma_async.aligned.m64n200k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99}, %100, %101, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n208k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103];
    asm volatile(
        "wgmma.mma_async.aligned.m64n208k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103}, %104, %105, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n216k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107];
    asm volatile(
        "wgmma.mma_async.aligned.m64n216k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107}, %108, %109, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n224k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111];
    asm volatile(
        "wgmma.mma_async.aligned.m64n224k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111}, %112, %113, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n232k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115];
    asm volatile(
        "wgmma.mma_async.aligned.m64n232k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115}, %116, %117, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n240k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119];
    asm volatile(
        "wgmma.mma_async.aligned.m64n240k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119}, %120, %121, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n248k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123];
    asm volatile(
        "wgmma.mma_async.aligned.m64n248k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123}, %124, %125, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

__device__ __forceinline__ void wgmma_m64n256k32_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3], d4 = acc[4], d5 = acc[5], d6 = acc[6], d7 = acc[7], d8 = acc[8], d9 = acc[9], d10 = acc[10], d11 = acc[11], d12 = acc[12], d13 = acc[13], d14 = acc[14], d15 = acc[15], d16 = acc[16], d17 = acc[17], d18 = acc[18], d19 = acc[19], d20 = acc[20], d21 = acc[21], d22 = acc[22], d23 = acc[23], d24 = acc[24], d25 = acc[25], d26 = acc[26], d27 = acc[27], d28 = acc[28], d29 = acc[29], d30 = acc[30], d31 = acc[31], d32 = acc[32], d33 = acc[33], d34 = acc[34], d35 = acc[35], d36 = acc[36], d37 = acc[37], d38 = acc[38], d39 = acc[39], d40 = acc[40], d41 = acc[41], d42 = acc[42], d43 = acc[43], d44 = acc[44], d45 = acc[45], d46 = acc[46], d47 = acc[47], d48 = acc[48], d49 = acc[49], d50 = acc[50], d51 = acc[51], d52 = acc[52], d53 = acc[53], d54 = acc[54], d55 = acc[55], d56 = acc[56], d57 = acc[57], d58 = acc[58], d59 = acc[59], d60 = acc[60], d61 = acc[61], d62 = acc[62], d63 = acc[63], d64 = acc[64], d65 = acc[65], d66 = acc[66], d67 = acc[67], d68 = acc[68], d69 = acc[69], d70 = acc[70], d71 = acc[71], d72 = acc[72], d73 = acc[73], d74 = acc[74], d75 = acc[75], d76 = acc[76], d77 = acc[77], d78 = acc[78], d79 = acc[79], d80 = acc[80], d81 = acc[81], d82 = acc[82], d83 = acc[83], d84 = acc[84], d85 = acc[85], d86 = acc[86], d87 = acc[87], d88 = acc[88], d89 = acc[89], d90 = acc[90], d91 = acc[91], d92 = acc[92], d93 = acc[93], d94 = acc[94], d95 = acc[95], d96 = acc[96], d97 = acc[97], d98 = acc[98], d99 = acc[99], d100 = acc[100], d101 = acc[101], d102 = acc[102], d103 = acc[103], d104 = acc[104], d105 = acc[105], d106 = acc[106], d107 = acc[107], d108 = acc[108], d109 = acc[109], d110 = acc[110], d111 = acc[111], d112 = acc[112], d113 = acc[113], d114 = acc[114], d115 = acc[115], d116 = acc[116], d117 = acc[117], d118 = acc[118], d119 = acc[119], d120 = acc[120], d121 = acc[121], d122 = acc[122], d123 = acc[123], d124 = acc[124], d125 = acc[125], d126 = acc[126], d127 = acc[127];
    asm volatile(
        "wgmma.mma_async.aligned.m64n256k32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35), "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47), "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59), "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63), "+f"(d64), "+f"(d65), "+f"(d66), "+f"(d67), "+f"(d68), "+f"(d69), "+f"(d70), "+f"(d71), "+f"(d72), "+f"(d73), "+f"(d74), "+f"(d75), "+f"(d76), "+f"(d77), "+f"(d78), "+f"(d79), "+f"(d80), "+f"(d81), "+f"(d82), "+f"(d83), "+f"(d84), "+f"(d85), "+f"(d86), "+f"(d87), "+f"(d88), "+f"(d89), "+f"(d90), "+f"(d91), "+f"(d92), "+f"(d93), "+f"(d94), "+f"(d95), "+f"(d96), "+f"(d97), "+f"(d98), "+f"(d99), "+f"(d100), "+f"(d101), "+f"(d102), "+f"(d103), "+f"(d104), "+f"(d105), "+f"(d106), "+f"(d107), "+f"(d108), "+f"(d109), "+f"(d110), "+f"(d111), "+f"(d112), "+f"(d113), "+f"(d114), "+f"(d115), "+f"(d116), "+f"(d117), "+f"(d118), "+f"(d119), "+f"(d120), "+f"(d121), "+f"(d122), "+f"(d123), "+f"(d124), "+f"(d125), "+f"(d126), "+f"(d127)
        : "l"(desc_a), "l"(desc_b));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
    acc[4] = d4;
    acc[5] = d5;
    acc[6] = d6;
    acc[7] = d7;
    acc[8] = d8;
    acc[9] = d9;
    acc[10] = d10;
    acc[11] = d11;
    acc[12] = d12;
    acc[13] = d13;
    acc[14] = d14;
    acc[15] = d15;
    acc[16] = d16;
    acc[17] = d17;
    acc[18] = d18;
    acc[19] = d19;
    acc[20] = d20;
    acc[21] = d21;
    acc[22] = d22;
    acc[23] = d23;
    acc[24] = d24;
    acc[25] = d25;
    acc[26] = d26;
    acc[27] = d27;
    acc[28] = d28;
    acc[29] = d29;
    acc[30] = d30;
    acc[31] = d31;
    acc[32] = d32;
    acc[33] = d33;
    acc[34] = d34;
    acc[35] = d35;
    acc[36] = d36;
    acc[37] = d37;
    acc[38] = d38;
    acc[39] = d39;
    acc[40] = d40;
    acc[41] = d41;
    acc[42] = d42;
    acc[43] = d43;
    acc[44] = d44;
    acc[45] = d45;
    acc[46] = d46;
    acc[47] = d47;
    acc[48] = d48;
    acc[49] = d49;
    acc[50] = d50;
    acc[51] = d51;
    acc[52] = d52;
    acc[53] = d53;
    acc[54] = d54;
    acc[55] = d55;
    acc[56] = d56;
    acc[57] = d57;
    acc[58] = d58;
    acc[59] = d59;
    acc[60] = d60;
    acc[61] = d61;
    acc[62] = d62;
    acc[63] = d63;
    acc[64] = d64;
    acc[65] = d65;
    acc[66] = d66;
    acc[67] = d67;
    acc[68] = d68;
    acc[69] = d69;
    acc[70] = d70;
    acc[71] = d71;
    acc[72] = d72;
    acc[73] = d73;
    acc[74] = d74;
    acc[75] = d75;
    acc[76] = d76;
    acc[77] = d77;
    acc[78] = d78;
    acc[79] = d79;
    acc[80] = d80;
    acc[81] = d81;
    acc[82] = d82;
    acc[83] = d83;
    acc[84] = d84;
    acc[85] = d85;
    acc[86] = d86;
    acc[87] = d87;
    acc[88] = d88;
    acc[89] = d89;
    acc[90] = d90;
    acc[91] = d91;
    acc[92] = d92;
    acc[93] = d93;
    acc[94] = d94;
    acc[95] = d95;
    acc[96] = d96;
    acc[97] = d97;
    acc[98] = d98;
    acc[99] = d99;
    acc[100] = d100;
    acc[101] = d101;
    acc[102] = d102;
    acc[103] = d103;
    acc[104] = d104;
    acc[105] = d105;
    acc[106] = d106;
    acc[107] = d107;
    acc[108] = d108;
    acc[109] = d109;
    acc[110] = d110;
    acc[111] = d111;
    acc[112] = d112;
    acc[113] = d113;
    acc[114] = d114;
    acc[115] = d115;
    acc[116] = d116;
    acc[117] = d117;
    acc[118] = d118;
    acc[119] = d119;
    acc[120] = d120;
    acc[121] = d121;
    acc[122] = d122;
    acc[123] = d123;
    acc[124] = d124;
    acc[125] = d125;
    acc[126] = d126;
    acc[127] = d127;
#else
    (void)desc_a; (void)desc_b; (void)acc;
#endif
}

template<int N>
__device__ __forceinline__ void wgmma_dispatch_f16(uint64_t desc_a, uint64_t desc_b, float* acc) {
    if constexpr (N == 8) { wgmma_m64n8k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 16) { wgmma_m64n16k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 24) { wgmma_m64n24k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 32) { wgmma_m64n32k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 40) { wgmma_m64n40k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 48) { wgmma_m64n48k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 56) { wgmma_m64n56k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 64) { wgmma_m64n64k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 72) { wgmma_m64n72k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 80) { wgmma_m64n80k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 88) { wgmma_m64n88k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 96) { wgmma_m64n96k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 104) { wgmma_m64n104k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 112) { wgmma_m64n112k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 120) { wgmma_m64n120k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 128) { wgmma_m64n128k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 136) { wgmma_m64n136k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 144) { wgmma_m64n144k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 152) { wgmma_m64n152k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 160) { wgmma_m64n160k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 168) { wgmma_m64n168k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 176) { wgmma_m64n176k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 184) { wgmma_m64n184k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 192) { wgmma_m64n192k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 200) { wgmma_m64n200k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 208) { wgmma_m64n208k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 216) { wgmma_m64n216k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 224) { wgmma_m64n224k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 232) { wgmma_m64n232k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 240) { wgmma_m64n240k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 248) { wgmma_m64n248k16_f16(desc_a, desc_b, acc); }
    else if constexpr (N == 256) { wgmma_m64n256k16_f16(desc_a, desc_b, acc); }
    else {
        static_assert(N == 0, "WGMMA dispatch: unsupported N");
    }
}

template<int N>
__device__ __forceinline__ void wgmma_dispatch_bf16(uint64_t desc_a, uint64_t desc_b, float* acc) {
    if constexpr (N == 8) { wgmma_m64n8k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 16) { wgmma_m64n16k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 24) { wgmma_m64n24k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 32) { wgmma_m64n32k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 40) { wgmma_m64n40k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 48) { wgmma_m64n48k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 56) { wgmma_m64n56k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 64) { wgmma_m64n64k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 72) { wgmma_m64n72k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 80) { wgmma_m64n80k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 88) { wgmma_m64n88k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 96) { wgmma_m64n96k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 104) { wgmma_m64n104k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 112) { wgmma_m64n112k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 120) { wgmma_m64n120k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 128) { wgmma_m64n128k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 136) { wgmma_m64n136k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 144) { wgmma_m64n144k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 152) { wgmma_m64n152k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 160) { wgmma_m64n160k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 168) { wgmma_m64n168k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 176) { wgmma_m64n176k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 184) { wgmma_m64n184k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 192) { wgmma_m64n192k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 200) { wgmma_m64n200k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 208) { wgmma_m64n208k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 216) { wgmma_m64n216k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 224) { wgmma_m64n224k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 232) { wgmma_m64n232k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 240) { wgmma_m64n240k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 248) { wgmma_m64n248k16_bf16(desc_a, desc_b, acc); }
    else if constexpr (N == 256) { wgmma_m64n256k16_bf16(desc_a, desc_b, acc); }
    else {
        static_assert(N == 0, "WGMMA dispatch: unsupported N");
    }
}

template<int N>
__device__ __forceinline__ void wgmma_dispatch_tf32(uint64_t desc_a, uint64_t desc_b, float* acc) {
    if constexpr (N == 8) { wgmma_m64n8k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 16) { wgmma_m64n16k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 24) { wgmma_m64n24k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 32) { wgmma_m64n32k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 40) { wgmma_m64n40k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 48) { wgmma_m64n48k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 56) { wgmma_m64n56k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 64) { wgmma_m64n64k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 72) { wgmma_m64n72k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 80) { wgmma_m64n80k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 88) { wgmma_m64n88k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 96) { wgmma_m64n96k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 104) { wgmma_m64n104k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 112) { wgmma_m64n112k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 120) { wgmma_m64n120k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 128) { wgmma_m64n128k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 136) { wgmma_m64n136k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 144) { wgmma_m64n144k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 152) { wgmma_m64n152k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 160) { wgmma_m64n160k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 168) { wgmma_m64n168k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 176) { wgmma_m64n176k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 184) { wgmma_m64n184k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 192) { wgmma_m64n192k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 200) { wgmma_m64n200k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 208) { wgmma_m64n208k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 216) { wgmma_m64n216k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 224) { wgmma_m64n224k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 232) { wgmma_m64n232k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 240) { wgmma_m64n240k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 248) { wgmma_m64n248k8_tf32(desc_a, desc_b, acc); }
    else if constexpr (N == 256) { wgmma_m64n256k8_tf32(desc_a, desc_b, acc); }
    else {
        static_assert(N == 0, "WGMMA dispatch: unsupported N");
    }
}

template<int N>
__device__ __forceinline__ void wgmma_dispatch_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
    if constexpr (N == 8) { wgmma_m64n8k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 16) { wgmma_m64n16k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 24) { wgmma_m64n24k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 32) { wgmma_m64n32k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 40) { wgmma_m64n40k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 48) { wgmma_m64n48k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 56) { wgmma_m64n56k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 64) { wgmma_m64n64k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 72) { wgmma_m64n72k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 80) { wgmma_m64n80k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 88) { wgmma_m64n88k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 96) { wgmma_m64n96k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 104) { wgmma_m64n104k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 112) { wgmma_m64n112k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 120) { wgmma_m64n120k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 128) { wgmma_m64n128k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 136) { wgmma_m64n136k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 144) { wgmma_m64n144k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 152) { wgmma_m64n152k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 160) { wgmma_m64n160k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 168) { wgmma_m64n168k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 176) { wgmma_m64n176k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 184) { wgmma_m64n184k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 192) { wgmma_m64n192k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 200) { wgmma_m64n200k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 208) { wgmma_m64n208k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 216) { wgmma_m64n216k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 224) { wgmma_m64n224k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 232) { wgmma_m64n232k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 240) { wgmma_m64n240k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 248) { wgmma_m64n248k32_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 256) { wgmma_m64n256k32_e4m3(desc_a, desc_b, acc); }
    else {
        static_assert(N == 0, "WGMMA dispatch: unsupported N");
    }
}

template<int N>
__device__ __forceinline__ void wgmma_dispatch_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
    if constexpr (N == 8) { wgmma_m64n8k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 16) { wgmma_m64n16k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 24) { wgmma_m64n24k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 32) { wgmma_m64n32k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 40) { wgmma_m64n40k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 48) { wgmma_m64n48k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 56) { wgmma_m64n56k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 64) { wgmma_m64n64k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 72) { wgmma_m64n72k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 80) { wgmma_m64n80k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 88) { wgmma_m64n88k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 96) { wgmma_m64n96k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 104) { wgmma_m64n104k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 112) { wgmma_m64n112k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 120) { wgmma_m64n120k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 128) { wgmma_m64n128k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 136) { wgmma_m64n136k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 144) { wgmma_m64n144k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 152) { wgmma_m64n152k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 160) { wgmma_m64n160k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 168) { wgmma_m64n168k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 176) { wgmma_m64n176k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 184) { wgmma_m64n184k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 192) { wgmma_m64n192k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 200) { wgmma_m64n200k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 208) { wgmma_m64n208k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 216) { wgmma_m64n216k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 224) { wgmma_m64n224k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 232) { wgmma_m64n232k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 240) { wgmma_m64n240k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 248) { wgmma_m64n248k32_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 256) { wgmma_m64n256k32_e5m2(desc_a, desc_b, acc); }
    else {
        static_assert(N == 0, "WGMMA dispatch: unsupported N");
    }
}

template<int N>
__device__ __forceinline__ void wgmma_dispatch_e4m3_e5m2(uint64_t desc_a, uint64_t desc_b, float* acc) {
    if constexpr (N == 8) { wgmma_m64n8k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 16) { wgmma_m64n16k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 24) { wgmma_m64n24k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 32) { wgmma_m64n32k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 40) { wgmma_m64n40k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 48) { wgmma_m64n48k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 56) { wgmma_m64n56k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 64) { wgmma_m64n64k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 72) { wgmma_m64n72k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 80) { wgmma_m64n80k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 88) { wgmma_m64n88k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 96) { wgmma_m64n96k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 104) { wgmma_m64n104k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 112) { wgmma_m64n112k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 120) { wgmma_m64n120k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 128) { wgmma_m64n128k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 136) { wgmma_m64n136k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 144) { wgmma_m64n144k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 152) { wgmma_m64n152k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 160) { wgmma_m64n160k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 168) { wgmma_m64n168k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 176) { wgmma_m64n176k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 184) { wgmma_m64n184k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 192) { wgmma_m64n192k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 200) { wgmma_m64n200k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 208) { wgmma_m64n208k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 216) { wgmma_m64n216k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 224) { wgmma_m64n224k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 232) { wgmma_m64n232k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 240) { wgmma_m64n240k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 248) { wgmma_m64n248k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else if constexpr (N == 256) { wgmma_m64n256k32_e4m3_e5m2(desc_a, desc_b, acc); }
    else {
        static_assert(N == 0, "WGMMA dispatch: unsupported N");
    }
}

template<int N>
__device__ __forceinline__ void wgmma_dispatch_e5m2_e4m3(uint64_t desc_a, uint64_t desc_b, float* acc) {
    if constexpr (N == 8) { wgmma_m64n8k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 16) { wgmma_m64n16k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 24) { wgmma_m64n24k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 32) { wgmma_m64n32k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 40) { wgmma_m64n40k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 48) { wgmma_m64n48k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 56) { wgmma_m64n56k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 64) { wgmma_m64n64k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 72) { wgmma_m64n72k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 80) { wgmma_m64n80k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 88) { wgmma_m64n88k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 96) { wgmma_m64n96k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 104) { wgmma_m64n104k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 112) { wgmma_m64n112k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 120) { wgmma_m64n120k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 128) { wgmma_m64n128k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 136) { wgmma_m64n136k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 144) { wgmma_m64n144k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 152) { wgmma_m64n152k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 160) { wgmma_m64n160k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 168) { wgmma_m64n168k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 176) { wgmma_m64n176k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 184) { wgmma_m64n184k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 192) { wgmma_m64n192k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 200) { wgmma_m64n200k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 208) { wgmma_m64n208k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 216) { wgmma_m64n216k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 224) { wgmma_m64n224k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 232) { wgmma_m64n232k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 240) { wgmma_m64n240k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 248) { wgmma_m64n248k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else if constexpr (N == 256) { wgmma_m64n256k32_e5m2_e4m3(desc_a, desc_b, acc); }
    else {
        static_assert(N == 0, "WGMMA dispatch: unsupported N");
    }
}

} // namespace axp::realize::sm90::detail
