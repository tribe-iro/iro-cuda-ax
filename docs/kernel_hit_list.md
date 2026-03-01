# Kernel Hit List Status

Tracks Section 7 must-ship kernels and verification evidence.

Status legend:

- `MANIFEST_VALIDATED`: schema/capability checks pass for manifest/tooling path.
- `NOT_IN_MANIFEST`: kernel/config not yet instantiated in manifests.
- `NOT_RUN`: numerical/perf runtime evidence not captured.
- `BLOCKED_NO_GPU`: no CUDA-capable runtime GPU available for capture.
- `BLOCKED_HW_SM100`: SM100 runtime closure blocked by hardware availability.

| Kernel | Arch | Config | Compile | Numerical | Perf | Notes |
|---|---|---|---|---|---|---|
| FA2 decode attention | sm90 | bf16 head_dim=128 causal=true | MANIFEST_VALIDATED | NOT_RUN | NOT_RUN | `fa2_decode_attention_64x64_hd128` is manifested and validator-covered. |
| FA2 decode attention | sm100 | bf16 head_dim=128 causal=true | NOT_IN_MANIFEST | BLOCKED_HW_SM100 | BLOCKED_HW_SM100 | Hardware/runtime gated. |
| FA2 prefill attention | sm90 | bf16 head_dim=128 causal=true | MANIFEST_VALIDATED | NOT_RUN | NOT_RUN | `fa2_prefill_attention_64x64_hd128` is manifested and validator-covered. |
| FA2 prefill attention | sm100 | bf16 head_dim=128 causal=true | NOT_IN_MANIFEST | BLOCKED_HW_SM100 | BLOCKED_HW_SM100 | Hardware/runtime gated. |
| GEMM | sm89/sm90/sm100 | bf16/fp16 common shapes | MANIFEST_VALIDATED | NOT_RUN | NOT_RUN | `gemm_16x16x16` (sm89) and `gemm_64x64x16` (sm90) are manifested and validator-covered; sm100 manifest stays intentionally empty until real SM100 realizations exist. |
| Fused GEMM+SiLU | sm90/sm100 | bf16 MLP shapes | MANIFEST_VALIDATED | NOT_RUN | NOT_RUN | `gemm_64x64x16_bias_silu` is manifested/validated on sm90; sm100 remains unimplemented. |
| RMSNorm | sm89/sm90/sm100 | bf16/fp16 hidden 4096/8192 | MANIFEST_VALIDATED | NOT_RUN | NOT_RUN | `rmsnorm_16x16` is manifested/validated for sm89/sm90; sm100 manifest remains intentionally empty until real SM100 realizations exist. |
| Softmax | sm89/sm90/sm100 | fp32 accum bf16 io | MANIFEST_VALIDATED | NOT_RUN | NOT_RUN | `softmax_row4` is manifested/validated for sm89/sm90; sm100 manifest remains intentionally empty until real SM100 realizations exist. |
| Vectorized elementwise | sm89/sm90/sm100 | contiguous/strided | MANIFEST_VALIDATED | NOT_RUN | NOT_RUN | `vectorized_elementwise_16x16` is manifested/validated for sm89/sm90; strided variants and sm100 remain pending. |

## Hardware-gated status

- status: `BLOCKED_HW_SM100`
- owner: `tribeiro`
- date: `2026-02-28`
- next_checkpoint: `2026-03-31`
