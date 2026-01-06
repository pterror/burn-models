# TODO

## Next Up

- [x] Rename project from burn-image to burn-models
- [x] Restructure crates for general model inference

## Completed (Image Generation)

- [x] Explore stable-diffusion-burn source to understand architecture
- [x] Explore stable-diffusion-xl-burn source for SDXL differences
- [x] Define crate structure (workspace with 7 crates)
- [x] Implement weight loading from safetensors
- [x] Implement CLIP tokenizer
- [x] Implement CLIP text encoder
- [x] Implement DDIM sampler
- [x] Implement VAE decoder
- [x] Implement UNet

## Roadmap

### Phase 1: Foundation

- [x] Weight loading infrastructure (safetensors → Burn tensors)
- [x] Model serialization (save/load converted weights)
- [x] Basic tensor ops (silu, groupnorm, layernorm)
- [x] Attention mechanism (qkv_attention, causal_mask)

### Phase 2: SD 1.x Components

- [x] CLIP text encoder
- [x] VAE decoder (latent → image)
- [x] UNet (diffusion backbone)
- [x] DDIM sampler

### Phase 3: SD 1.x Pipeline

- [x] Text-to-image generation
- [x] Classifier-free guidance
- [x] VAE encoder (for img2img)
- [x] Image-to-image pipeline

### Phase 4: SDXL Components

- [x] OpenCLIP text encoder (second encoder)
- [x] SDXL UNet (larger, different architecture)
- [x] SDXL VAE (scaling factors: 0.13025 vs 0.18215)
- [x] Refiner model

### Phase 5: SDXL Pipeline

- [x] SDXL text-to-image
- [x] SDXL img2img
- [x] Inpainting
- [x] Base + refiner workflow

### Phase 6: Polish

- [x] Multi-backend support (CUDA, Metal, WGPU)
- [x] Memory optimization
- [x] CLI interface
- [x] Documentation

## Backlog

### Samplers
- [x] DDIM
- [x] Euler
- [x] Euler Ancestral
- [x] DPM++ 2M
- [x] DPM++ SDE
- [x] Heun
- [x] HeunPP2
- [x] DPM 2
- [x] DPM 2 Ancestral
- [x] LMS
- [x] DPM++ 3M SDE
- [x] DDPM
- [x] LCM (Latent Consistency Model)
- [x] iPNDM
- [x] iPNDM-v
- [x] DEIS
- [x] UniPC
- [x] SA-Solver
- [x] Euler CFG++
- [x] Euler Ancestral CFG++
- [x] DPM Fast
- [x] DPM Adaptive
- [x] DPM++ 2S Ancestral
- [x] DPM++ 2S Ancestral CFG++
- [x] DPM++ 2M CFG++
- [x] DPM++ 2M SDE Heun
- [x] Res Multistep

### Model Extensions
- [x] LoRA support (Kohya, Diffusers formats)
- [x] ControlNet integration (SD 1.x, SDXL)
- [x] Textual inversion / embeddings
- [x] IP-Adapter
- [x] T2I-Adapter

### Performance
- [x] fp16/bf16 inference (precision config)
- [x] Flash Attention (tiled, memory-efficient, sliced)
- [x] xFormers-style memory efficient attention
- [x] Model offloading (sequential CPU/GPU)
- [x] Batch processing

### CubeCL Optimization (burn-models-cubecl crate)

Custom GPU kernels for operations that benefit from fusion/specialization.
Decision: Own crate rather than upstream (faster iteration, avoid "vibe code" concerns).

#### Phase 1: Infrastructure
- [x] Create burn-models-cubecl crate with cubecl dependency
- [x] Set up feature flags (cpu default, wgpu, cuda)
- [x] Add benchmark harness for comparing against tensor-ops implementations

#### Phase 2: Conv3d Kernel
- [x] Port conv_transpose3d pattern to conv3d (simple direct kernel, ~200 lines)
- [x] NTHWC layout handling (permute in, permute out)
- [x] Test harness comparing CubeCL vs im2col output (correctness) ✓ PASSING
  - CUDA: `cargo test -p burn-models-cubecl --features cuda --test correctness_cuda -- --ignored`
- [x] Benchmark against im2col implementation ✓ **630-40,900× speedup**
  - Run: `cargo bench -p burn-models-cubecl --features cuda --bench conv3d`
  - Results: see `docs/cubecl-guide.md` for full benchmark table
- [x] Provide CubeCL Conv3d layer (removed im2col from burn-models-core)
  - `Conv3dLayer<R>` in burn-models-cubecl for all CubeBackends
  - Helper functions `to_cube_tensor`/`from_cube_tensor` for type conversion
  - Comprehensive test suite: 9 CPU tests, 8 CUDA tests, 6 WGPU tests (includes NTHWC layout tests)

#### Phase 3: Conv3d Optimization
- [x] Add Line<E> vectorization with NTHWC kernel (`conv3d_nthwc`)
- [x] Recursive kernel_loop with comptime dimension unrolling
- [x] FastDivmod for efficient index calculation
- [x] Benchmark simple vs optimized kernel on CUDA

**Benchmark Results** (RTX 3060):

| Config | Simple (NCTHW) | Optimized (NTHWC) | Speedup | vs im2col |
|--------|----------------|-------------------|---------|-----------|
| tiny (1,2,4,4,8,8) | **12.5 µs** | 24.9 µs | 0.5× | 470× |
| small (1,4,8,8,32,32) | 63.8 µs | **41.3 µs** | 1.5× | 39,800× |
| medium (1,8,16,8,64,64) | 742 µs | **286 µs** | 2.6× | - |
| strided (stride=2) | 107 µs | **97.9 µs** | 1.1× | - |
| deep (1,32,64,4,32,32) | 1.04 ms | **866 µs** | 1.2× | - |

**Findings:**
- Simple kernel wins for tiny tensors (kernel launch overhead dominates)
- Optimized kernel wins 1.1-2.6× for larger tensors
- Permutation overhead is negligible BUT requires contiguous copy (memory allocation)
- Both kernels retained: simple for NCTHW (no memory overhead), optimized for NTHWC

See `docs/cubecl-conv3d.md` for detailed analysis.

#### Phase 4: Additional Kernels (as needed)
- [x] AvgPool3d kernel (5 CPU tests passing)
- [x] MaxPool3d kernel (5 CPU tests passing)
- [x] Expose burn-cubecl's flash attention (cubek-attention) via burn-models-cubecl
  - `flash_attention`, `flash_attention_masked` functions with `FlashAttentionOptions`
  - Supports both causal (LLMs) and non-causal (diffusion) via `FlashAttentionOptions::causal()`
  - Default is non-causal (suitable for diffusion models)
  - Calls cubek::attention directly (bypasses burn-cubecl's hardcoded causal=true)
  - Removed unused tensor-ops fallback from burn-models-core (was dead code)
  - Note: CPU backend has line_size constraints that cause assertion failures (GPU backends work)
  - FlashAttention3: Hopper-only (H100/H800), lower priority
- [x] Custom activation fusions (GroupNorm+SiLU)
  - `groupnorm_silu(input, weight, bias, options)` - fused GroupNorm + SiLU
  - `groupnorm(input, weight, bias, options)` - GroupNorm without SiLU
  - Two-phase kernel: compute group stats, then normalize + affine + SiLU
  - 5 CPU tests passing (tolerance 1e-2 due to biased vs unbiased variance)

#### Phase 5: Integration into Existing Models

Integrate CubeCL kernels into existing model implementations.

- [x] GroupNorm+SiLU → UNet ResNet blocks
  - Added `burn-models-unet/src/cubecl.rs` with `ResBlockCubeCL<R: CubeRuntime>`
  - Uses fused `groupnorm_silu` kernel in forward pass
  - `convert_resblock()` converts existing `ResBlock` to CubeCL version
  - Feature-gated: `--features cubecl`

- [x] FlashAttention → Transformer attention layers
  - Added `CrossAttentionCubeCL<R: CubeRuntime>` to `burn-models-unet/src/cubecl.rs`
  - Uses O(n) memory flash attention instead of materializing full attention matrix
  - `convert_crossattention()` converts existing `CrossAttention` to CubeCL version
  - Non-causal by default (diffusion models)

#### Phase 6: 3D VAE for Video Models

Implement actual 3D VAE encoder/decoder using Conv3d kernels.
Target: CogVideoX, Wan, Mochi video generation.

- [x] 3D VAE Encoder
  - `Vae3dEncoderCubeCL<R: CubeRuntime>` in `burn-models-core/src/vae3d.rs`
  - Uses Conv3d + fused GroupNorm+SiLU kernels
  - ResBlock3dCubeCL, Downsample3dCubeCL building blocks
  - Feature-gated: `--features cubecl`

- [x] 3D VAE Decoder
  - `Vae3dDecoderCubeCL<R: CubeRuntime>` in `burn-models-core/src/vae3d.rs`
  - Uses Conv3d + fused GroupNorm+SiLU kernels
  - Upsample3dCubeCL for temporal/spatial upsampling

- [x] Weight loading infrastructure for CogVideoX/Mochi safetensors
  - `Vae3dKeyMapping` enum for model-specific key naming conventions
  - `expected_vae3d_weights()` for validation
  - Works with existing `SafeTensorFile` from burn-models-convert

See `docs/cubecl-guide.md` for implementation details.

#### Phase 7: Polish & Usability

- [x] GPU integration tests for CubeCL VAE (WGPU/CUDA)
  - UNet: ResBlockCubeCL, CrossAttentionCubeCL shape tests
  - VAE: ResBlock3dCubeCL, Downsample3dCubeCL, Upsample3dCubeCL shape tests
  - VAE encoder/decoder tiny model integration tests
- [x] Benchmark CubeCL integrations
  - `benches/unet_blocks.rs`: ResBlock vs ResBlockCubeCL, CrossAttention vs CrossAttentionCubeCL
  - Run: `cargo bench -p burn-models-cubecl --features cuda --bench unet_blocks`
- [ ] MLX backend (Apple Silicon) - via [burn-mlx](https://lib.rs/crates/burn-mlx)
  - Note: burn-mlx 0.1.2 requires burn ^0.16, we're on 0.20 - wait for update
- [x] Wire up CLI `generate` command - now validates paths and shows helpful messages
- [x] WeightLoader for SD models - enables actual image generation
  - [x] CLIP text encoder loader (`SdWeightLoader::load_clip_text_encoder`)
    - Detects prefix automatically (text_model, text_encoder.text_model, etc.)
    - Loads embeddings, all transformer layers, final layer norm
  - [x] UNet loader (`SdWeightLoader::load_unet`)
    - Detects prefix (model.diffusion_model, unet, etc.)
    - Loads time embedding, conv_in, all down/mid/up blocks, conv_out
    - Each block: ResBlock, SpatialTransformer, Downsample/Upsample
    - **Note**: Currently supports HF diffusers naming only. Single-file checkpoints
      (CivitAI/CompVis format) use different naming and need mapping:
      - CompVis: `input_blocks`/`output_blocks`, `in_layers.0/2`
      - HF: `down_blocks`/`up_blocks`, `norm1/conv1`
  - [x] VAE decoder loader (`SdWeightLoader::load_vae_decoder`)
    - Detects prefix (decoder, vae.decoder, first_stage_model.decoder)
    - Loads conv_in, mid blocks, all up blocks, conv_out
    - Each block: ResnetBlock, SelfAttention, Upsample
  - [ ] CompVis naming support for single-file checkpoints (CivitAI models)
    - Map `input_blocks`/`middle_block`/`output_blocks` to HF naming
    - Or add separate CompVis loader variant
    - Affects UNet and VAE loaders

### Future Architectures

#### UNet-based (similar to current)
- [x] Stable Diffusion 2.x - OpenCLIP ViT-H/14 text encoder, v-prediction support in samplers
- [x] Stable Cascade - Würstchen architecture, latent cascade

#### DiT-based (new architecture needed)
- [x] Flux (Black Forest Labs) - Flow matching, DiT architecture, weight loading
- [x] Stable Diffusion 3 / 3.5 - MMDiT (Multimodal Diffusion Transformer), rectified flow
- [x] Z-Image (Alibaba) - Single-stream DiT (S3-DiT), 6B params, text+image tokens in single stream
- [x] Qwen-Image (Alibaba) - 20B MMDiT, Qwen2.5-VL text encoder, joint attention with pooled conditioning
- [x] PixArt-α/Σ - Efficient DiT, T5 text encoder, cross-attention to text
- [x] AuraFlow - Open source flow-based, MMDiT joint attention
- [x] Hunyuan-DiT - Bilingual DiT model, dual text encoder (CLIP + MT5), skip connections

### Video Generation

- [x] Wan 2.x (Alibaba) - DiT + MoE with sparse routing, 3D VAE, factorized spatial-temporal attention
- [x] CogVideoX - Open source video DiT, factorized spatial-temporal attention
- [x] Mochi - Genmo's open source video model, asymmetric DiT, factorized attention
- [x] LTX-Video - Lightricks video model, causal temporal attention

### Large Language Models

- [x] LLaMA 2/3 architecture (basic implementation)
- [x] LLaMA weight loading from safetensors
- [x] Mixtral 8x7B/8x22B (LLaMA + MoE)
- [x] Mistral 7B/Nemo - Sliding window attention
- [x] Qwen 2.5 - Alibaba's multilingual LLM (0.5B to 72B)
- [x] Gemma 2 - Google's open models (interleaved sliding/global attention, logit soft-capping, GeGLU)
- [x] Phi-3/3.5 - Microsoft's small models (fused QKV/gate-up, GQA)
- [x] DeepSeek V1/V2 - Multi-head Latent Attention (MLA) for compressed KV cache

### Shared Building Blocks (burn-models-core)

#### Layers
- [x] Transformer block (shared by DiT, LLM, encoders)
- [x] RoPE (Rotary Position Embedding)
- [x] RMSNorm
- [x] SwiGLU / GeGLU activations
- [x] Multi-head attention with GQA support
- [x] MoE (Mixture of Experts) routing
- [x] DiT components (AdaLayerNorm, DiTBlock, PatchEmbed)
- [x] Sliding window attention mask

#### Inference Optimization
- [x] KV cache for autoregressive models
- [x] Paged attention (block tables, memory management, scheduler)
- [x] Continuous batching (iteration-level scheduling, preemption)
- [x] Speculative decoding (draft/target verification, acceptance sampling)

#### Quantization
- [x] INT8 dynamic quantization (symmetric/asymmetric)
- [x] INT4 quantization (group-wise, GPTQ style)
- [x] FP8 support

#### Video-specific
- [x] 3D VAE (temporal compression)
- [x] Temporal attention layers
- [x] Frame interpolation

### Pre-Release

- [x] Architecture review - ensure consistent patterns across model implementations
- [x] API surface review - design public API for library consumers
- [x] Documentation - usage examples, model loading, inference patterns

### Future Backlog

#### Alternative Architectures (Production-Ready)
- [x] RWKV-7 "Goose" - RNN with transformer performance, linear time/constant space, no KV cache
- [x] Mamba / Mamba-2 - Selective state space models, linear scaling, 5x faster inference than transformers
- [x] Jamba - AI21's Transformer-Mamba-MoE hybrid, 1:7 attention:Mamba ratio, 256K context

#### Fast Image Generation
- [x] SANA - NVIDIA's linear DiT, 32x compression, 4K images on laptop GPU in <1s

#### Experimental / Research (Lower Priority)
- [ ] xLSTM - Extended LSTM with exponential gating and matrix memory (Sepp Hochreiter)
- [ ] Zamba/Zamba2 - Zyphra's Mamba backbone + shared attention layers
- [ ] Griffin/Hawk - Google DeepMind's gated linear recurrences (RecurrentGemma)
- [ ] RetNet - Microsoft's retentive network, parallel/recurrent/chunkwise modes
- [ ] Hyena/StripedHyena - Long convolutions + gating, subquadratic attention
- [ ] TTT (Test-Time Training) - Hidden state updated via gradient descent during inference
- [ ] LLaDA - Large Language Diffusion with Masking, bidirectional diffusion LM
- [ ] TESS-2 - Simplex diffusion LM, reward guidance for alignment

## Postmortems

### Dead Code Patterns (2026-01-06)

Cleaned up `let _var = ...` patterns where values were computed but never used.
Root causes:

1. **Incomplete implementations shipped as "good enough"**: Several samplers (DPM2, DEIS,
   DPM3M, DPM Fast) computed intermediate values needed for higher-order methods but then
   fell back to simpler approximations. The computation was left in place, presumably to
   be completed later. Example: `dpm2.rs` computed `sample_mid` for midpoint method but
   comments note "This would need another model call in practice" - then uses simplified
   extrapolation instead.

2. **Placeholder implementations**: `frame_interpolation.rs` computed all four bilinear
   interpolation weights but returns input unchanged with a comment explaining that full
   implementation "requires index_select or gather operations."

3. **Refactoring leftovers**: Variables like `_refiner_steps` in pipeline.rs and
   `_noise_pred` in euler_cfg.rs were computed for logic that was later simplified or
   removed, but the computation wasn't cleaned up.

4. **Speculative code for future features**: `rwkv_loader.rs` computed `lora_dim` for
   LoRA support that was never implemented.

**Lesson**: When deferring implementation, prefer one of:
- Don't write the computation at all - add a TODO comment instead
- Write the full implementation
- If partial implementation is necessary, add `// TODO: ` explaining what's missing

Suppressing warnings with `_` prefix hides technical debt. The warning exists to help.

### Incomplete Implementations (2026-01-06)

Found during dead code audit. These are stubs or broken implementations that need attention:

#### Fixable (no backend changes needed)

- [x] `attention.rs:30-34` `causal_mask` - Returns zeros instead of proper -inf upper triangular mask.
  Would allow model to attend to future tokens. Note: `transformer.rs` has a working version.

- [x] `speculative.rs:268-271` `sample_token` - Non-greedy mode just uses argmax.
  Temperature sampling is fake (comment says "Simplified sampling").

- [x] `dpm_fast.rs:200` `DpmAdaptiveSampler::step` - "accept all steps" comment.
  Adaptive step control is unimplemented, it's just fixed-step pretending to be adaptive.

#### Fixed

- [x] `vae3d.rs:174` `Conv3d::forward` - Now implements proper 3D convolution using im2col.
  Extracts 3D patches, reshapes to columns, matrix multiply with weights.
  **Future optimization**: CubeCL kernel for better GPU performance.

#### Fixed (were marked unfixable but actually solvable)

- [x] `paged_attention.rs:284` `store_single_kv` - Now uses `slice_assign()`.
  Burn has had this since 0.16 - the "needs custom kernels" comment was stale.

- [x] `precision.rs:94-104` `to_fp16`/`to_bf16` - Removed.
  These were no-op functions. Added docs explaining precision is compile-time via backend.

#### Intentional Simplifications (not bugs)

- [x] `main.rs` CLI generate - Fully functional SD 1.x image generation.
  Loads CLIP, UNet, VAE from safetensors and runs full diffusion pipeline.

- [x] `rwkv.rs:277-290` RWKV-7 dynamic mixing - Now implements full low-rank projection.
  Auto-detects RWKV-6 vs RWKV-7 by checking if `time_maa_w1` weights are present.

### Precision and Performance (2026-01-07)

**f16 vs f32 Precision:**
- Added `--precision f16|f32` flag to CLI
- f32: ~57 sec inference, ~11GB VRAM
- f16: ~6GB VRAM, but ops run on CPU (0-1% GPU usage)
- CubeCL issue #984 confirmed f16 should be ~2x faster - our slowness is something else

**Timing (SD 1.x @ 512x512, f32, 30 steps):**
```
[timing] tokenizer: 38ms
[timing] load CLIP: 578ms
[timing] load UNet: 2.8s
[timing] load VAE: 82ms
[timing] inference: 53.4s (~1.78s/step)
[timing] total: 57s
```

Comparison: ComfyUI does SDXL @ 20 steps in ~26s (~1.3s/step). We're 37% slower per step on a smaller model.

**TODO:**
- [x] Add `--debug timing,shapes` flag for diagnostics
- [ ] Debug garbled output from CompVis models

### f16 Performance Investigation (2026-01-07)

**Root cause found**: Timestep tensors are created from f32 every step:
```rust
// pipeline.rs:172 - runs 30x per generation
let t = Tensor::<B, 1>::from_data(
    TensorData::new(vec![timestep as f32], [1]),  // ALWAYS f32
    &self.device,
);
```

With f16 backend, this forces f32→f16 conversion each step, likely causing CPU→GPU sync stalls.

**Fix**:
1. Precompute all timestep tensors at sampler initialization
2. Use backend-native element type instead of hardcoded f32
3. Avoid CPU→GPU transfers in the denoising loop

### Linear Weight Convention (2026-01-07)

**RESOLVED**: Not a bug. Burn and PyTorch use different weight layouts:
- Burn: `[d_input, d_output]` (Row layout, default)
- PyTorch: `[d_output, d_input]`

Burn's `linear()` does `input.matmul(weight)` - no transpose.
PyTorch's `F.linear()` does `input @ weight.T`.

**Solution**: Transpose Linear weights when loading from PyTorch safetensors:
```rust
// In load_linear():
linear.weight = Param::from_tensor(weight.transpose());
```

Previously observed errors like `[1, 77, 768] @ [1, 3072, 768]` were from missing transpose.


### Session Notes (2026-01-06) - CubeCL Phase 4

#### Key Files Modified

**burn-models-cubecl crate:**
- `src/attention.rs` - Flash attention wrapper calling cubek::attention directly
  - `flash_attention(q, k, v, options)` - main function
  - `flash_attention_masked(q, k, v, mask, options)` - with explicit mask
  - `FlashAttentionOptions { out_dtype, causal }` - default is non-causal (diffusion)
  - `FlashAttentionOptions::causal()` - for LLMs
- `src/groupnorm_silu.rs` - Fused GroupNorm + SiLU kernel
  - `groupnorm_silu(input, weight, bias, options)` - main function
  - `groupnorm(input, weight, bias, options)` - without SiLU
  - `GroupNormSiLuOptions { num_groups, eps }` - configurable groups (default 32)
  - Two-phase: 1) compute mean/var per group, 2) normalize + affine + SiLU
- `src/pool3d.rs` - AvgPool3d/MaxPool3d kernels
- `src/lib.rs` - exports attention, pool3d, groupnorm_silu modules
- `Cargo.toml` - added `cubek = "=0.1.0-pre.1"` dependency with `attention` feature

**Tests:**
- `tests/correctness_cpu.rs` - 21 passing, 3 ignored (flash attention CPU limitation)
- `tests/correctness.rs` (WGPU) - compiles, GPU tests require hardware
- `tests/correctness_cuda.rs` - compiles, GPU tests require hardware

**Removed:**
- `burn-models-core/src/flash_attention.rs` - was dead code (exported but never used)

#### Technical Notes

**Flash Attention:**
- burn-cubecl's `flash_attention` hardcodes `causal: true`
- We bypass it by calling `cubek::attention::launch::launch_ref` directly
- CPU backend fails with assertion `unit_tile.layout.num_cols % line_size == 0`
- GPU backends (CUDA, WGPU) should work

**GroupNorm+SiLU Fusion (DONE):**
- Pattern: `silu(groupnorm(x, gamma, beta, num_groups))`
- Two-phase kernel: 1) compute mean/var per group, 2) normalize + affine + SiLU
- Avoids intermediate tensor allocation between GroupNorm and SiLU
- Uses population variance (n divisor) - 1e-2 tolerance needed vs Burn's unbiased variance

#### Test Commands

```sh
# CPU tests (all kernels)
cargo test -p burn-models-cubecl --features cpu --test correctness_cpu

# WGPU tests (requires GPU)
cargo test -p burn-models-cubecl --features wgpu --test correctness -- --ignored

# CUDA tests (requires CUDA GPU)
cargo test -p burn-models-cubecl --features cuda --test correctness_cuda -- --ignored
```
