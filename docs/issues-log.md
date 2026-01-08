# Issues Log

Running log of issues encountered during development and their resolution.

Each issue includes:
- **Symptom**: How it manifested
- **Root Cause**: Why it happened
- **Fix**: How we resolved it
- **Prevention**: How to avoid similar issues in the future

## 2026-01-07

### [FIXED] Blurry Output - Missing c_in Input Scaling

**Symptom**: Generated images were extremely blurry compared to ComfyUI/A1111 output.

**Root Cause**: k-diffusion formulation requires input variance normalization before UNet.

The UNet expects input with variance ~1.0. During diffusion, the noisy latent has variance
scaling with sigma. Without normalization, the UNet receives inputs with wrong magnitude.

k-diffusion uses: `c_in = 1 / sqrt(1 + sigma^2)` to normalize input variance.

**Fix**: Apply c_in scaling before each UNet call:
```rust
let c_in = 1.0 / (1.0 + sigma * sigma).sqrt();
let latent_scaled = latent.clone() * c_in;
let noise = unet.forward(latent_scaled, t, conditioning);
```

**Files**: `crates/burn-models/src/pipeline/sd1x.rs`

**Prevention**:
- When implementing samplers, document the expected input/output formulation
- k-diffusion vs diffusers vs paper formulations differ - be explicit about which is used
- Test: Compare intermediate values (scaled input magnitude) against reference implementation

---

### [FIXED] Gray/Brown Backgrounds - Incorrect Denoised Computation

**Symptom**: After c_in fix, backgrounds were gray/brown (neutral latent color) instead of proper colors.

**Root Cause**: Used scaled input for denoised computation instead of unscaled.

Initial (wrong) fix attempted:
```rust
let denoised = latent_scaled - noise_pred * sigma;  // WRONG
```

This dampens denoised by c_in factor, causing low-variance areas (backgrounds) to collapse toward zero.

**Correct formulation** (from ComfyUI `model_sampling.py`):
- `c_in` scaling is ONLY for UNet input normalization
- `denoised = x - noise * sigma` where x is the **original unscaled** latent
- The sampler's `step()` function already does this correctly

**Fix**: Just use the original sampler step function with unscaled latent:
```rust
// c_in scaling is ONLY for UNet input
let latent_scaled = latent.clone() * c_in;
let noise = unet.forward(latent_scaled, t, cond);

// Sampler uses UNSCALED latent - it computes denoised internally
latent = sampler.step(latent, noise_pred, step_idx);
```

**Files**: `crates/burn-models/src/pipeline/sd1x.rs`

**Prevention**:
- Read ComfyUI source carefully - `model_input` in `calculate_denoised` is unscaled x
- c_in is variance normalization for neural network input only
- The sampler step formula operates entirely in unscaled space
- Test: Check background colors match reference - gray = signal being dampened

---

### [FIXED] Patchy/Blurry Output with Karras Schedule

**Symptom**: With Karras sigma schedule, output looked coherent but "patchy" - slightly blurry with inconsistent texture.

**Root Cause**: Timestep-sigma mismatch with non-uniform schedules.

With the Karras schedule, sigmas are respaced using the Karras formula, but we were still
passing the original evenly-spaced timesteps to the UNet. For example:
- Karras sigma[1] = 15.01 (corresponds to training t≈910)
- We were passing t=900 (the original evenly-spaced timestep)

The UNet's timestep embedding is trained to expect a specific noise level for each timestep.
Passing the wrong timestep causes subtle conditioning errors.

**Fix**: Convert sigmas back to training timesteps using `sigma_to_timestep()`:
```rust
// For Karras/other non-uniform schedules, find the training timestep
// that corresponds to each sigma value
let timesteps = if sigma_schedule == SigmaSchedule::Normal {
    sampler.timesteps().to_vec()
} else {
    schedule.sigmas_to_timesteps(&sigmas[..sigmas.len() - 1])
};
```

The conversion uses: `alpha_cumprod = 1 / (1 + sigma^2)`, then finds the closest match.

**Files**:
- `crates/burn-models-samplers/src/scheduler.rs` - Added `sigma_to_timestep()` method
- `crates/burn-models/src/pipeline/sd1x.rs` - Use sigma-derived timesteps for non-Normal schedules

**Prevention**:
- In k-diffusion/ComfyUI, the model wrapper converts sigma→timestep internally
- When implementing samplers with non-uniform sigma schedules, always convert sigma to training timestep
- Test: Compare timestep sequence with ComfyUI output for same schedule

---

### [FIXED] f16 Type Mismatch in Samplers

**Symptom**: Runtime panic when using f16 precision:
```
TypeMismatch("Invalid target element type (expected F16, got F32)")
```

**Root Cause**: Tensor `.to_vec()` requires element type to match.

Code like this fails with f16 tensors:
```rust
let alpha: f32 = tensor.into_data().to_vec().unwrap()[0];
```

**Fix**: Add `.convert::<f32>()` before `.to_vec()`:
```rust
let alpha: f32 = tensor.into_data().convert::<f32>().to_vec().unwrap()[0];
```

**Files**: Multiple samplers - `scheduler.rs`, `lcm.rs`, `ddpm.rs`, `ipndm.rs`, `dpm_fast.rs`

**Prevention**:
- Always use `.convert::<f32>()` when extracting scalar values from tensors
- Test samplers with f16 backend explicitly
- Consider: wrapper function `fn scalar_f32<B: Backend>(t: Tensor<B, 0>) -> f32`

---

### [FIXED] Garbled/Psychedelic Output from VAE

**Symptom**: Generated images had structure but psychedelic/heat-map colors.

**Root Cause**: VAE decoder missing `post_quant_conv` layer.

The diffusers VAE has two extra 1x1 convolutions:
- `quant_conv` (8→8 channels) - applied after encoder
- `post_quant_conv` (4→4 channels) - applied before decoder

We were missing `post_quant_conv`, which transforms the latent before decoding.

**Fix**:
1. Added `post_quant_conv: Option<Conv2d<B>>` field to `Decoder` struct
2. Load `first_stage_model.post_quant_conv` weights in decoder loader
3. Apply in `forward_raw()` before main decoder path

**Files**: `crates/burn-models-vae/src/decoder.rs`, `crates/burn-models-convert/src/sd_loader.rs`

**Prevention**:
- When porting models, enumerate ALL layers in reference implementation
- Create checklist of expected weight names from safetensors before coding
- Test: VAE roundtrip test comparing output to reference (diffusers) on same input
- Could catch with: `expected_weights()` function that validates all required tensors are present

---

### [FIXED] Non-deterministic Tokenization

**Symptom**: Same prompt produced different outputs across runs.

**Root Cause**: HashMap iteration order is non-deterministic in Rust.

Vocab was built by iterating `byte_encoder.values()` and `bpe_ranks.keys()`.
Each run assigned different token IDs to the same tokens.

**Fix**: Sort iterators before assigning vocab indices:
```rust
byte_chars.sort();  // Sort byte-level tokens
pairs.sort_by_key(|&(_, rank)| rank);  // Sort BPE merges by rank
```

**Files**: `crates/burn-models-clip/src/tokenizer.rs`

**Prevention**:
- Never iterate HashMap/HashSet when order matters
- Prefer BTreeMap/BTreeSet or explicit sorting
- Test: Run tokenizer twice on same input, assert outputs are identical
- Test: Compare token IDs against known-good reference (official vocab)
- Clippy lint: Consider adding custom lint for HashMap iteration in determinism-critical code

---

### [FIXED] Token ID Mismatch vs Official CLIP

**Symptom**: Tokens like "cute" got wrong IDs, affecting text conditioning quality.

**Root Cause**: Built vocab programmatically from BPE merges instead of using official vocab.

The programmatic approach produced different token assignments than OpenAI's original.

**Fix**: Load official CLIP `vocab.json` directly:
1. Downloaded official vocab from HuggingFace
2. Added simple JSON parser (no serde dependency)
3. Embed vocab in binary with `include_str!`

**Files**: `crates/burn-models-clip/src/tokenizer.rs`, `crates/burn-models-clip/data/vocab.json`

**Verification**: Token IDs now match exactly:
- "a</w>" = 320
- "cute</w>" = 2242
- "cat</w>" = 2368

**Prevention**:
- Use official assets (vocab files, configs) rather than reconstructing
- When reconstruction is necessary, validate against official output
- Test: Golden test with known prompt → known token IDs
- General: Prefer loading authoritative data over deriving it

---

### [FIXED] f16 Produces NaN in UNet

**Symptom**: f16 precision produces NaN from step 0 in UNet forward pass.
```
[debug] Step 0 t=666 - noise_uncond: min=inf, max=-inf, mean=NaN
```

**Root Cause**: Attention softmax overflow in f16.
- f16 max representable: ~65504
- exp(x) overflows for x > ~11
- Attention scores before softmax can exceed this
- More specifically: Q @ K^T matmul itself can overflow, not just softmax

**Failed Attempt**: Manual stable softmax (max-subtraction before exp).
```rust
let attn_max = attn.clone().max_dim(3);
let attn = (attn - attn_max).exp();
let attn = attn.clone() / attn.clone().sum_dim(3);
```
**Result**: Still NaN. Overflow happens in matmul, before softmax.

**Solution**: Flash attention with f32 accumulation.
- Implemented in `burn-models-unet/src/cubecl.rs`
- `UNetCubeCL` type with `convert_unet()` function
- Uses flash attention kernel with f32 accumulation internally
- Wired to CLI with `--flash-attention` flag (enabled by default)

**Fix Details**:
1. Created `SpatialTransformerCubeCL`, `TransformerBlockCubeCL`, `CrossAttentionCubeCL`
2. Created `UNetCubeCL` that uses flash attention in all attention blocks
3. Added `run_sd1x_generate_flash` function in CLI
4. Flash attention uses `cubek-attention` kernel with f32 accumulation

**Files Modified**:
- `crates/burn-models-unet/src/cubecl.rs` - UNet and block types with flash attention
- `crates/burn-models-cli/src/main.rs` - Flash attention generation path
- `crates/burn-models-cli/Cargo.toml` - Added cubecl feature

**Usage**:
```bash
# Flash attention enabled by default for f16
cargo run -p burn-models-cli --features cuda -- generate --prompt "..." --weights /path

# Disable if needed (will likely produce NaN with f16)
cargo run -p burn-models-cli --features cuda -- generate --flash-attention=false ...
```

**Prevention**:
- Test attention implementations with f16 specifically, not just f32
- Test with large sequence lengths where overflow is more likely
- When using half-precision, prefer flash attention or algorithms with f32 accumulation
- Add tensor stats assertions in debug builds to catch NaN/inf early
- General: f16 has ~5 orders of magnitude less range than f32 - consider this in algorithm design

---

### [KNOWN] f16 Produces NaN in UNet Operations (Not Just Attention)

**Symptom**: f16 precision with flash attention produces NaN in output. Debug shows NaN in Q/K/V tensors BEFORE flash attention is called.

**Root Cause**: f16 overflow occurs in UNet operations BEFORE attention, not just in attention softmax.

Investigation found NaN appearing in:
1. Linear projections (to_q, to_k, to_v)
2. ResBlock operations (GroupNorm+SiLU+Conv)
3. Possibly other matmul operations

Flash attention with f32 accumulation only fixes overflow in the attention softmax computation itself. It cannot fix NaN that already exists in the input tensors.

**Attempted Fixes**:
1. ✅ Padding head_dim to power-of-2 (40→64) fixes cubek tiling assertion
2. ❌ But f16 still produces NaN because other operations overflow
3. ❌ Forced blueprint with tile_size=8 produces NaN (even with f32)
4. ❌ Forced blueprint with line_size=4 doesn't fix f16 NaN

**Why f32 Works**: f32 has ~5 orders of magnitude more range than f16. Operations that overflow in f16 remain within f32's representable range.

**Workaround**: Use f32 precision:
```bash
# f32 + flash attention (recommended)
cargo run -p burn-models-cli --features cuda -- generate --model sd1x --precision f32 --flash-attention true ...

# f32 without flash attention
cargo run -p burn-models-cli --features cuda -- generate --model sd1x --precision f32 --flash-attention false ...
```

**Files**:
- `crates/burn-models-unet/src/cubecl.rs` - CrossAttentionCubeCL with padding
- `crates/burn-models-cubecl/src/attention.rs` - flash_attention wrapper

**Future Work - How Existing Software Solves This**:

1. **ComfyUI's "upcast attention"**: Only compute Q@K^T in f32, keep everything else in f16:
   ```python
   if attn_precision == torch.float32:
       sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
   ```
   This targeted approach avoids overflow while maintaining speed.

2. **Diffusers recommends bfloat16**: bf16 has f32's exponent range (8 bits) but only
   8 bits of mantissa. It can represent the same magnitude as f32 without overflow.

3. **PyTorch autocast**: Automatically keeps LayerNorm, GroupNorm, Softmax in f32.

**Implementation Options for burn-models**:

| Approach | Effort | Status |
|----------|--------|--------|
| **bf16 support** | Low | burn-cubecl supports bf16 - just need to wire it up |
| **Upcast attention Q@K^T** | Medium | Compute similarity matrix in f32 only |
| **Upcast all Linear matmuls** | Medium | Keep nn::Linear forward in f32 |
| **Mixed precision pipeline** | High | Store weights f16, compute f32 |

**Recommended**: Try bf16 first - burn-cubecl already implements `FloatElement` and
`MatmulElement` for bf16. bf16 has f32's dynamic range so overflow is unlikely.

**Note**: SDXL uses `head_dim=64` (power-of-2) which avoids the cubek tiling issue. But SDXL may still have f16 overflow in other operations - needs testing.

---

### [KNOWN] cubek-attention BlackboxAccelerated wmma Compilation Error

**Symptom**: Using BlackboxAccelerated strategy produces CUDA compilation error:
```
error: incomplete type "nvcuda::wmma::fragment<...>" is not allowed
```

**Root Cause**: cubek-attention's BlackboxAccelerated routine uses WMMA (Tensor Core) operations that fail to compile on certain GPU/driver combinations.

This was observed on RTX 3060 (Ampere architecture, compute capability 8.6) which does support Tensor Cores, so the error is likely a cubek bug, not a hardware limitation.

**Workaround**: Use Unit strategy instead of BlackboxAccelerated (this is the default).

**Status**: Known cubek-attention issue. Unit strategy works correctly.

---

## Other Issues (Resolved)

### VAE num_res_blocks Mismatch

**Issue**: Config had `num_res_blocks=2`, safetensors had 3 blocks.

**Fix**: Changed `DecoderConfig::default()` to use 3.

**Prevention**:
- Validate model structure against reference implementation before hardcoding defaults
- Test: Load weights, check all layers have matching weight shapes

### GroupNorm Variance Calculation

**Issue**: Our GroupNorm used unbiased variance (n-1 divisor), PyTorch uses biased (n divisor).

**Fix**: Use `var_bias` for population variance.

**Prevention**:
- Document which variance formula reference implementations use
- Test: Compare layer output to PyTorch reference on same input

---

## Debug Tools

### --debug nan Flag

Use `--debug nan` to enable NaN/Inf checking at key points in the pipeline.
When a NaN or Inf value is detected, the program will panic with details:

```
[NaN check failed] step_0_noise_uncond: 1234/65536 values are NaN, 0/65536 are Inf
Stats: min=inf, max=-inf, mean=NaN, std=NaN [NaN=1234, Inf=0]
```

Checks are placed at:
- After CLIP encoding (text embeddings)
- After each UNet forward pass (noise predictions)
- After each sampling step (latent update)
- Before and after VAE decode

Usage:
```bash
cargo run -p burn-models-cli --features cuda -- generate --debug nan --weights /path/to/model ...
```

Debug modes can be combined: `--debug timing,nan,shapes` or use `--debug all`.

This is invaluable for debugging f16 overflow issues - it pinpoints exactly where NaN first appears.

---

## Testing Strategy (Derived from Issues)

These issues suggest the following testing priorities:

1. **Golden tests with reference outputs**: Compare our output to diffusers/PyTorch on identical inputs
2. **Weight completeness validation**: Assert all expected weights are loaded
3. **Determinism tests**: Run twice, outputs must be identical
4. **Precision-specific tests**: Test f16 and f32 separately
5. **Numeric stability tests**: Check for NaN/inf in intermediate tensors
