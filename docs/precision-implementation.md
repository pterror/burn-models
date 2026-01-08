# Precision Implementation Analysis

Technical analysis of implementing each precision strategy.

## 1. bf16 Support

### Requirements
- Add `Bf16` variant to CLI `Precision` enum
- Add match arms using `half::bf16` type
- Use `Cuda<bf16>` or `CubeBackend<CudaRuntime, bf16, i32, u32>`

### Code Changes
```rust
// In main.rs
pub enum Precision {
    F32,
    F16,
    Bf16,  // NEW
}

// In match precision block
Precision::Bf16 => {
    use half::bf16;
    type Backend = Cuda<bf16>;
    run_sd1x_generate_impl::<Backend>(...)
}
```

### Blockers
- **GPU requirement**: bf16 native support requires SM 80+ (Ampere+)
  - RTX 30xx, RTX 40xx: Native bf16
  - RTX 20xx: Emulated (slower than f16)
  - GTX 16xx: May not work
- **Testing needed**: Verify burn-cubecl bf16 works on CUDA

### Effort: Low (few hours)

---

## 2. Upcast Attention (Q@K^T in f32)

### Requirements
- Modify `CrossAttention::forward()` to cast Q, K to f32 before matmul
- Add `--upcast-attention` CLI flag
- Keep weights in original dtype

### Code Changes
```rust
// In blocks.rs CrossAttention::forward()
pub fn forward(&self, x: Tensor<B, 3>, context: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
    let q = self.to_q.forward(x);
    let k = self.to_k.forward(context);
    let v = self.to_v.forward(context);

    // Upcast Q and K for attention score computation
    let q_f32 = q.clone().float();  // Cast to f32
    let k_f32 = k.clone().float();

    // Compute attention in f32
    let scale = 1.0 / (self.head_dim as f32).sqrt();
    let attn = q_f32.matmul(k_f32.transpose()) * scale;
    let attn = softmax(attn, -1);

    // V matmul can stay in original dtype
    let out = attn.matmul(v);
    ...
}
```

### Blockers
- **Type system**: Burn tensors are typed, need to cast between `Tensor<B, D>` where B has different float types
- **May need**: Generic over precision or use `into_data().convert::<f32>()` approach
- **Performance**: Extra memory for f32 intermediates

### Effort: Medium (1-2 days)

---

## 3. Upcast All Matmuls

### Requirements
- Create `LinearUpcast<B>` wrapper that:
  - Stores weights in f16/bf16
  - Computes forward pass in f32
- Replace `nn::Linear` usage

### Code Changes
```rust
pub struct LinearUpcast<B: Backend> {
    weight: Tensor<B, 2>,  // Stored in f16
    bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> LinearUpcast<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Convert to f32 for computation
        let x_f32 = x.float();
        let w_f32 = self.weight.clone().float();

        let out = x_f32.matmul(w_f32.transpose());

        // Convert back to original dtype
        out.half()  // or original dtype
    }
}
```

### Blockers
- **Invasive**: Need to change all model definitions
- **Weight loading**: Need to handle precision conversion during load
- **Performance**: 2x memory traffic (convert in, convert out)

### Effort: Medium-High (2-3 days)

---

## 4. Mixed Precision Pipeline

### Requirements
- Different precision for UNet, VAE, CLIP
- Separate backend type parameters
- Complex type system changes

### Code Changes
Would require significant refactoring:
- Models need to be generic over precision
- Weight loaders need precision parameter
- Pipeline needs to handle multiple backend types

### Blockers
- **Type complexity**: Rust's type system makes mixing precisions hard
- **Large refactor**: Would touch most of the codebase
- **Testing**: Many combinations to test

### Effort: High (1+ weeks)

---

## Recommended Implementation Order

1. **bf16** - Easiest, biggest impact, test first
2. **Upcast attention** - If bf16 doesn't work on target GPU
3. **Upcast matmuls** - If attention upcast isn't enough
4. **Mixed precision** - Future work, low priority

---

## Cargo Feature Flag Design

Goal: Allow users to compile only the precision paths they need, avoiding binary bloat.

### Feature Structure (Cargo.toml)

```toml
[features]
# Default: all precisions available (maximum flexibility)
default = ["precision-all", "cubecl"]

# Individual precision support (additive)
precision-f32 = []      # Enable f32 precision
precision-f16 = []      # Enable f16 precision
precision-bf16 = []     # Enable bf16 precision
precision-all = ["precision-f32", "precision-f16", "precision-bf16"]

# Convenience presets (compile minimal code paths)
preset-fast = ["precision-bf16", "cubecl"]   # bf16 + flash attention only
preset-memory = ["precision-f16", "cubecl"]  # f16 + flash only (needs upcast work)
preset-quality = ["precision-f32"]            # f32 only, no JIT overhead
```

### Usage Examples

```bash
# Default: all precisions, can switch at runtime
cargo build -p burn-models-cli --features cuda

# Fast preset: only bf16+flash, minimal binary
cargo build -p burn-models-cli --no-default-features --features cuda,preset-fast

# Quality preset: f32 only, no half-precision code
cargo build -p burn-models-cli --no-default-features --features cuda,preset-quality

# Custom: just f32 and bf16, no f16
cargo build -p burn-models-cli --no-default-features --features cuda,precision-f32,precision-bf16,cubecl
```

### Code Conditional Compilation

```rust
// In main.rs Precision enum
pub enum Precision {
    #[cfg(feature = "precision-f32")]
    F32,
    #[cfg(feature = "precision-f16")]
    F16,
    #[cfg(feature = "precision-bf16")]
    Bf16,
}

// In match blocks
match precision {
    #[cfg(feature = "precision-f16")]
    Precision::F16 => { ... }

    #[cfg(feature = "precision-f32")]
    Precision::F32 => { ... }

    #[cfg(feature = "precision-bf16")]
    Precision::Bf16 => { ... }
}
```

### Binary Size Impact (Estimated)

| Configuration | Approx Binary Delta |
|--------------|---------------------|
| precision-all + cubecl | Baseline |
| preset-fast (bf16 only) | -20% |
| preset-quality (f32 only) | -30% (no half crate) |

### CLI Flag Design

Runtime selection (within compiled options):
```
--precision f32|f16|bf16
--flash-attention (enabled by default)
--upcast-attention (future, for f16 stability)
```

---

## GPU Compatibility Matrix

| GPU | f32 | bf16 | f16+upcast | f16 |
|-----|-----|------|------------|-----|
| RTX 40xx | Best | Native | Good | NaN |
| RTX 30xx | Good | Native | Good | NaN |
| RTX 20xx | Good | Emulated | Good | NaN |
| GTX 16xx | Good | Unknown | Good | NaN |
| GTX 10xx | Good | No | Good | NaN |
