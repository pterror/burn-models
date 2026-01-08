# Precision Strategies

This document outlines precision options for inference and how to choose the right one.

## Background: Why f16 Fails

Standard float16 (f16) has:
- 1 sign bit, 5 exponent bits, 10 mantissa bits
- Range: ~6e-5 to 65504
- When intermediate values exceed ~65504, they become Inf/NaN

In Stable Diffusion UNet:
- Attention Q@K^T matmul can produce large values
- GroupNorm variance computation accumulates large sums
- Linear layer outputs can exceed f16 range

## Precision Options

### 1. f32 (Full Precision)
- **Memory**: 100% (baseline)
- **Speed**: Slowest
- **Quality**: Best
- **Stability**: Perfect

### 2. bf16 (Brain Float16)
- **Memory**: 50%
- **Speed**: Fast (same as f16)
- **Quality**: Slightly lower than f32
- **Stability**: Excellent (f32's exponent range)

bf16 has 1 sign, 8 exponent, 7 mantissa bits. The 8 exponent bits give it
the same dynamic range as f32 (~1e-38 to ~3e38), avoiding overflow entirely.

### 3. f16 + Upcast Attention
- **Memory**: ~55% (f16 weights + f32 attention intermediates)
- **Speed**: Fast
- **Quality**: Good
- **Stability**: Good

Only the Q@K^T similarity matrix is computed in f32. This is ComfyUI's approach:
```python
sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
```

### 4. f16 + Upcast All Matmuls
- **Memory**: ~60% (f16 weights + f32 matmul intermediates)
- **Speed**: Medium
- **Quality**: Good
- **Stability**: Very good

All nn::Linear operations compute in f32, only weights stored in f16.

### 5. Mixed Precision Pipeline
- **Memory**: Variable
- **Speed**: Variable
- **Quality**: Configurable
- **Stability**: Configurable

Different components use different precision:
- UNet: f16 or bf16
- VAE: f32 (benefits from precision for image quality)
- CLIP: f16 (text encoding is stable)

## How Other Software Handles This

### ComfyUI
- `--force-upcast-attention`: Upcast Q@K^T to f32
- `--dont-upcast-attention`: Disable (may cause black images)
- `--fp16-vae`: Run VAE in f16 (may cause issues)
- Default: Automatic detection based on model/hardware

### Diffusers (HuggingFace)
- Recommends bf16 over f16
- Uses PyTorch autocast for automatic mixed precision
- SDPA (scaled_dot_product_attention) handles attention efficiently

### PyTorch Autocast
Automatically keeps these in f32:
- LayerNorm, GroupNorm, BatchNorm
- Softmax, LogSoftmax
- Cross-entropy loss
- Small matrix multiplications

## Implementation Plan for burn-models

### Phase 1: bf16 Support (Low Effort, High Impact)
- Add `--precision bf16` CLI flag
- Wire up `CubeBackend<CudaRuntime, bf16, i32, u32>`
- Test with existing pipeline

### Phase 2: Upcast Attention (Medium Effort)
- Modify CrossAttention to cast Q/K to f32 before matmul
- Add `--upcast-attention` flag
- Works with f16 weights

### Phase 3: Upcast All Matmuls (Medium Effort)
- Create wrapper for nn::Linear that upcasts inputs
- Store weights in f16, compute in f32
- Add `--upcast-matmul` flag

### Phase 4: Mixed Precision Pipeline (High Effort)
- Separate precision for UNet, VAE, CLIP
- `--unet-precision`, `--vae-precision`, `--clip-precision` flags
- Automatic precision selection based on model

## Recommended Presets

| Preset | Implementation | Use Case |
|--------|---------------|----------|
| `--preset quality` | f32 everything | Best output, slow |
| `--preset balanced` | bf16 or f32+flash | Good quality, good speed |
| `--preset fast` | bf16 + flash attention | Fast, slight quality loss |
| `--preset memory` | f16 + upcast attention | Low VRAM GPUs |

## Current Status

| Approach | Status | Notes |
|----------|--------|-------|
| f32 | Working | Stable, slowest |
| f32 + flash attention | Working | Recommended for quality |
| bf16 | Implemented | Default. Fast, stable on Ampere+ (RTX 30xx/40xx) |
| bf16 + flash attention | Implemented | Best balance of speed/memory |
| f16 | NaN issues | Overflow in UNet operations |
| f16 + flash attention | NaN issues | Flash attention works, but earlier ops overflow |
| Upcast attention | Not implemented | Needed to make f16 work |

## Cargo Feature Flags

Compile only the precision paths you need:

```bash
# Default: all precisions (maximum flexibility)
cargo build -p burn-models-cli --features cuda

# Fast preset: bf16 + flash attention only (minimal binary)
cargo build -p burn-models-cli --no-default-features --features cuda,preset-fast

# Quality preset: f32 only (no JIT overhead)
cargo build -p burn-models-cli --no-default-features --features cuda,preset-quality

# Custom: specific precisions
cargo build -p burn-models-cli --no-default-features --features cuda,precision-f32,precision-bf16,cubecl
```

Available precision features:
- `precision-f32` - 32-bit float (slowest, most stable)
- `precision-f16` - 16-bit float (fast, may NaN without upcast)
- `precision-bf16` - Brain float16 (fast, stable on Ampere+)
- `precision-all` - All precisions (default)
- `preset-fast` - bf16 + flash attention only
- `preset-quality` - f32 only

## References

- [Diffusers FP16 Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)
- [ComfyUI Precision Options](https://docs.comfy.org/interface/settings/server-config)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
