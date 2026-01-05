# Architecture

## Crate Structure

```
burn-models/
├── crates/
│   ├── burn-models/          # Main library, re-exports everything
│   ├── burn-models-core/     # Shared layers and utilities
│   ├── burn-models-clip/     # Text encoders (CLIP, OpenCLIP)
│   ├── burn-models-vae/      # Autoencoder (encoder + decoder)
│   ├── burn-models-unet/     # UNet diffusion models (SD 1.x, SDXL)
│   ├── burn-models-samplers/ # Diffusion samplers (DDIM, DPM++, Euler, etc.)
│   └── burn-models-convert/  # Weight conversion from safetensors
└── src/                     # CLI binary
```

## Module Breakdown

### burn-models-core

Shared building blocks:

- `attention.rs` - Multi-head attention, cross-attention
- `groupnorm.rs` - Group normalization
- `layernorm.rs` - Layer normalization
- `silu.rs` - SiLU activation
- `conv.rs` - Convolution helpers
- `linear.rs` - Linear layer helpers
- `timestep.rs` - Timestep embeddings

### burn-models-clip

Text encoding:

- `tokenizer.rs` - BPE tokenizer (pure Rust, no Python)
- `clip.rs` - CLIP text encoder (SD 1.x)
- `open_clip.rs` - OpenCLIP encoder (SDXL)
- `embedder.rs` - Unified interface for single/dual encoders

### burn-models-vae

Variational Autoencoder:

- `encoder.rs` - Image → latent (for img2img)
- `decoder.rs` - Latent → image
- `autoencoder.rs` - Combined VAE

### burn-models-unet

Diffusion backbone:

- `blocks.rs` - ResNet blocks, attention blocks, down/up blocks
- `unet_sd.rs` - SD 1.x UNet
- `unet_sdxl.rs` - SDXL UNet (larger, different architecture)
- `conditioning.rs` - Conditioning types

### burn-models-samplers

Noise schedulers and samplers:

- `scheduler.rs` - Noise schedule (linear, cosine, etc.)
- `ddim.rs` - DDIM sampler
- `ddpm.rs` - DDPM sampler
- `dpm.rs` - DPM++ variants
- `euler.rs` - Euler/Euler ancestral

### burn-models-convert

Weight conversion:

- `safetensors.rs` - Load .safetensors files
- `mapping.rs` - Map weight names to our architecture
- `sd.rs` - SD 1.x weight mapping
- `sdxl.rs` - SDXL weight mapping

## Pipeline Abstraction

```rust
pub trait DiffusionPipeline<B: Backend> {
    type Conditioning;

    fn encode_prompt(&self, prompt: &str, negative: &str) -> Self::Conditioning;
    fn sample(&self, conditioning: Self::Conditioning, config: SampleConfig) -> Tensor<B, 4>;
    fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4>;
}
```

Both SD 1.x and SDXL implement this trait, enabling unified usage.

## Design Principles

1. **Shared layers** - All common operations in `core`, no duplication
2. **Composable** - Each crate usable independently
3. **Pure Rust** - No Python dependencies
4. **Backend agnostic** - Works with any Burn backend (CUDA, Metal, WGPU, CPU)
5. **Feature flags** - Enable only what you need (`sd`, `sdxl`, `samplers-all`)
