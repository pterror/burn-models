# burn-image

**Stable Diffusion inference in pure Rust with Burn.**

burn-image provides a complete implementation of Stable Diffusion (1.x and XL) using the [Burn](https://burn.dev) deep learning framework.

## Features

- **Pure Rust** - No Python dependencies, runs anywhere Rust runs
- **Multiple Backends** - WGPU, CUDA, and more via Burn
- **SD 1.x & SDXL** - Support for both model families
- **Many Samplers** - DPM++, Euler, DDIM, LCM, and more
- **LoRA Support** - Load and apply LoRA weights
- **ControlNet** - Conditional generation with ControlNet
- **Textual Inversion** - Custom embeddings support

## Quick Start

```rust
use burn_image::prelude::*;

// Load a model and generate an image
let pipeline = StableDiffusion1x::load("model.safetensors", &device)?;

let image = pipeline.generate(
    "a photo of a cat",
    "",
    &SampleConfig::default(),
);
```

## Crates

| Crate | Description |
|-------|-------------|
| `burn-image` | Main pipeline and high-level API |
| `burn-image-core` | Core building blocks (attention, normalization) |
| `burn-image-unet` | UNet architecture for SD 1.x and SDXL |
| `burn-image-vae` | VAE encoder and decoder |
| `burn-image-clip` | CLIP and OpenCLIP text encoders |
| `burn-image-samplers` | Diffusion samplers (DPM++, Euler, etc.) |
| `burn-image-convert` | Weight loading from safetensors |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
burn-image = "0.1"
```
