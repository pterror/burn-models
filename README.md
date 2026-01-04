# burn-image

Stable Diffusion image generation in pure Rust using the [Burn](https://burn.dev) deep learning framework.

## Features

- **Pure Rust** - No Python dependencies, no ONNX runtime
- **Multiple Models** - Support for SD 1.x, SDXL, and SDXL + Refiner
- **Multiple Backends** - CPU (ndarray), CUDA, WebGPU, and PyTorch via libtorch
- **Multiple Pipelines** - Text-to-image, img2img, and inpainting
- **Memory Efficient** - VAE tiling for large images

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
burn-image = { version = "0.1", features = ["wgpu"] }
```

Available backend features:
- `ndarray` - CPU backend (no GPU required)
- `tch` - PyTorch backend via libtorch (CUDA, MPS support)
- `wgpu` - WebGPU backend (cross-platform GPU)
- `cuda` - Native CUDA backend (NVIDIA only)

## Quick Start

```rust
use burn_image::{
    backends::{Wgpu, WgpuDevice},
    clip::ClipTokenizer,
    StableDiffusionXL, SdxlSampleConfig,
};

fn main() -> anyhow::Result<()> {
    // Initialize backend
    let device = WgpuDevice::default();

    // Load tokenizer
    let tokenizer = ClipTokenizer::from_file("vocab.txt")?;

    // Create pipeline
    let pipeline = StableDiffusionXL::<Wgpu>::new(tokenizer, &device);

    // Configure generation
    let config = SdxlSampleConfig {
        width: 1024,
        height: 1024,
        steps: 30,
        guidance_scale: 7.5,
        seed: None,
    };

    // Generate image
    let image = pipeline.generate(
        "a sunset over mountains, dramatic lighting",
        "",  // negative prompt
        &config,
    );

    Ok(())
}
```

## CLI Usage

Install the CLI:

```bash
cargo install burn-image-cli
```

Generate an image:

```bash
burn-image generate \
    --prompt "a cat sitting on a windowsill" \
    --output cat.png \
    --model sdxl \
    --vocab vocab.txt \
    --weights ./models/sdxl
```

Show available backends:

```bash
burn-image info
```

## Pipelines

### Text-to-Image

Generate images from text prompts:

```rust
use burn_image::{StableDiffusion1x, SampleConfig};

let config = SampleConfig::default();
let image = pipeline.generate("a beautiful landscape", "", &config);
```

### Image-to-Image

Transform existing images based on prompts:

```rust
use burn_image::{StableDiffusion1xImg2Img, Img2ImgConfig};

let config = Img2ImgConfig {
    strength: 0.75,
    steps: 30,
    guidance_scale: 7.5,
    seed: None,
};
let transformed = pipeline.generate(input_image, "oil painting style", "", &config);
```

### Inpainting

Regenerate masked regions of an image:

```rust
use burn_image::{StableDiffusion1xInpaint, InpaintConfig};

let config = InpaintConfig::default();
// mask: 1 = regenerate, 0 = preserve
let inpainted = pipeline.inpaint(image, mask, "a red rose", "", &config);
```

### SDXL with Refiner

Highest quality generation using base + refiner workflow:

```rust
use burn_image::{StableDiffusionXLWithRefiner, BaseRefinerConfig};

let config = BaseRefinerConfig {
    width: 1024,
    height: 1024,
    steps: 40,
    refiner_start: 0.8,  // Refiner takes over at 80%
    ..Default::default()
};

let image = pipeline.generate("detailed portrait", "", &config);
```

## Memory Optimization

For systems with limited VRAM, enable VAE tiling:

```rust
use burn_image::{MemoryConfig, TiledVae};

let config = MemoryConfig::low_vram();
let tiler = TiledVae::new(config.vae_tile_size, config.vae_tile_overlap);
```

## Architecture

The library is organized into several crates:

- `burn-image` - Main crate with pipelines and re-exports
- `burn-image-core` - Core tensor operations (attention, normalization)
- `burn-image-clip` - CLIP/OpenCLIP text encoders
- `burn-image-vae` - VAE encoder/decoder
- `burn-image-unet` - UNet diffusion backbone
- `burn-image-samplers` - DDIM and other samplers
- `burn-image-convert` - Safetensors weight loading
- `burn-image-cli` - Command-line interface

## Model Weights

This library loads weights from safetensors format. You'll need to obtain weights
from sources like:

- [Hugging Face Hub](https://huggingface.co/models?library=diffusers)
- [Civitai](https://civitai.com/)

## License

MIT OR Apache-2.0
