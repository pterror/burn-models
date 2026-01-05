# Getting Started

## Installation

Add burn-image to your `Cargo.toml`:

```toml
[dependencies]
burn-image = "0.1"

# Choose a backend
burn-wgpu = "0.16"  # GPU via WebGPU (cross-platform)
# or
burn-cuda = "0.16"  # NVIDIA CUDA
# or
burn-ndarray = "0.16"  # CPU fallback
```

## Basic Usage

### Text-to-Image Generation

```rust
use burn_image::prelude::*;
use burn_wgpu::{Wgpu, WgpuDevice};

type Backend = Wgpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();

    // Load tokenizer vocabulary
    let tokenizer = ClipTokenizer::from_file("bpe_simple_vocab_16e6.txt")?;

    // Create pipeline (weights loaded separately)
    let pipeline = StableDiffusion1x::new(tokenizer, &device);

    // Generate an image
    let config = SampleConfig {
        width: 512,
        height: 512,
        steps: 20,
        guidance_scale: 7.5,
        seed: Some(42),
    };

    let image = pipeline.generate(
        "a photograph of an astronaut riding a horse",
        "",  // negative prompt
        &config,
    );

    // Convert to RGB bytes
    let rgb = tensor_to_rgb(image);

    Ok(())
}
```

### Using SDXL

```rust
use burn_image::prelude::*;

let pipeline = StableDiffusionXL::new(tokenizer, &device);

let config = SdxlSampleConfig {
    width: 1024,
    height: 1024,
    steps: 30,
    guidance_scale: 7.5,
    seed: None,
};

let image = pipeline.generate(prompt, negative_prompt, &config);
```

## Loading Weights

Weights are loaded from safetensors files:

```rust
use burn_image_convert::WeightLoader;

// Load from a single safetensors file
let loader = WeightLoader::open("model.safetensors")?;

// Or from a directory with multiple files
let loader = WeightLoader::open_dir("model/")?;

// Apply weights to the pipeline
loader.load_unet(&mut pipeline.unet)?;
loader.load_text_encoder(&mut pipeline.text_encoder)?;
loader.load_vae(&mut pipeline.vae_decoder)?;
```

## Choosing a Sampler

The default sampler is DDIM, but you can use others:

```rust
use burn_image_samplers::*;

// DPM++ 2M - fast and high quality
let sampler = DpmPlusPlusSampler::new(DpmConfig {
    num_inference_steps: 20,
    use_karras_sigmas: true,
    ..Default::default()
}, &schedule);

// Euler Ancestral - good variety
let sampler = EulerAncestralSampler::new(EulerConfig {
    num_inference_steps: 25,
    ..Default::default()
}, &schedule);

// LCM - very fast (4-8 steps)
let sampler = LcmSampler::new(LcmConfig {
    num_inference_steps: 4,
    ..Default::default()
}, &schedule);
```

See [Samplers](/samplers) for a complete comparison.

## Image-to-Image

Transform an existing image:

```rust
let pipeline = StableDiffusion1xImg2Img::new(tokenizer, &device);

let config = Img2ImgConfig {
    steps: 30,
    guidance_scale: 7.5,
    strength: 0.75,  // 0.0 = no change, 1.0 = full regeneration
    seed: None,
};

let output = pipeline.generate(
    input_image,  // [1, 3, H, W] tensor
    "oil painting style",
    "",
    &config,
);
```

## Inpainting

Regenerate specific regions:

```rust
let pipeline = StableDiffusion1xInpaint::new(tokenizer, &device);

let config = InpaintConfig {
    steps: 30,
    guidance_scale: 7.5,
    seed: None,
};

let output = pipeline.inpaint(
    image,  // [1, 3, H, W]
    mask,   // [1, 1, H, W] where 1 = regenerate
    "a red car",
    "",
    &config,
);
```

## Next Steps

- [Samplers](/samplers) - Detailed sampler comparison
- [Pipelines](/pipelines) - All pipeline types
- [Architecture](/architecture) - Crate structure
