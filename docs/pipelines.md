# Pipelines

burn-image provides several pipeline types for different generation tasks.

## Pipeline Overview

| Pipeline | Model | Task | Input | Output |
|----------|-------|------|-------|--------|
| `StableDiffusion1x` | SD 1.x | txt2img | Text prompt | Image |
| `StableDiffusion1xImg2Img` | SD 1.x | img2img | Image + text | Image |
| `StableDiffusion1xInpaint` | SD 1.x | inpainting | Image + mask + text | Image |
| `StableDiffusionXL` | SDXL | txt2img | Text prompt | Image |
| `StableDiffusionXLImg2Img` | SDXL | img2img | Image + text | Image |
| `StableDiffusionXLInpaint` | SDXL | inpainting | Image + mask + text | Image |
| `StableDiffusionXLRefiner` | SDXL Refiner | refinement | Latent + text | Image |
| `StableDiffusionXLWithRefiner` | SDXL + Refiner | txt2img | Text prompt | Image |

## Text-to-Image

### SD 1.x

```rust
use burn_image::prelude::*;

let pipeline = StableDiffusion1x::new(tokenizer, &device);

let config = SampleConfig {
    width: 512,
    height: 512,
    steps: 20,
    guidance_scale: 7.5,
    seed: Some(42),
};

let image = pipeline.generate(
    "a majestic lion in the savanna",
    "blurry, low quality",  // negative prompt
    &config,
);
```

### SDXL

SDXL produces higher quality 1024x1024 images using dual text encoders.

```rust
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

### SDXL with Refiner

The refiner improves fine details in SDXL outputs.

```rust
let pipeline = StableDiffusionXLWithRefiner::new(
    base_tokenizer,
    refiner_tokenizer,
    &device,
);

let config = BaseRefinerConfig {
    width: 1024,
    height: 1024,
    steps: 40,
    base_guidance_scale: 7.5,
    refiner_guidance_scale: 7.5,
    refiner_start: 0.8,  // Refiner takes over at 80%
    seed: None,
};

let image = pipeline.generate(prompt, negative_prompt, &config);
```

## Image-to-Image

Transform an existing image based on a text prompt.

### SD 1.x

```rust
let pipeline = StableDiffusion1xImg2Img::new(tokenizer, &device);

let config = Img2ImgConfig {
    steps: 30,
    guidance_scale: 7.5,
    strength: 0.75,
    seed: None,
};

// input_image: [1, 3, H, W] tensor with values 0-255
let output = pipeline.generate(input_image, prompt, negative_prompt, &config);
```

### Strength Parameter

The `strength` parameter controls how much the image changes:
- `0.0` - No change (output = input)
- `0.5` - Moderate transformation
- `0.75` - Significant change (recommended)
- `1.0` - Full regeneration (same as txt2img)

### SDXL

```rust
let pipeline = StableDiffusionXLImg2Img::new(tokenizer, &device);

let config = SdxlImg2ImgConfig {
    steps: 30,
    guidance_scale: 7.5,
    strength: 0.75,
    seed: None,
};

let output = pipeline.generate(input_image, prompt, negative_prompt, &config);
```

## Inpainting

Regenerate specific regions of an image based on a mask.

### Mask Format

- Shape: `[1, 1, H, W]`
- Values: `0.0` = preserve, `1.0` = regenerate
- Same dimensions as input image

### SD 1.x

```rust
let pipeline = StableDiffusion1xInpaint::new(tokenizer, &device);

let config = InpaintConfig {
    steps: 30,
    guidance_scale: 7.5,
    seed: None,
};

let output = pipeline.inpaint(
    image,   // [1, 3, H, W]
    mask,    // [1, 1, H, W]
    "a red sports car",
    "",
    &config,
);
```

### SDXL

```rust
let pipeline = StableDiffusionXLInpaint::new(tokenizer, &device);

let config = SdxlInpaintConfig {
    steps: 30,
    guidance_scale: 7.5,
    seed: None,
};

let output = pipeline.inpaint(image, mask, prompt, negative_prompt, &config);
```

## The DiffusionPipeline Trait

All txt2img pipelines implement a common trait:

```rust
pub trait DiffusionPipeline<B: Backend> {
    type Conditioning;

    /// Encode text prompt into conditioning
    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Self::Conditioning;

    /// Sample latent from conditioning
    fn sample_latent(&self, conditioning: &Self::Conditioning, config: &SampleConfig) -> Tensor<B, 4>;

    /// Decode latent to image
    fn decode(&self, latent: Tensor<B, 4>) -> Tensor<B, 4>;

    /// Full pipeline: prompt -> image
    fn generate(&self, prompt: &str, negative_prompt: &str, config: &SampleConfig) -> Tensor<B, 4>;
}
```

This allows writing generic code that works with any pipeline:

```rust
fn generate_with_progress<B: Backend, P: DiffusionPipeline<B>>(
    pipeline: &P,
    prompt: &str,
    config: &SampleConfig,
) -> Tensor<B, 4> {
    println!("Encoding prompt...");
    let conditioning = pipeline.encode_prompt(prompt, "");

    println!("Sampling...");
    let latent = pipeline.sample_latent(&conditioning, config);

    println!("Decoding...");
    pipeline.decode(latent)
}
```

## Image Tensor Format

### Input Images

- Shape: `[batch, channels, height, width]` = `[1, 3, H, W]`
- Values: `0.0` to `255.0` (RGB)
- Channel order: RGB

### Output Images

- Same format as input
- Use `tensor_to_rgb()` to convert to bytes:

```rust
use burn_image::pipeline::tensor_to_rgb;

let rgb_bytes: Vec<u8> = tensor_to_rgb(output_tensor);
// rgb_bytes is H*W*3 bytes in RGB order
```

## Pipeline Components

Each pipeline contains:

| Component | Description |
|-----------|-------------|
| `tokenizer` | BPE tokenizer for text encoding |
| `text_encoder` | CLIP (and OpenCLIP for SDXL) |
| `unet` | Diffusion backbone |
| `vae_encoder` | Image → latent (img2img/inpaint only) |
| `vae_decoder` | Latent → image |
| `scheduler` | Noise schedule |

Access components directly for custom workflows:

```rust
// Custom text encoding
let tokens = pipeline.tokenizer.encode_padded(text, 77);

// Direct UNet inference
let noise_pred = pipeline.unet.forward(latent, timestep, context);

// Manual VAE decoding
let image = pipeline.vae_decoder.decode_to_image(latent);
```
