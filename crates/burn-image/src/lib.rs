//! burn-image: Stable Diffusion in pure Rust
//!
//! A pure Rust implementation of Stable Diffusion using the Burn deep learning framework.
//! Supports both SD 1.x and SDXL models.
//!
//! # Backend Selection
//!
//! Choose a backend via feature flags:
//! - `ndarray`: CPU backend (no GPU required)
//! - `tch`: PyTorch backend via libtorch (CUDA, MPS support)
//! - `wgpu`: WebGPU backend (cross-platform GPU)
//! - `cuda`: Native CUDA backend (NVIDIA only)
//!
//! # Example
//!
//! ```toml
//! [dependencies]
//! burn-image = { version = "0.1", features = ["wgpu"] }
//! ```
//!
//! ```ignore
//! use burn_image::{backends::Wgpu, StableDiffusionXL, SdxlSampleConfig};
//!
//! let device = burn_image::backends::default_device();
//! let tokenizer = burn_image::clip::ClipTokenizer::from_file("vocab.txt")?;
//! let pipeline = StableDiffusionXL::<Wgpu>::new(tokenizer, &device);
//!
//! let config = SdxlSampleConfig::default();
//! let image = pipeline.generate("a sunset over mountains", "", &config);
//! ```

pub use burn_image_core as core;
pub use burn_image_clip as clip;
pub use burn_image_vae as vae;
pub use burn_image_unet as unet;
pub use burn_image_samplers as samplers;
pub use burn_image_convert as convert;

pub mod backends;
pub mod memory;
pub mod pipeline;

pub use pipeline::{
    DiffusionPipeline, Img2ImgConfig, SampleConfig, Sd1xConditioning,
    StableDiffusion1x, StableDiffusion1xImg2Img, tensor_to_rgb,
    // Inpainting
    InpaintConfig, StableDiffusion1xInpaint,
    // SDXL
    SdxlConditioning, SdxlSampleConfig, StableDiffusionXL,
    SdxlImg2ImgConfig, StableDiffusionXLImg2Img,
    // SDXL Inpainting
    SdxlInpaintConfig, StableDiffusionXLInpaint,
    // SDXL Refiner
    RefinerConfig, StableDiffusionXLRefiner,
    // SDXL Base + Refiner Workflow
    BaseRefinerConfig, StableDiffusionXLWithRefiner,
};

pub use memory::{MemoryConfig, TiledVae};
