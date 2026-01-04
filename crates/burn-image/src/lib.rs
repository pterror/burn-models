//! burn-image: Stable Diffusion in pure Rust
//!
//! Re-exports all sub-crates for convenience.

pub use burn_image_core as core;
pub use burn_image_clip as clip;
pub use burn_image_vae as vae;
pub use burn_image_unet as unet;
pub use burn_image_samplers as samplers;
pub use burn_image_convert as convert;

pub mod pipeline;

pub use pipeline::{
    DiffusionPipeline, Img2ImgConfig, SampleConfig, Sd1xConditioning,
    StableDiffusion1x, StableDiffusion1xImg2Img, tensor_to_rgb,
};
