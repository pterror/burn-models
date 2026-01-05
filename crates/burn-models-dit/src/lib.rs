//! DiT-based Diffusion Models
//!
//! This crate provides implementations of DiT-based (Diffusion Transformer)
//! image generation models.
//!
//! # Supported Models
//!
//! - **Flux**: Black Forest Labs' flow-matching DiT model
//!   - Flux.1-dev (guidance-distilled)
//!   - Flux.1-schnell (few-step)
//!
//! # Architecture
//!
//! DiT models differ from UNet-based diffusion models:
//! - Use transformer blocks instead of convolutional U-Net
//! - Adaptive layer norm (AdaLN) conditioned on timestep
//! - Bidirectional attention (no causal mask)
//! - Flow matching objective (vs noise prediction)

pub mod flux;
pub mod flux_loader;

pub use flux::{Flux, FluxConfig, FluxOutput, FluxRuntime};
pub use flux_loader::{load_flux, FluxLoadError};
