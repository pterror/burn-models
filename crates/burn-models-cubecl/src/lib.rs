//! CubeCL GPU kernels for burn-models
//!
//! Custom GPU kernels for operations that benefit from fusion/specialization.
//! These kernels complement Burn's built-in operations with optimized implementations.
//!
//! # Supported Operations
//!
//! - `Conv3d` - 3D convolution for video models (3D VAE, etc.)
//!
//! # Backend Selection
//!
//! Backend selection (wgpu, cuda) is handled by the consuming crate
//! via burn-wgpu or burn-cuda.

mod conv3d;
mod utils;

pub use conv3d::{conv3d, Conv3dOptions};
