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
mod conv3d_optimized;
mod utils;

pub use conv3d::{conv3d, Conv3dLayer, Conv3dOptions, Layout, from_cube_tensor, to_cube_tensor};
pub use conv3d_optimized::{conv3d_nthwc, Conv3dOptimizedOptions};

// Re-export key types for consumers
pub use burn_cubecl::{tensor::CubeTensor, CubeRuntime};
pub use cubecl::server::LaunchError;
