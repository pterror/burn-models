//! CubeCL GPU kernels for burn-models
//!
//! Custom GPU kernels for operations that benefit from fusion/specialization.
//! These kernels complement Burn's built-in operations with optimized implementations.
//!
//! # Supported Operations
//!
//! ## Attention
//! - `flash_attention` - GPU-accelerated flash attention (causal, O(n) memory)
//!
//! ## Conv3d
//! - `conv3d` - Simple kernel for NCTHW layout (works on non-contiguous data)
//! - `conv3d_nthwc` - Optimized kernel for NTHWC layout (1.5-2.6× faster for larger tensors)
//!
//! ## Pool3d
//! - `avg_pool3d` - 3D average pooling for temporal downsampling
//! - `max_pool3d` - 3D max pooling for temporal downsampling
//!
//! ## GroupNorm + SiLU
//! - `groupnorm_silu` - Fused group normalization + SiLU activation
//! - `groupnorm` - Group normalization (without SiLU)
//!
//! # Backend Selection
//!
//! Backend selection (wgpu, cuda) is handled by the consuming crate
//! via burn-wgpu or burn-cuda.

mod attention;
mod conv3d;
mod conv3d_optimized;
mod groupnorm_silu;
mod pool3d;
mod utils;

pub use attention::{flash_attention, flash_attention_masked, FlashAttentionOptions, AttentionSetupError};
pub use conv3d::{conv3d, Conv3dLayer, Conv3dOptions, Layout, from_cube_tensor, to_cube_tensor};
pub use conv3d_optimized::{conv3d_nthwc, Conv3dOptimizedOptions};
pub use groupnorm_silu::{
    groupnorm, groupnorm_silu, GroupNormSiLuOptions, GroupNormSiLuLayer,
    tensor_to_cube, cube_to_tensor,
};
pub use pool3d::{avg_pool3d, max_pool3d, Pool3dOptions};

// Re-export key types for consumers
pub use burn_cubecl::{tensor::CubeTensor, CubeRuntime};
pub use cubecl::server::LaunchError;
