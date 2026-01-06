//! Backend support for burn-models
//!
//! This module provides convenient access to different Burn backends.
//! Enable the desired backend via feature flags:
//!
//! - `ndarray`: CPU backend using ndarray (no GPU required)
//! - `tch`: PyTorch backend via libtorch (supports CUDA, MPS)
//! - `wgpu`: WebGPU backend (cross-platform GPU support)
//! - `cuda`: Native CUDA backend (NVIDIA GPUs only)
//!
//! # Example
//!
//! ```toml
//! [dependencies]
//! burn-models = { version = "0.1", features = ["wgpu"] }
//! ```
//!
//! ```ignore
//! use burn_models::backends::Wgpu;
//! use burn_models::StableDiffusionXL;
//!
//! let device = burn_models::backends::WgpuDevice::default();
//! let pipeline = StableDiffusionXL::<Wgpu>::new(tokenizer, &device);
//! ```

#[cfg(feature = "ndarray")]
pub use burn_ndarray::{NdArray, NdArrayDevice};

#[cfg(feature = "tch")]
pub use burn_tch::{LibTorch, LibTorchDevice};

#[cfg(feature = "wgpu")]
pub use burn_wgpu::{Wgpu, WgpuDevice};

#[cfg(feature = "cuda")]
pub use burn_cuda::{Cuda, CudaDevice};

/// Type alias for the default backend when using ndarray feature
#[cfg(feature = "ndarray")]
pub type DefaultBackend = NdArray;

/// Type alias for the default backend when using tch feature
#[cfg(all(feature = "tch", not(feature = "ndarray")))]
pub type DefaultBackend = LibTorch;

/// Type alias for the default backend when using wgpu feature
#[cfg(all(feature = "wgpu", not(any(feature = "ndarray", feature = "tch"))))]
pub type DefaultBackend = Wgpu;

/// Type alias for the default backend when using cuda feature
#[cfg(all(
    feature = "cuda",
    not(any(feature = "ndarray", feature = "tch", feature = "wgpu"))
))]
pub type DefaultBackend = Cuda;

/// Get the default device for the enabled backend
#[cfg(feature = "ndarray")]
pub fn default_device() -> NdArrayDevice {
    NdArrayDevice::default()
}

/// Get the default device for the enabled backend
#[cfg(all(feature = "tch", not(feature = "ndarray")))]
pub fn default_device() -> LibTorchDevice {
    // Try to use CUDA if available, otherwise CPU
    if burn_tch::is_cuda_available() {
        LibTorchDevice::Cuda(0)
    } else {
        LibTorchDevice::Cpu
    }
}

/// Get the default device for the enabled backend
#[cfg(all(feature = "wgpu", not(any(feature = "ndarray", feature = "tch"))))]
pub fn default_device() -> WgpuDevice {
    WgpuDevice::default()
}

/// Get the default device for the enabled backend
#[cfg(all(
    feature = "cuda",
    not(any(feature = "ndarray", feature = "tch", feature = "wgpu"))
))]
pub fn default_device() -> CudaDevice {
    CudaDevice::default()
}

#[cfg(test)]
mod tests {
    #[cfg(any(
        feature = "ndarray",
        feature = "tch",
        feature = "wgpu",
        feature = "cuda"
    ))]
    use super::*;

    #[test]
    #[cfg(any(
        feature = "ndarray",
        feature = "tch",
        feature = "wgpu",
        feature = "cuda"
    ))]
    fn test_default_device() {
        let _device = default_device();
    }
}
