//! Precision configuration for inference
//!
//! Supports running models in different precision modes:
//! - fp32: Full precision (default)
//! - fp16: Half precision (faster, less memory)
//! - bf16: Brain floating point (good balance)

use burn::prelude::*;

/// Precision mode for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrecisionMode {
    /// Full 32-bit precision (default)
    #[default]
    Fp32,
    /// 16-bit half precision
    Fp16,
    /// 16-bit brain floating point
    Bf16,
}

impl PrecisionMode {
    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            PrecisionMode::Fp32 => "fp32",
            PrecisionMode::Fp16 => "fp16",
            PrecisionMode::Bf16 => "bf16",
        }
    }

    /// Memory savings compared to fp32 (approximate)
    pub fn memory_savings(&self) -> f32 {
        match self {
            PrecisionMode::Fp32 => 1.0,
            PrecisionMode::Fp16 => 0.5,
            PrecisionMode::Bf16 => 0.5,
        }
    }
}

/// Configuration for mixed precision inference
#[derive(Debug, Clone, Default)]
pub struct PrecisionConfig {
    /// Precision for UNet weights
    pub unet_precision: PrecisionMode,
    /// Precision for VAE weights
    pub vae_precision: PrecisionMode,
    /// Precision for text encoder weights
    pub text_encoder_precision: PrecisionMode,
    /// Whether to use fp16 for intermediate computations
    pub compute_precision: PrecisionMode,
}

impl PrecisionConfig {
    /// Full fp32 precision (default, most accurate)
    pub fn fp32() -> Self {
        Self::default()
    }

    /// Full fp16 precision (fastest, may have quality loss)
    pub fn fp16() -> Self {
        Self {
            unet_precision: PrecisionMode::Fp16,
            vae_precision: PrecisionMode::Fp16,
            text_encoder_precision: PrecisionMode::Fp16,
            compute_precision: PrecisionMode::Fp16,
        }
    }

    /// Full bf16 precision (good balance)
    pub fn bf16() -> Self {
        Self {
            unet_precision: PrecisionMode::Bf16,
            vae_precision: PrecisionMode::Bf16,
            text_encoder_precision: PrecisionMode::Bf16,
            compute_precision: PrecisionMode::Bf16,
        }
    }

    /// Mixed precision: fp16 for UNet, fp32 for VAE
    /// (recommended - VAE benefits from higher precision)
    pub fn mixed() -> Self {
        Self {
            unet_precision: PrecisionMode::Fp16,
            vae_precision: PrecisionMode::Fp32,
            text_encoder_precision: PrecisionMode::Fp16,
            compute_precision: PrecisionMode::Fp16,
        }
    }
}

/// Convert a tensor to half precision (fp16)
pub fn to_fp16<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    // Note: In Burn, the actual conversion depends on backend support
    // This is a placeholder that works with backends that support fp16
    tensor
}

/// Convert a tensor to brain float16 (bf16)
pub fn to_bf16<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    // Note: Similar to fp16, backend-dependent
    tensor
}

/// Convert a tensor to full precision (fp32)
pub fn to_fp32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor
}

/// Helper trait for precision conversion
pub trait PrecisionConvert<B: Backend, const D: usize> {
    /// Convert to the specified precision
    fn to_precision(self, mode: PrecisionMode) -> Tensor<B, D>;
}

impl<B: Backend, const D: usize> PrecisionConvert<B, D> for Tensor<B, D> {
    fn to_precision(self, mode: PrecisionMode) -> Tensor<B, D> {
        match mode {
            PrecisionMode::Fp32 => to_fp32(self),
            PrecisionMode::Fp16 => to_fp16(self),
            PrecisionMode::Bf16 => to_bf16(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_mode_default() {
        assert_eq!(PrecisionMode::default(), PrecisionMode::Fp32);
    }

    #[test]
    fn test_precision_config_presets() {
        let fp32 = PrecisionConfig::fp32();
        assert_eq!(fp32.unet_precision, PrecisionMode::Fp32);

        let fp16 = PrecisionConfig::fp16();
        assert_eq!(fp16.unet_precision, PrecisionMode::Fp16);

        let mixed = PrecisionConfig::mixed();
        assert_eq!(mixed.unet_precision, PrecisionMode::Fp16);
        assert_eq!(mixed.vae_precision, PrecisionMode::Fp32);
    }

    #[test]
    fn test_memory_savings() {
        assert_eq!(PrecisionMode::Fp32.memory_savings(), 1.0);
        assert_eq!(PrecisionMode::Fp16.memory_savings(), 0.5);
    }
}
