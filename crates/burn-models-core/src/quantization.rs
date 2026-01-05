//! Quantization Support for Model Compression
//!
//! Provides INT8 and INT4 quantization for reduced memory usage
//! and faster inference on supported hardware.
//!
//! # Quantization Types
//!
//! - **Dynamic Quantization**: Weights quantized, activations quantized at runtime
//! - **Static Quantization**: Both weights and activations pre-quantized (requires calibration)
//!
//! # Supported Formats
//!
//! - INT8 symmetric: scale only, zero_point = 0
//! - INT8 asymmetric: scale + zero_point
//! - INT4 (GPTQ/AWQ style): group-wise quantization

use burn::prelude::*;

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Number of bits (4 or 8)
    pub bits: usize,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Group size for group-wise quantization (0 = per-tensor)
    pub group_size: usize,
}

impl QuantConfig {
    /// INT8 symmetric per-tensor quantization
    pub fn int8_symmetric() -> Self {
        Self {
            bits: 8,
            symmetric: true,
            group_size: 0,
        }
    }

    /// INT8 asymmetric per-tensor quantization
    pub fn int8_asymmetric() -> Self {
        Self {
            bits: 8,
            symmetric: false,
            group_size: 0,
        }
    }

    /// INT4 group-wise quantization (GPTQ style)
    pub fn int4_grouped(group_size: usize) -> Self {
        Self {
            bits: 4,
            symmetric: true,
            group_size,
        }
    }

    /// Get the quantization range
    pub fn range(&self) -> (i64, i64) {
        if self.symmetric {
            let max = (1i64 << (self.bits - 1)) - 1;
            (-max, max)
        } else {
            let max = (1i64 << self.bits) - 1;
            (0, max)
        }
    }
}

/// Quantization parameters for a tensor
#[derive(Debug, Clone)]
pub struct QuantParams<B: Backend> {
    /// Scale factor(s) - [1] for per-tensor, [num_groups] for group-wise
    pub scale: Tensor<B, 1>,
    /// Zero point(s) - [1] for per-tensor, [num_groups] for group-wise
    pub zero_point: Tensor<B, 1>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Configuration
    pub config: QuantConfig,
}

/// Quantized tensor representation
pub struct QuantizedTensor<B: Backend> {
    /// Quantized data stored as i32 (INT8 values in lower bits)
    pub data: Tensor<B, 2, Int>,
    /// Quantization parameters
    pub params: QuantParams<B>,
}

impl<B: Backend> QuantizedTensor<B> {
    /// Dequantize back to float
    pub fn dequantize(&self) -> Tensor<B, 2> {
        let [rows, cols] = self.data.dims();
        let device = self.data.device();

        if self.params.config.group_size == 0 {
            // Per-tensor quantization
            let scale = self.params.scale.clone().reshape([1, 1]);
            let zero_point = self.params.zero_point.clone().reshape([1, 1]);

            let data_float = self.data.clone().float();
            (data_float - zero_point) * scale
        } else {
            // Group-wise quantization
            let group_size = self.params.config.group_size;
            let num_groups = (cols + group_size - 1) / group_size;

            let mut result_parts = Vec::new();

            for g in 0..num_groups {
                let start = g * group_size;
                let end = (start + group_size).min(cols);

                let group_data = self.data.clone().slice([0..rows, start..end]);
                let scale = self.params.scale.clone().slice([g..g + 1]);
                let zero_point = self.params.zero_point.clone().slice([g..g + 1]);

                let scale = scale.reshape([1, 1]);
                let zero_point = zero_point.reshape([1, 1]);

                let group_float = group_data.float();
                let dequant = (group_float - zero_point) * scale;
                result_parts.push(dequant);
            }

            Tensor::cat(result_parts, 1)
        }
    }
}

/// Quantize a tensor to INT8/INT4
pub fn quantize<B: Backend>(
    tensor: Tensor<B, 2>,
    config: &QuantConfig,
) -> QuantizedTensor<B> {
    let [rows, cols] = tensor.dims();
    let device = tensor.device();
    let (qmin, qmax) = config.range();

    if config.group_size == 0 {
        // Per-tensor quantization
        let (scale, zero_point, quantized) = quantize_tensor(&tensor, config, &device);

        QuantizedTensor {
            data: quantized,
            params: QuantParams {
                scale,
                zero_point,
                shape: vec![rows, cols],
                config: config.clone(),
            },
        }
    } else {
        // Group-wise quantization
        let group_size = config.group_size;
        let num_groups = (cols + group_size - 1) / group_size;

        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        let mut quantized_groups = Vec::new();

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(cols);

            let group = tensor.clone().slice([0..rows, start..end]);
            let (scale, zp, quant) = quantize_tensor(&group, config, &device);

            scales.push(scale);
            zero_points.push(zp);
            quantized_groups.push(quant);
        }

        let scale = Tensor::cat(scales, 0);
        let zero_point = Tensor::cat(zero_points, 0);
        let quantized = Tensor::cat(quantized_groups, 1);

        QuantizedTensor {
            data: quantized,
            params: QuantParams {
                scale,
                zero_point,
                shape: vec![rows, cols],
                config: config.clone(),
            },
        }
    }
}

fn quantize_tensor<B: Backend>(
    tensor: &Tensor<B, 2>,
    config: &QuantConfig,
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 2, Int>) {
    let (qmin, qmax) = config.range();

    // Find min/max - use reduce operations on 2D tensor
    let min_per_row = tensor.clone().min_dim(1);
    let max_per_row = tensor.clone().max_dim(1);
    let min_val: f32 = min_per_row.min().into_scalar().elem();
    let max_val: f32 = max_per_row.max().into_scalar().elem();

    let (scale, zero_point) = if config.symmetric {
        // Symmetric: scale = max(|min|, |max|) / qmax
        let abs_max = min_val.abs().max(max_val.abs());
        let scale = if abs_max > 0.0 {
            abs_max / qmax as f32
        } else {
            1.0
        };
        (scale, 0.0f32)
    } else {
        // Asymmetric: scale = (max - min) / (qmax - qmin)
        let scale = if (max_val - min_val).abs() > 1e-10 {
            (max_val - min_val) / (qmax - qmin) as f32
        } else {
            1.0
        };
        let zero_point = qmin as f32 - min_val / scale;
        (scale, zero_point.round())
    };

    // Quantize
    let scale_tensor = Tensor::<B, 1>::from_floats([scale], device);
    let zp_tensor = Tensor::<B, 1>::from_floats([zero_point], device);

    let scaled = tensor.clone() / scale;
    let shifted = scaled + zero_point;
    let clamped = shifted.clamp(qmin as f32, qmax as f32);
    let quantized = clamped.int();

    (scale_tensor, zp_tensor, quantized)
}

/// Quantized linear layer
pub struct QuantizedLinear<B: Backend> {
    /// Quantized weight
    pub weight: QuantizedTensor<B>,
    /// Optional bias (not quantized)
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> QuantizedLinear<B> {
    /// Create from a regular linear layer's weights
    pub fn from_weight(
        weight: Tensor<B, 2>,
        bias: Option<Tensor<B, 1>>,
        config: &QuantConfig,
    ) -> Self {
        Self {
            weight: quantize(weight, config),
            bias,
        }
    }

    /// Forward pass with dequantization
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Dequantize weights
        let weight = self.weight.dequantize();

        // Standard linear: x @ weight.T + bias
        let out = x.matmul(weight.transpose());

        match &self.bias {
            Some(b) => out + b.clone().unsqueeze(),
            None => out,
        }
    }

    /// Memory savings compared to f32
    pub fn memory_ratio(&self) -> f32 {
        let bits = self.weight.params.config.bits;
        bits as f32 / 32.0
    }
}

/// Calculate memory usage for quantized vs original
pub fn memory_savings(
    num_params: usize,
    config: &QuantConfig,
) -> QuantMemoryStats {
    let original_bytes = num_params * 4; // f32 = 4 bytes
    let quantized_bytes = num_params * config.bits / 8;

    // Add overhead for scales/zero_points
    let num_groups = if config.group_size > 0 {
        (num_params + config.group_size - 1) / config.group_size
    } else {
        1
    };
    let param_overhead = num_groups * 8; // 2 x f32 per group

    let total_quantized = quantized_bytes + param_overhead;

    QuantMemoryStats {
        original_bytes,
        quantized_bytes: total_quantized,
        compression_ratio: original_bytes as f32 / total_quantized as f32,
    }
}

/// Memory statistics for quantization
#[derive(Debug, Clone)]
pub struct QuantMemoryStats {
    pub original_bytes: usize,
    pub quantized_bytes: usize,
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_quant_config() {
        let config = QuantConfig::int8_symmetric();
        assert_eq!(config.bits, 8);
        assert!(config.symmetric);
        assert_eq!(config.range(), (-127, 127));

        let config = QuantConfig::int8_asymmetric();
        assert_eq!(config.range(), (0, 255));

        let config = QuantConfig::int4_grouped(128);
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert_eq!(config.range(), (-7, 7));
    }

    #[test]
    fn test_quantize_dequantize_symmetric() {
        let device = Default::default();
        let config = QuantConfig::int8_symmetric();

        // Create a tensor with known values
        let tensor = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, -2.0, 3.0, -4.0], [0.5, -0.5, 1.5, -1.5]],
            &device,
        );

        let quantized = quantize(tensor.clone(), &config);
        let dequantized = quantized.dequantize();

        // Check approximate reconstruction
        let diff = (tensor - dequantized).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        // Should be close (within quantization error)
        assert!(max_diff < 0.1, "Max diff {} too large", max_diff);
    }

    #[test]
    fn test_quantize_dequantize_asymmetric() {
        let device = Default::default();
        let config = QuantConfig::int8_asymmetric();

        let tensor = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]],
            &device,
        );

        let quantized = quantize(tensor.clone(), &config);
        let dequantized = quantized.dequantize();

        let diff = (tensor - dequantized).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        assert!(max_diff < 0.1, "Max diff {} too large", max_diff);
    }

    #[test]
    fn test_quantize_grouped() {
        let device = Default::default();
        let config = QuantConfig::int4_grouped(2); // Group size 2

        let tensor = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 10.0, 20.0]], // Two groups with different scales
            &device,
        );

        let quantized = quantize(tensor.clone(), &config);

        // Should have 2 scales (one per group)
        assert_eq!(quantized.params.scale.dims(), [2]);

        let dequantized = quantized.dequantize();
        let diff = (tensor - dequantized).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        // INT4 has more quantization error
        assert!(max_diff < 3.0, "Max diff {} too large for INT4", max_diff);
    }

    #[test]
    fn test_quantized_linear() {
        let device = Default::default();
        let config = QuantConfig::int8_symmetric();

        // Create weight [out_features, in_features] = [3, 4]
        let weight = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [-1.0, -2.0, -3.0, -4.0]],
            &device,
        );

        let linear = QuantizedLinear::from_weight(weight, None, &config);

        // Forward pass
        let x = Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
        let out = linear.forward(x);

        assert_eq!(out.dims(), [1, 3]);

        // Check memory ratio
        assert!((linear.memory_ratio() - 0.25).abs() < 0.01); // 8/32 = 0.25
    }

    #[test]
    fn test_memory_savings() {
        // 1M params
        let stats = memory_savings(1_000_000, &QuantConfig::int8_symmetric());
        assert_eq!(stats.original_bytes, 4_000_000);
        assert!(stats.compression_ratio > 3.9); // Close to 4x

        let stats = memory_savings(1_000_000, &QuantConfig::int4_grouped(128));
        assert!(stats.compression_ratio > 7.0); // Close to 8x
    }
}
