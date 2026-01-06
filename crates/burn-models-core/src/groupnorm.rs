//! Group normalization implementation
//!
//! Provides group normalization as used in UNet and VAE architectures.
//! Divides channels into groups and normalizes within each group.

use burn::prelude::*;

/// Group normalization module
///
/// Divides channels into groups and normalizes each group independently.
/// This is commonly used in diffusion models (UNet, VAE) as an alternative
/// to batch normalization that works well with small batch sizes.
///
/// # Formula
///
/// For input with C channels divided into G groups:
/// ```text
/// y = (x - mean(x_group)) / sqrt(var(x_group) + eps) * weight + bias
/// ```
///
/// # Reference
///
/// "Group Normalization" - Wu & He, 2018
#[derive(Module, Debug)]
pub struct GroupNorm<B: Backend> {
    /// Number of groups to divide channels into
    pub num_groups: usize,
    /// Scale parameter (gamma), shape [num_channels]
    pub weight: Tensor<B, 1>,
    /// Bias parameter (beta), shape [num_channels]
    pub bias: Tensor<B, 1>,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl<B: Backend> GroupNorm<B> {
    /// Creates a new group normalization module
    ///
    /// # Arguments
    ///
    /// * `num_groups` - Number of groups to divide channels into (typically 32)
    /// * `num_channels` - Total number of input channels (must be divisible by num_groups)
    /// * `device` - Device to create tensors on
    pub fn new(num_groups: usize, num_channels: usize, device: &B::Device) -> Self {
        Self {
            num_groups,
            weight: Tensor::ones([num_channels], device),
            bias: Tensor::zeros([num_channels], device),
            eps: 1e-5,
        }
    }

    /// Applies group normalization to a 4D tensor
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, channels, height, width]`
    ///
    /// # Returns
    ///
    /// Normalized tensor with same shape as input
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let group_size = channels / self.num_groups;

        // Reshape to [batch, num_groups, group_size * height * width]
        let x = x.reshape([batch, self.num_groups, group_size * height * width]);

        // Compute mean and variance over the last dimension
        let mean = x.clone().mean_dim(2); // [batch, num_groups]
        let var = x.clone().var(2); // [batch, num_groups]

        // Expand for broadcasting: [batch, num_groups, 1]
        let mean = mean.unsqueeze::<3>(); // [batch, num_groups, 1]
        let var = var.unsqueeze::<3>(); // [batch, num_groups, 1]

        // Normalize
        let x = (x - mean) / (var + self.eps).sqrt();

        // Reshape back to [batch, channels, height, width]
        let x = x.reshape([batch, channels, height, width]);

        // Apply weight and bias
        let weight = self.weight.clone().reshape([1, channels, 1, 1]);
        let bias = self.bias.clone().reshape([1, channels, 1, 1]);

        x * weight + bias
    }
}
