use burn::prelude::*;

/// Group normalization layer
#[derive(Module, Debug)]
pub struct GroupNorm<B: Backend> {
    num_groups: usize,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> GroupNorm<B> {
    pub fn new(num_groups: usize, num_channels: usize, device: &B::Device) -> Self {
        Self {
            num_groups,
            weight: Tensor::ones([num_channels], device),
            bias: Tensor::zeros([num_channels], device),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let group_size = channels / self.num_groups;

        // Reshape to [batch, num_groups, group_size, height, width]
        let x = x.reshape([batch, self.num_groups, group_size, height, width]);

        // Compute mean and variance over group_size, height, width
        let mean = x.clone().mean_dim(2).mean_dim(2).mean_dim(2);
        let var = x.clone().var(2).mean_dim(2).mean_dim(2);

        // Normalize
        let x = (x - mean.unsqueeze_dim(2).unsqueeze_dim(2).unsqueeze_dim(2))
            / (var.unsqueeze_dim(2).unsqueeze_dim(2).unsqueeze_dim(2) + self.eps).sqrt();

        // Reshape back
        let x = x.reshape([batch, channels, height, width]);

        // Apply weight and bias
        let weight = self.weight.clone().reshape([1, channels, 1, 1]);
        let bias = self.bias.clone().reshape([1, channels, 1, 1]);

        x * weight + bias
    }
}
