use burn::prelude::*;

/// Layer normalization
#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(size: usize, device: &B::Device) -> Self {
        Self {
            weight: Tensor::ones([size], device),
            bias: Tensor::zeros([size], device),
            eps: 1e-5,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let last_dim = D - 1;
        let mean = x.clone().mean_dim(last_dim);
        let var = x.clone().var(last_dim);

        let x_norm = (x - mean) / (var + self.eps).sqrt();

        // Apply affine transformation
        // Weight and bias broadcasting handled by Burn
        x_norm * self.weight.clone().unsqueeze() + self.bias.clone().unsqueeze()
    }
}
