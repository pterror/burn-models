//! CubeCL-accelerated UNet blocks
//!
//! This module provides optimized versions of UNet building blocks
//! that use fused CubeCL kernels for better GPU performance.
//!
//! # Usage
//!
//! Enable the `cubecl` feature and use `ResBlockCubeCL` instead of `ResBlock`:
//!
//! ```ignore
//! use burn_models_unet::cubecl::ResBlockCubeCL;
//! use cubecl::wgpu::WgpuRuntime;
//!
//! let block = ResBlockCubeCL::<WgpuRuntime>::new(256, 512, 1024, &device);
//! let output = block.forward(input, time_emb);
//! ```

use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    Linear, LinearConfig, PaddingConfig2d,
};
use burn::prelude::*;
use burn_cubecl::{CubeBackend, CubeRuntime, tensor::CubeTensor};
use burn_models_cubecl::{
    GroupNormSiLuOptions, groupnorm_silu, tensor_to_cube, cube_to_tensor,
    flash_attention, FlashAttentionOptions,
};

/// CubeCL-accelerated ResNet block with fused GroupNorm+SiLU
///
/// Uses fused GPU kernels for the `norm → silu` pattern, reducing
/// memory traffic and kernel launch overhead.
#[derive(Debug)]
pub struct ResBlockCubeCL<R: CubeRuntime> {
    norm1_weight: CubeTensor<R>,
    norm1_bias: CubeTensor<R>,
    conv1: Conv2d<CubeBackend<R, f32, i32, u32>>,
    time_emb_proj: Linear<CubeBackend<R, f32, i32, u32>>,
    norm2_weight: CubeTensor<R>,
    norm2_bias: CubeTensor<R>,
    conv2: Conv2d<CubeBackend<R, f32, i32, u32>>,
    skip_conv: Option<Conv2d<CubeBackend<R, f32, i32, u32>>>,
    num_groups: usize,
}

type B<R> = CubeBackend<R, f32, i32, u32>;

impl<R: CubeRuntime> ResBlockCubeCL<R> {
    /// Creates a new CubeCL-accelerated residual block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        device: &R::Device,
    ) -> Self {
        let num_groups = 32;

        // Initialize norm weights (ones) and biases (zeros)
        let norm1_weight = Tensor::<B<R>, 1>::ones([in_channels], device);
        let norm1_bias = Tensor::<B<R>, 1>::zeros([in_channels], device);

        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let time_emb_proj = LinearConfig::new(time_emb_dim, out_channels).init(device);

        let norm2_weight = Tensor::<B<R>, 1>::ones([out_channels], device);
        let norm2_bias = Tensor::<B<R>, 1>::zeros([out_channels], device);

        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let skip_conv = if in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device))
        } else {
            None
        };

        Self {
            norm1_weight: tensor_to_cube(norm1_weight),
            norm1_bias: tensor_to_cube(norm1_bias),
            conv1,
            time_emb_proj,
            norm2_weight: tensor_to_cube(norm2_weight),
            norm2_bias: tensor_to_cube(norm2_bias),
            conv2,
            skip_conv,
            num_groups,
        }
    }

    /// Forward pass with fused GroupNorm+SiLU kernels
    pub fn forward(
        &self,
        x: Tensor<B<R>, 4>,
        time_emb: Tensor<B<R>, 2>,
    ) -> Tensor<B<R>, 4> {
        let [b, _, _h, _w] = x.dims();

        let residual = match &self.skip_conv {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        // First conv block with FUSED GroupNorm+SiLU
        let hidden = groupnorm_silu(
            tensor_to_cube(x),
            self.norm1_weight.clone(),
            self.norm1_bias.clone(),
            GroupNormSiLuOptions::with_groups(self.num_groups),
        );
        let hidden: Tensor<B<R>, 4> = cube_to_tensor(hidden);
        let hidden = self.conv1.forward(hidden);

        // Add time embedding (silu applied to time_emb)
        let time_emb = burn_models_core::silu::silu(time_emb);
        let time_emb = self.time_emb_proj.forward(time_emb);
        let emb_dim = time_emb.dims()[1];
        let time_emb = time_emb.reshape([b, emb_dim, 1, 1]);
        let hidden = hidden + time_emb;

        // Second conv block with FUSED GroupNorm+SiLU
        let hidden = groupnorm_silu(
            tensor_to_cube(hidden),
            self.norm2_weight.clone(),
            self.norm2_bias.clone(),
            GroupNormSiLuOptions::with_groups(self.num_groups),
        );
        let hidden: Tensor<B<R>, 4> = cube_to_tensor(hidden);
        let hidden = self.conv2.forward(hidden);

        hidden + residual
    }
}

/// Convert a standard ResBlock to CubeCL-accelerated version
///
/// This extracts the weights from an existing ResBlock and creates
/// a CubeCL-accelerated version that uses fused kernels.
pub fn convert_resblock<R: CubeRuntime>(
    block: &crate::blocks::ResBlock<B<R>>,
) -> ResBlockCubeCL<R> {
    ResBlockCubeCL {
        norm1_weight: tensor_to_cube(block.norm1.weight.clone()),
        norm1_bias: tensor_to_cube(block.norm1.bias.clone()),
        conv1: block.conv1.clone(),
        time_emb_proj: block.time_emb_proj.clone(),
        norm2_weight: tensor_to_cube(block.norm2.weight.clone()),
        norm2_bias: tensor_to_cube(block.norm2.bias.clone()),
        conv2: block.conv2.clone(),
        skip_conv: block.skip_conv.clone(),
        num_groups: block.norm1.num_groups,
    }
}

/// CubeCL-accelerated cross-attention with Flash Attention
///
/// Uses Flash Attention kernel for O(n) memory attention computation.
/// This is a drop-in replacement for `CrossAttention` that uses GPU-optimized
/// tiled attention instead of materializing the full attention matrix.
#[derive(Debug)]
pub struct CrossAttentionCubeCL<R: CubeRuntime> {
    to_q: Linear<B<R>>,
    to_k: Linear<B<R>>,
    to_v: Linear<B<R>>,
    to_out: Linear<B<R>>,
    num_heads: usize,
    head_dim: usize,
}

impl<R: CubeRuntime> CrossAttentionCubeCL<R> {
    /// Creates a new CubeCL-accelerated cross-attention module
    ///
    /// # Arguments
    ///
    /// * `query_dim` - Dimension of query input
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    /// * `context_dim` - Dimension of key/value context (None for self-attention)
    /// * `device` - Device to create tensors on
    pub fn new(
        query_dim: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: Option<usize>,
        device: &R::Device,
    ) -> Self {
        let inner_dim = num_heads * head_dim;
        let context_dim = context_dim.unwrap_or(query_dim);

        Self {
            to_q: LinearConfig::new(query_dim, inner_dim).with_bias(false).init(device),
            to_k: LinearConfig::new(context_dim, inner_dim).with_bias(false).init(device),
            to_v: LinearConfig::new(context_dim, inner_dim).with_bias(false).init(device),
            to_out: LinearConfig::new(inner_dim, query_dim).init(device),
            num_heads,
            head_dim,
        }
    }

    /// Computes attention using Flash Attention kernel
    ///
    /// # Arguments
    ///
    /// * `x` - Query input of shape `[batch, seq_len, query_dim]`
    /// * `context` - Key/value context (None uses x for self-attention)
    ///
    /// # Returns
    ///
    /// Attention output of shape `[batch, seq_len, query_dim]`
    pub fn forward(&self, x: Tensor<B<R>, 3>, context: Option<Tensor<B<R>, 3>>) -> Tensor<B<R>, 3> {
        let context = context.unwrap_or_else(|| x.clone());

        let [b, seq_len, _] = x.dims();
        let [_, ctx_len, _] = context.dims();

        let q = self.to_q.forward(x);
        let k = self.to_k.forward(context.clone());
        let v = self.to_v.forward(context);

        // Reshape to multi-head: [b, seq, heads*dim] -> [b, heads, seq, dim]
        let q = q.reshape([b, seq_len, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let k = k.reshape([b, ctx_len, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let v = v.reshape([b, ctx_len, self.num_heads, self.head_dim]).swap_dims(1, 2);

        // Flash attention (non-causal for diffusion models)
        let out = flash_attention(
            tensor_to_cube(q),
            tensor_to_cube(k),
            tensor_to_cube(v),
            FlashAttentionOptions::default(),  // non-causal
        ).expect("Flash attention failed");

        let out: Tensor<B<R>, 4> = cube_to_tensor(out);

        // Reshape back: [b, heads, seq, dim] -> [b, seq, heads*dim]
        let out = out.swap_dims(1, 2).reshape([b, seq_len, self.num_heads * self.head_dim]);

        self.to_out.forward(out)
    }
}

/// Convert a standard CrossAttention to CubeCL-accelerated version
pub fn convert_crossattention<R: CubeRuntime>(
    attn: &crate::blocks::CrossAttention<B<R>>,
) -> CrossAttentionCubeCL<R> {
    CrossAttentionCubeCL {
        to_q: attn.to_q.clone(),
        to_k: attn.to_k.clone(),
        to_v: attn.to_v.clone(),
        to_out: attn.to_out.clone(),
        num_heads: attn.num_heads,
        head_dim: attn.head_dim,
    }
}
