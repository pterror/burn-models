//! UNet building blocks: ResNet blocks, attention blocks, down/up sampling

use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    Linear, LinearConfig, PaddingConfig2d,
};
use burn::prelude::*;

use burn_image_core::groupnorm::GroupNorm;
use burn_image_core::silu::silu;

/// Timestep embedding using sinusoidal positional encoding
pub fn timestep_embedding<B: Backend>(
    timesteps: Tensor<B, 1>,
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let half_dim = dim / 2;
    let max_period = 10000.0f64;

    // Create frequency embedding
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-((i as f64) / half_dim as f64) * max_period.ln()).exp() as f32)
        .collect();

    let freqs = Tensor::<B, 1>::from_data(TensorData::new(freqs, [half_dim]), device);

    // timesteps: [batch], freqs: [half_dim]
    // args: [batch, half_dim]
    let [batch] = timesteps.dims();
    let args = timesteps.reshape([batch, 1]) * freqs.reshape([1, half_dim]);

    // Concatenate sin and cos
    let sin = args.clone().sin();
    let cos = args.cos();

    Tensor::cat(vec![sin, cos], 1) // [batch, dim]
}

/// ResNet block with time embedding
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    time_emb_proj: Linear<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    skip_conv: Option<Conv2d<B>>,
}

impl<B: Backend> ResBlock<B> {
    /// Creates a new residual block
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `time_emb_dim` - Dimension of the timestep embedding
    /// * `device` - Device to create tensors on
    pub fn new(in_channels: usize, out_channels: usize, time_emb_dim: usize, device: &B::Device) -> Self {
        let norm1 = GroupNorm::new(32, in_channels, device);
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let time_emb_proj = LinearConfig::new(time_emb_dim, out_channels).init(device);

        let norm2 = GroupNorm::new(32, out_channels, device);
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let skip_conv = if in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device))
        } else {
            None
        };

        Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            skip_conv,
        }
    }

    /// Forward pass through the residual block
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, in_channels, height, width]`
    /// * `time_emb` - Timestep embedding of shape `[batch, time_emb_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, out_channels, height, width]`
    pub fn forward(&self, x: Tensor<B, 4>, time_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        let [b, _, h, w] = x.dims();

        let residual = match &self.skip_conv {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        // First conv block
        let hidden = self.norm1.forward(x);
        let hidden = silu(hidden);
        let hidden = self.conv1.forward(hidden);

        // Add time embedding
        let time_emb = silu(time_emb);
        let time_emb = self.time_emb_proj.forward(time_emb);
        let emb_dim = time_emb.dims()[1];
        let time_emb = time_emb.reshape([b, emb_dim, 1, 1]);
        let hidden = hidden + time_emb;

        // Second conv block
        let hidden = self.norm2.forward(hidden);
        let hidden = silu(hidden);
        let hidden = self.conv2.forward(hidden);

        hidden + residual
    }
}

/// Spatial transformer block for cross-attention
#[derive(Module, Debug)]
pub struct SpatialTransformer<B: Backend> {
    norm: GroupNorm<B>,
    proj_in: Conv2d<B>,
    transformer_blocks: Vec<TransformerBlock<B>>,
    proj_out: Conv2d<B>,
}

impl<B: Backend> SpatialTransformer<B> {
    /// Creates a new spatial transformer block
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of input/output channels
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    /// * `context_dim` - Dimension of the cross-attention context (text embeddings)
    /// * `depth` - Number of transformer blocks to stack
    /// * `device` - Device to create tensors on
    pub fn new(
        channels: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        depth: usize,
        device: &B::Device,
    ) -> Self {
        let inner_dim = num_heads * head_dim;

        let norm = GroupNorm::new(32, channels, device);
        let proj_in = Conv2dConfig::new([channels, inner_dim], [1, 1]).init(device);

        let transformer_blocks = (0..depth)
            .map(|_| TransformerBlock::new(inner_dim, num_heads, head_dim, context_dim, device))
            .collect();

        let proj_out = Conv2dConfig::new([inner_dim, channels], [1, 1]).init(device);

        Self {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
        }
    }

    /// Forward pass with cross-attention to text context
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, channels, height, width]`
    /// * `context` - Text embedding context of shape `[batch, seq_len, context_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor with same shape as input
    pub fn forward(&self, x: Tensor<B, 4>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let residual = x.clone();

        let x = self.norm.forward(x);
        let x = self.proj_in.forward(x);

        // Reshape to sequence: [b, c, h, w] -> [b, h*w, c]
        let inner_dim = x.dims()[1];
        let x = x.reshape([b, inner_dim, h * w]).swap_dims(1, 2);

        // Apply transformer blocks
        let mut x = x;
        for block in &self.transformer_blocks {
            x = block.forward(x, context.clone());
        }

        // Reshape back: [b, h*w, c] -> [b, c, h, w]
        let x = x.swap_dims(1, 2).reshape([b, inner_dim, h, w]);
        let x = self.proj_out.forward(x);

        x + residual
    }
}

/// Transformer block with self-attention, cross-attention, and FFN
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    norm1: burn_image_core::layernorm::LayerNorm<B>,
    attn1: CrossAttention<B>, // Self-attention
    norm2: burn_image_core::layernorm::LayerNorm<B>,
    attn2: CrossAttention<B>, // Cross-attention
    norm3: burn_image_core::layernorm::LayerNorm<B>,
    ff: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    /// Creates a new transformer block
    ///
    /// # Arguments
    ///
    /// * `dim` - Hidden dimension
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    /// * `context_dim` - Dimension of cross-attention context
    /// * `device` - Device to create tensors on
    pub fn new(
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: burn_image_core::layernorm::LayerNorm::new(dim, device),
            attn1: CrossAttention::new(dim, num_heads, head_dim, None, device), // Self-attn
            norm2: burn_image_core::layernorm::LayerNorm::new(dim, device),
            attn2: CrossAttention::new(dim, num_heads, head_dim, Some(context_dim), device), // Cross-attn
            norm3: burn_image_core::layernorm::LayerNorm::new(dim, device),
            ff: FeedForward::new(dim, dim * 4, device),
        }
    }

    /// Forward pass through self-attention, cross-attention, and FFN
    ///
    /// # Arguments
    ///
    /// * `x` - Input sequence of shape `[batch, seq_len, dim]`
    /// * `context` - Cross-attention context of shape `[batch, ctx_len, context_dim]`
    ///
    /// # Returns
    ///
    /// Output sequence with same shape as input
    pub fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention
        let x = x.clone() + self.attn1.forward(self.norm1.forward(x.clone()), None);

        // Cross-attention
        let x = x.clone() + self.attn2.forward(self.norm2.forward(x.clone()), Some(context));

        // FFN
        x.clone() + self.ff.forward(self.norm3.forward(x))
    }
}

/// Cross-attention (or self-attention if context is None)
#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    to_q: Linear<B>,
    to_k: Linear<B>,
    to_v: Linear<B>,
    to_out: Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> CrossAttention<B> {
    /// Creates a new cross-attention (or self-attention) module
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
        device: &B::Device,
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

    /// Computes scaled dot-product attention
    ///
    /// # Arguments
    ///
    /// * `x` - Query input of shape `[batch, seq_len, query_dim]`
    /// * `context` - Key/value context (None uses x for self-attention)
    ///
    /// # Returns
    ///
    /// Attention output of shape `[batch, seq_len, query_dim]`
    pub fn forward(&self, x: Tensor<B, 3>, context: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
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

        // Attention
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = q.matmul(k.transpose()) * scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);

        // Reshape back: [b, heads, seq, dim] -> [b, seq, heads*dim]
        let out = out.swap_dims(1, 2).reshape([b, seq_len, self.num_heads * self.head_dim]);

        self.to_out.forward(out)
    }
}

/// Feed-forward network with GEGLU activation
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    net_0: Linear<B>,
    net_2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Creates a new feed-forward network with GEGLU activation
    ///
    /// # Arguments
    ///
    /// * `dim` - Input and output dimension
    /// * `mult_dim` - Hidden layer dimension
    /// * `device` - Device to create tensors on
    pub fn new(dim: usize, mult_dim: usize, device: &B::Device) -> Self {
        // GEGLU doubles the projection size
        Self {
            net_0: LinearConfig::new(dim, mult_dim * 2).init(device),
            net_2: LinearConfig::new(mult_dim, dim).init(device),
        }
    }

    /// Forward pass through the FFN with GEGLU gating
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, seq_len, dim]`
    ///
    /// # Returns
    ///
    /// Output tensor with same shape as input
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.net_0.forward(x);

        // GEGLU: split into two halves, one is gate
        let [b, s, d] = hidden.dims();
        let half = d / 2;
        let x = hidden.clone().slice([0..b, 0..s, 0..half]);
        let gate = hidden.slice([0..b, 0..s, half..d]);

        let hidden = x * burn::tensor::activation::gelu(gate);
        self.net_2.forward(hidden)
    }
}

/// Downsample block (strided conv)
#[derive(Module, Debug)]
pub struct Downsample<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> Downsample<B> {
    /// Creates a new downsample block (2x spatial reduction)
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of input/output channels
    /// * `device` - Device to create tensors on
    pub fn new(channels: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([channels, channels], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        Self { conv }
    }

    /// Downsamples input by 2x using strided convolution
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, channels, height, width]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, channels, height/2, width/2]`
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(x)
    }
}

/// Upsample block using nearest neighbor interpolation followed by convolution
#[derive(Module, Debug)]
pub struct Upsample<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> Upsample<B> {
    /// Creates a new upsample block (2x spatial increase)
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of input/output channels
    /// * `device` - Device to create tensors on
    pub fn new(channels: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        Self { conv }
    }

    /// Upsamples input by 2x using nearest neighbor + convolution
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, channels, height, width]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, channels, height*2, width*2]`
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();

        // Nearest neighbor 2x upsample
        let x = x.reshape([b, c, h, 1, w, 1]);
        let x = x.repeat_dim(3, 2).repeat_dim(5, 2);
        let x = x.reshape([b, c, h * 2, w * 2]);

        self.conv.forward(x)
    }
}
