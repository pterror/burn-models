//! SD 1.x UNet
//!
//! The diffusion backbone for Stable Diffusion 1.x models.

use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    Linear, LinearConfig, PaddingConfig2d,
};
use burn::prelude::*;

use burn_image_core::groupnorm::GroupNorm;
use burn_image_core::silu::silu;

use crate::blocks::{
    timestep_embedding, Downsample, ResBlock, SpatialTransformer, Upsample,
};

/// UNet configuration for SD 1.x
#[derive(Debug, Clone)]
pub struct UNetConfig {
    /// Input channels (latent channels, typically 4)
    pub in_channels: usize,
    /// Output channels (same as input)
    pub out_channels: usize,
    /// Base model channels
    pub model_channels: usize,
    /// Channel multipliers per resolution level
    pub channel_mult: Vec<usize>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Attention head dimension
    pub head_dim: usize,
    /// Context dimension (text embedding dim)
    pub context_dim: usize,
    /// Number of transformer blocks per spatial transformer
    pub transformer_depth: usize,
}

impl UNetConfig {
    /// Configuration for SD 1.x
    pub fn sd1x() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            channel_mult: vec![1, 2, 4, 4],
            num_heads: 8,
            head_dim: 40,
            context_dim: 768, // CLIP embedding dim
            transformer_depth: 1,
        }
    }

    /// Configuration for SD 2.x
    pub fn sd2x() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            channel_mult: vec![1, 2, 4, 4],
            num_heads: 8,
            head_dim: 80,
            context_dim: 1024, // OpenCLIP embedding dim
            transformer_depth: 1,
        }
    }
}

/// SD 1.x UNet
#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    // Time embedding
    time_embed_0: Linear<B>,
    time_embed_2: Linear<B>,

    // Input
    conv_in: Conv2d<B>,

    // Down blocks
    down_blocks: Vec<DownBlock<B>>,

    // Mid block
    mid_block: MidBlock<B>,

    // Up blocks
    up_blocks: Vec<UpBlock<B>>,

    // Output
    norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,

    // Config
    model_channels: usize,
}

impl<B: Backend> UNet<B> {
    /// Creates a new SD 1.x UNet
    ///
    /// # Arguments
    ///
    /// * `config` - UNet configuration
    /// * `device` - Device to create tensors on
    pub fn new(config: &UNetConfig, device: &B::Device) -> Self {
        let ch = config.model_channels;
        let time_embed_dim = ch * 4;

        // Time embedding MLP
        let time_embed_0 = LinearConfig::new(ch, time_embed_dim).init(device);
        let time_embed_2 = LinearConfig::new(time_embed_dim, time_embed_dim).init(device);

        // Input conv
        let conv_in = Conv2dConfig::new([config.in_channels, ch], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Build down blocks
        let mut down_blocks = Vec::new();
        let mut channels = vec![ch]; // Track channel sizes for skip connections
        let mut ch_in = ch;

        for (level, &mult) in config.channel_mult.iter().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;

            down_blocks.push(DownBlock::new(
                ch_in,
                ch_out,
                time_embed_dim,
                config.num_heads,
                config.head_dim,
                config.context_dim,
                config.transformer_depth,
                !is_last, // downsample except last
                device,
            ));

            channels.push(ch_out);
            channels.push(ch_out);
            if !is_last {
                channels.push(ch_out);
            }
            ch_in = ch_out;
        }

        // Mid block
        let mid_block = MidBlock::new(
            ch_in,
            time_embed_dim,
            config.num_heads,
            config.head_dim,
            config.context_dim,
            config.transformer_depth,
            device,
        );

        // Build up blocks (reverse order)
        let mut up_blocks = Vec::new();

        for (level, &mult) in config.channel_mult.iter().rev().enumerate() {
            let ch_out = ch * mult;
            let is_last = level == config.channel_mult.len() - 1;

            // Each up block has 3 res blocks that take skip connections
            for i in 0..3 {
                let skip_ch = channels.pop().unwrap();
                let block_in = ch_in + skip_ch;
                let block_out = if i == 2 && !is_last {
                    ch * config.channel_mult[config.channel_mult.len() - 2 - level]
                } else {
                    ch_out
                };

                let upsample = i == 2 && !is_last;

                up_blocks.push(UpBlock::new(
                    block_in,
                    block_out,
                    time_embed_dim,
                    config.num_heads,
                    config.head_dim,
                    config.context_dim,
                    config.transformer_depth,
                    upsample,
                    device,
                ));

                ch_in = block_out;
            }
        }

        // Output
        let norm_out = GroupNorm::new(32, ch, device);
        let conv_out = Conv2dConfig::new([ch, config.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self {
            time_embed_0,
            time_embed_2,
            conv_in,
            down_blocks,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
            model_channels: ch,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Noisy latent [batch, 4, h, w]
    /// * `timesteps` - Timestep for each sample [batch]
    /// * `context` - Text embeddings [batch, seq_len, context_dim]
    ///
    /// # Returns
    /// Predicted noise [batch, 4, h, w]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        // Time embedding
        let t_emb = timestep_embedding(timesteps, self.model_channels, &x.device());
        let t_emb = self.time_embed_0.forward(t_emb);
        let t_emb = silu(t_emb);
        let t_emb = self.time_embed_2.forward(t_emb);

        // Input
        let mut h = self.conv_in.forward(x);

        // Down blocks with skip connections
        let mut skips = vec![h.clone()];
        for block in &self.down_blocks {
            let (out, block_skips) = block.forward(h, t_emb.clone(), context.clone());
            h = out;
            skips.extend(block_skips);
        }

        // Mid block
        h = self.mid_block.forward(h, t_emb.clone(), context.clone());

        // Up blocks with skip connections
        for block in &self.up_blocks {
            let skip = skips.pop().unwrap();
            h = Tensor::cat(vec![h, skip], 1);
            h = block.forward(h, t_emb.clone(), context.clone());
        }

        // Output
        h = self.norm_out.forward(h);
        h = silu(h);
        self.conv_out.forward(h)
    }
}

/// Down block: ResBlock + SpatialTransformer + optional Downsample
#[derive(Module, Debug)]
struct DownBlock<B: Backend> {
    res1: ResBlock<B>,
    attn1: SpatialTransformer<B>,
    res2: ResBlock<B>,
    attn2: SpatialTransformer<B>,
    downsample: Option<Downsample<B>>,
}

impl<B: Backend> DownBlock<B> {
    /// Creates a new down block with residual and attention layers
    fn new(
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        transformer_depth: usize,
        downsample: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            res1: ResBlock::new(in_ch, out_ch, time_dim, device),
            attn1: SpatialTransformer::new(out_ch, num_heads, head_dim, context_dim, transformer_depth, device),
            res2: ResBlock::new(out_ch, out_ch, time_dim, device),
            attn2: SpatialTransformer::new(out_ch, num_heads, head_dim, context_dim, transformer_depth, device),
            downsample: if downsample { Some(Downsample::new(out_ch, device)) } else { None },
        }
    }

    /// Forward pass, returns output and skip connections for up blocks
    fn forward(
        &self,
        x: Tensor<B, 4>,
        t_emb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut skips = Vec::new();

        let h = self.res1.forward(x, t_emb.clone());
        let h = self.attn1.forward(h, context.clone());
        skips.push(h.clone());

        let h = self.res2.forward(h, t_emb);
        let h = self.attn2.forward(h, context);
        skips.push(h.clone());

        let h = if let Some(ds) = &self.downsample {
            let h = ds.forward(h);
            skips.push(h.clone());
            h
        } else {
            h
        };

        (h, skips)
    }
}

/// Mid block: ResBlock + Attention + ResBlock
#[derive(Module, Debug)]
struct MidBlock<B: Backend> {
    res1: ResBlock<B>,
    attn: SpatialTransformer<B>,
    res2: ResBlock<B>,
}

impl<B: Backend> MidBlock<B> {
    /// Creates a new mid block with ResBlock-Attention-ResBlock structure
    fn new(
        channels: usize,
        time_dim: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        transformer_depth: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            res1: ResBlock::new(channels, channels, time_dim, device),
            attn: SpatialTransformer::new(channels, num_heads, head_dim, context_dim, transformer_depth, device),
            res2: ResBlock::new(channels, channels, time_dim, device),
        }
    }

    /// Forward pass through the mid block
    fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let h = self.res1.forward(x, t_emb.clone());
        let h = self.attn.forward(h, context);
        self.res2.forward(h, t_emb)
    }
}

/// Up block with residual, attention, and optional upsampling
#[derive(Module, Debug)]
struct UpBlock<B: Backend> {
    res: ResBlock<B>,
    attn: SpatialTransformer<B>,
    upsample: Option<Upsample<B>>,
}

impl<B: Backend> UpBlock<B> {
    /// Creates a new up block
    fn new(
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        transformer_depth: usize,
        upsample: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            res: ResBlock::new(in_ch, out_ch, time_dim, device),
            attn: SpatialTransformer::new(out_ch, num_heads, head_dim, context_dim, transformer_depth, device),
            upsample: if upsample { Some(Upsample::new(out_ch, device)) } else { None },
        }
    }

    /// Forward pass through the up block
    fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let h = self.res.forward(x, t_emb);
        let h = self.attn.forward(h, context);

        if let Some(up) = &self.upsample {
            up.forward(h)
        } else {
            h
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unet_config() {
        let config = UNetConfig::sd1x();
        assert_eq!(config.in_channels, 4);
        assert_eq!(config.context_dim, 768);
    }
}
