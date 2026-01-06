//! VAE Decoder: latent -> image
//!
//! Decodes 4-channel latent representations to 3-channel RGB images.

use burn::nn::{
    PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;

use burn_models_core::groupnorm::GroupNorm;
use burn_models_core::silu::silu;

/// VAE Decoder configuration
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Input latent channels (typically 4)
    pub latent_channels: usize,
    /// Output image channels (typically 3 for RGB)
    pub out_channels: usize,
    /// Base channel multiplier
    pub base_channels: usize,
    /// Channel multipliers for each block
    pub channel_mult: Vec<usize>,
    /// Number of resnet blocks per decoder block
    pub num_res_blocks: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            latent_channels: 4,
            out_channels: 3,
            base_channels: 128,
            channel_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
        }
    }
}

impl DecoderConfig {
    /// SD 1.x VAE decoder config
    pub fn sd1x() -> Self {
        Self::default()
    }

    /// SD 1.x VAE decoder config (alias)
    pub fn sd() -> Self {
        Self::sd1x()
    }

    /// SDXL VAE decoder config (same architecture, different scaling)
    pub fn sdxl() -> Self {
        Self::default()
    }
}

/// VAE scaling factors for different model versions
pub mod scaling {
    /// SD 1.x / SD 2.x scaling factor
    pub const SD1X: f64 = 0.18215;
    /// SDXL scaling factor
    pub const SDXL: f64 = 0.13025;
}

/// VAE Decoder
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    pub conv_in: Conv2d<B>,
    pub mid_block1: ResnetBlock<B>,
    pub mid_attn: SelfAttention<B>,
    pub mid_block2: ResnetBlock<B>,
    pub up_blocks: Vec<DecoderBlock<B>>,
    pub norm_out: GroupNorm<B>,
    pub conv_out: Conv2d<B>,
}

impl<B: Backend> Decoder<B> {
    /// Creates a new VAE decoder
    ///
    /// # Arguments
    ///
    /// * `config` - Decoder configuration
    /// * `device` - Device to create tensors on
    pub fn new(config: &DecoderConfig, device: &B::Device) -> Self {
        let ch = config.base_channels;
        let ch_mult = &config.channel_mult;

        // Start with highest channel count (reversed from encoder)
        let block_in = ch * ch_mult[ch_mult.len() - 1];

        // Input conv: latent_channels -> block_in
        let conv_in = Conv2dConfig::new([config.latent_channels, block_in], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Mid blocks
        let mid_block1 = ResnetBlock::new(block_in, block_in, device);
        let mid_attn = SelfAttention::new(block_in, device);
        let mid_block2 = ResnetBlock::new(block_in, block_in, device);

        // Up blocks (reverse order, with upsampling)
        let mut up_blocks = Vec::new();
        let mut in_ch = block_in;

        for (i, &mult) in ch_mult.iter().rev().enumerate() {
            let out_ch = ch * mult;
            let upsample = i < ch_mult.len() - 1; // Don't upsample on last block

            up_blocks.push(DecoderBlock::new(
                in_ch,
                out_ch,
                config.num_res_blocks,
                upsample,
                device,
            ));
            in_ch = out_ch;
        }

        // Output layers
        let norm_out = GroupNorm::new(32, in_ch, device);
        let conv_out = Conv2dConfig::new([in_ch, config.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self {
            conv_in,
            mid_block1,
            mid_attn,
            mid_block2,
            up_blocks,
            norm_out,
            conv_out,
        }
    }

    /// Decode latent to image (raw, no scaling applied)
    ///
    /// Input: [batch, 4, h, w] latent (already unscaled)
    /// Output: [batch, 3, h*8, w*8] image (values in [-1, 1])
    pub fn forward_raw(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        // Input conv
        let mut h = self.conv_in.forward(z);

        // Mid blocks
        h = self.mid_block1.forward(h);
        h = self.mid_attn.forward(h);
        h = self.mid_block2.forward(h);

        // Up blocks
        for block in &self.up_blocks {
            h = block.forward(h);
        }

        // Output
        h = self.norm_out.forward(h);
        h = silu(h);
        self.conv_out.forward(h)
    }

    /// Decode latent to image with SD 1.x scaling
    ///
    /// Input: [batch, 4, h, w] latent
    /// Output: [batch, 3, h*8, w*8] image (values in [-1, 1])
    pub fn forward(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let z = z / scaling::SD1X;
        self.forward_raw(z)
    }

    /// Decode latent to image with custom scaling factor
    pub fn forward_scaled(&self, z: Tensor<B, 4>, scale: f64) -> Tensor<B, 4> {
        let z = z / scale;
        self.forward_raw(z)
    }

    /// Decode latent with SDXL scaling
    pub fn forward_sdxl(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_scaled(z, scaling::SDXL)
    }

    /// Decode and convert to [0, 255] range (SD 1.x scaling)
    pub fn decode_to_image(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let img = self.forward(z);
        // Convert from [-1, 1] to [0, 255]
        (img + 1.0) * 127.5
    }

    /// Decode and convert to [0, 255] range (SDXL scaling)
    pub fn decode_to_image_sdxl(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let img = self.forward_sdxl(z);
        (img + 1.0) * 127.5
    }
}

/// Decoder block with optional upsampling
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    pub res_blocks: Vec<ResnetBlock<B>>,
    pub upsample: Option<Upsample<B>>,
}

impl<B: Backend> DecoderBlock<B> {
    /// Creates a new decoder block with optional upsampling
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        upsample: bool,
        device: &B::Device,
    ) -> Self {
        let mut res_blocks = Vec::new();

        // First block handles channel change
        res_blocks.push(ResnetBlock::new(in_channels, out_channels, device));

        // Remaining blocks maintain channels
        for _ in 1..num_blocks {
            res_blocks.push(ResnetBlock::new(out_channels, out_channels, device));
        }

        let upsample = if upsample {
            Some(Upsample::new(out_channels, device))
        } else {
            None
        };

        Self {
            res_blocks,
            upsample,
        }
    }

    /// Forward pass through residual blocks and optional upsampling
    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for block in &self.res_blocks {
            x = block.forward(x);
        }

        if let Some(up) = &self.upsample {
            x = up.forward(x);
        }

        x
    }
}

/// Resnet block with skip connection
#[derive(Module, Debug)]
pub struct ResnetBlock<B: Backend> {
    pub norm1: GroupNorm<B>,
    pub conv1: Conv2d<B>,
    pub norm2: GroupNorm<B>,
    pub conv2: Conv2d<B>,
    pub skip_conv: Option<Conv2d<B>>,
}

impl<B: Backend> ResnetBlock<B> {
    /// Creates a new resnet block with optional skip connection
    pub fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let norm1 = GroupNorm::new(32, in_channels, device);
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

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
            norm2,
            conv2,
            skip_conv,
        }
    }

    /// Forward pass with residual connection
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = match &self.skip_conv {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        let h = self.norm1.forward(x);
        let h = silu(h);
        let h = self.conv1.forward(h);

        let h = self.norm2.forward(h);
        let h = silu(h);
        let h = self.conv2.forward(h);

        h + residual
    }
}

/// 2x Upsampling with conv
#[derive(Module, Debug)]
pub struct Upsample<B: Backend> {
    pub conv: Conv2d<B>,
}

impl<B: Backend> Upsample<B> {
    /// Creates a new 2x upsampling layer
    pub fn new(channels: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self { conv }
    }

    /// Forward pass with nearest neighbor upsampling and convolution
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();

        // Nearest neighbor upsampling 2x
        // Reshape and repeat: [b, c, h, w] -> [b, c, h, 1, w, 1] -> [b, c, h, 2, w, 2] -> [b, c, h*2, w*2]
        let x = x.reshape([b, c, h, 1, w, 1]);
        let x = x.repeat_dim(3, 2).repeat_dim(5, 2);
        let x = x.reshape([b, c, h * 2, w * 2]);

        self.conv.forward(x)
    }
}

/// Self-attention for VAE mid block
#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    pub norm: GroupNorm<B>,
    pub q: Conv2d<B>,
    pub k: Conv2d<B>,
    pub v: Conv2d<B>,
    pub proj_out: Conv2d<B>,
    pub channels: usize,
}

impl<B: Backend> SelfAttention<B> {
    /// Creates a new self-attention layer
    pub fn new(channels: usize, device: &B::Device) -> Self {
        let norm = GroupNorm::new(32, channels, device);

        let q = Conv2dConfig::new([channels, channels], [1, 1]).init(device);
        let k = Conv2dConfig::new([channels, channels], [1, 1]).init(device);
        let v = Conv2dConfig::new([channels, channels], [1, 1]).init(device);
        let proj_out = Conv2dConfig::new([channels, channels], [1, 1]).init(device);

        Self {
            norm,
            q,
            k,
            v,
            proj_out,
            channels,
        }
    }

    /// Forward pass computing scaled dot-product self-attention
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let residual = x.clone();

        let x = self.norm.forward(x);

        let q = self.q.forward(x.clone());
        let k = self.k.forward(x.clone());
        let v = self.v.forward(x);

        // Reshape for attention: [b, c, h, w] -> [b, c, h*w] -> [b, h*w, c]
        let q = q.reshape([b, c, h * w]).swap_dims(1, 2);
        let k = k.reshape([b, c, h * w]); // [b, c, h*w]
        let v = v.reshape([b, c, h * w]).swap_dims(1, 2); // [b, h*w, c]

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let scale = (c as f64).powf(-0.5);
        let attn = q.matmul(k) * scale; // [b, h*w, h*w]
        let attn = burn::tensor::activation::softmax(attn, 2);

        let out = attn.matmul(v); // [b, h*w, c]

        // Reshape back: [b, h*w, c] -> [b, c, h, w]
        let out = out.swap_dims(1, 2).reshape([b, c, h, w]);

        let out = self.proj_out.forward(out);

        out + residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_config() {
        let config = DecoderConfig::sd();
        assert_eq!(config.latent_channels, 4);
        assert_eq!(config.out_channels, 3);
    }
}
