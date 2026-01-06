//! Stable Diffusion Weight Loading from Safetensors
//!
//! Loads SD model weights from HuggingFace-format safetensors files.
//!
//! # Supported Formats
//!
//! - Single file models (model.safetensors)
//! - Multi-file models (text_encoder.safetensors, unet.safetensors, vae.safetensors)
//! - Diffusers-style directory structure
//!
//! # Recommended Workflow
//!
//! Since safetensors loading requires name mapping, we recommend a two-step process:
//!
//! 1. **First run**: Convert safetensors to Burn record format
//! 2. **Subsequent runs**: Load directly from Burn record (much faster)
//!
//! ```ignore
//! use burn::record::{BinFileRecorder, FullPrecisionSettings};
//!
//! // First time: convert and save
//! let loader = SdWeightLoader::open("model/")?;
//! let clip = loader.load_clip_text_encoder::<MyBackend>(&ClipConfig::sd1x(), &device)?;
//! clip.save_file("clip.bin", &BinFileRecorder::<FullPrecisionSettings>::new())?;
//!
//! // Later: load directly (fast)
//! let clip = ClipTextEncoder::new(&ClipConfig::sd1x(), &device)
//!     .load_file("clip.bin", &BinFileRecorder::<FullPrecisionSettings>::new(), &device)?;
//! ```
//!
//! # Current Status
//!
//! - [x] CLIP text encoder loader - fully implemented
//! - [x] UNet loader - fully implemented
//! - [x] VAE decoder loader - fully implemented

use std::path::{Path, PathBuf};

use burn::module::Param;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::nn::conv::Conv2dConfig;
use burn::nn::PaddingConfig2d;
use burn::prelude::*;

use crate::loader::{LoadError, SafeTensorFile};

use burn_models_clip::{ClipConfig, ClipTextEncoder};
use burn_models_core::groupnorm::GroupNorm;
use burn_models_core::layernorm::LayerNorm;

/// Error type for SD weight loading
#[derive(Debug, thiserror::Error)]
pub enum SdLoadError {
    #[error("Load error: {0}")]
    Load(#[from] LoadError),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Shape mismatch for {tensor}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// Weight loader for Stable Diffusion models
pub struct SdWeightLoader {
    /// Path to the weights (file or directory)
    path: PathBuf,
    /// Cached text encoder file
    text_encoder_file: Option<SafeTensorFile>,
    /// Cached UNet file
    unet_file: Option<SafeTensorFile>,
    /// Cached VAE file
    vae_file: Option<SafeTensorFile>,
}

impl SdWeightLoader {
    /// Open a weights path (file or directory)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SdLoadError> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(SdLoadError::FileNotFound(path.display().to_string()));
        }

        Ok(Self {
            path,
            text_encoder_file: None,
            unet_file: None,
            vae_file: None,
        })
    }

    /// Get the SafeTensorFile for text encoder weights
    fn get_text_encoder_file(&mut self) -> Result<&SafeTensorFile, SdLoadError> {
        if self.text_encoder_file.is_none() {
            let file_path = self.find_component_file(&["text_encoder.safetensors", "text_encoder/model.safetensors"])?;
            self.text_encoder_file = Some(SafeTensorFile::open(file_path)?);
        }
        Ok(self.text_encoder_file.as_ref().unwrap())
    }

    /// Get the SafeTensorFile for UNet weights
    fn get_unet_file(&mut self) -> Result<&SafeTensorFile, SdLoadError> {
        if self.unet_file.is_none() {
            let file_path = self.find_component_file(&["unet.safetensors", "unet/diffusion_pytorch_model.safetensors"])?;
            self.unet_file = Some(SafeTensorFile::open(file_path)?);
        }
        Ok(self.unet_file.as_ref().unwrap())
    }

    /// Get the SafeTensorFile for VAE weights
    fn get_vae_file(&mut self) -> Result<&SafeTensorFile, SdLoadError> {
        if self.vae_file.is_none() {
            let file_path = self.find_component_file(&["vae.safetensors", "vae/diffusion_pytorch_model.safetensors"])?;
            self.vae_file = Some(SafeTensorFile::open(file_path)?);
        }
        Ok(self.vae_file.as_ref().unwrap())
    }

    /// Find a component file from possible locations
    fn find_component_file(&self, candidates: &[&str]) -> Result<PathBuf, SdLoadError> {
        // If path is a file, return it directly (single-file model)
        if self.path.is_file() {
            return Ok(self.path.clone());
        }

        // Search for component files in directory
        for candidate in candidates {
            let full_path = self.path.join(candidate);
            if full_path.exists() {
                return Ok(full_path);
            }
        }

        Err(SdLoadError::FileNotFound(format!(
            "Could not find any of {:?} in {}",
            candidates,
            self.path.display()
        )))
    }

    /// Load CLIP text encoder weights
    ///
    /// Returns a ClipTextEncoder with loaded weights.
    pub fn load_clip_text_encoder<B: Backend>(
        &mut self,
        config: &ClipConfig,
        device: &B::Device,
    ) -> Result<ClipTextEncoder<B>, SdLoadError> {
        let file = self.get_text_encoder_file()?;
        load_clip_from_file(file, config, device)
    }

    /// Load UNet weights
    ///
    /// Returns a UNet with loaded weights.
    pub fn load_unet<B: Backend>(
        &mut self,
        config: &burn_models_unet::UNetConfig,
        device: &B::Device,
    ) -> Result<burn_models_unet::UNet<B>, SdLoadError> {
        let file = self.get_unet_file()?;
        load_unet_from_file(file, config, device)
    }

    /// Load VAE decoder weights
    ///
    /// Returns a Decoder with loaded weights.
    pub fn load_vae_decoder<B: Backend>(
        &mut self,
        config: &burn_models_vae::DecoderConfig,
        device: &B::Device,
    ) -> Result<burn_models_vae::Decoder<B>, SdLoadError> {
        let file = self.get_vae_file()?;
        load_vae_decoder_from_file(file, config, device)
    }
}

/// Load CLIP text encoder from a SafeTensorFile
fn load_clip_from_file<B: Backend>(
    file: &SafeTensorFile,
    config: &ClipConfig,
    device: &B::Device,
) -> Result<ClipTextEncoder<B>, SdLoadError> {
    // Detect prefix - could be "text_model" or "text_encoder.text_model"
    let prefix = detect_clip_prefix(file);

    // Load token embedding
    let token_emb_key = format!("{}.embeddings.token_embedding.weight", prefix);
    let token_emb_weight: Tensor<B, 2> = file.load_f32(&token_emb_key, device)?;

    let mut token_embedding = EmbeddingConfig::new(config.vocab_size, config.embed_dim).init(device);
    token_embedding.weight = Param::from_tensor(token_emb_weight);

    // Load position embedding
    let pos_emb_key = format!("{}.embeddings.position_embedding.weight", prefix);
    let position_embedding: Tensor<B, 2> = file.load_f32(&pos_emb_key, device)?;
    let position_embedding = Param::from_tensor(position_embedding);

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_clip_transformer_layer(file, &prefix, i, config, device)?;
        layers.push(layer);
    }

    // Load final layer norm
    let final_ln_weight_key = format!("{}.final_layer_norm.weight", prefix);
    let final_ln_bias_key = format!("{}.final_layer_norm.bias", prefix);
    let final_layer_norm = load_layer_norm(file, &final_ln_weight_key, &final_ln_bias_key, config.embed_dim, device)?;

    Ok(ClipTextEncoder {
        token_embedding,
        position_embedding,
        layers,
        final_layer_norm,
        context_length: config.context_length,
    })
}

/// Detect the prefix used for CLIP weights
fn detect_clip_prefix(file: &SafeTensorFile) -> String {
    // Check common prefixes
    let prefixes = [
        "text_model",
        "text_encoder.text_model",
        "cond_stage_model.transformer.text_model",
    ];

    for prefix in prefixes {
        let test_key = format!("{}.embeddings.token_embedding.weight", prefix);
        if file.contains(&test_key) {
            return prefix.to_string();
        }
    }

    // Default
    "text_model".to_string()
}

/// Load a single CLIP transformer layer
fn load_clip_transformer_layer<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    layer_idx: usize,
    config: &ClipConfig,
    device: &B::Device,
) -> Result<burn_models_clip::TransformerBlock<B>, SdLoadError> {
    let layer_prefix = format!("{}.encoder.layers.{}", prefix, layer_idx);

    // Load attention layer norm (layer_norm1)
    let attn_norm = load_layer_norm(
        file,
        &format!("{}.layer_norm1.weight", layer_prefix),
        &format!("{}.layer_norm1.bias", layer_prefix),
        config.embed_dim,
        device,
    )?;

    // Load self-attention
    let attn = load_clip_attention(file, &layer_prefix, config, device)?;

    // Load FFN layer norm (layer_norm2)
    let ffn_norm = load_layer_norm(
        file,
        &format!("{}.layer_norm2.weight", layer_prefix),
        &format!("{}.layer_norm2.bias", layer_prefix),
        config.embed_dim,
        device,
    )?;

    // Load feed-forward
    let ffn = load_clip_ffn(file, &layer_prefix, config, device)?;

    Ok(burn_models_clip::TransformerBlock {
        attn_norm,
        attn,
        ffn_norm,
        ffn,
    })
}

/// Load CLIP multi-head self-attention
fn load_clip_attention<B: Backend>(
    file: &SafeTensorFile,
    layer_prefix: &str,
    config: &ClipConfig,
    device: &B::Device,
) -> Result<burn_models_clip::MultiHeadSelfAttention<B>, SdLoadError> {
    let embed_dim = config.embed_dim;

    let q_proj = load_linear(
        file,
        &format!("{}.self_attn.q_proj.weight", layer_prefix),
        Some(&format!("{}.self_attn.q_proj.bias", layer_prefix)),
        embed_dim,
        embed_dim,
        device,
    )?;

    let k_proj = load_linear(
        file,
        &format!("{}.self_attn.k_proj.weight", layer_prefix),
        Some(&format!("{}.self_attn.k_proj.bias", layer_prefix)),
        embed_dim,
        embed_dim,
        device,
    )?;

    let v_proj = load_linear(
        file,
        &format!("{}.self_attn.v_proj.weight", layer_prefix),
        Some(&format!("{}.self_attn.v_proj.bias", layer_prefix)),
        embed_dim,
        embed_dim,
        device,
    )?;

    let out_proj = load_linear(
        file,
        &format!("{}.self_attn.out_proj.weight", layer_prefix),
        Some(&format!("{}.self_attn.out_proj.bias", layer_prefix)),
        embed_dim,
        embed_dim,
        device,
    )?;

    Ok(burn_models_clip::MultiHeadSelfAttention {
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        num_heads: config.num_heads,
        head_dim: embed_dim / config.num_heads,
    })
}

/// Load CLIP feed-forward network
fn load_clip_ffn<B: Backend>(
    file: &SafeTensorFile,
    layer_prefix: &str,
    config: &ClipConfig,
    device: &B::Device,
) -> Result<burn_models_clip::FeedForward<B>, SdLoadError> {
    let fc1 = load_linear(
        file,
        &format!("{}.mlp.fc1.weight", layer_prefix),
        Some(&format!("{}.mlp.fc1.bias", layer_prefix)),
        config.embed_dim,
        config.intermediate_size,
        device,
    )?;

    let fc2 = load_linear(
        file,
        &format!("{}.mlp.fc2.weight", layer_prefix),
        Some(&format!("{}.mlp.fc2.bias", layer_prefix)),
        config.intermediate_size,
        config.embed_dim,
        device,
    )?;

    Ok(burn_models_clip::FeedForward { fc1, fc2 })
}

// ============================================================================
// Helper functions for loading common layer types
// ============================================================================

/// Load a Linear layer from safetensors
fn load_linear<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: Option<&str>,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<burn::nn::Linear<B>, SdLoadError> {
    let weight: Tensor<B, 2> = file.load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;

    // Verify shape [out_features, in_features]
    let [out_f, in_f] = weight.dims();
    if out_f != out_features || in_f != in_features {
        return Err(SdLoadError::ShapeMismatch {
            tensor: weight_key.to_string(),
            expected: vec![out_features, in_features],
            actual: vec![out_f, in_f],
        });
    }

    let has_bias = bias_key.map(|k| file.contains(k)).unwrap_or(false);
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(has_bias)
        .init(device);

    linear.weight = Param::from_tensor(weight);

    if let Some(bias_k) = bias_key {
        if file.contains(bias_k) {
            let bias: Tensor<B, 1> = file.load_f32(bias_k, device)?;
            linear.bias = Some(Param::from_tensor(bias));
        }
    }

    Ok(linear)
}

/// Load a LayerNorm from safetensors
fn load_layer_norm<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: &str,
    _size: usize,
    device: &B::Device,
) -> Result<LayerNorm<B>, SdLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;
    let bias: Tensor<B, 1> = file.load_f32(bias_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", bias_key, e)))?;

    Ok(LayerNorm::from_weight_bias(weight, bias))
}

/// Load a GroupNorm from safetensors
#[allow(dead_code)]
fn load_group_norm<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: &str,
    num_groups: usize,
    device: &B::Device,
) -> Result<GroupNorm<B>, SdLoadError> {
    let weight: Tensor<B, 1> = file.load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;
    let bias: Tensor<B, 1> = file.load_f32(bias_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", bias_key, e)))?;

    Ok(GroupNorm {
        num_groups,
        weight,
        bias,
        eps: 1e-6,
    })
}

/// Load a Conv2d from safetensors
fn load_conv2d<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: Option<&str>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
    device: &B::Device,
) -> Result<burn::nn::conv::Conv2d<B>, SdLoadError> {
    let weight: Tensor<B, 4> = file.load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;

    let has_bias = bias_key.map(|k| file.contains(k)).unwrap_or(false);
    let mut conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_bias(has_bias)
        .with_padding(PaddingConfig2d::Explicit(padding, padding))
        .init(device);

    conv.weight = Param::from_tensor(weight);

    if let Some(bias_k) = bias_key {
        if file.contains(bias_k) {
            let bias: Tensor<B, 1> = file.load_f32(bias_k, device)?;
            conv.bias = Some(Param::from_tensor(bias));
        }
    }

    Ok(conv)
}

/// Load a Conv2d with stride from safetensors
fn load_conv2d_strided<B: Backend>(
    file: &SafeTensorFile,
    weight_key: &str,
    bias_key: Option<&str>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    device: &B::Device,
) -> Result<burn::nn::conv::Conv2d<B>, SdLoadError> {
    let weight: Tensor<B, 4> = file.load_f32(weight_key, device)
        .map_err(|e| SdLoadError::MissingTensor(format!("{}: {}", weight_key, e)))?;

    let has_bias = bias_key.map(|k| file.contains(k)).unwrap_or(false);
    let mut conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_bias(has_bias)
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(padding, padding))
        .init(device);

    conv.weight = Param::from_tensor(weight);

    if let Some(bias_k) = bias_key {
        if file.contains(bias_k) {
            let bias: Tensor<B, 1> = file.load_f32(bias_k, device)?;
            conv.bias = Some(Param::from_tensor(bias));
        }
    }

    Ok(conv)
}

// ============================================================================
// UNet Loading
// ============================================================================

use burn_models_unet::{
    UNet, UNetConfig, DownBlock, MidBlock, UpBlock,
    ResBlock, SpatialTransformer, TransformerBlock, CrossAttention, FeedForward,
    Downsample, Upsample,
};

/// Load UNet from a SafeTensorFile
fn load_unet_from_file<B: Backend>(
    file: &SafeTensorFile,
    config: &UNetConfig,
    device: &B::Device,
) -> Result<UNet<B>, SdLoadError> {
    // Detect prefix
    let prefix = detect_unet_prefix(file);
    let ch = config.model_channels;
    let time_embed_dim = ch * 4;

    // Detect naming convention: HF diffusers uses "linear_1"/"linear_2", original uses "0"/"2"
    let uses_hf_naming = file.contains(&format!("{}.time_embed.linear_1.weight", prefix));

    // Time embedding MLP
    let (te0_name, te2_name) = if uses_hf_naming {
        ("linear_1", "linear_2")
    } else {
        ("0", "2")
    };

    let time_embed_0 = load_linear(
        file,
        &format!("{}.time_embed.{}.weight", prefix, te0_name),
        Some(&format!("{}.time_embed.{}.bias", prefix, te0_name)),
        ch,
        time_embed_dim,
        device,
    )?;

    let time_embed_2 = load_linear(
        file,
        &format!("{}.time_embed.{}.weight", prefix, te2_name),
        Some(&format!("{}.time_embed.{}.bias", prefix, te2_name)),
        time_embed_dim,
        time_embed_dim,
        device,
    )?;

    // Input conv
    let conv_in = load_conv2d(
        file,
        &format!("{}.conv_in.weight", prefix),
        Some(&format!("{}.conv_in.bias", prefix)),
        config.in_channels,
        ch,
        3,
        1,
        device,
    )?;

    // Build down blocks
    let mut down_blocks = Vec::new();
    let mut ch_in = ch;

    for (level, &mult) in config.channel_mult.iter().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;

        let block = load_down_block(
            file,
            &prefix,
            level,
            ch_in,
            ch_out,
            time_embed_dim,
            config.num_heads,
            config.head_dim,
            config.context_dim,
            config.transformer_depth,
            !is_last,
            device,
        )?;

        down_blocks.push(block);
        ch_in = ch_out;
    }

    // Mid block
    let mid_block = load_mid_block(
        file,
        &prefix,
        ch_in,
        time_embed_dim,
        config.num_heads,
        config.head_dim,
        config.context_dim,
        config.transformer_depth,
        device,
    )?;

    // Build up blocks
    let mut up_blocks = Vec::new();
    let mut channels: Vec<usize> = vec![ch];
    for (level, &mult) in config.channel_mult.iter().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;
        channels.push(ch_out);
        channels.push(ch_out);
        if !is_last {
            channels.push(ch_out);
        }
    }

    let mut ch_in = ch * config.channel_mult[config.channel_mult.len() - 1];
    let mut block_idx = 0;

    for (level, &mult) in config.channel_mult.iter().rev().enumerate() {
        let ch_out = ch * mult;
        let is_last = level == config.channel_mult.len() - 1;

        for i in 0..3 {
            let skip_ch = channels.pop().unwrap();
            let block_in = ch_in + skip_ch;
            let block_out = if i == 2 && !is_last {
                ch * config.channel_mult[config.channel_mult.len() - 2 - level]
            } else {
                ch_out
            };

            let upsample = i == 2 && !is_last;

            let block = load_up_block(
                file,
                &prefix,
                block_idx,
                block_in,
                block_out,
                time_embed_dim,
                config.num_heads,
                config.head_dim,
                config.context_dim,
                config.transformer_depth,
                upsample,
                device,
            )?;

            up_blocks.push(block);
            ch_in = block_out;
            block_idx += 1;
        }
    }

    // Output
    let norm_out = load_group_norm(
        file,
        &format!("{}.conv_norm_out.weight", prefix),
        &format!("{}.conv_norm_out.bias", prefix),
        32,
        device,
    )?;

    let conv_out = load_conv2d(
        file,
        &format!("{}.conv_out.weight", prefix),
        Some(&format!("{}.conv_out.bias", prefix)),
        ch,
        config.out_channels,
        3,
        1,
        device,
    )?;

    Ok(UNet {
        time_embed_0,
        time_embed_2,
        conv_in,
        down_blocks,
        mid_block,
        up_blocks,
        norm_out,
        conv_out,
        model_channels: ch,
    })
}

/// Detect the prefix used for UNet weights
fn detect_unet_prefix(file: &SafeTensorFile) -> String {
    let prefixes = [
        "model.diffusion_model",
        "unet",
        "",
    ];

    for prefix in prefixes {
        let test_key = if prefix.is_empty() {
            "conv_in.weight".to_string()
        } else {
            format!("{}.conv_in.weight", prefix)
        };
        if file.contains(&test_key) {
            return prefix.to_string();
        }
    }

    "model.diffusion_model".to_string()
}

/// Load a DownBlock
fn load_down_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    level: usize,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    transformer_depth: usize,
    has_downsample: bool,
    device: &B::Device,
) -> Result<DownBlock<B>, SdLoadError> {
    let block_prefix = format!("{}.down_blocks.{}", prefix, level);

    let res1 = load_resblock(file, &format!("{}.resnets.0", block_prefix), in_ch, out_ch, time_dim, device)?;
    let attn1 = load_spatial_transformer(file, &format!("{}.attentions.0", block_prefix), out_ch, num_heads, head_dim, context_dim, transformer_depth, device)?;
    let res2 = load_resblock(file, &format!("{}.resnets.1", block_prefix), out_ch, out_ch, time_dim, device)?;
    let attn2 = load_spatial_transformer(file, &format!("{}.attentions.1", block_prefix), out_ch, num_heads, head_dim, context_dim, transformer_depth, device)?;

    let downsample = if has_downsample {
        Some(load_downsample(file, &format!("{}.downsamplers.0", block_prefix), out_ch, device)?)
    } else {
        None
    };

    Ok(DownBlock {
        res1,
        attn1,
        res2,
        attn2,
        downsample,
    })
}

/// Load a MidBlock
fn load_mid_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    time_dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    transformer_depth: usize,
    device: &B::Device,
) -> Result<MidBlock<B>, SdLoadError> {
    let block_prefix = format!("{}.mid_block", prefix);

    let res1 = load_resblock(file, &format!("{}.resnets.0", block_prefix), channels, channels, time_dim, device)?;
    let attn = load_spatial_transformer(file, &format!("{}.attentions.0", block_prefix), channels, num_heads, head_dim, context_dim, transformer_depth, device)?;
    let res2 = load_resblock(file, &format!("{}.resnets.1", block_prefix), channels, channels, time_dim, device)?;

    Ok(MidBlock {
        res1,
        attn,
        res2,
    })
}

/// Load an UpBlock
fn load_up_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    transformer_depth: usize,
    has_upsample: bool,
    device: &B::Device,
) -> Result<UpBlock<B>, SdLoadError> {
    let block_prefix = format!("{}.up_blocks.{}", prefix, block_idx);

    let res = load_resblock(file, &format!("{}.resnets.0", block_prefix), in_ch, out_ch, time_dim, device)?;
    let attn = load_spatial_transformer(file, &format!("{}.attentions.0", block_prefix), out_ch, num_heads, head_dim, context_dim, transformer_depth, device)?;

    let upsample = if has_upsample {
        Some(load_upsample(file, &format!("{}.upsamplers.0", block_prefix), out_ch, device)?)
    } else {
        None
    };

    Ok(UpBlock {
        res,
        attn,
        upsample,
    })
}

/// Load a ResBlock
fn load_resblock<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    time_dim: usize,
    device: &B::Device,
) -> Result<ResBlock<B>, SdLoadError> {
    let norm1 = load_group_norm(
        file,
        &format!("{}.norm1.weight", prefix),
        &format!("{}.norm1.bias", prefix),
        32,
        device,
    )?;

    let conv1 = load_conv2d(
        file,
        &format!("{}.conv1.weight", prefix),
        Some(&format!("{}.conv1.bias", prefix)),
        in_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let time_emb_proj = load_linear(
        file,
        &format!("{}.time_emb_proj.weight", prefix),
        Some(&format!("{}.time_emb_proj.bias", prefix)),
        time_dim,
        out_ch,
        device,
    )?;

    let norm2 = load_group_norm(
        file,
        &format!("{}.norm2.weight", prefix),
        &format!("{}.norm2.bias", prefix),
        32,
        device,
    )?;

    let conv2 = load_conv2d(
        file,
        &format!("{}.conv2.weight", prefix),
        Some(&format!("{}.conv2.bias", prefix)),
        out_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let skip_conv = if in_ch != out_ch {
        let conv_key = format!("{}.conv_shortcut.weight", prefix);
        if file.contains(&conv_key) {
            Some(load_conv2d(
                file,
                &conv_key,
                Some(&format!("{}.conv_shortcut.bias", prefix)),
                in_ch,
                out_ch,
                1,
                0,
                device,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(ResBlock {
        norm1,
        conv1,
        time_emb_proj,
        norm2,
        conv2,
        skip_conv,
    })
}

/// Load a SpatialTransformer
fn load_spatial_transformer<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    depth: usize,
    device: &B::Device,
) -> Result<SpatialTransformer<B>, SdLoadError> {
    let inner_dim = num_heads * head_dim;

    let norm = load_group_norm(
        file,
        &format!("{}.norm.weight", prefix),
        &format!("{}.norm.bias", prefix),
        32,
        device,
    )?;

    let proj_in = load_conv2d(
        file,
        &format!("{}.proj_in.weight", prefix),
        Some(&format!("{}.proj_in.bias", prefix)),
        channels,
        inner_dim,
        1,
        0,
        device,
    )?;

    let mut transformer_blocks = Vec::with_capacity(depth);
    for i in 0..depth {
        let block = load_transformer_block(
            file,
            &format!("{}.transformer_blocks.{}", prefix, i),
            inner_dim,
            num_heads,
            head_dim,
            context_dim,
            device,
        )?;
        transformer_blocks.push(block);
    }

    let proj_out = load_conv2d(
        file,
        &format!("{}.proj_out.weight", prefix),
        Some(&format!("{}.proj_out.bias", prefix)),
        inner_dim,
        channels,
        1,
        0,
        device,
    )?;

    Ok(SpatialTransformer {
        norm,
        proj_in,
        transformer_blocks,
        proj_out,
    })
}

/// Load a TransformerBlock
fn load_transformer_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: usize,
    device: &B::Device,
) -> Result<TransformerBlock<B>, SdLoadError> {
    let norm1 = load_layer_norm(
        file,
        &format!("{}.norm1.weight", prefix),
        &format!("{}.norm1.bias", prefix),
        dim,
        device,
    )?;

    let attn1 = load_cross_attention(
        file,
        &format!("{}.attn1", prefix),
        dim,
        num_heads,
        head_dim,
        None, // Self-attention
        device,
    )?;

    let norm2 = load_layer_norm(
        file,
        &format!("{}.norm2.weight", prefix),
        &format!("{}.norm2.bias", prefix),
        dim,
        device,
    )?;

    let attn2 = load_cross_attention(
        file,
        &format!("{}.attn2", prefix),
        dim,
        num_heads,
        head_dim,
        Some(context_dim),
        device,
    )?;

    let norm3 = load_layer_norm(
        file,
        &format!("{}.norm3.weight", prefix),
        &format!("{}.norm3.bias", prefix),
        dim,
        device,
    )?;

    let ff = load_feedforward(
        file,
        &format!("{}.ff", prefix),
        dim,
        dim * 4,
        device,
    )?;

    Ok(TransformerBlock {
        norm1,
        attn1,
        norm2,
        attn2,
        norm3,
        ff,
    })
}

/// Load a CrossAttention
fn load_cross_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    dim: usize,
    num_heads: usize,
    head_dim: usize,
    context_dim: Option<usize>,
    device: &B::Device,
) -> Result<CrossAttention<B>, SdLoadError> {
    let inner_dim = num_heads * head_dim;
    let kv_dim = context_dim.unwrap_or(dim);

    let to_q = load_linear(
        file,
        &format!("{}.to_q.weight", prefix),
        None, // CLIP-style attention often has no bias
        dim,
        inner_dim,
        device,
    )?;

    let to_k = load_linear(
        file,
        &format!("{}.to_k.weight", prefix),
        None,
        kv_dim,
        inner_dim,
        device,
    )?;

    let to_v = load_linear(
        file,
        &format!("{}.to_v.weight", prefix),
        None,
        kv_dim,
        inner_dim,
        device,
    )?;

    let to_out = load_linear(
        file,
        &format!("{}.to_out.0.weight", prefix),
        Some(&format!("{}.to_out.0.bias", prefix)),
        inner_dim,
        dim,
        device,
    )?;

    Ok(CrossAttention {
        to_q,
        to_k,
        to_v,
        to_out,
        num_heads,
        head_dim,
    })
}

/// Load a FeedForward
fn load_feedforward<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    dim: usize,
    mult_dim: usize,
    device: &B::Device,
) -> Result<FeedForward<B>, SdLoadError> {
    // GEGLU doubles the projection
    let net_0 = load_linear(
        file,
        &format!("{}.net.0.proj.weight", prefix),
        Some(&format!("{}.net.0.proj.bias", prefix)),
        dim,
        mult_dim * 2,
        device,
    )?;

    let net_2 = load_linear(
        file,
        &format!("{}.net.2.weight", prefix),
        Some(&format!("{}.net.2.bias", prefix)),
        mult_dim,
        dim,
        device,
    )?;

    Ok(FeedForward { net_0, net_2 })
}

/// Load a Downsample
fn load_downsample<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<Downsample<B>, SdLoadError> {
    let conv = load_conv2d_strided(
        file,
        &format!("{}.conv.weight", prefix),
        Some(&format!("{}.conv.bias", prefix)),
        channels,
        channels,
        3,
        2,
        1,
        device,
    )?;

    Ok(Downsample { conv })
}

/// Load an Upsample
fn load_upsample<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<Upsample<B>, SdLoadError> {
    let conv = load_conv2d(
        file,
        &format!("{}.conv.weight", prefix),
        Some(&format!("{}.conv.bias", prefix)),
        channels,
        channels,
        3,
        1,
        device,
    )?;

    Ok(Upsample { conv })
}

// =============================================================================
// VAE Decoder Loading
// =============================================================================

/// Load VAE decoder from a SafeTensorFile
fn load_vae_decoder_from_file<B: Backend>(
    file: &SafeTensorFile,
    config: &burn_models_vae::DecoderConfig,
    device: &B::Device,
) -> Result<burn_models_vae::Decoder<B>, SdLoadError> {
    let prefix = detect_vae_decoder_prefix(file);
    let ch = config.base_channels;
    let ch_mult = &config.channel_mult;

    // Start with highest channel count (reversed from encoder)
    let block_in = ch * ch_mult[ch_mult.len() - 1];

    // Input conv: latent_channels -> block_in
    let conv_in = load_conv2d(
        file,
        &format!("{}.conv_in.weight", prefix),
        Some(&format!("{}.conv_in.bias", prefix)),
        config.latent_channels,
        block_in,
        3,
        1,
        device,
    )?;

    // Mid blocks
    let mid_block1 = load_vae_resnet_block(file, &format!("{}.mid.block_1", prefix), block_in, block_in, device)?;
    let mid_attn = load_vae_self_attention(file, &format!("{}.mid.attn_1", prefix), block_in, device)?;
    let mid_block2 = load_vae_resnet_block(file, &format!("{}.mid.block_2", prefix), block_in, block_in, device)?;

    // Up blocks (reverse order, with upsampling)
    let mut up_blocks = Vec::new();
    let mut in_ch = block_in;

    for (i, &mult) in ch_mult.iter().rev().enumerate() {
        let out_ch = ch * mult;
        let upsample = i < ch_mult.len() - 1; // Don't upsample on last block

        let block = load_vae_decoder_block(
            file,
            &prefix,
            ch_mult.len() - 1 - i, // Map to original block index (3, 2, 1, 0)
            in_ch,
            out_ch,
            config.num_res_blocks,
            upsample,
            device,
        )?;
        up_blocks.push(block);
        in_ch = out_ch;
    }

    // Output layers
    let norm_out = load_group_norm(
        file,
        &format!("{}.norm_out.weight", prefix),
        &format!("{}.norm_out.bias", prefix),
        32,
        device,
    )?;

    let conv_out = load_conv2d(
        file,
        &format!("{}.conv_out.weight", prefix),
        Some(&format!("{}.conv_out.bias", prefix)),
        in_ch,
        config.out_channels,
        3,
        1,
        device,
    )?;

    Ok(burn_models_vae::Decoder {
        conv_in,
        mid_block1,
        mid_attn,
        mid_block2,
        up_blocks,
        norm_out,
        conv_out,
    })
}

/// Detect the prefix used for VAE decoder weights
fn detect_vae_decoder_prefix(file: &SafeTensorFile) -> String {
    let prefixes = [
        "decoder",
        "vae.decoder",
        "first_stage_model.decoder",
    ];

    for prefix in prefixes {
        let test_key = format!("{}.conv_in.weight", prefix);
        if file.contains(&test_key) {
            return prefix.to_string();
        }
    }

    "decoder".to_string()
}

/// Load a VAE decoder block with residual blocks and optional upsampling
fn load_vae_decoder_block<B: Backend>(
    file: &SafeTensorFile,
    vae_prefix: &str,
    block_idx: usize,
    in_ch: usize,
    out_ch: usize,
    num_blocks: usize,
    upsample: bool,
    device: &B::Device,
) -> Result<burn_models_vae::DecoderBlock<B>, SdLoadError> {
    let block_prefix = format!("{}.up.{}", vae_prefix, block_idx);

    let mut res_blocks = Vec::with_capacity(num_blocks);

    // First block handles channel change
    res_blocks.push(load_vae_resnet_block(
        file,
        &format!("{}.block.0", block_prefix),
        in_ch,
        out_ch,
        device,
    )?);

    // Remaining blocks maintain channels
    for i in 1..num_blocks {
        res_blocks.push(load_vae_resnet_block(
            file,
            &format!("{}.block.{}", block_prefix, i),
            out_ch,
            out_ch,
            device,
        )?);
    }

    let upsample_layer = if upsample {
        Some(load_vae_upsample(file, &format!("{}.upsample", block_prefix), out_ch, device)?)
    } else {
        None
    };

    Ok(burn_models_vae::DecoderBlock {
        res_blocks,
        upsample: upsample_layer,
    })
}

/// Load a VAE ResnetBlock
fn load_vae_resnet_block<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    device: &B::Device,
) -> Result<burn_models_vae::ResnetBlock<B>, SdLoadError> {
    let norm1 = load_group_norm(
        file,
        &format!("{}.norm1.weight", prefix),
        &format!("{}.norm1.bias", prefix),
        32,
        device,
    )?;

    let conv1 = load_conv2d(
        file,
        &format!("{}.conv1.weight", prefix),
        Some(&format!("{}.conv1.bias", prefix)),
        in_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let norm2 = load_group_norm(
        file,
        &format!("{}.norm2.weight", prefix),
        &format!("{}.norm2.bias", prefix),
        32,
        device,
    )?;

    let conv2 = load_conv2d(
        file,
        &format!("{}.conv2.weight", prefix),
        Some(&format!("{}.conv2.bias", prefix)),
        out_ch,
        out_ch,
        3,
        1,
        device,
    )?;

    let skip_conv = if in_ch != out_ch {
        let conv_key = format!("{}.nin_shortcut.weight", prefix);
        if file.contains(&conv_key) {
            Some(load_conv2d(
                file,
                &conv_key,
                Some(&format!("{}.nin_shortcut.bias", prefix)),
                in_ch,
                out_ch,
                1,
                0,
                device,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(burn_models_vae::ResnetBlock {
        norm1,
        conv1,
        norm2,
        conv2,
        skip_conv,
    })
}

/// Load a VAE SelfAttention block
fn load_vae_self_attention<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<burn_models_vae::SelfAttention<B>, SdLoadError> {
    let norm = load_group_norm(
        file,
        &format!("{}.norm.weight", prefix),
        &format!("{}.norm.bias", prefix),
        32,
        device,
    )?;

    let q = load_conv2d(
        file,
        &format!("{}.q.weight", prefix),
        Some(&format!("{}.q.bias", prefix)),
        channels,
        channels,
        1,
        0,
        device,
    )?;

    let k = load_conv2d(
        file,
        &format!("{}.k.weight", prefix),
        Some(&format!("{}.k.bias", prefix)),
        channels,
        channels,
        1,
        0,
        device,
    )?;

    let v = load_conv2d(
        file,
        &format!("{}.v.weight", prefix),
        Some(&format!("{}.v.bias", prefix)),
        channels,
        channels,
        1,
        0,
        device,
    )?;

    let proj_out = load_conv2d(
        file,
        &format!("{}.proj_out.weight", prefix),
        Some(&format!("{}.proj_out.bias", prefix)),
        channels,
        channels,
        1,
        0,
        device,
    )?;

    Ok(burn_models_vae::SelfAttention {
        norm,
        q,
        k,
        v,
        proj_out,
        channels,
    })
}

/// Load a VAE Upsample layer
fn load_vae_upsample<B: Backend>(
    file: &SafeTensorFile,
    prefix: &str,
    channels: usize,
    device: &B::Device,
) -> Result<burn_models_vae::Upsample<B>, SdLoadError> {
    let conv = load_conv2d(
        file,
        &format!("{}.conv.weight", prefix),
        Some(&format!("{}.conv.bias", prefix)),
        channels,
        channels,
        3,
        1,
        device,
    )?;

    Ok(burn_models_vae::Upsample { conv })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_clip_prefix() {
        // This would need a mock SafeTensorFile to test properly
    }

    #[test]
    fn test_sd_load_error_display() {
        let err = SdLoadError::MissingTensor("text_model.embeddings.weight".to_string());
        assert!(err.to_string().contains("Missing tensor"));
    }
}
