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
//! - [ ] UNet loader - not yet implemented (complex, many block types)
//! - [ ] VAE decoder loader - not yet implemented

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
#[allow(dead_code)]
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
